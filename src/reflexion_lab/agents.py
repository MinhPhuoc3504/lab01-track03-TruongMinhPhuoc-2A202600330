"""
agents.py — Định nghĩa ReActAgent và ReflexionAgent.

Tại sao có 2 agent?
-------------------
Bài lab yêu cầu SO SÁNH 2 phương pháp:
  - ReActAgent:    1 lần thử duy nhất, không có cơ chế tự sửa lỗi
  - ReflexionAgent: Tối đa N lần thử, mỗi lần sai → Reflector phân tích
                    → reflection_memory → Actor thử lại tốt hơn

Cả hai đều kế thừa BaseAgent — chỉ khác nhau ở max_attempts và agent_type.

Token tracking đầy đủ:
-----------------------
Mỗi lần gọi LLM (Actor, Evaluator, Reflector) đều trả về:
  (result, prompt_tokens, completion_tokens, latency_ms)

Tổng hợp lại vào RunRecord để tính:
  - token_estimate: tổng tokens tiêu thụ toàn run
  - prompt_tokens / completion_tokens: phân tách input/output
  - token_cost_usd: chi phí USD thực tế (từ compute_cost())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .real_runtime import (
    ACTOR_MODEL,
    EVALUATOR_MODEL,
    REFLECTOR_MODEL,
    actor_answer,
    compute_cost,
    evaluator,
    reflector,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    """
    Lớp cơ sở cho cả ReAct và Reflexion Agent.

    Thuộc tính:
        agent_type:   "react" hoặc "reflexion"
        max_attempts: Số lần thử tối đa (ReAct=1, Reflexion=3 mặc định)

    Phương thức chính:
        run(example) → RunRecord
            Chạy agent trên một QAExample và trả về toàn bộ kết quả.
    """
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        """
        Vòng lặp chính của Agent.

        Luồng (Reflexion):
        ┌─────────────────────────────────────────────┐
        │ for attempt in 1..max_attempts:              │
        │   1. Actor  → answer  (dùng reflection_memory) │
        │   2. Evaluator → judge (score 0/1 + reason)  │
        │   3. Nếu score=1: DỪNG (đã đúng)            │
        │   4. Nếu score=0 và còn lần thử:             │
        │      Reflector → reflection_entry            │
        │      reflection_memory.append(next_strategy) │
        └─────────────────────────────────────────────┘

        Token tracking:
            Mỗi lần gọi LLM trả về (result, pt, ct, lat).
            Cộng dồn tất cả vào total_prompt_tokens, total_completion_tokens.
            Cuối cùng tính cost = compute_cost() cho từng model.

        Failure mode detection:
            - "none": đúng ngay
            - "entity_drift": có claim sai trong judge.spurious_claims
            - "incomplete_multi_hop": missing_evidence không rỗng
            - "wrong_final_answer": sai nhưng không có thêm thông tin
            - "looping": cùng câu trả lời sai lặp lại ≥2 lần
            - "reflection_overfit": đúng trước đó nhưng reflection làm sai
        """
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []

        final_answer = ""
        final_score = 0

        # Accumulate tokens across ALL LLM calls (actor + evaluator + reflector)
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0
        total_cost_usd: float = 0.0
        total_latency: int = 0

        previous_answers: list[str] = []  # Để detect looping

        for attempt_id in range(1, self.max_attempts + 1):

            # ── 1. ACTOR: Sinh câu trả lời ────────────────────────────────
            answer, a_pt, a_ct, a_lat = actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )
            total_prompt_tokens     += a_pt
            total_completion_tokens += a_ct
            total_cost_usd          += compute_cost(ACTOR_MODEL, a_pt, a_ct)
            total_latency           += a_lat

            # ── 2. EVALUATOR: Chấm điểm ───────────────────────────────────
            judge, e_pt, e_ct, e_lat = evaluator(example, answer)
            total_prompt_tokens     += e_pt
            total_completion_tokens += e_ct
            total_cost_usd          += compute_cost(EVALUATOR_MODEL, e_pt, e_ct)
            total_latency           += e_lat

            # ── Ghi trace cho attempt này ──────────────────────────────────
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=a_pt + a_ct + e_pt + e_ct,
                prompt_tokens=a_pt + e_pt,
                completion_tokens=a_ct + e_ct,
                latency_ms=a_lat + e_lat,
            )

            final_answer = answer
            final_score  = judge.score

            if judge.score == 1:
                # Đúng rồi — không cần reflection
                traces.append(trace)
                break

            # ── 3. REFLEXION: Phân tích lỗi (chỉ khi là ReflexionAgent) ──
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection_entry, r_pt, r_ct, r_lat = reflector(
                    example, attempt_id, judge
                )
                total_prompt_tokens     += r_pt
                total_completion_tokens += r_ct
                total_cost_usd          += compute_cost(REFLECTOR_MODEL, r_pt, r_ct)
                total_latency           += r_lat

                # Cập nhật token estimate của trace để bao gồm cả reflector
                trace = AttemptTrace(
                    attempt_id=attempt_id,
                    answer=answer,
                    score=judge.score,
                    reason=judge.reason,
                    reflection=reflection_entry,
                    token_estimate=a_pt + a_ct + e_pt + e_ct + r_pt + r_ct,
                    prompt_tokens=a_pt + e_pt + r_pt,
                    completion_tokens=a_ct + e_ct + r_ct,
                    latency_ms=a_lat + e_lat + r_lat,
                )

                # Thêm chiến thuật vào memory → Actor sẽ dùng lần sau
                reflection_memory.append(reflection_entry.next_strategy)
                reflections.append(reflection_entry)

            traces.append(trace)
            previous_answers.append(answer)

        # ── Detect failure mode ─────────────────────────────────────────────
        failure_mode = _detect_failure_mode(
            final_score, traces, previous_answers
        )

        total_tokens = total_prompt_tokens + total_completion_tokens

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            token_cost_usd=round(total_cost_usd, 8),
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


def _detect_failure_mode(
    final_score: int,
    traces: list[AttemptTrace],
    previous_answers: list[str],
) -> str:
    """
    Tự động phát hiện loại lỗi dựa trên traces và judge feedback.

    Các loại failure mode:
        none:                  Đúng — không có lỗi
        looping:               Cùng câu trả lời sai lặp lại ≥2 lần
        incomplete_multi_hop:  Evaluator báo missing_evidence
        entity_drift:          Evaluator báo spurious_claims
        reflection_overfit:    Đúng ở lần thử giữa nhưng sai ở cuối
        wrong_final_answer:    Sai nhưng không thuộc các loại trên
    """
    if final_score == 1:
        return "none"

    # Detect looping: cùng answer lặp lại
    if len(previous_answers) >= 2 and len(set(previous_answers)) == 1:
        return "looping"

    # Detect reflection_overfit: có lần đúng ở giữa nhưng cuối lại sai
    scores = [t.score for t in traces]
    if 1 in scores and scores[-1] == 0:
        return "reflection_overfit"

    # Lấy trace cuối cùng để check judge reason
    last_trace = traces[-1]
    last_reason = last_trace.reason.lower() if last_trace.reason else ""

    if any(kw in last_reason for kw in ["incomplete", "hop", "missing", "intermediate"]):
        return "incomplete_multi_hop"

    if any(kw in last_reason for kw in ["drift", "wrong entity", "spurious", "incorrect entity"]):
        return "entity_drift"

    return "wrong_final_answer"


# ---------------------------------------------------------------------------
# Các Agent cụ thể
# ---------------------------------------------------------------------------

class ReActAgent(BaseAgent):
    """
    ReAct Agent — 1 lần thử duy nhất.

    Không có reflection. Nếu sai thì sai.
    Dùng làm baseline để so sánh với Reflexion.
    """
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent — tối đa max_attempts lần thử.

    Mỗi lần sai: Reflector phân tích lỗi → reflection_memory cập nhật
    → lần thử tiếp theo Actor dùng memory đó để cải thiện.

    Tham số:
        max_attempts: Số lần thử tối đa (mặc định 3)
                      3 là điểm cân bằng tốt: đủ cơ hội cải thiện
                      mà không tốn quá nhiều token/chi phí.
    """
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
