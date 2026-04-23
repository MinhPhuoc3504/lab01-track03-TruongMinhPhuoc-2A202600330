"""
real_runtime.py — Thay thế mock_runtime.py bằng LLM thật (OpenAI API).

Tại sao cần file này?
--------------------
mock_runtime.py chỉ giả lập phản hồi với dữ liệu cứng (hardcoded). Để
chạy benchmark thật và tính điểm tối đa, ta phải:
  1. Gọi LLM thật với mỗi vai trò dùng model phù hợp
  2. Lấy token count THỰC TẾ từ response (không ước tính)
  3. Đo latency thực tế bằng time.perf_counter()
  4. Tính chi phí USD dựa trên token usage thực tế

Chiến lược chọn model (3 model khác nhau):
============================================
  ACTOR     → gpt-4o-mini:   Trả lời câu hỏi multi-hop. Cần reasoning nhưng
                               không cần mô hình quá mạnh. Rẻ + nhanh.
  EVALUATOR → gpt-3.5-turbo: Chỉ so sánh 2 câu trả lời, chấm 0/1.
                               Task đơn giản nhất → dùng model rẻ nhất.
                               Được gọi MỖI attempt nên cost ảnh hưởng nhiều.
  REFLECTOR → gpt-4o:        Chẩn đoán lỗi sâu, đề xuất chiến thuật.
                               Task phức tạp nhất → cần model mạnh nhất.
                               Nếu Reflector yếu, strategy vô nghĩa → Actor
                               không cải thiện dù có reflection.

Pricing (tháng 4/2026):
  gpt-4o-mini:   $0.150/1M input,  $0.600/1M output
  gpt-3.5-turbo: $0.500/1M input,  $1.500/1M output
  gpt-4o:        $2.500/1M input, $10.000/1M output
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from openai import OpenAI

from .prompts import (
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    build_actor_user_message,
    build_evaluator_user_message,
    build_reflector_user_message,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry

# ---------------------------------------------------------------------------
# Khởi tạo OpenAI client (singleton)
# ---------------------------------------------------------------------------
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy-init OpenAI client. Lấy API key từ env OPENAI_API_KEY."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Cấu hình model — 3 model khác nhau, mỗi vai trò một model phù hợp
# ---------------------------------------------------------------------------
ACTOR_MODEL     = "gpt-3.5-turbo"  # Yếu nhất, rẻ nhất — hay sai → Reflexion có cơ hội cải thiện rõ
EVALUATOR_MODEL = "gpt-4o-mini"    # Mạnh hơn Actor — chấm điểm chính xác hơn, tránh false negative
REFLECTOR_MODEL = "gpt-4o-mini"    # Mạnh hơn Actor — phân tích lỗi và đề xuất chiến thuật chất lượng

# ---------------------------------------------------------------------------
# Pricing (USD per 1 token)
# ---------------------------------------------------------------------------
_PRICING = {
    "gpt-4o-mini":   {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.500 / 1_000_000, "output": 1.500 / 1_000_000},
    "gpt-4o":        {"input": 2.500 / 1_000_000, "output": 10.000 / 1_000_000},
}

def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Tính chi phí USD cho một lần gọi API.

    Công thức:
        cost = (prompt_tokens × price_input) + (completion_tokens × price_output)

    Ví dụ với gpt-4o, 300 input + 80 output:
        = (300 × 0.0000025) + (80 × 0.00001)
        = $0.00075 + $0.00080
        = $0.00155
    """
    pricing = _PRICING.get(model, _PRICING["gpt-4o-mini"])
    return (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])


# ---------------------------------------------------------------------------
# ACTOR — gpt-4o-mini
# ---------------------------------------------------------------------------
def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int, int]:
    """
    Gọi Actor (gpt-4o-mini) để sinh câu trả lời.

    Tại sao gpt-4o-mini?
        Multi-hop QA từ 2 đoạn văn ngắn không đòi hỏi model quá mạnh.
        gpt-4o-mini đủ khả năng reasoning trong tình huống này với chi phí thấp.

    Luồng:
        1. Build context + question + reflection_memory thành user message
        2. Gọi API với temperature=0 (deterministic)
        3. Parse dòng "Answer:" từ output Chain-of-Thought
        4. Trả về (answer, prompt_tokens, completion_tokens, latency_ms)
    """
    client = _get_client()
    context_passages = [{"title": c.title, "text": c.text} for c in example.context]
    user_msg = build_actor_user_message(example.question, context_passages, reflection_memory)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=ACTOR_MODEL,
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,   # Deterministic: giảm variance
        max_tokens=300,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Token count THỰC TẾ từ API response — không ước tính
    prompt_tokens     = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    raw_text = response.choices[0].message.content or ""
    answer = _parse_actor_answer(raw_text)

    return answer, prompt_tokens, completion_tokens, latency_ms


def _parse_actor_answer(raw_text: str) -> str:
    """
    Trích xuất phần trả lời từ output CoT của Actor.
    Ưu tiên dòng "Answer:", fallback lấy dòng cuối.
    """
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("answer:"):
            return stripped[len("answer:"):].strip()
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    return lines[-1] if lines else raw_text.strip()


# ---------------------------------------------------------------------------
# EVALUATOR — gpt-3.5-turbo (model rẻ nhất, task đơn giản nhất)
# ---------------------------------------------------------------------------
def evaluator(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, int, int]:
    """
    Gọi Evaluator (gpt-3.5-turbo) để chấm điểm.

    Tại sao gpt-3.5-turbo?
        Evaluator chỉ cần so sánh 2 chuỗi về cùng một sự kiện. Đây là
        task phán đoán đơn giản, không cần khả năng reasoning phức tạp.
        Dùng model rẻ nhất vì Evaluator được gọi MỌI attempt (N lần/câu).

    Tại sao không dùng exact-match string?
        - "River Thames" vs "Thames" → đều đúng nhưng exact match sẽ fail
        - "Oxford University" vs "Oxford" → cần ngữ cảnh để phán đoán
        - LLM-as-judge bắt được các biến thể này, tăng accuracy

    Dùng response_format=json_object để force JSON output, tránh text thừa.

    Returns: (JudgeResult, prompt_tokens, completion_tokens, latency_ms)
    """
    client = _get_client()
    user_msg = build_evaluator_user_message(example.question, example.gold_answer, answer)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    prompt_tokens     = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    raw_json = response.choices[0].message.content or "{}"
    judge = _parse_judge_result(raw_json, example.gold_answer, answer)

    return judge, prompt_tokens, completion_tokens, latency_ms


def _parse_judge_result(raw_json: str, gold_answer: str, predicted_answer: str) -> JudgeResult:
    """Parse JSON từ Evaluator. Fallback về exact match nếu JSON lỗi."""
    try:
        data = json.loads(raw_json)
        return JudgeResult(
            score=int(data.get("score", 0)),
            reason=data.get("reason", ""),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        from .utils import normalize_answer
        is_correct = normalize_answer(gold_answer) == normalize_answer(predicted_answer)
        return JudgeResult(
            score=1 if is_correct else 0,
            reason="Fallback to exact match due to JSON parse error.",
        )


# ---------------------------------------------------------------------------
# REFLECTOR — gpt-4o (model mạnh nhất, task phức tạp nhất)
# ---------------------------------------------------------------------------
def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, int, int]:
    """
    Gọi Reflector (gpt-4o) để phân tích lỗi và đề xuất chiến thuật.

    Tại sao gpt-4o?
        Đây là bước QUAN TRỌNG NHẤT trong Reflexion. Nếu Reflector chẩn đoán
        sai hoặc đề xuất chiến thuật mơ hồ ("try harder"), thì Actor sẽ
        không cải thiện dù có bao nhiêu lần thử. gpt-4o có khả năng:
          - Nhận biết loại lỗi reasoning (incomplete hop vs entity drift)
          - Đề xuất chiến thuật cụ thể, có thể thực thi ngay
          - Tham chiếu đúng đoạn văn cần đọc lại

        Chi phí cao hơn nhưng Reflector chỉ được gọi khi Agent SAI,
        và chỉ tối đa (max_attempts - 1) lần/câu hỏi.

    Luồng:
        1. Build message từ: context, câu hỏi, lý do sai (từ Judge)
        2. Gọi gpt-4o với temperature=0.2 (đủ creative để đề xuất đa dạng)
        3. Parse JSON ra ReflectionEntry
        4. .next_strategy sẽ được thêm vào reflection_memory của Agent

    Returns: (ReflectionEntry, prompt_tokens, completion_tokens, latency_ms)
    """
    client = _get_client()
    context_passages = [{"title": c.title, "text": c.text} for c in example.context]
    user_msg = build_reflector_user_message(
        question=example.question,
        context_passages=context_passages,
        predicted_answer="(from previous attempt — see evaluator feedback)",
        judge_reason=judge.reason,
        missing_evidence=judge.missing_evidence,
        spurious_claims=judge.spurious_claims,
        attempt_id=attempt_id,
    )

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=REFLECTOR_MODEL,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,   # Nhẹ creativity để đa dạng chiến thuật
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    prompt_tokens     = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    raw_json = response.choices[0].message.content or "{}"
    entry = _parse_reflection_entry(raw_json, attempt_id, judge)

    return entry, prompt_tokens, completion_tokens, latency_ms


def _parse_reflection_entry(raw_json: str, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    """Parse JSON từ Reflector. Fallback về nội dung từ judge nếu JSON lỗi."""
    try:
        data = json.loads(raw_json)
        return ReflectionEntry(
            attempt_id=int(data.get("attempt_id", attempt_id)),
            failure_reason=data.get("failure_reason", judge.reason),
            lesson=data.get("lesson", "The previous attempt was incorrect; try a different approach."),
            next_strategy=data.get("next_strategy", "Re-read all passages and trace each hop explicitly."),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="The previous attempt was incorrect.",
            next_strategy="Re-read all passages carefully and complete all reasoning hops before answering.",
        )
