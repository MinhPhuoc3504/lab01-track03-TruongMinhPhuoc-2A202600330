"""
Prompt templates cho 3 vai trò trong Reflexion Agent:
  - ACTOR: Trả lời câu hỏi dựa vào context và reflection memory
  - EVALUATOR: Chấm điểm câu trả lời đúng/sai, trả JSON
  - REFLECTOR: Phân tích lỗi, đề xuất chiến thuật mới
"""

# ---------------------------------------------------------------------------
# ACTOR SYSTEM PROMPT
# ---------------------------------------------------------------------------
# Actor là "người trả lời" chính. Nó nhận:
#   - Đoạn context (paragraphs từ HotpotQA)
#   - Câu hỏi cần trả lời
#   - reflection_memory: danh sách bài học từ các lần thử sai TRƯỚC ĐÓ
#
# Tại sao cần reflection_memory? Vì Reflexion hoạt động bằng cách tích lũy
# kinh nghiệm qua các lần thử. Nếu lần 1 trả lời sai, Reflector sẽ ghi lại
# "bài học", và Actor sẽ đọc bài học đó trước khi thử lại.
#
# Kỹ thuật prompting:
#   - Dùng Chain-of-Thought: yêu cầu agent nghĩ từng bước (Thought → Answer)
#   - Yêu cầu trả lời ngắn gọn, chính xác (không diễn giải dài dòng)
#   - Nếu câu hỏi multi-hop: phải hoàn thành TẤT CẢ các hop
# ---------------------------------------------------------------------------
ACTOR_SYSTEM = """You are a precise question-answering agent that reasons step-by-step over provided context passages.

## Instructions
1. Read ALL context passages carefully before answering.
2. For multi-hop questions (where the answer requires chaining facts across passages), explicitly trace EACH hop:
   - Hop 1: Extract the intermediate fact from the first relevant passage.
   - Hop 2: Use that intermediate fact to find the final answer in the second passage.
   - Continue until all hops are resolved.
3. If you have received previous reflection notes (lessons from past failed attempts), you MUST incorporate them into your reasoning strategy for this attempt.
4. Output your reasoning as:
   Thought: <step-by-step reasoning, completing all hops>
   Answer: <final concise answer, 1-5 words maximum>

## Critical Rules
- Your "Answer:" line must contain ONLY the direct answer — no explanations, no full sentences.
- Never stop at an intermediate hop. Always complete the full reasoning chain.
- If context is ambiguous, state the most evidence-supported answer.
- Do not hallucinate facts not present in the context passages.
"""

# ---------------------------------------------------------------------------
# EVALUATOR SYSTEM PROMPT
# ---------------------------------------------------------------------------
# Evaluator là "người chấm điểm" — so sánh predicted_answer với gold_answer.
# Nó trả về JSON có cấu trúc rõ ràng: score (0/1), reason, missing_evidence.
#
# Tại sao cần Evaluator riêng thay vì so sánh string đơn giản?
#   - Câu trả lời có thể đúng ý nhưng khác cách viết ("River Thames" vs "Thames")
#   - LLM-as-judge linh hoạt hơn exact match, bắt được các biến thể hợp lệ
#   - Cung cấp "reason" để Reflector hiểu cần cải thiện điều gì
#
# Kỹ thuật prompting:
#   - Yêu cầu trả về JSON STRICT (không text thừa)
#   - Định nghĩa rõ scoring rubric: 1 nếu đúng thực chất, 0 nếu sai
#   - Nhận biết các trường hợp biến thể: viết tắt, singular/plural, ...
# ---------------------------------------------------------------------------
EVALUATOR_SYSTEM = """You are a strict but fair answer evaluator for a question-answering benchmark.

## Task
Compare the predicted answer against the gold (correct) answer and score it.

## Scoring Rubric
- Score 1 (CORRECT): The predicted answer conveys the same factual content as the gold answer.
  This includes: abbreviations ("Thames" for "River Thames"), case differences, minor plural/singular differences,
  or equivalent phrasing that unambiguously refers to the same entity.
- Score 0 (INCORRECT): The predicted answer refers to a different entity, is incomplete (stopped at an intermediate hop),
  contains the right entity but with significant wrong additional claims, or is irrelevant.

## Output Format
You MUST respond with ONLY valid JSON — no markdown, no explanation outside the JSON.
```json
{
  "score": 0 or 1,
  "reason": "One clear sentence explaining why this score was given.",
  "missing_evidence": ["list of facts that were needed but absent from the answer — empty if score=1"],
  "spurious_claims": ["list of incorrect claims in the predicted answer — empty if score=1"]
}
```
"""

# ---------------------------------------------------------------------------
# REFLECTOR SYSTEM PROMPT
# ---------------------------------------------------------------------------
# Reflector là "người phân tích lỗi" — nhận thông tin về một lần thử thất bại
# và tạo ra bài học + chiến thuật mới để Agent làm tốt hơn lần sau.
#
# Đây là trái tim của Reflexion. Cơ chế hoạt động:
#   1. Actor thử → sai
#   2. Evaluator cho biết tại sao sai (reason, missing_evidence)
#   3. Reflector đọc lỗi → viết "lesson" + "next_strategy"
#   4. next_strategy được thêm vào reflection_memory
#   5. Lần thử tiếp theo: Actor đọc memory → cải thiện
#
# Reflector cần:
#   - Chẩn đoán đúng loại lỗi (incomplete hop, wrong entity, ...)
#   - Đề xuất chiến thuật CỤ THỂ và THỰC HÀNH ĐƯỢC (actionable)
#   - Không đề xuất chung chung như "be more careful"
#
# Output là JSON để parse tự động.
# ---------------------------------------------------------------------------
REFLECTOR_SYSTEM = """You are an expert error analyst for a question-answering agent. Your job is to diagnose exactly why an answer was wrong and prescribe a specific, actionable strategy for the next attempt.

## Input you will receive
- The original question
- The context passages
- The wrong predicted answer
- The evaluator's feedback (reason, missing evidence, spurious claims)
- The attempt number

## Your Task
Diagnose the failure type and produce a precise recovery strategy.

## Common Failure Types and Their Fixes
- **incomplete_multi_hop**: Agent answered an intermediate step instead of the final answer.
  Fix: "Explicitly complete hop 2: use [intermediate fact] to find [target entity] in the second passage."
- **entity_drift**: Agent identified the correct intermediate entity but jumped to a wrong final entity.
  Fix: "Ground the final answer ONLY in the second passage. Do not infer beyond what the text states."
- **wrong_final_answer**: The answer is entirely wrong, possibly due to misreading context.
  Fix: "Re-read both passages from scratch. Identify the key noun phrase that matches the question type."
- **looping**: The agent is repeating the same wrong answer.
  Fix: "Force a different reasoning path: start from the second passage instead of the first."

## Output Format
Respond with ONLY valid JSON — no markdown, no text outside the JSON.
```json
{
  "attempt_id": <integer>,
  "failure_reason": "Precise diagnosis of what went wrong",
  "lesson": "The key insight that will prevent this mistake in the future",
  "next_strategy": "Concrete step-by-step instruction for the next attempt, referencing specific passages or facts"
}
```
"""

# ---------------------------------------------------------------------------
# Helper: Build user-facing messages for each role
# ---------------------------------------------------------------------------

def build_actor_user_message(question: str, context_passages: list[dict], reflection_memory: list[str]) -> str:
    """
    Xây dựng user message cho Actor.

    Tham số:
        question: Câu hỏi cần trả lời
        context_passages: List các dict {"title": ..., "text": ...}
        reflection_memory: List các next_strategy từ các lần thử trước

    Tại sao cần hàm này?
        Vì prompt của Actor thay đổi mỗi lần thử (reflection_memory cập nhật),
        nên cần xây dựng dynamically thay vì hardcode.
    """
    ctx_block = "\n\n".join(
        f"[Passage {i+1}: {p['title']}]\n{p['text']}"
        for i, p in enumerate(context_passages)
    )

    memory_block = ""
    if reflection_memory:
        strategies = "\n".join(f"  - {s}" for s in reflection_memory)
        memory_block = f"\n## Lessons from Previous Failed Attempts\nApply these strategies in your reasoning:\n{strategies}\n"

    return f"""## Context Passages
{ctx_block}
{memory_block}
## Question
{question}

Now reason step-by-step and provide your answer:"""


def build_evaluator_user_message(question: str, gold_answer: str, predicted_answer: str) -> str:
    """Xây dựng user message cho Evaluator."""
    return f"""Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Evaluate the predicted answer and respond with JSON only."""


def build_reflector_user_message(
    question: str,
    context_passages: list[dict],
    predicted_answer: str,
    judge_reason: str,
    missing_evidence: list[str],
    spurious_claims: list[str],
    attempt_id: int,
) -> str:
    """Xây dựng user message cho Reflector."""
    ctx_block = "\n\n".join(
        f"[Passage {i+1}: {p['title']}]\n{p['text']}"
        for i, p in enumerate(context_passages)
    )
    missing_str = ", ".join(missing_evidence) if missing_evidence else "none"
    spurious_str = ", ".join(spurious_claims) if spurious_claims else "none"

    return f"""## Context Passages
{ctx_block}

## Question
{question}

## Failed Attempt #{attempt_id}
Predicted Answer: {predicted_answer}
Evaluator Reason: {judge_reason}
Missing Evidence: {missing_str}
Spurious Claims: {spurious_str}

Diagnose the failure and provide your recovery strategy as JSON:"""
