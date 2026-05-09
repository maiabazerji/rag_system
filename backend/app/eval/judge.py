from __future__ import annotations

from app.config import settings

JUDGE_RUBRIC = """You are an impartial judge. Score the answer on:
- faithfulness (0-1): claims grounded in context
- answer_relevance (0-1): addresses the question
- context_precision (0-1): retrieved chunks are relevant, ranked well
- context_recall (0-1): needed information is present

Return strict JSON: {"faithfulness": ..., "answer_relevance": ..., "context_precision": ..., "context_recall": ..., "reasoning": "..."}"""


async def judge(question: str, answer: str, context: list[str]) -> dict:
    """Call judge model. Stub returns zeros — wire to Anthropic/OpenAI."""
    _ = settings.judge_model
    return {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "reasoning": "stub",
    }
