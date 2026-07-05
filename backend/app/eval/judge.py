from __future__ import annotations

import json
import logging
from app.config import settings
from app.rag.providers.anthropic_provider import generate_with_usage

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an impartial judge scoring a RAG answer. Score on:

**Faithfulness (0-1):** What fraction of claims in the answer are grounded in the retrieved context?
- 1.0 = all claims supported by context
- 0.5 = some claims supported, some unsupported
- 0.0 = no claims grounded or hallucinated

**Answer Relevance (0-1):** Does the answer address the user's question?
- 1.0 = directly and completely answers the question
- 0.5 = partially addresses or tangential
- 0.0 = off-topic or irrelevant

**Context Precision (0-1):** Are the retrieved chunks relevant and well-ranked?
- 1.0 = all chunks are highly relevant
- 0.5 = mixed relevance
- 0.0 = chunks are irrelevant

**Context Recall (0-1):** Is all needed information present in the context?
- 1.0 = everything needed to answer is in context
- 0.5 = some needed info is missing
- 0.0 = critical info missing

Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Return ONLY valid JSON (no markdown, no extra text):
{{"faithfulness": <0-1>, "answer_relevance": <0-1>, "context_precision": <0-1>, "context_recall": <0-1>, "reasoning": "brief justification"}}"""


async def judge(question: str, answer: str, context: list[str]) -> dict:
    """Score answer using Claude as judge."""
    try:
        context_str = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context[:5]))
        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context_str or "(no context)",
            answer=answer,
        )

        result = await generate_with_usage(
            model=settings.judge_model or "claude-opus-4-1-20250805",
            prompt=prompt,
            max_tokens=256,
        )

        text = result["text"].strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        scores = json.loads(text)
        return {
            "faithfulness": max(0, min(1, float(scores.get("faithfulness", 0.0)))),
            "answer_relevance": max(0, min(1, float(scores.get("answer_relevance", 0.0)))),
            "context_precision": max(0, min(1, float(scores.get("context_precision", 0.0)))),
            "context_recall": max(0, min(1, float(scores.get("context_recall", 0.0)))),
            "reasoning": str(scores.get("reasoning", "")),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Judge JSON parse error: {e}")
        return {
            "faithfulness": 0.5,
            "answer_relevance": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "reasoning": "judge parse error",
        }
    except Exception as e:
        logger.exception(f"Judge error: {e}")
        return {
            "faithfulness": 0.5,
            "answer_relevance": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "reasoning": f"error: {type(e).__name__}",
        }
