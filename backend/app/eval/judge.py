"""Evaluation functions for scoring RAG answer quality.

This module provides LLM-based evaluation metrics for assessing RAG system output.
It measures answer quality along four key dimensions using Claude as an impartial judge:

1. Faithfulness: Are claims grounded in retrieved context (no hallucination)?
2. Answer Relevance: Does the answer address the user's question?
3. Context Precision: Are retrieved chunks relevant and well-ranked?
4. Context Recall: Is all necessary information present in the context?

These metrics are useful for:
- Comparing RAG strategies (classic vs. graph vs. agentic)
- Detecting hallucinations and off-topic answers
- Identifying retrieval quality issues
- Benchmarking against golden datasets

Reference:
    Inspired by RAGAS (RAG Assessment) metrics
    https://github.com/explodinggradients/ragas
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

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


class ScoreOutput(BaseModel):
    """Structured output from Claude for a single evaluation dimension.

    Used with instructor for structured output extraction from LLM responses.
    Ensures Claude returns properly formatted JSON with score and reasoning.

    Attributes:
        score: Numeric score from 0 to 1 representing the dimension (0=poor, 1=perfect).
        reasoning: Brief justification for the score (1-2 sentences).
    """
    score: float = Field(ge=0, le=1, description="Score between 0 and 1")
    reasoning: str = Field(description="Brief reasoning for the score")


async def score_faithfulness(answer: str, context: list[str]) -> dict[str, Any]:
    """Score faithfulness: what fraction of claims are grounded in retrieved context.

    Measures hallucination by checking whether statements in the answer are
    supported by the provided context chunks. High faithfulness means the answer
    stays close to the facts in context; low means the answer makes unsupported
    claims or uses outside knowledge.

    Uses Claude Opus as judge via instructor for structured output extraction.

    Args:
        answer: The generated answer text to evaluate.
        context: List of context chunk strings that the answer should be grounded in.
            Only first 5 chunks are used to keep prompt manageable.

    Returns:
        Dict with keys:
            - "score": Float 0-1 (0=hallucinated, 1=all claims supported)
            - "reasoning": Brief text explanation of the score

        On error, returns default score of 0.5 with error message.

    Raises:
        No exceptions. All errors caught and returned as failed evaluations.

    Example:
        >>> answer = "The capital of France is Paris, located in Northern Europe."
        >>> context = ["France is a country in Western Europe. Paris is its capital."]
        >>> result = await score_faithfulness(answer, context)
        >>> print(f"Faithfulness: {result['score']:.2f}")  # ~1.0
    """
    try:
        context_str = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context[:5]))
        prompt = f"""Evaluate the faithfulness of the following answer based on the provided context.
Faithfulness measures what fraction of claims in the answer are grounded in and supported by the context.

Context:
{context_str or "(no context)"}

Answer:
{answer}

Score from 0 to 1 where:
- 1.0 = all claims are supported by the context
- 0.5 = some claims are supported, some are unsupported
- 0.0 = no claims are grounded or answer is hallucinated

Provide a score and brief reasoning."""

        api_key = settings.anthropic_api_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        client = instructor.from_anthropic(AsyncAnthropic(api_key=api_key))
        response = await client.messages.create(
            model=settings.judge_model or "claude-opus-4-7",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            response_model=ScoreOutput,
        )

        return {
            "score": max(0, min(1, float(response.score))),
            "reasoning": response.reasoning,
        }
    except Exception as e:
        logger.exception(f"Faithfulness scoring error: {e}")
        return {
            "score": 0.5,
            "reasoning": f"Error: {type(e).__name__}",
        }


async def score_relevance(question: str, answer: str) -> dict[str, Any]:
    """Score answer relevance: does the answer address the user's question?

    Measures task completion by checking whether the answer is relevant to and
    directly addresses the question. An answer may be factually correct (faithful)
    but still irrelevant if it doesn't answer what was asked.

    Uses Claude Opus as judge via instructor for structured output.

    Args:
        question: The user's original question.
        answer: The generated answer to evaluate.

    Returns:
        Dict with keys:
            - "score": Float 0-1 (0=irrelevant, 1=perfectly addresses question)
            - "reasoning": Brief text explanation of the score

        On error, returns default score of 0.5 with error message.

    Raises:
        No exceptions. All errors caught and returned as failed evaluations.

    Example:
        >>> question = "What is photosynthesis?"
        >>> answer = "Plants use sunlight to make food through photosynthesis..."
        >>> result = await score_relevance(question, answer)
        >>> print(f"Relevance: {result['score']:.2f}")  # ~1.0
    """
    try:
        prompt = f"""Evaluate how well the answer addresses the user's question.
Answer relevance measures whether the answer directly and completely answers what was asked.

Question:
{question}

Answer:
{answer}

Score from 0 to 1 where:
- 1.0 = directly and completely answers the question
- 0.5 = partially addresses or tangential
- 0.0 = off-topic or irrelevant

Provide a score and brief reasoning."""

        api_key = settings.anthropic_api_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        client = instructor.from_anthropic(AsyncAnthropic(api_key=api_key))
        response = await client.messages.create(
            model=settings.judge_model or "claude-opus-4-7",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            response_model=ScoreOutput,
        )

        return {
            "score": max(0, min(1, float(response.score))),
            "reasoning": response.reasoning,
        }
    except Exception as e:
        logger.exception(f"Relevance scoring error: {e}")
        return {
            "score": 0.5,
            "reasoning": f"Error: {type(e).__name__}",
        }


async def judge(question: str, answer: str, context: list[str]) -> dict[str, Any]:
    """Comprehensive evaluation of RAG answer on four dimensions.

    Uses Claude as an impartial judge to score the answer on:
    1. Faithfulness: Are claims grounded in context (no hallucination)?
    2. Answer Relevance: Does the answer address the question?
    3. Context Precision: Are retrieved chunks relevant and well-ranked?
    4. Context Recall: Is all needed information present in context?

    This is the primary evaluation function for comparing RAG strategies and
    detecting quality issues. Each dimension scored 0-1.

    Args:
        question: The user's question that was answered.
        answer: The answer text generated by RAG system.
        context: List of retrieved context chunks provided to the generator.
            Only first 5 chunks used in evaluation prompt.

    Returns:
        Dict with keys:
            - "faithfulness": Float 0-1, fraction of claims grounded in context
            - "answer_relevance": Float 0-1, how well answer addresses question
            - "context_precision": Float 0-1, relevance of retrieved chunks
            - "context_recall": Float 0-1, completeness of context
            - "reasoning": Brief overall justification

        All scores clamped to [0, 1]. On error, returns scores of 0.5 with
        error message.

    Raises:
        No exceptions. Parsing errors and API errors caught and returned as
        degraded scores (0.5) with error reasoning.

    Example:
        >>> question = "What is AI?"
        >>> answer = "AI is artificial intelligence..."
        >>> context = ["AI stands for Artificial Intelligence..."]
        >>> scores = await judge(question, answer, context)
        >>> print(f"Avg score: {sum(s for k,s in scores.items() if isinstance(s,float))/4:.2f}")
    """
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
