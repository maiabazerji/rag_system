"""RAGAS scoring for evaluation runs (``run_eval.py --scorer ragas|both``).

`RAGAS <https://github.com/explodinggradients/ragas>`_ scores a RAG answer
with its own LLM prompts. It runs next to (or instead of) the project's own
judge so the two can be compared on the same answers. Its metrics are reported
under ``ragas_*`` names so they never collide with the judge's dimensions or
with the regression thresholds written for them.

Metrics (``ragas.metrics.collections``):

``ragas_faithfulness``
    Share of the answer's claims supported by the retrieved contexts.
``ragas_answer_relevancy``
    How directly the answer addresses the question (questions regenerated from
    the answer, compared with the real one by embedding similarity).
``ragas_context_precision``
    Whether the contexts that help answer are ranked first. Needs a reference.
``ragas_context_recall``
    Share of the reference answer's claims found in the contexts. Needs a
    reference.
``ragas_answer_correctness``
    Factual overlap with the reference answer, plus semantic similarity.
    Needs a reference.

The LLM is the configured judge model, called through the Anthropic client.
Embeddings are the pipeline's own local model (:mod:`app.rag.embed`), so no
document or question text goes to an embedding API.

RAGAS is optional: this module imports without it, and :func:`ragas_available`
says whether it can run (``pip install -e "backend[ragas]"``). Scoring never
raises: each metric runs on its own, so one failing metric records its error
and the others still score.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

#: Reported metric names, in report order.
RAGAS_METRICS: tuple[str, ...] = (
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_answer_correctness",
)

#: Metrics that compare against the example's reference (ideal) answer.
REFERENCE_METRICS = frozenset(
    {"ragas_context_precision", "ragas_context_recall", "ragas_answer_correctness"}
)

INSTALL_HINT = 'RAGAS is not installed: pip install -e "backend[ragas]"'


@dataclass
class RagasOutcome:
    """Result of scoring one answer with RAGAS.

    Attributes:
        scores: Measured metrics, by ``ragas_*`` name. Unmeasured ones are absent.
        errors: Per-metric error messages for metrics that failed.
        skipped: Metrics not attempted, with the reason (e.g. no reference).
        error_type: Set when nothing could run at all (RAGAS missing, no
            contexts, ...); ``error`` then says why.
        ragas_version: Installed RAGAS version, if any.
    """

    scores: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None
    ragas_version: str | None = None

    def as_row(self) -> dict[str, Any]:
        """The block stored on an evaluation row as ``row["ragas"]``."""
        return {
            "scores": self.scores,
            "errors": self.errors,
            "skipped": self.skipped,
            "error_type": self.error_type,
            "error": self.error,
            "version": self.ragas_version,
        }


def ragas_available() -> bool:
    """Whether RAGAS (and its LLM adapter) can be imported."""
    return all(importlib.util.find_spec(m) is not None for m in ("ragas", "instructor"))


def ragas_version() -> str | None:
    if not ragas_available():
        return None
    try:
        import ragas

        return str(ragas.__version__)
    except Exception:  # pragma: no cover - broken install
        return None


def _embeddings_class() -> type:
    """A RAGAS embedding backed by the pipeline's own embedding model."""
    from ragas.embeddings.base import BaseRagasEmbedding

    from app.rag.embed import embed_texts, embed_texts_async

    class PipelineEmbeddings(BaseRagasEmbedding):
        """Embeds with :mod:`app.rag.embed`, as queries (symmetric similarity)."""

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embed_texts([text], kind="query")[0]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return (await embed_texts_async([text], kind="query"))[0]

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return embed_texts(list(texts), kind="query")

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return await embed_texts_async(list(texts), kind="query")

    return PipelineEmbeddings


_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def _sampling_safe_client(client: Any) -> Any:
    """Wrap an AsyncAnthropic client so RAGAS's sampling arguments cannot break calls.

    RAGAS's LLM adapter (instructor) always sends ``temperature``. Recent
    Anthropic SDKs no longer accept it as a keyword, and Claude 5 models reject
    it outright. Sampling parameters are therefore dropped for models that
    reject them (:func:`supports_temperature`, the pipeline's own rule) and
    passed through ``extra_body`` for the models that still honour them.
    """
    from app.rag.providers.anthropic_provider import supports_temperature

    create = client.messages.create

    async def safe_create(*args: Any, **kwargs: Any) -> Any:
        sampling = {k: kwargs.pop(k) for k in _SAMPLING_PARAMS if k in kwargs}
        if sampling and supports_temperature(str(kwargs.get("model", ""))):
            kwargs["extra_body"] = {**sampling, **(kwargs.get("extra_body") or {})}
        return await create(*args, **kwargs)

    client.messages.create = safe_create
    return client


_metrics_cache: dict[str, Any] | None = None


def _build_metrics() -> dict[str, Any]:
    """Instantiate the RAGAS metrics once: one LLM client, one embedding wrapper."""
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    import warnings

    from anthropic import AsyncAnthropic

    with warnings.catch_warnings():
        # ragas emits deprecation chatter on import; it is not actionable here.
        warnings.simplefilter("ignore")
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )

    llm = llm_factory(
        settings.judge_model,
        provider="anthropic",
        client=_sampling_safe_client(AsyncAnthropic(api_key=settings.anthropic_api_key or None)),
    )
    embeddings = _embeddings_class()()
    _metrics_cache = {
        "ragas_faithfulness": Faithfulness(llm=llm),
        "ragas_answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "ragas_context_precision": ContextPrecisionWithReference(llm=llm),
        "ragas_context_recall": ContextRecall(llm=llm),
        "ragas_answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }
    return _metrics_cache


def _call_args(
    name: str, question: str, answer: str, contexts: list[str], reference: str | None
) -> dict[str, Any]:
    """Keyword arguments for each metric's ``ascore``."""
    if name == "ragas_faithfulness":
        return {"user_input": question, "response": answer, "retrieved_contexts": contexts}
    if name == "ragas_answer_relevancy":
        return {"user_input": question, "response": answer}
    if name == "ragas_context_precision":
        return {"user_input": question, "reference": reference, "retrieved_contexts": contexts}
    if name == "ragas_context_recall":
        return {"user_input": question, "retrieved_contexts": contexts, "reference": reference}
    return {"user_input": question, "response": answer, "reference": reference}


def _as_float(result: Any) -> float | None:
    value = getattr(result, "value", result)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN means "could not compute"


async def score_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None = None,
    *,
    metrics: dict[str, Any] | None = None,
) -> RagasOutcome:
    """Score one answer with every applicable RAGAS metric.

    Skips (no score, reason recorded): every metric when there are no contexts
    or no answer (a refusal has nothing to check), and the reference-based
    metrics when ``reference`` is empty. Never raises.

    Args:
        question: The user question.
        answer: The generated answer.
        contexts: Retrieved chunk texts, best-ranked first.
        reference: Ideal answer from the golden dataset, if any.
        metrics: Pre-built metrics by name (tests inject fakes); built from
            RAGAS when omitted.
    """
    outcome = RagasOutcome(ragas_version=ragas_version())
    contexts = [c for c in contexts if c and c.strip()]
    if metrics is None:
        if not ragas_available():
            outcome.error_type, outcome.error = "RagasNotInstalled", INSTALL_HINT
            return outcome
        try:
            metrics = _build_metrics()
        except Exception as e:
            outcome.error_type, outcome.error = type(e).__name__, str(e)
            return outcome

    if not answer.strip() or not contexts:
        reason = "no answer" if not answer.strip() else "no retrieved contexts"
        outcome.skipped = dict.fromkeys(metrics, reason)
        return outcome

    has_reference = bool(reference and reference.strip())
    names = [n for n in RAGAS_METRICS if n in metrics]
    for name in names:
        if name in REFERENCE_METRICS and not has_reference:
            outcome.skipped[name] = "no reference answer"
    runnable = [n for n in names if n not in outcome.skipped]

    async def run(name: str) -> tuple[str, float | None, str | None]:
        try:
            result = await metrics[name].ascore(
                **_call_args(name, question, answer, contexts, reference)
            )
        except Exception as e:  # one metric failing must not drop the others
            return name, None, f"{type(e).__name__}: {e}"
        value = _as_float(result)
        return name, value, None if value is not None else "no numeric score returned"

    for name, value, err in await asyncio.gather(*(run(n) for n in runnable)):
        if value is not None:
            outcome.scores[name] = value
        elif err:
            outcome.errors[name] = err
    if not outcome.scores and outcome.errors:
        first = next(iter(outcome.errors.values()))
        outcome.error_type = first.split(":", 1)[0]
        outcome.error = first
        logger.warning("RAGAS scored nothing: %s", first)
    return outcome
