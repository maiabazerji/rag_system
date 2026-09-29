"""The versioned LLM-judge rubric.

Every judge score is only meaningful relative to the rubric that produced it.
This module is that rubric, written down: one definition per dimension and an
anchor description for each of the five allowed reference points
(0, 0.25, 0.5, 0.75, 1). The rendered text is embedded verbatim in the judge
prompt, and every eval run records :data:`RUBRIC_VERSION` and
:func:`rubric_fingerprint` so runs scored under different rubrics are never
compared as if they were the same measurement.

Changing any wording here changes the fingerprint. Bump :data:`RUBRIC_VERSION`
at the same time -- ``tests/test_eval_rubric_judge.py`` pins the fingerprint of
the current version and fails until you do.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

RUBRIC_VERSION = "2026-09.v2"

#: Reference points every dimension describes. Scores between anchors are
#: allowed; the anchors exist so different runs of the judge mean the same
#: thing by, say, 0.75.
ANCHORS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Dimension:
    """One scored dimension of the rubric.

    Attributes:
        name: JSON key the judge must use.
        title: Human-readable name.
        definition: What the dimension measures, in one or two sentences.
        anchors: Description of each anchor score, keyed by the values in
            :data:`ANCHORS`.
        requires_reference: Only scored when the example has a reference
            (ideal) answer.
    """

    name: str
    title: str
    definition: str
    anchors: dict[float, str]
    requires_reference: bool = False


DIMENSIONS_SPEC: tuple[Dimension, ...] = (
    Dimension(
        name="faithfulness",
        title="Faithfulness",
        definition=(
            "The share of factual claims in the generated answer that are supported by "
            "the retrieved context. Judge only against the context, not world knowledge: "
            "a true statement absent from the context is unsupported. A refusal that "
            "makes no claims is fully faithful."
        ),
        anchors={
            1.0: "Every claim is directly supported by the context.",
            0.75: "Nearly all claims are supported; one minor detail is unsupported.",
            0.5: "Roughly half the claims are supported; others are unsupported or embellished.",
            0.25: "Most claims are unsupported; only a few trace back to the context.",
            0.0: "No claim is supported, or the answer contradicts the context.",
        },
    ),
    Dimension(
        name="answer_relevance",
        title="Answer Relevance",
        definition=(
            "How directly and completely the answer addresses the question that was "
            "asked, regardless of whether it is correct. The answer is expected in the "
            "language of the question; an answer in another language scores at most 0.5."
        ),
        anchors={
            1.0: "Directly and completely answers the question, with no irrelevant material.",
            0.75: "Answers the question but misses a minor part or includes some padding.",
            0.5: "Partially answers, answers a related question, or is in the wrong language.",
            0.25: "Mostly tangential; touches the topic but does not answer the question.",
            0.0: "Off-topic, or a refusal when the question was answerable.",
        },
    ),
    Dimension(
        name="context_precision",
        title="Context Precision",
        definition=(
            "The share of the retrieved context passages that are relevant to answering "
            "the question, weighted toward the top: irrelevant passages ranked first "
            "cost more than irrelevant passages ranked last."
        ),
        anchors={
            1.0: "Every passage is relevant and the most useful ones come first.",
            0.75: "Most passages are relevant; a few irrelevant ones, mostly low in the ranking.",
            0.5: "About half the passages are relevant, or relevant ones are ranked low.",
            0.25: "Few passages are relevant.",
            0.0: "No passage is relevant to the question.",
        },
    ),
    Dimension(
        name="context_recall",
        title="Context Recall",
        definition=(
            "Whether the retrieved context contains all the information needed to answer "
            "the question fully (use the reference answer, when given, as the list of "
            "facts that must be present)."
        ),
        anchors={
            1.0: "Everything needed for a complete answer is in the context.",
            0.75: "The key facts are present; a minor supporting detail is missing.",
            0.5: "About half of the needed information is present.",
            0.25: "Only a small part of the needed information is present.",
            0.0: "The information needed to answer is absent.",
        },
    ),
    Dimension(
        name="answer_correctness",
        title="Answer Correctness",
        definition=(
            "Agreement between the generated answer and the reference answer: the same "
            "facts, nothing that contradicts it. Extra correct detail is not penalised; "
            "missing key facts and contradictions are."
        ),
        anchors={
            1.0: "States all key facts of the reference and contradicts none.",
            0.75: "States most key facts; one minor fact missing or imprecise.",
            0.5: "Partially correct: about half the key facts, or a notable omission.",
            0.25: "Mostly incorrect or missing the main point, with a little overlap.",
            0.0: "Contradicts the reference or is entirely wrong.",
        },
        requires_reference=True,
    ),
)

DIMENSION_BY_NAME: dict[str, Dimension] = {d.name: d for d in DIMENSIONS_SPEC}


def dimensions_for(has_reference: bool) -> tuple[Dimension, ...]:
    """Dimensions scored for an example with or without a reference answer."""
    return tuple(d for d in DIMENSIONS_SPEC if has_reference or not d.requires_reference)


def render_dimension(dim: Dimension) -> str:
    """Render one dimension's definition and anchors as prompt text."""
    lines = [f"**{dim.title}** (`{dim.name}`, 0-1): {dim.definition}"]
    for anchor in sorted(dim.anchors, reverse=True):
        lines.append(f"- {anchor:.2f} = {dim.anchors[anchor]}")
    return "\n".join(lines)


def render_rubric(has_reference: bool) -> str:
    """The rubric text embedded in the judge prompt."""
    header = (
        f"Rubric version {RUBRIC_VERSION}. Score each dimension on 0-1 using the anchors "
        "below. Pick the closest anchor; use a value between two anchors only when the "
        "answer sits clearly between their descriptions."
    )
    body = "\n\n".join(render_dimension(d) for d in dimensions_for(has_reference))
    return f"{header}\n\n{body}"


def rubric_fingerprint() -> str:
    """Short SHA-256 over the full rubric text (both variants).

    Recorded with every run so that an edit made without a version bump is
    still detectable after the fact.
    """
    text = RUBRIC_VERSION + "\n" + render_rubric(True) + "\n" + render_rubric(False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
