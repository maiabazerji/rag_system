"""Golden dataset schema and offline validation.

A golden example is one JSON object per line. Recognised fields:

``question`` (required)
    The question put to the pipeline.
``ideal_answer`` / ``expected_answer``
    Reference answer; enables the judge's ``answer_correctness``. The two
    names are aliases; ``ideal_answer`` wins if both are present.
``relevant_doc_ids`` / ``expected_sources``
    Documents relevant to the question (filenames or extension-less ids),
    used for deterministic retrieval metrics. ``relevant_doc_ids`` wins.
``question_type``, ``difficulty``
    Free-form labels used to break aggregates down.
``expected_behavior``
    ``answer`` (default) or ``refuse``. A ``refuse`` example is unanswerable
    from the corpus: it is scored by whether the system declined, and is
    excluded from the judge and from retrieval metrics.
``id``
    Stable example id; defaults to the 1-based line position (``q001``...).
``note`` / ``notes``
    Free text; explains, among other things, why an example deliberately
    carries no relevance labels.
``evidence``
    Supporting passages or pointers; carried through, not interpreted.

Unknown fields are kept (datasets may carry extra annotations).

:func:`validate_dataset_file` checks a file without calling any model:
JSON, schema, duplicate ids and questions, and that every referenced document
exists in the corpus directory. ``scripts/run_eval.py --dry-run`` uses it.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.eval.retrieval import doc_key, relevant_documents

UNSPECIFIED = "unspecified"


class GoldenExample(BaseModel):
    """Schema of one golden example (see the module docstring)."""

    model_config = ConfigDict(extra="allow")

    id: str | int | None = None
    question: str = Field(min_length=1)
    ideal_answer: str | None = None
    expected_answer: str | None = None
    expected_sources: list[str] | None = None
    relevant_doc_ids: list[str] | None = None
    question_type: str | None = None
    difficulty: str | int | None = None
    expected_behavior: Literal["answer", "refuse"] | None = None
    note: str | None = None
    notes: str | None = None
    evidence: Any = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question is blank")
        return v


def ideal_answer_of(example: Mapping[str, Any]) -> str | None:
    """The reference answer under either field name, or ``None``."""
    for key in ("ideal_answer", "expected_answer"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def example_id(example: Mapping[str, Any], index: int) -> str:
    """Stable id: the example's own ``id`` or its 1-based position."""
    raw = example.get("id")
    return str(raw) if raw not in (None, "") else f"q{index + 1:03d}"


def question_type_of(example: Mapping[str, Any]) -> str:
    value = example.get("question_type")
    return str(value) if value not in (None, "") else UNSPECIFIED


def difficulty_of(example: Mapping[str, Any]) -> str | None:
    value = example.get("difficulty")
    return str(value) if value not in (None, "") else None


def expected_behavior_of(example: Mapping[str, Any]) -> str:
    """``"refuse"`` for unanswerable examples, else ``"answer"``."""
    return "refuse" if example.get("expected_behavior") == "refuse" else "answer"


def corpus_keys(corpus_dir: Path) -> set[str]:
    """Document keys (see :func:`~app.eval.retrieval.doc_key`) of every file
    under ``corpus_dir``, recursively."""
    if not corpus_dir.is_dir():
        return set()
    return {doc_key(p.name) for p in corpus_dir.rglob("*") if p.is_file()}


@dataclass
class DatasetReport:
    """Outcome of validating one dataset file."""

    path: str
    n_examples: int = 0
    n_with_reference: int = 0
    n_with_relevance: int = 0
    n_expect_refusal: int = 0
    question_types: dict[str, int] = field(default_factory=dict)
    difficulties: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "n_examples": self.n_examples,
            "n_with_reference": self.n_with_reference,
            "n_with_relevance": self.n_with_relevance,
            "n_expect_refusal": self.n_expect_refusal,
            "question_types": self.question_types,
            "difficulties": self.difficulties,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_dataset_file(path: Path, corpus_dir: Path | None) -> DatasetReport:
    """Validate a golden JSONL file without calling any model.

    Errors (the file is unusable or would silently mis-score): invalid JSON,
    schema violations, duplicate ids, relevance labels naming a document that
    is not in the corpus. Warnings (legal but suspicious): duplicate questions,
    examples with neither relevance labels nor a ``note``, examples with no
    reference answer, an empty or missing corpus directory.

    Args:
        path: The ``.jsonl`` file.
        corpus_dir: Directory of indexed documents; ``None`` skips the
            existence check.
    """
    report = DatasetReport(path=str(path))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        report.errors.append(f"cannot read file: {e}")
        return report

    known = corpus_keys(corpus_dir) if corpus_dir is not None else None
    if corpus_dir is not None and not known:
        report.warnings.append(f"corpus directory {corpus_dir} is missing or empty")
        known = None

    ids: Counter[str] = Counter()
    questions: Counter[str] = Counter()
    types: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    index = 0
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"line {lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            report.errors.append(f"{where}: invalid JSON ({e.msg})")
            continue
        if not isinstance(raw, dict):
            report.errors.append(f"{where}: expected an object, got {type(raw).__name__}")
            continue
        try:
            GoldenExample.model_validate(raw)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "(root)"
                report.errors.append(f"{where}: {loc}: {err['msg']}")
            continue

        report.n_examples += 1
        ids[example_id(raw, index)] += 1
        questions[raw["question"].strip()] += 1
        types[question_type_of(raw)] += 1
        diff = difficulty_of(raw)
        if diff is not None:
            difficulties[diff] += 1
        index += 1

        refuse = expected_behavior_of(raw) == "refuse"
        if refuse:
            report.n_expect_refusal += 1
        if ideal_answer_of(raw):
            report.n_with_reference += 1
        elif not refuse:
            report.warnings.append(f"{where}: no ideal_answer/expected_answer")

        relevant = relevant_documents(raw)
        if relevant:
            # Refusal examples are never retrieval-scored, but a label naming a
            # missing document is still a broken label.
            if not refuse:
                report.n_with_relevance += 1
            if known is not None:
                missing = [r for r in relevant if doc_key(r) not in known]
                if missing:
                    report.errors.append(
                        f"{where}: relevant documents not in corpus: {', '.join(missing)}"
                    )
        elif not refuse and not (raw.get("note") or raw.get("notes")):
            report.warnings.append(
                f"{where}: no relevant_doc_ids/expected_sources and no note explaining why"
            )

    for dup, count in ids.items():
        if count > 1:
            report.errors.append(f"duplicate id {dup!r} ({count} times)")
    for dup, count in questions.items():
        if count > 1:
            report.warnings.append(f"duplicate question ({count} times): {dup[:60]!r}")
    if report.n_examples == 0 and not report.errors:
        report.errors.append("no examples")

    report.question_types = dict(sorted(types.items()))
    report.difficulties = dict(sorted(difficulties.items()))
    return report
