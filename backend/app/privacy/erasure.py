"""Right to erasure (GDPR art. 17): remove a document, a source or a principal everywhere.

An erasure runs in two phases:

1. **Resolve** the subject into an :class:`ErasureRequest`: the document ids
   (and filenames) it covers, read from the vector store, plus the principal
   when one is being erased. Resolution happens once, up front, so every
   target sees the same ids even after the vectors themselves are gone.
2. **Erase** by calling every registered target with that request. Each target
   returns how many records it removed; the counts make up the report.

Targets are plain callables registered with :func:`register_erasure_target`,
so stores added later (an audit log, an external tracing backend) plug in
without touching this module. Every target must be idempotent: repeating an
erasure, e.g. after a partial failure, removes what is left and reports 0 for
the rest.

Eval runs are scrubbed, not deleted: rows whose answer drew on an erased
document keep their scores (so historical aggregates and regression baselines
stay comparable) but have their answer text and cited filenames replaced with
``[erased]``.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client.http import models as qm

from app.logging_config import get_structured_logger
from app.rag import graph_store
from app.rag.store import delete_matching, scroll_payloads
from app.tracing import erase_traces

logger = get_structured_logger(__name__)

# Payload keys identifying who owns a chunk and which source it came from.
# Chunks indexed before source keys existed are matched on their filename.
OWNER_PAYLOAD_KEY = "owner"
SOURCE_PAYLOAD_KEY = "source_key"

ERASED = "[erased]"


@dataclass(frozen=True)
class ErasureRequest:
    """What an erasure covers, as resolved before any target runs.

    Attributes:
        subject: Human-readable description, e.g. ``document:abc123``.
        doc_ids: Every document id to erase.
        filenames: Filenames of those documents, for stores that only know
            documents by name (older eval runs).
        principal_id: The principal being erased, if any. Targets holding
            per-principal data (traces, usage, audit) erase by this too.
    """

    subject: str
    doc_ids: frozenset[str] = frozenset()
    filenames: frozenset[str] = frozenset()
    principal_id: str | None = None


ErasureTarget = Callable[[ErasureRequest], int | Awaitable[int]]


@dataclass
class ErasureReport:
    """Per-target outcome of an erasure."""

    subject: str
    doc_ids: list[str]
    targets: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "doc_ids": self.doc_ids,
            "targets": self.targets,
            "errors": self.errors,
            "complete": self.complete,
        }


_targets: dict[str, ErasureTarget] = {}


def register_erasure_target(name: str, fn: ErasureTarget) -> None:
    """Add (or replace) a store to erase from.

    Args:
        name: Key under which the target's count appears in the report.
        fn: Called with the :class:`ErasureRequest`; returns, or resolves to,
            the number of records removed. Must be idempotent.
    """
    _targets[name] = fn


def unregister_erasure_target(name: str) -> None:
    """Remove a target. A no-op if it is not registered."""
    _targets.pop(name, None)


def erasure_targets() -> list[str]:
    """Names of the registered targets, in the order they run."""
    return list(_targets)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _match(key: str, value: str) -> qm.FieldCondition:
    return qm.FieldCondition(key=key, match=qm.MatchValue(value=value))


async def _documents_matching(selector: qm.Filter) -> tuple[set[str], set[str]]:
    payloads = await scroll_payloads(selector, fields=["doc_id", "filename"])
    doc_ids = {p["doc_id"] for p in payloads if p.get("doc_id")}
    filenames = {p["filename"] for p in payloads if p.get("filename")}
    return doc_ids, filenames


async def resolve_documents(doc_ids: Iterable[str]) -> ErasureRequest:
    """Erase specific documents. Ids no longer in the vector store are kept,
    so a retried erasure still reaches the other stores."""
    wanted = sorted(set(doc_ids))
    if not wanted:
        return ErasureRequest(subject="document:")
    _, filenames = await _documents_matching(
        qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=wanted))])
    )
    return ErasureRequest(
        subject="document:" + ",".join(wanted),
        doc_ids=frozenset(wanted),
        filenames=frozenset(filenames),
    )


async def resolve_source(source_key: str) -> ErasureRequest:
    """Erase every document ingested from one source (all its revisions)."""
    doc_ids, filenames = await _documents_matching(
        qm.Filter(
            should=[_match(SOURCE_PAYLOAD_KEY, source_key), _match("filename", source_key)]
        )
    )
    return ErasureRequest(
        subject=f"source:{source_key}",
        doc_ids=frozenset(doc_ids),
        filenames=frozenset(filenames | {source_key}),
    )


async def resolve_principal(principal_id: str) -> ErasureRequest:
    """Erase every document a principal owns, and their own records."""
    doc_ids, filenames = await _documents_matching(
        qm.Filter(must=[_match(OWNER_PAYLOAD_KEY, principal_id)])
    )
    return ErasureRequest(
        subject=f"principal:{principal_id}",
        doc_ids=frozenset(doc_ids),
        filenames=frozenset(filenames),
        principal_id=principal_id,
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


async def erase(request: ErasureRequest) -> ErasureReport:
    """Run every registered target against ``request``.

    A failing target does not stop the others; its error is recorded and the
    report is marked incomplete, so the caller can retry.
    """
    report = ErasureReport(subject=request.subject, doc_ids=sorted(request.doc_ids))
    for name, fn in list(_targets.items()):
        try:
            result = fn(request)
            if inspect.isawaitable(result):
                result = await result
            report.targets[name] = int(result)
        except Exception as e:
            report.errors[name] = f"{type(e).__name__}: {e}"
            logger.error(
                f"Erasure target '{name}' failed: {type(e).__name__}: {e}",
                extra_fields={"target": name, "subject": request.subject},
            )
    logger.info(
        "Erasure completed" if report.complete else "Erasure incomplete",
        extra_fields={
            "subject": request.subject,
            "doc_ids": len(request.doc_ids),
            "targets": report.targets,
            "failed_targets": sorted(report.errors),
        },
    )
    return report


# ---------------------------------------------------------------------------
# Built-in targets
# ---------------------------------------------------------------------------


async def erase_vectors(request: ErasureRequest) -> int:
    """Delete chunks of the resolved documents (and any the principal owns)."""
    conditions: list[qm.Condition] = []
    if request.doc_ids:
        conditions.append(
            qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=sorted(request.doc_ids)))
        )
    if request.principal_id is not None:
        conditions.append(_match(OWNER_PAYLOAD_KEY, request.principal_id))
    if not conditions:
        return 0
    return await delete_matching(qm.Filter(should=conditions))


def erase_graph(request: ErasureRequest) -> int:
    """Delete knowledge-graph triples extracted from the resolved documents."""
    return graph_store.remove_docs(request.doc_ids)


def erase_trace_records(request: ErasureRequest) -> int:
    """Delete traces that touched the documents or were made by the principal."""
    return erase_traces(request.doc_ids, request.principal_id)


def _runs_dir() -> Path:
    # Read at call time: tests (and deployments) repoint metrics.RUNS_DIR.
    from app.eval import metrics

    return metrics.RUNS_DIR


def _scrub_row(row: dict, doc_ids: Collection[str], filenames: Collection[str]) -> bool:
    retrieval = row.get("retrieval") or {}
    cited_docs = set(row.get("doc_ids") or ())
    cited_files = set(retrieval.get("retrieved") or ())
    if not (cited_docs & set(doc_ids) or cited_files & set(filenames)):
        return False

    row["answer"] = ERASED
    row["doc_ids"] = sorted(cited_docs - set(doc_ids))
    for key in ("retrieved", "expected"):
        if isinstance(retrieval.get(key), list):
            retrieval[key] = [ERASED if f in filenames else f for f in retrieval[key]]
    row["erased"] = True
    return True


def scrub_eval_runs(doc_ids: Collection[str], filenames: Collection[str] = ()) -> int:
    """Replace answers and filenames that cite erased documents in saved eval runs.

    Rows are matched on the document ids they cite, or, for runs saved before
    rows recorded document ids, on the filenames retrieval cited. Scores are
    kept. A scrubbed row no longer matches, so scrubbing again changes nothing.

    Returns:
        The number of rows scrubbed.
    """
    runs_dir = _runs_dir()
    if not (doc_ids or filenames) or not runs_dir.exists():
        return 0

    scrubbed = 0
    for path in sorted(runs_dir.glob("*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping unreadable eval run {path.name}: {type(e).__name__}")
            continue
        changed = sum(
            _scrub_row(row, doc_ids, filenames)
            for row in run.get("per_example") or ()
            if isinstance(row, dict)
        )
        if changed:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(run, indent=2), encoding="utf-8")
            tmp.replace(path)
            scrubbed += changed
    return scrubbed


def erase_eval_runs(request: ErasureRequest) -> int:
    return scrub_eval_runs(request.doc_ids, request.filenames)


def erase_usage(request: ErasureRequest) -> int:
    """Delete the principal's usage rows. Nothing to do for document erasures."""
    if request.principal_id is None:
        return 0
    from app.privacy.usage import delete_usage

    return delete_usage(request.principal_id)


def register_builtin_targets() -> None:
    """(Re)register the built-in targets. Called at import."""
    register_erasure_target("vectors", erase_vectors)
    register_erasure_target("graph", erase_graph)
    register_erasure_target("traces", erase_trace_records)
    register_erasure_target("eval_runs", erase_eval_runs)
    register_erasure_target("usage", erase_usage)


register_builtin_targets()
