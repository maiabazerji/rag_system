"""Tiny on-disk graph store used by Graph RAG.

Why so simple?
==============
For an educational/demo project, a full graph DB (Neo4j, KuzuDB) is overkill.
We persist triples to a JSONL file and rebuild the in-memory index from it.
That keeps the data inspectable (`cat triples.jsonl`) and zero-deps.

Schema (one JSON per line):
    {"subject": "...", "predicate": "...", "object": "...",
     "chunk_id": "...", "doc_id": "..."}

Index:
    entity_to_chunks[entity]   = set[chunk_id]    # which chunks mention this entity
    entity_to_edges[entity]    = list[(predicate, neighbor, chunk_id)]
    chunk_to_doc[chunk_id]     = doc_id           # for access-control filtering

Read functions take an optional ``doc_ids`` set: when given, only triples from
those documents are visible. Callers pass the documents the requesting
principal may read, so a graph walk cannot surface relations (or chunk ids)
from documents outside the caller's access scope.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from app.config import settings


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    chunk_id: str
    doc_id: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "subject": self.subject,
                "predicate": self.predicate,
                "object": self.object,
                "chunk_id": self.chunk_id,
                "doc_id": self.doc_id,
            },
            ensure_ascii=False,
        )


def _norm(s: str) -> str:
    return s.strip().lower()


@dataclass
class GraphIndex:
    triples: list[Triple] = field(default_factory=list)
    entity_to_chunks: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    entity_to_edges: dict[str, list[tuple[str, str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    chunk_to_doc: dict[str, str] = field(default_factory=dict)
    _triple_ids: set[tuple[str, str, str, str]] = field(default_factory=set)

    def add(self, t: Triple) -> bool:
        tid = (t.subject, t.predicate, t.object, t.chunk_id)
        if tid in self._triple_ids:
            return False
        self._triple_ids.add(tid)
        self.triples.append(t)
        self.chunk_to_doc[t.chunk_id] = t.doc_id
        s, o = _norm(t.subject), _norm(t.object)
        self.entity_to_chunks[s].add(t.chunk_id)
        self.entity_to_chunks[o].add(t.chunk_id)
        self.entity_to_edges[s].append((t.predicate, t.object, t.chunk_id))
        self.entity_to_edges[o].append((f"inv:{t.predicate}", t.subject, t.chunk_id))
        return True

    @property
    def entities(self) -> list[str]:
        return sorted(self.entity_to_chunks.keys())

    def visible(self, chunk_id: str, doc_ids: Collection[str] | None) -> bool:
        """Whether a chunk's triples are visible when restricted to ``doc_ids``."""
        return doc_ids is None or self.chunk_to_doc.get(chunk_id) in doc_ids

    def entities_in(self, doc_ids: Collection[str] | None) -> list[str]:
        """Entities mentioned by at least one visible chunk."""
        if doc_ids is None:
            return self.entities
        return sorted(
            e
            for e, chunks in self.entity_to_chunks.items()
            if any(self.visible(c, doc_ids) for c in chunks)
        )


_INDEX: GraphIndex | None = None
# Reentrant: writers call load() while holding it, so the index they update is
# the one currently installed, never one a concurrent remove_docs has replaced.
_LOCK = RLock()


def _triples_path() -> Path:
    p = Path(settings.graph_data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "triples.jsonl"


def load() -> GraphIndex:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        idx = GraphIndex()
        path = _triples_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                idx.add(Triple(**d))
        _INDEX = idx
        return idx


def append(triples: list[Triple]) -> dict:
    """Append triples, deduplicating. Returns {added: int, skipped: int}."""
    if not triples:
        return {"added": 0, "skipped": 0}
    path = _triples_path()
    added, skipped = 0, 0
    # load() inside the lock: a remove_docs() between fetching the index and
    # writing would otherwise leave these triples in an index it has replaced.
    with _LOCK:
        idx = load()
        with path.open("a", encoding="utf-8") as f:
            for t in triples:
                if idx.add(t):
                    f.write(t.to_json() + "\n")
                    added += 1
                else:
                    skipped += 1
    return {"added": added, "skipped": skipped}


def remove_docs(doc_ids: Collection[str]) -> int:
    """Drop every triple extracted from the given document revisions.

    Triples are tied to the chunk they came from, and chunk ids derive from the
    document's content hash, so re-indexing an edited document strands the old
    revision's triples: they point at chunks the vector store no longer holds,
    and an entity walk keeps surfacing text that is no longer in the corpus.
    Calling this when the vector store drops a revision keeps the two in step.

    The index is rebuilt from the survivors rather than unpicked in place: the
    entity maps are many-to-one, so an entity may be reachable through triples
    from several documents and cannot simply be deleted alongside one of them.

    Args:
        doc_ids: Document ids whose triples should be removed.

    Returns:
        The number of triples removed.
    """
    global _INDEX
    if not doc_ids:
        return 0

    with _LOCK:
        idx = load()
        survivors = [t for t in idx.triples if t.doc_id not in doc_ids]
        removed = len(idx.triples) - len(survivors)
        if not removed:
            return 0

        rebuilt = GraphIndex()
        for t in survivors:
            rebuilt.add(t)

        # Write beside the store and swap, so an interrupted rewrite cannot
        # leave a half-truncated file where the graph used to be.
        path = _triples_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            "".join(t.to_json() + "\n" for t in survivors), encoding="utf-8"
        )
        tmp.replace(path)
        _INDEX = rebuilt
        return removed


def reset() -> None:
    """Wipe the on-disk store and the in-memory index. Used by /graph/rebuild."""
    global _INDEX
    with _LOCK:
        path = _triples_path()
        if path.exists():
            path.unlink()
        _INDEX = None


def stats(doc_ids: Collection[str] | None = None) -> dict:
    """Graph size, counting only triples from ``doc_ids`` when given."""
    idx = load()
    triples = [t for t in idx.triples if doc_ids is None or t.doc_id in doc_ids]
    return {
        "triples": len(triples),
        "entities": len(idx.entities_in(doc_ids)),
        "chunks_indexed": len({t.chunk_id for t in triples}),
        "docs_indexed": len({t.doc_id for t in triples}),
    }


def neighbors(
    entity: str, hops: int = 1, doc_ids: Collection[str] | None = None
) -> tuple[set[str], set[str]]:
    """Return (chunks_touched, related_entities) within `hops` of `entity`.

    With ``doc_ids``, only edges extracted from those documents are walked.
    """
    idx = load()
    e0 = _norm(entity)
    seen_entities: set[str] = {e0}
    chunks: set[str] = {
        c for c in idx.entity_to_chunks.get(e0, set()) if idx.visible(c, doc_ids)
    }
    frontier = {e0}
    for _ in range(hops):
        next_frontier: set[str] = set()
        for e in frontier:
            for _pred, neigh, cid in idx.entity_to_edges.get(e, []):
                if not idx.visible(cid, doc_ids):
                    continue
                n = _norm(neigh)
                chunks.add(cid)
                if n not in seen_entities:
                    seen_entities.add(n)
                    next_frontier.add(n)
        frontier = next_frontier
        if not frontier:
            break
    seen_entities.discard(e0)
    return chunks, seen_entities


def describe_subgraph(
    entities: list[str], max_edges: int = 30, doc_ids: Collection[str] | None = None
) -> str:
    """Pretty-print a small subgraph as text the LLM can read.

    With ``doc_ids``, only edges extracted from those documents are shown.
    """
    idx = load()
    seen: set[tuple[str, str, str]] = set()
    lines: list[str] = []
    for e in entities:
        for pred, neigh, cid in idx.entity_to_edges.get(_norm(e), []):
            if pred.startswith("inv:") or not idx.visible(cid, doc_ids):
                continue
            key = (_norm(e), pred, _norm(neigh))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {e}-[{pred}]→ {neigh}")
            if len(lines) >= max_edges:
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(no edges found)"
