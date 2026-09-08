"""Tests for the on-disk triple store behind Graph RAG.

The store is a JSONL file plus derived entity maps rebuilt from it, so the
thing worth testing is that the two never disagree: whatever survives a
removal must be exactly what the maps still point at.
"""
import pytest

from app.rag import graph_store
from app.rag.graph_store import Triple


@pytest.fixture
def graph(tmp_path, settings, monkeypatch):
    """An empty graph store rooted in tmp_path.

    `_INDEX` is a module-level cache, so it has to be cleared or a previous
    test's graph leaks into this one.
    """
    monkeypatch.setattr(settings, "graph_data_dir", str(tmp_path))
    monkeypatch.setattr(graph_store, "_INDEX", None)
    return graph_store


def _triple(subject, obj, chunk_id, doc_id):
    return Triple(
        subject=subject,
        predicate="relates_to",
        object=obj,
        chunk_id=chunk_id,
        doc_id=doc_id,
    )


class TestRemoveDocs:
    def test_drops_only_the_named_revisions(self, graph):
        graph.append([
            _triple("bm25", "sparse retrieval", "old:0", "old"),
            _triple("rrf", "fusion", "new:0", "new"),
        ])

        assert graph.remove_docs({"old"}) == 1

        remaining = graph.load().triples
        assert [t.doc_id for t in remaining] == ["new"]

    def test_entity_maps_forget_the_removed_document(self, graph):
        graph.append([_triple("bm25", "sparse retrieval", "old:0", "old")])
        graph.remove_docs({"old"})

        index = graph.load()
        assert index.entities == []
        assert graph.neighbors("bm25") == (set(), set())

    def test_an_entity_shared_with_another_document_survives(self, graph):
        """The regression this guards: entity maps are many-to-one, so deleting
        a document's entities outright would erase entities another document
        still mentions."""
        graph.append([
            _triple("bm25", "sparse retrieval", "old:0", "old"),
            _triple("bm25", "hybrid search", "new:0", "new"),
        ])

        graph.remove_docs({"old"})

        assert "bm25" in graph.load().entities
        chunks, _ = graph.neighbors("bm25")
        assert chunks == {"new:0"}

    def test_the_removal_survives_a_reload_from_disk(self, graph, monkeypatch):
        graph.append([
            _triple("bm25", "sparse retrieval", "old:0", "old"),
            _triple("rrf", "fusion", "new:0", "new"),
        ])
        graph.remove_docs({"old"})

        monkeypatch.setattr(graph_store, "_INDEX", None)  # force a re-read

        assert [t.doc_id for t in graph.load().triples] == ["new"]

    def test_removing_nothing_is_a_no_op(self, graph):
        graph.append([_triple("bm25", "sparse retrieval", "c:0", "doc")])

        assert graph.remove_docs(set()) == 0
        assert graph.remove_docs({"never-indexed"}) == 0
        assert len(graph.load().triples) == 1

    def test_stats_reflect_the_removal(self, graph):
        graph.append([
            _triple("bm25", "sparse retrieval", "old:0", "old"),
            _triple("rrf", "fusion", "new:0", "new"),
        ])
        graph.remove_docs({"old"})

        assert graph.stats() == {
            "triples": 1,
            "entities": 2,
            "chunks_indexed": 1,
            "docs_indexed": 1,
        }
