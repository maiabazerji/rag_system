"""Tests for the Qdrant store wrapper and the code that depends on its contract.

The contract that matters: an outage is an exception (`StoreUnavailable`),
never an empty result, because callers such as the graph build decide what to
delete from what the store says exists.
"""
import importlib.util
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qm

from app.rag import graph_store, store
from app.rag.graph_store import Triple
from app.rag.store import StoreUnavailable
from app.resilience import CircuitBreaker
from app.schemas import Chunk


@pytest.fixture
def qdrant(mock_qdrant_client, monkeypatch):
    """Install a mock Qdrant client and a fresh, closed circuit breaker."""
    monkeypatch.setattr(store, "_client", mock_qdrant_client)
    monkeypatch.setattr(
        store, "_qdrant_breaker", CircuitBreaker(failure_threshold=5, service_name="Qdrant")
    )
    return mock_qdrant_client


@pytest.fixture
def open_breaker(monkeypatch):
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=3600, service_name="Qdrant")
    breaker.record_failure()
    monkeypatch.setattr(store, "_qdrant_breaker", breaker)


@pytest.fixture
def graph(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "graph_data_dir", str(tmp_path))
    monkeypatch.setattr(graph_store, "_INDEX", None)
    return graph_store


def _record(chunk_id, doc_id, text="some text"):
    return qm.Record(
        id=store._point_id(chunk_id),
        payload={"chunk_id": chunk_id, "doc_id": doc_id, "text": text},
    )


def _triple(chunk_id, doc_id, subject="a", obj="b"):
    return Triple(subject=subject, predicate="rel", object=obj, chunk_id=chunk_id, doc_id=doc_id)


class TestOutagesRaise:
    async def test_search_raises_on_qdrant_error(self, qdrant):
        qdrant.query_points.side_effect = ConnectionError("down")
        with pytest.raises(StoreUnavailable):
            await store.search([0.0] * 384)

    async def test_count_raises_on_qdrant_error(self, qdrant):
        qdrant.count.side_effect = ConnectionError("down")
        with pytest.raises(StoreUnavailable):
            await store.count()

    @pytest.mark.parametrize(
        "call",
        [
            lambda: store.search([0.0] * 384),
            lambda: store.count(),
            lambda: store.fetch_chunks(["x:0"]),
            lambda: store.scroll_chunks(),
        ],
    )
    async def test_open_breaker_raises(self, open_breaker, call):
        with pytest.raises(StoreUnavailable, match="circuit breaker open"):
            await call()

    async def test_count_still_works(self, qdrant):
        assert await store.count() == 10

    def test_is_mapped_to_503(self, client):
        with patch(
            "app.api.ingest.store_count",
            new=AsyncMock(side_effect=StoreUnavailable("Qdrant count failed")),
        ):
            r = client.get("/ingest/stats")
        assert r.status_code == 503
        assert r.json()["code"] == "store_unavailable"


class TestFetchChunks:
    async def test_looks_points_up_by_id(self, qdrant):
        qdrant.retrieve = AsyncMock(return_value=[_record("d:1", "d")])

        hits = await store.fetch_chunks(["d:1", "d:2", "d:1"])

        kwargs = qdrant.retrieve.await_args.kwargs
        assert kwargs["ids"] == [store._point_id("d:1"), store._point_id("d:2")]
        assert kwargs["with_payload"] is True
        assert [h.payload["chunk_id"] for h in hits] == ["d:1"]
        qdrant.query_points.assert_not_awaited()

    async def test_no_ids_no_request(self, qdrant):
        qdrant.retrieve = AsyncMock()
        assert await store.fetch_chunks([]) == []
        qdrant.retrieve.assert_not_awaited()

    async def test_graph_strategy_fetches_by_id(self):
        from app.rag.strategies.graph import _fetch_chunks_by_id

        with patch(
            "app.rag.strategies.graph.fetch_chunks",
            new=AsyncMock(return_value=[_record("d:1", "d", "hello")]),
        ) as mock_fetch:
            chunks = await _fetch_chunks_by_id({"d:1"})

        mock_fetch.assert_awaited_once()
        assert [c.id for c in chunks] == ["d:1"]
        assert chunks[0].text == "hello"

    async def test_agentic_fetch_tool_fetches_by_id(self):
        from app.rag.strategies.agentic import AgenticRAG

        captured = {}

        async def fake_loop(**kwargs):
            captured["out"] = await kwargs["tool_handlers"]["fetch_chunk"]({"chunk_id": "d:1"})
            return {"text": "", "input_tokens": 0, "output_tokens": 0, "iterations": 1, "trace": []}

        with (
            patch("app.rag.strategies.agentic.tool_use_loop", new=fake_loop),
            patch(
                "app.rag.strategies.agentic.fetch_chunks",
                new=AsyncMock(return_value=[_record("d:1", "d", "chunk body")]),
            ) as mock_fetch,
            patch("app.rag.strategies.agentic.vector_search", new=AsyncMock()) as mock_search,
        ):
            await AgenticRAG().run("q", top_k=8, model="claude-sonnet-5", prompt_version="v1")

        mock_fetch.assert_awaited_once_with(["d:1"])
        mock_search.assert_not_awaited()
        assert captured["out"] == "chunk body"


class TestScrollChunks:
    async def test_pages_through_the_whole_collection(self, qdrant, monkeypatch):
        monkeypatch.setattr(store, "_SCROLL_PAGE", 2)
        pages = [
            ([_record("d:0", "d"), _record("d:1", "d")], 11),
            ([_record("d:2", "d"), _record("e:0", "e")], 22),
            ([_record("e:1", "e")], None),
        ]
        qdrant.scroll = AsyncMock(side_effect=pages)

        records = await store.scroll_chunks()

        assert len(records) == 5
        offsets = [c.kwargs["offset"] for c in qdrant.scroll.await_args_list]
        assert offsets == [None, 11, 22]

    async def test_respects_a_limit(self, qdrant, monkeypatch):
        monkeypatch.setattr(store, "_SCROLL_PAGE", 2)
        qdrant.scroll = AsyncMock(
            side_effect=[([_record("d:0", "d"), _record("d:1", "d")], 11), ([_record("d:2", "d")], 22)]
        )

        records = await store.scroll_chunks(limit=3)

        assert len(records) == 3
        assert qdrant.scroll.await_args_list[1].kwargs["limit"] == 1

    async def test_a_failed_page_raises_rather_than_returning_part(self, qdrant):
        qdrant.scroll = AsyncMock(
            side_effect=[([_record("d:0", "d")], 11), ConnectionError("down")]
        )
        with pytest.raises(StoreUnavailable):
            await store.scroll_chunks()


class TestUpsert:
    async def test_batches_points(self, qdrant):
        chunks = [Chunk(id=f"d:{i}", doc_id="d", text="t", tokens=1) for i in range(300)]
        await store.upsert(chunks, [[0.0] * 3] * 300)

        sizes = [len(c.kwargs["points"]) for c in qdrant.upsert.await_args_list]
        assert sizes == [128, 128, 44]

    async def test_failure_raises_store_unavailable(self, qdrant):
        qdrant.upsert.side_effect = ConnectionError("down")
        chunks = [Chunk(id="d:0", doc_id="d", text="t", tokens=1)]
        with pytest.raises(StoreUnavailable):
            await store.upsert(chunks, [[0.0] * 3])


class TestClose:
    async def test_closes_and_forgets_the_client(self, monkeypatch):
        c = AsyncMock()
        monkeypatch.setattr(store, "_client", c)
        await store.close()
        c.close.assert_awaited_once()
        assert store._client is None
        await store.close()  # idempotent

    async def test_embedding_model_loads_off_the_event_loop(self, mock_qdrant_client):
        loop_thread = threading.get_ident()
        seen = {}

        def fake_dim():
            seen["thread"] = threading.get_ident()
            return 384

        with patch("app.rag.store.embedding_dim", new=fake_dim):
            await store._ensure_collection(mock_qdrant_client)
        assert seen["thread"] != loop_thread


class TestStaleRevisions:
    async def test_matches_on_source_key_not_filename(self, qdrant):
        qdrant.count = AsyncMock(return_value=MagicMock(count=0))

        await store.delete_stale_revisions("key:7:docs/notes.md", "newrev")

        selector = qdrant.count.await_args.kwargs["count_filter"]
        (cond,) = selector.must
        assert cond.key == "source_key"
        assert cond.match.value == "key:7:docs/notes.md"

    async def test_ingest_records_and_uses_an_owner_scoped_key(self):
        from app.rag.ingest import enqueue_document

        with (
            patch("app.rag.ingest.embed_texts_async", new=AsyncMock(return_value=[[0.0]])),
            patch("app.rag.ingest.upsert", new=AsyncMock()) as mock_upsert,
            patch(
                "app.rag.ingest.delete_stale_revisions",
                new=AsyncMock(return_value=store.StaleRevisions()),
            ) as mock_delete,
        ):
            await enqueue_document("sub/a.txt", b"hello world", owner="key:7")

        chunk = mock_upsert.await_args.args[0][0]
        assert chunk.metadata["source_key"] == "key:7:sub/a.txt"
        assert mock_delete.await_args.args[0] == "key:7:sub/a.txt"

    def test_upload_owner_comes_from_the_api_key(self, client):
        with patch(
            "app.api.ingest.enqueue_document",
            new=AsyncMock(return_value={"chunks": 1}),
        ) as mock_enqueue, patch("app.api.ingest.store_count", new=AsyncMock(return_value=1)):
            r = client.post("/ingest", files={"file": ("a.txt", b"hello")})
        assert r.status_code == 200
        assert mock_enqueue.await_args.kwargs["owner"] == "local"

    def test_script_names_files_relative_to_the_ingest_root(self, tmp_path):
        path = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"
        spec = importlib.util.spec_from_file_location("ingest_script", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        (tmp_path / "a").mkdir()
        f = tmp_path / "a" / "notes.md"
        f.write_text("x")
        assert mod._relative_name(f, tmp_path) == "a/notes.md"
        assert mod._relative_name(mod.ROOT / "README.md", tmp_path) == "README.md"


class TestGraphBuild:
    async def _build(self, limit=None):
        from app.api import graph as graph_api

        job_id = "job"
        graph_api._JOBS.pop(job_id, None)
        await graph_api._run_build(job_id, limit, 2)
        return graph_api._JOBS[job_id]

    async def test_store_outage_fails_the_job_and_prunes_nothing(self, graph):
        graph.append([_triple("d:0", "d")])
        with patch(
            "app.api.graph.scroll_chunks",
            new=AsyncMock(side_effect=StoreUnavailable("Qdrant scroll failed")),
        ):
            job = await self._build()

        assert job["status"] == "failed"
        assert "nothing was pruned" in job["error"]
        assert len(graph.load().triples) == 1

    async def test_empty_store_beside_a_populated_graph_prunes_nothing(self, graph):
        graph.append([_triple("d:0", "d")])
        with patch("app.api.graph.scroll_chunks", new=AsyncMock(return_value=[])):
            job = await self._build()

        assert job["status"] == "failed"
        assert "refusing to prune" in job["error"]
        assert len(graph.load().triples) == 1

    async def test_full_build_sees_more_than_2048_chunks(self, graph):
        """The old similarity-search probe capped at 2048 hits, so every document
        past that looked orphaned and had its triples pruned."""
        graph.append([_triple("d:0", "d"), _triple("gone:0", "gone")])
        records = [_record(f"d:{i}", "d") for i in range(2500)]
        records[-1] = _record("late:0", "late")
        with (
            patch("app.api.graph.scroll_chunks", new=AsyncMock(return_value=records)) as scroll,
            patch("app.api.graph.extract_triples", new=AsyncMock(return_value=[])),
        ):
            job = await self._build()

        scroll.assert_awaited_once_with(limit=None)
        assert job["status"] == "completed"
        assert job["pruned_triples"] == 1
        assert {t.doc_id for t in graph.load().triples} == {"d"}
        assert job["processed_chunks"] == 2499  # everything but the already-done d:0

    async def test_limited_build_prunes_nothing(self, graph):
        graph.append([_triple("gone:0", "gone")])
        with (
            patch(
                "app.api.graph.scroll_chunks", new=AsyncMock(return_value=[_record("d:0", "d")])
            ) as scroll,
            patch("app.api.graph.extract_triples", new=AsyncMock(return_value=[])),
        ):
            job = await self._build(limit=1)

        scroll.assert_awaited_once_with(limit=1)
        assert job["status"] == "completed"
        assert len(graph.load().triples) == 1


class TestGraphAdminRoutes:
    @pytest.mark.parametrize("path", ["/graph/reset", "/graph/build"])
    def test_requires_the_admin_key(self, client, settings, monkeypatch, path):
        monkeypatch.setattr(settings, "admin_key", "correct-horse")
        assert client.post(path).status_code == 403
        assert client.post(path, headers={"X-Admin-Key": "wrong"}).status_code == 403

    def test_reset_with_the_admin_key(self, client, settings, monkeypatch, graph):
        monkeypatch.setattr(settings, "admin_key", "correct-horse")
        r = client.post("/graph/reset", headers={"X-Admin-Key": "correct-horse"})
        assert r.status_code == 200
        assert r.json()["triples"] == 0

    def test_stats_stays_open_to_api_key_holders(self, client, settings, monkeypatch, graph):
        monkeypatch.setattr(settings, "admin_key", "correct-horse")
        assert client.get("/graph/stats").status_code == 200


class TestGraphStoreConsistency:
    def test_append_racing_remove_docs_lands_in_the_live_index(self, graph):
        """append() used to fetch the index before taking the lock, so a
        remove_docs() in between swapped the index out from under it and the
        new triples went into the discarded copy."""
        graph.append([_triple("old:0", "old"), _triple("keep:0", "keep")])

        with graph._LOCK:
            t = threading.Thread(target=graph.append, args=([_triple("new:0", "new")],))
            t.start()
            time.sleep(0.1)  # let append reach the lock
            graph.remove_docs({"old"})
        t.join(timeout=5)

        live = {t.doc_id for t in graph.load().triples}
        assert live == {"keep", "new"}
        graph._INDEX = None  # and the file agrees
        assert {t.doc_id for t in graph.load().triples} == {"keep", "new"}

