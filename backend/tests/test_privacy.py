"""Tests for erasure, subject access export and retention.

The vector store is replaced by a small in-memory fake that evaluates the same
Qdrant filters the real one would, so these tests check which points a filter
selects, not merely that some delete call was made. The graph, eval runs and
traces are the real implementations pointed at tmp_path.
"""
import asyncio
import json
import os
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import auth, tracing
from app.eval import metrics
from app.privacy import erasure, export, retention
from app.rag import graph_store, store
from app.rag.graph_store import Triple
from app.resilience import CircuitBreaker

ADMIN = {"X-Admin-Key": "admin-secret"}


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


def _cond_matches(payload: dict, cond) -> bool:
    value = payload.get(cond.key)
    if isinstance(cond.match, qm.MatchValue):
        return value == cond.match.value
    if isinstance(cond.match, qm.MatchAny):
        return value in cond.match.any
    raise AssertionError(f"unsupported match {cond.match!r}")


def _filter_matches(payload: dict, f: qm.Filter) -> bool:
    if f.must and not all(_cond_matches(payload, c) for c in f.must):
        return False
    if f.should and not any(_cond_matches(payload, c) for c in f.should):
        return False
    return not (f.must_not and any(_cond_matches(payload, c) for c in f.must_not))


class FakeVectors:
    """Chunk payloads plus the two store calls privacy code makes."""

    def __init__(self, payloads):
        self.payloads = [dict(p) for p in payloads]
        self.fail = False

    async def scroll(self, selector, fields=None):
        if self.fail:
            raise ConnectionError("qdrant down")
        hits = [p for p in self.payloads if _filter_matches(p, selector)]
        if fields is None:
            return [dict(p) for p in hits]
        return [{k: p[k] for k in fields if k in p} for p in hits]

    async def delete(self, selector):
        before = len(self.payloads)
        self.payloads = [p for p in self.payloads if not _filter_matches(p, selector)]
        return before - len(self.payloads)

    def doc_ids(self):
        return sorted({p["doc_id"] for p in self.payloads})


def _chunk(doc_id, n, filename, owner=None, source_key=None):
    payload = {
        "chunk_id": f"{doc_id}:{n}",
        "doc_id": doc_id,
        "filename": filename,
        "text": "...",
        "ingested_at": f"2026-09-0{n + 1}T10:00:00+00:00",
        "pii_counts": {"EMAIL": 1} if n == 0 else {},
    }
    if owner:
        payload["owner"] = owner
    if source_key:
        payload["source_key"] = source_key
    return payload


@pytest.fixture
def vectors():
    fake = FakeVectors(
        [
            _chunk("docA", 0, "alice_cv.md", owner="alice", source_key="drive://alice/cv"),
            _chunk("docA", 1, "alice_cv.md", owner="alice", source_key="drive://alice/cv"),
            _chunk("docA2", 0, "alice_cv.md", owner="alice", source_key="drive://alice/cv"),
            _chunk("docB", 0, "bob_notes.md", owner="bob"),
            _chunk("legacy", 0, "old.md"),
        ]
    )
    with (
        patch.object(erasure, "scroll_payloads", new=fake.scroll),
        patch.object(erasure, "delete_matching", new=fake.delete),
        patch.object(export, "scroll_payloads", new=fake.scroll),
    ):
        yield fake


@pytest.fixture
def graph(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "graph_data_dir", str(tmp_path / "graph"))
    monkeypatch.setattr(graph_store, "_INDEX", None)
    graph_store.append(
        [
            Triple("alice", "wrote", "cv", "docA:0", "docA"),
            Triple("alice", "lives_in", "lyon", "docA:1", "docA"),
            Triple("bob", "likes", "rag", "docB:0", "docB"),
        ]
    )
    return graph_store


@pytest.fixture
def traces(monkeypatch):
    monkeypatch.setattr(tracing, "_TRACES", OrderedDict())
    ids = {}
    for name, chunk_ids, principal in [
        ("usesA", ["docA:0"], None),
        ("usesB", ["docB:0"], None),
        ("byAlice", ["legacy:0"], "alice"),
    ]:
        with tracing.start_trace(name, {"question": name}, principal_id=principal) as t:
            t.log("sources", {"chunk_ids": chunk_ids})
        ids[name] = t.id
    return ids


def _run_file(runs_dir, name, rows, created_at="2026-09-01T00:00:00+00:00"):
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / name
    path.write_text(
        json.dumps({"dataset": "golden_v1", "created_at": created_at, "per_example": rows}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def eval_run():
    rows = [
        {
            "question": "What did Alice write?",
            "answer": "Alice wrote a CV.",
            "score": {"faithfulness": 0.9},
            "retrieval": {"retrieved": ["alice_cv.md"], "expected": ["alice_cv.md"]},
            "doc_ids": ["docA"],
        },
        {
            "question": "What does Bob like?",
            "answer": "RAG.",
            "score": {"faithfulness": 0.8},
            "retrieval": {"retrieved": ["bob_notes.md"], "expected": ["bob_notes.md"]},
            "doc_ids": ["docB"],
        },
        {
            # Saved before rows recorded doc ids: matched on the filename.
            "question": "Old question",
            "answer": "Old answer from Alice's CV.",
            "score": None,
            "retrieval": {"retrieved": ["alice_cv.md", "bob_notes.md"], "expected": []},
        },
    ]
    return _run_file(metrics.RUNS_DIR, "run.json", rows)


@pytest.fixture
def usage_db(monkeypatch):
    """The auth tables in an in-memory SQLite database, with two keys and usage."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    auth.Base.metadata.create_all(engine)
    monkeypatch.setattr(auth, "_SessionLocal", sessionmaker(bind=engine))
    with auth.session_scope() as db:
        db.add_all(
            [
                auth.APIKey(id=1, key_hash="h1", name="alice", key_hint="sk_aaa"),
                auth.APIKey(id=2, key_hash="h2", name="bob", key_hint="sk_bbb"),
            ]
        )
        db.add_all(
            [
                auth.APIKeyUsage(api_key_id=1, endpoint="/ask", tokens_input=10, tokens_output=5),
                auth.APIKeyUsage(api_key_id=1, endpoint="/ask", tokens_input=20, tokens_output=5),
                auth.APIKeyUsage(api_key_id=1, endpoint="/ingest"),
                auth.APIKeyUsage(api_key_id=2, endpoint="/ask", tokens_input=1),
            ]
        )
    return engine


def _usage_rows(key_id):
    with auth.session_scope() as db:
        return db.query(auth.APIKeyUsage).filter_by(api_key_id=key_id).count()


@pytest.fixture
def admin(settings, monkeypatch):
    monkeypatch.setattr(settings, "admin_key", ADMIN["X-Admin-Key"])


@pytest.fixture(autouse=True)
def _isolated_registries(monkeypatch):
    """Tests that register targets must not leak them into their neighbours."""
    monkeypatch.setattr(erasure, "_targets", dict(erasure._targets))
    monkeypatch.setattr(retention, "_targets", dict(retention._targets))
    monkeypatch.setattr(export, "_sources", dict(export._sources))


# ---------------------------------------------------------------------------
# Erasure
# ---------------------------------------------------------------------------


class TestEraseDocument:
    def test_removes_the_document_everywhere(
        self, client, admin, vectors, graph, traces, eval_run
    ):
        resp = client.delete("/privacy/documents/docA", headers=ADMIN)

        assert resp.status_code == 200
        body = resp.json()
        assert body["complete"] is True
        assert body["doc_ids"] == ["docA"]
        assert body["targets"] == {
            "vectors": 2,
            "graph": 2,
            "traces": 1,
            "eval_runs": 2,
            "usage": 0,
        }
        # Only docA went; its other revision and other documents stay.
        assert vectors.doc_ids() == ["docA2", "docB", "legacy"]
        assert {t.doc_id for t in graph_store.load().triples} == {"docB"}
        assert tracing.get_trace(traces["usesA"]) is None
        assert tracing.get_trace(traces["usesB"]) is not None

    def test_eval_rows_keep_scores_but_lose_text(
        self, client, admin, vectors, graph, traces, eval_run
    ):
        client.delete("/privacy/documents/docA", headers=ADMIN)
        rows = json.loads(eval_run.read_text())["per_example"]

        assert rows[0]["answer"] == erasure.ERASED
        assert rows[0]["score"] == {"faithfulness": 0.9}
        assert rows[0]["retrieval"] == {"retrieved": ["[erased]"], "expected": ["[erased]"]}
        assert rows[0]["doc_ids"] == []
        assert rows[0]["erased"] is True
        assert rows[1]["answer"] == "RAG."
        assert rows[2]["answer"] == erasure.ERASED
        assert rows[2]["retrieval"]["retrieved"] == ["[erased]", "bob_notes.md"]

    def test_is_idempotent(self, client, admin, vectors, graph, traces, eval_run):
        client.delete("/privacy/documents/docA", headers=ADMIN)
        again = client.delete("/privacy/documents/docA", headers=ADMIN)

        assert again.status_code == 200
        assert set(again.json()["targets"].values()) == {0}

    def test_unknown_document_is_a_harmless_no_op(self, client, admin, vectors, graph, traces):
        resp = client.delete("/privacy/documents/nope", headers=ADMIN)
        assert resp.status_code == 200
        assert set(resp.json()["targets"].values()) == {0}
        assert len(vectors.payloads) == 5


class TestEraseSource:
    def test_matches_source_key_across_revisions(self, client, admin, vectors, graph, traces):
        resp = client.delete(
            "/privacy/sources", params={"source_key": "drive://alice/cv"}, headers=ADMIN
        )
        assert resp.status_code == 200
        assert resp.json()["doc_ids"] == ["docA", "docA2"]
        assert resp.json()["targets"]["vectors"] == 3
        assert vectors.doc_ids() == ["docB", "legacy"]

    def test_falls_back_to_filename(self, client, admin, vectors, graph, traces):
        resp = client.delete("/privacy/sources", params={"source_key": "old.md"}, headers=ADMIN)
        assert resp.json()["doc_ids"] == ["legacy"]
        assert vectors.doc_ids() == ["docA", "docA2", "docB"]

    def test_source_key_is_required(self, client, admin):
        assert client.delete("/privacy/sources", headers=ADMIN).status_code == 422


class TestErasePrincipal:
    def test_removes_owned_documents_traces_and_usage(
        self, client, admin, vectors, graph, traces, eval_run, usage_db
    ):
        resp = client.delete("/privacy/principals/alice", headers=ADMIN)

        assert resp.status_code == 200
        body = resp.json()
        assert body["subject"] == "principal:alice"
        assert body["doc_ids"] == ["docA", "docA2"]
        assert body["targets"]["vectors"] == 3
        # One trace used docA, one was made by alice herself.
        assert body["targets"]["traces"] == 2
        assert body["targets"]["usage"] == 3
        assert vectors.doc_ids() == ["docB", "legacy"]
        assert tracing.get_trace(traces["byAlice"]) is None
        assert _usage_rows(1) == 0
        assert _usage_rows(2) == 1

    def test_numeric_principal_is_an_api_key_id(self, client, admin, vectors, usage_db):
        resp = client.delete("/privacy/principals/2", headers=ADMIN)
        assert resp.json()["targets"]["usage"] == 1
        assert _usage_rows(1) == 3


class TestErasureFailures:
    def test_failing_target_gives_503_with_partial_report(
        self, client, admin, vectors, graph, traces
    ):
        def broken(request):
            raise OSError("disk full")

        erasure.register_erasure_target("broken", broken)
        resp = client.delete("/privacy/documents/docA", headers=ADMIN)

        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["complete"] is False
        assert detail["errors"] == {"broken": "OSError: disk full"}
        # The other targets still ran.
        assert detail["targets"]["vectors"] == 2
        assert "docA" not in vectors.doc_ids()

    def test_unresolvable_subject_erases_nothing(self, client, admin, vectors, graph, traces):
        vectors.fail = True
        resp = client.delete("/privacy/principals/alice", headers=ADMIN)
        assert resp.status_code == 503
        assert "Nothing was erased" in resp.json()["detail"]
        assert len(vectors.payloads) == 5
        assert tracing.get_trace(traces["byAlice"]) is not None


class TestErasureAuth:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("delete", "/privacy/documents/x"),
            ("delete", "/privacy/sources?source_key=x"),
            ("delete", "/privacy/principals/x"),
            ("get", "/privacy/export/x"),
            ("post", "/privacy/retention/run"),
        ],
    )
    def test_requires_the_admin_key(self, client, settings, monkeypatch, method, path):
        monkeypatch.setattr(settings, "admin_key", "")
        assert getattr(client, method)(path).status_code == 503
        monkeypatch.setattr(settings, "admin_key", "right")
        assert getattr(client, method)(path, headers={"X-Admin-Key": "wrong"}).status_code == 403


@pytest.mark.asyncio
class TestErasureRegistry:
    async def test_custom_targets_get_the_resolved_request(self):
        seen = []

        async def audit(request):
            seen.append(request)
            return 4

        erasure.register_erasure_target("audit", audit)
        assert erasure.erasure_targets()[-1] == "audit"
        request = erasure.ErasureRequest(
            subject="principal:p", doc_ids=frozenset({"d"}), principal_id="p"
        )
        with (
            patch.object(erasure, "delete_matching", new=AsyncMock(return_value=0)),
            patch.object(graph_store, "remove_docs", return_value=0),
        ):
            report = await erasure.erase(request)

        assert seen == [request]
        assert report.targets["audit"] == 4

    async def test_unregister(self):
        erasure.register_erasure_target("x", lambda r: 1)
        erasure.unregister_erasure_target("x")
        erasure.unregister_erasure_target("x")
        assert "x" not in erasure.erasure_targets()

    async def test_vectors_target_with_nothing_to_match_does_not_call_the_store(self):
        with patch.object(erasure, "delete_matching", new=AsyncMock()) as delete:
            assert await erasure.erase_vectors(erasure.ErasureRequest(subject="x")) == 0
        delete.assert_not_awaited()

    async def test_vectors_target_deletes_by_owner_too(self):
        with patch.object(erasure, "delete_matching", new=AsyncMock(return_value=3)) as delete:
            await erasure.erase_vectors(
                erasure.ErasureRequest(subject="p", principal_id="alice")
            )
        selector = delete.await_args.args[0]
        assert [c.key for c in selector.should] == [erasure.OWNER_PAYLOAD_KEY]

    async def test_resolve_documents_with_no_ids(self):
        request = await erasure.resolve_documents([])
        assert request.doc_ids == frozenset()


class TestScrubEvalRuns:
    def test_skips_unreadable_files(self):
        metrics.RUNS_DIR.mkdir(parents=True)
        (metrics.RUNS_DIR / "broken.json").write_text("{not json")
        assert erasure.scrub_eval_runs({"docA"}) == 0

    def test_missing_directory(self):
        assert erasure.scrub_eval_runs({"docA"}) == 0

    def test_untouched_file_is_not_rewritten(self, eval_run):
        before = eval_run.stat().st_mtime_ns
        assert erasure.scrub_eval_runs({"nothing"}) == 0
        assert eval_run.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_documents_usage_and_traces(self, client, admin, vectors, traces, usage_db):
        resp = client.get("/privacy/export/alice", headers=ADMIN)

        assert resp.status_code == 200
        body = resp.json()
        assert body["principal_id"] == "alice"
        assert body["complete"] is True
        assert body["documents"] == [
            {
                "doc_id": "docA",
                "filename": "alice_cv.md",
                "source_key": "drive://alice/cv",
                "created": "2026-09-01T10:00:00+00:00",
                "chunks": 2,
                "pii_counts": {"EMAIL": 1},
            },
            {
                "doc_id": "docA2",
                "filename": "alice_cv.md",
                "source_key": "drive://alice/cv",
                "created": "2026-09-01T10:00:00+00:00",
                "chunks": 1,
                "pii_counts": {"EMAIL": 1},
            },
        ]
        usage = body["usage"]
        assert usage["requests"] == 3
        assert usage["tokens_input"] == 30
        assert usage["tokens_output"] == 10
        assert usage["by_endpoint"] == {"/ask": 2, "/ingest": 1}
        assert [k["name"] for k in usage["api_keys"]] == ["alice"]
        assert "key_hash" not in usage["api_keys"][0]
        assert body["traces"]["count"] == 1
        assert body["traces"]["items"][0]["name"] == "byAlice"

    def test_unknown_principal_is_empty(self, client, admin, vectors, traces, usage_db):
        body = client.get("/privacy/export/nobody", headers=ADMIN).json()
        assert body["documents"] == []
        assert body["usage"]["requests"] == 0
        assert body["usage"]["api_keys"] == []
        assert body["traces"]["count"] == 0

    def test_unreadable_section_is_reported(self, client, admin, vectors, traces, usage_db):
        vectors.fail = True
        body = client.get("/privacy/export/alice", headers=ADMIN).json()
        assert body["complete"] is False
        assert "documents" in body["errors"]
        assert "documents" not in body
        assert body["usage"]["requests"] == 3

    @pytest.mark.asyncio
    async def test_custom_section(self, vectors, traces, usage_db):
        async def audit(principal_id):
            return [{"event": "login", "principal": principal_id}]

        export.register_export_source("audit", audit)
        body = await export.export_principal("alice")
        assert body["audit"] == [{"event": "login", "principal": "alice"}]
        export.unregister_export_source("audit")
        assert "audit" not in await export.export_principal("alice")


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
class TestRetention:
    async def test_purges_old_traces_and_eval_runs(self, settings, monkeypatch, traces):
        monkeypatch.setattr(settings, "retention_traces_days", 7)
        monkeypatch.setattr(settings, "retention_eval_runs_days", 30)
        tracing._TRACES[traces["usesA"]]["created_at"] = (NOW - timedelta(days=8)).isoformat()
        for tid in (traces["usesB"], traces["byAlice"]):
            tracing._TRACES[tid]["created_at"] = (NOW - timedelta(days=1)).isoformat()
        old = _run_file(metrics.RUNS_DIR, "old.json", [], (NOW - timedelta(days=31)).isoformat())
        new = _run_file(metrics.RUNS_DIR, "new.json", [], (NOW - timedelta(days=2)).isoformat())

        report = await retention.run_retention(NOW)

        assert report.targets == {"traces": 1, "eval_runs": 1}
        assert report.cutoffs["traces"] == (NOW - timedelta(days=7)).isoformat(timespec="seconds")
        assert tracing.get_trace(traces["usesA"]) is None
        assert not old.exists() and new.exists()

    async def test_eval_run_without_created_at_uses_file_time(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "retention_eval_runs_days", 30)
        metrics.RUNS_DIR.mkdir(parents=True)
        legacy = metrics.RUNS_DIR / "legacy.json"
        legacy.write_text("{}")
        stamp = (NOW - timedelta(days=40)).timestamp()
        os.utime(legacy, (stamp, stamp))
        unreadable = metrics.RUNS_DIR / "unreadable.json"
        unreadable.write_text("{nope")

        assert retention.purge_eval_runs(NOW - timedelta(days=30)) == 1
        assert not legacy.exists() and unreadable.exists()

    async def test_zero_days_keeps_forever(self, settings, monkeypatch, traces):
        monkeypatch.setattr(settings, "retention_traces_days", 0)
        for t in tracing._TRACES.values():
            t["created_at"] = (NOW - timedelta(days=1000)).isoformat()

        report = await retention.run_retention(NOW)

        assert "traces" in report.skipped
        assert "traces" not in report.targets
        assert len(tracing._TRACES) == 3

    async def test_custom_target_uses_its_own_setting(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "retention_audit_days", 365)
        seen = []

        async def purge_audit_events(before):
            seen.append(before)
            return 5

        retention.register_retention_target(
            "audit", purge_audit_events, days_setting="retention_audit_days"
        )
        report = await retention.run_retention(NOW)

        assert seen == [NOW - timedelta(days=365)]
        assert report.targets["audit"] == 5

    async def test_failing_target_does_not_stop_the_others(self, traces):
        def broken(before):
            raise RuntimeError("db down")

        retention.unregister_retention_target("traces")
        retention.register_retention_target("broken", broken, days_setting="retention_audit_days")
        retention.register_retention_target(
            "traces", tracing.purge_traces, days_setting="retention_traces_days"
        )
        report = await retention.run_retention(NOW)

        assert report.errors == {"broken": "RuntimeError: db down"}
        assert "traces" in report.targets

    async def test_unknown_setting_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown retention setting"):
            retention.register_retention_target("x", lambda b: 0, days_setting="nope")

    async def test_background_task_runs_at_once_and_cancels_cleanly(self):
        ran = asyncio.Event()

        async def fake_run(now=None):
            ran.set()
            return retention.RetentionReport(ran_at="now")

        with patch.object(retention, "run_retention", new=fake_run):
            task = retention.start_retention_task(interval_seconds=3600)
            await asyncio.wait_for(ran.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert task.cancelled()

    async def test_background_task_survives_a_failed_run(self):
        calls = 0

        async def flaky(now=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("boom")
            return retention.RetentionReport(ran_at="now")

        with patch.object(retention, "run_retention", new=flaky):
            task = retention.start_retention_task(interval_seconds=0)
            for _ in range(50):
                await asyncio.sleep(0)
                if calls >= 2:
                    break
            task.cancel()
        assert calls >= 2


class TestRetentionEndpoint:
    def test_runs_now(self, client, admin, traces):
        resp = client.post("/privacy/retention/run", headers=ADMIN)
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["targets"]) == {"traces", "eval_runs"}
        assert body["errors"] == {}


class TestLifespan:
    def test_starts_and_stops_the_retention_task(self):
        from fastapi.testclient import TestClient

        from app import main

        started = []

        def fake_start():
            async def idle():
                await asyncio.Event().wait()

            real = asyncio.get_running_loop().create_task(idle())
            started.append(real)
            return real

        with patch.object(main, "start_retention_task", new=fake_start):
            with TestClient(main.app) as c:
                assert c.get("/health").status_code == 200
                assert not started[0].done()
        assert started[0].cancelled()


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStoreHelpers:
    @pytest.fixture(autouse=True)
    def _fresh_breaker(self, monkeypatch):
        monkeypatch.setattr(store, "_qdrant_breaker", CircuitBreaker(service_name="Qdrant"))

    async def test_scroll_follows_pages(self, mock_qdrant_client):
        page1 = [MagicMock(payload={"doc_id": "a"}), MagicMock(payload={"doc_id": "b"})]
        page2 = [MagicMock(payload=None)]
        mock_qdrant_client.scroll = AsyncMock(side_effect=[(page1, 42), (page2, None)])
        with patch.object(store, "client", new=AsyncMock(return_value=mock_qdrant_client)):
            payloads = await store.scroll_payloads(qm.Filter(), fields=["doc_id"])

        assert payloads == [{"doc_id": "a"}, {"doc_id": "b"}, {}]
        assert mock_qdrant_client.scroll.await_args_list[1].kwargs["offset"] == 42

    async def test_scroll_raises_instead_of_degrading(self, mock_qdrant_client):
        mock_qdrant_client.scroll = AsyncMock(side_effect=ConnectionError("down"))
        with patch.object(store, "client", new=AsyncMock(return_value=mock_qdrant_client)):
            with pytest.raises(ConnectionError):
                await store.scroll_payloads(qm.Filter())
        assert store._qdrant_breaker.failure_count == 1

    async def test_delete_counts_then_deletes(self, mock_qdrant_client):
        mock_qdrant_client.count = AsyncMock(return_value=MagicMock(count=3))
        mock_qdrant_client.delete = AsyncMock()
        with patch.object(store, "client", new=AsyncMock(return_value=mock_qdrant_client)):
            assert await store.delete_matching(qm.Filter()) == 3
        mock_qdrant_client.delete.assert_awaited_once()

    async def test_delete_nothing_matched_skips_the_delete(self, mock_qdrant_client):
        mock_qdrant_client.count = AsyncMock(return_value=MagicMock(count=0))
        mock_qdrant_client.delete = AsyncMock()
        with patch.object(store, "client", new=AsyncMock(return_value=mock_qdrant_client)):
            assert await store.delete_matching(qm.Filter()) == 0
        mock_qdrant_client.delete.assert_not_awaited()

    async def test_open_breaker_refuses(self):
        store._qdrant_breaker.state = "open"
        store._qdrant_breaker.last_failure_time = float("inf")
        with pytest.raises(RuntimeError, match="circuit breaker open"):
            await store.delete_matching(qm.Filter())
        with pytest.raises(RuntimeError, match="circuit breaker open"):
            await store.scroll_payloads(qm.Filter())
