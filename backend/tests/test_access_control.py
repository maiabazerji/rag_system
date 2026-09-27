"""Per-document access control: principals, scopes, and enforcement on every retrieval path.

The store tests run against Qdrant's in-memory mode, so the payload filters
are evaluated by Qdrant itself rather than by a mock that would agree with
whatever the code sends.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app import auth
from app.access import (
    PUBLIC_GROUP,
    AccessScope,
    Principal,
    local_principal,
    normalize_groups,
    resolve_document_acl,
)
from app.rag import graph_store, store
from app.rag.graph_store import Triple
from app.resilience import CircuitBreaker
from app.schemas import Answer, Chunk, Source

VEC = [1.0, 0.0, 0.0, 0.0]

# doc_id -> payload fields that decide who may read it.
CORPUS: dict[str, dict] = {
    "pub": {"tenant": "default", "acl_groups": ["public"]},
    "hr": {"tenant": "default", "acl_groups": ["hr"]},
    "fin": {"tenant": "default", "acl_groups": ["finance"]},
    "both": {"tenant": "default", "acl_groups": ["hr", "finance"]},
    "legacy": {},  # indexed before ACLs: no tenant, no groups
    "acme": {"tenant": "acme", "acl_groups": ["public"]},
    "acmehr": {"tenant": "acme", "acl_groups": ["hr"]},
}


def scope(tenant="default", *groups) -> AccessScope:
    return AccessScope(tenant=tenant, groups=frozenset({PUBLIC_GROUP, *groups}))


def key_principal(groups=(), tenant="default", is_admin=False) -> Principal:
    return Principal(
        id="key:1",
        kind="api_key",
        display_name="ci",
        tenant=tenant,
        groups=tuple(groups),
        is_admin=is_admin,
        key_id=1,
    )


@pytest.fixture
async def mem_store(monkeypatch, settings):
    """A real (in-memory) Qdrant collection holding CORPUS, one chunk per doc."""
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=len(VEC), distance=qm.Distance.COSINE),
    )
    monkeypatch.setattr(store, "_client", client)
    monkeypatch.setattr(
        store, "_qdrant_breaker", CircuitBreaker(failure_threshold=5, service_name="Qdrant")
    )
    chunks = [
        Chunk(
            id=f"{doc}:0",
            doc_id=doc,
            text=f"text of {doc}",
            tokens=3,
            metadata={"filename": f"{doc}.md", **fields},
        )
        for doc, fields in CORPUS.items()
    ]
    await store.upsert(chunks, [VEC] * len(chunks))
    yield client
    await client.close()


def _docs(points) -> set[str]:
    return {p.payload["doc_id"] for p in points}


def _expected(access: AccessScope) -> set[str]:
    return {doc for doc, fields in CORPUS.items() if access.permits({"doc_id": doc, **fields})}


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


class TestAccessScope:
    def test_tenant_and_group_must_both_match(self):
        s = scope("default", "hr")
        assert s.permits({"tenant": "default", "acl_groups": ["hr"]})
        assert not s.permits({"tenant": "acme", "acl_groups": ["hr"]})
        assert not s.permits({"tenant": "default", "acl_groups": ["finance"]})

    def test_any_shared_group_is_enough(self):
        assert scope("default", "hr").permits({"tenant": "default", "acl_groups": ["x", "hr"]})

    def test_public_is_readable_by_everyone_in_the_tenant(self):
        assert scope().permits({"tenant": "default", "acl_groups": ["public"]})
        assert not scope("acme").permits({"tenant": "default", "acl_groups": ["public"]})

    def test_legacy_chunks_are_public_in_the_default_tenant(self):
        assert scope().permits({})
        assert not scope("acme").permits({})

    def test_legacy_chunks_hidden_when_configured(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "acl_legacy_public", False)
        assert not scope("default", "hr").permits({})

    def test_empty_payload_is_not_a_bypass(self):
        assert not scope("default", "hr").permits({"tenant": "default", "acl_groups": ["x"]})
        assert not scope("acme").permits(None)


class TestPrincipal:
    def test_scope_always_includes_public(self):
        assert key_principal(["hr"], "acme").scope() == scope("acme", "hr")

    def test_local_principal_is_a_public_admin_in_the_default_tenant(self):
        p = local_principal()
        assert p.id == "local"
        assert p.is_admin
        assert p.scope() == scope()

    def test_as_auth_keeps_the_legacy_keys(self):
        d = key_principal(["hr"]).as_auth()
        assert d["id"] == 1 and d["name"] == "ci" and d["usage_id"] is None
        assert d["groups"] == ["hr"] and d["tenant"] == "default"

    def test_normalize_groups(self):
        assert normalize_groups([" hr ", "hr", "", "fin"]) == ("fin", "hr")
        with pytest.raises(ValueError):
            normalize_groups([str(i) for i in range(201)])
        with pytest.raises(ValueError):
            normalize_groups(["x" * 257])


class TestDocumentAcl:
    def test_local_uploads_are_public(self):
        assert resolve_document_acl(local_principal()) == ("default", ["public"])

    def test_default_is_the_uploaders_groups(self):
        assert resolve_document_acl(key_principal(["hr", "fin"], "acme")) == (
            "acme",
            ["fin", "hr"],
        )

    def test_uploader_without_groups_publishes(self):
        assert resolve_document_acl(key_principal()) == ("default", ["public"])

    def test_acl_default_public(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "acl_default_public", True)
        assert resolve_document_acl(key_principal(["hr"])) == ("default", ["public"])

    def test_explicit_subset_is_allowed(self):
        assert resolve_document_acl(key_principal(["hr", "fin"]), ["hr"]) == (
            "default",
            ["hr"],
        )

    def test_public_is_always_allowed(self):
        assert resolve_document_acl(key_principal(["hr"]), ["public"])[1] == ["public"]

    def test_foreign_group_is_forbidden(self):
        with pytest.raises(HTTPException) as exc:
            resolve_document_acl(key_principal(["hr"]), ["hr", "finance"])
        assert exc.value.status_code == 403
        assert "finance" in exc.value.detail

    def test_admin_may_use_any_group(self):
        assert resolve_document_acl(key_principal(is_admin=True), ["finance"])[1] == [
            "finance"
        ]

    @pytest.mark.parametrize("groups", [[], [" ", ""]])
    def test_empty_group_list_is_rejected(self, groups):
        with pytest.raises(HTTPException) as exc:
            resolve_document_acl(key_principal(["hr"]), groups)
        assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Store: the Qdrant filter, evaluated by Qdrant
# ---------------------------------------------------------------------------

SCOPES = [
    scope(),
    scope("default", "hr"),
    scope("default", "finance"),
    scope("default", "hr", "finance"),
    scope("acme"),
    scope("acme", "hr"),
    scope("nowhere", "hr"),
]


class TestStoreEnforcement:
    @pytest.mark.parametrize("access", SCOPES, ids=str)
    async def test_search_returns_exactly_the_permitted_docs(self, mem_store, access):
        hits = await store.search(VEC, top_k=50, access=access)
        assert _docs(hits) == _expected(access)

    @pytest.mark.parametrize("access", SCOPES, ids=str)
    async def test_the_filter_alone_matches_the_rule(self, mem_store, settings, access):
        """Without the Python re-check, which would hide an over-permissive filter."""
        result = await mem_store.query_points(
            settings.qdrant_collection,
            query=VEC,
            limit=50,
            query_filter=store.access_filter(access),
        )
        assert _docs(result.points) == _expected(access)

    async def test_expected_sets_are_what_we_think(self):
        assert _expected(scope("default", "hr")) == {"pub", "hr", "both", "legacy"}
        assert _expected(scope("acme", "hr")) == {"acme", "acmehr"}

    async def test_legacy_hidden_by_the_filter_too(self, mem_store, settings, monkeypatch):
        monkeypatch.setattr(settings, "acl_legacy_public", False)
        hits = await store.search(VEC, top_k=50, access=scope("default", "hr"))
        assert _docs(hits) == {"pub", "hr", "both"}

    async def test_unrestricted_search_sees_everything(self, mem_store):
        assert _docs(await store.search(VEC, top_k=50)) == set(CORPUS)

    @pytest.mark.parametrize("access", SCOPES, ids=str)
    async def test_count_is_scoped(self, mem_store, access):
        assert await store.count(access) == len(_expected(access))

    @pytest.mark.parametrize("access", SCOPES, ids=str)
    async def test_fetch_by_id_cannot_reach_forbidden_chunks(self, mem_store, access):
        hits = await store.fetch_chunks([f"{d}:0" for d in CORPUS], access=access)
        assert _docs(hits) == _expected(access)

    @pytest.mark.parametrize("access", SCOPES, ids=str)
    async def test_scroll_is_scoped(self, mem_store, access):
        assert _docs(await store.scroll_chunks(access=access)) == _expected(access)

    async def test_readable_doc_ids(self, mem_store):
        assert await store.readable_doc_ids(scope("default", "finance")) == {
            "pub",
            "fin",
            "both",
            "legacy",
        }

    async def test_payload_is_rechecked_after_the_filter(self, qdrant_mock):
        """A point the filter let through by mistake is still dropped."""
        leaked = qm.ScoredPoint(
            id=1, version=1, score=1.0, payload={"doc_id": "fin", "acl_groups": ["finance"]}
        )
        qdrant_mock.query_points = AsyncMock(return_value=qm.QueryResponse(points=[leaked]))
        assert await store.search(VEC, access=scope("default", "hr")) == []

    async def test_search_sends_the_filter(self, qdrant_mock):
        qdrant_mock.query_points = AsyncMock(return_value=qm.QueryResponse(points=[]))
        await store.search(VEC, access=scope("acme", "hr"))
        sent = qdrant_mock.query_points.await_args.kwargs["query_filter"]
        assert sent == store.access_filter(scope("acme", "hr"))

    async def test_acl_fields_get_payload_indexes(self, qdrant_mock, settings):
        await store._ensure_collection(qdrant_mock)
        indexed = {
            c.kwargs["field_name"]: c.kwargs["field_schema"]
            for c in qdrant_mock.create_payload_index.await_args_list
        }
        assert indexed == {
            "tenant": qm.PayloadSchemaType.KEYWORD,
            "acl_groups": qm.PayloadSchemaType.KEYWORD,
        }

    async def test_payload_index_failure_is_not_fatal(self, qdrant_mock):
        qdrant_mock.create_payload_index = AsyncMock(side_effect=RuntimeError("nope"))
        await store._ensure_collection(qdrant_mock)  # does not raise


@pytest.fixture
def qdrant_mock(mock_qdrant_client, monkeypatch):
    monkeypatch.setattr(store, "_client", mock_qdrant_client)
    monkeypatch.setattr(
        store, "_qdrant_breaker", CircuitBreaker(failure_threshold=5, service_name="Qdrant")
    )
    with patch("app.rag.store.embedding_dim", return_value=384):
        yield mock_qdrant_client


# ---------------------------------------------------------------------------
# Ingest: every chunk is labelled
# ---------------------------------------------------------------------------


class TestIngestLabels:
    async def _ingest(self, **kwargs):
        from app.rag.ingest import enqueue_document

        with (
            patch("app.rag.ingest.embed_texts_async", new=AsyncMock(return_value=[VEC])),
            patch("app.rag.ingest.upsert", new=AsyncMock()) as up,
            patch(
                "app.rag.ingest.delete_stale_revisions",
                new=AsyncMock(return_value=store.StaleRevisions()),
            ),
        ):
            result = await enqueue_document("a.txt", b"hello world", **kwargs)
        return result, up.await_args.args[0]

    async def test_defaults_to_public_in_the_default_tenant(self):
        result, chunks = await self._ingest()
        assert chunks[0].metadata["owner"] == "local"
        assert chunks[0].metadata["tenant"] == "default"
        assert chunks[0].metadata["acl_groups"] == ["public"]
        assert result["acl_groups"] == ["public"]

    async def test_labels_every_chunk(self):
        _, chunks = await self._ingest(tenant="acme", acl_groups=["hr", "fin"])
        assert all(c.metadata["tenant"] == "acme" for c in chunks)
        assert all(c.metadata["acl_groups"] == ["fin", "hr"] for c in chunks)

    async def test_public_default_tenant_doc_ids_are_unchanged(self):
        from app.rag.ingest import _doc_id

        result, _ = await self._ingest()
        assert result["doc_id"] == _doc_id("hello world")

    async def test_other_scopes_get_their_own_doc_ids(self):
        """Same text in two tenants must not share (and overwrite) points."""
        a, _ = await self._ingest(tenant="acme")
        b, _ = await self._ingest(tenant="globex")
        c, _ = await self._ingest(tenant="acme", acl_groups=["hr"])
        d, _ = await self._ingest()
        assert len({a["doc_id"], b["doc_id"], c["doc_id"], d["doc_id"]}) == 4


# ---------------------------------------------------------------------------
# Graph store and graph strategy
# ---------------------------------------------------------------------------


@pytest.fixture
def graph(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "graph_data_dir", str(tmp_path))
    monkeypatch.setattr(graph_store, "_INDEX", None)
    graph_store.append(
        [
            Triple("Alice", "manages", "Bob", "hr:0", "hr"),
            Triple("Alice", "audits", "Ledger", "fin:0", "fin"),
            Triple("Bob", "reports_to", "Carol", "hr:0", "hr"),
        ]
    )
    return graph_store


class TestGraphStoreFiltering:
    def test_neighbors_only_walk_visible_edges(self, graph):
        chunks, related = graph.neighbors("alice", hops=2, doc_ids={"hr"})
        assert chunks == {"hr:0"}
        assert "ledger" not in related
        assert {"bob", "carol"} <= related

    def test_unrestricted_walk_is_unchanged(self, graph):
        chunks, related = graph.neighbors("alice", hops=1)
        assert chunks == {"hr:0", "fin:0"}
        assert "ledger" in related

    def test_subgraph_hides_foreign_triples(self, graph):
        text = graph.describe_subgraph(["Alice"], doc_ids={"hr"})
        assert "manages" in text
        assert "audits" not in text and "Ledger" not in text

    def test_no_visible_docs_means_no_edges(self, graph):
        assert graph.describe_subgraph(["Alice"], doc_ids=set()) == "(no edges found)"
        assert graph.neighbors("alice", doc_ids=set()) == (set(), set())

    def test_stats_and_entities_are_scoped(self, graph):
        assert graph.stats({"fin"}) == {
            "triples": 1,
            "entities": 2,
            "chunks_indexed": 1,
            "docs_indexed": 1,
        }
        assert graph.load().entities_in({"fin"}) == ["alice", "ledger"]
        assert graph.stats()["triples"] == 3

    def test_filter_survives_reload(self, graph, monkeypatch):
        monkeypatch.setattr(graph_store, "_INDEX", None)
        assert graph.describe_subgraph(["Alice"], doc_ids={"fin"}).count("\n") == 0


class TestGraphStrategy:
    async def test_foreign_triples_and_chunks_never_reach_the_prompt(self, mem_store, graph):
        from app.rag.strategies.graph import GraphRAG

        async def fake_generate(**kwargs):
            prompts.append(kwargs["prompt"])
            return {"text": "ok", "input_tokens": 1, "output_tokens": 1}

        prompts: list[str] = []
        with (
            patch(
                "app.rag.strategies.graph.extract_question_entities",
                new=AsyncMock(return_value=["Alice"]),
            ),
            patch("app.rag.retrieve.embed_query_async", new=AsyncMock(return_value=VEC)),
            patch(
                "app.rag.strategies.graph.rerank_async",
                new=AsyncMock(side_effect=lambda q, chunks, top_k: chunks[:top_k]),
            ),
            patch("app.rag.strategies.graph.generate_with_usage", new=fake_generate),
        ):
            result = await GraphRAG().run(
                "Who does Alice work with?",
                top_k=20,
                model="claude-sonnet-5",
                prompt_version="default",
                access=scope("default", "hr"),
            )

        prompt = prompts[0]
        assert "manages" in prompt
        assert "audits" not in prompt and "Ledger" not in prompt
        assert "text of fin" not in prompt and "text of acme" not in prompt
        assert {s.chunk_id.split(":")[0] for s in result.sources} <= _expected(
            scope("default", "hr")
        )
        assert "ledger" not in result.extra["related_entities"]


# ---------------------------------------------------------------------------
# Classic and agentic strategies
# ---------------------------------------------------------------------------


class TestClassicStrategy:
    async def test_context_holds_only_permitted_chunks(self, mem_store):
        from app.rag.strategies.classic import ClassicRAG

        with (
            patch("app.rag.retrieve.embed_query_async", new=AsyncMock(return_value=VEC)),
            patch(
                "app.rag.strategies.classic.rerank_async",
                new=AsyncMock(side_effect=lambda q, chunks, top_k: chunks[:top_k]),
            ),
            patch(
                "app.rag.strategies.classic.generate_with_usage",
                new=AsyncMock(return_value={"text": "a", "input_tokens": 1, "output_tokens": 1}),
            ),
        ):
            result = await ClassicRAG().run(
                "q", top_k=20, model="m", prompt_version="default", access=scope("acme")
            )
        assert {c.split(":")[0] for c in result.extra["retrieved_ids"]} == {"acme"}
        assert "text of pub" not in result.extra["context_text"]


class TestAgenticStrategy:
    async def test_tools_cannot_reach_forbidden_chunks(self, mem_store):
        from app.rag.strategies.agentic import AgenticRAG

        seen = {}

        async def fake_loop(**kwargs):
            tools = kwargs["tool_handlers"]
            seen["search"] = await tools["search"]({"query": "anything", "top_k": 12})
            seen["fetch_forbidden"] = await tools["fetch_chunk"]({"chunk_id": "fin:0"})
            seen["fetch_allowed"] = await tools["fetch_chunk"]({"chunk_id": "hr:0"})
            return {"text": "", "input_tokens": 0, "output_tokens": 0, "iterations": 1, "trace": []}

        with (
            patch("app.rag.strategies.agentic.tool_use_loop", new=fake_loop),
            patch("app.rag.strategies.agentic.embed_query_async", new=AsyncMock(return_value=VEC)),
        ):
            result = await AgenticRAG().run(
                "q", top_k=8, model="m", prompt_version="v", access=scope("default", "hr")
            )

        assert "fin:0" not in seen["search"] and "acme" not in seen["search"]
        assert seen["fetch_forbidden"] == "error: chunk 'fin:0' not found"
        assert seen["fetch_allowed"] == "text of hr"
        assert "fin:0" not in result.extra["retrieved_ids"]


# ---------------------------------------------------------------------------
# The orchestrator threads the scope through
# ---------------------------------------------------------------------------


class _RecordingStrategy:
    name = "classic"
    provider = "anthropic"

    def __init__(self):
        self.access = "unset"

    async def run(self, question, *, top_k, model, prompt_version, access=None):
        from app.rag.strategies.base import StrategyResult

        self.access = access
        return StrategyResult(answer="a", sources=[Source(chunk_id="pub:0", quote="q")])


class TestGenerate:
    async def test_answer_question_passes_the_scope(self):
        from app.rag.generate import answer_question

        strat = _RecordingStrategy()
        counted = AsyncMock(return_value=5)
        with (
            patch("app.rag.generate.store_count", new=counted),
            patch("app.rag.generate.get_strategy", return_value=strat),
        ):
            await answer_question("q", access=scope("acme"))
        assert strat.access == scope("acme")
        counted.assert_awaited_once_with(scope("acme"))

    async def test_run_strategy_raw_passes_the_scope(self):
        from app.rag.generate import run_strategy_raw

        strat = _RecordingStrategy()
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=5)),
            patch("app.rag.generate.get_strategy", return_value=strat),
        ):
            await run_strategy_raw("q", strategy="classic", access=scope("default", "hr"))
        assert strat.access == scope("default", "hr")

    async def test_nothing_readable_looks_like_an_empty_index(self):
        from app.rag.generate import answer_question

        with patch("app.rag.generate.store_count", new=AsyncMock(return_value=0)):
            ans = await answer_question("q", access=scope("acme"))
        assert ans.refusal


# ---------------------------------------------------------------------------
# Routes always pass the caller's scope
# ---------------------------------------------------------------------------


def _answer(**kw) -> Answer:
    return Answer(
        question="q",
        answer="a",
        sources=[Source(chunk_id="hr:0", quote="x")],
        confidence=0.9,
        refusal=False,
        **kw,
    )


@pytest.fixture
def hr_key(auth_db, settings, monkeypatch):
    """Auth on, and a key in tenant 'acme' belonging to group 'hr'."""
    monkeypatch.setattr(settings, "require_api_key", True)
    key = auth.create_api_key("hr-bot", requests_per_minute=100, groups=["hr"], tenant="acme")
    return {"Authorization": f"Bearer {key}"}


HR_SCOPE = scope("acme", "hr")


class TestRoutesPassTheScope:
    def test_ask(self, client, hr_key):
        mock = AsyncMock(return_value=_answer())
        with patch("app.api.ask.answer_question", new=mock):
            r = client.post("/ask", json={"question": "q"}, headers=hr_key)
        assert r.status_code == 200
        assert mock.await_args.kwargs["access"] == HR_SCOPE

    def test_ask_in_local_mode(self, client):
        mock = AsyncMock(return_value=_answer())
        with patch("app.api.ask.answer_question", new=mock):
            assert client.post("/ask", json={"question": "q"}).status_code == 200
        assert mock.await_args.kwargs["access"] == scope()

    def test_compare_variants(self, client, hr_key):
        mock = AsyncMock(return_value=_answer())
        with patch("app.api.compare.answer_question", new=mock):
            r = client.post(
                "/compare",
                json={"question": "q", "variants": [{"strategy": "classic"}]},
                headers=hr_key,
            )
        assert r.status_code == 200
        assert all(c.kwargs["access"] == HR_SCOPE for c in mock.await_args_list)

    def test_compare_strategies(self, client, hr_key):
        mock = AsyncMock(return_value=(None, "x"))
        with patch("app.api.compare.run_strategy_raw", new=mock):
            r = client.post(
                "/compare/strategies",
                json={"question": "q", "strategies": ["classic", "graph"]},
                headers=hr_key,
            )
        assert r.status_code == 200
        assert [c.kwargs["access"] for c in mock.await_args_list] == [HR_SCOPE, HR_SCOPE]

    def test_eval_run(self, client, hr_key):
        mock = AsyncMock(return_value={"cost": {}})
        with (
            patch("app.api.eval_routes.run_evaluation", new=mock),
            patch("app.api.eval_routes.load_dataset", return_value=[{}]),
        ):
            r = client.post("/eval/run", json={"dataset": "d"}, headers=hr_key)
        assert r.status_code == 200
        assert mock.await_args.kwargs["access"] == HR_SCOPE

    def test_eval_metrics_threads_it_to_every_example(self, tmp_path, monkeypatch):
        from app.eval.metrics import run_evaluation

        monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
        (tmp_path / "d.jsonl").write_text('{"question": "one"}\n', encoding="utf-8")
        answer = AsyncMock(return_value=(_answer(), None))
        with patch("app.eval.metrics.answer_question_detailed", new=answer):
            import asyncio

            asyncio.run(run_evaluation(dataset="d", judge_answers=False, access=HR_SCOPE))
        assert answer.await_args.kwargs["access"] == HR_SCOPE

    def test_advise_validate(self, client, hr_key):
        from app.advisor.schemas import ValidateResponse

        empty = ValidateResponse(
            strategies=["classic"], scorecard=[], measured_winner=None, winner_reason="", rows=[]
        )
        mock = AsyncMock(return_value=(empty, 0, 0))
        with patch("app.api.advisor.run_validation", new=mock):
            r = client.post(
                "/advise/validate",
                json={"questions": [{"question": "q"}], "strategies": ["classic"]},
                headers=hr_key,
            )
        assert r.status_code == 200
        assert mock.await_args.kwargs["access"] == HR_SCOPE

    def test_advisor_validation_passes_it_to_each_run(self):
        import asyncio

        from app.advisor.schemas import ValidateRequest
        from app.advisor.validate import run_validation

        mock = AsyncMock(return_value=(None, "x"))
        req = ValidateRequest(questions=[{"question": "q"}], strategies=["classic"], judge=False)
        with patch("app.advisor.validate.run_strategy_raw", new=mock):
            asyncio.run(run_validation(req, access=HR_SCOPE))
        assert mock.await_args.kwargs["access"] == HR_SCOPE

    def test_ingest_stats_is_scoped(self, client, hr_key):
        mock = AsyncMock(return_value=3)
        with patch("app.api.ingest.store_count", new=mock):
            assert client.get("/ingest/stats", headers=hr_key).json() == {"indexed_chunks": 3}
        mock.assert_awaited_once_with(HR_SCOPE)

    def test_graph_entities_are_scoped(self, client, hr_key, graph):
        with patch("app.api.graph.readable_doc_ids", new=AsyncMock(return_value={"fin"})) as rd:
            body = client.get("/graph/entities", headers=hr_key).json()
        rd.assert_awaited_once_with(HR_SCOPE)
        assert body == {"entities": ["alice", "ledger"], "total": 2}


class TestIngestRoute:
    def _post(self, client, headers=None, groups=None):
        enqueue = AsyncMock(return_value={"doc_id": "d1", "chunks": 1})
        data = {"groups": groups} if groups is not None else None
        with (
            patch("app.api.ingest.enqueue_document", new=enqueue),
            patch("app.api.ingest.store_count", new=AsyncMock(return_value=1)),
        ):
            r = client.post(
                "/ingest", files={"file": ("a.txt", b"hello")}, data=data, headers=headers
            )
        return r, enqueue

    def test_defaults_to_the_uploaders_groups(self, client, hr_key):
        r, enqueue = self._post(client, hr_key)
        assert r.status_code == 200
        kwargs = enqueue.await_args.kwargs
        assert (kwargs["tenant"], kwargs["acl_groups"], kwargs["owner"]) == (
            "acme",
            ["hr"],
            "key:1",
        )

    def test_explicit_groups_comma_separated(self, client, hr_key):
        r, enqueue = self._post(client, hr_key, groups="hr,public")
        assert r.status_code == 200
        assert enqueue.await_args.kwargs["acl_groups"] == ["hr", "public"]

    def test_explicit_groups_repeated(self, client, hr_key):
        r, enqueue = self._post(client, hr_key, groups=["hr", "public"])
        assert r.status_code == 200
        assert enqueue.await_args.kwargs["acl_groups"] == ["hr", "public"]

    def test_foreign_group_is_403_and_nothing_is_indexed(self, client, hr_key):
        r, enqueue = self._post(client, hr_key, groups="finance")
        assert r.status_code == 403
        enqueue.assert_not_awaited()

    def test_admin_key_may_label_any_group(self, client, auth_db, settings, monkeypatch):
        monkeypatch.setattr(settings, "require_api_key", True)
        key = auth.create_api_key("root", groups=[], is_admin=True)
        r, enqueue = self._post(client, {"Authorization": f"Bearer {key}"}, groups="finance")
        assert r.status_code == 200
        assert enqueue.await_args.kwargs["acl_groups"] == ["finance"]

    def test_local_mode_uploads_are_public(self, client):
        r, enqueue = self._post(client)
        assert r.status_code == 200
        assert enqueue.await_args.kwargs["acl_groups"] == ["public"]
        assert enqueue.await_args.kwargs["owner"] == "local"

