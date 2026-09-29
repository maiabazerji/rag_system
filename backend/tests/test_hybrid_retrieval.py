"""Hybrid retrieval: query preprocessing, BM25, RRF fusion and the pipeline."""
from __future__ import annotations

import itertools
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.access import PUBLIC_GROUP, AccessScope
from app.config import Settings
from app.rag import rerank as rerank_module
from app.rag import sparse, store
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.query import MAX_QUERY_CHARS, bm25_tokenize, preprocess_query
from app.rag.rerank import RerankOutcome
from app.rag.retrieve import hybrid_search
from app.rag.sparse import SparseIndex
from app.resilience import CircuitBreaker
from app.schemas import Chunk, RetrievalDiagnostics
from app.tracing.spans import Trace, activate


def _chunk(cid: str, text: str = "", **metadata) -> Chunk:
    return Chunk(
        id=cid,
        doc_id=cid.split(":")[0],
        text=text or f"text of {cid}",
        tokens=3,
        metadata=metadata,
    )


def _record(cid: str, text: str, **fields) -> SimpleNamespace:
    return SimpleNamespace(
        payload={"chunk_id": cid, "doc_id": cid.split(":")[0], "text": text, **fields}
    )


def scope(tenant: str = "default", *groups: str) -> AccessScope:
    return AccessScope(tenant=tenant, groups=frozenset({PUBLIC_GROUP, *groups}))


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_nfkc_whitespace_and_invisible_characters(self):
        q = preprocess_query("  ＲＡＧ vs​  BM25\n\tﬁne-tuning  ")
        assert q.text == "RAG vs BM25 fine-tuning"
        assert q.original.startswith("  Ｒ")  # generation keeps the original
        assert not q.truncated

    def test_long_queries_are_capped(self):
        q = preprocess_query("word " * 1000)
        assert q.truncated
        assert len(q.text) <= MAX_QUERY_CHARS
        assert not q.text.endswith(" ")

    def test_tokens_are_folded_and_stopworded(self):
        assert preprocess_query("Qu'est-ce qu'une REQUÊTE ?").tokens == ("requete",)
        assert preprocess_query("   ").tokens == ()

    def test_rerank_still_exports_the_shared_tokenizer(self):
        assert rerank_module.bm25_tokenize is bm25_tokenize


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestRRF:
    def test_hand_computed_scores_and_order(self):
        a, b, c, d = (_chunk(x) for x in "abcd")
        fused = reciprocal_rank_fusion({"dense": [a, b, c], "sparse": [c, d]}, k=60)

        scores = {f.chunk_id: f.fused_score for f in fused}
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)
        assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
        assert scores["d"] == pytest.approx(1 / 62)
        # c wins on agreement; b and d tie on score and on best rank (2), so id.
        assert [f.chunk_id for f in fused] == ["c", "a", "b", "d"]
        assert (fused[0].dense_rank, fused[0].sparse_rank) == (3, 1)
        assert (fused[1].dense_rank, fused[1].sparse_rank) == (1, None)
        assert (fused[3].dense_rank, fused[3].sparse_rank) == (None, 2)

    def test_small_k_example(self):
        x, y, z = _chunk("x"), _chunk("y"), _chunk("z")
        fused = reciprocal_rank_fusion({"dense": [x, y], "sparse": [z, y]}, k=1)
        # y: 1/3 + 1/3 = 0.667 > x: 1/2 = z: 1/2 (tie -> best rank 1 both -> id)
        assert [f.chunk_id for f in fused] == ["y", "x", "z"]
        assert fused[0].fused_score == pytest.approx(2 / 3)

    def test_same_chunk_in_both_lists_is_one_entry_with_summed_score(self):
        dense_copy = _chunk("d:1", "dense payload")
        sparse_copy = _chunk("d:1", "sparse payload")
        other = _chunk("d:2")
        fused = reciprocal_rank_fusion(
            {"dense": [other, dense_copy], "sparse": [sparse_copy]}, k=60
        )
        assert [f.chunk_id for f in fused].count("d:1") == 1
        top = next(f for f in fused if f.chunk_id == "d:1")
        assert top.fused_score == pytest.approx(1 / 62 + 1 / 61)
        # The payload comes from the list that ranked it best (sparse, rank 1).
        assert top.chunk.text == "sparse payload"

    def test_duplicate_within_one_list_counts_once_at_its_best_rank(self):
        a, b = _chunk("a"), _chunk("b")
        fused = reciprocal_rank_fusion({"dense": [a, b, a]}, k=60)
        assert len(fused) == 2
        assert fused[0].fused_score == pytest.approx(1 / 61)
        assert fused[0].dense_rank == 1

    def test_ranking_is_stable_across_runs_and_input_permutations(self):
        chunks = [_chunk(f"c{i}") for i in range(6)]
        # Every chunk appears once at the same rank in one list or the other,
        # so many exact ties: only the tie-break decides the order.
        expected = None
        for perm in itertools.permutations(["dense", "sparse"]):
            for _ in range(5):
                rankings = {perm[0]: chunks[:3], perm[1]: chunks[3:]}
                order = [f.chunk_id for f in reciprocal_rank_fusion(rankings, k=60)]
                expected = expected or order
                assert order == expected
        assert expected == ["c0", "c3", "c1", "c4", "c2", "c5"]

    def test_input_list_order_of_tied_items_does_not_matter(self):
        pool = [_chunk(f"t{i}") for i in range(8)]
        baseline = None
        rng = random.Random(7)
        for _ in range(10):
            shuffled = pool[:]
            rng.shuffle(shuffled)
            # Each chunk alone in its own list at rank 1: all tie completely.
            rankings = {f"r{i}": [c] for i, c in enumerate(shuffled)}
            order = [f.chunk_id for f in reciprocal_rank_fusion(rankings)]
            baseline = baseline or order
            assert order == baseline == sorted(order)

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({"dense": []}, k=0)


# ---------------------------------------------------------------------------
# Sparse (BM25) index
# ---------------------------------------------------------------------------


CORPUS = [
    _record("en:0", "BM25 is a lexical ranking function based on term frequency"),
    _record("en:1", "Dense retrieval embeds the query into a vector space"),
    _record("en:2", "Reciprocal rank fusion merges several ranked lists"),
    _record("fr:0", "Le résumé de la requête est envoyé au modèle de langage"),
    _record("fr:1", "La fusion des rangs réciproques combine plusieurs listes"),
    _record("hr:0", "Salary bands and bonus policy", tenant="default", acl_groups=["hr"]),
    _record("fin:0", "Quarterly bonus accruals ledger", tenant="default", acl_groups=["finance"]),
    _record("acme:0", "Acme bonus scheme", tenant="acme", acl_groups=["public"]),
]


@pytest.fixture
def fake_store(monkeypatch):
    """Point the sparse index at an in-memory list of records."""
    state = {"records": list(CORPUS)}
    scroll = AsyncMock(side_effect=lambda **_: list(state["records"]))
    count = AsyncMock(side_effect=lambda *a, **k: len(state["records"]))
    monkeypatch.setattr(sparse.store, "scroll_chunks", scroll)
    monkeypatch.setattr(sparse.store, "count", count)
    state["scroll"] = scroll
    return state


class TestSparseIndex:
    async def test_ranks_lexical_matches(self, fake_store):
        idx = SparseIndex()
        hits = await idx.search(bm25_tokenize("What does BM25 measure?"), top_k=5)
        assert [h.chunk_id for h in hits] == ["en:0"]
        assert hits[0].score > 0

    async def test_french_accents_are_folded(self, fake_store):
        idx = SparseIndex()
        # No accents in the query, accents in the document, and vice versa.
        hits = await idx.search(bm25_tokenize("resume de la requete"), top_k=5)
        assert hits[0].chunk_id == "fr:0"
        hits = await idx.search(bm25_tokenize("Fusion des RANGS RÉCIPROQUES"), top_k=5)
        assert hits[0].chunk_id == "fr:1"

    async def test_non_matching_chunks_are_not_returned(self, fake_store):
        hits = await SparseIndex().search(bm25_tokenize("zeppelin"), top_k=5)
        assert hits == []

    async def test_top_k_and_deterministic_ties(self, fake_store):
        fake_store["records"] = [_record(f"d:{i}", "same words here") for i in (3, 1, 2)]
        hits = await SparseIndex().search(bm25_tokenize("same words"), top_k=2)
        assert [h.chunk_id for h in hits] == ["d:1", "d:2"]

    async def test_acl_filters_before_ranking(self, fake_store):
        idx = SparseIndex()
        tokens = bm25_tokenize("bonus")
        assert {h.chunk_id for h in await idx.search(tokens, 10)} == {
            "hr:0",
            "fin:0",
            "acme:0",
        }
        assert [h.chunk_id for h in await idx.search(tokens, 10, scope("default", "hr"))] == [
            "hr:0"
        ]
        assert [h.chunk_id for h in await idx.search(tokens, 10, scope("acme"))] == ["acme:0"]
        # A scope that can read none of the matches gets nothing at all.
        assert await idx.search(tokens, 10, scope("default")) == []

    async def test_cached_scope_models_never_cross_scopes(self, fake_store):
        idx = SparseIndex()
        tokens = bm25_tokenize("bonus ledger accruals")
        fin = await idx.search(tokens, 10, scope("default", "finance"))
        hr = await idx.search(tokens, 10, scope("default", "hr"))
        assert "fin:0" in {h.chunk_id for h in fin}
        assert "fin:0" not in {h.chunk_id for h in hr}

    async def test_index_is_cached_until_the_corpus_changes(self, fake_store, monkeypatch):
        idx = SparseIndex()
        tokens = bm25_tokenize("zeppelin airship")
        await idx.search(tokens, 5)
        await idx.search(tokens, 5)
        assert idx.rebuilds == 1
        assert fake_store["scroll"].await_count == 1

        # A new point (count changes) is picked up.
        fake_store["records"].append(_record("new:0", "zeppelin airship history"))
        assert [h.chunk_id for h in await idx.search(tokens, 5)] == ["new:0"]
        assert idx.rebuilds == 2

        # An in-place edit keeps the count; the generation counter catches it.
        fake_store["records"][-1] = _record("new:0", "something else entirely")
        assert [h.chunk_id for h in await idx.search(tokens, 5)] == ["new:0"]  # stale
        store.bump_corpus_generation()
        assert await idx.search(tokens, 5) == []
        assert idx.rebuilds == 3

    async def test_store_writes_bump_the_generation(self, monkeypatch):
        qdrant = AsyncMock()
        qdrant.count = AsyncMock(return_value=SimpleNamespace(count=2))
        monkeypatch.setattr(store, "_client", qdrant)
        monkeypatch.setattr(
            store, "_qdrant_breaker", CircuitBreaker(failure_threshold=5, service_name="Q")
        )
        before = store.corpus_generation()
        await store.upsert([_chunk("u:0")], [[0.0]])
        assert store.corpus_generation() == before + 1
        await store.delete_matching(qm.Filter())
        assert store.corpus_generation() == before + 2


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


DENSE = [_chunk("a:0", filename="a.md"), _chunk("b:0", filename="b.md")]
SPARSE_HITS = [
    sparse.SparseHit("c:0", {"chunk_id": "c:0", "doc_id": "c", "text": "text of c"}, 3.0),
    sparse.SparseHit("a:0", {"chunk_id": "a:0", "doc_id": "a", "text": "text of a"}, 2.0),
]


def _scored(q, chunks, top_k):
    kept = chunks[:top_k]
    return RerankOutcome(
        chunks=kept, scores=[10.0 - i for i in range(len(kept))], reranker="cross-encoder"
    )


@pytest.fixture
def mocked_retrievers():
    with (
        patch("app.rag.retrieve.dense_search", new=AsyncMock(return_value=DENSE)) as dense,
        patch("app.rag.retrieve.sparse_search", new=AsyncMock(return_value=SPARSE_HITS)) as sp,
        patch("app.rag.retrieve.rerank_scored_async", new=AsyncMock(side_effect=_scored)) as rr,
    ):
        yield SimpleNamespace(dense=dense, sparse=sp, rerank=rr)


class TestHybridSearch:
    async def test_hybrid_fuses_both_and_reports_diagnostics(self, mocked_retrievers):
        result = await hybrid_search("  what is  a? ", scope(), mode="hybrid", final_k=3)

        mocked_retrievers.dense.assert_awaited_once()
        assert mocked_retrievers.dense.await_args.args[0] == "what is a?"
        mocked_retrievers.sparse.assert_awaited_once()
        # a:0 is in both lists, so it leads the fused list.
        assert result.chunk_ids == ["a:0", "c:0", "b:0"]

        diag = result.diagnostics
        assert diag.mode == "hybrid" and diag.reranker == "cross-encoder"
        assert diag.counts.model_dump() == {
            "dense": 2, "sparse": 2, "fused": 3, "reranked": 3, "final": 3,
        }
        first = diag.chunks[0]
        assert (first.chunk_id, first.doc_id) == ("a:0", "a")
        assert (first.dense_rank, first.sparse_rank) == (1, 2)
        assert first.fused_score == pytest.approx(1 / 61 + 1 / 62)
        assert first.rerank_score == 10.0
        assert diag.chunks[1].dense_rank is None and diag.chunks[1].sparse_rank == 1
        assert set(diag.latency_ms.model_dump()) == {
            "preprocess", "dense", "sparse", "fusion", "rerank",
        }
        assert "text of" not in diag.model_dump_json()

    async def test_dense_mode_skips_bm25(self, mocked_retrievers):
        result = await hybrid_search("q", None, mode="dense")
        mocked_retrievers.sparse.assert_not_awaited()
        assert result.chunk_ids == ["a:0", "b:0"]
        assert result.diagnostics.counts.sparse == 0
        assert all(c.sparse_rank is None for c in result.diagnostics.chunks)

    async def test_sparse_mode_skips_dense(self, mocked_retrievers):
        result = await hybrid_search("q", None, mode="sparse")
        mocked_retrievers.dense.assert_not_awaited()
        assert result.chunk_ids == ["c:0", "a:0"]
        assert result.diagnostics.mode == "sparse"

    async def test_mode_defaults_to_setting(self, mocked_retrievers, settings):
        settings.retrieval_mode = "sparse"
        await hybrid_search("q")
        mocked_retrievers.dense.assert_not_awaited()

    async def test_unknown_mode_is_rejected(self, mocked_retrievers):
        with pytest.raises(ValueError):
            await hybrid_search("q", mode="telepathy")

    async def test_rerank_off_returns_fused_order_unscored(self, mocked_retrievers):
        result = await hybrid_search("q", mode="hybrid", rerank=False, final_k=2)
        mocked_retrievers.rerank.assert_not_awaited()
        assert result.chunk_ids == ["a:0", "c:0"]
        assert result.diagnostics.reranker == "none"
        assert result.diagnostics.chunks[0].rerank_score is None

    async def test_final_k_cuts_the_reranked_list(self, mocked_retrievers, settings):
        settings.rerank_top_k = 3
        result = await hybrid_search("q", mode="hybrid", final_k=1)
        assert mocked_retrievers.rerank.await_args.kwargs["top_k"] == 3
        assert len(result.chunks) == 1
        assert result.diagnostics.counts.reranked == 3

    async def test_blank_query_retrieves_nothing(self, mocked_retrievers):
        result = await hybrid_search(" ​ ", mode="hybrid")
        assert result.chunks == []
        mocked_retrievers.dense.assert_not_awaited()

    async def test_bm25_outage_degrades_hybrid_but_fails_sparse(self, mocked_retrievers):
        mocked_retrievers.sparse.side_effect = store.StoreUnavailable("down")
        result = await hybrid_search("q", mode="hybrid")
        assert result.chunk_ids == ["a:0", "b:0"]
        assert result.diagnostics.degraded == ["sparse"]
        with pytest.raises(store.StoreUnavailable):
            await hybrid_search("q", mode="sparse")

    async def test_each_stage_is_a_span(self, mocked_retrievers):
        trace = Trace.new("test", {})
        with activate(trace):
            await hybrid_search("q", mode="hybrid")
        names = {s.name for s in trace.spans}
        assert {
            "retrieval.preprocess",
            "retrieval.dense",
            "retrieval.sparse",
            "retrieval.fusion",
            "retrieval.rerank",
        } <= names
        fusion = next(s for s in trace.spans if s.name == "retrieval.fusion")
        assert fusion.metadata["fused"] == 3
        # Stage spans carry counts, never content.
        for s in trace.spans:
            if s.name.startswith("retrieval."):
                assert s.input is None and s.output is None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self):
        s = Settings(_env_file=None)
        assert s.retrieval_mode == "hybrid"
        assert (s.dense_top_k, s.bm25_top_k, s.rrf_k) == (50, 50, 60)
        assert s.final_context_k == s.rerank_top_k

    def test_mode_is_validated(self):
        assert Settings(_env_file=None, retrieval_mode=" Dense ").retrieval_mode == "dense"
        with pytest.raises(ValidationError):
            Settings(_env_file=None, retrieval_mode="fuzzy")

    def test_retrieval_top_k_is_a_deprecated_alias(self):
        assert Settings(_env_file=None, retrieval_top_k=30).dense_top_k == 30
        both = Settings(_env_file=None, retrieval_top_k=30, dense_top_k=40)
        assert both.dense_top_k == 40

    def test_final_context_k_follows_rerank_top_k_and_is_bounded_by_it(self):
        assert Settings(_env_file=None, rerank_top_k=5).final_context_k == 5
        assert Settings(_env_file=None, rerank_top_k=10, final_context_k=4).final_context_k == 4
        with pytest.raises(ValidationError):
            Settings(_env_file=None, rerank_top_k=4, final_context_k=6)


# ---------------------------------------------------------------------------
# End to end: API response and access control on a real in-memory Qdrant
# ---------------------------------------------------------------------------


class TestAskResponse:
    def test_ask_exposes_retrieval_diagnostics(self, client, settings, mocked_retrievers):
        settings.retrieval_mode = "hybrid"
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
            patch(
                "app.rag.strategies.classic.generate_structured",
                new=AsyncMock(
                    return_value={"text": "An answer [S1].", "input_tokens": 1, "output_tokens": 1}
                ),
            ),
        ):
            r = client.post("/ask", json={"question": "what is a?", "top_k": 2})

        assert r.status_code == 200, r.text
        body = r.json()
        diag = RetrievalDiagnostics.model_validate(body["retrieval"])
        assert diag.mode == "hybrid"
        assert [c.chunk_id for c in diag.chunks] == ["a:0", "c:0"]
        assert diag.counts.final == 2
        assert "text of" not in str(body["retrieval"])

    def test_compare_strategies_exposes_retrieval_diagnostics(
        self, client, settings, mocked_retrievers
    ):
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
            patch(
                "app.rag.strategies.classic.generate_structured",
                new=AsyncMock(
                    return_value={"text": "An answer [S1].", "input_tokens": 1, "output_tokens": 1}
                ),
            ),
        ):
            r = client.post(
                "/compare/strategies", json={"question": "what is a?", "strategies": ["classic"]}
            )
        assert r.status_code == 200, r.text
        row = r.json()["results"][0]
        assert row["retrieval"]["mode"] == "dense"
        assert row["extra"]["retrieved_ids"] == ["a:0", "b:0"]


VEC = [1.0, 0.0, 0.0, 0.0]
ACL_CORPUS = {
    "pub": {"tenant": "default", "acl_groups": ["public"]},
    "hr": {"tenant": "default", "acl_groups": ["hr"]},
    "fin": {"tenant": "default", "acl_groups": ["finance"]},
    "acme": {"tenant": "acme", "acl_groups": ["public"]},
}


@pytest.fixture
async def mem_store(monkeypatch, settings):
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
            text=f"payroll ledger of {doc}",
            tokens=4,
            metadata={"filename": f"{doc}.md", **fields},
        )
        for doc, fields in ACL_CORPUS.items()
    ]
    await store.upsert(chunks, [VEC] * len(chunks))
    yield client
    await client.close()


class TestHybridAccessControl:
    @pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
    @pytest.mark.parametrize(
        "access, readable",
        [
            (scope("default"), {"pub"}),
            (scope("default", "hr"), {"pub", "hr"}),
            (scope("acme"), {"acme"}),
        ],
    )
    async def test_only_readable_chunks_come_back(self, mem_store, mode, access, readable):
        with (
            patch("app.rag.retrieve.embed_query_async", new=AsyncMock(return_value=VEC)),
            patch(
                "app.rag.retrieve.rerank_scored_async",
                new=AsyncMock(side_effect=lambda q, c, top_k: RerankOutcome.unscored(c[:top_k])),
            ),
        ):
            result = await hybrid_search("payroll ledger", access, mode=mode, final_k=10)
        assert {c.doc_id for c in result.chunks} == readable
        assert {c.doc_id for c in result.diagnostics.chunks} == readable

    async def test_ingest_invalidates_the_live_index(self, mem_store):
        tokens = bm25_tokenize("zeppelin")
        assert await sparse.sparse_search(tokens, 5) == []
        await store.upsert([_chunk("z:0", "zeppelin notes")], [VEC])
        assert [h.chunk_id for h in await sparse.sparse_search(tokens, 5)] == ["z:0"]
