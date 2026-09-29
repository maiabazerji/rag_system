"""End-to-end tests of the critical paths, fully offline.

Everything between the HTTP request and the model call is the real code:
parsers, chunking, PII masking, the Qdrant store (a real in-memory
``AsyncQdrantClient``, created and model-stamped by ``store.client()``), the
BM25 index and its invalidation, RRF fusion, the reranker's BM25 fallback,
citation validation, pricing, tracing, access control, the eval harness, the
regression gate and the erasure targets.

Three test doubles stand in for what needs the network or a model download:

* **Embedding model** -- ``HashedBagOfWords`` replaces the SentenceTransformer
  behind ``app.rag.embed._model``. It is deterministic (md5-hashed token
  counts, L2-normalised, 256 dimensions), so dense retrieval does rank lexically
  similar text higher, but it has no semantics. The real ``embed_*`` functions,
  input prefixes and thread pool still run.
* **Cross-encoder** -- ``sentence_transformers`` is made unimportable, so the
  reranker's real load fails fast and its real BM25 fallback runs.
* **Anthropic API** -- ``ScriptedAnthropic`` replaces the SDK client behind
  ``anthropic_provider.get_client()``. It returns a scripted ``submit_answer``
  tool call for generation and scripted rubric JSON for the judge. The real
  provider code (retries, circuit breaker, usage spans, cost) runs around it.

What these tests cannot tell you: the quality of real embeddings or of the
cross-encoder, how a real model follows the prompt, or anything about a
networked Qdrant (timeouts, payload index performance).
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from document_builders import make_docx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app import pricing
from app.eval import metrics as eval_metrics
from app.eval import regression
from app.rag import embed, graph_store, rerank, sparse, store
from app.rag.collection_model import EMBEDDING_MODEL_KEY
from app.rag.ingest import enqueue_document
from app.rag.providers import anthropic_provider
from app.rag.query import bm25_tokenize
from app.resilience import CircuitBreaker

REPO = Path(__file__).resolve().parents[2]
FR_DOC = REPO / "data" / "demo_fr_business" / "docs" / "06_facture_FA-2026-0311.md"

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class HashedBagOfWords:
    """Deterministic stand-in for a SentenceTransformer (test double).

    Each lowercase word token is hashed (md5, stable across processes) into
    one of ``DIM`` buckets; the vector is the bucket counts, L2-normalised.
    Texts sharing words get a positive cosine similarity; nothing else does.
    """

    DIM = 256

    def encode(self, texts: list[str], normalize_embeddings: bool = False) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self.DIM)
            for token in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if normalize_embeddings and norm else vec)
        return np.array(rows)

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM


def _rubric(score: float) -> str:
    dims = ("faithfulness", "answer_relevance", "context_precision", "context_recall",
            "answer_correctness")
    return json.dumps({d: {"score": score, "reasoning": f"{d} ok"} for d in dims})


class ScriptedAnthropic:
    """Stand-in for ``AsyncAnthropic`` (test double): scripted Messages API replies.

    A request carrying ``tools`` is a generation call and gets ``answer`` back as
    a ``submit_answer`` tool call. Any other request is a judge call: its reply is
    ``judge_reply``, or ``"not json"`` when the prompt contains a phrase listed
    in ``judge_breaks_on`` (so both the first try and the repair retry fail).
    """

    GEN_USAGE = (1200, 150)
    JUDGE_USAGE = (800, 120)

    def __init__(self) -> None:
        self.answer: dict[str, Any] = {}
        self.judge_score = 0.9
        self.judge_breaks_on: tuple[str, ...] = ()
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    @staticmethod
    def _response(content: list, usage: tuple[int, int], stop: str) -> SimpleNamespace:
        return SimpleNamespace(
            content=content,
            usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
            stop_reason=stop,
        )

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if kwargs.get("tools"):
            block = SimpleNamespace(type="tool_use", name="submit_answer", input=self.answer)
            return self._response([block], self.GEN_USAGE, "tool_use")
        prompt = kwargs["messages"][-1]["content"]
        broken = any(marker in prompt for marker in self.judge_breaks_on)
        text = "not json" if broken else _rubric(self.judge_score)
        return self._response(
            [SimpleNamespace(type="text", text=text)], self.JUDGE_USAGE, "end_turn"
        )


def cited_answer(text: str, *handles: str, status: str = "answered") -> dict[str, Any]:
    """A ``submit_answer`` tool input with one claim per handle."""
    markers = "".join(f"[{h}]" for h in handles)
    return {
        "answer": f"{text} {markers}".strip(),
        "claims": [
            {"text": f"{text} [{h}]", "citations": [h], "supported": True} for h in handles
        ],
        "status": status,
        "unsupported_notes": "",
    }


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

DOCS: dict[str, bytes] = {
    "remote_work.md": (
        b"# Remote work policy\n\n"
        b"Employees may work remotely up to three days per week. Remote work "
        b"requires written approval from the line manager, and the remaining days "
        b"are spent in the office.\n\n"
        b"## Equipment\n\nThe company provides a laptop and a headset for remote work.\n"
    ),
    "expenses.md": (
        b"# Expense reimbursement\n\n"
        b"Submit expense claims with the original receipts within 30 days of "
        b"purchase. Claims above 500 euros need approval from the finance "
        b"director. Contact jane.doe@example.com with questions.\n"
    ),
    "security.md": (
        b"# Password security\n\n"
        b"Passwords must be rotated every 90 days and contain at least 14 "
        b"characters. Multi-factor authentication is mandatory for email and VPN.\n"
    ),
}
LEGAL_DOC = (
    "legal_hold.md",
    b"# Litigation hold\n\nThe zanzibar litigation hold freezes every contract "
    b"signed with the supplier Kestrel since 2024. Do not delete these files.\n",
)
DOCX_NAME = "rapport_annuel.docx"
FR_NAME = FR_DOC.name


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def pipeline(settings, monkeypatch, tmp_path):
    """Wire the real pipeline to offline doubles; yield the scripted model.

    The Qdrant client is created by ``store.client()`` itself (through
    ``_ensure_collection``), so collection creation and model stamping are
    exercised; only its constructor is redirected to in-memory mode.
    """
    qdrant = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(store, "AsyncQdrantClient", lambda **_: qdrant)
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(
        store, "_qdrant_breaker", CircuitBreaker(failure_threshold=5, service_name="Qdrant")
    )

    monkeypatch.setattr(embed, "_model", HashedBagOfWords)

    # The cross-encoder cannot be imported, so its load fails and BM25 takes over.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(rerank, "_cross_encoder_model", None)
    monkeypatch.setattr(rerank, "_cross_encoder_failed_at", None)
    monkeypatch.setattr(rerank, "_cross_encoder_error", None)

    model = ScriptedAnthropic()
    monkeypatch.setattr(anthropic_provider, "_client", model)

    settings.retrieval_mode = "hybrid"
    monkeypatch.setattr(settings, "graph_data_dir", str(tmp_path / "graph"))
    monkeypatch.setattr(graph_store, "_INDEX", None)

    yield SimpleNamespace(model=model, qdrant=qdrant)
    _run(qdrant.close())


@pytest.fixture
def corpus(pipeline):
    """Ingest the corpus through the real ingest path; return filename -> report."""
    reports: dict[str, dict] = {}

    async def ingest_all():
        for name, content in DOCS.items():
            reports[name] = await enqueue_document(name, content)
        reports[DOCX_NAME] = await enqueue_document(DOCX_NAME, make_docx())
        reports[FR_NAME] = await enqueue_document(FR_NAME, FR_DOC.read_bytes())
        reports[LEGAL_DOC[0]] = await enqueue_document(*LEGAL_DOC, acl_groups=["legal"])

    _run(ingest_all())
    return reports


async def _payloads(doc_id: str | None = None) -> list[dict]:
    selector = (
        qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
        if doc_id
        else qm.Filter()
    )
    return await store.scroll_payloads(selector)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_collection_is_created_and_stamped_with_the_embedding_model(
        self, pipeline, corpus, settings
    ):
        info = _run(pipeline.qdrant.get_collection(settings.qdrant_collection))
        assert info.config.params.vectors.size == HashedBagOfWords.DIM
        assert info.config.metadata[EMBEDDING_MODEL_KEY] == settings.embedding_model

    def test_every_format_is_chunked_and_indexed_with_its_acl(self, corpus):
        payloads = _run(_payloads())
        by_file: dict[str, list[dict]] = {}
        for p in payloads:
            by_file.setdefault(p["filename"], []).append(p)

        assert set(by_file) == set(corpus)
        for name, report in corpus.items():
            assert len(by_file[name]) == report["chunks"] > 0
            assert {p["doc_id"] for p in by_file[name]} == {report["doc_id"]}
        assert {tuple(p["acl_groups"]) for p in by_file[LEGAL_DOC[0]]} == {("legal",)}
        assert {tuple(p["acl_groups"]) for p in by_file["security.md"]} == {("public",)}
        docx_text = " ".join(p["text"] for p in by_file[DOCX_NAME])
        assert "Loyer" in docx_text and "1200" in docx_text

    def test_personal_data_is_masked_before_it_is_stored(self, corpus):
        stored = json.dumps(_run(_payloads()), ensure_ascii=False)
        assert "jane.doe@example.com" not in stored
        assert "contact@atelier-fictif.example.com" not in stored
        assert corpus["expenses.md"]["pii_counts"].get("EMAIL") == 1
        assert corpus[FR_NAME]["pii_counts"]

    def test_reingesting_unchanged_content_is_idempotent(self, corpus):
        before = _run(store.count())
        again = _run(enqueue_document("remote_work.md", DOCS["remote_work.md"]))
        assert again["doc_id"] == corpus["remote_work.md"]["doc_id"]
        assert _run(store.count()) == before


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

QUESTION = "How many days per week can employees work remotely?"
PIPELINE_STAGES = [
    "preprocess", "dense", "sparse", "fusion", "rerank",
    "context_selection", "generation", "citation_validation", "response",
]


class TestAsk:
    def test_hybrid_answer_is_validated_measured_and_traced(
        self, client, pipeline, corpus, settings
    ):
        pipeline.model.answer = cited_answer(
            "Employees may work remotely up to three days per week.", "S1", "S9"
        )
        r = client.post("/ask", json={"question": QUESTION})
        assert r.status_code == 200, r.text
        body = r.json()

        # Retrieval: both retrievers ran, were fused and reranked by BM25.
        diag = body["retrieval"]
        assert diag["mode"] == "hybrid"
        assert diag["reranker"] == "bm25-fallback"
        assert diag["degraded"] == []
        counts = diag["counts"]
        assert counts["dense"] > 0 and counts["sparse"] > 0
        assert max(counts["dense"], counts["sparse"]) <= counts["fused"]
        assert counts["fused"] <= counts["dense"] + counts["sparse"]
        assert 0 < counts["final"] <= settings.final_context_k < 9
        top = diag["chunks"][0]
        assert top["rerank_score"] is not None and top["fused_score"] is not None

        # Citations: S1 kept and mapped to the real chunk; S9 was never shown.
        assert body["invalid_citations"] == ["S9"]
        assert [s["handle"] for s in body["sources"]] == ["S1"]
        source = body["sources"][0]
        assert source["chunk_id"] == top["chunk_id"]
        assert source["document_id"] == corpus["remote_work.md"]["doc_id"]
        (stored,) = _run(store.fetch_chunks([source["chunk_id"]]))
        assert stored.payload["doc_id"] == source["document_id"]
        assert stored.payload["title"] == source["title"]
        assert source["quote"] in stored.payload["text"]
        assert "[S9]" not in body["answer"] and "[S1]" in body["answer"]
        assert body["grounded"] is False
        assert body["status"] == "answered"
        assert body["refusal"] is False
        assert body["citation_count"] == 1

        # Cost and latency, priced with the configured generator model.
        metrics = body["metrics"]
        tokens_in, tokens_out = ScriptedAnthropic.GEN_USAGE
        assert (metrics["input_tokens"], metrics["output_tokens"]) == (tokens_in, tokens_out)
        assert metrics["llm_calls"] == 1
        assert metrics["model"] == settings.generator_model
        assert metrics["estimated_cost_usd"] == pytest.approx(
            pricing.estimate_cost(settings.generator_model, tokens_in, tokens_out)
        )
        assert metrics["estimated_cost_usd"] > 0
        latency = metrics["latency_ms"]
        for stage in ("retrieval", "rerank", "generation", "citation_validation"):
            assert latency[stage] >= 0
        assert latency["total"] >= latency["retrieval"] + latency["generation"]
        assert metrics["context_chunks"] == counts["final"]

        # The model saw the context under handles, never raw chunk ids.
        (call,) = pipeline.model.calls
        prompt = call["messages"][-1]["content"]
        assert "[S1]" in prompt and source["chunk_id"] not in prompt

        trace = client.get(f"/traces/{body['trace_id']}").json()
        assert [s["stage"] for s in trace["stages"]] == PIPELINE_STAGES
        starts = [s["start_ms"] for s in trace["stages"]]
        assert starts == sorted(starts)
        assert corpus["remote_work.md"]["doc_id"] in trace["doc_ids"]

    def test_fully_supported_answer_is_grounded(self, client, pipeline, corpus):
        pipeline.model.answer = cited_answer("Up to three days per week.", "S1")
        body = client.post("/ask", json={"question": QUESTION}).json()
        assert body["grounded"] is True
        assert body["invalid_citations"] == []
        assert body["confidence"] > 0

    def test_insufficient_context_is_a_localized_refusal(self, client, pipeline, corpus):
        pipeline.model.answer = cited_answer(
            "Les documents ne parlent pas des stagiaires.", status="insufficient_context"
        )
        question = "Quelle est la politique de télétravail pour les stagiaires ?"
        body = client.post("/ask", json={"question": question}).json()

        assert body["refusal"] is True
        assert body["status"] == "insufficient_context"
        assert body["grounded"] is False
        assert body["confidence"] == 0
        assert body["answer"].startswith("Je ne peux pas répondre")
        assert body["unsupported_notes"] == "Les documents ne parlent pas des stagiaires."
        # The model was still called and billed.
        assert body["metrics"]["llm_calls"] == 1

    def test_french_question_retrieves_the_french_invoice(self, client, pipeline, corpus):
        pipeline.model.answer = cited_answer("Le total TTC est de 9 577,37 €.", "S1")
        body = client.post(
            "/ask", json={"question": "Quel est le total TTC de la facture FA-2026-0311 ?"}
        ).json()
        assert body["sources"][0]["document_id"] == corpus[FR_NAME]["doc_id"]


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


@pytest.fixture
def keys(auth_db, settings, monkeypatch):
    from app import auth

    monkeypatch.setattr(settings, "require_api_key", True)
    return {
        group: {
            "Authorization": "Bearer "
            + auth.create_api_key(f"{group}-bot", requests_per_minute=100, groups=[group])
        }
        for group in ("legal", "hr")
    }


class TestAccessControl:
    ASK = {"question": "What does the zanzibar litigation hold freeze?"}

    def _retrieved_docs(self, body: dict) -> set[str]:
        return {c["doc_id"] for c in body["retrieval"]["chunks"]}

    def test_group_document_is_invisible_without_the_group(
        self, client, pipeline, corpus, keys
    ):
        legal_doc = corpus[LEGAL_DOC[0]]["doc_id"]
        pipeline.model.answer = cited_answer("It freezes the Kestrel contracts.", "S1")

        member = client.post("/ask", json=self.ASK, headers=keys["legal"]).json()
        assert member["sources"][0]["document_id"] == legal_doc

        outsider = client.post("/ask", json=self.ASK, headers=keys["hr"]).json()
        assert legal_doc not in self._retrieved_docs(outsider)
        assert all(s["document_id"] != legal_doc for s in outsider["sources"])
        # Its text never reached the prompt the model saw.
        assert "Kestrel" not in pipeline.model.calls[-1]["messages"][-1]["content"]

    def test_group_document_is_not_counted_or_bm25_ranked_for_outsiders(
        self, client, pipeline, corpus, keys
    ):
        total = client.get("/ingest/stats", headers=keys["legal"]).json()["indexed_chunks"]
        visible = client.get("/ingest/stats", headers=keys["hr"]).json()["indexed_chunks"]
        assert total - visible == corpus[LEGAL_DOC[0]]["chunks"]

        from app.access import AccessScope

        tokens = bm25_tokenize("zanzibar")
        outsider = AccessScope(tenant="default", groups=frozenset({"public", "hr"}))
        assert _run(sparse.sparse_search(tokens, 5, outsider)) == []
        assert _run(sparse.sparse_search(tokens, 5))  # unrestricted internal search finds it


# ---------------------------------------------------------------------------
# Erasure
# ---------------------------------------------------------------------------


class TestErasure:
    def test_document_erasure_removes_vectors_bm25_entries_and_traces(
        self, client, pipeline, corpus, settings
    ):
        settings.admin_key = "admin-secret"
        doc_id = corpus["security.md"]["doc_id"]
        tokens = bm25_tokenize("passwords rotated multi-factor")

        pipeline.model.answer = cited_answer("Every 90 days.", "S1")
        asked = client.post("/ask", json={"question": "How often must passwords be rotated?"})
        assert asked.json()["sources"][0]["document_id"] == doc_id
        hits = _run(sparse.sparse_search(tokens, 10))
        assert doc_id in {h.payload["doc_id"] for h in hits}
        rebuilds = sparse.get_index().rebuilds

        r = client.delete(f"/privacy/documents/{doc_id}", headers={"X-Admin-Key": "admin-secret"})
        assert r.status_code == 200, r.text
        report = r.json()
        assert report["complete"] is True
        assert report["targets"]["vectors"] == corpus["security.md"]["chunks"]
        assert report["targets"]["traces"] >= 1

        assert _run(_payloads(doc_id)) == []
        hits = _run(sparse.sparse_search(tokens, 10))
        assert doc_id not in {h.payload["doc_id"] for h in hits}
        assert sparse.get_index().rebuilds == rebuilds + 1

        # Idempotent: a repeat finds nothing left to delete.
        again = client.delete(
            f"/privacy/documents/{doc_id}", headers={"X-Admin-Key": "admin-secret"}
        ).json()
        assert again["targets"]["vectors"] == 0


# ---------------------------------------------------------------------------
# Evaluation and the regression gate
# ---------------------------------------------------------------------------

GOLDEN = [
    {
        "id": "remote",
        "question": "How many days per week can employees work remotely?",
        "ideal_answer": "Up to three days per week.",
        "relevant_doc_ids": ["remote_work.md"],
    },
    {
        "id": "receipts",
        "question": "Within how many days must expense receipts be submitted?",
        "ideal_answer": "Within 30 days of purchase.",
        "relevant_doc_ids": ["expenses.md"],
    },
    {
        "id": "passwords",
        "question": "How often must passwords be rotated?",
        "ideal_answer": "Every 90 days.",
        "relevant_doc_ids": ["security.md"],
    },
    {
        "id": "facture",
        "question": "Quel est le total TTC de la facture FA-2026-0311 ?",
        "ideal_answer": "9 577,37 €.",
        "relevant_doc_ids": [FR_NAME],
    },
    {
        "id": "loyer",
        "question": "What is the Loyer amount in the Rapport annuel budget?",
        "ideal_answer": "1200.",
        "relevant_doc_ids": [DOCX_NAME],
    },
]
JUDGE_BREAKS_ON = "expense receipts"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location(
        "_integration_run_eval", REPO / "scripts" / "run_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEvaluation:
    @pytest.fixture
    def golden(self, tmp_path, monkeypatch, corpus):
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir()
        (golden_dir / "tiny.jsonl").write_text(
            "\n".join(json.dumps(e) for e in GOLDEN) + "\n", encoding="utf-8"
        )
        monkeypatch.setattr(eval_metrics, "GOLDEN_DIR", golden_dir)
        return "tiny"

    def test_run_accounting_retrieval_metrics_and_regression_gate(
        self, pipeline, golden, capsys
    ):
        pipeline.model.answer = cited_answer("An answer from the context.", "S1")
        pipeline.model.judge_breaks_on = (JUDGE_BREAKS_ON,)

        run = _run(eval_metrics.run_evaluation(dataset=golden, strategy="classic"))

        # Accounting: one judge failure, recorded and excluded, never scored.
        assert run["n_examples"] == 5
        assert (run["n_scored"], run["n_judge_failed"], run["n_generation_failed"]) == (4, 1, 0)
        rows = {r["id"]: r for r in run["per_example"]}
        assert set(rows) == {e["id"] for e in GOLDEN}
        failed = rows["receipts"]
        assert failed["status"] == "judge_failed"
        assert failed["score"] is None
        assert failed["judge"]["attempts"] == 2
        assert failed["error"]["stage"] == "judge"
        assert not any(k in failed["metrics"] for k in ("faithfulness", "answer_relevance"))
        assert run["aggregates"]["faithfulness"] == {
            "mean": pytest.approx(0.9), "std": pytest.approx(0.0), "n": 4
        }
        assert [f["id"] for f in run["failures"]] == ["receipts"]

        # Retrieval @K comes from the real hybrid retrieval, for every labelled row,
        # the judge-failed one included.
        assert run["n_retrieval_scored"] == 5
        assert run["retrieval_mode"] == "hybrid"
        for row in rows.values():
            assert row["retrieval_mode"] == "hybrid"
            assert row["retrieval"]["hit_rate@5"] == 1.0, row["id"]
            assert row["metrics"]["recall@5"] == 1.0
            assert row["retrieved_ids"]
        assert run["aggregates"]["mrr"]["n"] == 5
        assert run["cost"]["total_input_tokens"] == 5 * ScriptedAnthropic.GEN_USAGE[0]
        # 4 good verdicts in one call each, the broken one tried twice.
        assert run["cost"]["judge_retries"] == 1
        assert run["cost"]["judge_input_tokens"] == 6 * ScriptedAnthropic.JUDGE_USAGE[0]

        # Second run: the judge scores every answer lower.
        pipeline.model.judge_score = 0.5
        run_eval = _load_run_eval()
        code = run_eval.main([
            "--dataset", golden, "--fail-on-regression", "--min-examples", "2",
            # Wall-clock latency of an in-process run is noise; gate on quality.
            "--threshold", "latency_p50_ms=off",
        ])
        assert code == run_eval.EXIT_REGRESSION
        assert "Regression gate: FAIL" in capsys.readouterr().err

        reports = sorted(eval_metrics.RUNS_DIR.glob(f"*{regression.REPORT_SUFFIX}.json"))
        latest = json.loads(reports[-1].read_text(encoding="utf-8"))
        assert latest["status"] == regression.FAIL
        assert latest["baseline_source"] == "previous"
        assert latest["baseline"]["id"] == run["id"]
        checks = {c["metric"]: c for c in latest["checks"]}
        assert checks["faithfulness"]["status"] == regression.FAIL
        assert checks["faithfulness"]["delta"] == pytest.approx(-0.4)
        # Retrieval did not change, so its checks pass.
        assert checks["recall@5"]["status"] == regression.PASS
        assert checks["mrr"]["status"] == regression.PASS
        assert "latency_p50_ms" not in checks

    def test_unchanged_rerun_passes_the_gate(self, pipeline, golden):
        pipeline.model.answer = cited_answer("An answer from the context.", "S1")
        run_eval = _load_run_eval()
        args = ["--dataset", golden, "--fail-on-regression", "--min-examples", "2",
                "--threshold", "latency_p50_ms=off", "--json"]
        assert run_eval.main(args) == run_eval.EXIT_OK
        assert run_eval.main(args) == run_eval.EXIT_OK
