"""POST /compare/strategies with evaluate=true, and GET /eval/golden.

The judge model call and the strategies are mocked; retrieval metrics are the
real deterministic functions.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.eval import online
from app.eval.judge import JudgeOutcome
from app.rag.strategies.base import StrategyResult
from app.schemas import Source, StrategyComparison

GOLDEN = [
    {
        "id": "g-1",
        "question": "What does BM25 tune?",
        "ideal_answer": "k1 and b.",
        "relevant_doc_ids": ["bm25.md"],
        "question_type": "single_hop",
        "difficulty": "easy",
    },
    {
        "id": "g-2",
        "question": "What is the CEO's salary?",
        "expected_behavior": "refuse",
        "question_type": "unanswerable",
    },
    {"question": "No id here", "expected_sources": ["other.md"]},
]

VERDICT = {
    "faithfulness": 0.9,
    "answer_relevance": 0.8,
    "context_precision": 0.7,
    "context_recall": 0.6,
    "answer_correctness": 1.0,
}


def _result(docs: list[str], *, refusal: bool = False, status: str = "answered") -> StrategyResult:
    return StrategyResult(
        answer="Tune k1 and b [S1].",
        sources=[Source(chunk_id="bm25:0", quote="k1 and b", document="bm25.md", handle="S1")],
        refusal=refusal,
        status=status,  # type: ignore[arg-type]
        latency_ms=10,
        input_tokens=5,
        output_tokens=3,
        extra={
            "retrieved_ids": [f"{d.split('.')[0]}:0" for d in docs],
            "retrieved_docs": docs,
            "context_text": "[S1] k1 and b",
        },
    )


def _ok_outcome(*_: object, **__: object) -> JudgeOutcome:
    return JudgeOutcome(
        scores=dict(VERDICT),
        reasoning=dict.fromkeys(VERDICT, "because"),
        attempts=1,
        input_tokens=100,
        output_tokens=20,
        judge_model="claude-opus-5",
    )


@pytest.fixture
def golden():
    with patch("app.eval.online.load_dataset", side_effect=lambda n: GOLDEN if n == "gold" else []):
        yield


class TestGoldenLookup:
    def test_questions_omit_answers_and_labels(self, golden):
        qs = online.golden_questions("gold")
        assert [q["id"] for q in qs] == ["g-1", "g-2", "q003"]
        assert qs[0] == {
            "id": "g-1",
            "question": "What does BM25 tune?",
            "question_type": "single_hop",
            "difficulty": "easy",
        }
        assert qs[2]["question_type"] == "unspecified"
        assert "ideal_answer" not in json.dumps(qs)

    def test_missing_dataset_or_example(self, golden):
        with pytest.raises(online.GoldenNotFound):
            online.golden_questions("nope")
        with pytest.raises(online.GoldenNotFound):
            online.golden_example("gold", "g-404")

    def test_reference_from_example(self, golden):
        ref = online.resolve_reference(None, online.GoldenRef(dataset="gold", id="g-1"))
        assert ref.ideal_answer == "k1 and b."
        assert ref.relevant_docs == ["bm25.md"]
        assert not ref.expect_refusal
        assert online.resolve_reference(None, online.GoldenRef(dataset="gold", id="g-2")).expect_refusal


class TestRankedDocuments:
    def _row(self, extra: dict, sources: list[Source] | None = None) -> StrategyComparison:
        return StrategyComparison(
            strategy="classic",
            question="q",
            answer="a",
            sources=sources or [],
            confidence=0,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            iterations=1,
            extra=extra,
        )

    def test_prefers_retrieved_docs_and_dedupes(self):
        row = self._row({"retrieved_docs": ["a.md", "a.md", None, "b.md"], "retrieved_ids": ["a:0", "a:1", "c:0", "b:0"]})
        assert online.ranked_documents(row) == ["a.md", "c", "b.md"]

    def test_falls_back_to_chunk_ids_then_sources(self):
        assert online.ranked_documents(self._row({"retrieved_ids": ["x:1", "y:0"]})) == ["x", "y"]
        src = [Source(chunk_id="z:0", quote="", document="z.md")]
        assert online.ranked_documents(self._row({}, src)) == ["z.md"]


class TestEndpoint:
    def _post(self, client, body: dict, results: list, judge=None):
        charged = MagicMock()
        judge = judge or AsyncMock(side_effect=_ok_outcome)
        with (
            patch("app.api.compare.charge", new=charged),
            patch("app.api.compare.run_strategy_raw", new=AsyncMock(side_effect=results)),
            patch("app.eval.online.judge_detailed", new=judge),
        ):
            r = client.post("/compare/strategies", json=body)
        return r, charged, judge

    def test_without_evaluate_nothing_is_scored(self, client):
        r, charged, judge = self._post(
            client, {"question": "q", "strategies": ["classic"]}, [(_result(["bm25.md"]), None)]
        )
        assert r.status_code == 200, r.text
        assert r.json()["results"][0]["evaluation"] is None
        judge.assert_not_called()
        assert charged.call_args.args[1] == 1

    def test_scores_with_golden_reference(self, client, golden):
        r, charged, judge = self._post(
            client,
            {
                "question": "What does BM25 tune?",
                "strategies": ["classic", "agentic"],
                "evaluate": True,
                "golden": {"dataset": "gold", "id": "g-1"},
            },
            [(_result(["other.md", "bm25.md"]), None), (_result(["bm25.md"]), None)],
        )
        assert r.status_code == 200, r.text
        # 1 (classic) + 3 (agentic) + one judge unit per strategy.
        assert charged.call_args.args[1] == 6
        rows = r.json()["results"]
        first = rows[0]["evaluation"]
        assert first["status"] == "scored"
        assert first["scores"]["answer_correctness"] == 1.0
        assert first["has_reference_answer"] is True
        assert first["retrieval"]["mrr"] == 0.5
        assert first["retrieval"]["recall@5"] == 1.0
        assert first["retrieved_docs"] == ["other.md", "bm25.md"]
        assert first["judge_input_tokens"] == 100
        assert rows[1]["evaluation"]["retrieval"]["mrr"] == 1.0
        kwargs = judge.call_args.kwargs
        assert kwargs["ideal_answer"] == "k1 and b."
        assert judge.call_args.args[2] == ["[S1] k1 and b"]

    def test_judge_failure_yields_null_scores_with_error(self, client):
        failing = AsyncMock(
            return_value=JudgeOutcome.failed("JudgeOutputInvalid", "bad json", attempts=2)
        )
        r, _, _ = self._post(
            client,
            {
                "question": "q",
                "strategies": ["classic"],
                "evaluate": True,
                "reference": {"relevant_doc_ids": ["bm25.md"]},
            },
            [(_result(["bm25.md"]), None)],
            judge=failing,
        )
        ev = r.json()["results"][0]["evaluation"]
        assert ev["status"] == "judge_failed"
        assert ev["scores"] is None
        assert ev["error"] == "bad json"
        # Retrieval metrics need no model, so they are still measured.
        assert ev["retrieval"]["recall@1"] == 1.0
        assert failing.call_args.kwargs["ideal_answer"] is None

    def test_no_relevant_docs_means_no_retrieval_metrics(self, client):
        r, _, _ = self._post(
            client,
            {"question": "q", "strategies": ["classic"], "evaluate": True},
            [(_result(["bm25.md"]), None)],
        )
        ev = r.json()["results"][0]["evaluation"]
        assert ev["retrieval"] is None
        assert ev["relevant_docs"] is None
        assert ev["has_reference_answer"] is False

    def test_failed_strategy_is_not_judged(self, client):
        r, _, judge = self._post(
            client,
            {"question": "q", "strategies": ["graph"], "evaluate": True},
            [(None, "boom")],
        )
        ev = r.json()["results"][0]["evaluation"]
        assert ev["status"] == "not_evaluated"
        assert ev["scores"] is None
        judge.assert_not_called()

    def test_refusal_example_is_checked_without_judge(self, client, golden):
        r, charged, judge = self._post(
            client,
            {
                "question": "What is the CEO's salary?",
                "strategies": ["classic"],
                "evaluate": True,
                "golden": {"dataset": "gold", "id": "g-2"},
            },
            [(_result([], refusal=True, status="insufficient_context"), None)],
        )
        ev = r.json()["results"][0]["evaluation"]
        assert ev["status"] == "refusal_checked"
        assert ev["correct_refusal"] == 1.0
        judge.assert_not_called()
        assert charged.call_args.args[1] == 1

    def test_unknown_golden_example_is_404(self, client, golden):
        r, charged, _ = self._post(
            client,
            {
                "question": "q",
                "strategies": ["classic"],
                "evaluate": True,
                "golden": {"dataset": "gold", "id": "missing"},
            },
            [],
        )
        assert r.status_code == 404
        charged.assert_not_called()

    def test_reference_and_golden_are_exclusive(self, client):
        r = client.post(
            "/compare/strategies",
            json={
                "question": "q",
                "evaluate": True,
                "reference": {"ideal_answer": "x"},
                "golden": {"dataset": "gold", "id": "g-1"},
            },
        )
        assert r.status_code == 422


class TestGoldenRoutes:
    def test_lists_questions(self, client, golden):
        r = client.get("/eval/golden/gold")
        assert r.status_code == 200
        assert r.json()[1] == {
            "id": "g-2",
            "question": "What is the CEO's salary?",
            "question_type": "unanswerable",
            "difficulty": None,
        }

    def test_unknown_dataset_is_404(self, client, golden):
        assert client.get("/eval/golden/nope").status_code == 404

    def test_rejects_path_like_names(self, client):
        assert client.get("/eval/golden/.hidden").status_code == 422

    def test_lists_datasets(self, client, tmp_path):
        (tmp_path / "b.jsonl").write_text("{}\n")
        (tmp_path / "a.jsonl").write_text("{}\n")
        (tmp_path / "notes.md").write_text("x")
        with patch("app.api.eval_routes.GOLDEN_DIR", tmp_path):
            assert client.get("/eval/golden").json() == ["a", "b"]
