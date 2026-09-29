"""Run accounting: statuses, failures, aggregates and breakdowns of an eval run.

The pipeline and the judge are mocked per question, so each test can state
exactly which examples succeed, which fail and where.
"""
import json
import statistics
from unittest.mock import AsyncMock, patch

import pytest

from app.eval import metrics
from app.eval.judge import JudgeOutcome
from app.eval.metrics import (
    aggregate_rows,
    breakdown,
    config_hash,
    percentile,
    retrieval_config,
    run_evaluation,
    summarize,
)
from app.rag.strategies.base import StrategyResult
from app.schemas import Answer, Source

DATASET = [
    {"id": "a", "question": "qa", "ideal_answer": "A", "relevant_doc_ids": ["a"],
     "question_type": "single_hop", "difficulty": "easy"},
    {"id": "b", "question": "qb", "expected_sources": ["b.md"],
     "question_type": "single_hop", "difficulty": "hard"},
    {"id": "c", "question": "qc", "relevant_doc_ids": ["c"], "question_type": "multi_hop"},
    {"id": "d", "question": "qd", "note": "open question", "question_type": "multi_hop"},
    {"id": "e", "question": "qe", "expected_behavior": "refuse", "question_type": "unanswerable"},
    {"id": "f", "question": "qf", "expected_behavior": "refuse", "question_type": "unanswerable"},
]

#: Ranked documents each question's strategy retrieves.
RETRIEVED = {
    "qa": ["a.md", "x.md"],  # relevant at rank 1
    "qb": ["x.md", "b.md"],  # relevant at rank 2
    "qc": ["x.md", "y.md"],  # judge will fail on this one
    "qd": ["x.md"],
    "qe": ["x.md"],
    "qf": ["x.md"],
}


def _pipeline(question, **kwargs):
    if question == "qd":
        raise RuntimeError("vector store exploded")
    refused = question == "qe"  # qf answers an unanswerable question
    ans = Answer(
        question=question,
        answer="I cannot answer that." if refused else f"answer to {question}",
        sources=[Source(chunk_id="d1:0", quote="q", document=RETRIEVED[question][0])],
        confidence=0.8,
        refusal=refused,
        provider="anthropic",
        model="gen-model",
        latency_ms={"qa": 100, "qb": 300, "qc": 200}.get(question, 50),
        input_tokens=90,
        output_tokens=10,
    )
    result = StrategyResult(
        answer=ans.answer,
        sources=ans.sources,
        refusal=refused,
        extra={
            "retrieved_ids": [f"{d}:0" for d in RETRIEVED[question]],
            "retrieved_docs": RETRIEVED[question],
            "context_text": f"context for {question}",
        },
    )
    return ans, result


def _judge(expected, got, context_text=None):
    q = expected["question"]
    if q == "qc":
        return JudgeOutcome.failed(
            "JudgeOutputInvalid", "faithfulness.score: too big", attempts=2,
            input_tokens=20, output_tokens=4,
        )
    score = {"qa": 1.0, "qb": 0.5}[q]
    return JudgeOutcome(
        scores={
            "faithfulness": score,
            "answer_relevance": score,
            "context_precision": score,
            "context_recall": score,
            "answer_correctness": 0.75 if expected.get("ideal_answer") else None,
        },
        reasoning={"faithfulness": f"because {q}"},
        attempts=1,
        input_tokens=10,
        output_tokens=2,
        judge_model="judge-model",
    )


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "GOLDEN_DIR", tmp_path)
    (tmp_path / "acc.jsonl").write_text(
        "\n".join(json.dumps(r) for r in DATASET) + "\n", encoding="utf-8"
    )

    async def go(**kwargs):
        with (
            patch("app.eval.metrics.answer_question_detailed",
                  new=AsyncMock(side_effect=_pipeline)),
            patch("app.eval.metrics.score_example_detailed", new=AsyncMock(side_effect=_judge)),
        ):
            return await run_evaluation(dataset="acc", **kwargs)

    return go


def _row(result, rid):
    return next(r for r in result["per_example"] if r["id"] == rid)


@pytest.mark.asyncio
class TestAccounting:
    async def test_counts(self, run):
        r = await run()
        assert r["n_examples"] == r["n"] == 6
        assert r["n_scored"] == 4  # a, b judged; e, f refusal-checked
        assert r["n_judge_failed"] == 1  # c
        assert r["n_generation_failed"] == 1  # d
        assert r["n_unscored"] == 2
        assert r["n_retrieval_scored"] == 3  # a, b, c (d failed; e, f are refusals)
        assert [_row(r, x)["status"] for x in "abcdef"] == [
            "scored", "scored", "judge_failed", "generation_failed", "scored", "scored",
        ]

    async def test_failures_list_names_stage_and_error(self, run):
        r = await run()
        by_id = {f["id"]: f for f in r["failures"]}
        assert set(by_id) == {"c", "d"}
        assert by_id["c"]["stage"] == "judge"
        assert by_id["c"]["error_type"] == "JudgeOutputInvalid"
        assert by_id["d"]["stage"] == "generation"
        assert by_id["d"]["error_type"] == "RuntimeError"
        assert by_id["d"]["error"] == "vector store exploded"

    async def test_judge_aggregates_exclude_unscored_examples(self, run):
        r = await run()
        faith = r["aggregates"]["faithfulness"]
        assert faith["n"] == 2  # not 3 (c failed) and never a fabricated value
        assert faith["mean"] == pytest.approx(0.75)
        assert faith["std"] == pytest.approx(statistics.stdev([1.0, 0.5]))
        assert r["aggregates"]["answer_correctness"] == {"mean": 0.75, "std": None, "n": 1}
        assert r["aggregate"]["faithfulness"] == pytest.approx(0.75)  # legacy block agrees

    async def test_retrieval_aggregates(self, run):
        r = await run()
        agg = r["aggregates"]
        assert agg["mrr"]["n"] == 3
        assert agg["mrr"]["mean"] == pytest.approx((1 + 0.5 + 0) / 3)
        assert agg["recall@1"]["mean"] == pytest.approx(1 / 3)
        assert agg["recall@3"]["mean"] == pytest.approx(2 / 3)
        assert "ndcg@10" in agg

    async def test_refusal_behaviour(self, run):
        r = await run()
        assert _row(r, "e")["metrics"]["correct_refusal"] == 1.0
        assert _row(r, "f")["metrics"]["correct_refusal"] == 0.0  # answered: hallucination
        assert r["aggregates"]["correct_refusal"] == {
            "mean": 0.5, "std": pytest.approx(statistics.stdev([1.0, 0.0])), "n": 2,
        }
        assert r["aggregates"]["false_refusal"]["mean"] == 0.0
        assert _row(r, "e")["retrieval"] is None
        assert _row(r, "e")["judge"] is None

    async def test_per_question_rows(self, run):
        r = await run()
        a = _row(r, "a")
        assert a["question_type"] == "single_hop"
        assert a["difficulty"] == "easy"
        assert a["strategy"] == "classic"
        assert a["latency_ms"] == 100
        assert a["total_tokens"] == 100
        assert a["judge"]["reasoning"] == {"faithfulness": "because qa"}
        assert a["judge"]["input_tokens"] == 10
        assert a["metrics"]["faithfulness"] == 1.0
        assert a["metrics"]["recall@1"] == 1.0
        assert a["retrieval"]["retrieved"] == ["a.md", "x.md"]
        assert _row(r, "c")["judge"]["attempts"] == 2
        assert _row(r, "d")["metrics"] == {}

    async def test_per_question_type_breakdown(self, run):
        r = await run()
        by_type = r["by_question_type"]
        assert set(by_type) == {"single_hop", "multi_hop", "unanswerable"}
        single = by_type["single_hop"]
        assert single["n_examples"] == 2
        assert single["n_scored"] == 2
        assert single["aggregates"]["faithfulness"]["mean"] == pytest.approx(0.75)
        multi = by_type["multi_hop"]
        assert multi["n_judge_failed"] == 1
        assert multi["n_generation_failed"] == 1
        assert "faithfulness" not in multi["aggregates"]  # nothing scored there
        assert multi["aggregates"]["mrr"] == {"mean": 0.0, "std": None, "n": 1}
        assert set(r["by_difficulty"]) == {"easy", "hard", "unspecified"}

    async def test_cost_block(self, run):
        r = await run()
        cost = r["cost"]
        # Generated rows: a 100, b 300, c 200, e 50, f 50 (d failed)
        assert cost["p50_latency_ms"] == 100
        assert cost["mean_tokens_per_question"] == 100
        assert cost["judge_input_tokens"] == 10 + 10 + 20
        assert cost["judge_output_tokens"] == 2 + 2 + 4
        assert cost["judge_retries"] == 1
        assert r["aggregates"]["latency_ms"]["n"] == 5

    async def test_configuration_is_recorded(self, run):
        r = await run(k_values=[2, 1])
        assert r["k_values"] == [1, 2]
        assert "recall@2" in r["aggregates"] and "recall@5" not in r["aggregates"]
        assert r["judge_model"]
        assert r["rubric_version"] == "2026-09.v2"
        assert r["rubric_fingerprint"]
        assert r["retrieval_mode"] == "dense"
        assert r["config_hash"] == config_hash(r["retrieval_config"])
        assert r["schema_version"] == 2

    async def test_no_judge_run(self, run):
        r = await run(judge_answers=False)
        assert r["n_not_judged"] == 3
        assert r["n_judge_failed"] == 0
        assert r["judge_model"] is None and r["rubric_version"] is None
        assert "faithfulness" not in r["aggregates"]
        assert r["aggregates"]["mrr"]["n"] == 3

    async def test_saved_file_matches_and_has_a_regression_report(self, run, tmp_path):
        r = await run()
        saved = json.loads((metrics.RUNS_DIR / f"{r['id']}.json").read_text())
        assert saved["n_judge_failed"] == 1
        assert saved["regression"]["status"] == "SKIPPED"  # first run: no baseline
        assert (metrics.RUNS_DIR / saved["regression"]["report_markdown"]).is_file()

    async def test_second_run_is_compared_with_the_first(self, run):
        first = await run()
        second = await run()
        assert second["regression"]["baseline_id"] == first["id"]
        assert second["regression"]["baseline_source"] == "previous"


class TestStatsHelpers:
    def test_summarize(self):
        assert summarize([]) == {"mean": None, "std": None, "n": 0}
        assert summarize([2.0]) == {"mean": 2.0, "std": None, "n": 1}
        s = summarize([1.0, 2.0, 3.0])
        assert s["mean"] == 2.0 and s["std"] == 1.0 and s["n"] == 3

    def test_percentile(self):
        assert percentile([], 50) is None
        assert percentile([5], 50) == 5
        assert percentile([1, 2, 3, 4], 50) == 2.5
        assert percentile([1, 2, 3, 4, 100], 95) == pytest.approx(80.8)

    def test_aggregate_rows_omits_unmeasured_metrics(self):
        rows = [{"metrics": {"faithfulness": 1.0}}, {"metrics": {}}]
        agg = aggregate_rows(rows)
        assert agg == {"faithfulness": {"mean": 1.0, "std": None, "n": 1}}

    def test_breakdown_groups_none_as_unspecified(self):
        rows = [{"difficulty": None, "status": "scored", "metrics": {}}]
        assert set(breakdown(rows, "difficulty")) == {"unspecified"}


class TestRetrievalConfig:
    def test_hash_changes_with_retrieval_settings(self, monkeypatch):
        before = config_hash(retrieval_config())
        monkeypatch.setattr(metrics.settings, "rerank_top_k", 3)
        assert config_hash(retrieval_config()) != before

    def test_hash_changes_with_mode(self):
        assert config_hash(retrieval_config("dense")) != config_hash(retrieval_config("hybrid"))

    def test_covers_models_and_chunking(self):
        cfg = retrieval_config()
        for key in ("embedding_model", "reranker_model", "chunk_size_tokens", "retrieval_top_k"):
            assert key in cfg
        assert not any("key" in k or "secret" in k for k in cfg)


def test_erasure_scrubs_judge_reasoning(tmp_path, monkeypatch):
    """Reasoning can quote the erased context, so it goes with the answer."""
    from app.privacy import erasure

    monkeypatch.setattr(metrics, "RUNS_DIR", tmp_path)
    row = {
        "answer": "secret",
        "doc_ids": ["docA"],
        "score": {"faithfulness": 1.0},
        "judge": {"reasoning": {"faithfulness": "quotes the secret"}, "attempts": 1},
    }
    (tmp_path / "r.json").write_text(json.dumps({"per_example": [row]}), encoding="utf-8")

    assert erasure.scrub_eval_runs({"docA"}) == 1
    saved = json.loads((tmp_path / "r.json").read_text())["per_example"][0]
    assert saved["judge"]["reasoning"] == {"faithfulness": erasure.ERASED}
    assert saved["score"] == {"faithfulness": 1.0}
