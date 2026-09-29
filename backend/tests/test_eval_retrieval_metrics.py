"""Ranked retrieval metrics against hand-computed values.

Every expected number below is worked out in the comment next to it, from the
formulas in the ``app.eval.retrieval`` module docstring.
"""
import math

import pytest

from app.eval.retrieval import (
    DEFAULT_K_VALUES,
    aggregate_retrieval,
    doc_key,
    ndcg_at_k,
    precision_at_k,
    ranked_metric_names,
    ranked_retrieval_metrics,
    recall_at_k,
    reciprocal_rank,
    relevant_documents,
    score_example_retrieval,
)

LOG2_3 = math.log2(3)
LOG2_5 = math.log2(5)


class TestWorkedExample:
    """Relevant {a, b}; retrieved [x, a, y, b, z] -> rel = [0, 1, 0, 1, 0]."""

    @pytest.fixture
    def m(self):
        return ranked_retrieval_metrics(["a.md", "b.md"], ["x.md", "a.md", "y.md", "b.md", "z.md"])

    def test_mrr_is_one_over_the_first_relevant_rank(self, m):
        assert m["mrr"] == 0.5  # first relevant at rank 2

    def test_precision_uses_k_as_the_denominator(self, m):
        assert m["precision@1"] == 0.0
        assert m["precision@3"] == pytest.approx(1 / 3)
        assert m["precision@5"] == pytest.approx(2 / 5)
        assert m["precision@10"] == pytest.approx(2 / 10)  # only 5 retrieved, still /10

    def test_recall(self, m):
        assert m["recall@1"] == 0.0
        assert m["recall@3"] == 0.5
        assert m["recall@5"] == 1.0
        assert m["recall@10"] == 1.0

    def test_hit_rate(self, m):
        assert m["hit_rate@1"] == 0.0
        assert m["hit_rate@3"] == 1.0

    def test_ndcg(self, m):
        # DCG@3 = 1/log2(3); IDCG@3 = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
        assert m["ndcg@3"] == pytest.approx((1 / LOG2_3) / (1 + 1 / LOG2_3))
        assert m["ndcg@3"] == pytest.approx(0.386853, abs=1e-6)
        # DCG@5 = 1/log2(3) + 1/log2(5); IDCG@5 = IDCG@3 (only two relevant)
        assert m["ndcg@5"] == pytest.approx((1 / LOG2_3 + 1 / LOG2_5) / (1 + 1 / LOG2_3))
        assert m["ndcg@5"] == pytest.approx(0.650920, abs=1e-6)
        assert m["ndcg@1"] == 0.0

    def test_reports_every_default_cutoff(self, m):
        assert set(m) == set(ranked_metric_names(DEFAULT_K_VALUES))
        assert DEFAULT_K_VALUES == (1, 3, 5, 10)


class TestNdcgEdgeCases:
    def test_perfect_ranking_is_one_at_every_k(self):
        m = ranked_retrieval_metrics(["a", "b"], ["a", "b", "c"])
        assert all(m[f"ndcg@{k}"] == pytest.approx(1.0) for k in DEFAULT_K_VALUES)

    def test_more_relevant_than_k_caps_the_ideal(self):
        # IDCG@1 = 1 (one slot), DCG@1 = 1 -> 1.0 even though b, c were never found
        m = ranked_retrieval_metrics(["a", "b", "c"], ["a"], k_values=[1, 3])
        assert m["ndcg@1"] == 1.0
        assert m["recall@1"] == pytest.approx(1 / 3)
        # IDCG@3 = 1 + 1/log2(3) + 1/2; DCG@3 = 1
        assert m["ndcg@3"] == pytest.approx(1 / (1 + 1 / LOG2_3 + 0.5))

    def test_single_relevant_at_rank_two(self):
        # DCG = 1/log2(3), IDCG = 1
        assert ndcg_at_k([0, 1], 5, 1) == pytest.approx(1 / LOG2_3)

    def test_nothing_retrieved_scores_zero_not_none(self):
        m = ranked_retrieval_metrics(["a"], [])
        assert m["ndcg@5"] == 0.0
        assert m["mrr"] == 0.0
        assert m["recall@10"] == 0.0

    def test_no_relevant_documents_is_undefined(self):
        """Excluded, not zero: there is nothing a retriever could have found."""
        assert ranked_retrieval_metrics([], ["a"]) is None
        assert ranked_retrieval_metrics(None, ["a"]) is None
        assert ranked_retrieval_metrics(["  "], ["a"]) is None

    def test_zero_ideal_is_guarded(self):
        assert ndcg_at_k([0, 0], 3, 0) == 0.0


class TestRanking:
    def test_duplicates_collapse_to_the_best_position(self):
        # [a, a, b] -> [a, b]; b is at rank 2, not 3
        m = ranked_retrieval_metrics(["b"], ["a", "a", "b"])
        assert m["mrr"] == 0.5
        assert m["precision@3"] == pytest.approx(1 / 3)

    def test_matching_ignores_path_case_and_extension(self):
        m = ranked_retrieval_metrics(["rag_overview"], ["docs/RAG_Overview.md"])
        assert m["hit_rate@1"] == 1.0

    def test_doc_key(self):
        assert doc_key("docs/A.md") == "a"
        assert doc_key("a") == "a"
        assert doc_key("v1.2") == "v1.2"  # not a document extension

    @pytest.mark.parametrize("ks", [[0], [-1, 3], []])
    def test_rejects_bad_cutoffs(self, ks):
        with pytest.raises(ValueError):
            ranked_retrieval_metrics(["a"], ["a"], k_values=ks)

    def test_custom_cutoffs(self):
        m = ranked_retrieval_metrics(["a"], ["x", "a"], k_values=[2])
        assert set(m) == {"mrr", "recall@2", "precision@2", "hit_rate@2", "ndcg@2"}


class TestPrimitives:
    def test_precision(self):
        assert precision_at_k([1, 0, 1], 2) == 0.5

    def test_recall(self):
        assert recall_at_k([1, 0, 1], 3, 4) == 0.5

    def test_reciprocal_rank(self):
        assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
        assert reciprocal_rank([]) == 0.0


class TestRelevanceLabels:
    def test_new_field_wins_over_the_old_one(self):
        ex = {"relevant_doc_ids": ["new"], "expected_sources": ["old.md"]}
        assert relevant_documents(ex) == ["new"]

    def test_falls_back_to_expected_sources(self):
        assert relevant_documents({"expected_sources": ["a.md"]}) == ["a.md"]
        assert relevant_documents({"relevant_doc_ids": [], "expected_sources": ["a.md"]}) == [
            "a.md"
        ]

    def test_none_when_unlabelled(self):
        assert relevant_documents({"question": "q"}) is None


class TestExampleRecord:
    def test_combines_unranked_and_ranked_metrics(self):
        rec = score_example_retrieval({"relevant_doc_ids": ["a"]}, ["x.md", "a.md"])
        assert rec["hit"] is True  # legacy RetrievalScore fields survive
        assert rec["precision"] == 0.5
        assert rec["recall@1"] == 0.0
        assert rec["recall@3"] == 1.0

    def test_none_when_unlabelled(self):
        assert score_example_retrieval({"question": "q"}, ["a.md"]) is None

    def test_aggregate_averages_at_k_metrics(self):
        hit = score_example_retrieval({"expected_sources": ["a.md"]}, ["a.md"])
        miss = score_example_retrieval({"expected_sources": ["a.md"]}, ["b.md"])
        agg = aggregate_retrieval([hit, miss])
        assert agg["n"] == 2
        assert agg["recall@5"] == 0.5
        assert agg["ndcg@1"] == 0.5
