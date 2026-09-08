"""Integrity checks on the shipped golden datasets.

`golden_v1` was labelled against a six-document seed corpus. The corpus later
grew to 41 documents and the labels were never revisited, so retrieval that
surfaced a better document scored as a miss and the resulting numbers were
meaningless. These tests make that class of drift fail loudly.
"""
from pathlib import Path

import pytest

from app.eval.metrics import GOLDEN_DIR, load_dataset

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "docs"
DATASETS = [p.stem for p in sorted(GOLDEN_DIR.glob("*.jsonl"))]


@pytest.fixture(scope="module")
def corpus() -> set[str]:
    """Filenames present in the document corpus."""
    return {p.name for p in DOCS_DIR.glob("*.md")}


def test_at_least_one_dataset_ships():
    assert DATASETS, "no golden dataset found in data/golden/"


@pytest.mark.parametrize("name", DATASETS)
class TestDatasetIntegrity:
    def test_parses_and_is_not_empty(self, name):
        assert load_dataset(name), f"{name} has no examples"

    def test_every_example_has_a_question_and_ideal_answer(self, name):
        for i, ex in enumerate(load_dataset(name), 1):
            assert ex.get("question", "").strip(), f"{name} #{i}: blank question"
            assert ex.get("ideal_answer", "").strip(), f"{name} #{i}: blank ideal_answer"

    def test_questions_are_unique(self, name):
        questions = [ex["question"] for ex in load_dataset(name)]
        duplicates = {q for q in questions if questions.count(q) > 1}
        assert not duplicates, f"{name} repeats: {duplicates}"

    def test_expected_sources_is_a_list_of_strings(self, name):
        for i, ex in enumerate(load_dataset(name), 1):
            sources = ex.get("expected_sources", [])
            assert isinstance(sources, list), f"{name} #{i}: expected_sources not a list"
            assert all(isinstance(s, str) for s in sources), f"{name} #{i}: non-string"

    def test_every_expected_source_exists_in_the_corpus(self, name, corpus):
        """A label naming a file that does not exist can never be satisfied."""
        broken = [
            (ex["question"][:60], s)
            for ex in load_dataset(name)
            for s in ex.get("expected_sources", [])
            if s not in corpus
        ]
        assert not broken, f"{name} references missing documents: {broken}"

    def test_examples_without_labels_explain_themselves(self, name):
        """An unlabelled example is excluded from retrieval scoring silently.

        That is the right behaviour only when it is deliberate, so it has to say
        so -- otherwise a forgotten label looks identical to an intentional one.
        """
        unexplained = [
            ex["question"][:60]
            for ex in load_dataset(name)
            if not ex.get("expected_sources") and not ex.get("note")
        ]
        assert not unexplained, (
            f"{name}: examples with no expected_sources and no note explaining why: "
            f"{unexplained}"
        )


class TestGoldenV2Coverage:
    """v2 is the dataset corrected against the full corpus."""

    def test_v2_exists(self):
        assert "golden_v2" in DATASETS

    def test_v2_labels_far_more_of_the_corpus_than_v1(self, corpus):
        def labelled(name: str) -> set[str]:
            return {s for ex in load_dataset(name) for s in ex.get("expected_sources", [])}

        v1, v2 = labelled("golden_v1"), labelled("golden_v2")
        assert len(v2) > len(v1) * 3, (
            f"v2 labels {len(v2)} documents, v1 labels {len(v1)}; "
            "v2 is meant to span the corpus, not a seed subset"
        )

    def test_v2_allows_several_acceptable_documents_per_question(self):
        """Single-label ground truth on a corpus with near-duplicates under-reports."""
        examples = load_dataset("golden_v2")
        multi = [ex for ex in examples if len(ex.get("expected_sources", [])) > 1]
        assert len(multi) > len(examples) / 3

    def test_v1_is_left_untouched(self):
        """Rewriting a dataset in place invalidates every score recorded against it."""
        v1 = load_dataset("golden_v1")
        assert len(v1) == 34
        single = [ex for ex in v1 if len(ex["expected_sources"]) == 1]
        assert len(single) == 32, "golden_v1 should still be the original labelling"
