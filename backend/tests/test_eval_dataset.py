"""Golden dataset schema helpers and offline validation (the --dry-run path)."""
import json

import pytest

from app.eval.dataset import (
    GoldenExample,
    difficulty_of,
    example_id,
    expected_behavior_of,
    ideal_answer_of,
    question_type_of,
    validate_dataset_file,
)


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("rag_overview.md", "bm25.md"):
        (docs / name).write_text("x", encoding="utf-8")
    return docs


def _write(tmp_path, rows, name="d.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


class TestFieldHelpers:
    def test_reference_answer_aliases(self):
        assert ideal_answer_of({"ideal_answer": "a", "expected_answer": "b"}) == "a"
        assert ideal_answer_of({"expected_answer": "b"}) == "b"
        assert ideal_answer_of({"ideal_answer": "  "}) is None

    def test_ids_default_to_position(self):
        assert example_id({}, 0) == "q001"
        assert example_id({"id": "fr-7"}, 0) == "fr-7"
        assert example_id({"id": 3}, 0) == "3"

    def test_labels(self):
        assert question_type_of({}) == "unspecified"
        assert question_type_of({"question_type": "multi_hop"}) == "multi_hop"
        assert difficulty_of({"difficulty": 2}) == "2"
        assert difficulty_of({}) is None
        assert expected_behavior_of({"expected_behavior": "refuse"}) == "refuse"
        assert expected_behavior_of({}) == "answer"

    def test_schema_rejects_unknown_behaviour(self):
        with pytest.raises(ValueError):
            GoldenExample.model_validate({"question": "q", "expected_behavior": "shrug"})


class TestValidateDatasetFile:
    def test_a_clean_file(self, tmp_path, corpus):
        path = _write(
            tmp_path,
            [
                {
                    "id": "a",
                    "question": "What is RAG?",
                    "ideal_answer": "x",
                    "relevant_doc_ids": ["rag_overview"],
                    "question_type": "single_hop",
                    "difficulty": "easy",
                },
                {
                    "id": "b",
                    "question": "Which bank?",
                    "expected_behavior": "refuse",
                    "question_type": "unanswerable",
                },
            ],
        )
        report = validate_dataset_file(path, corpus)
        assert report.ok, report.errors
        assert report.n_examples == 2
        assert report.n_with_relevance == 1
        assert report.n_expect_refusal == 1
        assert report.question_types == {"single_hop": 1, "unanswerable": 1}
        assert report.difficulties == {"easy": 1}
        assert report.warnings == []  # refusals need neither labels nor a reference

    def test_missing_documents_are_errors(self, tmp_path, corpus):
        path = _write(tmp_path, [{"question": "q", "ideal_answer": "a", "expected_sources": ["gone.md"]}])
        report = validate_dataset_file(path, corpus)
        assert not report.ok
        assert "gone.md" in report.errors[0]

    def test_schema_and_json_errors_name_the_line(self, tmp_path, corpus):
        path = _write(tmp_path, ['{"question": ""}', "{oops", "[1]", '{"question": "q", "relevant_doc_ids": "a"}'])
        report = validate_dataset_file(path, corpus)
        text = "\n".join(report.errors)
        assert "line 1: question" in text
        assert "line 2: invalid JSON" in text
        assert "line 3: expected an object" in text
        assert "line 4: relevant_doc_ids" in text

    def test_duplicates(self, tmp_path, corpus):
        path = _write(tmp_path, [{"id": "x", "question": "q", "note": "n"}] * 2)
        report = validate_dataset_file(path, corpus)
        assert any("duplicate id 'x'" in e for e in report.errors)
        assert any("duplicate question" in w for w in report.warnings)

    def test_unlabelled_without_note_is_a_warning(self, tmp_path, corpus):
        path = _write(tmp_path, [{"question": "q", "ideal_answer": "a"}, {"question": "r", "ideal_answer": "a", "notes": "why"}])
        report = validate_dataset_file(path, corpus)
        assert report.ok
        assert len([w for w in report.warnings if "no note" in w]) == 1

    def test_no_corpus_check_when_dir_is_none(self, tmp_path):
        path = _write(tmp_path, [{"question": "q", "ideal_answer": "a", "expected_sources": ["gone.md"]}])
        assert validate_dataset_file(path, None).ok

    def test_empty_corpus_is_a_warning_not_a_pass(self, tmp_path):
        path = _write(tmp_path, [{"question": "q", "ideal_answer": "a", "expected_sources": ["a.md"]}])
        report = validate_dataset_file(path, tmp_path / "nowhere")
        assert any("missing or empty" in w for w in report.warnings)

    def test_empty_file(self, tmp_path, corpus):
        path = tmp_path / "e.jsonl"
        path.write_text("\n", encoding="utf-8")
        assert validate_dataset_file(path, corpus).errors == ["no examples"]

    def test_report_serializes(self, tmp_path, corpus):
        path = _write(tmp_path, [{"question": "q", "ideal_answer": "a", "note": "n"}])
        assert json.loads(json.dumps(validate_dataset_file(path, corpus).to_dict()))["ok"]
