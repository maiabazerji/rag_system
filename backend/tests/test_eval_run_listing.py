"""Listing saved eval runs, and the threshold-file errors the gate reports."""
from __future__ import annotations

import json

import pytest

from app.eval import regression


def _save(name: str, payload) -> None:
    regression.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (regression.RUNS_DIR / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summaries_are_newest_first_without_per_example_detail():
    _save("20260101T000000Z-a", {"dataset": "d", "n": 3, "per_example": [{"q": 1}],
                                 "regression": {"status": "PASS"}})
    _save("20260102T000000Z-b", {"dataset": "d", "n": 2, "n_scored": 1})
    _save("20260102T000000Z-b.regression", {"status": "FAIL"})  # a report, not a run
    _save("20260103T000000Z-c", ["not", "a", "run"])
    (regression.RUNS_DIR / "20260104T000000Z-d.json").write_text("{broken", encoding="utf-8")

    summaries = regression.list_run_summaries()

    assert [s["id"] for s in summaries] == ["20260102T000000Z-b", "20260101T000000Z-a"]
    newest, oldest = summaries
    assert "per_example" not in oldest
    assert oldest["regression_status"] == "PASS"
    assert (oldest["n"], oldest["n_scored"]) == (3, 3)  # older runs: every example scored
    assert (newest["n_scored"], newest["regression_status"]) == (1, None)


def test_no_runs_directory_lists_nothing():
    assert regression.list_run_summaries() == []


@pytest.mark.parametrize(
    "toml, error",
    [
        ("min_examples = 2\n", "missing or empty"),
        ("[metrics]\nfaithfulness = 0.1\n", "must be a table"),
        ("[metrics.faithfulness]\nmin_delta = -0.1\nfloor = 1\n", "unknown keys"),
        ("[metrics.faithfulness]\nmin_delta = 'x'\n", "expected a number"),
        ("min_examples = 0\n[metrics.mrr]\nmin_delta = -0.1\n", "positive integer"),
        ("[metrics\n", "invalid TOML"),
    ],
)
def test_malformed_threshold_files_are_rejected(tmp_path, toml, error):
    path = tmp_path / "thresholds.toml"
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        regression.load_thresholds(path)


def test_min_examples_override_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        regression.load_thresholds(min_examples=0)
