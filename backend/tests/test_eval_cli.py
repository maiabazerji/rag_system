"""scripts/run_eval.py and scripts/regression_report.py: flags and exit codes.

The eval itself is replaced by a stub that saves a canned run, so these tests
exercise the command-line layer: dry-run validation, the regression gate and
its exit code, baseline pinning, and the offline report script.
"""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.eval import metrics, regression

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load("run_eval")
regression_report = _load("regression_report")


def _result(run_id, faith=0.9, n=10, **extra):
    result = {
        "id": run_id,
        "dataset": "d",
        "strategy": "classic",
        "provider": "anthropic",
        "model": "m",
        "prompt_version": None,
        "judge_model": "j",
        "rubric_version": "r",
        "config_hash": "h",
        "n": n,
        "n_scored": n,
        "judged": True,
        "aggregates": {
            "faithfulness": {"mean": faith, "std": 0.1, "n": n},
            "mrr": {"mean": 0.5, "std": 0.1, "n": n},
            "latency_ms": {"mean": 100.0, "std": 1.0, "n": n},
        },
        "cost": {"p50_latency_ms": 100.0, "mean_tokens_per_question": 50.0},
        "failures": [],
        "per_example": [],
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    result.update(extra)
    return result


def _save(result, directory=None):
    directory = directory or metrics.RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['id']}.json"
    path.write_text(json.dumps({k: v for k, v in result.items() if k != "id"}), encoding="utf-8")
    return path


@pytest.fixture
def fake_eval(monkeypatch):
    """Make run_evaluation save and return the next canned result."""
    monkeypatch.delenv("REGRESSION_THRESHOLDS_PATH", raising=False)
    queue: list[dict] = []

    async def fake(**kwargs):
        result = queue.pop(0)
        _save(result)
        return result

    mock = AsyncMock(side_effect=fake)
    monkeypatch.setattr(run_eval, "run_evaluation", mock)
    return queue, mock


class TestRunEvalGate:
    def test_first_run_has_no_baseline_and_passes(self, fake_eval, capsys):
        queue, mock = fake_eval
        queue.append(_result("20260101T000000000000Z-a"))
        assert run_eval.main(["--dataset", "d", "--fail-on-regression"]) == 0
        out = capsys.readouterr().out
        assert "Regression report: SKIPPED" in out
        assert mock.await_args.kwargs["k_values"] == [1, 3, 5, 10]

    def test_regression_exits_2_only_with_the_flag(self, fake_eval, capsys):
        queue, _ = fake_eval
        _save(_result("20260101T000000000000Z-a", faith=0.9))
        queue.append(_result("20260102T000000000000Z-b", faith=0.8))
        assert run_eval.main(["--dataset", "d", "--fail-on-regression"]) == 2
        assert "| faithfulness | 0.9000 | 0.8000 |" in capsys.readouterr().out

        queue.append(_result("20260103T000000000000Z-c", faith=0.7))
        assert run_eval.main(["--dataset", "d"]) == 0

    def test_report_is_saved_next_to_the_run_and_linked(self, fake_eval):
        queue, _ = fake_eval
        _save(_result("20260101T000000000000Z-a", faith=0.9))
        queue.append(_result("20260102T000000000000Z-b", faith=0.8))
        run_eval.main(["--dataset", "d", "--json"])

        report = json.loads((metrics.RUNS_DIR / "20260102T000000000000Z-b.regression.json").read_text())
        assert report["status"] == "FAIL"
        assert (metrics.RUNS_DIR / "20260102T000000000000Z-b.regression.md").is_file()
        saved = json.loads((metrics.RUNS_DIR / "20260102T000000000000Z-b.json").read_text())
        assert saved["regression"]["status"] == "FAIL"
        assert saved["regression"]["baseline_id"] == "20260101T000000000000Z-a"

    def test_cli_threshold_override_turns_a_fail_into_a_pass(self, fake_eval):
        queue, _ = fake_eval
        _save(_result("20260101T000000000000Z-a", faith=0.9))
        queue.append(_result("20260102T000000000000Z-b", faith=0.8))
        argv = ["--dataset", "d", "--fail-on-regression", "--threshold", "faithfulness=-0.2"]
        assert run_eval.main(argv) == 0

    def test_min_examples_flag_skips_small_runs(self, fake_eval):
        queue, _ = fake_eval
        _save(_result("20260101T000000000000Z-a", faith=0.9))
        queue.append(_result("20260102T000000000000Z-b", faith=0.1))
        assert run_eval.main(["--dataset", "d", "--fail-on-regression", "--min-examples", "50"]) == 0

    def test_thresholds_file_flag(self, fake_eval, tmp_path):
        queue, _ = fake_eval
        f = tmp_path / "t.toml"
        f.write_text("min_examples = 1\n[metrics.mrr]\nmin_delta = -0.01\n", encoding="utf-8")
        _save(_result("20260101T000000000000Z-a", faith=0.9))
        queue.append(_result("20260102T000000000000Z-b", faith=0.1))  # faithfulness unchecked
        assert run_eval.main(["--dataset", "d", "--fail-on-regression", "--thresholds", str(f)]) == 0

    def test_invalid_thresholds_exit_1_before_running(self, fake_eval, tmp_path):
        _, mock = fake_eval
        assert run_eval.main(["--thresholds", str(tmp_path / "missing.toml")]) == 1
        assert run_eval.main(["--threshold", "nonsense"]) == 1
        mock.assert_not_awaited()

    def test_explicit_baseline_file(self, fake_eval, tmp_path):
        queue, _ = fake_eval
        base = _save(_result("elsewhere", faith=0.99), tmp_path / "other")
        queue.append(_result("20260102T000000000000Z-b", faith=0.9))
        assert run_eval.main(["--dataset", "d", "--fail-on-regression", "--baseline", str(base)]) == 2

    def test_set_baseline_after_a_run_pins_it(self, fake_eval):
        queue, _ = fake_eval
        queue.append(_result("20260101T000000000000Z-a", faith=0.95))
        assert run_eval.main(["--dataset", "d", "--set-baseline"]) == 0

        queue.append(_result("20260102T000000000000Z-b", faith=0.99))
        queue.append(_result("20260103T000000000000Z-c", faith=0.9))
        run_eval.main(["--dataset", "d"])
        # Compared with the pinned run (0.95), not the previous one (0.99): -0.05 fails.
        assert run_eval.main(["--dataset", "d", "--fail-on-regression"]) == 2
        saved = json.loads((metrics.RUNS_DIR / "20260103T000000000000Z-c.json").read_text())
        assert saved["regression"]["baseline_source"] == "pinned"
        assert saved["regression"]["baseline_id"] == "20260101T000000000000Z-a"

    def test_set_baseline_with_a_file_pins_without_running(self, fake_eval):
        _, mock = fake_eval
        path = _save(_result("20260101T000000000000Z-a"))
        assert run_eval.main(["--set-baseline", str(path)]) == 0
        mock.assert_not_awaited()
        assert list((regression.RUNS_DIR / "baselines").glob("d__classic__*.json"))

    def test_set_baseline_rejects_a_non_run(self, fake_eval, tmp_path):
        bad = tmp_path / "x.json"
        bad.write_text("[]", encoding="utf-8")
        assert run_eval.main(["--set-baseline", str(bad)]) == 1

    def test_nothing_measured_exits_1(self, fake_eval):
        queue, _ = fake_eval
        queue.append(_result("20260101T000000000000Z-a", aggregates={}, aggregate=None))
        assert run_eval.main(["--dataset", "d"]) == 1

    def test_missing_dataset_exits_1(self, monkeypatch):
        monkeypatch.setattr(
            run_eval, "run_evaluation", AsyncMock(return_value={"error": "No examples found"})
        )
        assert run_eval.main(["--dataset", "nope"]) == 1

    def test_k_flag(self, fake_eval):
        queue, mock = fake_eval
        queue.append(_result("20260101T000000000000Z-a"))
        run_eval.main(["--dataset", "d", "--k", "20,1,5"])
        assert mock.await_args.kwargs["k_values"] == [1, 5, 20]
        with pytest.raises(SystemExit):
            run_eval.main(["--k", "0"])


class TestDryRun:
    def test_shipped_datasets_are_valid(self, capsys):
        assert run_eval.main(["--dry-run", "--all-datasets"]) == 0
        out = capsys.readouterr().out
        assert "golden_v1.jsonl: OK" in out
        assert "dataset files valid" in out

    def test_invalid_dataset_exits_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(metrics, "GOLDEN_DIR", tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("x", encoding="utf-8")
        (tmp_path / "bad.jsonl").write_text(
            '{"question": "q", "ideal_answer": "a", "relevant_doc_ids": ["gone"]}\n',
            encoding="utf-8",
        )
        argv = ["--dry-run", "--dataset", "bad", "--corpus-dir", str(docs)]
        assert run_eval.main(argv) == 1
        assert "ERROR   line 1: relevant documents not in corpus: gone" in capsys.readouterr().out

    def test_json_output_and_explicit_path(self, tmp_path, capsys):
        f = tmp_path / "x.jsonl"
        f.write_text('{"question": "q", "ideal_answer": "a", "note": "n"}\n', encoding="utf-8")
        assert run_eval.main(["--dry-run", "--dataset", str(f), "--json"]) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["datasets"][0]["n_examples"] == 1

    def test_never_calls_the_pipeline(self, monkeypatch):
        mock = AsyncMock()
        monkeypatch.setattr(run_eval, "run_evaluation", mock)
        run_eval.main(["--dry-run", "--all-datasets"])
        mock.assert_not_awaited()


class TestRegressionReportScript:
    @pytest.fixture(autouse=True)
    def _defaults(self, monkeypatch):
        monkeypatch.delenv("REGRESSION_THRESHOLDS_PATH", raising=False)

    def test_pass(self, tmp_path, capsys):
        a = _save(_result("a", faith=0.9), tmp_path)
        b = _save(_result("b", faith=0.9), tmp_path)
        assert regression_report.main([str(a), str(b), "--fail-on-regression"]) == 0
        assert "Regression report: PASS" in capsys.readouterr().out

    def test_fail_exit_code_needs_the_flag(self, tmp_path):
        a = _save(_result("a", faith=0.9), tmp_path)
        b = _save(_result("b", faith=0.5), tmp_path)
        assert regression_report.main([str(a), str(b)]) == 0
        assert regression_report.main([str(a), str(b), "--fail-on-regression"]) == 2

    def test_json_and_output_dir(self, tmp_path, capsys):
        a = _save(_result("a", faith=0.9), tmp_path)
        b = _save(_result("b", faith=0.5), tmp_path)
        out_dir = tmp_path / "reports"
        assert regression_report.main([str(a), str(b), "--json", "--output-dir", str(out_dir)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "FAIL"
        assert report["baseline_source"] == "explicit"
        assert (out_dir / "b.regression.md").is_file()

    def test_overrides(self, tmp_path):
        a = _save(_result("a", faith=0.9), tmp_path)
        b = _save(_result("b", faith=0.5), tmp_path)
        argv = [str(a), str(b), "--fail-on-regression", "--threshold", "faithfulness=off"]
        assert regression_report.main(argv) == 0

    def test_unreadable_input_exits_1(self, tmp_path):
        a = _save(_result("a"), tmp_path)
        assert regression_report.main([str(a), str(tmp_path / "missing.json")]) == 1
