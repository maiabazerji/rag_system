"""The regression gate: thresholds, baselines, comparison and reports."""
import json

import pytest

from app.eval import regression
from app.eval.regression import (
    DEFAULT_THRESHOLDS,
    DEFAULT_THRESHOLDS_PATH,
    FAIL,
    PASS,
    SKIPPED,
    Threshold,
    ThresholdConfig,
    check_metric,
    compare_runs,
    find_baseline,
    load_thresholds,
    metric_value,
    parse_threshold_override,
    pin_baseline,
    render_markdown,
    save_report,
)


def _run(
    *,
    faith=0.9,
    relevance=0.8,
    recall5=0.8,
    mrr=0.7,
    p50=1000.0,
    tokens=1000.0,
    n=10,
    run_id="20260101T000000Z-a",
    **extra,
):
    agg = {
        "faithfulness": {"mean": faith, "std": 0.1, "n": n},
        "answer_relevance": {"mean": relevance, "std": 0.1, "n": n},
        "recall@5": {"mean": recall5, "std": 0.1, "n": n},
        "mrr": {"mean": mrr, "std": 0.1, "n": n},
        "latency_ms": {"mean": p50, "std": 1.0, "n": n},
    }
    run = {
        "id": run_id,
        "dataset": "golden_v3",
        "strategy": "classic",
        "provider": "anthropic",
        "model": "m",
        "prompt_version": None,
        "judge_model": "j",
        "rubric_version": "r1",
        "config_hash": "h1",
        "n": n,
        "aggregates": agg,
        "cost": {"p50_latency_ms": p50, "mean_tokens_per_question": tokens},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    run.update(extra)
    return run


CONFIG = ThresholdConfig(thresholds=dict(DEFAULT_THRESHOLDS), min_examples=5, source="test")


def _status(report, metric):
    return next(c for c in report["checks"] if c["metric"] == metric)["status"]


class TestThresholdFile:
    def test_shipped_file_matches_the_built_in_defaults(self, monkeypatch):
        monkeypatch.delenv("REGRESSION_THRESHOLDS_PATH", raising=False)
        assert DEFAULT_THRESHOLDS_PATH.is_file()
        config = load_thresholds()
        assert config.thresholds == DEFAULT_THRESHOLDS
        assert config.min_examples == 5
        assert config.source == str(DEFAULT_THRESHOLDS_PATH)

    def test_defaults_are_the_documented_ones(self):
        t = DEFAULT_THRESHOLDS
        assert t["faithfulness"].min_delta == -0.03
        assert t["answer_relevance"].min_delta == -0.03
        assert t["recall@5"].min_delta == -0.03
        assert t["mrr"].min_delta == -0.03
        assert t["latency_p50_ms"].max_increase_pct == 20.0
        assert t["tokens_per_question"].max_increase_pct == 15.0

    def test_env_var_selects_another_file(self, tmp_path, monkeypatch):
        f = tmp_path / "t.toml"
        f.write_text("min_examples = 2\n[metrics.ndcg@10]\nmin_delta = -0.1\n".replace(
            "ndcg@10", '"ndcg@10"'), encoding="utf-8")
        monkeypatch.setenv("REGRESSION_THRESHOLDS_PATH", str(f))
        config = load_thresholds()
        assert set(config.thresholds) == {"ndcg@10"}
        assert config.min_examples == 2

    def test_explicit_path_beats_env_var(self, tmp_path, monkeypatch):
        a, b = tmp_path / "a.toml", tmp_path / "b.toml"
        a.write_text("[metrics.mrr]\nmin_delta = -0.5\n", encoding="utf-8")
        b.write_text("[metrics.mrr]\nmin_delta = -0.9\n", encoding="utf-8")
        monkeypatch.setenv("REGRESSION_THRESHOLDS_PATH", str(b))
        assert load_thresholds(a).thresholds["mrr"].min_delta == -0.5

    def test_named_file_must_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REGRESSION_THRESHOLDS_PATH", str(tmp_path / "missing.toml"))
        with pytest.raises(FileNotFoundError):
            load_thresholds()

    def test_missing_default_file_falls_back_to_built_ins(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REGRESSION_THRESHOLDS_PATH", raising=False)
        monkeypatch.setattr(regression, "DEFAULT_THRESHOLDS_PATH", tmp_path / "none.toml")
        config = load_thresholds()
        assert config.thresholds == DEFAULT_THRESHOLDS
        assert config.source == "built-in defaults"

    @pytest.mark.parametrize(
        "body, fragment",
        [
            ("min_examples = 3\n", "metrics"),
            ("[metrics.mrr]\nminimum = -0.1\n", "unknown keys"),
            ('[metrics.mrr]\nmin_delta = "a lot"\n', "expected a number"),
            ("[metrics.mrr]\n", "no bound"),
            ("min_examples = 0\n[metrics.mrr]\nmin_delta = -0.1\n", "min_examples"),
            ("[metrics\n", "invalid TOML"),
        ],
    )
    def test_malformed_files_are_rejected(self, tmp_path, body, fragment):
        f = tmp_path / "t.toml"
        f.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=fragment):
            load_thresholds(f)


class TestOverrides:
    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("faithfulness=-0.05", Threshold("faithfulness", min_delta=-0.05)),
            ("x=+0.1", Threshold("x", max_delta=0.1)),
            ("latency_p50_ms=+25%", Threshold("latency_p50_ms", max_increase_pct=25.0)),
            ("recall@5=-10%", Threshold("recall@5", max_decrease_pct=10.0)),
            ("mrr=off", None),
        ],
    )
    def test_parse(self, spec, expected):
        assert parse_threshold_override(spec)[1] == expected

    @pytest.mark.parametrize("spec", ["mrr", "=0.1", "mrr=", "mrr=lots", "mrr=1x%"])
    def test_parse_rejects(self, spec):
        with pytest.raises(ValueError):
            parse_threshold_override(spec)

    def test_overrides_apply_on_top_of_the_file(self, monkeypatch):
        monkeypatch.delenv("REGRESSION_THRESHOLDS_PATH", raising=False)
        config = load_thresholds(
            overrides=["faithfulness=-0.1", "mrr=off", "ndcg@10=-0.02"], min_examples=20
        )
        assert config.thresholds["faithfulness"].min_delta == -0.1
        assert "mrr" not in config.thresholds
        assert config.thresholds["ndcg@10"].min_delta == -0.02
        assert config.thresholds["answer_relevance"].min_delta == -0.03  # untouched
        assert config.min_examples == 20
        assert config.source.endswith("+ CLI overrides")


class TestCheckMetric:
    def test_absolute_drop_within_threshold_passes(self):
        row = check_metric(Threshold("f", min_delta=-0.03), (0.90, 10), (0.88, 10), 5)
        assert row["status"] == PASS
        assert row["delta"] == pytest.approx(-0.02)

    def test_drop_exactly_at_the_threshold_passes(self):
        row = check_metric(Threshold("f", min_delta=-0.03), (0.90, 10), (0.87, 10), 5)
        assert row["status"] == PASS

    def test_absolute_drop_beyond_threshold_fails(self):
        row = check_metric(Threshold("f", min_delta=-0.03), (0.90, 10), (0.85, 10), 5)
        assert row["status"] == FAIL
        assert "< -0.03" in row["reason"]

    def test_improvement_passes(self):
        assert check_metric(Threshold("f", min_delta=-0.03), (0.5, 10), (0.9, 10), 5)["status"] == PASS

    def test_percentage_increase(self):
        t = Threshold("latency_p50_ms", max_increase_pct=20.0)
        ok = check_metric(t, (1000.0, 10), (1200.0, 10), 5)
        bad = check_metric(t, (1000.0, 10), (1250.0, 10), 5)
        assert ok["status"] == PASS and ok["delta_pct"] == pytest.approx(20.0)
        assert bad["status"] == FAIL and bad["delta_pct"] == pytest.approx(25.0)

    def test_percentage_decrease_bound(self):
        t = Threshold("x", max_decrease_pct=10.0)
        assert check_metric(t, (100.0, 10), (85.0, 10), 5)["status"] == FAIL
        assert check_metric(t, (100.0, 10), (95.0, 10), 5)["status"] == PASS

    def test_zero_baseline_percentage_is_skipped(self):
        row = check_metric(Threshold("x", max_increase_pct=10.0), (0.0, 10), (5.0, 10), 5)
        assert row["status"] == SKIPPED
        assert "baseline is 0" in row["reason"]

    def test_too_few_examples_is_skipped(self):
        row = check_metric(Threshold("f", min_delta=-0.03), (0.9, 4), (0.1, 10), 5)
        assert row["status"] == SKIPPED
        assert row["reason"].startswith("insufficient data")

    def test_unmeasured_metric_is_skipped(self):
        row = check_metric(Threshold("f", min_delta=-0.03), (0.9, 10), None, 5)
        assert row["status"] == SKIPPED
        assert "current run" in row["reason"]


class TestMetricValue:
    def test_current_layout(self):
        run = _run()
        assert metric_value(run, "faithfulness") == (0.9, 10)
        assert metric_value(run, "latency_p50_ms") == (1000.0, 10)
        assert metric_value(run, "tokens_per_question") == (1000.0, 10)
        assert metric_value(run, "ndcg@10") is None

    def test_legacy_layout(self):
        legacy = {
            "n": 8,
            "n_scored": 6,
            "aggregate": {"faithfulness": 0.7},
            "retrieval_aggregate": {"mrr": 0.4, "n": 7},
            "cost": {"mean_latency_ms": 900, "total_input_tokens": 700, "total_output_tokens": 100},
        }
        assert metric_value(legacy, "faithfulness") == (0.7, 6)
        assert metric_value(legacy, "mrr") == (0.4, 7)
        assert metric_value(legacy, "tokens_per_question") == (100.0, 8)
        assert metric_value(legacy, "latency_p50_ms") is None  # never recorded


class TestCompareRuns:
    def test_all_within_thresholds_passes(self):
        report = compare_runs(_run(), _run(faith=0.89, run_id="b"), CONFIG)
        assert report["status"] == PASS
        assert report["n_fail"] == 0
        assert report["comparable"] is True

    def test_any_failure_fails_the_report(self):
        current = _run(faith=0.8, p50=1300.0, tokens=1100.0, run_id="b")
        report = compare_runs(_run(), current, CONFIG)
        assert report["status"] == FAIL
        assert _status(report, "faithfulness") == FAIL
        assert _status(report, "latency_p50_ms") == FAIL  # +30% > +20%
        assert _status(report, "tokens_per_question") == PASS  # +10% <= +15%
        assert report["n_fail"] == 2

    def test_no_baseline_skips_everything(self):
        report = compare_runs(None, _run(), CONFIG)
        assert report["status"] == SKIPPED
        assert report["n_skipped"] == len(DEFAULT_THRESHOLDS)
        assert all("no baseline" in c["reason"] for c in report["checks"])

    def test_config_mismatch_is_flagged(self):
        report = compare_runs(_run(), _run(config_hash="h2", run_id="b"), CONFIG)
        assert report["comparable"] is False
        assert report["config_mismatch"] == ["config_hash"]

    def test_report_is_json_serializable(self):
        json.dumps(compare_runs(_run(), _run(run_id="b"), CONFIG))


class TestRenderMarkdown:
    def test_table_rows_and_statuses(self):
        report = compare_runs(_run(), _run(faith=0.8, n=10, run_id="b"), CONFIG)
        report["checks"].append(
            check_metric(Threshold("ndcg@10", min_delta=-0.03), None, (0.5, 10), 5)
        )
        md = render_markdown(report)
        assert md.startswith("## Regression report: FAIL")
        assert "| Metric | Baseline | Current | Delta | Threshold | Status |" in md
        assert "| faithfulness | 0.9000 | 0.8000 | -0.1000 (-11.1%) | delta >= -0.03 | FAIL |" in md
        assert "| latency_p50_ms | 1,000.0 | 1,000.0 | +0.0 (+0.0%) | <= +20% | PASS |" in md
        assert "| ndcg@10 | - | 0.5000 | - | delta >= -0.03 | SKIPPED-insufficient-data |" in md

    def test_no_baseline(self):
        md = render_markdown(compare_runs(None, _run(), CONFIG))
        assert "Baseline: none" in md

    def test_mismatch_warning(self):
        md = render_markdown(compare_runs(_run(), _run(model="other", run_id="b"), CONFIG))
        assert "differ in model" in md


class TestBaselines:
    @pytest.fixture
    def runs_dir(self, tmp_path, monkeypatch):
        d = tmp_path / "runs"
        d.mkdir()
        monkeypatch.setattr(regression, "RUNS_DIR", d)
        return d

    @staticmethod
    def _save(d, run):
        body = {k: v for k, v in run.items() if k != "id"}
        path = d / f"{run['id']}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_previous_run_of_the_same_configuration(self, runs_dir):
        self._save(runs_dir, _run(run_id="20260101T000000Z-a"))
        self._save(runs_dir, _run(run_id="20260102T000000Z-b", config_hash="other"))
        self._save(runs_dir, _run(run_id="20260103T000000Z-c", faith=0.5))
        current = _run(run_id="20260104T000000Z-d")
        self._save(runs_dir, current)

        baseline, source = find_baseline(current)
        assert source == "previous"
        assert baseline["id"] == "20260103T000000Z-c"

    def test_later_runs_are_never_baselines(self, runs_dir):
        current = _run(run_id="20260101T000000Z-a")
        self._save(runs_dir, current)
        self._save(runs_dir, _run(run_id="20260102T000000Z-b"))
        assert find_baseline(current) == (None, None)

    def test_regression_reports_are_not_runs(self, runs_dir):
        current = _run(run_id="20260102T000000Z-b")
        (runs_dir / "20260101T000000Z-a.regression.json").write_text(
            json.dumps(_run()), encoding="utf-8"
        )
        assert find_baseline(current) == (None, None)
        assert regression.list_runs() == []

    def test_pinned_baseline_wins(self, runs_dir):
        pinned_src = self._save(runs_dir, _run(run_id="20260101T000000Z-a", faith=0.95))
        self._save(runs_dir, _run(run_id="20260102T000000Z-b"))
        target = pin_baseline(pinned_src)
        assert target.parent == runs_dir / "baselines"
        assert target.name.startswith("golden_v3__classic__")

        current = _run(run_id="20260103T000000Z-c")
        baseline, source = find_baseline(current)
        assert source == "pinned"
        assert baseline["id"] == "20260101T000000Z-a"
        assert baseline["aggregates"]["faithfulness"]["mean"] == 0.95

    def test_pinned_baseline_is_per_configuration(self, runs_dir):
        pin_baseline(self._save(runs_dir, _run(run_id="20260101T000000Z-a", config_hash="x")))
        current = _run(run_id="20260103T000000Z-c")
        assert find_baseline(current) == (None, None)

    def test_a_run_is_not_its_own_pinned_baseline(self, runs_dir):
        current = _run(run_id="20260101T000000Z-a")
        pin_baseline(self._save(runs_dir, current))
        assert find_baseline(current) == (None, None)

    def test_save_report_writes_json_and_markdown(self, runs_dir):
        report = compare_runs(_run(), _run(run_id="b"), CONFIG)
        json_path, md_path = save_report(report, "b")
        assert json_path.name == "b.regression.json"
        assert json.loads(json_path.read_text())["status"] == PASS
        assert md_path.read_text().startswith("## Regression report")

    def test_pin_rejects_a_non_run(self, runs_dir):
        bad = runs_dir / "x.json"
        bad.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError):
            pin_baseline(bad)


class TestGroupingKey:
    def test_retrieval_config_separates_groups(self, tmp_path, monkeypatch):
        """Dense and hybrid runs are different configurations, not a regression."""
        monkeypatch.setattr(regression, "RUNS_DIR", tmp_path)
        for name, cfg, faith in (("20260101T000000Z-a", "dense", 0.9), ("20260102T000000Z-b", "hybrid", 0.3)):
            (tmp_path / f"{name}.json").write_text(
                json.dumps({"dataset": "d", "config_hash": cfg, "aggregate": {"faithfulness": faith}}),
                encoding="utf-8",
            )
        assert regression.load_regressions() == []

    def test_rubric_version_separates_groups(self, tmp_path, monkeypatch):
        monkeypatch.setattr(regression, "RUNS_DIR", tmp_path)
        for name, rv, faith in (("20260101T000000Z-a", "v1", 0.9), ("20260102T000000Z-b", "v2", 0.3)):
            (tmp_path / f"{name}.json").write_text(
                json.dumps({"dataset": "d", "rubric_version": rv, "aggregate": {"faithfulness": faith}}),
                encoding="utf-8",
            )
        assert regression.load_regressions() == []
