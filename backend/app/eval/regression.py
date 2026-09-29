"""Comparison of eval runs over time, for spotting metric regressions.

Runs are only ever compared against other runs with the same dataset, strategy,
model, provider, prompt version, judge, rubric and retrieval configuration.
Comparing across configurations would report every deliberate change as a
regression.

Two layers live here:

* :func:`load_regressions` -- the dashboard view: every drop larger than a
  single threshold between consecutive comparable runs.
* :func:`compare_runs` -- the gate: one run against one baseline (a pinned
  baseline file, or the previous comparable run), checked metric by metric
  against the thresholds in ``eval/regression_thresholds.toml``, producing a
  structured report plus a markdown table (:func:`render_markdown`).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tomllib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from app.config import settings
from app.eval.judge import DIMENSIONS

logger = logging.getLogger(__name__)

RUNS_DIR = settings.data_path / "eval_runs"

#: Suffix of regression reports saved next to a run; never loaded as runs.
REPORT_SUFFIX = ".regression"

#: Repository-level thresholds file (``<repo>/eval/regression_thresholds.toml``).
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "eval" / "regression_thresholds.toml"
THRESHOLDS_ENV_VAR = "REGRESSION_THRESHOLDS_PATH"

PASS, FAIL, SKIPPED = "PASS", "FAIL", "SKIPPED"


def _run_files() -> list[Path]:
    """Return run files sorted oldest first. Filenames start with a timestamp.

    Regression reports saved next to runs are excluded.
    """
    if not RUNS_DIR.exists():
        return []
    return sorted(p for p in RUNS_DIR.glob("*.json") if not p.stem.endswith(REPORT_SUFFIX))


def _load(path: Path) -> dict | None:
    """Load one run file, returning None (and logging) if it is unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Skipping unreadable eval run %s: %s", path.name, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Skipping eval run %s: expected an object", path.name)
        return None
    data["id"] = path.stem
    return data


def list_runs() -> list[dict]:
    """Load every run in full, oldest first.

    Prefer :func:`list_run_summaries` for anything user-facing -- this loads
    per-example detail for every run and grows without bound.
    """
    return [r for r in (_load(p) for p in _run_files()) if r is not None]


def list_run_summaries() -> list[dict]:
    """Summarize every run, newest first, without per-example detail.

    Returns:
        One dict per run: id, configuration, counts and aggregate scores.
    """
    summaries = []
    for run in list_runs():
        summaries.append(
            {
                "id": run["id"],
                "dataset": run.get("dataset"),
                "strategy": run.get("strategy"),
                "provider": run.get("provider"),
                "model": run.get("model"),
                "prompt_version": run.get("prompt_version"),
                "judge_model": run.get("judge_model"),
                "n": run.get("n", 0),
                "n_scored": run.get("n_scored", run.get("n", 0)),
                "n_unscored": run.get("n_unscored", 0),
                "n_judge_failed": run.get("n_judge_failed"),
                "n_generation_failed": run.get("n_generation_failed"),
                "rubric_version": run.get("rubric_version"),
                "retrieval_mode": run.get("retrieval_mode"),
                "config_hash": run.get("config_hash"),
                "regression_status": (run.get("regression") or {}).get("status"),
                "aggregates": run.get("aggregates"),
                "aggregate": run.get("aggregate"),
                "retrieval_aggregate": run.get("retrieval_aggregate"),
                "cost": run.get("cost"),
                "created_at": run.get("created_at"),
            }
        )
    summaries.reverse()
    return summaries


def get_run(run_id: str) -> dict | None:
    """Load one run in full by its id.

    Args:
        run_id: The run's filename stem, as returned by list_run_summaries.

    Returns:
        The run dict, or None if no such run exists.
    """
    # Resolve against the known set rather than joining user input onto a path.
    for path in _run_files():
        if path.stem == run_id:
            return _load(path)
    return None


def _config_key(run: dict) -> tuple:
    """Group key identifying runs that are legitimately comparable.

    Strategy is part of the key: Classic scoring lower than Agentic is the
    finding the project exists to produce, not a regression. ``model`` and
    ``provider`` are what the run actually used (older runs recorded only the
    request override), and the judge model is included because a different
    judge scores differently -- and the rubric version for the same reason.
    ``config_hash`` covers the retrieval configuration (mode, top-k, chunking,
    embedding and reranker models), so dense and hybrid runs, or runs over
    differently chunked indexes, are never compared as the same configuration.
    Older runs without the last two fields group together as before.
    """
    return (
        run.get("dataset"),
        run.get("strategy"),
        run.get("provider"),
        run.get("model"),
        run.get("prompt_version"),
        run.get("judge_model"),
        run.get("rubric_version"),
        run.get("config_hash"),
    )


def load_regressions(threshold: float = 0.05) -> list[dict]:
    """Find metric drops between consecutive runs of the same configuration.

    Args:
        threshold: Minimum drop (in absolute score) counted as a regression.

    Returns:
        One entry per regressed metric, newest first, naming the two runs and
        the size of the drop. Runs with no aggregate (nothing scored) are
        skipped rather than treated as zero.
    """
    groups: dict[tuple, list[dict]] = {}
    for run in list_runs():
        if run.get("aggregate"):
            groups.setdefault(_config_key(run), []).append(run)

    regressions: list[dict] = []
    for runs in groups.values():
        for prev, curr in pairwise(runs):
            for metric in DIMENSIONS:
                before = prev["aggregate"].get(metric)
                after = curr["aggregate"].get(metric)
                if before is None or after is None:
                    continue
                delta = after - before
                if delta < -threshold:
                    regressions.append(
                        {
                            "metric": metric,
                            "delta": round(delta, 4),
                            "before": round(before, 4),
                            "after": round(after, 4),
                            "dataset": curr.get("dataset"),
                            "strategy": curr.get("strategy"),
                            "model": curr.get("model"),
                            "prompt_version": curr.get("prompt_version"),
                            "from_run": prev["id"],
                            "to_run": curr["id"],
                            "detected_at": curr.get("created_at"),
                        }
                    )

    regressions.sort(key=lambda r: r.get("to_run") or "", reverse=True)
    return regressions


# ---------------------------------------------------------------------------
# Regression gate: thresholds, baselines, comparison, reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Threshold:
    """Allowed change of one metric between a baseline and a current run.

    Absolute bounds apply to ``delta = current - baseline``; percentage bounds
    to ``delta_pct = 100 * (current - baseline) / baseline``. A check fails
    when any configured bound is violated.

    Attributes:
        metric: Metric name (``faithfulness``, ``recall@5``, ``latency_p50_ms``...).
        min_delta: Fail if ``delta < min_delta`` (e.g. ``-0.03`` for a score).
        max_delta: Fail if ``delta > max_delta``.
        max_increase_pct: Fail if ``delta_pct > max_increase_pct`` (cost metrics).
        max_decrease_pct: Fail if ``delta_pct < -max_decrease_pct``.
    """

    metric: str
    min_delta: float | None = None
    max_delta: float | None = None
    max_increase_pct: float | None = None
    max_decrease_pct: float | None = None

    def describe(self) -> str:
        parts = []
        if self.min_delta is not None:
            parts.append(f"delta >= {self.min_delta:+g}")
        if self.max_delta is not None:
            parts.append(f"delta <= {self.max_delta:+g}")
        if self.max_increase_pct is not None:
            parts.append(f"<= +{self.max_increase_pct:g}%")
        if self.max_decrease_pct is not None:
            parts.append(f">= -{self.max_decrease_pct:g}%")
        return ", ".join(parts) or "(none)"

    @property
    def relative(self) -> bool:
        return self.max_increase_pct is not None or self.max_decrease_pct is not None


_THRESHOLD_FIELDS = ("min_delta", "max_delta", "max_increase_pct", "max_decrease_pct")

#: Built-in defaults, identical to the shipped TOML file. Used when the file
#: is absent (e.g. a container without the repository checkout).
DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "faithfulness": Threshold("faithfulness", min_delta=-0.03),
    "answer_relevance": Threshold("answer_relevance", min_delta=-0.03),
    "recall@5": Threshold("recall@5", min_delta=-0.03),
    "mrr": Threshold("mrr", min_delta=-0.03),
    "latency_p50_ms": Threshold("latency_p50_ms", max_increase_pct=20.0),
    "tokens_per_question": Threshold("tokens_per_question", max_increase_pct=15.0),
}
DEFAULT_MIN_EXAMPLES = 5


@dataclass
class ThresholdConfig:
    """The thresholds in force for a comparison, and where they came from."""

    thresholds: dict[str, Threshold] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    min_examples: int = DEFAULT_MIN_EXAMPLES
    source: str = "built-in defaults"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "min_examples": self.min_examples,
            "metrics": {
                name: {f: getattr(t, f) for f in _THRESHOLD_FIELDS if getattr(t, f) is not None}
                for name, t in self.thresholds.items()
            },
        }


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _parse_thresholds_doc(doc: dict[str, Any], source: str) -> ThresholdConfig:
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"{source}: missing or empty [metrics] table")
    thresholds: dict[str, Threshold] = {}
    for name, spec in metrics.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{source}: metrics.{name} must be a table")
        unknown = set(spec) - set(_THRESHOLD_FIELDS)
        if unknown:
            raise ValueError(f"{source}: metrics.{name}: unknown keys {sorted(unknown)}")
        bounds = {k: _as_float(v, f"{source}: metrics.{name}.{k}") for k, v in spec.items()}
        if not bounds:
            raise ValueError(f"{source}: metrics.{name} sets no bound")
        thresholds[name] = Threshold(name, **bounds)
    min_examples = doc.get("min_examples", DEFAULT_MIN_EXAMPLES)
    if isinstance(min_examples, bool) or not isinstance(min_examples, int) or min_examples < 1:
        raise ValueError(f"{source}: min_examples must be a positive integer")
    return ThresholdConfig(thresholds=thresholds, min_examples=min_examples, source=source)


def parse_threshold_override(spec: str) -> tuple[str, Threshold | None]:
    """Parse a ``METRIC=VALUE`` override from the command line.

    ``VALUE`` forms: ``-0.05`` (min_delta), ``+0.05`` or ``0.05`` (max_delta),
    ``+25%`` (max_increase_pct), ``-10%`` (max_decrease_pct), ``off`` (drop the
    check). Returns ``(metric, threshold)``; ``threshold`` is ``None`` for ``off``.

    Raises:
        ValueError: On a malformed override.
    """
    metric, sep, raw = spec.partition("=")
    metric, raw = metric.strip(), raw.strip()
    if not sep or not metric or not raw:
        raise ValueError(f"threshold override {spec!r} is not METRIC=VALUE")
    if raw.lower() == "off":
        return metric, None
    try:
        if raw.endswith("%"):
            pct = float(raw[:-1])
            if pct < 0:
                return metric, Threshold(metric, max_decrease_pct=-pct)
            return metric, Threshold(metric, max_increase_pct=pct)
        value = float(raw)
    except ValueError as e:
        raise ValueError(f"threshold override {spec!r}: {raw!r} is not a number") from e
    if raw.startswith("-"):
        return metric, Threshold(metric, min_delta=value)
    return metric, Threshold(metric, max_delta=value)


def load_thresholds(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    min_examples: int | None = None,
) -> ThresholdConfig:
    """Resolve the thresholds in force.

    Precedence (highest first): ``overrides`` / ``min_examples`` (CLI flags),
    ``path``, the ``REGRESSION_THRESHOLDS_PATH`` environment variable, the
    repository file ``eval/regression_thresholds.toml``, built-in defaults.
    An override replaces the bounds of its kind (absolute or percentage) for
    that metric and drops the other kind.

    Raises:
        FileNotFoundError: If an explicitly named file (argument or env var)
            does not exist.
        ValueError: If the file or an override is malformed.
    """
    explicit = path or os.environ.get(THRESHOLDS_ENV_VAR) or None
    if explicit is not None:
        file = Path(explicit)
        if not file.is_file():
            raise FileNotFoundError(f"regression thresholds file not found: {file}")
    else:
        file = DEFAULT_THRESHOLDS_PATH

    if file.is_file():
        with file.open("rb") as f:
            try:
                doc = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise ValueError(f"{file}: invalid TOML: {e}") from e
        config = _parse_thresholds_doc(doc, str(file))
    else:
        config = ThresholdConfig()

    if overrides:
        thresholds = dict(config.thresholds)
        for spec in overrides:
            metric, threshold = parse_threshold_override(spec)
            if threshold is None:
                thresholds.pop(metric, None)
            else:
                thresholds[metric] = threshold
        config = replace(
            config, thresholds=thresholds, source=f"{config.source} + CLI overrides"
        )
    if min_examples is not None:
        if min_examples < 1:
            raise ValueError("min_examples must be a positive integer")
        config = replace(config, min_examples=min_examples)
    return config


def metric_value(run: dict, metric: str) -> tuple[float, int] | None:
    """A run's value for ``metric`` and the number of examples behind it.

    Understands both run layouts: the ``aggregates`` block of current runs
    and the ``aggregate`` / ``retrieval_aggregate`` blocks of older ones.
    Derived metrics: ``latency_p50_ms`` and ``tokens_per_question`` (both from
    the ``cost`` block).

    Returns:
        ``(value, n)``, or ``None`` when the run did not measure it.
    """
    aggregates = run.get("aggregates") or {}
    cost = run.get("cost") or {}
    n_generated = (aggregates.get("latency_ms") or {}).get("n")
    if n_generated is None:
        n_generated = max(0, int(run.get("n") or 0) - int(run.get("n_generation_failed") or 0))

    if metric == "latency_p50_ms":
        value = cost.get("p50_latency_ms")
        return (float(value), n_generated) if value is not None else None
    if metric == "tokens_per_question":
        value = cost.get("mean_tokens_per_question")
        if value is None and n_generated and "total_input_tokens" in cost:
            value = (cost["total_input_tokens"] + cost.get("total_output_tokens", 0)) / n_generated
        return (float(value), n_generated) if value is not None else None

    block = aggregates.get(metric)
    if isinstance(block, dict) and block.get("mean") is not None:
        return float(block["mean"]), int(block.get("n") or 0)

    legacy = run.get("aggregate") or {}
    if legacy.get(metric) is not None:
        return float(legacy[metric]), int(run.get("n_scored") or 0)
    retrieval = run.get("retrieval_aggregate") or {}
    if retrieval.get(metric) is not None:
        return float(retrieval[metric]), int(retrieval.get("n") or 0)
    return None


def check_metric(
    threshold: Threshold,
    baseline: tuple[float, int] | None,
    current: tuple[float, int] | None,
    min_examples: int,
) -> dict[str, Any]:
    """Evaluate one threshold. Returns a report row (see :func:`compare_runs`)."""
    row: dict[str, Any] = {
        "metric": threshold.metric,
        "baseline": baseline[0] if baseline else None,
        "current": current[0] if current else None,
        "baseline_n": baseline[1] if baseline else None,
        "current_n": current[1] if current else None,
        "delta": None,
        "delta_pct": None,
        "threshold": threshold.describe(),
        "status": SKIPPED,
        "reason": None,
    }
    if baseline is None or current is None:
        missing = "baseline" if baseline is None else "current run"
        row["reason"] = f"insufficient data: not measured in {missing}"
        return row
    if baseline[1] < min_examples or current[1] < min_examples:
        row["reason"] = (
            f"insufficient data: n={baseline[1]} (baseline) / {current[1]} (current), "
            f"minimum {min_examples}"
        )
        return row

    base, cur = baseline[0], current[0]
    delta = cur - base
    row["delta"] = delta
    row["delta_pct"] = (100.0 * delta / base) if base != 0 else None

    eps = 1e-9
    violations = []
    if threshold.min_delta is not None and delta < threshold.min_delta - eps:
        violations.append(f"delta {delta:+.4f} < {threshold.min_delta:+g}")
    if threshold.max_delta is not None and delta > threshold.max_delta + eps:
        violations.append(f"delta {delta:+.4f} > {threshold.max_delta:+g}")
    if threshold.relative:
        pct = row["delta_pct"]
        if pct is None:
            row["reason"] = "insufficient data: baseline is 0, percentage change undefined"
            return row
        if threshold.max_increase_pct is not None and pct > threshold.max_increase_pct + eps:
            violations.append(f"{pct:+.1f}% > +{threshold.max_increase_pct:g}%")
        if threshold.max_decrease_pct is not None and pct < -threshold.max_decrease_pct - eps:
            violations.append(f"{pct:+.1f}% < -{threshold.max_decrease_pct:g}%")

    row["status"] = FAIL if violations else PASS
    row["reason"] = "; ".join(violations) or None
    return row


_IDENTITY_FIELDS = (
    "dataset",
    "strategy",
    "provider",
    "model",
    "prompt_version",
    "judge_model",
    "rubric_version",
    "config_hash",
)


def _run_ref(run: dict | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "id": run.get("id"),
        "created_at": run.get("created_at"),
        **{f: run.get(f) for f in _IDENTITY_FIELDS},
    }


def compare_runs(
    baseline: dict | None,
    current: dict,
    config: ThresholdConfig | None = None,
    baseline_source: str = "explicit",
) -> dict[str, Any]:
    """Check ``current`` against ``baseline`` metric by metric.

    Args:
        baseline: The baseline run, or ``None`` when there is none (every
            check is then SKIPPED).
        current: The run under test.
        config: Thresholds; defaults to :func:`load_thresholds`.
        baseline_source: How the baseline was chosen: ``"pinned"``,
            ``"previous"`` or ``"explicit"``.

    Returns:
        A JSON-ready report: ``status`` (FAIL if any check failed, else PASS
        if any passed, else SKIPPED), counts, ``checks`` (one row per metric:
        metric, baseline, current, delta, delta_pct, threshold, status,
        reason, and the n behind each side), both run references, whether the
        two runs share a configuration (``comparable`` and ``config_mismatch``)
        and the thresholds used.
    """
    config = config or load_thresholds()
    checks = [
        check_metric(
            t,
            metric_value(baseline, name) if baseline is not None else None,
            metric_value(current, name),
            config.min_examples,
        )
        for name, t in config.thresholds.items()
    ]
    if baseline is None:
        for c in checks:
            c["reason"] = "insufficient data: no baseline run for this configuration"

    counts = {s: sum(1 for c in checks if c["status"] == s) for s in (PASS, FAIL, SKIPPED)}
    status = FAIL if counts[FAIL] else PASS if counts[PASS] else SKIPPED
    mismatch = (
        [f for f in _IDENTITY_FIELDS if baseline.get(f) != current.get(f)]
        if baseline is not None
        else []
    )
    return {
        "status": status,
        "n_pass": counts[PASS],
        "n_fail": counts[FAIL],
        "n_skipped": counts[SKIPPED],
        "baseline_source": baseline_source if baseline is not None else None,
        "baseline": _run_ref(baseline),
        "current": _run_ref(current),
        "comparable": baseline is not None and not mismatch,
        "config_mismatch": mismatch,
        "thresholds": config.to_dict(),
        "checks": checks,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _fmt(value: float | None, metric: str) -> str:
    if value is None:
        return "-"
    if metric.endswith("_ms") or metric == "tokens_per_question":
        return f"{value:,.1f}"
    return f"{value:.4f}"


def _fmt_delta(row: dict) -> str:
    if row["delta"] is None:
        return "-"
    text = _fmt(row["delta"], row["metric"])
    if not text.startswith("-"):
        text = "+" + text
    if row["delta_pct"] is not None:
        text += f" ({row['delta_pct']:+.1f}%)"
    return text


def render_markdown(report: dict[str, Any]) -> str:
    """Render a :func:`compare_runs` report as a markdown summary and table."""
    base = report.get("baseline") or {}
    cur = report.get("current") or {}
    lines = [
        f"## Regression report: {report['status']}",
        "",
        f"- Current: `{cur.get('id')}` ({cur.get('dataset')}, {cur.get('strategy')}, "
        f"{cur.get('model')})",
        (
            f"- Baseline ({report.get('baseline_source')}): `{base.get('id')}`"
            if report.get("baseline")
            else "- Baseline: none for this configuration"
        ),
        f"- Checks: {report['n_pass']} pass, {report['n_fail']} fail, "
        f"{report['n_skipped']} skipped (thresholds: {report['thresholds']['source']})",
    ]
    if report.get("config_mismatch"):
        lines.append(
            "- Warning: the runs differ in "
            + ", ".join(report["config_mismatch"])
            + "; deltas may reflect a configuration change."
        )
    lines += [
        "",
        "| Metric | Baseline | Current | Delta | Threshold | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["checks"]:
        status = row["status"]
        if status == SKIPPED:
            status = "SKIPPED-insufficient-data"
        lines.append(
            f"| {row['metric']} | {_fmt(row['baseline'], row['metric'])} "
            f"| {_fmt(row['current'], row['metric'])} | {_fmt_delta(row)} "
            f"| {row['threshold']} | {status} |"
        )
    notes = [f"- {r['metric']}: {r['reason']}" for r in report["checks"] if r.get("reason")]
    if notes:
        lines += ["", "Notes:", *notes]
    return "\n".join(lines) + "\n"


def config_id(run: dict) -> str:
    """Filesystem-safe id of a run's comparison group, for baseline files.

    Readable prefix (dataset, strategy) plus a hash of the full group key.
    """
    key_hash = hashlib.sha256(
        json.dumps(_config_key(run), default=str).encode("utf-8")
    ).hexdigest()[:12]

    def safe(s: Any) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s or "default"))

    return f"{safe(run.get('dataset'))}__{safe(run.get('strategy'))}__{key_hash}"


def baselines_dir() -> Path:
    return RUNS_DIR / "baselines"


def baseline_path(run: dict) -> Path:
    """Where the pinned baseline for ``run``'s configuration lives."""
    return baselines_dir() / f"{config_id(run)}.json"


def pin_baseline(run_file: Path) -> Path:
    """Pin a run file as the baseline for its configuration.

    The run is copied to ``<RUNS_DIR>/baselines/<config>.json`` with
    ``pinned_id`` (the source run's id) and ``pinned_at`` added, and without
    its ``per_example`` rows: comparison needs only the aggregates, and a
    copy of every answer outside the runs directory would escape erasure and
    retention.

    Returns:
        The baseline file written.

    Raises:
        ValueError: If the file is not a readable run.
    """
    run = load_run_file(run_file)
    run.pop("per_example", None)
    run["pinned_id"] = run.pop("id")
    run["pinned_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    target = baseline_path(run)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return target


def load_run_file(path: Path) -> dict:
    """Load a run from any path (for offline comparison).

    Raises:
        ValueError: If the file is missing or not a run object.
    """
    run = _load(Path(path))
    if run is None:
        raise ValueError(f"{path} is not a readable eval run")
    return run


def _measured(run: dict) -> bool:
    return bool(run.get("aggregates") or run.get("aggregate") or run.get("retrieval_aggregate"))


def find_baseline(current: dict) -> tuple[dict | None, str | None]:
    """The baseline to compare ``current`` against.

    A pinned baseline for the configuration wins (unless it *is* the current
    run); otherwise the most recent earlier run of the same configuration that
    measured anything.

    Returns:
        ``(baseline_run, source)`` with source ``"pinned"`` or ``"previous"``,
        or ``(None, None)``.
    """
    current_id = current.get("id") or ""
    pinned = baseline_path(current)
    if pinned.is_file():
        run = _load(pinned)
        if run is not None and run.get("pinned_id") != current_id:
            run["id"] = run.get("pinned_id") or f"baselines/{pinned.stem}"
            return run, "pinned"

    key = _config_key(current)
    previous = [
        r
        for r in list_runs()
        if r["id"] < current_id and _config_key(r) == key and _measured(r)
    ]
    if previous:
        return previous[-1], "previous"
    return None, None


def save_report(
    report: dict[str, Any], run_id: str, directory: Path | None = None
) -> tuple[Path, Path]:
    """Save a report as ``<run_id>.regression.json`` and ``.md`` next to the run."""
    directory = directory or RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{run_id}{REPORT_SUFFIX}.json"
    md_path = directory / f"{run_id}{REPORT_SUFFIX}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def report_summary(report: dict[str, Any], json_path: Path, md_path: Path) -> dict[str, Any]:
    """The compact form of a report stored in the run record itself."""
    return {
        "status": report["status"],
        "n_pass": report["n_pass"],
        "n_fail": report["n_fail"],
        "n_skipped": report["n_skipped"],
        "baseline_id": (report.get("baseline") or {}).get("id"),
        "baseline_source": report.get("baseline_source"),
        "thresholds_source": report["thresholds"]["source"],
        "report_json": json_path.name,
        "report_markdown": md_path.name,
    }


def regression_report_for(run: dict, config: ThresholdConfig | None = None) -> dict[str, Any]:
    """Compare a run against its baseline (pinned, else previous comparable run)."""
    baseline, source = find_baseline(run)
    return compare_runs(baseline, run, config, baseline_source=source or "previous")
