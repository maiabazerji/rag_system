"""Comparison of eval runs over time, for spotting metric regressions.

Runs are only ever compared against other runs with the same dataset, model,
provider and prompt version. Comparing across configurations would report every
deliberate change as a regression.
"""
from __future__ import annotations

import json
import logging
from itertools import pairwise
from pathlib import Path

from app.eval.judge import DIMENSIONS

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parents[3] / "data" / "eval_runs"


def _run_files() -> list[Path]:
    """Return run files sorted oldest first. Filenames start with a timestamp."""
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.glob("*.json"))


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
                "n": run.get("n", 0),
                "n_scored": run.get("n_scored", run.get("n", 0)),
                "n_unscored": run.get("n_unscored", 0),
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
    finding the project exists to produce, not a regression.
    """
    return (
        run.get("dataset"),
        run.get("strategy"),
        run.get("provider"),
        run.get("model"),
        run.get("prompt_version"),
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
