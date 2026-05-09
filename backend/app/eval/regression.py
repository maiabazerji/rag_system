from __future__ import annotations

import json
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[3] / "data" / "eval_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def list_runs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("*.json"))]


def load_regressions(threshold: float = 0.05) -> list[dict]:
    runs = list_runs()
    if len(runs) < 2:
        return []
    regressions: list[dict] = []
    for prev, curr in zip(runs, runs[1:]):
        p_agg = (prev.get("aggregate") or {})
        c_agg = (curr.get("aggregate") or {})
        for metric in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
            delta = c_agg.get(metric, 0) - p_agg.get(metric, 0)
            if delta < -threshold:
                regressions.append({
                    "metric": metric,
                    "delta": delta,
                    "from": prev.get("prompt_version"),
                    "to": curr.get("prompt_version"),
                })
    return regressions
