"""Optional Weights & Biases tracer.

Applies patterns from DeepLearning.AI "Evaluating & Debugging Generative AI":
  - per-run config logging
  - W&B Tables for prompt/response traces
  - artifact versioning for prompts + golden datasets

No-ops unless the telemetry policy allows W&B (WANDB_ENABLED=true, and
TELEMETRY_MODE permits the requested WANDB_MODE; see app/tracing/policy.py) and
the wandb package is installed. When W&B is not allowed, WANDB_MODE is forced to
``disabled`` so the SDK never opens a connection, whoever imports it.
"""
from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import settings
from app.tracing import policy

# Imported dynamically so the module type-checks the same way whether or not the
# optional `tracing` extra is installed.
wandb: Any
try:
    wandb = importlib.import_module("wandb")
except ImportError:  # pragma: no cover
    wandb = None


def _enabled() -> bool:
    """Apply the policy to the SDK's environment and report whether to log."""
    decision = policy.evaluate(settings)
    policy.apply_wandb_env(decision, settings)
    return decision.wandb.enabled and wandb is not None


@contextmanager
def wandb_run(name: str, config: dict | None = None, tags: list[str] | None = None) -> Iterator[Any]:
    """Start a W&B run if configured; else yield None."""
    if not _enabled():
        yield None
        return
    run = wandb.init(
        project=settings.wandb_project,
        name=name,
        config=config or {},
        tags=tags or [],
        reinit=True,
    )
    try:
        yield run
    finally:
        run.finish()


def log_eval_table(run: Any, rows: list[dict]) -> None:
    """Log per-example scores as a W&B Table (course pattern)."""
    if run is None or wandb is None or not rows:
        return
    columns = sorted({k for r in rows for k in r})
    table = wandb.Table(columns=columns)
    for r in rows:
        table.add_data(*[r.get(c) for c in columns])
    run.log({"eval/examples": table})


def log_metrics(run: Any, metrics: dict) -> None:
    if run is None:
        return
    run.log({f"eval/{k}": v for k, v in metrics.items() if v is not None})


def log_prompt_artifact(run: Any, prompt_name: str, prompt_text: str) -> None:
    if run is None or wandb is None:
        return
    art = wandb.Artifact(f"prompt-{prompt_name}", type="prompt")
    with art.new_file(f"{prompt_name}.md", mode="w") as f:
        f.write(prompt_text)
    run.log_artifact(art)
