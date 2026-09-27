"""Storage limitation (GDPR art. 5(1)(e)): purge records past their retention period.

Each store registers a purge function with :func:`register_retention_target`,
naming the setting that holds its retention period in days. A period of 0
keeps that store's records forever. :func:`run_retention` purges every target
once; :func:`start_retention_task` runs it at startup and then daily for as
long as the app is up.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import settings
from app.logging_config import get_structured_logger
from app.tracing import purge_traces

logger = get_structured_logger(__name__)

RETENTION_INTERVAL_SECONDS = 24 * 60 * 60

RetentionFn = Callable[[datetime], int | Awaitable[int]]


@dataclass(frozen=True)
class RetentionTarget:
    """A store with a retention period.

    Attributes:
        name: Key under which the target appears in the report.
        purge: Deletes records created before the datetime it is given and
            returns, or resolves to, how many it deleted.
        days_setting: Name of the :class:`~app.config.Settings` field holding
            the retention period in days, read at every run.
    """

    name: str
    purge: RetentionFn
    days_setting: str


@dataclass
class RetentionReport:
    """What one retention run deleted, per target."""

    ran_at: str
    targets: dict[str, int] = field(default_factory=dict)
    cutoffs: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran_at": self.ran_at,
            "targets": self.targets,
            "cutoffs": self.cutoffs,
            "skipped": self.skipped,
            "errors": self.errors,
        }


_targets: dict[str, RetentionTarget] = {}


def register_retention_target(name: str, fn: RetentionFn, *, days_setting: str) -> None:
    """Add (or replace) a store to purge.

    Args:
        name: Report key for this target.
        fn: ``fn(before) -> int``: delete records created before ``before``.
        days_setting: Settings field with the retention period in days, e.g.
            ``"retention_audit_days"``.

    Raises:
        ValueError: If ``days_setting`` is not a settings field.
    """
    if not hasattr(settings, days_setting):
        raise ValueError(f"Unknown retention setting '{days_setting}'")
    _targets[name] = RetentionTarget(name=name, purge=fn, days_setting=days_setting)


def unregister_retention_target(name: str) -> None:
    """Remove a target. A no-op if it is not registered."""
    _targets.pop(name, None)


def retention_targets() -> list[str]:
    """Names of the registered targets, in the order they run."""
    return list(_targets)


async def run_retention(now: datetime | None = None) -> RetentionReport:
    """Purge every registered target once.

    A failing target does not stop the others; its error is logged and
    reported.

    Args:
        now: Reference time. Defaults to the current UTC time.
    """
    now = now or datetime.now(UTC)
    report = RetentionReport(ran_at=now.isoformat(timespec="seconds"))
    for target in list(_targets.values()):
        days = int(getattr(settings, target.days_setting))
        if days <= 0:
            report.skipped.append(target.name)
            continue
        before = now - timedelta(days=days)
        report.cutoffs[target.name] = before.isoformat(timespec="seconds")
        try:
            result = target.purge(before)
            if inspect.isawaitable(result):
                result = await result
            report.targets[target.name] = int(result)
        except Exception as e:
            report.errors[target.name] = f"{type(e).__name__}: {e}"
            logger.error(
                f"Retention target '{target.name}' failed: {type(e).__name__}: {e}",
                extra_fields={"target": target.name},
            )
    logger.info(
        "Retention run completed",
        extra_fields={
            "targets": report.targets,
            "skipped": report.skipped,
            "failed_targets": sorted(report.errors),
        },
    )
    return report


async def _retention_loop(interval_seconds: float) -> None:
    while True:
        try:
            await run_retention()
        except Exception as e:  # run_retention isolates targets; this is a backstop
            logger.error(f"Retention run failed: {type(e).__name__}: {e}")
        await asyncio.sleep(interval_seconds)


def start_retention_task(
    interval_seconds: float = RETENTION_INTERVAL_SECONDS,
) -> asyncio.Task[None]:
    """Run retention now and then every ``interval_seconds`` in the background.

    The caller owns the task and must cancel it on shutdown.
    """
    return asyncio.create_task(_retention_loop(interval_seconds), name="retention")


# ---------------------------------------------------------------------------
# Built-in targets
# ---------------------------------------------------------------------------


def _run_created_at(path: Path, run: dict) -> datetime:
    """When a saved eval run was created: its ``created_at``, else the file's mtime."""
    raw = run.get("created_at")
    if isinstance(raw, str):
        try:
            created = datetime.fromisoformat(raw)
            return created if created.tzinfo else created.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def purge_eval_runs(before: datetime) -> int:
    """Delete saved eval runs created before ``before``. Returns how many."""
    from app.eval import metrics

    runs_dir = metrics.RUNS_DIR
    if not runs_dir.exists():
        return 0
    deleted = 0
    for path in sorted(runs_dir.glob("*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run = {}
        if _run_created_at(path, run if isinstance(run, dict) else {}) < before:
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted


def register_builtin_targets() -> None:
    """(Re)register the built-in targets. Called at import."""
    register_retention_target("traces", purge_traces, days_setting="retention_traces_days")
    register_retention_target(
        "eval_runs", purge_eval_runs, days_setting="retention_eval_runs_days"
    )


register_builtin_targets()
