"""Telemetry policy: decides which exporters may send data, and where.

One place answers "may this process send traces or eval runs anywhere?", so the
answer cannot drift between the Langfuse exporter, the W&B tracer and /health.

``TELEMETRY_MODE``:

- ``off`` (default): nothing is exported. W&B may still run in ``offline`` mode,
  which only writes files under ``./wandb`` on this machine.
- ``self_hosted``: exporters may only talk to hosts listed in
  ``TELEMETRY_ALLOWED_HOSTS`` (default ``localhost,langfuse,127.0.0.1``). A
  Langfuse or W&B URL pointing anywhere else is refused and logged at startup.
- ``cloud``: any host, but only with ``TELEMETRY_CLOUD_OPT_IN=true`` as well, so
  copying a mode string between environments cannot start a cross-border
  transfer on its own.

The decision is recomputed from settings on each call. It is cheap, and tests
and operators that change settings see the new answer immediately.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# The public W&B service. Used when WANDB_BASE_URL is unset.
WANDB_CLOUD_URL = "https://api.wandb.ai"


@dataclass(frozen=True)
class ExporterDecision:
    """Whether one exporter may run, and why.

    Attributes:
        enabled: True when the exporter may send data.
        reason: Human-readable explanation, suitable for logs.
        refused: True when the exporter was configured but the policy blocked
            it. Refusals are logged as errors; "simply not configured" is not.
    """

    enabled: bool
    reason: str
    refused: bool = False


@dataclass(frozen=True)
class TelemetryDecision:
    """The telemetry policy applied to the current settings.

    Attributes:
        mode: The effective TELEMETRY_MODE.
        langfuse: Decision for the Langfuse trace exporter.
        wandb: Decision for W&B eval dashboards.
        wandb_mode: The WANDB_MODE the SDK must run with: ``online``,
            ``offline`` or ``disabled``.
    """

    mode: str
    langfuse: ExporterDecision
    wandb: ExporterDecision
    wandb_mode: str


def host_of(url: str) -> str:
    """Return the lower-cased host of a URL, or "" if it has none."""
    return (urlparse(url).hostname or "").lower()


def is_host_allowed(url: str, allowed: list[str]) -> bool:
    """True when the URL's host exactly matches an allowed host.

    Exact matching is deliberate: a suffix match would let ``langfuse.evil.com``
    through an allow-list entry of ``langfuse``.
    """
    host = host_of(url)
    return bool(host) and host in {a.lower() for a in allowed}


def _cloud_gate(s: Any) -> ExporterDecision | None:
    """Refusal when cloud mode lacks its explicit opt-in, else None."""
    if not s.telemetry_cloud_opt_in:
        return ExporterDecision(
            False,
            "TELEMETRY_MODE=cloud also requires TELEMETRY_CLOUD_OPT_IN=true",
            refused=True,
        )
    return None


def _decide_langfuse(s: Any) -> ExporterDecision:
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return ExporterDecision(False, "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    if s.telemetry_mode == "off":
        return ExporterDecision(
            False, "TELEMETRY_MODE=off; set it to self_hosted to export traces", refused=True
        )
    if s.telemetry_mode == "cloud":
        gate = _cloud_gate(s)
        if gate is not None:
            return gate
        return ExporterDecision(True, f"cloud export to {host_of(s.langfuse_host)}")
    if not is_host_allowed(s.langfuse_host, s.telemetry_allowed_host_list):
        return ExporterDecision(
            False,
            f"LANGFUSE_HOST host '{host_of(s.langfuse_host)}' is not in "
            f"TELEMETRY_ALLOWED_HOSTS ({', '.join(s.telemetry_allowed_host_list)})",
            refused=True,
        )
    return ExporterDecision(True, f"self-hosted export to {host_of(s.langfuse_host)}")


def _decide_wandb(s: Any) -> tuple[ExporterDecision, str]:
    requested = s.wandb_mode
    if not s.wandb_enabled:
        return ExporterDecision(False, "WANDB_ENABLED is false"), "disabled"
    if requested == "disabled":
        return ExporterDecision(False, "WANDB_MODE=disabled"), "disabled"
    if requested == "offline":
        # Local files only; permitted in every mode.
        return ExporterDecision(True, "offline: runs are written to ./wandb only"), "offline"

    # online
    if s.telemetry_mode == "off":
        return (
            ExporterDecision(
                False,
                "WANDB_MODE=online is not allowed with TELEMETRY_MODE=off; "
                "use WANDB_MODE=offline",
                refused=True,
            ),
            "disabled",
        )
    base_url = s.wandb_base_url or WANDB_CLOUD_URL
    if s.telemetry_mode == "cloud":
        gate = _cloud_gate(s)
        if gate is not None:
            return gate, "disabled"
    elif not is_host_allowed(base_url, s.telemetry_allowed_host_list):
        return (
            ExporterDecision(
                False,
                f"W&B host '{host_of(base_url)}' is not in TELEMETRY_ALLOWED_HOSTS; "
                "set WANDB_BASE_URL to a self-hosted server or use WANDB_MODE=offline",
                refused=True,
            ),
            "disabled",
        )
    if not s.wandb_api_key:
        return ExporterDecision(False, "WANDB_MODE=online needs WANDB_API_KEY"), "disabled"
    return ExporterDecision(True, f"online to {host_of(base_url)}"), "online"


def _app_settings() -> Any:
    from app.config import settings

    return settings


def evaluate(s: Any | None = None) -> TelemetryDecision:
    """Apply the telemetry policy to settings (the app settings by default)."""
    s = s if s is not None else _app_settings()
    wandb_decision, wandb_mode = _decide_wandb(s)
    return TelemetryDecision(
        mode=s.telemetry_mode,
        langfuse=_decide_langfuse(s),
        wandb=wandb_decision,
        wandb_mode=wandb_mode,
    )


def apply_wandb_env(decision: TelemetryDecision, s: Any | None = None) -> None:
    """Make the W&B SDK obey the decision, whoever imports it.

    ``WANDB_MODE`` is overwritten, not defaulted: a stray ``WANDB_MODE=online``
    in the environment must not outrank the policy. When W&B is not allowed the
    SDK is set to ``disabled``, so it never opens a network connection.
    """
    s = s if s is not None else _app_settings()
    os.environ["WANDB_MODE"] = decision.wandb_mode
    if decision.wandb_mode == "online":
        os.environ["WANDB_API_KEY"] = s.wandb_api_key
        if s.wandb_base_url:
            os.environ["WANDB_BASE_URL"] = s.wandb_base_url


def log_decision(decision: TelemetryDecision) -> None:
    """Log the policy outcome once at startup. Refusals are errors."""
    logger.info("Telemetry mode: %s", decision.mode)
    for name, d in (("Langfuse", decision.langfuse), ("W&B", decision.wandb)):
        if d.refused:
            logger.error("%s export refused by telemetry policy: %s", name, d.reason)
        elif d.enabled:
            logger.info("%s export enabled: %s", name, d.reason)
        else:
            logger.info("%s export disabled: %s", name, d.reason)
