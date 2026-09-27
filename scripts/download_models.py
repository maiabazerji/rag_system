#!/usr/bin/env python3
"""Download the embedding and reranker models into the Hugging Face cache.

Compose bind-mounts ``.hf_cache/`` into the backend container. Pre-filling it
lets the backend start with ``HF_HUB_OFFLINE=1``, which skips Hugging Face's
online "check for updates" call on every model load. Offline mode is opt-in:
with the default ``HF_HUB_OFFLINE=0`` the backend downloads on first use, and
this script is only a way to do that ahead of time.

Model ids come from, in order: the ``EMBEDDING_MODEL`` / ``RERANKER_MODEL``
environment variables, then the defaults in ``backend/app/config.py``.

Usage:
    # On the host, into the directory Compose mounts:
    pip install huggingface_hub
    python scripts/download_models.py --cache-dir .hf_cache

    # Or inside the running backend container (its cache is the same mount):
    docker compose --env-file .env -f infra/docker-compose.yml \\
        exec backend python /scripts/download_models.py
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# /scripts is mounted into the container next to /app, so look in both places.
CONFIG_CANDIDATES = (
    REPO_ROOT / "backend" / "app" / "config.py",
    Path("/app/app/config.py"),
)
SETTINGS = {"embedding_model": "EMBEDDING_MODEL", "reranker_model": "RERANKER_MODEL"}


def _config_defaults() -> dict[str, str]:
    """Read the ``Field(default=...)`` values for the model settings.

    Parsed with ``ast`` instead of imported, so this runs on a host without the
    backend's dependencies installed.
    """
    for path in CONFIG_CANDIDATES:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: dict[str, str] = {}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in SETTINGS
                and isinstance(node.value, ast.Call)
            ):
                continue
            for kw in node.value.keywords:
                if (
                    kw.arg == "default"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    found[node.target.id] = kw.value.value
        if found:
            return found
    return {}


def resolve_models() -> list[str]:
    """Return the model ids to download, environment first, then config defaults."""
    defaults = _config_defaults()
    models: list[str] = []
    for field, env_name in SETTINGS.items():
        model = os.environ.get(env_name) or defaults.get(field)
        if not model:
            raise SystemExit(
                f"Could not determine {env_name}: set it in the environment or check "
                "backend/app/config.py."
            )
        models.append(model)
    return models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the embedding and reranker models into the HF cache."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Hugging Face cache root (the directory that holds hub/). Defaults to "
            "HF_HOME, or ~/.cache/huggingface. Use .hf_cache on the host for Compose."
        ),
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed: pip install huggingface_hub", file=sys.stderr)
        return 1

    if os.environ.get("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}:
        print(
            "HF_HUB_OFFLINE is set, so nothing can be downloaded. Unset it for this run.",
            file=sys.stderr,
        )
        return 1

    cache_dir: Path | None = args.cache_dir
    hub_dir = str(cache_dir.resolve() / "hub") if cache_dir else None

    for model in resolve_models():
        print(f"Downloading {model} ...", flush=True)
        path = snapshot_download(repo_id=model, cache_dir=hub_dir)
        print(f"  -> {path}")

    print("Done. Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in .env to run offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
