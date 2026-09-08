#!/usr/bin/env python3
"""Manage EvalRAG API keys.

Creates the auth tables if they do not exist, then mints, lists or revokes API
keys. Keys are stored as SHA-256 hashes, so a created key is shown once and
cannot be recovered afterwards.

Requires POSTGRES_URL to point at a reachable database (see .env).

Usage:
    python scripts/setup_auth.py                          # initialise and show status
    python scripts/setup_auth.py --create-key "my-laptop" # mint a key
    python scripts/setup_auth.py --list-keys              # list keys and usage
    python scripts/setup_auth.py --deactivate-key 1       # revoke a key
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `app` importable when running this from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.auth import (  # noqa: E402
    create_api_key,
    deactivate_api_key,
    init_db,
    list_api_keys,
)
from app.config import settings  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage EvalRAG API keys.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Initialise the auth tables and report status
  python scripts/setup_auth.py

  # Mint a key (shown once -- save it)
  python scripts/setup_auth.py --create-key "my-laptop"

  # Mint a key with a higher rate limit
  python scripts/setup_auth.py --create-key "ci" --rpm 60

  # List keys with their 24-hour usage
  python scripts/setup_auth.py --list-keys

  # Revoke a key (takes effect immediately)
  python scripts/setup_auth.py --deactivate-key 1
""",
    )
    parser.add_argument(
        "--create-key", metavar="NAME", help="Mint a new API key with the given name"
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=10,
        metavar="N",
        help="Requests per minute for the new key (default: 10)",
    )
    parser.add_argument(
        "--list-keys", action="store_true", help="List all API keys and their usage"
    )
    parser.add_argument(
        "--deactivate-key",
        type=int,
        metavar="ID",
        help="Deactivate the API key with this id",
    )
    return parser


def _create(name: str, rpm: int) -> None:
    key = create_api_key(name, requests_per_minute=rpm)
    print(f"\nCreated API key '{name}' ({rpm} requests/minute).\n")
    print(f"  {key}\n")
    print("This is the only time the key is shown -- save it now.")
    print("\nUse it from the UI (paste it into the API key field), or directly:")
    print(f"  curl -H 'Authorization: Bearer {key}' http://localhost:8011/ingest/stats")


def _list() -> None:
    keys = list_api_keys()
    if not keys:
        print("\nNo API keys yet.")
        print('  Create one: python scripts/setup_auth.py --create-key "my-laptop"')
        return

    print(f"\n{len(keys)} API key(s):\n")
    for key in keys:
        status = "active" if key["is_active"] else "REVOKED"
        usage = key["usage_24h"]
        print(f"  [{key['id']}] {key['name']}  ({status}, {key['requests_per_minute']} rpm)")
        print(f"      prefix:    {key['key_hint']}...")
        print(f"      created:   {key['created_at'] or 'unknown'}")
        print(f"      last used: {key['last_used'] or 'never'}")
        print(
            f"      last 24h:  {usage['requests']} requests, "
            f"{usage['total_tokens']} tokens"
        )
        print()


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args()

    print(f"Connecting to {settings.postgres_url.rsplit('@', 1)[-1]} ...")
    try:
        init_db()
    except Exception as e:
        print(f"\nCould not reach the auth database: {type(e).__name__}: {e}")
        print("\nCheck that POSTGRES_URL is correct and Postgres is running:")
        print("  docker compose -f infra/docker-compose.yml up -d postgres")
        return 1
    print("Auth tables are ready.")

    try:
        if args.create_key:
            _create(args.create_key, args.rpm)
        if args.list_keys:
            _list()
        if args.deactivate_key is not None:
            if deactivate_api_key(args.deactivate_key):
                print(f"\nKey {args.deactivate_key} revoked. It stops working immediately.")
            else:
                print(f"\nNo API key with id {args.deactivate_key}.")
                return 1
    except Exception as e:
        print(f"\nFailed: {type(e).__name__}: {e}")
        return 1

    if not (args.create_key or args.list_keys or args.deactivate_key is not None):
        print("\nStatus:")
        print(f"  REQUIRE_API_KEY = {settings.require_api_key}")
        print(f"  ADMIN_KEY       = {'set' if settings.admin_key else 'not set'}")
        print(f"  Existing keys   = {len(list_api_keys())}")
        if not settings.require_api_key:
            print(
                "\n  Auth is off, so keys are not checked yet. "
                "Set REQUIRE_API_KEY=true in .env to enforce it."
            )
        print('\nNext: python scripts/setup_auth.py --create-key "my-laptop"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
