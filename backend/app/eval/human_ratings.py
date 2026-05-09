"""Persistent storage for human ratings as JSONL.

One file, append-only. Cheap, inspectable, and easy to swap for Postgres later.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import HumanRating, HumanRatingRequest

RATINGS_DIR = Path(__file__).resolve().parents[3] / "data" / "human_ratings"
RATINGS_FILE = RATINGS_DIR / "ratings.jsonl"


def _ensure_file() -> None:
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    RATINGS_FILE.touch(exist_ok=True)


def save(req: HumanRatingRequest) -> HumanRating:
    _ensure_file()
    record = HumanRating(
        id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **req.model_dump(),
    )
    with RATINGS_FILE.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    return record


def load_all(provider: str | None = None) -> list[HumanRating]:
    _ensure_file()
    out: list[HumanRating] = []
    for line in RATINGS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = HumanRating.model_validate_json(line)
        if provider and rec.provider != provider:
            continue
        out.append(rec)
    return out
