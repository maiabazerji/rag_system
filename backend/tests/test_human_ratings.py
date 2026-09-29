"""Human ratings: stored under DATA_DIR, listed back, filtered by provider."""
from __future__ import annotations

import importlib

from app.config import settings as app_settings
from app.eval import human_ratings


def test_ratings_live_under_the_data_directory():
    # Reloaded because the conftest repoints the module attributes per test.
    module = importlib.reload(human_ratings)
    try:
        assert app_settings.data_path / "human_ratings" == module.RATINGS_DIR
        assert module.RATINGS_FILE.parent == module.RATINGS_DIR
    finally:
        importlib.reload(human_ratings)


def test_save_and_list_through_the_api(client):
    for provider, rating in (("anthropic", 5), ("local", 2), ("anthropic", 4)):
        r = client.post(
            "/eval/human-rate",
            json={"question": "q?", "answer": "a.", "rating": rating, "provider": provider},
        )
        assert r.status_code == 200, r.text
        assert r.json()["id"] and r.json()["created_at"]

    everything = client.get("/eval/human-ratings").json()
    assert [r["rating"] for r in everything] == [5, 2, 4]
    anthropic = client.get("/eval/human-ratings", params={"provider": "anthropic"}).json()
    assert [r["rating"] for r in anthropic] == [5, 4]


def test_blank_lines_are_skipped_and_an_empty_store_lists_nothing():
    assert human_ratings.load_all() == []
    human_ratings.RATINGS_FILE.write_text("\n\n", encoding="utf-8")
    assert human_ratings.load_all() == []
