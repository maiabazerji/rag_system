"""Model price table and cost estimates."""
import pytest

from app import pricing


@pytest.fixture(autouse=True)
def _fresh_table(settings):
    pricing.reload_price_table()
    yield
    settings.model_pricing_path = ""
    pricing.reload_price_table()


def test_checked_in_table_is_found_and_documented():
    table = pricing.price_table()
    assert table.path is not None and table.path.name == "model_pricing.toml"
    assert table.source and table.as_of


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # 1M input at $4 + 1M output at $20
        ("claude-opus-5-5", 24.0),
        ("claude-sonnet-5", 12.0),
        ("claude-fable-5-1", 60.0),
        ("claude-haiku-4-5", 6.0),
        # The dated snapshot bills like its alias.
        ("claude-haiku-4-5-20251001", 6.0),
    ],
)
def test_known_model_is_priced(model, expected):
    assert pricing.estimate_cost(model, 1_000_000, 1_000_000) == pytest.approx(expected)


def test_small_request_cost():
    # 1,500 input tokens at $2/MTok + 500 output at $10/MTok
    assert pricing.estimate_cost("claude-sonnet-5", 1500, 500) == pytest.approx(0.008)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "llama3.1:8b", "claude-unreleased-9", "", None])
def test_unknown_model_has_no_cost(model):
    assert pricing.estimate_cost(model, 1000, 1000) is None


def test_zero_tokens_cost_nothing():
    assert pricing.estimate_cost("claude-sonnet-5", 0, 0) == 0.0


def test_override_file(tmp_path, settings):
    path = tmp_path / "prices.toml"
    path.write_text(
        '[meta]\nsource = "negotiated"\nas_of = "2026-01-01"\n'
        '[models."claude-sonnet-5"]\ninput = 1.0\noutput = 2.0\n'
        '[models."broken"]\ninput = "free"\noutput = 1.0\n'
        '[aliases]\n"sonnet-alias" = "claude-sonnet-5"\n"dangling" = "nope"\n'
    )
    settings.model_pricing_path = str(path)
    table = pricing.reload_price_table()

    assert table.source == "negotiated"
    assert pricing.estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(3.0)
    assert pricing.estimate_cost("sonnet-alias", 1_000_000, 0) == pytest.approx(1.0)
    # Malformed rows and dangling aliases are skipped, not guessed.
    assert pricing.estimate_cost("broken", 1, 1) is None
    assert pricing.estimate_cost("dangling", 1, 1) is None
    # Models only in the default file are unknown to the override.
    assert pricing.estimate_cost("claude-opus-5-5", 1, 1) is None


def test_missing_or_invalid_file_yields_no_costs(tmp_path, settings):
    settings.model_pricing_path = str(tmp_path / "absent.toml")
    assert pricing.reload_price_table().models == {}
    assert pricing.estimate_cost("claude-sonnet-5", 10, 10) is None

    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml")
    settings.model_pricing_path = str(bad)
    assert pricing.reload_price_table().models == {}
