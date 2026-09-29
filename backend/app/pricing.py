"""Per-model prices and request cost estimates.

Prices live in a checked-in TOML file, ``config/model_pricing.toml`` at the
repository root, in US dollars per million input and output tokens. A
deployment can point ``MODEL_PRICING_PATH`` at its own copy.

An estimate is only ever made from a listed price: a model without a row (an
OpenAI or Ollama model, a newly released Claude model) gets ``None``, never a
guess. Estimates use standard list rates and leave out prompt caching, batch
discounts and partner-platform pricing; they are for comparing requests and
strategies, not for reconciling an invoice.
"""
from __future__ import annotations

import logging
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

TOKENS_PER_UNIT = 1_000_000
PRICING_FILENAME = "model_pricing.toml"
# The repository checkout (backend/app/pricing.py -> repo) and, in a container
# where app/ sits at /app/app, the directory above the package.
_DEFAULT_LOCATIONS = (
    Path(__file__).resolve().parents[2] / "config" / PRICING_FILENAME,
    Path(__file__).resolve().parents[1] / "config" / PRICING_FILENAME,
)


@dataclass(frozen=True)
class ModelPrice:
    """List price of one model, in USD per million tokens."""

    input: float
    output: float


@dataclass(frozen=True)
class PriceTable:
    """The loaded price table and where it came from.

    Attributes:
        models: Model ID to price. Aliases are resolved into their own entries.
        source: Where the prices were taken from (the file's ``meta.source``).
        as_of: Date the prices were last checked (``meta.as_of``).
        path: The file the table was read from, or None when none was found.
    """

    models: dict[str, ModelPrice] = field(default_factory=dict)
    source: str | None = None
    as_of: str | None = None
    path: Path | None = None

    def price(self, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        return self.models.get(model)


def pricing_path() -> Path | None:
    """The price file to load: MODEL_PRICING_PATH, else the checked-in one."""
    if settings.model_pricing_path:
        return Path(settings.model_pricing_path)
    return next((p for p in _DEFAULT_LOCATIONS if p.is_file()), None)


def _price(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def parse_price_table(data: dict[str, Any], path: Path | None = None) -> PriceTable:
    """Build a :class:`PriceTable` from parsed TOML, skipping malformed rows."""
    models: dict[str, ModelPrice] = {}
    for model, row in (data.get("models") or {}).items():
        row = row if isinstance(row, dict) else {}
        inp, out = _price(row.get("input")), _price(row.get("output"))
        if inp is None or out is None:
            logger.warning("Ignoring price entry for %s: input/output must be numbers >= 0", model)
            continue
        models[str(model)] = ModelPrice(input=inp, output=out)
    for alias, target in (data.get("aliases") or {}).items():
        if isinstance(target, str) and target in models:
            models.setdefault(str(alias), models[target])
        else:
            logger.warning("Ignoring price alias %s: %r has no price", alias, target)
    meta = data.get("meta") or {}
    return PriceTable(
        models=models,
        source=str(meta["source"]) if meta.get("source") else None,
        as_of=str(meta["as_of"]) if meta.get("as_of") else None,
        path=path,
    )


def load_price_table(path: Path | None = None) -> PriceTable:
    """Read a price file. A missing or invalid file gives an empty table.

    Cost estimation is observability, so a bad price file must never fail a
    request: it is logged and every estimate becomes None.
    """
    path = path if path is not None else pricing_path()
    if path is None:
        logger.warning("No model price file found; request costs will be null")
        return PriceTable()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.warning("Could not read model prices from %s: %s", path, e)
        return PriceTable(path=path)
    return parse_price_table(data, path)


_table: PriceTable | None = None
_lock = threading.Lock()


def price_table() -> PriceTable:
    """The process-wide price table, loaded on first use."""
    global _table
    if _table is None:
        with _lock:
            if _table is None:
                _table = load_price_table()
    return _table


def reload_price_table() -> PriceTable:
    """Drop the cached table (after MODEL_PRICING_PATH changes) and reload it."""
    global _table
    with _lock:
        _table = None
    return price_table()


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD cost of one call, or None when ``model`` has no listed price.

    Args:
        model: The model that served the call. Judge and extraction calls are
            priced with their own model, not the request's generator.
        input_tokens: Input tokens billed.
        output_tokens: Output tokens billed.
    """
    price = price_table().price(model)
    if price is None:
        return None
    cost = (
        max(0, input_tokens) * price.input + max(0, output_tokens) * price.output
    ) / TOKENS_PER_UNIT
    return round(cost, 8)
