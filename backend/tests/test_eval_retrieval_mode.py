"""Eval rows record the retrieval mode the strategy reported."""
import pytest

from app.eval.metrics import _retrieval_mode


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({"retrieval": {"mode": "hybrid", "counts": {}}}, "hybrid"),
        ({"retrieval_mode": "sparse", "retrieval": {"mode": "hybrid"}}, "sparse"),
        ({"retrieval": None}, None),  # agentic run that never searched
        ({"retrieval": "not a dict"}, None),
        ({}, None),
    ],
)
def test_retrieval_mode_is_read_from_the_diagnostics(extra, expected):
    assert _retrieval_mode(extra) == expected
