"""Regression tests for structured logging.

`StructuredLogger.process` once assigned an attribute onto a plain dict, which
raised AttributeError on every call that passed `extra_fields=`. Because nearly
every code path logs that way, it broke the whole application. These tests pin
the behaviour down.
"""
import json
import logging

import pytest

from app.logging_config import StructuredJSONFormatter, get_structured_logger


@pytest.fixture
def capture(caplog):
    caplog.set_level(logging.DEBUG, logger="app.test")
    return caplog


def test_extra_fields_does_not_raise(capture):
    """The original bug: logging with extra_fields must not raise."""
    logger = get_structured_logger("app.test")
    logger.info("hello", extra_fields={"request_id": "abc", "count": 3})

    assert len(capture.records) == 1
    assert capture.records[0].getMessage() == "hello"


def test_extra_fields_reach_the_record(capture):
    """extra_fields must land on the record where the formatter looks for them."""
    logger = get_structured_logger("app.test")
    logger.warning("with fields", extra_fields={"error_type": "Boom", "attempt": 2})

    record = capture.records[0]
    assert record.extra_fields == {"error_type": "Boom", "attempt": 2}


def test_extra_fields_are_serialized_into_the_json_line(capture):
    """The JSON formatter must merge extra_fields into its output."""
    logger = get_structured_logger("app.test")
    logger.info("serialized", extra_fields={"strategy": "classic", "latency_ms": 42})

    payload = json.loads(StructuredJSONFormatter().format(capture.records[0]))
    assert payload["message"] == "serialized"
    assert payload["level"] == "INFO"
    assert payload["strategy"] == "classic"
    assert payload["latency_ms"] == 42


def test_caller_supplied_extra_is_preserved(capture):
    """A caller passing both `extra` and `extra_fields` keeps both."""
    logger = get_structured_logger("app.test")
    logger.info("both", extra={"custom_attr": "kept"}, extra_fields={"a": 1})

    record = capture.records[0]
    assert record.custom_attr == "kept"
    assert record.extra_fields == {"a": 1}


def test_logging_without_extra_fields_still_works(capture):
    """The common path -- a plain message -- must be untouched."""
    logger = get_structured_logger("app.test")
    logger.info("plain message")

    record = capture.records[0]
    assert record.getMessage() == "plain message"
    assert not hasattr(record, "extra_fields")


def test_every_level_accepts_extra_fields(capture):
    """debug/info/warning/error all route through process()."""
    logger = get_structured_logger("app.test")
    for level in ("debug", "info", "warning", "error"):
        getattr(logger, level)(f"{level} line", extra_fields={"level_name": level})

    assert len(capture.records) == 4
    assert [r.extra_fields["level_name"] for r in capture.records] == [
        "debug",
        "info",
        "warning",
        "error",
    ]


def test_formatter_includes_location_for_warnings(capture):
    """Warnings and above carry a file:line location for debugging."""
    logger = get_structured_logger("app.test")
    logger.error("boom", extra_fields={"error_type": "ValueError"})

    payload = json.loads(StructuredJSONFormatter().format(capture.records[0]))
    assert "location" in payload
    assert "test_logging.py" in payload["location"]
