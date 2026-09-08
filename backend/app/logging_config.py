import json
import logging
import logging.config
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.middleware.request_id import get_request_id


class StructuredJSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs with request ID."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with structured fields."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info and record.exc_text:
            log_data["exception"] = record.exc_text

        # Add line number and function info for debugging
        if record.levelno >= logging.WARNING:
            log_data["location"] = f"{record.filename}:{record.lineno} in {record.funcName}"

        return json.dumps(log_data)


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that supports structured logging with extra fields."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        """Move `extra_fields=` into the stdlib `extra` dict.

        Keys of the `extra` dict become attributes on the LogRecord, so passing
        `extra={"extra_fields": {...}}` is what makes `record.extra_fields`
        available to StructuredJSONFormatter.
        """
        extra_fields = kwargs.pop("extra_fields", None)
        if extra_fields:
            supplied = kwargs.get("extra")
            extra = dict(supplied) if isinstance(supplied, dict) else {}
            extra["extra_fields"] = extra_fields
            kwargs["extra"] = extra
        return msg, kwargs


def get_structured_logger(name: str) -> StructuredLogger:
    """Create a structured logger with request ID support."""
    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger, {})


LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _build_config(file_logging: bool) -> dict:
    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "structured",
            "stream": "ext://sys.stdout",
        },
    }
    app_handlers = ["console"]

    if file_logging:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "structured",
            "filename": str(LOG_DIR / "evalrag.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        app_handlers.append("file")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {"()": "app.logging_config.StructuredJSONFormatter"},
        },
        "handlers": handlers,
        "loggers": {
            "app": {
                "level": "DEBUG" if file_logging else LOG_LEVEL,
                "handlers": app_handlers,
                "propagate": False,
            },
        },
        "root": {"level": LOG_LEVEL, "handlers": ["console"]},
    }


def setup_logging() -> None:
    """Initialize logging.

    Console logging is always configured. File logging is added only when the
    log directory is writable, so a read-only or missing directory degrades to
    console-only instead of silently dropping the structured formatter.
    """
    file_logging = False
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_logging = True
    except OSError:
        pass

    try:
        logging.config.dictConfig(_build_config(file_logging))
    except Exception:
        logging.config.dictConfig(_build_config(file_logging=False))
        logging.getLogger(__name__).warning(
            "File logging unavailable; falling back to console-only logging."
        )
