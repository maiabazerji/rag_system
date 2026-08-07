import json
import logging
import logging.config
from datetime import datetime, timezone
from typing import Any

from app.middleware.request_id import get_request_id


class StructuredJSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs with request ID."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with structured fields."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
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
        """Process log message to extract extra fields."""
        extra_fields = kwargs.pop("extra_fields", {})
        if extra_fields:
            record = kwargs.get("extra", {})
            if not isinstance(record, dict):
                record = {}
            record.extra_fields = extra_fields
            kwargs["extra"] = record
        return msg, kwargs


def get_structured_logger(name: str) -> StructuredLogger:
    """Create a structured logger with request ID support."""
    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger, {})


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "()": "app.logging_config.StructuredJSONFormatter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "structured",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "structured",
            "filename": "logs/evalrag.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "app": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


def setup_logging():
    """Initialize logging configuration."""
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
    except Exception:
        # Fallback to basic config if file handler fails
        logging.basicConfig(level=logging.INFO)
