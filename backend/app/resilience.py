"""Resilience patterns for external service calls.

Provides timeout wrappers, circuit breaker, and retry logic for handling
failures in Qdrant, Postgres, and Anthropic API calls.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from app.logging_config import get_structured_logger

logger = get_structured_logger(__name__)

T = TypeVar("T")


async def async_timeout_wrapper(
    coro: Awaitable[T],
    timeout: float,
    service_name: str = "service",
) -> T:
    """Wrap an async call with timeout.

    Args:
        coro: The coroutine to execute
        timeout: Timeout in seconds
        service_name: Name of the service for logging

    Returns:
        The result of the coroutine

    Raises:
        TimeoutError: If the call exceeds the timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError as e:
        logger.error(
            f"{service_name} call timed out after {timeout}s",
            extra_fields={
                "error_type": "timeout",
                "service": service_name,
                "timeout_seconds": timeout,
            },
        )
        raise TimeoutError(f"{service_name} call timed out after {timeout}s") from e


class CircuitBreaker:
    """Circuit breaker for service calls.

    Tracks failures per service and prevents cascading failures by entering
    an "open" state after a threshold is reached.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        service_name: str = "service",
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting half-open
            service_name: Name of the service
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.service_name = service_name
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open

    def record_success(self) -> None:
        """Record a successful call; reset failure count."""
        self.failure_count = 0
        self.state = "closed"
        logger.debug(
            "Circuit breaker success",
            extra_fields={
                "service": self.service_name,
                "state": "closed",
                "failure_count": 0,
            },
        )

    def record_failure(self) -> None:
        """Record a failed call; track failure count and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "Circuit breaker opened",
                extra_fields={
                    "service": self.service_name,
                    "state": "open",
                    "failure_count": self.failure_count,
                    "failure_threshold": self.failure_threshold,
                },
            )
        else:
            logger.warning(
                "Circuit breaker recording failure",
                extra_fields={
                    "service": self.service_name,
                    "state": "closed",
                    "failure_count": self.failure_count,
                    "failure_threshold": self.failure_threshold,
                },
            )

    def can_execute(self) -> bool:
        """Check if a call can be executed.

        Returns True if circuit is closed or if enough time has passed
        to attempt a half-open state.
        """
        if self.state == "closed":
            return True

        if self.state == "open":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = "half_open"
                logger.info(
                    "Circuit breaker attempting recovery",
                    extra_fields={
                        "service": self.service_name,
                        "state": "half_open",
                        "elapsed_since_failure": elapsed,
                        "recovery_timeout": self.recovery_timeout,
                    },
                )
                return True
            return False

        # half_open: allow one attempt
        return True

    def get_state(self) -> dict:
        """Get current state of the circuit breaker."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
        }


def _default_retryable() -> tuple:
    """Exception types worth retrying, including the Anthropic SDK's own.

    The SDK raises typed errors (`RateLimitError`, `APIStatusError`,
    `APIConnectionError`), never `httpx.HTTPStatusError`, so matching on httpx
    types alone would silently retry nothing.
    """
    types: list[type[BaseException]] = [
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ]
    try:
        import anthropic

        types += [
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ]
    except ImportError:  # pragma: no cover
        pass
    return tuple(types)


def _is_retryable_status(exc: BaseException, statuses: tuple) -> bool:
    """True when the exception carries an HTTP status code worth retrying."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status in statuses


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple | None = None,
    retryable_http_status: tuple = (408, 429, 500, 502, 503, 504),
) -> Callable:
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        backoff_factor: Multiplier for exponential backoff (wait = backoff_factor^attempt)
        jitter: Whether to add random jitter to backoff
        retryable_exceptions: Exception types to retry on
        retryable_http_status: HTTP status codes to retry on

    Returns:
        Decorated function that retries on transient failures
    """
    retryable = _default_retryable() if retryable_exceptions is None else retryable_exceptions

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            attempt = 0

            while attempt <= max_retries:
                try:
                    if attempt > 0:
                        logger.debug(
                            f"Retry attempt {attempt} for {func.__name__}",
                            extra_fields={
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_retries": max_retries,
                            },
                        )
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    is_retryable = isinstance(e, retryable) or _is_retryable_status(
                        e, retryable_http_status
                    )

                    if not is_retryable or attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} failed after {attempt + 1} attempts: {type(e).__name__}: {e}",
                            extra_fields={
                                "error_type": type(e).__name__,
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries + 1,
                                "is_retryable": is_retryable,
                            },
                        )
                        raise

                    # Calculate backoff with optional jitter
                    wait_time = backoff_factor ** attempt
                    if jitter:
                        wait_time = wait_time * (0.5 + random.random())

                    attempt += 1
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_retries + 1} failed: {type(e).__name__}. "
                        f"Retrying in {wait_time:.2f}s...",
                        extra_fields={
                            "error_type": type(e).__name__,
                            "function": func.__name__,
                            "attempt": attempt,
                            "max_retries": max_retries + 1,
                            "backoff_seconds": round(wait_time, 2),
                            "is_retryable": is_retryable,
                        },
                    )
                    await asyncio.sleep(wait_time)

            # Should not reach here, but just in case
            logger.error(
                f"{func.__name__} exhausted all {max_retries} retries",
                extra_fields={
                    "function": func.__name__,
                    "max_retries": max_retries,
                    "error_type": type(last_exception).__name__ if last_exception else "unknown",
                },
            )
            raise last_exception if last_exception else RuntimeError("Retry exhausted")

        return wrapper

    return decorator
