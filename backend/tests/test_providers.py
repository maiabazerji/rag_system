"""Tests for the Anthropic provider and the retry/circuit-breaker layer.

Two regressions are pinned here: retries used to match only `httpx` exceptions
(so the SDK's own 429s were never retried), and a fresh client was constructed
for every call.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from app.rag.providers import anthropic_provider as ap
from app.rag.providers.base import MissingKeyError, ProviderError
from app.resilience import CircuitBreaker, _default_retryable, _is_retryable_status, with_retry


@pytest.fixture(autouse=True)
def reset_provider_state():
    """Give each test a fresh client and a closed circuit breaker."""
    ap._client = None
    ap._anthropic_breaker = CircuitBreaker(
        failure_threshold=5, recovery_timeout=30.0, service_name="Anthropic"
    )
    yield
    ap._client = None


def _response(text="Generated response text.", input_tokens=100, output_tokens=50):
    resp = MagicMock()
    resp.content = [MagicMock(type="text", text=text)]
    resp.stop_reason = "end_turn"
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


@pytest.fixture
def mock_client():
    """Patch the shared client with an AsyncMock and hand it to the test."""
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_response())
    with patch.object(ap, "get_client", return_value=client):
        yield client


class TestRequireKey:
    def test_raises_when_unset(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY"):
            ap._require_key()

    def test_returns_the_key(self, settings):
        assert ap._require_key() == settings.anthropic_api_key


class TestClientReuse:
    def test_the_client_is_created_once_and_reused(self, settings):
        """A new client per call meant a new TLS handshake per call."""
        with patch.object(ap, "AsyncAnthropic") as ctor:
            ctor.return_value = MagicMock()
            first = ap.get_client()
            second = ap.get_client()

        assert first is second
        assert ctor.call_count == 1

    def test_sdk_retries_are_disabled_so_they_do_not_compound(self, settings):
        """with_retry owns the retry policy; the SDK must not add its own."""
        with patch.object(ap, "AsyncAnthropic") as ctor:
            ctor.return_value = MagicMock()
            ap.get_client()

        assert ctor.call_args.kwargs["max_retries"] == 0
        assert ctor.call_args.kwargs["timeout"] == settings.provider_timeout_seconds

    def test_client_creation_requires_a_key(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        with pytest.raises(MissingKeyError):
            ap.get_client()


@pytest.mark.asyncio
class TestGenerateWithUsage:
    async def test_returns_text_and_usage(self, mock_client):
        result = await ap.generate_with_usage(
            model="claude-sonnet-5", prompt="Test prompt", max_tokens=100
        )
        assert result["text"] == "Generated response text."
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    async def test_passes_a_system_prompt_through(self, mock_client):
        await ap.generate_with_usage(
            model="claude-sonnet-5", prompt="p", system="You are a judge."
        )
        assert mock_client.messages.create.call_args.kwargs["system"] == "You are a judge."

    async def test_omits_system_when_not_given(self, mock_client):
        await ap.generate_with_usage(model="claude-sonnet-5", prompt="p")
        assert "system" not in mock_client.messages.create.call_args.kwargs

    async def test_never_leaks_the_internal_operation_marker(self, mock_client):
        await ap.generate_with_usage(model="claude-sonnet-5", prompt="p")
        assert "_operation" not in mock_client.messages.create.call_args.kwargs

    async def test_handles_a_response_with_no_text_blocks(self, mock_client):
        empty = _response()
        empty.content = []
        mock_client.messages.create = AsyncMock(return_value=empty)

        result = await ap.generate_with_usage(model="claude-sonnet-5", prompt="p")
        assert result["text"] == ""

    async def test_generate_returns_just_the_text(self, mock_client):
        assert await ap.generate(model="claude-sonnet-5", prompt="p") == (
            "Generated response text."
        )


@pytest.mark.asyncio
class TestCircuitBreaker:
    async def test_open_breaker_raises_a_provider_error(self, mock_client):
        ap._anthropic_breaker.state = "open"
        ap._anthropic_breaker.last_failure_time = float("inf")

        with pytest.raises(ProviderError, match="circuit breaker open"):
            await ap.generate_with_usage(model="claude-sonnet-5", prompt="p")

    async def test_success_resets_the_failure_count(self, mock_client):
        ap._anthropic_breaker.failure_count = 3
        await ap.generate_with_usage(model="claude-sonnet-5", prompt="p")
        assert ap._anthropic_breaker.failure_count == 0
        assert ap._anthropic_breaker.state == "closed"

    async def test_failures_are_recorded(self, mock_client, settings, monkeypatch):
        monkeypatch.setattr(settings, "provider_max_retries", 0)
        mock_client.messages.create = AsyncMock(side_effect=ValueError("nope"))

        with pytest.raises(ValueError):
            await ap._create_message(model="claude-sonnet-5", max_tokens=10, messages=[])
        assert ap._anthropic_breaker.failure_count == 1

    async def test_opens_after_the_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, service_name="test")
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.can_execute() is False

    async def test_half_opens_after_the_recovery_window(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        breaker.record_failure()
        assert breaker.can_execute() is True
        assert breaker.state == "half_open"


class TestRetryPolicy:
    def test_anthropic_exceptions_are_retryable(self):
        """The regression: the policy only knew about httpx, so 429s never retried."""
        retryable = _default_retryable()
        assert anthropic.RateLimitError in retryable
        assert anthropic.APIConnectionError in retryable
        assert anthropic.InternalServerError in retryable
        assert ConnectionError in retryable
        assert TimeoutError in retryable

    def test_status_codes_are_read_off_the_exception(self):
        exc = MagicMock()
        exc.status_code = 429
        assert _is_retryable_status(exc, (429, 500)) is True

    def test_status_codes_are_read_off_a_nested_response(self):
        exc = MagicMock(spec=["response"])
        exc.response.status_code = 503
        assert _is_retryable_status(exc, (503,)) is True

    def test_non_retryable_status_is_rejected(self):
        exc = MagicMock()
        exc.status_code = 400
        assert _is_retryable_status(exc, (429, 500)) is False


@pytest.mark.asyncio
class TestWithRetry:
    async def test_retries_a_transient_failure_then_succeeds(self):
        calls = []

        @with_retry(max_retries=2, backoff_factor=1.0, jitter=False)
        async def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("transient")
            return "ok"

        assert await flaky() == "ok"
        assert len(calls) == 2

    async def test_gives_up_after_max_retries(self):
        calls = []

        @with_retry(max_retries=1, backoff_factor=0.0, jitter=False)
        async def always_fails():
            calls.append(1)
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            await always_fails()
        assert len(calls) == 2  # initial attempt plus one retry

    async def test_does_not_retry_a_non_transient_error(self):
        calls = []

        @with_retry(max_retries=3, backoff_factor=0.0, jitter=False)
        async def bad_request():
            calls.append(1)
            raise ValueError("your fault")

        with pytest.raises(ValueError):
            await bad_request()
        assert len(calls) == 1


@pytest.mark.asyncio
class TestToolUseLoop:
    @staticmethod
    def _tool_response(tool_name="search", tool_input=None):
        resp = MagicMock()
        # `name` is consumed by the Mock constructor itself, so it has to be
        # assigned afterwards to land as an attribute.
        block = MagicMock(type="tool_use", id="tu_1")
        block.name = tool_name
        block.input = tool_input or {"query": "test"}
        resp.content = [block]
        resp.stop_reason = "tool_use"
        resp.usage = MagicMock(input_tokens=50, output_tokens=25)
        return resp

    @staticmethod
    def _tool_results_sent(mock_client):
        """Pull the tool_result blocks handed back to the model.

        The `messages` list is mutated in place across the loop, so the
        recorded call args keep growing -- select by shape, not by index.
        """
        messages = mock_client.messages.create.call_args_list[-1].kwargs["messages"]
        for message in messages:
            content = message.get("content")
            if (
                isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and content[0].get("type") == "tool_result"
            ):
                return content
        raise AssertionError("no tool_result blocks were sent back to the model")

    async def test_runs_a_tool_then_finishes(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=[self._tool_response(), _response("Final answer.")]
        )
        handler = AsyncMock(return_value="tool output")

        result = await ap.tool_use_loop(
            model="claude-sonnet-5",
            system="s",
            user_message="u",
            tools=[{"name": "search"}],
            tool_handlers={"search": handler},
            max_iters=5,
        )

        assert result["text"] == "Final answer."
        assert result["iterations"] == 2
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 75
        assert handler.await_count == 1

    async def test_stops_at_the_iteration_limit(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=self._tool_response())

        result = await ap.tool_use_loop(
            model="claude-sonnet-5",
            system="s",
            user_message="u",
            tools=[{"name": "search"}],
            tool_handlers={"search": AsyncMock(return_value="out")},
            max_iters=2,
        )

        assert result["iterations"] == 2
        assert "2-step limit" in result["text"]

    async def test_reports_an_unknown_tool_back_to_the_model(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=[self._tool_response("mystery"), _response("Recovered.")]
        )

        result = await ap.tool_use_loop(
            model="claude-sonnet-5",
            system="s",
            user_message="u",
            tools=[],
            tool_handlers={},
            max_iters=3,
        )

        tool_result = self._tool_results_sent(mock_client)[0]
        assert tool_result["is_error"] is True
        assert "unknown tool" in tool_result["content"]
        assert result["text"] == "Recovered."

    async def test_a_failing_handler_is_surfaced_not_swallowed(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=[self._tool_response(), _response("Handled.")]
        )
        handler = AsyncMock(side_effect=RuntimeError("tool blew up"))

        await ap.tool_use_loop(
            model="claude-sonnet-5",
            system="s",
            user_message="u",
            tools=[{"name": "search"}],
            tool_handlers={"search": handler},
            max_iters=3,
        )

        tool_result = self._tool_results_sent(mock_client)[0]
        assert tool_result["is_error"] is True
        assert "RuntimeError" in tool_result["content"]

    async def test_returns_partial_telemetry_when_a_step_fails(
        self, mock_client, settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "provider_max_retries", 0)
        mock_client.messages.create = AsyncMock(side_effect=ValueError("boom"))

        result = await ap.tool_use_loop(
            model="claude-sonnet-5",
            system="s",
            user_message="u",
            tools=[],
            tool_handlers={},
            max_iters=3,
        )

        assert result["iterations"] == 0
        assert "provider error" in result["text"]


@pytest.mark.asyncio
class TestTimeoutWrapper:
    async def test_raises_timeout_error_past_the_deadline(self):
        from app.resilience import async_timeout_wrapper

        async def slow():
            await asyncio.sleep(1)

        with pytest.raises(TimeoutError):
            await async_timeout_wrapper(slow(), timeout=0.01, service_name="test")

    async def test_returns_the_value_within_the_deadline(self):
        from app.resilience import async_timeout_wrapper

        async def quick():
            return "done"

        assert await async_timeout_wrapper(quick(), timeout=1.0) == "done"
