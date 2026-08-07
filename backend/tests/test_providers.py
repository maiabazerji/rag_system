import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.rag.providers.anthropic_provider import (
    generate,
    generate_with_usage,
    tool_use_loop,
    _anthropic_breaker,
)
from app.rag.providers.base import MissingKeyError
from app.resilience import CircuitBreaker


@pytest.mark.asyncio
class TestAnthropicGenerate:
    async def test_generate_success(self):
        """Test successful text generation."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = [MagicMock(type="text", text="Generated response text.")]
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    result = await generate(
                        model="claude-sonnet-4-6",
                        prompt="Test prompt",
                        max_tokens=100,
                    )

                    assert result == "Generated response text."
                    mock_breaker.record_success.assert_called_once()

    async def test_generate_missing_api_key(self):
        """Test generate with missing API key."""
        with patch("app.rag.providers.anthropic_provider._require_key", side_effect=MissingKeyError("Key missing")):
            with pytest.raises(MissingKeyError):
                await generate(
                    model="claude-sonnet-4-6",
                    prompt="Test prompt",
                )

    async def test_generate_api_timeout(self):
        """Test generate with API timeout."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    with pytest.raises(asyncio.TimeoutError):
                        await generate(
                            model="claude-sonnet-4-6",
                            prompt="Test prompt",
                        )

                    mock_breaker.record_failure.assert_called()

    async def test_generate_api_error(self):
        """Test generate with API error."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=RuntimeError("API Error"))
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    with pytest.raises(RuntimeError):
                        await generate(
                            model="claude-sonnet-4-6",
                            prompt="Test prompt",
                        )

                    mock_breaker.record_failure.assert_called()

    async def test_generate_circuit_breaker_open(self):
        """Test generate when circuit breaker is open."""
        with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
            mock_breaker.can_execute.return_value = False
            with pytest.raises(RuntimeError, match="circuit breaker open"):
                await generate(
                    model="claude-sonnet-4-6",
                    prompt="Test prompt",
                )


@pytest.mark.asyncio
class TestGenerateWithUsage:
    async def test_generate_with_usage_success(self):
        """Test generation with usage tracking."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = [MagicMock(type="text", text="Response text.")]
                mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    result = await generate_with_usage(
                        model="claude-sonnet-4-6",
                        prompt="Test prompt",
                    )

                    assert result["text"] == "Response text."
                    assert result["input_tokens"] == 100
                    assert result["output_tokens"] == 50
                    mock_breaker.record_success.assert_called_once()

    async def test_generate_with_usage_system_message(self):
        """Test generation with system message."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = [MagicMock(type="text", text="Response.")]
                mock_response.usage = MagicMock(input_tokens=50, output_tokens=25)
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    result = await generate_with_usage(
                        model="claude-sonnet-4-6",
                        prompt="User message",
                        system="System prompt",
                    )

                    # Verify system message was passed
                    call_kwargs = mock_client.messages.create.call_args[1]
                    assert call_kwargs["system"] == "System prompt"
                    assert result["text"] == "Response."

    async def test_generate_with_usage_timeout(self):
        """Test generate_with_usage timeout."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    with pytest.raises(TimeoutError):
                        await generate_with_usage(
                            model="claude-sonnet-4-6",
                            prompt="Test prompt",
                        )

    async def test_generate_with_usage_empty_response(self):
        """Test generate_with_usage with empty response."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = []
                mock_response.usage = MagicMock(input_tokens=50, output_tokens=0)
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    result = await generate_with_usage(
                        model="claude-sonnet-4-6",
                        prompt="Test prompt",
                    )

                    assert result["text"] == ""
                    assert result["output_tokens"] == 0


@pytest.mark.asyncio
class TestToolUseLoop:
    async def test_tool_use_loop_success(self):
        """Test tool-use loop that completes successfully."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()

                # First response: tool use
                resp1 = MagicMock()
                resp1.stop_reason = "tool_use"
                resp1.content = [
                    MagicMock(type="text", text="I'll search for this."),
                    MagicMock(type="tool_use", name="search", id="tool_0", input={"query": "test"}),
                ]
                resp1.usage = MagicMock(input_tokens=100, output_tokens=50)

                # Second response: end
                resp2 = MagicMock()
                resp2.stop_reason = "end_turn"
                resp2.content = [MagicMock(type="text", text="Final answer based on search.")]
                resp2.usage = MagicMock(input_tokens=100, output_tokens=75)

                mock_client.messages.create = AsyncMock(side_effect=[resp1, resp2])
                mock_client_class.return_value.__aenter__.return_value = mock_client

                async def mock_search_handler(args):
                    return "Search results."

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    result = await tool_use_loop(
                        model="claude-sonnet-4-6",
                        system="You are helpful.",
                        user_message="Search for something.",
                        tools=[{"name": "search", "input_schema": {"type": "object"}}],
                        tool_handlers={"search": mock_search_handler},
                        max_iters=5,
                    )

                    assert "Final answer based on search." in result["text"]
                    assert result["iterations"] == 2
                    assert result["input_tokens"] == 200
                    assert result["output_tokens"] == 125
                    assert len(result["trace"]) == 2
                    mock_breaker.record_success.assert_called_once()

    async def test_tool_use_loop_max_iterations(self):
        """Test tool-use loop hitting max iterations."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()

                # Always respond with tool_use
                resp = MagicMock()
                resp.stop_reason = "tool_use"
                resp.content = [MagicMock(type="tool_use", name="search", id="tool_0", input={"query": "q"})]
                resp.usage = MagicMock(input_tokens=100, output_tokens=50)

                mock_client.messages.create = AsyncMock(return_value=resp)
                mock_client_class.return_value.__aenter__.return_value = mock_client

                async def mock_handler(args):
                    return "Result"

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    result = await tool_use_loop(
                        model="claude-sonnet-4-6",
                        system="You are helpful.",
                        user_message="Search",
                        tools=[{"name": "search"}],
                        tool_handlers={"search": mock_handler},
                        max_iters=2,
                    )

                    assert "exceeded max iterations" in result["text"]
                    assert result["iterations"] == 2

    async def test_tool_use_loop_unknown_tool(self):
        """Test tool-use loop with unknown tool."""
        with patch("app.rag.providers.anthropic_provider._require_key", return_value="test-key"):
            with patch("app.rag.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()

                resp1 = MagicMock()
                resp1.stop_reason = "tool_use"
                resp1.content = [MagicMock(type="tool_use", name="unknown_tool", id="tool_0", input={})]
                resp1.usage = MagicMock(input_tokens=100, output_tokens=50)

                resp2 = MagicMock()
                resp2.stop_reason = "end_turn"
                resp2.content = [MagicMock(type="text", text="Done")]
                resp2.usage = MagicMock(input_tokens=100, output_tokens=25)

                mock_client.messages.create = AsyncMock(side_effect=[resp1, resp2])
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
                    mock_breaker.can_execute.return_value = True
                    result = await tool_use_loop(
                        model="claude-sonnet-4-6",
                        system="You are helpful.",
                        user_message="Search",
                        tools=[],
                        tool_handlers={},
                        max_iters=5,
                    )

                    assert "error: unknown tool" in result["trace"][0]["tool_calls"][0].get("input", "")

    async def test_tool_use_loop_circuit_breaker_open(self):
        """Test tool-use loop when circuit breaker is open."""
        with patch("app.rag.providers.anthropic_provider._anthropic_breaker") as mock_breaker:
            mock_breaker.can_execute.return_value = False
            result = await tool_use_loop(
                model="claude-sonnet-4-6",
                system="You are helpful.",
                user_message="Search",
                tools=[],
                tool_handlers={},
            )

            assert "circuit breaker open" in result["text"]
            assert result["iterations"] == 0


@pytest.mark.asyncio
class TestCircuitBreaker:
    async def test_circuit_breaker_success_resets(self):
        """Test circuit breaker resets on success."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2
        breaker.record_success()
        assert breaker.failure_count == 0
        assert breaker.state == "closed"

    async def test_circuit_breaker_opens_on_threshold(self):
        """Test circuit breaker opens after threshold."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"
        assert not breaker.can_execute()

    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open recovery."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

        import time
        time.sleep(0.1)  # Wait for recovery timeout

        assert breaker.can_execute()
        assert breaker.state == "half_open"

    async def test_circuit_breaker_get_state(self):
        """Test circuit breaker state reporting."""
        breaker = CircuitBreaker(failure_threshold=5)
        breaker.record_failure()
        breaker.record_failure()

        state = breaker.get_state()
        assert state["failure_count"] == 2
        assert state["failure_threshold"] == 5
        assert state["state"] == "closed"
