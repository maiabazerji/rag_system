"""The OpenAI and Ollama providers: request shape, usage parsing and failures.

Neither service is reachable from the tests. Ollama is served by an
``httpx.MockTransport``, so the real HTTP client builds and parses the
requests; the optional ``openai`` SDK is replaced by a stub module.
"""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.rag.providers import MissingKeyError, ProviderError, base, local, openai_provider
from app.tracing import start_trace

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama(monkeypatch, settings):
    """Route the local provider's HTTP client to a scripted handler."""
    monkeypatch.setattr(settings, "ollama_host", "http://ollama.test:11434/")
    state = SimpleNamespace(requests=[], respond=None)
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        state.requests.append(request)
        return state.respond(request)

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(local.httpx, "AsyncClient", client)
    return state


class TestOllama:
    async def test_request_shape_and_usage(self, ollama):
        ollama.respond = lambda r: httpx.Response(
            200, json={"response": "  Bonjour.  ", "prompt_eval_count": 31, "eval_count": 4}
        )
        out = await local.generate_with_usage(
            model="llama3", prompt="Salut", max_tokens=64, system="Be brief.", temperature=0.2
        )

        assert out == {"text": "Bonjour.", "input_tokens": 31, "output_tokens": 4}
        (request,) = ollama.requests
        assert str(request.url) == "http://ollama.test:11434/api/generate"
        assert json.loads(request.content) == {
            "model": "llama3",
            "prompt": "Salut",
            "stream": False,
            "options": {"num_predict": 64, "temperature": 0.2},
            "system": "Be brief.",
        }

    async def test_optional_fields_are_omitted_and_missing_counts_are_zero(self, ollama):
        ollama.respond = lambda r: httpx.Response(200, json={"response": None})
        out = await local.generate_with_usage(model="llama3", prompt="p")
        body = json.loads(ollama.requests[0].content)
        assert "system" not in body and "temperature" not in body["options"]
        assert out == {"text": "", "input_tokens": 0, "output_tokens": 0}

    async def test_generate_returns_the_text(self, ollama):
        ollama.respond = lambda r: httpx.Response(200, json={"response": "ok"})
        assert await local.generate(model="llama3", prompt="p") == "ok"

    @pytest.mark.parametrize(
        "respond",
        [
            lambda r: httpx.Response(500, json={"error": "model not found"}),
            lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)),
        ],
        ids=["http-error", "unreachable"],
    )
    async def test_transport_failures_are_provider_errors(self, ollama, respond):
        ollama.respond = respond
        with pytest.raises(ProviderError, match="unreachable or slow"):
            await local.generate_with_usage(model="llama3", prompt="p")

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="<html>502 Bad Gateway</html>"),
            httpx.Response(200, json=["not", "an", "object"]),
        ],
        ids=["html", "json-array"],
    )
    async def test_a_reply_that_is_not_a_json_object_is_a_provider_error(self, ollama, response):
        ollama.respond = lambda r: response
        with pytest.raises(ProviderError, match="not a JSON object"):
            await local.generate_with_usage(model="llama3", prompt="p")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _completion(content, prompt_tokens=12, completion_tokens=3, usage=True):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=(
            SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            if usage
            else None
        ),
    )


@pytest.fixture
def openai_sdk(monkeypatch, settings):
    """Install a stub ``openai`` module; return its call log."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai-test")
    state = SimpleNamespace(clients=[], create=AsyncMock(return_value=_completion(" Hi. ")))

    class AsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=state.create))
            state.clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True
            return False

    module = types.ModuleType("openai")
    module.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return state


class TestOpenAI:
    async def test_request_shape_and_usage(self, openai_sdk, settings):
        out = await openai_provider.generate_with_usage(
            model="gpt-4o-mini", prompt="Hello", max_tokens=50, system="Be brief.", temperature=0
        )

        assert out == {"text": "Hi.", "input_tokens": 12, "output_tokens": 3}
        openai_sdk.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            max_tokens=50,
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
            ],
            temperature=0,
        )
        (client,) = openai_sdk.clients
        assert client.kwargs == {
            "api_key": "sk-openai-test",
            "timeout": settings.provider_timeout_seconds,
        }
        assert client.closed

    async def test_no_system_no_temperature_no_usage(self, openai_sdk):
        openai_sdk.create.return_value = _completion(None, usage=False)
        out = await openai_provider.generate_with_usage(model="m", prompt="p")
        kwargs = openai_sdk.create.await_args.kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "p"}]
        assert "temperature" not in kwargs
        assert out == {"text": "", "input_tokens": 0, "output_tokens": 0}

    async def test_generate_returns_the_text(self, openai_sdk):
        assert await openai_provider.generate(model="m", prompt="p") == "Hi."

    async def test_missing_key(self, openai_sdk, settings, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "")
        with pytest.raises(MissingKeyError, match="OPENAI_API_KEY"):
            await openai_provider.generate_with_usage(model="m", prompt="p")
        openai_sdk.create.assert_not_awaited()

    async def test_sdk_errors_become_provider_errors_without_details(self, openai_sdk):
        openai_sdk.create.side_effect = RuntimeError("401 key sk-openai-test rejected")
        with pytest.raises(ProviderError) as info:
            await openai_provider.generate_with_usage(model="m", prompt="p")
        assert str(info.value) == "OpenAI request failed (RuntimeError)."


# ---------------------------------------------------------------------------
# The dispatcher in front of them
# ---------------------------------------------------------------------------


class TestDispatch:
    async def test_non_anthropic_calls_are_traced_and_priced_like_anthropic(self, openai_sdk):
        with start_trace("t", {}) as trace:
            text = await base.generate("openai", model="gpt-4o-mini", prompt="p", max_tokens=9)
        assert text == "Hi."
        (span,) = [s for s in trace.spans if s.as_type == "generation"]
        assert span.model == "gpt-4o-mini"
        assert (span.usage["input"], span.usage["output"]) == (12, 3)
        assert span.metadata["max_tokens"] == 9

    async def test_local_dispatch(self, ollama):
        ollama.respond = lambda r: httpx.Response(200, json={"response": "ok", "eval_count": 2})
        out = await base.generate_with_usage("local", model="llama3", prompt="p")
        assert out["text"] == "ok" and out["output_tokens"] == 2

    async def test_structured_request_to_other_providers_is_a_json_instruction(self, ollama):
        ollama.respond = lambda r: httpx.Response(200, json={"response": '{"answer": "x"}'})
        tool = {"name": "submit_answer", "input_schema": {"type": "object"}}
        out = await base.generate_structured(
            "local", model="llama3", prompt="p", tool=tool, system="Cite sources."
        )
        assert out["structured"] is None and out["text"] == '{"answer": "x"}'
        system = json.loads(ollama.requests[0].content)["system"]
        assert system.startswith("Cite sources.\n\nReply with a single JSON object")

    async def test_unknown_provider(self):
        with pytest.raises(ProviderError, match="unknown generator provider"):
            await base.generate_with_usage("mistral", model="m", prompt="p")
