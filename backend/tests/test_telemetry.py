"""Telemetry policy, span recording and the Langfuse exporter.

The policy is the compliance boundary: these tests pin that nothing is exported
unless the mode, the host allow-list and (for cloud) the opt-in all agree.
"""
import base64
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.tracing import langfuse_exporter as lf
from app.tracing import policy, start_trace
from app.tracing.spans import span, traced


@pytest.fixture
def langfuse_keys(settings, monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://langfuse:3000")
    return settings


@pytest.fixture
def restore_redactor():
    yield
    lf.set_redactor(None)


class _Recorder:
    """An httpx transport that records requests and answers with ``status``."""

    def __init__(self, status: int = 200):
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json={})

    @property
    def spans(self) -> list[dict]:
        out = []
        for r in self.requests:
            body = json.loads(r.content)
            for rs in body["resourceSpans"]:
                for ss in rs["scopeSpans"]:
                    out.extend(ss["spans"])
        return out


def _attrs(span_json: dict) -> dict:
    """OTLP attribute list -> plain dict."""
    out = {}
    for a in span_json["attributes"]:
        (v,) = a["value"].values()
        out[a["key"]] = v
    return out


# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------


class TestLangfusePolicy:
    def test_off_by_default_even_with_keys(self, langfuse_keys):
        d = policy.evaluate(langfuse_keys).langfuse
        assert d.enabled is False
        assert "TELEMETRY_MODE=off" in d.reason

    def test_not_configured_is_not_a_refusal(self, settings):
        d = policy.evaluate(settings).langfuse
        assert d.enabled is False
        assert d.refused is False

    def test_self_hosted_allows_listed_hosts(self, langfuse_keys, monkeypatch):
        monkeypatch.setattr(langfuse_keys, "telemetry_mode", "self_hosted")
        for host in ("http://langfuse:3000", "http://localhost:3100", "http://127.0.0.1:3100"):
            monkeypatch.setattr(langfuse_keys, "langfuse_host", host)
            assert policy.evaluate(langfuse_keys).langfuse.enabled, host

    def test_self_hosted_refuses_other_hosts(self, langfuse_keys, monkeypatch):
        monkeypatch.setattr(langfuse_keys, "telemetry_mode", "self_hosted")
        monkeypatch.setattr(langfuse_keys, "langfuse_host", "https://cloud.langfuse.com")
        d = policy.evaluate(langfuse_keys).langfuse
        assert d.enabled is False
        assert d.refused is True
        assert "cloud.langfuse.com" in d.reason

    def test_allow_list_is_exact_not_suffix(self):
        assert not policy.is_host_allowed("http://langfuse.evil.com", ["langfuse"])
        assert not policy.is_host_allowed("http://evil-localhost", ["localhost"])
        assert policy.is_host_allowed("http://LANGFUSE:3000", ["langfuse"])

    def test_custom_allow_list(self, langfuse_keys, monkeypatch):
        monkeypatch.setattr(langfuse_keys, "telemetry_mode", "self_hosted")
        monkeypatch.setattr(langfuse_keys, "telemetry_allowed_hosts", "lf.internal.example.eu")
        monkeypatch.setattr(langfuse_keys, "langfuse_host", "https://lf.internal.example.eu")
        assert policy.evaluate(langfuse_keys).langfuse.enabled

    def test_cloud_requires_opt_in(self, langfuse_keys, monkeypatch):
        monkeypatch.setattr(langfuse_keys, "telemetry_mode", "cloud")
        monkeypatch.setattr(langfuse_keys, "langfuse_host", "https://cloud.langfuse.com")
        d = policy.evaluate(langfuse_keys).langfuse
        assert d.enabled is False and d.refused is True
        assert "TELEMETRY_CLOUD_OPT_IN" in d.reason

        monkeypatch.setattr(langfuse_keys, "telemetry_cloud_opt_in", True)
        assert policy.evaluate(langfuse_keys).langfuse.enabled

    def test_refusal_is_logged_as_an_error(self, langfuse_keys, monkeypatch):
        monkeypatch.setattr(langfuse_keys, "telemetry_mode", "self_hosted")
        monkeypatch.setattr(langfuse_keys, "langfuse_host", "https://example.com")
        with patch.object(policy.logger, "error") as error:
            policy.log_decision(policy.evaluate(langfuse_keys))
        error.assert_called_once()
        assert error.call_args.args[1] == "Langfuse"

    def test_invalid_mode_is_rejected(self):
        from app.config import Settings

        with pytest.raises(ValueError, match="TELEMETRY_MODE"):
            Settings(telemetry_mode="everywhere", _env_file=None)
        assert Settings(telemetry_mode="Self-Hosted", _env_file=None).telemetry_mode == "self_hosted"


class TestWandbPolicy:
    def test_disabled_unless_opted_in(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "wandb_api_key", "key")
        monkeypatch.setattr(settings, "wandb_mode", "online")
        monkeypatch.setattr(settings, "telemetry_mode", "cloud")
        monkeypatch.setattr(settings, "telemetry_cloud_opt_in", True)
        d = policy.evaluate(settings)
        assert d.wandb.enabled is False
        assert d.wandb_mode == "disabled"

    def test_offline_is_allowed_in_every_mode(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "wandb_enabled", True)
        monkeypatch.setattr(settings, "wandb_mode", "offline")
        for mode in ("off", "self_hosted", "cloud"):
            monkeypatch.setattr(settings, "telemetry_mode", mode)
            d = policy.evaluate(settings)
            assert d.wandb.enabled and d.wandb_mode == "offline", mode

    def test_online_is_refused_when_off(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "wandb_enabled", True)
        monkeypatch.setattr(settings, "wandb_api_key", "key")
        monkeypatch.setattr(settings, "wandb_mode", "online")
        d = policy.evaluate(settings)
        assert d.wandb.refused and d.wandb_mode == "disabled"

    def test_online_self_hosted_needs_an_allowed_server(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "wandb_enabled", True)
        monkeypatch.setattr(settings, "wandb_api_key", "key")
        monkeypatch.setattr(settings, "wandb_mode", "online")
        monkeypatch.setattr(settings, "telemetry_mode", "self_hosted")
        d = policy.evaluate(settings)
        assert d.wandb.refused and "api.wandb.ai" in d.wandb.reason

        monkeypatch.setattr(settings, "wandb_base_url", "http://localhost:8080")
        d = policy.evaluate(settings)
        assert d.wandb.enabled and d.wandb_mode == "online"

    def test_online_cloud_needs_opt_in(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "wandb_enabled", True)
        monkeypatch.setattr(settings, "wandb_api_key", "key")
        monkeypatch.setattr(settings, "wandb_mode", "online")
        monkeypatch.setattr(settings, "telemetry_mode", "cloud")
        assert not policy.evaluate(settings).wandb.enabled
        monkeypatch.setattr(settings, "telemetry_cloud_opt_in", True)
        assert policy.evaluate(settings).wandb.enabled

    def test_sdk_is_forced_to_disabled(self, settings, monkeypatch):
        """A stray WANDB_MODE=online in the environment must not win."""
        monkeypatch.setenv("WANDB_MODE", "online")
        policy.apply_wandb_env(policy.evaluate(settings), settings)
        assert os.environ["WANDB_MODE"] == "disabled"

    def test_wandb_run_yields_none_when_not_allowed(self, settings, monkeypatch):
        from app.tracing import wandb_tracer

        monkeypatch.setattr(settings, "wandb_api_key", "key")
        fake = SimpleNamespace(init=AsyncMock())
        monkeypatch.setattr(wandb_tracer, "wandb", fake)
        with wandb_tracer.wandb_run("x") as run:
            assert run is None
        fake.init.assert_not_called()


# ----------------------------------------------------------------------------
# Spans
# ----------------------------------------------------------------------------


class TestSpans:
    async def test_spans_nest_under_the_current_trace(self):
        @traced("outer")
        async def outer():
            with span("inner", as_type="tool"):
                pass
            return "ok"

        with start_trace("t", {"strategy": "classic"}) as t:
            await outer()

        assert [s.name for s in t.spans] == ["outer", "inner"]
        outer_span, inner_span = t.spans
        assert outer_span.parent_id is None
        assert inner_span.parent_id == outer_span.id
        assert all(s.end_ns is not None for s in t.spans)

    async def test_span_outside_a_trace_is_a_no_op(self):
        with span("lonely") as s:
            assert s is None

    async def test_errors_are_recorded_and_re_raised(self):
        @traced("boom")
        async def boom():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError), start_trace("t", {}) as t:
            await boom()
        assert t.spans[0].error == "RuntimeError"
        assert t.outcome == "error"

    async def test_concurrent_traces_do_not_share_spans(self):
        import asyncio

        @traced("step")
        async def step():
            await asyncio.sleep(0)

        async def one(name):
            with start_trace(name, {}) as t:
                await step()
                await step()
            return t

        a, b = await asyncio.gather(one("a"), one("b"))
        assert len(a.spans) == 2 and len(b.spans) == 2


# ----------------------------------------------------------------------------
# Exporter
# ----------------------------------------------------------------------------


def _finished_trace(question="What is X?", answer="X is Y."):
    with start_trace("ask:classic", {"question": question, "strategy": "classic"}) as t:
        with span("retrieve", as_type="retriever", input=question) as s:
            s.output = ["c1", "c2"]
        with span("generate", as_type="generation") as s:
            s.model = "claude-sonnet-5"
            s.usage = {"input": 10, "output": 5}
            s.input = {"message": f"Context: secret. Question: {question}"}
            s.output = answer
        t.finish(SimpleNamespace(answer=answer, refusal=False, input_tokens=10, output_tokens=5))
    return t


class TestPayload:
    def test_shape_and_langfuse_attributes(self):
        t = _finished_trace()
        payload = lf.build_payload([t])
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        root, retrieve, generate = spans

        assert root["spanId"] == t.root_span_id
        assert len(root["traceId"]) == 32
        assert retrieve["parentSpanId"] == t.root_span_id
        assert _attrs(root)["langfuse.trace.name"] == "ask:classic"
        assert _attrs(root)["langfuse.trace.input"] == "What is X?"
        assert _attrs(retrieve)["langfuse.observation.type"] == "retriever"
        gen = _attrs(generate)
        assert gen["langfuse.observation.type"] == "generation"
        assert gen["langfuse.observation.model.name"] == "claude-sonnet-5"
        assert json.loads(gen["langfuse.observation.usage_details"]) == {"input": 10, "output": 5}
        assert int(generate["endTimeUnixNano"]) >= int(generate["startTimeUnixNano"])

    def test_every_text_field_goes_through_the_redactor(self, restore_redactor):
        lf.set_redactor(lambda s: s.replace("secret", "[REDACTED]").replace("X", "[E]"))
        body = json.dumps(lf.build_payload([_finished_trace()]))
        assert "secret" not in body
        assert "What is X" not in body
        assert "[REDACTED]" in body

    def test_content_can_be_left_out_entirely(self):
        body = json.dumps(lf.build_payload([_finished_trace()], include_content=False))
        assert "What is X" not in body
        assert "X is Y" not in body
        assert "secret" not in body
        assert "claude-sonnet-5" in body  # metadata still flows

    def test_long_text_is_truncated(self):
        long_q = "a" * (lf.MAX_TEXT_CHARS + 500)
        body = json.dumps(lf.build_payload([_finished_trace(question=long_q)]))
        assert "truncated 500 chars" in body


class TestExporter:
    def _exporter(self, recorder, **kw):
        return lf.LangfuseExporter(
            host="http://langfuse:3000",
            public_key="pk",
            secret_key="sk",
            transport=httpx.MockTransport(recorder),
            flush_interval=0.05,
            **kw,
        )

    def test_sends_otlp_json_with_basic_auth(self):
        rec = _Recorder()
        exp = self._exporter(rec)
        assert exp.submit(_finished_trace())
        exp.shutdown()

        assert len(rec.requests) == 1
        req = rec.requests[0]
        assert req.url.path == "/api/public/otel/v1/traces"
        assert req.headers["content-type"] == "application/json"
        assert req.headers["authorization"] == "Basic " + base64.b64encode(b"pk:sk").decode()
        assert exp.exported == 1 and exp.dropped == 0

    def test_full_queue_drops_and_counts(self):
        from app import monitoring

        before = monitoring.TELEMETRY_DROPPED.labels("langfuse", "queue_full")._value.get()
        exp = self._exporter(_Recorder(), max_queue=1)
        with patch.object(exp, "_ensure_worker"):
            assert exp.submit(_finished_trace()) is True
            assert exp.submit(_finished_trace()) is False
        assert exp.dropped == 1
        after = monitoring.TELEMETRY_DROPPED.labels("langfuse", "queue_full")._value.get()
        assert after == before + 1
        exp.shutdown()

    def test_backend_errors_are_counted_not_raised(self):
        rec = _Recorder(status=500)
        exp = self._exporter(rec)
        exp.submit(_finished_trace())
        exp.shutdown()
        assert rec.requests
        assert exp.dropped == 1 and exp.exported == 0

    def test_unreachable_backend_never_raises(self):
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)

        exp = self._exporter(refuse)
        exp.submit(_finished_trace())
        exp.shutdown()
        assert exp.dropped == 1

    def test_nothing_is_configured_when_policy_says_no(self, langfuse_keys):
        assert lf.configure(langfuse_keys) is None
        assert lf.export(_finished_trace()) is False


class TestEndToEnd:
    """One /ask becomes one exported trace with retrieve, rerank and generate spans."""

    def test_ask_exports_a_redacted_trace(
        self, client, langfuse_keys, monkeypatch, fake_chunks, mock_anthropic_response,
        restore_redactor,
    ):
        from app.rag.providers import anthropic_provider as ap

        monkeypatch.setattr(langfuse_keys, "telemetry_mode", "self_hosted")
        rec = _Recorder()
        lf.configure(langfuse_keys, transport=httpx.MockTransport(rec))
        lf.set_redactor(lambda s: s.replace("alice@example.com", "<EMAIL>"))

        hits = [
            SimpleNamespace(payload={"chunk_id": c.id, "doc_id": c.doc_id, "text": c.text})
            for c in fake_chunks(2)
        ]
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(return_value=mock_anthropic_response("An answer.", 321, 45))
            )
        )
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
            patch("app.rag.retrieve.embed_query_async", new=AsyncMock(return_value=[0.0])),
            patch("app.rag.retrieve.search", new=AsyncMock(return_value=hits)),
            patch("app.rag.rerank.rerank", side_effect=lambda q, c, k: c[:k]),
            patch.object(ap, "get_client", return_value=fake_client),
        ):
            r = client.post("/ask", json={"question": "Who is alice@example.com?"})
        assert r.status_code == 200, r.text
        lf.shutdown()

        spans = rec.spans
        names = [s["name"] for s in spans]
        assert names[0] == "ask:classic"
        assert {"retrieve", "rerank", "generate"} <= set(names)
        gen = _attrs(next(s for s in spans if s["name"] == "generate"))
        assert json.loads(gen["langfuse.observation.usage_details"]) == {
            "input": 321,
            "output": 45,
        }
        assert gen["langfuse.observation.model.name"] == "claude-sonnet-5"
        body = "".join(r.content.decode() for r in rec.requests)
        assert "alice@example.com" not in body
        assert "<EMAIL>" in body

    def test_health_reports_langfuse_only_when_allowed(self, client, langfuse_keys, monkeypatch):
        assert client.get("/health").json()["providers"]["langfuse"] is False
        monkeypatch.setattr(langfuse_keys, "telemetry_mode", "self_hosted")
        body = client.get("/health").json()
        assert body["providers"]["langfuse"] is True
        assert body["telemetry"]["mode"] == "self_hosted"
        monkeypatch.setattr(langfuse_keys, "langfuse_host", "https://cloud.langfuse.com")
        assert client.get("/health").json()["providers"]["langfuse"] is False


class TestAgentToolSpans:
    async def test_tool_calls_become_tool_spans(self, mock_anthropic_response):
        from unittest.mock import MagicMock

        from app.rag.providers import anthropic_provider as ap

        tool_resp = MagicMock()
        block = MagicMock(type="tool_use", id="tu_1")
        block.name = "search"
        block.input = {"query": "q"}
        tool_resp.content = [block]
        tool_resp.stop_reason = "tool_use"
        tool_resp.usage = MagicMock(input_tokens=5, output_tokens=2)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(side_effect=[tool_resp, mock_anthropic_response("Done.")])
            )
        )
        with (
            patch.object(ap, "get_client", return_value=fake_client),
            start_trace("ask:agentic", {"strategy": "agentic"}) as t,
        ):
            await ap.tool_use_loop(
                model="claude-sonnet-5",
                system="s",
                user_message="u",
                tools=[{"name": "search"}],
                tool_handlers={"search": AsyncMock(return_value="found it")},
            )

        kinds = [(s.name, s.as_type) for s in t.spans]
        assert kinds == [
            ("agent_step", "generation"),
            ("tool:search", "tool"),
            ("agent_step", "generation"),
        ]
        assert t.spans[1].output == "found it"
        assert t.spans[0].usage == {"input": 5, "output": 2}
