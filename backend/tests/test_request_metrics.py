"""Per-request cost and latency metrics, stage timing, spans and their privacy."""
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import monitoring, pricing
from app.rag import timing
from app.rag.generate import answer_question_detailed, run_strategy_raw
from app.rag.request_metrics import build_request_metrics
from app.rag.rerank import RerankOutcome
from app.rag.strategies.base import StrategyResult
from app.schemas import Chunk, Source
from app.tracing import get_trace, start_trace
from app.tracing.langfuse_exporter import build_payload
from app.tracing.spans import Span

CHUNK_TEXT = "Confidential clause ZX-4471: the escrow releases on the ninth day."
QUESTION = "When does the escrow release under clause ZX?"
CLASSIC_ORDER = [
    "preprocess",
    "dense",
    "sparse",
    "fusion",
    "rerank",
    "context_selection",
    "generation",
    "citation_validation",
    "response",
]


@pytest.fixture(autouse=True)
def _prices():
    pricing.reload_price_table()


def _chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            id=f"doc{i}:0",
            doc_id=f"doc{i}",
            text=f"{CHUNK_TEXT} ({i})",
            tokens=12,
            metadata={"filename": f"f{i}.md"},
        )
        for i in range(n)
    ]


def _hits(chunks):
    return [
        SimpleNamespace(payload={"chunk_id": c.id, "doc_id": c.doc_id, "text": c.text})
        for c in chunks
    ]


def _text_response(text: str, input_tokens: int, output_tokens: int):
    resp = MagicMock()
    resp.content = [MagicMock(type="text", text=text)]
    resp.stop_reason = "end_turn"
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


def _tool_response(name: str, tool_input: dict, input_tokens: int, output_tokens: int):
    block = MagicMock(type="tool_use", id=f"tu_{name}", input=tool_input)
    block.name = name
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "tool_use"
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


def _fake_client(create):
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def _classic_patches(chunks, create, *, hybrid_sparse=True):
    from app.rag.providers import anthropic_provider as ap

    return [
        patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
        patch("app.rag.retrieve.dense_search", new=AsyncMock(return_value=chunks)),
        patch(
            "app.rag.retrieve.sparse_search",
            new=AsyncMock(return_value=_hits(chunks) if hybrid_sparse else []),
        ),
        patch(
            "app.rag.retrieve.rerank_scored_async",
            new=AsyncMock(return_value=RerankOutcome.unscored(chunks[:2])),
        ),
        patch.object(ap, "get_client", return_value=_fake_client(create)),
    ]


async def _run_classic(settings, chunks=None):
    settings.retrieval_mode = "hybrid"
    chunks = chunks or _chunks()
    create = AsyncMock(return_value=_text_response("The escrow releases on day nine [S1].", 321, 45))
    patches = _classic_patches(chunks, create)
    for p in patches:
        p.start()
    try:
        return await answer_question_detailed(QUESTION, strategy="classic")
    finally:
        for p in reversed(patches):
            p.stop()


# ---------------------------------------------------------------------------
# RequestMetrics arithmetic
# ---------------------------------------------------------------------------


def _check_latency_adds_up(latency):
    values = latency.model_dump()
    assert all(v >= 0 for v in values.values()), values
    parts = sum(v for k, v in values.items() if k != "total")
    assert parts == pytest.approx(values["total"], abs=0.01)


def test_totals_add_up_and_are_non_negative():
    timer = timing.StageTimer()
    timer.add("retrieval", 12.5)
    timer.add("rerank", 30.0)
    timer.add("generation", 400.0)
    timer.add("citation_validation", 1.5)
    timer.add("context_selection", 2.0)  # folded into "other"
    with start_trace("ask:classic", {"strategy": "classic"}) as trace:
        gen = Span(name="generate", as_type="generation", model="claude-sonnet-5")
        gen.usage = {"input": 1000, "output": 100}
        trace.spans.append(gen)
    result = StrategyResult(
        answer="a", sources=[Source(chunk_id="none", quote="")],
        input_tokens=1000, output_tokens=100, candidate_count=40, context_count=8,
    )

    m = build_request_metrics(
        result, strategy="classic", model="claude-sonnet-5", provider="anthropic",
        total_ms=500.0, timer=timer, trace=trace,
    )

    assert m.latency_ms.total == 500.0
    assert m.latency_ms.other == pytest.approx(56.0)
    _check_latency_adds_up(m.latency_ms)
    assert m.llm_calls == 1
    assert m.estimated_cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    assert (m.retrieved_chunks, m.context_chunks) == (40, 8)


def test_parts_exceeding_wall_time_never_go_negative():
    timer = timing.StageTimer()
    timer.add("generation", 10.004)
    result = StrategyResult(answer="a", sources=[Source(chunk_id="none", quote="")])
    m = build_request_metrics(
        result, strategy="classic", model=None, provider=None,
        total_ms=10.0, timer=timer, trace=None,
    )
    assert m.latency_ms.other == 0
    _check_latency_adds_up(m.latency_ms)
    assert m.estimated_cost_usd == 0.0  # no call was made


def test_each_call_is_priced_with_its_own_model():
    with start_trace("ask:graph", {"strategy": "graph"}) as trace:
        for model, usage in (
            ("claude-haiku-4-5-20251001", {"input": 2000, "output": 200}),
            ("claude-sonnet-5", {"input": 1000, "output": 100}),
        ):
            s = Span(name="generate", as_type="generation", model=model)
            s.usage = usage
            trace.spans.append(s)
    result = StrategyResult(answer="a", sources=[], input_tokens=3000, output_tokens=300)
    m = build_request_metrics(
        result, strategy="graph", model="claude-sonnet-5", provider="anthropic",
        total_ms=1.0, timer=timing.StageTimer(), trace=trace,
    )
    haiku = (2000 * 1 + 200 * 5) / 1e6
    sonnet = (1000 * 2 + 100 * 10) / 1e6
    assert m.estimated_cost_usd == pytest.approx(haiku + sonnet)
    assert m.llm_calls == 2


def test_an_unpriced_call_makes_the_cost_unknown():
    with start_trace("ask:classic", {"strategy": "classic"}) as trace:
        s = Span(name="generate", as_type="generation", model="gpt-4o-mini")
        s.usage = {"input": 10, "output": 10}
        trace.spans.append(s)
    result = StrategyResult(answer="a", sources=[], input_tokens=10, output_tokens=10)
    m = build_request_metrics(
        result, strategy="classic", model="gpt-4o-mini", provider="openai",
        total_ms=1.0, timer=timing.StageTimer(), trace=trace,
    )
    assert m.estimated_cost_usd is None
    assert m.unpriced_models == ["gpt-4o-mini"]


def test_nested_stages_are_timed_exclusively():
    with timing.activate() as timer:
        with timing.stage("generation", log=False):
            timing.record("retrieval", 5.0)
            with timing.stage("citation_validation", log=False):
                pass
    assert timer.get("retrieval") == 5.0
    assert "citation_validation" in timer.durations_ms
    # The outer stage took well under 5 ms of wall time, all of it attributed
    # to the stages inside it, so none is left over as its own.
    assert timer.durations_ms["generation"] == 0


def test_stages_are_no_ops_without_a_timer():
    timing.record("retrieval", 5.0)
    with timing.stage("generation", span_name="x", log=False) as s:
        assert s is None
    assert timing.current_timer() is None


# ---------------------------------------------------------------------------
# Per-stage timing in each strategy
# ---------------------------------------------------------------------------


async def test_classic_times_every_stage(settings):
    answer, result = await _run_classic(settings)
    assert result is not None
    m = answer.metrics
    assert m is not None
    assert m.strategy == "classic" and m.provider == "anthropic"
    assert m.model == "claude-sonnet-5"
    assert m.latency_ms.retrieval > 0
    assert m.latency_ms.rerank > 0
    assert m.latency_ms.generation > 0
    assert m.latency_ms.citation_validation > 0
    _check_latency_adds_up(m.latency_ms)
    assert m.latency_ms.total >= answer.latency_ms
    assert (m.input_tokens, m.output_tokens, m.llm_calls) == (321, 45, 1)
    assert m.estimated_cost_usd == pytest.approx((321 * 2 + 45 * 10) / 1e6)
    assert m.retrieved_chunks == 3  # fused candidates
    assert m.context_chunks == 2


async def test_graph_times_stages_and_prices_extraction_separately(settings):
    from app.rag.providers import anthropic_provider as ap

    settings.graph_extraction_model = "claude-haiku-4-5-20251001"
    chunks = _chunks(3)

    async def create(**kwargs):
        if kwargs["model"] == "claude-haiku-4-5-20251001":
            return _text_response('["escrow"]', 40, 8)
        return _text_response("Day nine [S1].", 500, 60)

    graph = MagicMock(triples=[("escrow", "releases_on", "day nine")])
    with (
        patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
        patch("app.rag.strategies.graph.load_graph", return_value=graph),
        patch("app.rag.strategies.graph.neighbors", return_value=({"doc2:0"}, {"day nine"})),
        patch("app.rag.strategies.graph.describe_subgraph", return_value="escrow -> day nine"),
        patch("app.rag.retrieve.dense_search", new=AsyncMock(return_value=chunks[:2])),
        patch("app.rag.strategies.graph._fetch_chunks_by_id", new=AsyncMock(return_value=chunks[2:])),
        patch("app.rag.strategies.graph.rerank_async", new=AsyncMock(return_value=chunks[:2])),
        patch.object(ap, "get_client", return_value=_fake_client(AsyncMock(side_effect=create))),
    ):
        result, err = await run_strategy_raw(QUESTION, strategy="graph")

    assert err is None and result is not None
    m = result.metrics
    assert m is not None
    assert (m.input_tokens, m.output_tokens) == (540, 68)  # extraction included
    assert m.llm_calls == 2
    haiku = (40 * 1 + 8 * 5) / 1e6
    sonnet = (500 * 2 + 60 * 10) / 1e6
    assert m.estimated_cost_usd == pytest.approx(haiku + sonnet)
    for stage in ("retrieval", "rerank", "generation", "citation_validation"):
        assert getattr(m.latency_ms, stage) > 0, stage
    _check_latency_adds_up(m.latency_ms)
    assert (m.retrieved_chunks, m.context_chunks) == (3, 2)
    stages = [s["stage"] for s in get_trace(result.trace_id)["stages"]]
    assert stages[:2] == ["entity_extraction", "graph_walk"]
    assert stages[-3:] == ["generation", "citation_validation", "response"]


async def test_agentic_counts_every_call_and_times_stages(settings):
    from app.rag.providers import anthropic_provider as ap

    chunks = _chunks(2)
    finish = {
        "answer": "Day nine [S1].",
        "claims": [{"text": "Day nine.", "citations": ["S1"], "supported": True}],
        "status": "answered",
        "unsupported_notes": "",
    }
    create = AsyncMock(
        side_effect=[
            _tool_response("search", {"query": "escrow release"}, 300, 20),
            _tool_response("finish", finish, 450, 70),
        ]
    )
    with (
        patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
        patch("app.rag.retrieve.dense_search", new=AsyncMock(return_value=chunks)),
        patch.object(ap, "get_client", return_value=_fake_client(create)),
    ):
        result, err = await run_strategy_raw(QUESTION, strategy="agentic")

    assert err is None and result is not None
    m = result.metrics
    assert m is not None
    assert (m.input_tokens, m.output_tokens, m.llm_calls) == (750, 90, 2)
    assert m.estimated_cost_usd == pytest.approx((750 * 2 + 90 * 10) / 1e6)
    assert m.latency_ms.retrieval > 0
    assert m.latency_ms.generation > 0
    assert m.latency_ms.citation_validation > 0
    _check_latency_adds_up(m.latency_ms)
    assert m.context_chunks == 2


# ---------------------------------------------------------------------------
# Spans and the trace API
# ---------------------------------------------------------------------------


async def test_classic_span_order(settings):
    answer, _ = await _run_classic(settings)
    stored = get_trace(answer.trace_id)
    assert [s["stage"] for s in stored["stages"]] == CLASSIC_ORDER
    starts = [s["start_ms"] for s in stored["stages"]]
    assert starts == sorted(starts)
    assert all(s["duration_ms"] >= 0 for s in stored["stages"])
    by_stage = {s["stage"]: s for s in stored["stages"]}
    gen = by_stage["generation"]
    assert gen["usage"] == {"input": 321, "output": 45}
    assert gen["cost_usd"] == pytest.approx((321 * 2 + 45 * 10) / 1e6)
    validation = by_stage["citation_validation"]["attributes"]
    assert validation["cited_handles"] == ["S1"]
    assert validation["invalid_count"] == 0
    assert validation["grounded"] is True and validation["status"] == "answered"
    response = by_stage["response"]["attributes"]
    assert response["estimated_cost_usd"] == answer.metrics.estimated_cost_usd
    assert response["llm_calls"] == 1
    assert by_stage["context_selection"]["attributes"]["selected"] == 2


async def test_no_chunk_or_question_text_in_spans_or_logs(settings, caplog):
    caplog.set_level(logging.DEBUG, logger="app")
    captured = {}

    def keep(trace):
        captured["trace"] = trace
        return False

    with patch("app.tracing.langfuse_exporter.export", side_effect=keep):
        answer, _ = await _run_classic(settings)
    trace = captured["trace"]

    # Span attributes and the stored stage list are content-free.
    metadata = json.dumps([s.metadata for s in trace.spans], default=str)
    stages = json.dumps(get_trace(answer.trace_id)["stages"], default=str)
    for blob in (metadata, stages):
        assert "ZX-4471" not in blob and "escrow" not in blob.lower()

    # With content capture off, the exported payload carries no text either.
    payload = json.dumps(build_payload([trace], include_content=False))
    assert "ZX-4471" not in payload
    assert QUESTION not in payload
    assert "langfuse.observation.cost_details" in payload
    assert '"evalrag.stage"' in payload

    # Structured logs: one line per stage, with the request id field slot,
    # and never the chunk text.
    stage_lines = [r for r in caplog.records if r.getMessage() == "Pipeline stage completed"]
    assert {r.extra_fields["stage"] for r in stage_lines} >= {
        "context_selection", "generation", "citation_validation", "response",
    }
    for record in caplog.records:
        text = record.getMessage() + json.dumps(getattr(record, "extra_fields", {}), default=str)
        assert "ZX-4471" not in text, record.getMessage()


def test_request_id_is_on_stage_logs():
    from app.logging_config import StructuredJSONFormatter
    from app.middleware.request_id import request_id_context

    token = request_id_context.set("request-1-abc")
    try:
        with timing.activate(), timing.stage("generation"):
            pass
        record = logging.LogRecord("app.rag.timing", logging.INFO, __file__, 1,
                                   "Pipeline stage completed", None, None)
        record.extra_fields = {"stage": "generation"}
        line = json.loads(StructuredJSONFormatter().format(record))
    finally:
        request_id_context.reset(token)
    assert line["request_id"] == "request-1-abc"


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------


async def test_prometheus_counters_increment(settings):
    cost = monitoring.LLM_COST.labels("claude-sonnet-5", "classic")
    stage_hist = monitoring.STAGE_LATENCY.labels("classic", "generation")
    chunks_hist = monitoring.CONTEXT_CHUNKS.labels("classic")
    request_cost = monitoring.REQUEST_COST.labels("classic")

    def count(h):
        return next(s.value for s in h.collect()[0].samples if s.name.endswith("_count"))

    before = (cost._value.get(), count(stage_hist), count(chunks_hist), count(request_cost))
    await _run_classic(settings)
    after = (cost._value.get(), count(stage_hist), count(chunks_hist), count(request_cost))

    assert after[0] - before[0] == pytest.approx((321 * 2 + 45 * 10) / 1e6)
    assert after[1] == before[1] + 1
    assert after[2] == before[2] + 1
    assert after[3] == before[3] + 1


def test_unpriced_calls_are_counted():
    counter = monitoring.LLM_UNPRICED_CALLS.labels("llama3.1:8b", "classic")
    before = counter._value.get()
    monitoring.observe_cost("llama3.1:8b", "classic", None)
    assert counter._value.get() == before + 1


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_ask_response_includes_metrics(client, settings):
    settings.retrieval_mode = "hybrid"
    create = AsyncMock(return_value=_text_response("The escrow releases on day nine [S1].", 321, 45))
    patches = _classic_patches(_chunks(), create)
    for p in patches:
        p.start()
    try:
        r = client.post("/ask", json={"question": QUESTION})
    finally:
        for p in reversed(patches):
            p.stop()

    assert r.status_code == 200, r.text
    body = r.json()
    m = body["metrics"]
    assert m["strategy"] == "classic"
    assert m["input_tokens"] == body["input_tokens"] == 321
    assert m["estimated_cost_usd"] == pytest.approx((321 * 2 + 45 * 10) / 1e6)
    assert set(m["latency_ms"]) == {
        "total", "retrieval", "rerank", "generation", "citation_validation", "other",
    }
    assert m["llm_calls"] == 1 and m["context_chunks"] == 2

    trace = client.get(f"/traces/{body['trace_id']}").json()
    assert [s["stage"] for s in trace["stages"]] == CLASSIC_ORDER
    assert trace["output"]["estimated_cost_usd"] == m["estimated_cost_usd"]


async def test_non_anthropic_calls_get_a_generation_span():
    from app.rag.providers import base

    fake = AsyncMock(return_value={"text": "ok", "input_tokens": 12, "output_tokens": 3})
    with (
        patch("app.rag.providers.local.generate_with_usage", new=fake),
        start_trace("ask:classic", {"strategy": "classic"}) as trace,
    ):
        await base.generate_with_usage("local", model="llama3.1:8b", prompt="p")
    (s,) = trace.spans
    assert (s.name, s.as_type, s.model) == ("generate", "generation", "llama3.1:8b")
    assert s.usage == {"input": 12, "output": 3}
    assert s.cost is None and s.metadata["provider"] == "local"
