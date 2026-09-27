"""Prometheus metrics: what /metrics exposes, and who may read it."""
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from app import monitoring
from app.resilience import CircuitBreaker


@pytest.fixture
def local_client():
    """A client whose requests come from 127.0.0.1."""
    from app.main import app

    return TestClient(app, client=("127.0.0.1", 50000))


def _samples(text: str) -> list:
    return [s for fam in text_string_to_metric_families(text) for s in fam.samples]


def _value(text: str, name: str, **labels) -> float | None:
    for s in _samples(text):
        if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
            return s.value
    return None


class TestAccess:
    def test_disabled_by_default(self, local_client):
        assert local_client.get("/metrics").status_code == 404

    def test_served_to_localhost(self, local_client, settings, monkeypatch):
        monkeypatch.setattr(settings, "metrics_enabled", True)
        r = local_client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")

    def test_remote_callers_need_the_admin_key(self, client, settings, monkeypatch):
        monkeypatch.setattr(settings, "metrics_enabled", True)
        # No admin key configured: the admin dependency answers 503.
        assert client.get("/metrics").status_code == 503

        monkeypatch.setattr(settings, "admin_key", "admin-secret")
        assert client.get("/metrics").status_code == 403
        assert client.get("/metrics", headers={"X-Admin-Key": "wrong"}).status_code == 403
        r = client.get("/metrics", headers={"X-Admin-Key": "admin-secret"})
        assert r.status_code == 200

    def test_not_in_the_openapi_schema(self, client):
        assert "/metrics" not in client.get("/openapi.json").json()["paths"]


class TestContent:
    @pytest.fixture(autouse=True)
    def _enable(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "metrics_enabled", True)

    def test_requests_are_labelled_by_route_template(self, local_client):
        local_client.get("/traces/some-unknown-id")
        text = local_client.get("/metrics").text
        assert _value(
            text,
            "evalrag_http_requests_total",
            route="/traces/{trace_id}",
            status="404",
        )
        assert "some-unknown-id" not in text

    def test_unmatched_paths_share_one_label(self, local_client):
        local_client.get("/no/such/path/123")
        text = local_client.get("/metrics").text
        assert _value(text, "evalrag_http_requests_total", route="unmatched")
        assert "/no/such/path" not in text

    def test_ask_records_tokens_latency_and_refusals(self, local_client, fake_chunks):
        chunks = fake_chunks(2)
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
            patch("app.rag.strategies.classic.dense_search", new=AsyncMock(return_value=chunks)),
            patch("app.rag.strategies.classic.rerank_async", new=AsyncMock(return_value=chunks)),
            patch(
                "app.rag.strategies.classic.generate_with_usage",
                new=AsyncMock(
                    return_value={"text": "An answer.", "input_tokens": 3, "output_tokens": 2}
                ),
            ),
        ):
            local_client.post("/ask", json={"question": "q?"})
        with patch("app.rag.generate.store_count", new=AsyncMock(return_value=0)):
            local_client.post("/ask", json={"question": "q?"})

        text = local_client.get("/metrics").text
        assert _value(
            text, "evalrag_strategy_duration_seconds_count", strategy="classic", outcome="ok"
        )
        assert _value(
            text, "evalrag_refusals_total", strategy="classic", reason="no_documents"
        )

    def test_token_counter_labels(self):
        before = monitoring.LLM_TOKENS.labels("m-test", "classic", "input")._value.get()
        monitoring.observe_tokens("m-test", "classic", 7, 0)
        assert monitoring.LLM_TOKENS.labels("m-test", "classic", "input")._value.get() == before + 7

    def test_circuit_breaker_state_gauge(self, local_client):
        breaker = CircuitBreaker(failure_threshold=1, service_name="TestService")
        text = local_client.get("/metrics").text
        assert _value(text, "evalrag_circuit_breaker_state", service="TestService") == 0

        breaker.record_failure()
        text = local_client.get("/metrics").text
        assert _value(text, "evalrag_circuit_breaker_state", service="TestService") == 2
        assert _value(text, "evalrag_circuit_breaker_failures", service="TestService") == 1

    def test_telemetry_dropped_counter_is_exposed(self, local_client):
        monitoring.observe_dropped("langfuse", "queue_full")
        text = local_client.get("/metrics").text
        assert _value(
            text, "evalrag_telemetry_dropped_total", exporter="langfuse", reason="queue_full"
        )


def test_grafana_dashboard_only_uses_exported_metrics():
    """The provisioned dashboard must not drift from the metric names."""
    dashboard = (
        Path(__file__).resolve().parents[2]
        / "infra/monitoring/grafana/dashboards/evalrag-overview.json"
    )
    if not dashboard.exists():  # the backend image carries backend/ only
        pytest.skip("infra/ not available")
    panels = json.loads(dashboard.read_text())["panels"]
    used = {
        m for p in panels for t in p["targets"] for m in re.findall(r"\b(evalrag_\w+)", t["expr"])
    }
    families = list(monitoring.REGISTRY.collect())
    exported = {f.name for f in families} | {s.name for f in families for s in f.samples}
    for name in used:
        base = re.sub(r"_(bucket|count|sum|total)$", "", name)
        assert name in exported or base in exported, name
