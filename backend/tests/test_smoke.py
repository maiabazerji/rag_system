"""End-to-end smoke tests over the HTTP surface.

These cover the failure that shipped once already: routes gaining an auth
dependency the caller does not satisfy.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "ok"
    assert body["auth_required"] is False
    for provider in ("anthropic", "openai", "langfuse", "wandb"):
        assert provider in body["providers"]


def test_health_reports_auth_state(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    assert client.get("/health").json()["auth_required"] is True


def test_ask_refuses_when_nothing_is_indexed(client):
    """With an empty index, /ask returns a refusal rather than an error."""
    with patch("app.rag.generate.store_count", new=AsyncMock(return_value=0)):
        r = client.post("/ask", json={"question": "what is foo?"})

    assert r.status_code == 200
    body = r.json()
    assert body["refusal"] is True
    assert body["confidence"] == 0.0
    assert "Ingest" in body["answer"]


def test_ask_rejects_an_empty_question(client):
    """Validation happens in the schema, so the handler never sees a blank question."""
    assert client.post("/ask", json={"question": "   "}).status_code in (200, 422)
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_rejects_an_unknown_strategy(client):
    r = client.post("/ask", json={"question": "hi", "strategy": "telepathy"})
    assert r.status_code == 422


def test_ask_returns_token_telemetry(client, fake_chunks):
    """Answer must carry the token counts usage accounting depends on."""
    chunks = fake_chunks(2)

    with (
        patch("app.rag.generate.store_count", new=AsyncMock(return_value=10)),
        patch(
            "app.rag.strategies.classic.dense_search",
            new=AsyncMock(return_value=chunks),
        ),
        patch(
            "app.rag.strategies.classic.rerank_async",
            new=AsyncMock(return_value=chunks),
        ),
        patch(
            "app.rag.strategies.classic.generate_with_usage",
            new=AsyncMock(
                return_value={
                    "text": "An answer.",
                    "input_tokens": 321,
                    "output_tokens": 45,
                }
            ),
        ),
    ):
        r = client.post("/ask", json={"question": "what is foo?"})

    assert r.status_code == 200
    body = r.json()
    assert body["input_tokens"] == 321
    assert body["output_tokens"] == 45
    assert body["latency_ms"] >= 0
    assert body["refusal"] is False


class TestAuthEnforcement:
    """When REQUIRE_API_KEY is on, protected routes must actually be protected."""

    PROTECTED = [
        ("post", "/ask", {"question": "hi"}),
        ("post", "/compare", {"question": "hi", "variants": [{"strategy": "classic"}]}),
        ("post", "/compare/strategies", {"question": "hi", "strategies": ["classic"]}),
        ("post", "/eval/run", {"dataset": "golden_v1"}),
        ("post", "/graph/reset", None),
        ("get", "/graph/stats", None),
        ("get", "/ingest/stats", None),
        ("post", "/eval/human-rate", {"question": "q", "answer": "a", "rating": 5}),
    ]

    @pytest.fixture
    def stub_expensive_handlers(self):
        """Stub the handlers that would otherwise spend money or write files.

        These tests are about the auth dependency, which runs before the
        handler. Letting the real bodies run made /eval/run execute a full
        34-example evaluation on every parametrised case.
        """
        graph_stub = MagicMock()
        graph_stub.stats.return_value = {
            "triples": 0,
            "entities": 0,
            "chunks_indexed": 0,
            "docs_indexed": 0,
        }

        with (
            patch(
                "app.api.eval_routes.run_evaluation",
                new=AsyncMock(return_value={"n": 0, "aggregate": None}),
            ),
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=0)),
            patch("app.api.graph.graph_store", graph_stub),
        ):
            yield

    @pytest.mark.parametrize("method,path,payload", PROTECTED)
    def test_open_when_auth_is_disabled(
        self, client, stub_expensive_handlers, method, path, payload
    ):
        """Default configuration: a fresh clone works without a key."""
        call = getattr(client, method)
        r = call(path, json=payload) if payload is not None else call(path)
        assert r.status_code != 401

    @pytest.mark.parametrize("method,path,payload", PROTECTED)
    def test_401_without_a_key_when_auth_is_enabled(
        self, client, settings, monkeypatch, method, path, payload
    ):
        monkeypatch.setattr(settings, "require_api_key", True)
        call = getattr(client, method)
        r = call(path, json=payload) if payload is not None else call(path)
        assert r.status_code == 401
        assert "Authorization" in r.json()["detail"]

    def test_401_on_a_malformed_authorization_header(
        self, client, settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "require_api_key", True)
        r = client.post(
            "/ask", json={"question": "hi"}, headers={"Authorization": "sk_rawkey"}
        )
        assert r.status_code == 401
        assert "Bearer" in r.json()["detail"]

    def test_health_stays_open(self, client, settings, monkeypatch):
        monkeypatch.setattr(settings, "require_api_key", True)
        assert client.get("/health").status_code == 200


class TestAdminRoutes:
    def test_admin_is_disabled_until_a_key_is_configured(self, client):
        r = client.get("/admin/keys")
        assert r.status_code == 503
        assert "ADMIN_KEY" in r.json()["detail"]

    def test_admin_rejects_a_wrong_key(self, client, settings, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "correct-horse")
        r = client.get("/admin/keys", headers={"X-Admin-Key": "wrong"})
        assert r.status_code == 403

    def test_admin_rejects_a_missing_header(self, client, settings, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "correct-horse")
        assert client.get("/admin/keys").status_code == 403


def test_openapi_schema_builds(client):
    """A malformed response_model or dependency shows up here first."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/ask" in r.json()["paths"]
