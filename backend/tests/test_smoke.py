from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "providers" in body
    for p in ("anthropic", "openai", "cohere", "langfuse", "wandb"):
        assert p in body["providers"]


def test_ask_refusal_when_no_context():
    r = client.post("/ask", json={"question": "what is foo?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refusal"] is True
