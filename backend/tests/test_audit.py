"""Audit log, key access administration, and the API-key principal."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app import audit, auth
from app.access import Principal
from app.schemas import Answer, Source

ADMIN = {"X-Admin-Key": "correct-horse"}


def _principal(pid="key:1", tenant="acme") -> Principal:
    return Principal(id=pid, kind="api_key", display_name="ci", tenant=tenant)


def _events() -> list[audit.AuditEvent]:
    with auth.session_scope() as db:
        rows = db.query(audit.AuditEvent).order_by(audit.AuditEvent.id).all()
        db.expunge_all()
        return rows


class TestAudited:
    def test_writes_one_event(self, auth_db):
        with audit.audited(_principal(), "ask", strategy="classic", question="Salaire de Bob ?") as ev:
            ev.add_sources(
                [Source(chunk_id="d1:0", quote=""), Source(chunk_id="d1:3", quote=""),
                 Source(chunk_id="none", quote="")]
            )
        [e] = _events()
        assert (e.principal_id, e.tenant, e.action, e.strategy, e.status) == (
            "key:1", "acme", "ask", "classic", "ok"
        )
        assert e.doc_ids == ["d1"]
        assert e.question_hash == audit.question_hash("Salaire de Bob ?")
        assert e.question is None, "question text is not stored by default"

    def test_question_text_is_opt_in(self, auth_db, settings, monkeypatch):
        monkeypatch.setattr(settings, "audit_store_questions", True)
        with audit.audited(_principal(), "ask", question="q?"):
            pass
        assert _events()[0].question == "q?"

    def test_hash_ignores_whitespace_noise(self):
        assert audit.question_hash(" a  b\n") == audit.question_hash("a b")

    def test_http_errors_are_recorded_and_reraised(self, auth_db):
        with pytest.raises(HTTPException), audit.audited(_principal(), "ingest"):
            raise HTTPException(status_code=403, detail="nope")
        e = _events()[0]
        assert e.status == "rejected"
        assert e.detail == "403: nope"

    def test_other_errors_are_recorded_and_reraised(self, auth_db):
        with pytest.raises(ZeroDivisionError), audit.audited(_principal(), "ask"):
            1 / 0  # noqa: B018
        assert _events()[0].status == "error"

    def test_inactive_without_a_database(self, monkeypatch):
        monkeypatch.setattr(auth, "_SessionLocal", None)
        assert audit.emit(audit.AuditRecord("key:1", None, "ask")) is None

    def test_a_failing_write_never_raises(self, auth_db, caplog):
        with patch.object(auth, "session_scope", side_effect=RuntimeError("db gone")):
            audit._write(audit.AuditRecord("key:1", None, "ask"))
        assert "Failed to write audit event" in caplog.text

    def test_emit_does_not_block_on_the_database(self, auth_db, monkeypatch):
        """With the real executor the write happens on another thread."""
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(audit, "_executor", ThreadPoolExecutor(max_workers=1))
        future = audit.emit(audit.AuditRecord("key:9", None, "ask"))
        future.result(timeout=5)
        assert _events()[0].principal_id == "key:9"


class TestQueries:
    @pytest.fixture
    def seeded(self, auth_db):
        now = datetime.now(UTC)
        with auth.session_scope() as db:
            for i, (pid, action, age) in enumerate(
                [
                    ("key:1", "ask", 40),
                    ("key:1", "ingest", 20),
                    ("oidc:u", "ask", 10),
                    ("oidc:u", "compare", 1),
                ]
            ):
                db.add(
                    audit.AuditEvent(
                        id=i + 1,
                        ts=now - timedelta(days=age),
                        principal_id=pid,
                        tenant="acme",
                        action=action,
                    )
                )
        return now

    def test_filters_and_newest_first(self, seeded):
        page = audit.list_audit_events(principal_id="oidc:u")
        assert [e["action"] for e in page["events"]] == ["compare", "ask"]
        assert audit.list_audit_events(action="ask")["total"] == 2
        assert audit.list_audit_events(since=seeded - timedelta(days=15))["total"] == 2
        assert audit.list_audit_events(until=seeded - timedelta(days=15))["total"] == 2

    def test_pagination(self, seeded):
        page = audit.list_audit_events(limit=3, offset=3)
        assert page["total"] == 4
        assert [e["id"] for e in page["events"]] == [1]

    def test_purge_before(self, seeded):
        assert audit.purge_audit_events(seeded - timedelta(days=15)) == 2
        assert audit.list_audit_events()["total"] == 2

    def test_erase_principal(self, seeded):
        assert audit.erase_principal_audit("key:1") == 2
        assert audit.erase_principal_audit("key:1") == 0
        assert {e["principal_id"] for e in audit.list_audit_events()["events"]} == {"oidc:u"}


class TestRoutesAreAudited:
    def test_ask_records_the_documents_it_disclosed(self, client, auth_db):
        answer = Answer(
            question="q",
            answer="a",
            sources=[Source(chunk_id="docA:2", quote="x")],
            confidence=0.9,
            refusal=False,
        )
        with patch("app.api.ask.answer_question", new=AsyncMock(return_value=answer)):
            client.post("/ask", json={"question": "q", "strategy": "graph"})
        [e] = _events()
        assert (e.principal_id, e.action, e.strategy, e.doc_ids) == (
            "local", "ask", "graph", ["docA"]
        )

    def test_rejected_ingest_is_audited(self, client, auth_db):
        client.post("/ingest", files={"file": ("a.exe", b"x")})
        [e] = _events()
        assert (e.action, e.status) == ("ingest", "rejected")

    def test_successful_ingest_records_the_doc(self, client, auth_db):
        with (
            patch(
                "app.api.ingest.enqueue_document",
                new=AsyncMock(return_value={"doc_id": "d9", "chunks": 1}),
            ),
            patch("app.api.ingest.store_count", new=AsyncMock(return_value=1)),
        ):
            client.post("/ingest", files={"file": ("a.txt", b"x")})
        assert _events()[0].doc_ids == ["d9"]

    def test_compare_strategies_is_audited(self, client, auth_db):
        with patch("app.api.compare.run_strategy_raw", new=AsyncMock(return_value=(None, "x"))):
            client.post("/compare/strategies", json={"question": "q", "strategies": ["classic"]})
        assert _events()[0].action == "compare"


class TestAdminRoutes:
    @pytest.fixture(autouse=True)
    def _admin(self, auth_db, settings, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "correct-horse")

    def test_create_key_with_access(self, client):
        r = client.post(
            "/admin/keys",
            json={"name": "hr", "groups": ["hr", " hr"], "tenant": "acme"},
            headers=ADMIN,
        )
        assert r.status_code == 201
        body = r.json()
        assert (body["tenant"], body["groups"], body["is_admin"]) == ("acme", ["hr"], False)
        [listed] = client.get("/admin/keys", headers=ADMIN).json()
        assert (listed["tenant"], listed["groups"]) == ("acme", ["hr"])

    def test_set_key_access(self, client):
        auth.create_api_key("k")
        r = client.put(
            "/admin/keys/1/access", json={"groups": ["finance"], "is_admin": True}, headers=ADMIN
        )
        assert r.status_code == 200
        assert r.json() == {
            "key_id": 1, "tenant": "default", "groups": ["finance"], "is_admin": True
        }
        r = client.put("/admin/keys/1/access", json={"tenant": "acme"}, headers=ADMIN)
        assert r.json()["groups"] == ["finance"], "omitted fields are unchanged"
        assert r.json()["tenant"] == "acme"

    def test_set_access_takes_effect_on_the_next_request(self, client, settings, monkeypatch):
        monkeypatch.setattr(settings, "require_api_key", True)
        key = auth.create_api_key("k", requests_per_minute=100)
        request = MagicMock()
        request.url.path = "/ask"
        assert auth.require_principal(request, f"Bearer {key}").groups == ()
        client.put("/admin/keys/1/access", json={"groups": ["hr"]}, headers=ADMIN)
        assert auth.require_principal(request, f"Bearer {key}").groups == ("hr",)

    def test_set_access_unknown_key(self, client):
        assert client.put("/admin/keys/42/access", json={}, headers=ADMIN).status_code == 404

    def test_set_access_malformed_groups(self, client):
        auth.create_api_key("k")
        r = client.put("/admin/keys/1/access", json={"groups": ["x" * 300]}, headers=ADMIN)
        assert r.status_code == 422

    def test_audit_endpoint(self, client):
        auth.create_api_key("k")
        client.put("/admin/keys/1/access", json={"groups": ["hr"]}, headers=ADMIN)
        body = client.get("/admin/audit?action=admin", headers=ADMIN).json()
        assert body["total"] == 1
        assert body["events"][0]["principal_id"] == audit.ADMIN_KEY_PRINCIPAL
        assert body["events"][0]["detail"] == "set_key_access id=1"

    def test_audit_endpoint_filters_and_pages(self, client):
        for _ in range(3):
            audit.emit(audit.AuditRecord("oidc:u", "acme", "ask"))
        body = client.get(
            "/admin/audit", params={"principal": "oidc:u", "limit": 2}, headers=ADMIN
        ).json()
        assert (body["total"], len(body["events"]), body["limit"]) == (3, 2, 2)
        since = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        assert client.get(
            "/admin/audit", params={"since": since}, headers=ADMIN
        ).json()["total"] == 0

    def test_audit_endpoint_needs_the_admin_key(self, client):
        assert client.get("/admin/audit").status_code == 403


class TestApiKeyPrincipal:
    def test_require_principal_carries_the_keys_access(self, auth_db, settings, monkeypatch):
        monkeypatch.setattr(settings, "require_api_key", True)
        key = auth.create_api_key("k", groups=["hr"], tenant="acme", is_admin=True)
        request = MagicMock()
        request.url.path = "/ask"
        p = auth.require_principal(request, f"Bearer {key}")
        assert (p.id, p.kind, p.tenant, p.groups, p.is_admin) == (
            "key:1", "api_key", "acme", ("hr",), True
        )
        assert p.key_id == 1 and p.usage_id is not None

    def test_legacy_dict_dependency_still_works(self, auth_db, settings, monkeypatch):
        monkeypatch.setattr(settings, "require_api_key", True)
        key = auth.create_api_key("k", groups=["hr"])
        request = MagicMock()
        request.url.path = "/ask"
        d = auth.require_api_key(request, f"Bearer {key}")
        assert d["id"] == 1 and d["usage_id"] is not None and d["name"] == "k"
        assert d["groups"] == ["hr"] and d["principal_id"] == "key:1"

    def test_charge_and_record_tokens_accept_a_principal(self, auth_db, settings, monkeypatch):
        monkeypatch.setattr(settings, "require_api_key", True)
        key = auth.create_api_key("k", requests_per_minute=10)
        request = MagicMock()
        request.url.path = "/ask"
        p = auth.require_principal(request, f"Bearer {key}")
        auth.charge(p, 3)
        auth.record_tokens(p, tokens_input=5, tokens_output=6)
        with auth.session_scope() as db:
            row = db.get(auth.APIKeyUsage, p.usage_id)
            assert (row.units, row.tokens_input, row.tokens_output) == (3, 5, 6)

    def test_one_usage_row_per_request(self, client, auth_db, settings, monkeypatch):
        """Router and handler share one dependency, so a request is metered once."""
        monkeypatch.setattr(settings, "require_api_key", True)
        key = auth.create_api_key("k", requests_per_minute=100)
        with patch("app.api.ingest.store_count", new=AsyncMock(return_value=0)):
            client.get("/ingest/stats", headers={"Authorization": f"Bearer {key}"})
        with auth.session_scope() as db:
            assert db.query(auth.APIKeyUsage).count() == 1


def test_legacy_key_table_gains_the_access_columns(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE api_keys (id INTEGER PRIMARY KEY, key_hash VARCHAR(64), "
                "key_hint VARCHAR(16), name VARCHAR(256), requests_per_minute INTEGER, "
                "is_active INTEGER, created_at DATETIME, last_used DATETIME)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO api_keys (id, key_hash, key_hint, name, requests_per_minute, "
                "is_active) VALUES (1, 'h', 'sk_', 'old', 10, 1)"
            )
        )
    auth._add_missing_columns(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("api_keys")}
    assert {"tenant", "acl_groups", "is_admin"} <= columns

    with engine.connect() as conn:
        row = conn.execute(text("SELECT tenant, acl_groups, is_admin FROM api_keys")).one()
    assert tuple(row) == (None, None, 0)
    # Idempotent.
    auth._add_missing_columns(engine)


def test_init_db_creates_the_audit_table(tmp_path, settings, monkeypatch):
    from sqlalchemy import create_engine as real_create_engine
    from sqlalchemy import inspect

    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    monkeypatch.setattr(auth, "_SessionLocal", None)
    monkeypatch.setattr(auth, "_engine", None)
    monkeypatch.setattr(auth, "create_engine", lambda *_a, **_k: real_create_engine(url))
    auth.init_db()
    assert "audit_events" in inspect(real_create_engine(url)).get_table_names()
