"""How the access-control, GDPR and monitoring features fit together.

Each feature has its own tests; these cover the seams between them: the
principal id that access control assigns is the one erasure, export,
retention and traces key on.
"""
from datetime import UTC, datetime, timedelta

from app import audit
from app.access import Principal
from app.privacy import erasure, export, retention
from app.privacy.usage import usage_summary
from app.tracing import start_trace, traces_for_principal


def _principal(pid: str = "key:7") -> Principal:
    return Principal(id=pid, kind="api_key", display_name="k", groups=("legal",), tenant="acme")


def test_scope_carries_the_principal_id_for_traces():
    scope = _principal().scope()
    assert scope.principal_id == "key:7"
    # It identifies the reader; it must not change what the reader can see.
    assert scope.permits({"tenant": "acme", "acl_groups": ["legal"]})


def test_traces_opened_with_a_scope_are_found_by_principal():
    scope = _principal("key:8").scope()
    with start_trace("ask:classic", {"question": "q"}, principal_id=scope.principal_id):
        pass
    assert [t["name"] for t in traces_for_principal("key:8")] == ["ask:classic"]


def _write(record: audit.AuditRecord) -> None:
    future = audit.emit(record)
    assert future is not None
    future.result()


def test_audit_events_are_erased_with_the_principal(auth_db):
    _write(audit.AuditRecord(principal_id="key:7", tenant="acme", action="ask"))
    _write(audit.AuditRecord(principal_id="key:9", tenant="acme", action="ask"))

    removed = erasure.erase_audit(erasure.ErasureRequest(subject="p", principal_id="key:7"))

    assert removed == 1
    remaining = audit.list_audit_events()["events"]
    assert [e["principal_id"] for e in remaining] == ["key:9"]


def test_audit_erasure_is_a_no_op_without_the_auth_database():
    assert erasure.erase_audit(erasure.ErasureRequest(subject="p", principal_id="key:7")) == 0


def test_audit_is_a_registered_erasure_retention_and_export_target():
    assert "audit" in erasure._targets
    assert "audit" in retention._targets
    assert "audit" in export._sources


def test_retention_purges_old_audit_events(auth_db):
    _write(audit.AuditRecord(principal_id="key:7", tenant="acme", action="ask"))
    assert retention.purge_audit(datetime.now(UTC) + timedelta(seconds=1)) == 1


def test_usage_accepts_the_principal_id_spelling(auth_db):
    from app.auth import create_api_key

    create_api_key("svc")
    key_id = usage_summary("svc")["api_keys"][0]["id"]
    summary = usage_summary(f"key:{key_id}")
    assert [k["id"] for k in summary["api_keys"]] == [key_id]
    assert usage_summary("oidc:someone")["api_keys"] == []


def test_erasure_requests_are_themselves_audited(auth_db, monkeypatch):
    import asyncio

    from app.api import privacy as privacy_api

    async def resolved():
        return erasure.ErasureRequest(
            subject="document:docA", doc_ids=frozenset({"docA"}), principal_id=None
        )

    async def fake_erase(request):
        return erasure.ErasureReport(subject=request.subject, doc_ids=["docA"])

    monkeypatch.setattr(erasure, "erase", fake_erase)
    asyncio.run(privacy_api._run(resolved()))

    (event,) = audit.list_audit_events(action="delete")["events"]
    assert event["principal_id"] == audit.ADMIN_KEY_PRINCIPAL
    assert event["doc_ids"] == ["docA"]
    assert event["detail"] == "erase document:docA"
