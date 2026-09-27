"""PII redaction where personal data would otherwise be persisted: ingest, logs, traces."""
import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app import tracing
from app.logging_config import PIIRedactingFilter, StructuredJSONFormatter, _build_config
from app.privacy.pii import PIIRejected
from app.rag.ingest import enqueue_document
from app.rag.store import StaleRevisions

DOC = b"Contact Jean at jean.dupont@example.fr or 06 12 34 56 78 about the RAG demo."


@pytest.fixture
def indexing():
    """Stub out embedding and the vector store; expose the upsert mock."""
    with (
        patch("app.rag.ingest.embed_texts_async", new=AsyncMock(return_value=[[0.0]])),
        patch("app.rag.ingest.upsert", new=AsyncMock()) as mock_upsert,
        patch(
            "app.rag.ingest.delete_stale_revisions",
            new=AsyncMock(return_value=StaleRevisions()),
        ),
    ):
        yield mock_upsert


@pytest.mark.asyncio
class TestIngest:
    async def test_mask_mode_stores_placeholders_and_counts_only(
        self, indexing, settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "pii_mode_ingest", "mask")
        result = await enqueue_document("notes.txt", DOC)

        chunks = indexing.await_args.args[0]
        stored = " ".join(c.text for c in chunks)
        assert "jean.dupont@example.fr" not in stored
        assert "06 12 34 56 78" not in stored
        assert "[EMAIL]" in stored and "[PHONE]" in stored
        assert chunks[0].metadata["pii_counts"] == {"EMAIL": 1, "PHONE": 1}
        assert chunks[0].metadata["ingested_at"]
        assert result["pii_counts"] == {"EMAIL": 1, "PHONE": 1}

    async def test_doc_id_is_derived_from_the_redacted_text(
        self, indexing, settings, monkeypatch
    ):
        """No hash of the raw personal data is kept as an identifier."""
        monkeypatch.setattr(settings, "pii_mode_ingest", "mask")
        a = await enqueue_document("a.txt", b"write to a@example.fr")
        b = await enqueue_document("b.txt", b"write to b@example.fr")
        assert a["doc_id"] == b["doc_id"]

    async def test_off_mode_indexes_verbatim(self, indexing, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_mode_ingest", "off")
        result = await enqueue_document("notes.txt", DOC)
        chunks = indexing.await_args.args[0]
        assert "jean.dupont@example.fr" in chunks[0].text
        assert result["pii_counts"] == {}

    async def test_reject_mode_refuses_before_anything_is_stored(
        self, indexing, settings, monkeypatch
    ):
        monkeypatch.setattr(settings, "pii_mode_ingest", "reject")
        with pytest.raises(PIIRejected) as exc:
            await enqueue_document("notes.txt", DOC)
        assert exc.value.types == ["EMAIL", "PHONE"]
        indexing.assert_not_awaited()


class TestIngestEndpoint:
    def test_reject_mode_is_a_422_listing_the_types(self, client, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_mode_ingest", "reject")
        resp = client.post("/ingest", files={"file": ("notes.txt", DOC, "text/plain")})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["code"] == "pii_rejected"
        assert detail["types"] == ["EMAIL", "PHONE"]
        assert "jean.dupont" not in resp.text


def _record(msg, *args, extra_fields=None):
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, msg, args, None)
    if extra_fields is not None:
        record.extra_fields = extra_fields
    return record


class TestLogFilter:
    def test_masks_message_and_args(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_redact_logs", True)
        record = _record("login by %s from %s", "a@b.fr", "192.168.0.1")
        assert PIIRedactingFilter().filter(record) is True
        assert record.getMessage() == "login by [EMAIL] from [IPV4]"

    def test_masks_extra_fields(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_redact_logs", True)
        record = _record("ok", extra_fields={"user": "a@b.fr", "n": 3, "nested": ["06 12 34 56 78"]})
        PIIRedactingFilter().filter(record)
        payload = json.loads(StructuredJSONFormatter().format(record))
        assert payload["user"] == "[EMAIL]"
        assert payload["n"] == 3
        assert payload["nested"] == ["[PHONE]"]

    def test_disabled_leaves_records_alone(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_redact_logs", False)
        record = _record("login by %s", "a@b.fr", extra_fields={"user": "a@b.fr"})
        PIIRedactingFilter().filter(record)
        assert record.getMessage() == "login by a@b.fr"
        assert record.extra_fields == {"user": "a@b.fr"}

    def test_clean_record_keeps_its_args(self, settings, monkeypatch):
        """Records without PII are not rewritten, so lazy %-formatting survives."""
        monkeypatch.setattr(settings, "pii_redact_logs", True)
        record = _record("took %d ms", 42)
        PIIRedactingFilter().filter(record)
        assert record.args == (42,)

    @pytest.mark.parametrize("file_logging", [True, False])
    def test_installed_on_every_handler(self, file_logging):
        config = _build_config(file_logging)
        assert config["filters"]["pii"]["()"].endswith("PIIRedactingFilter")
        for handler in config["handlers"].values():
            assert handler["filters"] == ["pii"]


@pytest.fixture
def traces(monkeypatch):
    """An empty trace store."""
    store = tracing.OrderedDict()
    monkeypatch.setattr(tracing, "_TRACES", store)
    return store


class TestTraces:
    def test_redacts_inputs_and_events(self, traces, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_redact_traces", True)
        with tracing.start_trace("ask:classic", {"question": "Who is a@b.fr?"}) as t:
            t.log("retrieve", {"context": "call 06 12 34 56 78"})
        stored = tracing.get_trace(t.id)
        assert stored["inputs"]["question"] == "Who is [EMAIL]?"
        assert stored["events"][0]["data"]["context"] == "call [PHONE]"

    def test_redaction_can_be_disabled(self, traces, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_redact_traces", False)
        with tracing.start_trace("ask:classic", {"question": "Who is a@b.fr?"}) as t:
            pass
        assert tracing.get_trace(t.id)["inputs"]["question"] == "Who is a@b.fr?"

    def test_records_principal_time_and_documents(self, traces):
        with tracing.start_trace("ask:agentic", {"question": "q"}, principal_id="7") as t:
            t.log("step", {"tool_calls": [{"name": "fetch", "input": {"chunk_id": "doc1:4"}}]})
            t.log("sources", {"chunk_ids": ["doc2:0", "doc2:1", "none"]})
            t.log("meta", {"doc_id": "doc3"})
        stored = tracing.get_trace(t.id)
        assert stored["principal_id"] == "7"
        assert stored["doc_ids"] == ["doc1", "doc2", "doc3"]
        assert datetime.fromisoformat(stored["created_at"]) <= datetime.now(UTC)

    def test_erase_by_document_or_principal(self, traces):
        with tracing.start_trace("a", {}) as a:
            a.log("sources", {"chunk_ids": ["d1:0"]})
        with tracing.start_trace("b", {}, principal_id="p") as b:
            pass
        with tracing.start_trace("c", {}) as c:
            c.log("sources", {"chunk_ids": ["d2:0"]})

        assert tracing.erase_traces({"d1"}) == 1
        assert tracing.erase_traces((), principal_id="p") == 1
        assert tracing.erase_traces({"d1"}, principal_id="p") == 0
        assert list(traces) == [c.id]
        assert tracing.get_trace(a.id) is None and tracing.get_trace(b.id) is None

    def test_purge_before(self, traces):
        with tracing.start_trace("old", {}) as old:
            pass
        with tracing.start_trace("new", {}) as new:
            pass
        traces[old.id]["created_at"] = (datetime.now(UTC) - timedelta(days=10)).isoformat()

        assert tracing.purge_traces(datetime.now(UTC) - timedelta(days=7)) == 1
        assert list(traces) == [new.id]

    def test_traces_for_principal(self, traces):
        with tracing.start_trace("a", {}, principal_id="p"):
            pass
        with tracing.start_trace("b", {}, principal_id="q"):
            pass
        assert [t["name"] for t in tracing.traces_for_principal("p")] == ["a"]
