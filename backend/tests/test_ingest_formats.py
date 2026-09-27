"""Tests for the ingest routes and pipeline around the parser registry."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.rag import parsers
from app.rag.ingest import enqueue_document
from app.rag.parsers import SUPPORTED_EXTENSIONS
from app.rag.store import StaleRevisions
from tests.test_parsers import make_docx, make_pdf


class TestFormatsRoute:
    def test_lists_extensions_from_the_registry(self, client):
        response = client.get("/ingest/formats")

        assert response.status_code == 200
        body = response.json()
        assert body["extensions"] == sorted(SUPPORTED_EXTENSIONS)
        assert {f["name"] for f in body["formats"]} >= {"pdf", "docx", "odt", "eml", "html"}
        docx = next(f for f in body["formats"] if f["name"] == "docx")
        assert docx["extensions"] == [".docx"]
        assert body["max_upload_mb"] == 25

    @pytest.mark.parametrize("available", [True, False])
    def test_reports_whether_ocr_can_run(self, client, available):
        with patch("app.rag.parsers.ocr.ocr_available", return_value=available):
            ocr = client.get("/ingest/formats").json()["ocr"]

        assert ocr["available"] is available
        assert ocr["active"] is available  # OCR_ENABLED defaults to true
        assert ocr["languages"] == "fra+eng"

    def test_ocr_is_inactive_when_disabled(self, client, settings, monkeypatch):
        monkeypatch.setattr(settings, "ocr_enabled", False)
        with patch("app.rag.parsers.ocr.ocr_available", return_value=True):
            ocr = client.get("/ingest/formats").json()["ocr"]
        assert ocr["enabled"] is False and ocr["active"] is False


class TestUploadRoute:
    def test_an_unsupported_type_is_a_400(self, client):
        response = client.post("/ingest", files={"file": ("a.exe", b"MZ", "application/x-msdownload")})
        assert response.status_code == 400
        assert ".docx" in response.json()["detail"]

    def test_a_supported_mime_type_is_accepted_without_extension(self, client):
        with patch(
            "app.api.ingest.enqueue_document",
            new=AsyncMock(return_value={"chunks": 1, "doc_id": "d", "filename": "scan"}),
        ) as enqueue, patch("app.api.ingest.store_count", new=AsyncMock(return_value=1)):
            response = client.post("/ingest", files={"file": ("scan", b"%PDF-1.4", "application/pdf")})
        assert response.status_code == 200
        enqueue.assert_awaited_once()

    def test_an_encrypted_file_is_a_422(self, client):
        response = client.post(
            "/ingest", files={"file": ("secret.pdf", make_pdf(password="pw"), "application/pdf")}
        )
        assert response.status_code == 422
        assert "password-protected" in response.json()["detail"]


@pytest.mark.asyncio
class TestEnqueueStructuredDocument:
    async def test_chunks_carry_heading_path_and_document_metadata(self):
        with (
            patch(
                "app.rag.ingest.embed_texts_async",
                new=AsyncMock(side_effect=lambda texts: [[0.0]] * len(texts)),
            ),
            patch("app.rag.ingest.upsert", new=AsyncMock()) as upsert,
            patch(
                "app.rag.ingest.delete_stale_revisions",
                new=AsyncMock(return_value=StaleRevisions()),
            ),
        ):
            await enqueue_document("rapport.docx", make_docx())

        chunks = upsert.await_args.args[0]
        assert [c.metadata["heading_path"] for c in chunks] == ["Chapitre 1", "Chapitre 1 > Budget"]
        meta = chunks[0].metadata["doc_metadata"]
        assert meta["source_format"] == "docx"
        assert meta["title"] == "Rapport annuel"
        assert chunks[0].metadata["filename"] == "rapport.docx"

    async def test_parsing_runs_off_the_event_loop(self):
        """OCR can take seconds a page; it must not stall other requests."""
        with (
            patch("app.rag.ingest.asyncio.to_thread", wraps=asyncio.to_thread) as to_thread,
            patch("app.rag.ingest.embed_texts_async", new=AsyncMock(return_value=[[0.0]])),
            patch("app.rag.ingest.upsert", new=AsyncMock()),
            patch(
                "app.rag.ingest.delete_stale_revisions",
                new=AsyncMock(return_value=StaleRevisions()),
            ),
        ):
            await enqueue_document("a.txt", b"hello world")
        # Graph-store cleanup also runs in a thread; parsing must be one of them.
        assert any(c.args[0] is parsers.extract for c in to_thread.call_args_list)
