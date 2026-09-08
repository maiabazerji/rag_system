"""Tests for ingestion and prompt rendering.

Both areas had silent-corruption bugs: chunking ignored its configuration, and
Jinja HTML-escaped every prompt before it reached the model.
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.prompts.loader import PromptManager
from app.rag.ingest import SUPPORTED_SUFFIXES, _doc_id, chunk_text, enqueue_document, load_document
from app.rag.response_clean import clean_response, extract_citations
from app.rag.store import StaleRevisions


class TestChunkText:
    def test_uses_configured_size_and_overlap(self, settings):
        """The regression: chunk_text ignored settings and used hardcoded 600/80."""
        text = " ".join(f"w{i}" for i in range(250))
        chunks = chunk_text(text)  # settings: size 100, overlap 20 -> step 80

        assert len(chunks[0].split()) == 100
        assert len(chunks) == 4  # ceil(250 / 80)

    def test_respects_a_changed_setting(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "chunk_size_tokens", 50)
        monkeypatch.setattr(settings, "chunk_overlap_tokens", 0)
        text = " ".join(f"w{i}" for i in range(100))

        chunks = chunk_text(text)
        assert len(chunks) == 2
        assert len(chunks[0].split()) == 50

    def test_explicit_arguments_win_over_settings(self):
        text = " ".join(f"w{i}" for i in range(60))
        assert len(chunk_text(text, size=30, overlap=0)) == 2

    def test_chunks_overlap(self):
        text = " ".join(f"w{i}" for i in range(30))
        chunks = chunk_text(text, size=20, overlap=10)
        assert chunks[0].split()[10:] == chunks[1].split()[:10]

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_overlap_not_smaller_than_size_is_rejected(self):
        """Otherwise the window never advances and chunking loops forever."""
        with pytest.raises(ValueError, match="smaller than size"):
            chunk_text("a b c", size=10, overlap=10)


class TestDocId:
    def test_is_derived_from_content(self):
        assert _doc_id("hello") == _doc_id("hello")

    def test_differs_for_different_content_of_the_same_length(self):
        """The regression: ids were hash(filename + length), so these collided."""
        assert _doc_id("aaaaa") != _doc_id("bbbbb")

    def test_is_independent_of_filename(self):
        assert _doc_id("same text") == _doc_id("same text")

    def test_ignores_whitespace_that_chunking_discards(self):
        """The regression: hashing raw bytes meant a file whose line endings
        changed got a second id, so the index kept both copies for ever."""
        assert _doc_id("a b\nc") == _doc_id("a  b\r\nc") == _doc_id("  a b c  ")

    def test_still_distinguishes_a_real_edit(self):
        assert _doc_id("the reranker is fast") != _doc_id("the reranker is slow")


class TestLoadDocument:
    def test_reads_utf8_text(self):
        assert load_document("a.md", b"# Title\nBody") == "# Title\nBody"

    def test_preserves_non_ascii(self):
        text = "café — naïve — 日本語"
        assert load_document("a.txt", text.encode("utf-8")) == text

    def test_rejects_an_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("archive.zip", b"PK\x03\x04")

    def test_rejects_invalid_utf8_instead_of_indexing_mojibake(self):
        """The old code decoded with errors='ignore' and indexed the garbage."""
        with pytest.raises(ValueError, match="not valid UTF-8"):
            load_document("a.txt", b"\xff\xfe\x00\x01binary")

    def test_reports_an_unreadable_pdf_clearly(self):
        with pytest.raises(ValueError, match="Could not read"):
            load_document("a.pdf", b"not actually a pdf")

    def test_supported_suffixes_include_the_documented_types(self):
        assert {".pdf", ".txt", ".md"} <= SUPPORTED_SUFFIXES


@pytest.mark.asyncio
class TestEnqueueDocument:
    async def test_indexes_chunks_and_reports_counts(self, settings):
        text = " ".join(f"w{i}" for i in range(250))

        with (
            patch(
                "app.rag.ingest.embed_texts_async",
                new=AsyncMock(return_value=[[0.0] * 384] * 4),
            ),
            patch("app.rag.ingest.upsert", new=AsyncMock()) as mock_upsert,
            patch(
                "app.rag.ingest.delete_stale_revisions",
                new=AsyncMock(return_value=StaleRevisions()),
            ),
        ):
            result = await enqueue_document("notes.md", text.encode("utf-8"))

        assert result["chunks"] == 4
        assert result["filename"] == "notes.md"
        assert result["doc_id"] == _doc_id(text)
        assert mock_upsert.await_count == 1

    async def test_removes_the_previous_revision_after_indexing_the_new_one(self):
        """Point ids derive from the content hash, so an edited document lands on
        fresh points and its old chunks survive unless they are deleted."""
        stale = StaleRevisions(points=3, doc_ids={"oldrevision"})
        with (
            patch("app.rag.ingest.embed_texts_async", new=AsyncMock(return_value=[[0.0]])),
            patch("app.rag.ingest.upsert", new=AsyncMock()) as mock_upsert,
            patch(
                "app.rag.ingest.delete_stale_revisions", new=AsyncMock(return_value=stale)
            ) as mock_delete,
            patch("app.rag.ingest.graph_store.remove_docs", return_value=7),
        ):
            result = await enqueue_document("a.txt", b"hello world")

        mock_delete.assert_awaited_once_with("a.txt", result["doc_id"])
        assert result["stale_chunks_removed"] == 3
        # The replacement has to be indexed before the old copy is dropped, so the
        # document is never briefly missing from search.
        assert mock_upsert.await_count == 1

    async def test_clears_the_graph_of_the_revision_it_just_dropped(self):
        """Triples are keyed to chunks, so a revision removed from the vector
        store leaves the graph pointing at text that is no longer retrievable."""
        stale = StaleRevisions(points=3, doc_ids={"oldrevision"})
        with (
            patch("app.rag.ingest.embed_texts_async", new=AsyncMock(return_value=[[0.0]])),
            patch("app.rag.ingest.upsert", new=AsyncMock()),
            patch(
                "app.rag.ingest.delete_stale_revisions", new=AsyncMock(return_value=stale)
            ),
            patch("app.rag.ingest.graph_store.remove_docs", return_value=7) as mock_graph,
        ):
            result = await enqueue_document("a.txt", b"hello world")

        mock_graph.assert_called_once_with({"oldrevision"})
        assert result["stale_triples_removed"] == 7

    async def test_rejects_a_document_with_no_extractable_text(self):
        with pytest.raises(ValueError, match="No text could be extracted"):
            await enqueue_document("empty.txt", b"   \n\n  ")

    async def test_uses_the_async_embedder(self):
        """Blocking embed_texts in an async handler stalled the whole event loop."""
        with (
            patch(
                "app.rag.ingest.embed_texts_async", new=AsyncMock(return_value=[[0.0]])
            ) as mock_embed,
            patch("app.rag.ingest.upsert", new=AsyncMock()),
            patch(
                "app.rag.ingest.delete_stale_revisions",
                new=AsyncMock(return_value=StaleRevisions()),
            ),
        ):
            await enqueue_document("a.txt", b"hello world")
        assert mock_embed.await_count == 1


class TestPromptRendering:
    """Prompts go to a model, not a browser: nothing may be HTML-escaped."""

    def test_apostrophes_survive(self, tmp_path):
        (tmp_path / "t.md").write_text("Q: {{ question }}", encoding="utf-8")
        out = PromptManager(tmp_path).render_prompt("t", question="What's RAG?")
        assert out == "Q: What's RAG?"
        assert "&#39;" not in out

    def test_angle_brackets_and_ampersands_survive(self, tmp_path):
        (tmp_path / "t.md").write_text("{{ context }}", encoding="utf-8")
        context = "if (a < b && c > d) { return \"x\"; }"
        out = PromptManager(tmp_path).render_prompt("t", context=context)
        assert out == context
        assert "&amp;" not in out and "&lt;" not in out

    def test_the_shipped_default_template_renders_verbatim(self):
        from app.prompts.loader import render_prompt

        out = render_prompt("default", question="What's <this> & that?", context="a < b")
        assert "What's <this> & that?" in out
        assert "a < b" in out
        assert "&" in out and "&amp;" not in out

    def test_unknown_version_falls_back_to_default(self):
        from app.prompts.loader import render_prompt

        out = render_prompt("no-such-version", question="q", context="c")
        assert "q" in out and "c" in out

    def test_missing_variable_is_an_error_not_a_blank(self, tmp_path):
        from jinja2 import UndefinedError

        (tmp_path / "t.md").write_text("{{ question }} {{ context }}", encoding="utf-8")
        with pytest.raises(UndefinedError):
            PromptManager(tmp_path).render_prompt("t", question="only one")


class TestResponseClean:
    """Citations are the product; cleaning must not delete them."""

    def test_citations_are_preserved(self):
        text = "RAG combines retrieval and generation [9fa3c1b0e2:4]."
        assert clean_response(text) == text

    def test_code_blocks_are_preserved(self):
        text = "Use this:\n```python\nprint('hi')\n```"
        assert "```python" in clean_response(text)

    def test_headings_are_preserved(self):
        assert clean_response("## Overview\nText") == "## Overview\nText"

    def test_json_envelope_is_unwrapped(self):
        assert clean_response('{"answer": "The real answer."}') == "The real answer."

    def test_a_json_looking_answer_is_left_alone(self):
        assert clean_response('{"not_an_answer": 1}') == '{"not_an_answer": 1}'

    def test_excess_blank_lines_collapse(self):
        assert clean_response("a\n\n\n\n\nb") == "a\n\nb"

    def test_empty_input_round_trips(self):
        assert clean_response("") == ""

    def test_em_dashes_become_commas(self):
        assert (
            clean_response("It detects signals—like negation—that bi-encoders miss.")
            == "It detects signals, like negation, that bi-encoders miss."
        )

    def test_a_spaced_dash_joins_two_clauses(self):
        assert (
            clean_response("Reranking is slow — so run it on the top 50 only.")
            == "Reranking is slow, so run it on the top 50 only."
        )

    def test_a_numeric_range_becomes_a_hyphen(self):
        assert clean_response("Recall improves 15–25%.") == "Recall improves 15-25%."

    def test_a_dash_inside_code_is_left_alone(self):
        text = "Run:\n```bash\nsearch --top-k 50 — verbose\n```"
        assert "—" in clean_response(text)

    def test_a_dash_bullet_stays_a_bullet(self):
        assert clean_response("— first\n— second") == "- first\n- second"

    def test_a_dash_after_inline_code_is_punctuation_not_a_bullet(self):
        """The regression: splitting on code spans made the following text look
        like the start of a line, so the dash was rewritten as a list bullet."""
        assert (
            clean_response("Use `--top-k` — it controls depth.")
            == "Use `--top-k`, it controls depth."
        )

    def test_a_trailing_dash_is_dropped(self):
        assert clean_response("An unfinished thought —") == "An unfinished thought"

    def test_citations_survive_dash_rewriting(self):
        assert (
            clean_response("Precision ranks chunks — recall covers them [9fa3c1b0e2:4].")
            == "Precision ranks chunks, recall covers them [9fa3c1b0e2:4]."
        )

    def test_extract_citations_finds_chunk_ids(self):
        text = "See [9fa3c1b0e2:4] and [9fa3c1b0e2:7], plus [9fa3c1b0e2:4] again."
        assert extract_citations(text) == ["9fa3c1b0e2:4", "9fa3c1b0e2:7"]

    def test_extract_citations_on_plain_text(self):
        assert extract_citations("No citations here.") == []
