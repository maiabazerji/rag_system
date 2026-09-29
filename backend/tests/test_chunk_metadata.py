"""Tests for page tracking and the citation metadata every indexed chunk carries."""
from unittest.mock import AsyncMock, patch

import pytest
from document_builders import make_docx, make_pptx, make_text_pdf

from app.privacy.pii import redact
from app.rag import parsers
from app.rag.ingest import _acl_namespace, _doc_id, enqueue_document
from app.rag.store import StaleRevisions

CITATION_FIELDS = {
    "document_id", "chunk_id", "chunk_index", "source", "filename", "title",
    "section", "heading_path", "headings", "page", "page_end", "char_start", "char_end",
    "token_count",
}

PAGES = [
    ["# Rapport annuel", "", "Premiere page du rapport. Elle presente le contexte."],
    ["Deuxieme page. Les chiffres du trimestre sont stables."],
    [],  # a page with no text keeps its number: the next page is page 4
    ["## Conclusion", "", "Quatrieme page. Le bilan est positif."],
]


@pytest.fixture(autouse=True)
def _no_ocr(settings, monkeypatch):
    monkeypatch.setattr(settings, "ocr_enabled", False)


async def ingest(filename, content, **kwargs):
    """Run ingestion with the embedder and the vector store mocked; return the chunks."""
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
        result = await enqueue_document(filename, content, **kwargs)
    return result, upsert.await_args.args[0]


def indexed_text(filename, content):
    """The normalised text ingestion chunks: parsed, then PII-masked."""
    return redact(parsers.extract(filename, content).text, "mask").text


class TestPageSpans:
    def test_pdf_pages_map_to_their_text(self):
        parsed = parsers.extract("rapport.pdf", make_text_pdf(PAGES))

        assert [p.number for p in parsed.pages] == [1, 2, 4]
        by_number = {p.number: parsed.text[p.start : p.end] for p in parsed.pages}
        assert by_number[1].startswith("# Rapport") and by_number[1].endswith("contexte.")
        assert by_number[2] == "Deuxieme page. Les chiffres du trimestre sont stables."
        assert by_number[4].startswith("## Conclusion")
        assert parsed.metadata.page_count == 4

    def test_pdf_text_is_unchanged_by_page_tracking(self):
        """Page tracking must not move the text, or document ids would change."""
        parsed = parsers.extract("rapport.pdf", make_text_pdf(PAGES))
        assert "\n\n\n" not in parsed.text
        joined = "\n\n".join(parsed.text[p.start : p.end] for p in parsed.pages)
        assert joined == parsed.text

    def test_slides_are_pages(self):
        parsed = parsers.extract("bilan.pptx", make_pptx())
        assert [p.number for p in parsed.pages] == [1, 2]
        assert parsed.text[parsed.pages[1].start :].startswith("## Slide 2: Chiffres")

    def test_formats_without_pages_have_none(self):
        assert parsers.extract("rapport.docx", make_docx()).pages == []
        assert parsers.extract("notes.md", b"# Notes\n\nTexte.").pages == []


class TestRedactionOffsets:
    def test_moved_maps_offsets_onto_the_masked_text(self):
        text = "Ecrire a jean.dupont@example.fr avant lundi. Fin."
        result = redact(text, "mask")
        before, after = text.index("avant"), result.text.index("avant")
        assert result.moved(before) == after
        assert result.moved(0) == 0
        assert result.moved(len(text)) == len(result.text)
        inside = text.index("dupont")
        assert result.moved(inside) == result.text.index("[EMAIL]")

    def test_nothing_masked_moves_nothing(self):
        assert redact("Rien.", "mask").moved(3) == 3
        assert redact("a@b.fr", "off").moved(3) == 3


@pytest.mark.asyncio
class TestChunkPayload:
    async def test_markdown_chunks_carry_every_citation_field(self):
        content = (
            b"# Guide\n\nIntroduction du guide. Elle est courte.\n\n"
            b"## Installation\n\nLancer le script. Puis redemarrer."
        )
        _, chunks = await ingest("docs/guide-utilisateur.md", content)
        text = indexed_text("guide-utilisateur.md", content)

        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            assert meta.keys() >= CITATION_FIELDS
            assert meta["document_id"] == chunk.doc_id
            assert meta["chunk_id"] == chunk.id == f"{chunk.doc_id}:{i}"
            assert meta["chunk_index"] == i
            assert meta["source"] == meta["filename"] == "docs/guide-utilisateur.md"
            assert meta["title"] == "guide-utilisateur"  # no title metadata: file stem
            assert meta["page"] is None and meta["page_end"] is None
            assert meta["token_count"] == chunk.tokens == len(chunk.text.split())
            assert text[meta["char_start"] : meta["char_end"]] == chunk.text
        assert [c.metadata["section"] for c in chunks] == ["Guide", "Installation"]
        assert chunks[1].metadata["heading_path"] == "Guide > Installation"
        assert chunks[1].metadata["headings"] == ["Guide", "Installation"]
        assert chunks[1].section == "Installation"

    async def test_existing_payload_fields_are_kept(self):
        _, chunks = await ingest("a.md", b"# A\n\nTexte.", owner="key:1")
        meta = chunks[0].metadata
        for field in ("filename", "source_key", "owner", "ingested_at", "pii_counts",
                      "heading_path", "doc_metadata", "tenant", "acl_groups"):
            assert field in meta
        assert meta["source_key"] == "key:1:a.md"

    async def test_docx_title_and_sections(self):
        content = make_docx()
        _, chunks = await ingest("rapport.docx", content)
        text = indexed_text("rapport.docx", content)

        assert {c.metadata["title"] for c in chunks} == {"Rapport annuel"}
        assert [c.metadata["section"] for c in chunks] == ["Chapitre 1", "Budget"]
        for chunk in chunks:
            meta = chunk.metadata
            assert text[meta["char_start"] : meta["char_end"]] == chunk.text
            assert meta["page"] is None

    async def test_pdf_chunks_know_their_pages(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "chunk_size_tokens", 100)
        monkeypatch.setattr(settings, "chunk_overlap_tokens", 0)
        content = make_text_pdf(PAGES, title="Rapport annuel 2026")
        _, chunks = await ingest("rapport.pdf", content)
        text = indexed_text("rapport.pdf", content)

        assert {c.metadata["title"] for c in chunks} == {"Rapport annuel 2026"}
        pages = {c.metadata["section"]: (c.metadata["page"], c.metadata["page_end"]) for c in chunks}
        assert pages == {"Rapport annuel": (1, 2), "Conclusion": (4, 4)}
        for chunk in chunks:
            meta = chunk.metadata
            assert text[meta["char_start"] : meta["char_end"]] == chunk.text

    async def test_pdf_page_per_chunk_with_small_chunks(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "chunk_size_tokens", 100)
        monkeypatch.setattr(settings, "chunk_overlap_tokens", 0)
        pages = [[f"Page {n} phrase {i} du document de test." for i in range(12)] for n in range(1, 6)]
        _, chunks = await ingest("long.pdf", make_text_pdf(pages))

        assert len(chunks) > 1
        assert chunks[0].metadata["page"] == 1
        assert chunks[-1].metadata["page_end"] == 5
        for chunk in chunks:
            meta = chunk.metadata
            first, last = meta["page"], meta["page_end"]
            assert first <= last
            assert f"Page {first} phrase" in chunk.text and f"Page {last} phrase" in chunk.text
        pages_seen = [c.metadata["page"] for c in chunks]
        assert pages_seen == sorted(pages_seen)

    async def test_page_numbers_survive_pii_masking(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "chunk_size_tokens", 100)
        monkeypatch.setattr(settings, "chunk_overlap_tokens", 0)
        pages = [
            ["Contact: jean.dupont@example.fr et marie.curie@example.fr pour la suite."] * 8,
            ["Deuxieme page sans donnees personnelles. Elle parle du budget."] * 8,
        ]
        content = make_text_pdf(pages)
        result, chunks = await ingest("contacts.pdf", content)
        text = indexed_text("contacts.pdf", content)

        assert result["pii_counts"]["EMAIL"] == 16
        for chunk in chunks:
            meta = chunk.metadata
            assert text[meta["char_start"] : meta["char_end"]] == chunk.text
            if "Deuxieme page" in chunk.text and "Contact" not in chunk.text:
                assert meta["page"] == 2
            if "Contact" in chunk.text and "Deuxieme" not in chunk.text:
                assert meta["page_end"] == 1

    async def test_pptx_slides_are_pages(self):
        _, chunks = await ingest("bilan.pptx", make_pptx())
        assert [(c.metadata["page"], c.metadata["page_end"]) for c in chunks] == [(1, 1), (2, 2)]
        assert chunks[0].metadata["title"] == "Bilan"

    @pytest.mark.parametrize("strategy", ["structured", "fixed"])
    async def test_doc_ids_do_not_depend_on_chunking(self, settings, monkeypatch, strategy):
        monkeypatch.setattr(settings, "chunk_strategy", strategy)
        content = make_text_pdf(PAGES)
        first, chunks = await ingest("rapport.pdf", content)
        second, again = await ingest("rapport.pdf", content)

        expected = _doc_id(indexed_text("rapport.pdf", content), _acl_namespace(settings.default_tenant, ["public"]))
        assert first["doc_id"] == second["doc_id"] == expected
        assert [c.id for c in chunks] == [c.id for c in again]
        assert [c.metadata["char_start"] for c in chunks] == [c.metadata["char_start"] for c in again]

    async def test_fixed_strategy_chunks_carry_the_same_fields(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "chunk_strategy", "fixed")
        _, chunks = await ingest("rapport.pdf", make_text_pdf(PAGES))
        for chunk in chunks:
            assert chunk.metadata.keys() >= CITATION_FIELDS
            assert chunk.metadata["page"] == 1
