"""Tests for the document parsers and the format registry.

Every fixture file is built in the test itself (python-docx, odfpy,
python-pptx, pypdf, the email package), so the suite needs no binary fixtures.
"""
from email.message import EmailMessage
from unittest.mock import MagicMock, patch

import pytest
from document_builders import (
    make_docx,
    make_eml,
    make_ods,
    make_odt,
    make_pdf,
    make_pptx,
    zip_bytes,
)

from app.rag import parsers
from app.rag.parsers import (
    ArchiveTooLargeError,
    EncryptedDocumentError,
    ParseError,
    UnsupportedFormatError,
    extract,
)
from app.rag.parsers.html import html_to_text

# --- formats -----------------------------------------------------------------


class TestDocx:
    def test_headings_lists_and_tables_become_markdown(self):
        text = extract("rapport.docx", make_docx()).text

        assert "# Chapitre 1" in text
        assert "## Budget" in text
        assert "- premier point" in text
        assert "| Poste | Montant |\n| --- | --- |\n| Loyer | 1200 |" in text

    def test_headers_and_footers_are_skipped(self):
        text = extract("rapport.docx", make_docx()).text
        assert "EN-TÊTE" not in text
        assert "PIED DE PAGE" not in text

    def test_core_properties_become_metadata(self):
        meta = extract("rapport.docx", make_docx()).metadata

        assert meta.source_format == "docx"
        assert meta.title == "Rapport annuel"
        assert meta.author == "Marie Curie"
        assert meta.language == "fr-FR"
        assert meta.created is not None

    def test_sections_carry_the_heading_path(self):
        document = extract("rapport.docx", make_docx())
        assert [s.breadcrumb for s in document.sections] == [
            "Chapitre 1",
            "Chapitre 1 > Budget",
        ]

    def test_a_password_protected_file_is_rejected_clearly(self):
        # Encrypted OOXML is an OLE2 container with an EncryptedPackage stream.
        ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
        content = ole + "EncryptedPackage".encode("utf-16-le")
        with pytest.raises(EncryptedDocumentError, match="password-protected"):
            extract("secret.docx", content)

    def test_a_corrupt_file_is_a_parse_error(self):
        with pytest.raises(ParseError, match="corrupt"):
            extract("broken.docx", b"PK\x03\x04 truncated")


class TestOdf:
    def test_odt_headings_lists_and_tables(self):
        text = extract("arrete.odt", make_odt()).text

        assert "# Titre I" in text
        assert "## Dispositions générales" in text
        assert "- premier alinéa" in text
        assert "| Commune | Population |" in text
        assert "| Lyon | 522000 |" in text

    def test_odt_metadata(self):
        meta = extract("arrete.odt", make_odt()).metadata
        assert meta.source_format == "odt"
        assert meta.title == "Arrêté préfectoral"
        assert meta.author == "Préfecture"
        assert meta.language == "fr-FR"

    def test_ods_sheet_becomes_a_table_under_its_name(self):
        document = extract("budget.ods", make_ods())

        assert document.metadata.source_format == "ods"
        assert document.text.startswith("## Budget")
        assert "| Loyer | 1200 |" in document.text

    def test_ods_repeated_empty_cells_are_not_expanded(self):
        """A million declared empty rows must not become a million lines."""
        text = extract("budget.ods", make_ods()).text
        assert len(text.splitlines()) == 5
        assert "|  |" not in text

    def test_an_encrypted_odf_is_rejected(self):
        content = zip_bytes(
            {
                "mimetype": b"application/vnd.oasis.opendocument.text",
                "META-INF/manifest.xml": b"<manifest:encryption-data/>",
                "content.xml": b"<x/>",
            }
        )
        with pytest.raises(EncryptedDocumentError):
            extract("secret.odt", content)


class TestPptx:
    def test_one_heading_per_slide_with_tables(self):
        document = extract("bilan.pptx", make_pptx())

        assert "## Slide 1: Introduction" in document.text
        assert "- Objectifs atteints" in document.text
        assert "| Clé | Valeur |" in document.text
        # The title is the heading; it is not repeated as a bullet.
        assert "- Introduction" not in document.text
        assert document.metadata.page_count == 2
        assert document.metadata.title == "Bilan"


class TestHtml:
    PAGE = """<!DOCTYPE html><html lang="fr"><head><title>Guide</title>
        <meta name="author" content="Service public">
        <style>body { color: red }</style><script>alert("x")</script></head>
        <body><h1>Démarches</h1><p>Un <b>texte</b>
        sur deux lignes.</p>
        <ul><li>un</li><li>deux<ul><li>sous-point</li></ul></li></ul>
        <table><tr><th>Pièce</th><th>Délai</th></tr><tr><td>CNI</td><td>3 semaines</td></tr></table>
        <h2>Suite</h2><p>Fin<br>du texte</p><noscript>activez JS</noscript></body></html>"""

    def test_structure_is_preserved(self):
        text = extract("guide.html", self.PAGE.encode()).text

        assert "# Démarches" in text
        assert "## Suite" in text
        assert "Un texte sur deux lignes." in text
        assert "- deux\n  - sous-point" in text
        assert "| Pièce | Délai |\n| --- | --- |\n| CNI | 3 semaines |" in text
        assert "Fin\ndu texte" in text

    def test_script_style_and_noscript_are_dropped(self):
        text = extract("guide.html", self.PAGE.encode()).text
        assert "alert" not in text
        assert "color" not in text
        assert "activez" not in text
        assert "Guide" not in text  # the <title> is metadata, not body

    def test_metadata_from_markup(self):
        meta = extract("guide.htm", self.PAGE.encode()).metadata
        assert (meta.source_format, meta.title, meta.author, meta.language) == (
            "html", "Guide", "Service public", "fr",
        )

    def test_an_unclosed_head_does_not_swallow_the_body(self):
        """<head> may legally be left unclosed; <body> still ends it."""
        assert html_to_text("<html><head><title>t</title><body><p>visible</p>") == "visible"

    def test_omitted_cell_and_row_end_tags_keep_every_cell(self):
        markup = "<table><tr><th>A<th>B<tr><td>1<td>2</table>"
        assert html_to_text(markup) == "| A | B |\n| --- | --- |\n| 1 | 2 |"

    def test_single_column_layout_tables_are_unwrapped(self):
        markup = "<table><tr><td><h1>Newsletter</h1></td></tr><tr><td>Bonjour</td></tr></table>"
        text = html_to_text(markup)
        assert "|" not in text
        assert "Newsletter" in text and "Bonjour" in text

    def test_declared_charset_is_honoured(self):
        markup = '<html><head><meta charset="iso-8859-1"></head><body>café</body></html>'
        assert extract("latin.html", markup.encode("latin-1")).text == "café"


class TestEml:
    def test_headers_become_metadata(self):
        meta = extract("cr.eml", make_eml()).metadata

        assert meta.source_format == "eml"
        assert meta.title == "Compte rendu de réunion"
        assert meta.author == "Alice Martin <alice@example.fr>"
        assert meta.created == "2026-03-02T10:00:00+01:00"
        assert meta.language == "fr"
        assert meta.extra["to"] == "bob@example.fr"

    def test_plain_text_body_is_preferred_over_html(self):
        text = extract("cr.eml", make_eml()).text
        assert "Voici le compte rendu." in text
        assert "Version HTML" not in text
        assert text.startswith("# Compte rendu de réunion")

    def test_html_only_body_is_converted_safely(self):
        message = EmailMessage()
        message["Subject"] = "Lettre"
        message["From"] = "a@example.fr"
        message.set_content(
            "<style>p{}</style><script>evil()</script><p>Corps</p>", subtype="html"
        )
        text = extract("lettre.eml", message.as_bytes()).text
        assert "Corps" in text
        assert "evil" not in text and "p{}" not in text

    def test_attachments_are_listed_and_supported_ones_parsed(self):
        content = make_eml(
            attachments=[
                ("annexe.md", b"# Annexe\n\nContenu de l'annexe.", "text", "markdown"),
                ("rapport.docx", make_docx(), "application", "octet-stream"),
                ("photo.bin", b"\x00\x01\x02", "application", "octet-stream"),
            ]
        )
        document = extract("cr.eml", content)

        assert "- annexe.md" in document.text
        assert "- photo.bin" in document.text
        assert "Contenu de l'annexe." in document.text
        assert "| Loyer | 1200 |" in document.text
        # Attachment headings nest under the attachment's own heading.
        breadcrumbs = [s.breadcrumb for s in document.sections]
        assert "Compte rendu de réunion > Attachment: annexe.md > Annexe" in breadcrumbs

    def test_a_forwarded_message_is_parsed(self):
        forwarded = EmailMessage()
        forwarded["Subject"] = "Message transféré"
        forwarded["From"] = "carole@example.fr"
        forwarded.set_content("Contenu transféré.")
        message = EmailMessage()
        message["Subject"] = "Fwd"
        message["From"] = "a@example.fr"
        message.set_content("Voir ci-dessous.")
        message.add_attachment(forwarded)

        text = extract("fwd.eml", message.as_bytes()).text
        assert "Contenu transféré." in text
        assert "### Message transféré" in text

    def test_nested_emails_stop_at_the_depth_limit(self):
        inner = make_eml(attachments=[("deep.md", b"TRES PROFOND", "text", "markdown")])
        middle = make_eml(attachments=[("inner.eml", inner, "application", "octet-stream")])
        outer = make_eml(attachments=[("middle.eml", middle, "application", "octet-stream")])

        text = extract("outer.eml", outer).text
        assert "- inner.eml" in text  # listed at depth 2...
        assert "- deep.md" in text
        assert "TRES PROFOND" not in text  # ...but not parsed past the limit

    def test_oversized_attachments_are_listed_not_parsed(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "max_upload_mb", 1)
        big = b"mot " * 300_000  # 1.2 MB
        text = extract("cr.eml", make_eml([("gros.txt", big, "text", "plain")])).text
        assert "- gros.txt" in text
        assert "mot mot" not in text

    def test_a_file_with_no_headers_is_rejected(self):
        with pytest.raises(ParseError, match="not an .eml"):
            extract("vide.eml", b"just some words")


class TestPdf:
    def test_an_encrypted_pdf_is_rejected_clearly(self):
        with pytest.raises(EncryptedDocumentError, match="password-protected"):
            extract("secret.pdf", make_pdf(password="secret"))

    def test_page_count_is_recorded(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "ocr_enabled", False)
        assert extract("blank.pdf", make_pdf(pages=3)).metadata.page_count == 3

    def test_scanned_pages_are_ocrd(self, settings, monkeypatch):
        """pytesseract and pdf2image are mocked: the path, not tesseract, is under test."""
        monkeypatch.setattr(settings, "ocr_languages", "fra+eng")
        image = MagicMock()
        with (
            patch("app.rag.parsers.ocr.ocr_available", return_value=True),
            patch("pdf2image.convert_from_bytes", return_value=[image]) as convert,
            patch("pytesseract.image_to_string", return_value="Texte numérisé") as ocr,
        ):
            document = extract("scan.pdf", make_pdf(pages=2))

        assert document.text.count("Texte numérisé") == 2
        assert document.metadata.extra["ocr_pages"] == 2
        ocr.assert_called_with(image, lang="fra+eng")
        assert convert.call_args.kwargs["first_page"] == 2

    def test_ocr_stops_at_the_page_cap(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "ocr_max_pages", 1)
        with (
            patch("app.rag.parsers.ocr.ocr_available", return_value=True),
            patch("app.rag.parsers.ocr.ocr_pdf_page", return_value="Texte") as ocr,
        ):
            document = extract("scan.pdf", make_pdf(pages=3))

        assert ocr.call_count == 1
        assert any("OCR_MAX_PAGES" in w for w in document.warnings)

    def test_missing_ocr_binaries_warn_and_keep_the_text(self):
        with (
            patch("app.rag.parsers.ocr.ocr_available", return_value=False),
            patch("app.rag.parsers.ocr.ocr_pdf_page") as ocr,
        ):
            document = extract("scan.pdf", make_pdf())

        ocr.assert_not_called()
        assert any("OCR is unavailable" in w for w in document.warnings)

    def test_ocr_can_be_disabled(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "ocr_enabled", False)
        with patch("app.rag.parsers.ocr.ocr_pdf_page") as ocr:
            extract("scan.pdf", make_pdf())
        ocr.assert_not_called()

    def test_a_failed_page_keeps_its_extracted_text(self):
        with (
            patch("app.rag.parsers.ocr.ocr_available", return_value=True),
            patch("app.rag.parsers.ocr.ocr_pdf_page", side_effect=RuntimeError("boom")),
        ):
            document = extract("scan.pdf", make_pdf())
        assert "ocr_pages" not in document.metadata.extra


class TestCsv:
    def test_becomes_a_markdown_table(self):
        text = extract("t.csv", "nom;âge\nAlice;30\nBob;25\n".encode()).text
        assert text == "| nom | âge |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"


# --- safety ------------------------------------------------------------------


class TestZipBombGuard:
    def test_total_uncompressed_size_is_capped(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "max_uncompressed_mb", 1)
        content = zip_bytes(
            {"word/document.xml": b"<w/>", "word/media/big.bin": b"\x00" * (2 * 1024 * 1024)}
        )
        with pytest.raises(ArchiveTooLargeError, match="MAX_UNCOMPRESSED_MB"):
            extract("bomb.docx", content)

    def test_extreme_compression_ratio_is_refused(self):
        content = zip_bytes({"content.xml": b"\x00" * (8 * 1024 * 1024)})
        with pytest.raises(ArchiveTooLargeError, match="zip bomb"):
            extract("bomb.odt", content)

    def test_the_guard_runs_before_the_parser(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "max_uncompressed_mb", 1)
        content = zip_bytes({"word/document.xml": b"<w/>" * (400 * 1024)})
        with patch("docx.Document") as document:
            with pytest.raises(ArchiveTooLargeError):
                extract("bomb.docx", content)
        document.assert_not_called()

    def test_errors_are_value_errors_so_the_api_answers_422(self):
        assert issubclass(ArchiveTooLargeError, ValueError)
        assert issubclass(EncryptedDocumentError, ValueError)


# --- registry ----------------------------------------------------------------


class TestRegistry:
    def test_every_documented_extension_is_supported(self):
        expected = {
            ".pdf", ".txt", ".md", ".markdown", ".rst", ".csv", ".json",
            ".docx", ".odt", ".ods", ".eml", ".html", ".htm", ".pptx",
        }
        assert expected == set(parsers.SUPPORTED_EXTENSIONS)

    def test_unknown_extensions_are_rejected(self):
        with pytest.raises(UnsupportedFormatError, match="Unsupported file type '.zip'"):
            extract("archive.zip", b"PK\x03\x04")

    def test_content_wins_over_a_wrong_extension(self):
        document = extract("rapport.pdf", make_docx())
        assert document.metadata.source_format == "docx"

    def test_mime_type_is_used_without_an_extension(self):
        assert parsers.is_supported("scan", "application/pdf; charset=binary")
        assert not parsers.is_supported("scan", "application/zip")
        fmt = parsers.detect_format("upload", b"<p>x</p>", "text/html")
        assert fmt is not None and fmt.name == "html"

    def test_content_is_sniffed_without_extension_or_mime(self):
        assert parsers.detect_format("rapport", make_docx()).name == "docx"
        assert parsers.detect_format("tableur", make_ods()).name == "ods"
        assert parsers.detect_format("page", b"<!DOCTYPE html><p>x</p>").name == "html"
        assert parsers.detect_format("mail", make_eml()).name == "eml"
        assert parsers.detect_format("notes", "du texte éè".encode()).name == "text"
        assert parsers.detect_format("binaire", b"\x00\xff\x00") is None
