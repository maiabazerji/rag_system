"""Tests for heading detection and structure-aware chunking."""
import pytest

from app.rag.chunking import chunk_structured
from app.rag.parsers.structure import detect_heading, split_sections

CODE_EXCERPT = """Code de la consommation

Titre II : Pratiques commerciales

Chapitre Ier : Pratiques commerciales interdites

Section 1 : Pratiques commerciales déloyales

Article L. 121-1

Les pratiques commerciales déloyales sont interdites. Une pratique commerciale
est déloyale lorsqu'elle est contraire aux exigences de la diligence
professionnelle.

Article L. 121-2

Une pratique commerciale est trompeuse si elle est commise dans l'une des
circonstances suivantes.

Section 2 : Pratiques commerciales agressives

Article L. 121-6

Une pratique commerciale est agressive lorsqu'elle contraint le consommateur.

Chapitre II : Dispositions pénales

Article 5

Les infractions sont punies d'une amende.

Art. 12. - Le présent décret entre en vigueur le lendemain de sa publication au Journal officiel.
"""


def paths(text, **kwargs):
    return [c.heading_path for c in chunk_structured(text, **kwargs)]


class TestDetectHeading:
    @pytest.mark.parametrize(
        "line, label",
        [
            ("Titre I", "Titre I"),
            ("TITRE IER DISPOSITIONS GÉNÉRALES", "TITRE IER DISPOSITIONS GÉNÉRALES"),
            ("Chapitre 2", "Chapitre 2"),
            ("Chapitre II - Des contrats", "Chapitre II - Des contrats"),
            ("Section 3", "Section 3"),
            ("Sous-section 1 : Champ d'application", "Sous-section 1 : Champ d'application"),
            ("Article L. 121-1", "Article L. 121-1"),
            ("Article L121-1", "Article L121-1"),
            ("Article R*123-4", "Article R*123-4"),
            ("Article 5", "Article 5"),
            ("Article 1er", "Article 1er"),
            ("Article premier", "Article premier"),
            ("Article 5 : Objet du contrat", "Article 5 : Objet du contrat"),
            ("Art. 12", "Art. 12"),
            ("## Annexe technique", "Annexe technique"),
        ],
    )
    def test_recognises_headings(self, line, label):
        heading = detect_heading(line)
        assert heading is not None
        assert heading.label == label

    @pytest.mark.parametrize(
        "line",
        [
            "Article 5 dispose que le contrat est nul.",
            "Section 3 du présent chapitre s'applique.",
            "article 5",  # lower case: a sentence fragment, not a heading
            "Articles 5 et 6",
            "Le titre I est abrogé.",
            "| Article 5 | abrogé |",
            "Titre de séjour",
        ],
    )
    def test_ignores_prose(self, line):
        assert detect_heading(line) is None

    def test_an_article_line_with_its_text_is_an_inline_heading(self):
        heading = detect_heading("Art. 12. - Le présent décret entre en vigueur demain.")
        assert heading is not None
        assert heading.label == "Art. 12"
        assert heading.inline_body


class TestSplitSections:
    def test_legal_structure_nests(self):
        sections = split_sections(CODE_EXCERPT)
        crumbs = [s.breadcrumb for s in sections]

        assert crumbs[0] == ""
        assert (
            "Titre II : Pratiques commerciales > Chapitre Ier : Pratiques commerciales "
            "interdites > Section 1 : Pratiques commerciales déloyales > Article L. 121-1"
        ) in crumbs
        # A new section closes the previous section's articles.
        assert any(c.endswith("Section 2 : Pratiques commerciales agressives > Article L. 121-6") for c in crumbs)
        # A new chapter closes the previous chapter's sections.
        assert (
            "Titre II : Pratiques commerciales > Chapitre II : Dispositions pénales > Article 5"
        ) in crumbs

    def test_bare_articles_nest_under_markdown_divisions(self):
        text = "# Chapitre 2\n\nArticle 5\n\nTexte.\n\n# Chapitre 3\n\nArticle 6\n\nAutre."
        crumbs = [s.breadcrumb for s in split_sections(text)]
        assert crumbs == [
            "Chapitre 2", "Chapitre 2 > Article 5", "Chapitre 3", "Chapitre 3 > Article 6",
        ]

    def test_markdown_levels_nest(self):
        text = "# Chapitre 1\n\nIntro\n\n## Budget\n\nChiffres\n\n# Annexe\n\nFin"
        crumbs = [s.breadcrumb for s in split_sections(text)]
        assert crumbs == ["Chapitre 1", "Chapitre 1 > Budget", "Annexe"]

    def test_headings_inside_code_fences_are_ignored(self):
        text = "# Guide\n\n```\n# not a heading\nArticle 5\n```\n"
        assert [s.breadcrumb for s in split_sections(text)] == ["Guide"]

    def test_sections_cover_the_text(self):
        sections = split_sections(CODE_EXCERPT)
        assert sections[0].start == 0
        assert sections[-1].end == len(CODE_EXCERPT)
        for before, after in zip(sections, sections[1:], strict=False):
            assert before.end == after.start


class TestLegalChunking:
    def test_one_chunk_per_article_with_its_breadcrumb(self):
        chunks = chunk_structured(CODE_EXCERPT, size=100, overlap=10)
        by_path = {c.heading_path.split(" > ")[-1]: c.text for c in chunks}

        assert by_path["Article L. 121-1"].startswith("Article L. 121-1")
        assert "diligence" in by_path["Article L. 121-1"]
        assert "trompeuse" not in by_path["Article L. 121-1"]
        assert "agressive" in by_path["Article L. 121-6"]
        assert "amende" in by_path["Article 5"]
        assert by_path["Art. 12"].startswith("Art. 12. - Le présent décret")

    def test_heading_only_sections_produce_no_chunk(self):
        texts = [c.text for c in chunk_structured(CODE_EXCERPT, size=100, overlap=10)]
        assert "Titre II : Pratiques commerciales" not in texts
        assert "Chapitre II : Dispositions pénales" not in texts

    def test_a_long_article_is_split_by_size_under_the_same_heading(self):
        text = "Chapitre 2\n\nArticle 5\n\n" + " ".join(f"mot{i}" for i in range(250))
        chunks = chunk_structured(text, size=100, overlap=20)

        assert len(chunks) == 4
        assert {c.heading_path for c in chunks} == {"Chapitre 2 > Article 5"}
        first, second = chunks[0].text.split(), chunks[1].text.split()
        assert first[-20:] == second[:20]


class TestTableChunking:
    TABLE = "\n".join(
        ["| Commune | Code | Population |", "| --- | --- | --- |"]
        + [f"| Ville{i} | {i:05d} | {i * 100} |" for i in range(60)]
    )

    def test_a_table_is_never_split_inside_a_row(self):
        for chunk in chunk_structured(self.TABLE, size=60, overlap=10):
            for line in chunk.text.splitlines():
                assert line.startswith("|") and line.endswith("|")

    def test_each_row_group_repeats_the_header(self):
        chunks = chunk_structured(self.TABLE, size=60, overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.startswith("| Commune | Code | Population |\n| --- | --- | --- |")

    def test_every_row_lands_in_exactly_one_chunk(self):
        rows = [
            line
            for chunk in chunk_structured(self.TABLE, size=60, overlap=10)
            for line in chunk.text.splitlines()[2:]
        ]
        assert rows == self.TABLE.splitlines()[2:]

    def test_a_small_table_stays_with_its_paragraph(self):
        text = "## Budget\n\nRépartition :\n\n| Poste | Montant |\n| --- | --- |\n| Eau | 30 |"
        chunks = chunk_structured(text, size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0].heading_path == "Budget"
        assert "Répartition" in chunks[0].text and "| Eau | 30 |" in chunks[0].text

    def test_table_rows_get_no_prose_overlap(self):
        text = " ".join(f"mot{i}" for i in range(90)) + "\n\n" + self.TABLE
        chunks = chunk_structured(text, size=60, overlap=10)
        table_chunks = [c for c in chunks if "| Commune" in c.text]
        assert all(c.text.startswith("| Commune") for c in table_chunks)


class TestPacking:
    def test_short_paragraphs_share_a_chunk(self):
        text = "Premier paragraphe.\n\nDeuxième paragraphe."
        chunks = chunk_structured(text, size=100, overlap=10)
        assert [c.text for c in chunks] == ["Premier paragraphe.\n\nDeuxième paragraphe."]

    def test_plain_text_matches_word_windows(self):
        """Unstructured text chunks exactly as the old word-window chunker did."""
        from app.rag.chunking import chunk_text

        text = " ".join(f"w{i}" for i in range(250))
        assert [c.text for c in chunk_structured(text)] == chunk_text(text)

    def test_empty_text_yields_nothing(self):
        assert chunk_structured("") == []
        assert chunk_structured("  \n\n ") == []

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError, match="smaller than size"):
            chunk_structured("a b c", size=10, overlap=10)
