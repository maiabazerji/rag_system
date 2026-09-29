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


def sentences(n, words=8, start=0):
    """n distinct sentences of `words` words each, ending in a period."""
    filler = " ".join(["mot"] * (words - 3))
    return [f"Phrase numéro {start + i} {filler}." for i in range(n)]


def assert_offsets_map_back(text, chunks):
    for chunk in chunks:
        span = text[chunk.start : chunk.end]
        assert span and chunk.text.endswith(span)
        assert span == span.strip()


class TestParagraphPacking:
    def test_whole_paragraphs_are_packed_up_to_the_size(self):
        paragraphs = [" ".join(sentences(3, start=10 * i)) for i in range(12)]  # 24 words each
        text = "\n\n".join(paragraphs)
        chunks = chunk_structured(text, size=100, overlap=0)

        assert len(chunks) == 3
        for chunk in chunks:
            assert set(chunk.text.split("\n\n")) <= set(paragraphs)
            assert chunk.token_count <= 100

    def test_a_long_paragraph_is_split_between_sentences(self):
        text = " ".join(sentences(40))
        chunks = chunk_structured(text, size=50, overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.startswith("Phrase numéro")
            assert chunk.text.endswith(".")
            assert chunk.token_count <= 50

    def test_a_paragraph_is_split_only_when_it_exceeds_the_size(self):
        short = " ".join(sentences(3))
        long = " ".join(sentences(12, start=100))  # 96 words
        chunks = chunk_structured(f"{short}\n\n{long}", size=100, overlap=0)

        # The long paragraph fits a chunk of its own, so it is not split.
        assert [c.text for c in chunks] == [short, long]

    def test_overlap_is_made_of_whole_trailing_sentences(self):
        text = " ".join(sentences(40))
        chunks = chunk_structured(text, size=50, overlap=20)

        for before, after in zip(chunks, chunks[1:], strict=False):
            assert after.start < before.end
            shared = text[after.start : before.end]
            assert shared.startswith("Phrase numéro") and shared.endswith(".")
            assert len(shared.split()) <= 20
            assert before.text.endswith(shared) and after.text.startswith(shared)

    def test_no_overlap_when_the_last_sentence_is_longer_than_the_overlap(self):
        text = " ".join(sentences(20, words=30))
        chunks = chunk_structured(text, size=100, overlap=20)
        assert len(chunks) > 1
        for before, after in zip(chunks, chunks[1:], strict=False):
            assert after.start > before.end

    def test_every_sentence_is_kept(self):
        body = sentences(40)
        chunks = chunk_structured(" ".join(body), size=50, overlap=15)
        seen = {s for c in chunks for s in body if s in c.text}
        assert seen == set(body)

    def test_a_sentence_longer_than_the_size_falls_back_to_word_windows(self):
        giant = " ".join(f"mot{i}" for i in range(120)) + "."
        text = f"{' '.join(sentences(2))} {giant} {' '.join(sentences(2, start=50))}"
        chunks = chunk_structured(text, size=50, overlap=10)

        assert max(c.token_count for c in chunks) <= 50
        assert any(c.text.startswith("mot") and "Phrase" not in c.text for c in chunks)
        assert chunks[-1].text.endswith("Phrase numéro 51 mot mot mot mot mot.")


class TestSentenceSplitting:
    @staticmethod
    def split(text):
        from app.rag.chunking import _sentence_spans

        return [text[s:e] for s, e in _sentence_spans(text, 0, len(text))]

    def test_ends_at_terminal_punctuation_before_a_capital(self):
        assert self.split("Il pleut. Elle sort! Pourquoi ? Parce que… Voilà.") == [
            "Il pleut.", "Elle sort!", "Pourquoi ?", "Parce que…", "Voilà.",
        ]

    def test_french_quotes(self):
        assert self.split("Il dit : « Bonjour. » Puis il part. « Oui », dit-elle.") == [
            "Il dit : « Bonjour. »", "Puis il part.", "« Oui », dit-elle.",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "M. Dupont est venu.",
            "Voir l'art. L. 121-1 du code.",
            "Voir art. 5 et p. 12 du rapport.",
            "Signé J. Martin et Mme. Durand.",
            "Cf. Annexe 2 pour le détail.",
            "Selon le U.S. Congress, rien.",
        ],
    )
    def test_abbreviations_do_not_end_a_sentence(self, text):
        assert self.split(text) == [text]

    def test_lowercase_after_a_period_does_not_start_a_sentence(self):
        assert self.split("Version 2.1 est prête. puis rien") == ["Version 2.1 est prête. puis rien"]

    def test_list_items_start_sentences(self):
        assert self.split("Pièces :\n- un devis\n- une facture") == [
            "Pièces :", "- un devis", "- une facture",
        ]


class TestOffsets:
    def test_prose_chunks_are_exact_spans_of_the_text(self):
        text = "# Guide\n\n" + "\n\n".join(" ".join(sentences(5, start=i * 10)) for i in range(10))
        chunks = chunk_structured(text, size=60, overlap=15)
        assert len(chunks) > 1
        for chunk in chunks:
            assert text[chunk.start : chunk.end] == chunk.text

    def test_legal_and_table_chunks_map_back(self):
        text = CODE_EXCERPT + "\n\n## Annexe\n\n" + TestTableChunking.TABLE + "\n"
        chunks = chunk_structured(text, size=60, overlap=10)
        assert_offsets_map_back(text, chunks)
        # Only a repeated table header is not part of the span.
        for chunk in chunks:
            if text[chunk.start : chunk.end] != chunk.text:
                assert chunk.text.startswith("| Commune | Code | Population |\n| --- |")

    def test_a_heading_stays_in_the_first_span(self):
        text = "  ## Budget\n\nChiffres. " + " ".join(sentences(30))
        first = chunk_structured(text, size=60, overlap=10)[0]
        assert text[first.start :].startswith("## Budget")
        assert first.headings == ("Budget",) and first.section == "Budget"

    def test_offsets_hold_with_irregular_whitespace(self):
        text = "Titre I\n\n\n  Premier   paragraphe.  \nsuite.\t\n\n\n\nSecond.\r\n\r\nTroisième."
        chunks = chunk_structured(text, size=100, overlap=10)
        assert_offsets_map_back(text, chunks)
        assert "Premier   paragraphe." in chunks[0].text


class TestStrategies:
    TEXT = "# Guide\n\n" + " ".join(sentences(60)) + "\n\n## Annexe\n\n" + " ".join(sentences(5))

    def test_fixed_is_the_legacy_word_windows(self):
        from app.rag.chunking import chunk_document, chunk_text

        chunks = chunk_document(self.TEXT, strategy="fixed", size=100, overlap=20)
        assert [c.text for c in chunks] == chunk_text(self.TEXT, size=100, overlap=20)
        for chunk in chunks:
            assert " ".join(self.TEXT[chunk.start : chunk.end].split()) == chunk.text
        # A fixed chunk is filed under the section it starts in.
        paths = [c.heading_path for c in chunk_document(self.TEXT, strategy="fixed", size=50, overlap=0)]
        assert paths[0] == "Guide" and paths[-1] == "Guide > Annexe"

    def test_structured_is_the_default(self, settings):
        from app.rag.chunking import chunk_document

        assert settings.chunk_strategy == "structured"
        assert chunk_document(self.TEXT, size=100, overlap=20) == chunk_structured(
            self.TEXT, size=100, overlap=20
        )

    def test_the_setting_selects_the_strategy(self, settings, monkeypatch):
        from app.rag.chunking import chunk_document, chunk_text

        monkeypatch.setattr(settings, "chunk_strategy", "fixed")
        assert [c.text for c in chunk_document(self.TEXT, size=100, overlap=20)] == chunk_text(
            self.TEXT, size=100, overlap=20
        )

    def test_unknown_strategy_is_refused(self):
        from app.rag.chunking import chunk_document

        with pytest.raises(ValueError, match="chunk strategy"):
            chunk_document(self.TEXT, strategy="semantic")

    def test_setting_is_validated(self):
        from app.config import Settings

        assert Settings(_env_file=None, chunk_strategy="FIXED").chunk_strategy == "fixed"
        with pytest.raises(ValueError, match="CHUNK_STRATEGY"):
            Settings(_env_file=None, chunk_strategy="semantic")

    @pytest.mark.parametrize("strategy", ["structured", "fixed"])
    def test_chunking_is_deterministic(self, strategy):
        from app.rag.chunking import chunk_document

        text = CODE_EXCERPT + TestTableChunking.TABLE + self.TEXT
        runs = [chunk_document(text, strategy=strategy, size=60, overlap=10) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]


class TestModelWindowGuard:
    """Chunks must fit the 512-token input of the default embedder and reranker."""

    TEXT = " ".join(sentences(200))  # 1600 words, one paragraph

    def test_defaults_fit_a_512_token_model(self):
        from app.config import TOKENS_PER_WORD, Settings

        s = Settings(_env_file=None)
        assert s.chunk_max_model_tokens == 512
        assert s.chunk_word_budget == s.chunk_size_tokens == 300
        assert s.chunk_word_budget * TOKENS_PER_WORD <= 512

    @pytest.mark.parametrize(
        "size, window, budget", [(600, 512, 341), (300, 512, 300), (600, 8192, 600), (600, 0, 600)]
    )
    def test_the_budget_is_capped_by_the_model_window(self, size, window, budget):
        from app.config import Settings

        s = Settings(_env_file=None, chunk_size_tokens=size, chunk_max_model_tokens=window)
        assert s.chunk_word_budget == budget

    def test_overlap_must_fit_the_capped_budget(self):
        from app.config import Settings

        with pytest.raises(ValueError, match="CHUNK_MAX_MODEL_TOKENS"):
            Settings(_env_file=None, chunk_overlap_tokens=100, chunk_max_model_tokens=128)

    @pytest.mark.parametrize("strategy", ["structured", "fixed"])
    def test_configured_chunks_respect_the_cap(self, settings, monkeypatch, strategy):
        from app.rag.chunking import chunk_document, chunk_text

        monkeypatch.setattr(settings, "chunk_size_tokens", 600)
        monkeypatch.setattr(settings, "chunk_overlap_tokens", 50)
        monkeypatch.setattr(settings, "chunk_max_model_tokens", 512)

        chunks = chunk_document(self.TEXT, strategy=strategy)
        assert max(c.token_count for c in chunks) <= 341
        if strategy == "fixed":
            assert [c.text for c in chunks] == chunk_text(self.TEXT, size=341, overlap=50)

    def test_an_explicit_size_is_taken_as_given(self, settings, monkeypatch):
        from app.rag.chunking import chunk_document

        monkeypatch.setattr(settings, "chunk_max_model_tokens", 512)
        chunks = chunk_document(self.TEXT, strategy="fixed", size=600, overlap=50)
        assert chunks[0].token_count == 600


class TestPages:
    def test_chunks_record_the_pages_they_span(self):
        from app.rag.chunking import chunk_document
        from app.rag.parsers import PageSpan

        page_texts = [" ".join(sentences(10, start=100 * p)) for p in range(3)]
        text = "\n\n".join(page_texts)
        pages, offset = [], 0
        for number, page in enumerate(page_texts, start=1):
            pages.append(PageSpan(number, offset, offset + len(page)))
            offset += len(page) + 2

        chunks = chunk_document(text, size=50, overlap=0, pages=pages)
        assert chunks[0].page_start == 1
        assert chunks[-1].page_end == 3
        for chunk in chunks:
            assert chunk.page_start is not None and chunk.page_end is not None
            touched = {p.number for p in pages if p.start < chunk.end and chunk.start < p.end}
            assert set(range(chunk.page_start, chunk.page_end + 1)) == touched

    def test_no_pages_means_no_page_numbers(self):
        from app.rag.chunking import chunk_document

        chunk = chunk_document("Texte court.")[0]
        assert chunk.page_start is None and chunk.page_end is None
