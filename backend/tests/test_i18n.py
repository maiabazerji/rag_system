"""Tests for the lightweight language layer in app/i18n.py."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.i18n import (
    MESSAGES,
    SUPPORTED_LANGUAGES,
    AcceptLanguageMiddleware,
    detect_language,
    fold,
    localized,
    message,
    parse_accept_language,
    preferred_language,
    tokenize,
)


class TestFold:
    def test_strips_accents_and_lowercases(self):
        assert fold("Élève à l'ÉCOLE") == "eleve a l'ecole"

    def test_casefolds_sharp_s(self):
        assert fold("Straße") == "strasse"

    def test_tokenize_is_unicode_aware_and_splits_elisions(self):
        assert tokenize("Qu'est-ce que la requête ?") == ["qu", "est", "ce", "que", "la", "requete"]


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("What is Reciprocal Rank Fusion?", "en"),
            ("How does the reranker work with dense retrieval?", "en"),
            ("Qu'est-ce que la fusion Reciprocal Rank Fusion ?", "fr"),
            ("Comment fonctionne le reclassement avec un cross-encoder ?", "fr"),
            ("Pourquoi utiliser BM25 dans un système RAG ?", "fr"),
            ("Que mesure BM25 ?", "fr"),
            ("Wie funktioniert die Suche mit einem Cross-Encoder?", "de"),
            ("Was ist der Unterschied zwischen BM25 und dichten Vektoren?", "de"),
            ("¿Cómo funciona la búsqueda híbrida con BM25?", "es"),
            ("¿Qué es la generación aumentada por recuperación?", "es"),
            ("Come funziona la ricerca ibrida con il reranker?", "it"),
            ("Qual è la differenza tra ricerca densa e sparsa?", "it"),
        ],
    )
    def test_detects_common_questions(self, text, expected):
        assert detect_language(text) == expected

    def test_no_evidence_falls_back_to_default(self):
        assert detect_language("BM25 RRF 42", default="fr") == "fr"
        assert detect_language("", default="de") == "de"
        assert detect_language("BM25") == "en"

    def test_always_returns_a_supported_code(self):
        for text in ("x", "Hello there", "Bonjour à tous", "日本語のテキスト"):
            assert detect_language(text) in SUPPORTED_LANGUAGES


class TestAcceptLanguage:
    @pytest.mark.parametrize(
        "header, expected",
        [
            ("fr-FR,fr;q=0.9,en;q=0.8", "fr"),
            ("en-US,en;q=0.9", "en"),
            ("ja,de;q=0.5,fr;q=0.7", "fr"),
            ("zh-CN", None),
            ("fr;q=0", None),
            ("fr;q=abc,en", "en"),
            ("", None),
            (None, None),
        ],
    )
    def test_parse(self, header, expected):
        assert parse_accept_language(header) == expected

    def test_middleware_sets_preference_for_the_request_only(self):
        app = FastAPI()
        app.add_middleware(AcceptLanguageMiddleware)

        @app.get("/lang")
        async def lang():
            return {"preferred": preferred_language(), "short": detect_language("BM25 ?")}

        client = TestClient(app)
        body = client.get("/lang", headers={"Accept-Language": "fr-CA,fr;q=0.9"}).json()
        assert body == {"preferred": "fr", "short": "fr"}
        assert client.get("/lang").json()["preferred"] == "en"
        assert preferred_language() == "en"


class TestMessages:
    def test_every_message_has_english_and_french(self):
        for key, variants in MESSAGES.items():
            assert variants.get("en"), key
            assert variants.get("fr"), key

    def test_unknown_language_falls_back_to_english(self):
        assert message("no_context", "de") == MESSAGES["no_context"]["en"]

    def test_localized_follows_the_question(self):
        assert localized("no_context", "Que mesure BM25 ?") == MESSAGES["no_context"]["fr"]
        assert localized("no_context", "What does BM25 measure?") == MESSAGES["no_context"]["en"]
