"""Multilingual behaviour: model-aware embedding prefixes, accent-folded BM25,
the embedding-model check on the Qdrant collection, and answer-language prompts."""
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.config import Settings
from app.rag import store
from app.rag.collection_model import (
    EMBEDDING_MODEL_KEY,
    EmbeddingModelMismatch,
    check_collection_model,
)
from app.rag.embed import (
    embed_query,
    embed_query_async,
    embed_texts,
    embed_texts_async,
    embedding_profile,
    is_multilingual_model,
)
from app.rag.rerank import _rerank_with_bm25, bm25_tokenize
from app.schemas import Chunk

E5 = "intfloat/multilingual-e5-small"


def _model():
    m = MagicMock()
    m.encode = MagicMock(side_effect=lambda texts, **_: np.ones((len(texts), 4)))
    return m


class TestDefaults:
    def test_defaults_are_multilingual(self):
        fields = Settings.model_fields
        assert fields["embedding_model"].default == E5
        assert fields["reranker_model"].default == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        assert is_multilingual_model(fields["embedding_model"].default)

    def test_download_script_reads_the_new_defaults(self):
        path = Path(__file__).resolve().parents[2] / "scripts" / "download_models.py"
        spec = importlib.util.spec_from_file_location("download_models", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        defaults = module._config_defaults()
        assert defaults["embedding_model"] == E5
        assert defaults["reranker_model"].startswith("cross-encoder/mmarco")


class TestEmbeddingProfile:
    @pytest.mark.parametrize(
        "model, family, query, passage, multilingual",
        [
            (E5, "e5", "query: ", "passage: ", True),
            ("intfloat/multilingual-e5-large", "e5", "query: ", "passage: ", True),
            ("intfloat/e5-base-v2", "e5", "query: ", "passage: ", False),
            ("/models/multilingual-e5-small/", "e5", "query: ", "passage: ", True),
            ("BAAI/bge-m3", "bge-m3", "", "", True),
            ("BAAI/bge-small-en-v1.5", "bge-en", "", "", False),
            ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
             "multilingual", "", "", True),
            ("some/unknown-model", "generic", "", "", False),
        ],
    )
    def test_known_families(self, model, family, query, passage, multilingual):
        p = embedding_profile(model)
        assert (p.family, p.query_prefix, p.passage_prefix, p.multilingual) == (
            family, query, passage, multilingual,
        )

    def test_e5_instruct_prefixes_queries_only(self):
        p = embedding_profile("intfloat/multilingual-e5-large-instruct")
        assert p.query_prefix.startswith("Instruct: ")
        assert p.query_prefix.endswith("Query: ")
        assert p.passage_prefix == ""

    def test_settings_override_the_table(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", E5)
        monkeypatch.setattr(settings, "embedding_query_prefix", "")
        p = embedding_profile()
        assert p.query_prefix == ""
        assert p.passage_prefix == "passage: "


class TestEmbedPrefixing:
    def test_passages_get_the_passage_prefix(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", E5)
        model = _model()
        with patch("app.rag.embed._model", return_value=model):
            embed_texts(["BM25 est lexical."])
        model.encode.assert_called_once_with(
            ["passage: BM25 est lexical."], normalize_embeddings=True
        )

    def test_queries_get_the_query_prefix(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", E5)
        model = _model()
        with patch("app.rag.embed._model", return_value=model):
            embed_query("Que mesure BM25 ?")
        model.encode.assert_called_once_with(
            ["query: Que mesure BM25 ?"], normalize_embeddings=True
        )

    async def test_async_paths_prefix_too(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", E5)
        model = _model()
        with patch("app.rag.embed._model", return_value=model):
            await embed_texts_async(["doc"])
            await embed_query_async("q")
        calls = [c.args[0] for c in model.encode.call_args_list]
        assert calls == [["passage: doc"], ["query: q"]]

    def test_bge_english_is_left_alone(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        model = _model()
        with patch("app.rag.embed._model", return_value=model):
            embed_query("What is BM25?")
        model.encode.assert_called_once_with(["What is BM25?"], normalize_embeddings=True)


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"c{i}", doc_id=f"d{i}", text=text, tokens=len(text.split()))


class TestAccentFoldedBM25:
    def test_tokenizer_folds_accents_case_and_drops_stopwords(self):
        assert bm25_tokenize("La REQUÊTE et le résumé de l'index") == [
            "requete", "resume", "index",
        ]
        assert bm25_tokenize("What is the reranker?") == ["reranker"]

    def test_tokenizer_keeps_stopword_only_text(self):
        assert bm25_tokenize("the") == ["the"]

    def test_unaccented_query_matches_accented_text(self):
        chunks = [
            _chunk(0, "Les embeddings denses capturent la similarité sémantique."),
            _chunk(1, "Le reclassement utilise un modèle cross-encoder."),
            _chunk(2, "La mise en cache réduit la latence des requêtes."),
        ]
        ranked = _rerank_with_bm25("modele de reclassement", chunks, top_k=1)
        assert [c.id for c in ranked] == ["c1"]

    def test_punctuation_does_not_hide_terms(self):
        chunks = [
            _chunk(0, "Retrieval uses (BM25)."),
            _chunk(1, "Something unrelated about caching layers."),
            _chunk(2, "Another text on chunk overlap."),
        ]
        ranked = _rerank_with_bm25("bm25", chunks, top_k=1)
        assert [c.id for c in ranked] == ["c0"]


def _info(size=384, metadata=None, points=0):
    return MagicMock(
        config=MagicMock(params=MagicMock(vectors=MagicMock(size=size)), metadata=metadata),
        points_count=points,
    )


class TestCollectionModelCheck:
    def test_matching_model_passes(self, settings):
        info = _info(metadata={EMBEDDING_MODEL_KEY: settings.embedding_model}, points=10)
        assert check_collection_model(info, 384) is False

    def test_other_model_with_same_dim_fails_clearly(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", E5)
        info = _info(metadata={EMBEDDING_MODEL_KEY: "BAAI/bge-small-en-v1.5"}, points=10)
        with pytest.raises(EmbeddingModelMismatch) as exc:
            check_collection_model(info, 384)
        msg = str(exc.value)
        assert "BAAI/bge-small-en-v1.5" in msg and E5 in msg
        assert "scripts/ingest.py" in msg
        assert "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" in msg

    def test_dimension_mismatch_fails(self, settings):
        info = _info(size=1024, metadata={EMBEDDING_MODEL_KEY: "BAAI/bge-m3"}, points=5)
        with pytest.raises(EmbeddingModelMismatch, match="dim=1024"):
            check_collection_model(info, 384)

    def test_unstamped_non_empty_collection_is_refused(self, settings):
        with pytest.raises(EmbeddingModelMismatch, match="predates model tracking") as exc:
            check_collection_model(_info(metadata=None, points=107), 384)
        assert "curl -X PATCH" in str(exc.value)

    def test_unstamped_empty_collection_asks_to_be_stamped(self, settings):
        assert check_collection_model(_info(metadata=None, points=0), 384) is True


class TestEnsureCollection:
    @pytest.fixture
    def qdrant(self, settings):
        c = AsyncMock()
        coll = MagicMock()
        coll.name = settings.qdrant_collection
        c.get_collections = AsyncMock(return_value=MagicMock(collections=[coll]))
        return c

    async def test_new_collection_records_the_model(self, settings):
        c = AsyncMock()
        c.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        with patch("app.rag.store.embedding_dim", return_value=384):
            await store._ensure_collection(c)
        metadata = c.create_collection.call_args.kwargs["metadata"]
        assert metadata == {EMBEDDING_MODEL_KEY: settings.embedding_model, "embedding_dim": 384}

    async def test_empty_legacy_collection_is_stamped(self, qdrant, settings):
        qdrant.get_collection = AsyncMock(return_value=_info(metadata=None, points=0))
        with patch("app.rag.store.embedding_dim", return_value=384):
            await store._ensure_collection(qdrant)
        kwargs = qdrant.update_collection.call_args.kwargs
        assert kwargs["metadata"][EMBEDDING_MODEL_KEY] == settings.embedding_model

    async def test_mismatch_raises_and_deletes_nothing(self, qdrant, monkeypatch):
        breaker = MagicMock()
        monkeypatch.setattr(store, "_qdrant_breaker", breaker)
        qdrant.get_collection = AsyncMock(
            return_value=_info(metadata={EMBEDDING_MODEL_KEY: "other/model"}, points=3)
        )
        with patch("app.rag.store.embedding_dim", return_value=384):
            with pytest.raises(EmbeddingModelMismatch):
                await store._ensure_collection(qdrant)
        qdrant.delete_collection.assert_not_called()
        qdrant.update_collection.assert_not_called()
        breaker.record_failure.assert_not_called()

    async def test_failed_check_is_not_cached(self, monkeypatch):
        monkeypatch.setattr(store, "_client", None)
        ensure = AsyncMock(side_effect=EmbeddingModelMismatch("nope"))
        with patch.object(store, "AsyncQdrantClient"), patch.object(
            store, "_ensure_collection", ensure
        ):
            for _ in range(2):
                with pytest.raises(EmbeddingModelMismatch):
                    await store.client()
        assert ensure.await_count == 2
        assert store._client is None

    def test_mismatch_maps_to_a_named_503(self, client):
        async def boom(*_a, **_k):
            try:
                raise EmbeddingModelMismatch("built with other/model")
            except EmbeddingModelMismatch as e:
                raise store.StoreUnavailable(f"Qdrant count failed: {e}") from e

        with patch("app.rag.generate.store_count", new=boom):
            r = client.post("/ask", json={"question": "Que mesure BM25 ?"})
        assert r.status_code == 503
        body = r.json()
        assert body["code"] == "embedding_model_mismatch"
        assert "other/model" in body["detail"]


class TestAnswerLanguagePrompts:
    def test_generation_prompt_asks_for_the_question_language(self):
        from app.prompts import render_prompt

        text = render_prompt("default", question="Que mesure BM25 ?", context="[c1] BM25...")
        assert "Answer in the language the question is written in" in text
        assert "French question gets a French" in text
        assert "natural English" not in text

    def test_agentic_system_prompt_asks_for_the_question_language(self):
        from app.rag.strategies.agentic import _SYSTEM

        assert "language of the user's question" in _SYSTEM
        assert "French question gets a French answer" in _SYSTEM

    def test_judge_scores_across_languages_and_checks_answer_language(self):
        from app.eval.judge import _build_prompt

        prompt = _build_prompt("Que mesure BM25 ?", "BM25 est...", ["BM25 is..."], None)
        assert "Judge meaning, not wording or language" in prompt
        assert "expected in the language of the question" in prompt

    def test_graph_entity_extraction_asks_for_english_forms(self):
        import inspect

        from app.rag import graph_extract

        assert "English form" in inspect.getsource(graph_extract.extract_question_entities)


class TestLocalizedRefusals:
    async def test_no_documents_refusal_follows_the_question(self):
        from app.i18n import MESSAGES
        from app.rag.generate import answer_question

        with patch("app.rag.generate.store_count", new=AsyncMock(return_value=0)):
            fr = await answer_question("Comment fonctionne le reclassement ?")
            en = await answer_question("How does reranking work?")
        assert fr.refusal and fr.answer == MESSAGES["no_documents"]["fr"]
        assert en.refusal and en.answer == MESSAGES["no_documents"]["en"]

    async def test_classic_empty_context_refusal_is_french(self):
        from app.i18n import MESSAGES
        from app.rag.strategies.classic import ClassicRAG

        with patch(
            "app.rag.strategies.classic.dense_search", new=AsyncMock(return_value=[])
        ), patch("app.rag.strategies.classic.rerank_async", new=AsyncMock(return_value=[])):
            result = await ClassicRAG().run(
                "Pourquoi utiliser un reranker ?",
                top_k=4,
                model="claude-sonnet-5",
                prompt_version="default",
            )
        assert result.refusal
        assert result.answer == MESSAGES["no_context"]["fr"]
