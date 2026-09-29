"""Triple and entity extraction (app.rag.graph_extract): parsing model replies."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.rag import graph_extract
from app.rag.graph_extract import _parse_triples, extract_question_entities, extract_triples

TRIPLE = {"subject": "BM25", "predicate": "is_a", "object": "ranking function"}


class TestParseTriples:
    def test_a_json_array(self):
        assert _parse_triples('[{"subject":"BM25","predicate":"is_a","object":"ranking function"}]') == [
            TRIPLE
        ]

    def test_prose_and_code_fences_around_the_array(self):
        raw = 'Here you go:\n```json\n[{"subject":"BM25","predicate":"is_a","object":"ranking function"}]\n```'
        assert _parse_triples(raw) == [TRIPLE]

    def test_a_single_object_instead_of_an_array(self):
        assert _parse_triples('{"subject":"BM25","predicate":"is_a","object":"ranking function"}') == [
            TRIPLE
        ]

    @pytest.mark.parametrize("raw", ["", "no triples here", "[not json", "null", "42", '"text"', "[]"])
    def test_replies_without_triples_yield_none(self, raw):
        assert _parse_triples(raw) == []

    def test_incomplete_or_null_fields_are_dropped(self):
        raw = (
            '[{"subject":"A","predicate":"uses"},'
            ' {"subject":null,"predicate":"uses","object":"B"},'
            ' {"subject":"  ","predicate":"uses","object":"B"},'
            ' {"subject":true,"predicate":"uses","object":"B"},'
            ' "a string",'
            ' {"subject":"Python","predicate":"version","object":3.12}]'
        )
        assert _parse_triples(raw) == [
            {"subject": "Python", "predicate": "version", "object": 3.12}
        ]


def _reply(text: str, tokens: tuple[int, int] = (40, 12)):
    return AsyncMock(
        return_value={"text": text, "input_tokens": tokens[0], "output_tokens": tokens[1]}
    )


class TestExtractTriples:
    async def test_builds_stripped_triples_tied_to_the_chunk(self, settings):
        reply = _reply('[{"subject":" BM25 ","predicate":"is_a","object":"ranking function "}]')
        with patch.object(graph_extract, "generate_with_usage", new=reply):
            triples = await extract_triples(chunk_id="d:0", doc_id="d", text="x" * 5000)

        assert [(t.subject, t.predicate, t.object, t.chunk_id, t.doc_id) for t in triples] == [
            ("BM25", "is_a", "ranking function", "d:0", "d")
        ]
        kwargs = reply.await_args.kwargs
        assert kwargs["model"] == settings.graph_extraction_model
        # The passage is capped so one huge chunk cannot blow the prompt up.
        assert "x" * 3500 in kwargs["prompt"] and "x" * 3501 not in kwargs["prompt"]

    async def test_a_null_reply_is_no_triples_not_an_error(self):
        with patch.object(graph_extract, "generate_with_usage", new=_reply("null")):
            assert await extract_triples(chunk_id="d:0", doc_id="d", text="t") == []


class TestQuestionEntities:
    async def test_entities_and_usage(self):
        with patch.object(
            graph_extract, "generate_with_usage", new=_reply('Sure: ["BM25", " RRF ", "", 3]')
        ):
            entities = await extract_question_entities("How does BM25 relate to RRF?")
        assert list(entities) == ["BM25", "RRF"]
        assert (entities.input_tokens, entities.output_tokens) == (40, 12)

    @pytest.mark.parametrize("text", ["no list", "[broken", '{"a": 1}'])
    async def test_unparseable_replies_still_report_usage(self, text):
        with patch.object(graph_extract, "generate_with_usage", new=_reply(text, (7, 3))):
            entities = await extract_question_entities("q")
        assert list(entities) == []
        assert (entities.input_tokens, entities.output_tokens) == (7, 3)
