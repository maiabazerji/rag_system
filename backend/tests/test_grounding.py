"""Grounded generation: handles, structured parsing, citation validation, evidence score."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.i18n import MESSAGES
from app.rag import grounding as g
from app.rag.grounding import (
    SUBMIT_ANSWER_TOOL,
    CitedChunk,
    RawAnswer,
    RawClaim,
    cite_chunks,
    cite_payload,
    evidence_score,
    format_context,
    normalize_handle,
    normalize_relevance,
    parse_structured,
    raw_from_mapping,
    validate_citations,
)
from app.schemas import Answer, Chunk, Claim, Source, StrategyComparison

EN_REFUSAL = MESSAGES["insufficient_context"]["en"]
FR_REFUSAL = MESSAGES["insufficient_context"]["fr"]
Q = "What does BM25 measure?"


def _ctx(n: int = 3, scores: list[float | None] | None = None) -> list[CitedChunk]:
    scores = scores or [None] * n
    return [
        CitedChunk(
            handle=f"S{i + 1}",
            chunk_id=f"doc{i}:{i}",
            text=f"text {i}",
            document=f"doc{i}.md",
            score=scores[i],
        )
        for i in range(n)
    ]


def _raw(answer: str, claims=None, status: str = "answered", notes=None) -> RawAnswer:
    return RawAnswer(
        answer=answer,
        claims=[RawClaim(*c) if isinstance(c, tuple) else c for c in (claims or [])],
        status=status,
        unsupported_notes=notes,
        mode="tool",
    )


# --- handles and context ---------------------------------------------------------


class TestContext:
    def test_handles_are_sequential_and_map_to_chunk_ids(self, fake_chunks):
        cited = cite_chunks(fake_chunks(3))
        assert [c.handle for c in cited] == ["S1", "S2", "S3"]
        assert [c.chunk_id for c in cited] == ["chunk_0", "chunk_1", "chunk_2"]

    def test_context_shows_handles_and_metadata_never_chunk_ids(self):
        chunk = Chunk(
            id="9fa3c1b0e2:4",
            doc_id="9fa3c1b0e2",
            text="Leave is 25 days.",
            tokens=4,
            metadata={
                "filename": "handbook.pdf",
                "title": "Employee Handbook",
                "section": "Leave",
                "page": 12,
                "page_end": 13,
            },
        )
        block = format_context(cite_chunks([chunk]))
        assert block.startswith("[S1] Employee Handbook | section: Leave | pp. 12-13\n")
        assert "Leave is 25 days." in block
        assert "9fa3c1b0e2" not in block

    def test_missing_metadata_is_tolerated(self):
        cited = cite_payload("S1", "c:1", "body", {})
        assert g.context_header(cited) == "[S1]"
        assert cited.page is None and cited.title is None and cited.score is None

    def test_single_page_and_filename_fallback(self):
        cited = cite_payload("S2", "c:1", "body", {"filename": "a.md", "page": "3"})
        assert g.context_header(cited) == "[S2] a.md | p. 3"

    def test_score_prefers_rerank_then_fused_then_dense(self):
        assert cite_payload("S1", "c", "t", {"dense_score": 0.2, "rerank_score": 4.0}).score == 4.0
        assert cite_payload("S1", "c", "t", {"dense_score": 0.2, "fused_score": 0.03}).score == 0.03
        assert cite_payload("S1", "c", "t", {"dense_score": 0.2}).score == 0.2
        assert cite_payload("S1", "c", "t", {}, score=0.7).score == 0.7

    def test_document_id_prefers_payload_then_doc_id(self):
        assert cite_payload("S1", "c", "t", {"document_id": "D"}, doc_id="d").document_id == "D"
        assert cite_payload("S1", "c", "t", {}, doc_id="d").document_id == "d"


# --- parsing ---------------------------------------------------------------------------


class TestParse:
    def test_tool_call(self):
        raw = parse_structured(
            {
                "answer": "BM25 ranks by term frequency [S1].",
                "claims": [{"text": "BM25 ranks by tf.", "citations": ["S1"], "supported": True}],
                "status": "answered",
                "unsupported_notes": "",
            },
            "ignored preamble",
        )
        assert raw.mode == "tool"
        assert raw.claims == [RawClaim("BM25 ranks by tf.", ["S1"], True)]
        assert raw.status == "answered" and raw.unsupported_notes is None

    def test_json_block_fallback(self):
        text = 'Here you go:\n```json\n{"answer": "X [S2].", "status": "partial", "claims": []}\n```'
        raw = parse_structured(None, text)
        assert raw.mode == "json"
        assert raw.status == "partial"
        # Claims missing: derived from the answer's sentences.
        assert raw.claims == [RawClaim("X [S2].", ["S2"], True)]

    def test_bare_json_object_fallback(self):
        raw = parse_structured(None, 'prefix {"answer": "Y [S1]", "status": "answered"} suffix')
        assert raw.mode == "json" and raw.answer == "Y [S1]"

    def test_plain_text_fallback(self):
        raw = parse_structured(None, "BM25 is lexical [S1]. It ignores synonyms. [S3]")
        assert raw.mode == "text" and raw.status == "answered"
        # A marker after the full stop belongs to the sentence before it.
        assert raw.claims == [
            RawClaim("BM25 is lexical [S1].", ["S1"], True),
            RawClaim("It ignores synonyms.", ["S3"], True),
        ]

    def test_uncited_sentence_is_an_unsupported_claim(self):
        raw = parse_structured(None, "Cited [S1]. Not cited here.")
        assert raw.claims[1] == RawClaim("Not cited here.", [], False)

    @pytest.mark.parametrize("text", ["", None, "   "])
    def test_empty_output_is_insufficient(self, text):
        raw = parse_structured(None, text)
        assert raw.status == "insufficient_context" and raw.answer == ""

    def test_malformed_fields_never_raise(self):
        raw = raw_from_mapping(
            {"answer": 42, "claims": "S1 says so [S1]", "status": "bogus", "unsupported_notes": 3}
        )
        assert raw.answer == "42" and raw.status == "answered"
        assert raw.claims[0].citations == ["S1"]
        assert raw.unsupported_notes is None

    def test_claim_with_scalar_citation_and_bad_supported(self):
        raw = raw_from_mapping(
            {"answer": "A", "claims": [{"text": "A", "citations": "S2", "supported": "yes"}]}
        )
        assert raw.claims == [RawClaim("A", ["S2"], True)]

    def test_legacy_finish_shape(self):
        raw = raw_from_mapping({"answer": "A thing.", "citations": ["S2"], "refusal": False})
        assert raw.claims == [RawClaim("A thing.", ["S2"], True)]
        assert raw_from_mapping({"answer": "no", "citations": [], "refusal": True}).status == (
            "insufficient_context"
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("insufficient", "insufficient_context"), ("Partial", "partial"), ("refusal", "insufficient_context")],
    )
    def test_status_synonyms(self, value, expected):
        assert raw_from_mapping({"answer": "a", "status": value}).status == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("S1", "S1"), ("[s2]", "S2"), (" 3 ", "S3"), ("S01", "S1"), ("doc:4", "doc:4"), ("", None), (None, None)],
    )
    def test_normalize_handle(self, value, expected):
        assert normalize_handle(value) == expected


# --- validation ------------------------------------------------------------------------------


class TestValidate:
    def test_valid_citations_are_kept_and_sources_ordered_by_first_citation(self):
        raw = _raw(
            "First [S3]. Second [S1][S3].",
            [("First.", ["S3"], True), ("Second.", ["S1", "S3"], True)],
        )
        out = validate_citations(raw, _ctx(), Q)
        assert out.refusal is False and out.grounded is True
        assert [c.handle for c in out.cited] == ["S3", "S1"]
        assert [s.chunk_id for s in out.sources()] == ["doc2:2", "doc0:0"]
        assert out.citation_count == 2
        assert out.claims[1].chunk_ids == ["doc0:0", "doc2:2"]
        assert out.invalid_citations == []

    def test_invalid_handles_are_recorded_and_stripped(self):
        raw = _raw(
            "Real [S1]. Made up [S9]. Mixed [S2, S7].",
            [("Real.", ["S1"], True), ("Made up.", ["S9"], True), ("Mixed.", ["S2", "S7"], True)],
        )
        out = validate_citations(raw, _ctx(), Q)
        assert out.answer == "Real [S1]. Made up. Mixed [S2]."
        assert out.invalid_citations == ["S9", "S7"]
        assert out.grounded is False  # an invalid citation, and a claim left uncited
        assert out.refusal is False
        assert out.claims[1].citations == []

    def test_out_of_range_handles(self):
        out = validate_citations(_raw("Zero [S0]. Far [S4].", [("x", ["S0", "S4"], True)]), _ctx(3), Q)
        assert out.invalid_citations == ["S0", "S4"]
        assert out.refusal is True and out.refusal_reason == "no_valid_citations"

    def test_duplicates_collapse(self):
        raw = _raw("A [S1][S1] and [S1, s1].", [("A", ["S1", "S1", "s1"], True)])
        out = validate_citations(raw, _ctx(), Q)
        assert out.answer == "A [S1] and [S1]."
        assert out.claims[0].citations == ["S1"]
        assert out.citation_count == 1

    def test_raw_chunk_id_markers_are_invalid(self):
        out = validate_citations(
            _raw("Yes [S1] and [9fa3c1b0e2:4]. At [10:30].", [("Yes", ["S1", "doc0:0"], True)]),
            _ctx(),
            Q,
        )
        assert out.answer == "Yes [S1] and. At [10:30]."
        # The answer text is checked first, then the claims.
        assert out.invalid_citations == ["9fa3c1b0e2:4", "doc0:0"]

    def test_markers_in_claim_text_are_validated_and_stripped(self):
        out = validate_citations(_raw("A [S2].", [("A [S2] [S8]", [], True)]), _ctx(), Q)
        assert out.claims == [Claim(text="A", citations=["S2"], chunk_ids=["doc1:1"], supported=True)]
        assert out.invalid_citations == ["S8"]

    def test_insufficient_context_returns_localized_refusal(self):
        raw = _raw("The documents don't cover BM25.", [], status="insufficient_context")
        out = validate_citations(raw, _ctx(), Q)
        assert out.refusal is True and out.answer == EN_REFUSAL
        assert out.status == "insufficient_context"
        assert out.unsupported_notes == "The documents don't cover BM25."
        assert out.sources()[0].chunk_id == "none"
        assert out.confidence == 0.0 and out.grounded is False

    def test_insufficient_context_in_french(self):
        out = validate_citations(
            _raw("", status="insufficient_context"), _ctx(), "Que mesure BM25 dans ce corpus ?"
        )
        assert out.answer == FR_REFUSAL

    def test_no_surviving_citation_is_a_refusal(self):
        out = validate_citations(_raw("BM25 is great.", [("BM25 is great.", [], True)]), _ctx(), Q)
        assert out.refusal is True and out.refusal_reason == "no_valid_citations"
        assert out.answer == EN_REFUSAL
        assert out.unsupported_notes is None  # ungrounded text is not echoed back

    def test_empty_answer_is_a_refusal(self):
        out = validate_citations(_raw("  ", [("A", ["S1"], True)]), _ctx(), Q)
        assert out.refusal_reason == "empty_answer"

    def test_empty_context_refuses_everything(self):
        out = validate_citations(_raw("A [S1].", [("A", ["S1"], True)]), [], Q)
        assert out.refusal is True and out.invalid_citations == ["S1"]


class TestGroundedFlag:
    def test_partial_is_never_grounded(self):
        out = validate_citations(_raw("A [S1].", [("A", ["S1"], True)], status="partial"), _ctx(), Q)
        assert out.grounded is False and out.refusal is False and out.status == "partial"

    def test_unsupported_claim_is_not_grounded(self):
        raw = _raw("A [S1]. B [S2].", [("A", ["S1"], True), ("B", ["S2"], False)])
        assert validate_citations(raw, _ctx(), Q).grounded is False

    def test_claim_without_citation_is_not_grounded(self):
        raw = _raw("A [S1]. B.", [("A", ["S1"], True), ("B", [], True)])
        assert validate_citations(raw, _ctx(), Q).grounded is False

    def test_fully_cited_answer_is_grounded(self):
        raw = _raw("A [S1]. B [S2].", [("A", ["S1"], True), ("B", ["S2"], True)])
        assert validate_citations(raw, _ctx(), Q).grounded is True


# --- evidence score ------------------------------------------------------------------------------


class TestEvidenceScore:
    def _claims(self, cited: int, uncited: int) -> list[Claim]:
        return [Claim(text="c", citations=["S1"])] * cited + [Claim(text="u", citations=[])] * uncited

    def test_refusal_scores_zero(self):
        assert evidence_score(self._claims(2, 0), "answered", _ctx(1), refusal=True) == 0.0

    def test_full_evidence_without_scores_is_one(self):
        assert evidence_score(self._claims(3, 0), "answered", _ctx(2)) == 1.0

    def test_partial_status_and_coverage_lower_the_score(self):
        full = evidence_score(self._claims(2, 0), "answered", _ctx(1))
        half = evidence_score(self._claims(1, 1), "answered", _ctx(1))
        partial = evidence_score(self._claims(2, 0), "partial", _ctx(1))
        assert full > partial > 0 and full > half > 0
        assert half == pytest.approx((0.5 * 0.5 + 0.2) / 0.7, abs=1e-3)

    def test_relevance_of_cited_chunks_counts(self):
        high = evidence_score(self._claims(1, 0), "answered", _ctx(1, [0.9]))
        low = evidence_score(self._claims(1, 0), "answered", _ctx(1, [0.1]))
        assert high == pytest.approx(0.5 + 0.2 + 0.3 * 0.9)
        assert high > low

    def test_logit_scores_are_squashed(self):
        assert normalize_relevance(0.4) == 0.4
        assert 0.5 < normalize_relevance(3.0) < 1.0
        assert 0.0 < normalize_relevance(-8.0) < 0.5
        assert normalize_relevance(1e9) <= 1.0

    def test_invalid_citations_are_penalised(self):
        clean = evidence_score(self._claims(1, 0), "answered", _ctx(1))
        dirty = evidence_score(self._claims(1, 0), "answered", _ctx(1), invalid_count=1)
        assert dirty == pytest.approx(clean * 0.75)

    @pytest.mark.parametrize("status", ["answered", "partial", "insufficient_context", "junk"])
    @pytest.mark.parametrize("score", [None, -50.0, 0.0, 0.5, 1.0, 50.0])
    @pytest.mark.parametrize(("cited", "uncited", "invalid"), [(0, 0, 0), (0, 3, 5), (2, 1, 1), (4, 0, 0)])
    def test_bounds(self, status, score, cited, uncited, invalid):
        value = evidence_score(self._claims(cited, uncited), status, _ctx(1, [score]), invalid)
        assert 0.0 <= value <= 1.0

    def test_validated_answer_confidence_is_the_evidence_score(self):
        raw = _raw("A [S1].", [("A", ["S1"], True)])
        out = validate_citations(raw, _ctx(1, [0.8]), Q)
        assert out.confidence == evidence_score(out.claims, "answered", out.cited)


# --- schemas ---------------------------------------------------------------------------------


class TestSchemas:
    def test_answer_serializes_grounding_fields(self):
        raw = _raw("A [S1].", [("A", ["S1"], True)])
        out = validate_citations(raw, _ctx(1, [2.0]), Q)
        fields = out.result_fields()
        answer = Answer(question=Q, **fields)
        data = answer.model_dump(mode="json")
        assert data["grounded"] is True and data["status"] == "answered"
        assert data["citation_count"] == 1 and data["invalid_citations"] == []
        assert data["claims"] == [
            {"text": "A", "citations": ["S1"], "chunk_ids": ["doc0:0"], "supported": True}
        ]
        src = data["sources"][0]
        assert src["handle"] == "S1" and src["relevance_score"] == 2.0
        assert 0 < src["score"] <= 1  # the bounded field gets the normalised score
        assert Answer.model_validate(data) == answer

    def test_old_payloads_still_validate(self):
        legacy = Answer(
            question="q", answer="a", sources=[Source(chunk_id="c", quote="x")], confidence=0.5
        )
        assert legacy.grounded is False and legacy.status is None and legacy.claims == []
        row = StrategyComparison(
            strategy="classic",
            question="q",
            answer="a",
            sources=[],
            confidence=0.1,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            iterations=1,
        )
        assert row.citation_count == 0 and row.invalid_citations == []


# --- provider plumbing -----------------------------------------------------------------------


def _tool_response(tool_input=None, text=""):
    content = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    if tool_input is not None:
        content.append(SimpleNamespace(type="tool_use", name="submit_answer", input=tool_input, id="t1"))
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = "tool_use" if tool_input is not None else "end_turn"
    resp.usage = MagicMock(input_tokens=11, output_tokens=7)
    return resp


@pytest.mark.asyncio
class TestStructuredProvider:
    async def test_modern_model_gets_auto_and_strict(self):
        from app.rag.providers import anthropic_provider as ap

        create = AsyncMock(return_value=_tool_response({"answer": "a"}))
        with patch.object(ap, "_create_message", new=create):
            out = await ap.generate_structured(
                model="claude-sonnet-5", prompt="p", tool=SUBMIT_ANSWER_TOOL, system="s"
            )
        kwargs = create.await_args.kwargs
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert "temperature" not in kwargs
        assert kwargs["tools"][0]["strict"] is True
        assert "strict" not in SUBMIT_ANSWER_TOOL  # the shared definition is not mutated
        assert out["structured"] == {"answer": "a"} and out["input_tokens"] == 11

    async def test_legacy_model_is_forced_without_strict(self):
        from app.rag.providers import anthropic_provider as ap

        create = AsyncMock(return_value=_tool_response({"answer": "a"}))
        with patch.object(ap, "_create_message", new=create):
            await ap.generate_structured(model="claude-sonnet-4-6", prompt="p", tool=SUBMIT_ANSWER_TOOL)
        kwargs = create.await_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_answer"}
        assert kwargs["temperature"] == 0
        assert "strict" not in kwargs["tools"][0]

    async def test_no_tool_call_returns_text(self):
        from app.rag.providers import anthropic_provider as ap

        with patch.object(ap, "_create_message", new=AsyncMock(return_value=_tool_response(None, "prose [S1]"))):
            out = await ap.generate_structured(model="claude-opus-5-5", prompt="p", tool=SUBMIT_ANSWER_TOOL)
        assert out["structured"] is None and out["text"] == "prose [S1]"

    async def test_other_providers_are_asked_for_json(self):
        from app.rag.providers import base

        gen = AsyncMock(return_value={"text": '{"answer": "x"}', "input_tokens": 1, "output_tokens": 1})
        with patch.object(base, "generate_with_usage", new=gen):
            out = await base.generate_structured(
                "local", model="m", prompt="p", tool=SUBMIT_ANSWER_TOOL, system="rules"
            )
        assert out["structured"] is None
        system = gen.await_args.kwargs["system"]
        assert system.startswith("rules") and '"status"' in system


def test_system_prompt_names_the_tool_only_for_anthropic():
    assert "submit_answer tool" in g.grounded_system("anthropic")
    assert "JSON" in g.grounded_system("openai")


# --- end to end through a strategy ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_classic_end_to_end_with_invalid_citation(fake_chunks):
    from app.rag.strategies.classic import ClassicRAG

    chunks = fake_chunks(3)
    chunks[1].metadata["rerank_score"] = 5.0
    out = {
        "structured": {
            "answer": "Chunk one says so [S2]. And [S5].",
            "claims": [
                {"text": "Chunk one says so.", "citations": ["S2"], "supported": True},
                {"text": "And.", "citations": ["S5"], "supported": True},
            ],
            "status": "answered",
            "unsupported_notes": "",
        },
        "text": "",
        "input_tokens": 5,
        "output_tokens": 5,
    }
    with (
        patch("app.rag.strategies.classic.dense_search", new=AsyncMock(return_value=chunks)),
        patch("app.rag.strategies.classic.rerank_async", new=AsyncMock(return_value=chunks)),
        patch("app.rag.strategies.classic.generate_structured", new=AsyncMock(return_value=out)),
    ):
        result = await ClassicRAG().run(Q, top_k=3, model="claude-sonnet-5", prompt_version="default")

    assert result.answer == "Chunk one says so [S2]. And."
    assert result.invalid_citations == ["S5"]
    assert result.grounded is False and result.refusal is False
    assert [s.chunk_id for s in result.sources] == ["chunk_1"]
    assert result.sources[0].relevance_score == 5.0
    assert len(result.extra["context_sources"]) == 3
    assert 0 < result.confidence < 1


@pytest.mark.asyncio
async def test_generate_propagates_grounding_to_the_answer(fake_chunks):
    from app.rag.generate import answer_question

    chunks = fake_chunks(2)
    out = {"text": "Answer [S1].", "input_tokens": 1, "output_tokens": 1}
    with (
        patch("app.rag.generate.store_count", new=AsyncMock(return_value=2)),
        patch("app.rag.strategies.classic.dense_search", new=AsyncMock(return_value=chunks)),
        patch("app.rag.strategies.classic.rerank_async", new=AsyncMock(return_value=chunks)),
        patch("app.rag.strategies.classic.generate_structured", new=AsyncMock(return_value=out)),
    ):
        ans = await answer_question(Q, provider="anthropic")

    assert ans.grounded is True and ans.status == "answered" and ans.citation_count == 1
    assert ans.sources[0].handle == "S1" and ans.claims[0].chunk_ids == ["chunk_0"]
