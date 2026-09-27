"""Tests for the strategy advisor: scoring, profile extraction, and the /advise routes."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.advisor.profile import (
    TOOL_NAME,
    _request_options,
    apply_overrides,
    build_profile,
    extract_heuristic,
)
from app.advisor.schemas import (
    AdviseOverrides,
    ProjectProfile,
    QuestionMix,
    StrategyScorecard,
    ValidateRequest,
    corpus_bucket,
)
from app.advisor.scoring import (
    BASELINE_RELATIVE_TOKENS,
    compliance_notes,
    hybrid_routing,
    score_strategies,
    suggested_config,
)
from app.advisor.validate import default_strategies, pick_winner, token_f1
from app.rag.strategies.base import StrategyResult
from app.schemas import Source


def _profile(**kw) -> ProjectProfile:
    return ProjectProfile(**kw)


def _ranked(profile: ProjectProfile) -> list[str]:
    return [r.strategy for r in score_strategies(profile)]


def _tool_response(payload: dict, input_tokens: int = 400, output_tokens: int = 120):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input=payload, id="tu_1")],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


FAQ = _profile(
    corpus_size="small",
    question_mix=QuestionMix(single_fact=85, relational_multi_hop=10, exploratory_multi_step=5),
    latency_budget_ms=2_000,
    cost_sensitivity="high",
    entity_richness="low",
)
RELATIONAL = _profile(
    corpus_size="xlarge",
    corpus_size_docs=250_000,
    question_mix=QuestionMix(single_fact=20, relational_multi_hop=70, exploratory_multi_step=10),
    latency_budget_ms=20_000,
    cost_sensitivity="medium",
    entity_richness="high",
    document_types=["contracts"],
)
RESEARCH = _profile(
    corpus_size="medium",
    question_mix=QuestionMix(single_fact=10, relational_multi_hop=15, exploratory_multi_step=75),
    latency_budget_ms=60_000,
    cost_sensitivity="low",
    entity_richness="medium",
)


# --- scoring ------------------------------------------------------------------


class TestScoring:
    def test_classic_wins_single_fact_tight_latency_low_budget(self):
        recs = score_strategies(FAQ)
        assert recs[0].strategy == "classic"
        assert [r.rank for r in recs] == [1, 2, 3]
        assert recs[0].score > recs[1].score
        assert any("single-fact" in r for r in recs[0].reasons)

    def test_graph_wins_large_entity_rich_relational(self):
        recs = score_strategies(RELATIONAL)
        assert recs[0].strategy == "graph"
        cfg = recs[0].suggested_config
        assert cfg.retrieval_top_k < 50  # README: graph needs a smaller RETRIEVAL_TOP_K
        assert "graph_extraction" in cfg.models

    def test_agentic_wins_exploratory_with_generous_budget(self):
        recs = score_strategies(RESEARCH)
        assert recs[0].strategy == "agentic"
        assert recs[0].suggested_config.expected_relative_cost == BASELINE_RELATIVE_TOKENS[
            "agentic"
        ]
        assert any("1.9x" in t for t in recs[0].tradeoffs)

    def test_agentic_capped_when_cost_sensitive_even_if_exploratory(self):
        prof = RESEARCH.model_copy(update={"cost_sensitivity": "high"})
        recs = {r.strategy: r for r in score_strategies(prof)}
        assert recs["agentic"].score <= 40
        assert _ranked(prof)[0] != "agentic"

    def test_graph_never_beats_classic_on_small_corpus(self):
        prof = RELATIONAL.model_copy(update={"corpus_size": "small", "corpus_size_docs": 300})
        recs = {r.strategy: r for r in score_strategies(prof)}
        assert recs["graph"].score < recs["classic"].score
        assert any("extra steps" in r for r in recs["graph"].reasons)

    def test_scores_bounded_and_deterministic(self):
        for prof in (FAQ, RELATIONAL, RESEARCH, ProjectProfile()):
            a, b = score_strategies(prof), score_strategies(prof)
            assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
            assert all(0 <= r.score <= 100 for r in a)

    def test_neutral_profile_defaults_to_classic(self):
        assert _ranked(ProjectProfile())[0] == "classic"

    def test_very_tight_latency_turns_rerank_off(self):
        cfg = suggested_config("classic", FAQ.model_copy(update={"latency_budget_ms": 1_500}))
        assert cfg.rerank is False

    def test_multilingual_embedding_for_non_english(self):
        cfg = suggested_config("classic", FAQ.model_copy(update={"languages": ["fr", "en"]}))
        assert "bge-m3" in cfg.embedding_model

    def test_hybrid_routing_when_mix_split(self):
        prof = RESEARCH.model_copy(
            update={
                "question_mix": QuestionMix(
                    single_fact=50, relational_multi_hop=0, exploratory_multi_step=50
                )
            }
        )
        routing = hybrid_routing(prof)
        assert routing.recommended
        routes = {r.question_type: r.strategy for r in routing.routes}
        assert routes == {"single_fact": "classic", "exploratory_multi_step": "agentic"}

    def test_no_hybrid_when_one_type_dominates(self):
        assert hybrid_routing(FAQ).recommended is False

    def test_no_hybrid_when_constraints_collapse_routes(self):
        prof = FAQ.model_copy(
            update={
                "question_mix": QuestionMix(
                    single_fact=50, relational_multi_hop=0, exploratory_multi_step=50
                )
            }
        )  # tight latency + high cost: exploratory falls back to classic
        routing = hybrid_routing(prof)
        assert routing.recommended is False
        assert {r.strategy for r in routing.routes} == {"classic"}

    def test_compliance_notes_eu_gdpr_and_language(self):
        prof = FAQ.model_copy(
            update={"compliance": ["EU only", "GDPR"], "languages": ["fr"], "data_residency": "EU"}
        )
        notes = " ".join(compliance_notes(prof))
        assert "EU region" in notes
        assert "GDPR" in notes
        assert "bge-m3" in notes
        assert "not legal advice" in notes

    def test_no_compliance_notes_for_plain_english_project(self):
        assert compliance_notes(FAQ) == []

    def test_neutral_word_does_not_trigger_eu(self):
        prof = FAQ.model_copy(update={"compliance": ["neutral hosting"]})
        assert not any("EU" in n for n in compliance_notes(prof))


# --- profile extraction -------------------------------------------------------


class TestProfile:
    def test_corpus_bucket(self):
        assert corpus_bucket(50) == "tiny"
        assert corpus_bucket(500) == "small"
        assert corpus_bucket(5_000) == "medium"
        assert corpus_bucket(50_000) == "large"
        assert corpus_bucket(500_000) == "xlarge"

    def test_question_mix_normalizes(self):
        mix = QuestionMix(single_fact=2, relational_multi_hop=1, exploratory_multi_step=1)
        assert mix.single_fact == 50.0
        assert mix.single_fact + mix.relational_multi_hop + mix.exploratory_multi_step == 100.0

    def test_heuristic_english(self):
        p = extract_heuristic(
            "HR FAQ chatbot over 300 documents, answers under 2 seconds, tight budget. "
            "What is the parental leave policy?"
        )
        assert p.source == "heuristic"
        assert p.corpus_size_docs == 300 and p.corpus_size == "small"
        assert p.latency_budget_ms == 2_000
        assert p.cost_sensitivity == "high"
        assert "What is the parental leave policy?" in p.example_questions
        assert _ranked(p)[0] == "classic"

    def test_heuristic_french(self):
        p = extract_heuristic(
            "Nous avons 5 000 contrats en français entre nos fournisseurs et nos filiales. "
            "Les données doivent être hébergées en France, RGPD. Quel fournisseur est lié "
            "à quelle filiale ?"
        )
        assert p.corpus_size_docs == 5_000
        assert "fr" in p.languages
        assert "GDPR" in p.compliance
        assert p.data_residency == "EU"
        assert p.entity_richness == "high"
        assert p.question_mix.relational_multi_hop > 25

    def test_overrides_win(self):
        base = extract_heuristic("FAQ over 300 documents, tight budget, 2 seconds.")
        o = AdviseOverrides(
            corpus_size_docs=200_000,
            cost_sensitivity="med",
            latency_budget_ms=45_000,
            languages=["FR"],
            compliance=["EU only"],
            question_examples=["Who owns subsidiary X?"],
        )
        p = apply_overrides(base, o)
        assert p.corpus_size == "xlarge" and p.corpus_size_docs == 200_000
        assert p.cost_sensitivity == "medium"
        assert p.latency_budget_ms == 45_000
        assert p.languages == ["fr"]
        assert p.data_residency == "EU"
        assert p.example_questions == ["Who owns subsidiary X?"]
        assert set(p.overridden_fields) >= {"corpus_size_docs", "cost_sensitivity", "languages"}

    def test_request_options_by_model(self):
        legacy = _request_options("claude-haiku-4-5")
        assert legacy["temperature"] == 0
        assert legacy["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
        modern = _request_options("claude-sonnet-5")
        assert "temperature" not in modern
        assert modern["tool_choice"] == {"type": "auto"}

    async def test_llm_extraction_used_and_overrides_win(self):
        payload = {
            "corpus_size_docs": 40_000,
            "corpus_size": "small",  # contradicted by the count; the count wins
            "document_types": ["contracts"],
            "languages": ["fr"],
            "question_mix": {
                "single_fact": 20,
                "relational_multi_hop": 70,
                "exploratory_multi_step": 10,
            },
            "latency_budget_ms": None,
            "cost_sensitivity": "medium",
            "data_freshness": "weekly",
            "compliance": ["GDPR"],
            "data_residency": "EU",
            "entity_richness": "high",
            "example_questions": ["Quel fournisseur est lié à X ?"],
        }
        mock = AsyncMock(return_value=_tool_response(payload))
        with patch("app.advisor.profile._create_message", mock):
            p, tin, tout = await build_profile(
                "Nous avons 40 000 contrats...", AdviseOverrides(cost_sensitivity="high")
            )
        assert p.source == "llm"
        assert p.corpus_size == "large"
        assert p.cost_sensitivity == "high"  # override beats extraction
        assert p.overridden_fields == ["cost_sensitivity"]
        assert (tin, tout) == (400, 120)
        kwargs = mock.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["tools"][0]["name"] == TOOL_NAME

    async def test_falls_back_to_heuristic_on_provider_error(self):
        mock = AsyncMock(side_effect=RuntimeError("503 overloaded"))
        with patch("app.advisor.profile._create_message", mock):
            p, tin, tout = await build_profile("FAQ over 300 documents", AdviseOverrides())
        assert p.source == "heuristic"
        assert (tin, tout) == (0, 0)

    async def test_falls_back_when_model_skips_tool(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Sure!")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        with patch("app.advisor.profile._create_message", AsyncMock(return_value=resp)):
            p, _, _ = await build_profile("FAQ over 300 documents", AdviseOverrides())
        assert p.source == "heuristic"

    async def test_no_key_skips_provider(self, settings):
        settings.anthropic_api_key = ""
        mock = AsyncMock()
        with patch("app.advisor.profile._create_message", mock):
            p, _, _ = await build_profile("FAQ over 300 documents", AdviseOverrides())
        assert p.source == "heuristic"
        mock.assert_not_called()


# --- validation helpers -------------------------------------------------------


def _card(strategy, **kw):
    base = {
        "strategy": strategy,
        "questions": 4,
        "errors": 0,
        "refusals": 0,
        "refusal_rate": 0.0,
        "avg_latency_ms": 1000.0,
        "max_latency_ms": 1500,
        "total_input_tokens": 400,
        "total_output_tokens": 100,
        "avg_tokens_per_question": 125.0,
        "judged": 0,
    }
    base.update(kw)
    return StrategyScorecard(**base)


class TestValidateHelpers:
    def test_token_f1(self):
        assert token_f1("Paris is the capital", "the capital is Paris") == 1.0
        assert token_f1("", "x") == 0.0
        assert 0 < token_f1("Paris, France", "Paris") < 1

    def test_default_strategies(self):
        assert default_strategies(None) == ["classic", "graph"]
        assert default_strategies(RESEARCH)[0] == "agentic"

    def test_winner_by_judge_then_refusals(self):
        winner, reason = pick_winner(
            [
                _card("classic", avg_judge_score=0.7, judged=4),
                _card("agentic", avg_judge_score=0.8, judged=4, refusals=1, refusal_rate=0.25),
            ]
        )
        assert winner == "agentic" and "judge score" in reason

    def test_winner_without_quality_signal_uses_refusals_then_tokens(self):
        winner, _ = pick_winner(
            [
                _card("graph", avg_tokens_per_question=200.0),
                _card("classic", avg_tokens_per_question=120.0),
                _card("agentic", refusals=2, refusal_rate=0.5),
            ]
        )
        assert winner == "classic"

    def test_no_winner_when_everything_failed(self):
        winner, _ = pick_winner([_card("classic", errors=4)])
        assert winner is None

    def test_request_caps(self):
        with pytest.raises(ValueError):
            ValidateRequest(questions=[{"question": f"q{i}"} for i in range(21)])
        with pytest.raises(ValueError):
            ValidateRequest(
                questions=[{"question": "q"}],
                strategies=["classic", "graph", "agentic", "classic"],
            )
        req = ValidateRequest(questions=[{"question": "q"}], strategies=["classic", "classic"])
        assert req.strategies == ["classic"]


# --- endpoints ----------------------------------------------------------------


def _result(answer="Answer.", refusal=False, tin=100, tout=20, latency=900):
    return StrategyResult(
        answer=answer,
        sources=[Source(chunk_id="c1", quote="context quote")],
        refusal=refusal,
        latency_ms=latency,
        input_tokens=tin,
        output_tokens=tout,
    )


class TestEndpoints:
    def test_advise_heuristic_without_key(self, client, settings):
        settings.anthropic_api_key = ""
        r = client.post(
            "/advise",
            json={
                "description": "Chatbot FAQ RH, 300 documents en français, réponse en 2 secondes.",
                "overrides": {"compliance": ["EU only", "GDPR"]},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["profile"]["source"] == "heuristic"
        assert body["top_strategy"] == "classic"
        assert len(body["recommendations"]) == 3
        assert body["recommendations"][0]["suggested_config"]["embedding_model"] == "BAAI/bge-m3"
        assert any("EU region" in n for n in body["compliance_notes"])
        assert body["next_step"]["validate_endpoint"] == "/advise/validate"
        assert len(body["next_step"]["suggested_strategies"]) == 2
        assert "hybrid_routing" in body

    def test_advise_with_llm(self, client):
        payload = {
            "corpus_size": "xlarge",
            "corpus_size_docs": 300_000,
            "document_types": ["contracts"],
            "languages": ["en"],
            "question_mix": {
                "single_fact": 15,
                "relational_multi_hop": 75,
                "exploratory_multi_step": 10,
            },
            "cost_sensitivity": "medium",
            "data_freshness": "monthly",
            "compliance": [],
            "entity_richness": "high",
            "example_questions": ["Which subsidiaries does supplier X serve?"],
        }
        with patch(
            "app.advisor.profile._create_message", AsyncMock(return_value=_tool_response(payload))
        ), patch("app.api.advisor.record_tokens") as rec:
            r = client.post("/advise", json={"description": "300k supplier contracts..."})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["profile"]["source"] == "llm"
        assert body["top_strategy"] == "graph"
        assert body["next_step"]["example_payload"]["questions"][0]["question"].startswith(
            "Which subsidiaries"
        )
        assert rec.call_args.kwargs["tokens_input"] == 400

    def test_advise_rejects_empty_description(self, client):
        assert client.post("/advise", json={"description": ""}).status_code == 422

    def test_validate_scorecard_and_winner(self, client):
        async def fake_run(question, *, strategy, model=None, **_):
            if strategy == "agentic":
                return _result("I cannot answer.", refusal=True, tin=300, tout=50), None
            return _result("Paris is the capital of France.", tin=100, tout=20), None

        judge_mock = AsyncMock(
            return_value={
                "faithfulness": 0.8,
                "answer_relevance": 1.0,
                "context_precision": 0.5,
                "context_recall": 0.5,
                "reasoning": "",
            }
        )
        with patch("app.advisor.validate.run_strategy_raw", side_effect=fake_run), patch(
            "app.advisor.validate.judge", judge_mock
        ), patch("app.api.advisor.record_tokens") as rec, patch(
            "app.api.advisor.charge"
        ) as charge:
            r = client.post(
                "/advise/validate",
                json={
                    "questions": [
                        {"question": "Capital of France?", "ideal_answer": "Paris"},
                        {"question": "Capital again?"},
                    ],
                    "strategies": ["classic", "agentic"],
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        cards = {c["strategy"]: c for c in body["scorecard"]}
        assert cards["classic"]["refusals"] == 0
        assert cards["agentic"]["refusal_rate"] == 1.0
        assert cards["classic"]["avg_judge_score"] == 0.9
        assert cards["agentic"]["avg_judge_score"] == 0.0
        assert body["measured_winner"] == "classic"
        assert len(body["rows"]) == 4
        assert judge_mock.await_count == 1  # only the non-refused answer with an ideal answer
        assert judge_mock.await_args.kwargs["ideal_answer"] == "Paris"
        assert rec.call_args.kwargs == {"tokens_input": 800, "tokens_output": 140}
        # 2 questions x (classic 1 + agentic 3) rate-limit units, charged up front.
        assert charge.call_args.args[1] == 8

    def test_validate_defaults_to_top_two_and_reports_errors(self, client):
        calls: list[str] = []

        async def fake_run(question, *, strategy, model=None, **_):
            calls.append(strategy)
            return None, "No documents indexed yet."

        with patch("app.advisor.validate.run_strategy_raw", side_effect=fake_run):
            r = client.post("/advise/validate", json={"questions": [{"question": "q?"}]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["strategies"] == ["classic", "graph"]
        assert calls == ["classic", "graph"]
        assert body["measured_winner"] is None
        assert all(c["errors"] == 1 for c in body["scorecard"])

    def test_validate_rejects_too_many_questions(self, client):
        r = client.post(
            "/advise/validate", json={"questions": [{"question": f"q{i}"} for i in range(21)]}
        )
        assert r.status_code == 422
