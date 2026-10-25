"""The advisor speaks English and French: same scores, translated text."""
import string

from app.advisor.messages import MESSAGES, msg, normalize, pct
from app.advisor.schemas import ProjectProfile, QuestionMix, StrategyScorecard
from app.advisor.scoring import compliance_notes, hybrid_routing, score_strategies
from app.advisor.validate import pick_winner


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


class TestCatalog:
    def test_every_key_exists_in_both_languages_with_the_same_placeholders(self):
        assert set(MESSAGES["en"]) == set(MESSAGES["fr"])
        for key, en in MESSAGES["en"].items():
            assert _placeholders(en) == _placeholders(MESSAGES["fr"][key]), key

    def test_language_tags_normalize_and_fall_back_to_english(self):
        assert normalize("fr-FR") == "fr"
        assert normalize("de") == "en"
        assert normalize(None) == "en"
        assert msg("hybrid.single", "de") == MESSAGES["en"]["hybrid.single"]

    def test_percent_formatting(self):
        assert pct(0.8, "en") == "80%"
        assert pct(0.8, "fr") == "80 %"


RICH = ProjectProfile(
    corpus_size="xlarge",
    corpus_size_docs=250_000,
    languages=["fr", "en"],
    question_mix=QuestionMix(single_fact=40, relational_multi_hop=30, exploratory_multi_step=30),
    latency_budget_ms=2_000,
    cost_sensitivity="high",
    data_freshness="daily",
    entity_richness="high",
    compliance=["EU only", "GDPR", "on-prem"],
    data_residency="EU",
)


class TestScoring:
    def test_scores_do_not_depend_on_language(self):
        en = score_strategies(RICH, "en")
        fr = score_strategies(RICH, "fr")
        assert [(r.strategy, r.score) for r in en] == [(r.strategy, r.score) for r in fr]

    def test_french_text_differs_and_english_is_the_default(self):
        default = score_strategies(RICH)
        en = score_strategies(RICH, "en")
        fr = score_strategies(RICH, "fr")
        for d, e, f in zip(default, en, fr, strict=True):
            assert d.reasons == e.reasons and d.tradeoffs == e.tradeoffs
            assert set(f.reasons).isdisjoint(e.reasons)
            assert set(f.tradeoffs).isdisjoint(e.tradeoffs)
            assert d.suggested_config.notes == e.suggested_config.notes
        assert any("très grand" in r for rec in fr for r in rec.reasons)

    def test_routing_and_notes_are_translated(self):
        assert hybrid_routing(RICH, "fr").rationale != hybrid_routing(RICH, "en").rationale
        fr_notes = compliance_notes(RICH, "fr")
        en_notes = compliance_notes(RICH, "en")
        assert len(fr_notes) == len(en_notes) > 0
        assert any(n.startswith("RGPD") for n in fr_notes)
        assert any(n.startswith("Sur site") for n in fr_notes)
        assert fr_notes[-1].startswith("Ces notes sont")


def _card(strategy: str, judge: float | None) -> StrategyScorecard:
    return StrategyScorecard(
        strategy=strategy, questions=4, errors=0, refusals=1, refusal_rate=0.25,
        avg_latency_ms=1200.0, max_latency_ms=2000, total_input_tokens=400,
        total_output_tokens=40, avg_tokens_per_question=110.0, judged=4,
        avg_judge_score=judge,
    )


class TestWinner:
    def test_winner_reason_in_french(self):
        winner, reason = pick_winner([_card("classic", 0.8), _card("graph", 0.6)], "fr")
        assert winner == "classic"
        assert reason == (
            "classic obtient le meilleur score du juge (0,80), un taux de refus de "
            "25 %, 110 tokens et 1200 ms par question."
        )

    def test_winner_reason_in_english_is_unchanged(self):
        _, reason = pick_winner([_card("classic", 0.8), _card("graph", 0.6)])
        assert reason == (
            "classic has the best judge score (0.80), refusal rate 25%, "
            "110 tokens and 1200 ms per question."
        )

    def test_everything_failed_in_french(self):
        card = _card("classic", None).model_copy(update={"errors": 4})
        assert pick_winner([card], "fr")[1].startswith("Toutes les exécutions ont échoué")


class TestEndpoint:
    def test_advise_in_french(self, client, settings):
        settings.anthropic_api_key = ""
        body = {
            "description": "Assistant RH, 300 documents en français, réponse en 2 secondes.",
            "overrides": {"compliance": ["EU only", "GDPR"]},
        }
        en = client.post("/advise", json=body).json()
        fr = client.post("/advise", json={**body, "language": "fr"}).json()
        assert fr["top_strategy"] == en["top_strategy"]
        assert fr["next_step"]["summary"].startswith("Ce classement est une estimation")
        assert en["next_step"]["summary"].startswith("This ranking is a prior")
        assert fr["recommendations"][0]["reasons"] != en["recommendations"][0]["reasons"]
        assert any(n.startswith("RGPD") for n in fr["compliance_notes"])

    def test_unknown_language_is_rejected(self, client):
        r = client.post("/advise", json={"description": "x", "language": "de"})
        assert r.status_code == 422
