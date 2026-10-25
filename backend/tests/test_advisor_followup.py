"""Tests for the advisor's follow-up questions and confidence."""
from app.advisor.followup import MAX_QUESTIONS, confidence, follow_up_questions
from app.advisor.profile import apply_overrides
from app.advisor.schemas import AdviseOverrides, ProjectProfile, QuestionMix
from app.advisor.scoring import score_strategies


def _ask(profile: ProjectProfile):
    ranked = score_strategies(profile)
    return ranked, follow_up_questions(profile, ranked)


class TestFollowUpQuestions:
    def test_vague_project_asks_what_could_flip_the_pick_first(self):
        ranked, questions = _ask(ProjectProfile())
        assert 0 < len(questions) <= MAX_QUESTIONS
        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids))
        # Questions that can change the top strategy come before the rest.
        flags = [q.changes_recommendation for q in questions]
        assert flags == sorted(flags, reverse=True)
        assert flags[0] is True
        # Each option says which strategy would win with that answer: a big
        # corpus is what lets Graph overtake Classic on a neutral profile.
        size = next(q for q in questions if q.id == "corpus_size_docs")
        assert size.changes_recommendation
        assert {o.id: o.top_strategy for o in size.options}["more"] == "graph"
        assert {o.id: o.top_strategy for o in size.options}["lt100"] == "classic"
        # Data location never moves the ranking but always gets a slot.
        assert questions[-1].id == "compliance"
        assert not questions[-1].changes_recommendation

    def test_a_first_answer_can_open_the_next_question(self):
        # Research questions alone do not beat Classic at medium cost; once the
        # client says so, the budget becomes the deciding question.
        research = AdviseOverrides(question_mix=QuestionMix(
            single_fact=20, relational_multi_hop=20, exploratory_multi_step=60))
        _, questions = _ask(apply_overrides(ProjectProfile(), research))
        cost = next(q for q in questions if q.id == "cost_sensitivity")
        assert cost.changes_recommendation
        assert {o.id: o.top_strategy for o in cost.options}["low"] == "agentic"

    def test_confirmed_facts_are_never_asked_again(self):
        base = ProjectProfile()
        overrides = AdviseOverrides(
            question_mix=QuestionMix(single_fact=80, relational_multi_hop=10,
                                     exploratory_multi_step=10),
            corpus_size_docs=500,
            compliance=["EU only"],
        )
        profile = apply_overrides(base, overrides)
        _, questions = _ask(profile)
        asked = {q.id for q in questions}
        assert not asked & {"question_mix", "corpus_size_docs", "compliance"}

    def test_options_round_trip_through_overrides(self):
        _, questions = _ask(ProjectProfile())
        for q in questions:
            for option in q.options:
                profile = apply_overrides(ProjectProfile(), AdviseOverrides(**option.overrides))
                assert q.id in profile.overridden_fields
                assert score_strategies(profile)[0].strategy == option.top_strategy

    def test_answering_every_question_settles_the_advice(self):
        # Like the UI: answers accumulate and are re-sent with the description.
        answers: dict = {}
        for _ in range(4):
            profile = apply_overrides(ProjectProfile(), AdviseOverrides(**answers))
            _, questions = _ask(profile)
            if not any(q.changes_recommendation for q in questions):
                break
            for q in questions:
                answers.update(q.options[0].overrides)
        profile = apply_overrides(ProjectProfile(), AdviseOverrides(**answers))
        _, questions = _ask(profile)
        assert not any(q.changes_recommendation for q in questions)

    def test_current_guess_is_reported(self):
        profile = ProjectProfile(corpus_size="medium", corpus_size_docs=2_000,
                                 latency_budget_ms=3_000)
        _, questions = _ask(profile)
        by_id = {q.id: q for q in questions}
        if "corpus_size_docs" in by_id:
            assert by_id["corpus_size_docs"].current == "lt10k"
        if "latency_budget_ms" in by_id:
            assert by_id["latency_budget_ms"].current == "fast"

    def test_settled_pick_only_asks_where_data_may_go(self):
        # A clear lead that no answer can overturn: nudging questions would not
        # change the advice, so only the deployment question remains.
        profile = ProjectProfile(
            corpus_size="small",
            question_mix=QuestionMix(single_fact=90, relational_multi_hop=5,
                                     exploratory_multi_step=5),
            latency_budget_ms=2_000,
            cost_sensitivity="high",
        )
        ranked, questions = _ask(profile)
        assert ranked[0].score - ranked[1].score >= 15
        assert [q.id for q in questions] == ["compliance"]
        assert confidence(ranked, questions).level == "high"


class TestConfidence:
    def test_clear_margin_and_nothing_open_is_high(self):
        profile = ProjectProfile(
            corpus_size="small",
            question_mix=QuestionMix(single_fact=90, relational_multi_hop=5,
                                     exploratory_multi_step=5),
            latency_budget_ms=2_000,
            cost_sensitivity="high",
        )
        ranked = score_strategies(profile)
        c = confidence(ranked, [])
        assert c.level == "high"
        assert c.margin == ranked[0].score - ranked[1].score

    def test_open_questions_that_flip_the_pick_lower_confidence(self):
        ranked, questions = _ask(ProjectProfile())
        c = confidence(ranked, questions)
        assert c.open_questions == sum(q.changes_recommendation for q in questions)
        assert c.level in {"low", "medium"}


class TestEndpoint:
    def test_advise_returns_follow_ups_and_confidence(self, client, settings):
        settings.anthropic_api_key = ""
        r = client.post("/advise", json={"description": "Un assistant pour nos documents."})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["confidence"]["level"] in {"low", "medium", "high"}
        questions = body["follow_up_questions"]
        assert questions and all(q["options"] for q in questions)

        # Sending the chosen answers back as overrides removes those questions.
        answers: dict = {}
        for q in questions:
            answers.update(q["options"][0]["overrides"])
        r2 = client.post(
            "/advise", json={"description": "Un assistant pour nos documents.", "overrides": answers}
        )
        assert r2.status_code == 200, r2.text
        asked_again = {q["id"] for q in r2.json()["follow_up_questions"]}
        assert not asked_again & {q["id"] for q in questions}
