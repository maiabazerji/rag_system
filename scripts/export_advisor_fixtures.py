"""Export the Python advisor for the in-browser (static demo) engine.

The static Netlify build runs the advisor in the browser
(``frontend/src/advisor-engine``), a TypeScript port of ``app/advisor``. This
script keeps the two from drifting apart. It writes, under
``frontend/src/advisor-engine/generated/``:

- ``messages.json``: the English and French message catalog, verbatim.
- ``engine.json``: the settings the scorer reads (at their defaults), the
  multilingual flag of the default embedding model, and the profile-extraction
  tool, system prompt and request options used for "bring your own key".
- ``fixtures.json``: ~30 POST /advise requests and the exact responses the real
  route returns for them (heuristic extraction, default settings). A vitest test
  asserts the TypeScript engine reproduces every one.

Re-run it whenever the Python advisor (profile, scoring, follow-ups, messages,
or the /advise route) or one of the settings above changes, then run the
frontend tests:

    PYTHONPATH=backend backend/.venv/Scripts/python scripts/export_advisor_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.advisor.messages import MESSAGES  # noqa: E402
from app.advisor.profile import (  # noqa: E402
    PROFILE_TOOL,
    SYSTEM_PROMPT,
    TOOL_NAME,
    _request_options,
)
from app.config import Settings, settings  # noqa: E402
from app.rag.embed import is_multilingual_model  # noqa: E402

OUT = ROOT / "frontend" / "src" / "advisor-engine" / "generated"

#: Settings the scorer and the route read. Fixtures use their defaults, so a
#: developer's .env cannot leak into them.
ENGINE_SETTINGS = (
    "embedding_model",
    "reranker_model",
    "generator_model",
    "graph_extraction_model",
    "agentic_model",
    "agentic_max_iters",
)

#: Model the browser uses for "bring your own key" extraction: the cheapest.
BYOK_MODEL = "claude-haiku-4-5"

DESCRIPTIONS = [
    "We are an insurance company with about 20,000 PDF contracts and policy documents in "
    "French and English. Agents ask things like 'Which clauses cover water damage for policy "
    "X?' and 'Which partners are linked to claim Y?'. Answers should come back in under 5 "
    "seconds. Data must stay in the EU (GDPR).",
    "Nous sommes une PME de 40 personnes. Nous voulons un assistant RH qui répond aux "
    "questions des salariés sur nos 300 documents : congés, mutuelle, télétravail. Quel est "
    "le délai de prévenance pour un congé ? Réponse en 2 secondes, budget serré, hébergé en "
    "France (RGPD).",
    "Internal FAQ chatbot for our support team. 150 help-center articles in English, updated "
    "daily. Quick answers, cheap to run.",
    "Due diligence research assistant for analysts: 2 million pages of filings, annual "
    "reports and news. Open-ended questions, why did margins drop, compare competitors, "
    "summarise trends. Cost is not a concern, answers can take a minute.",
    "Knowledge graph of suppliers, subsidiaries and contracts: who supplies which "
    "subsidiary, what is the ownership chain between entities? 50k documents, German and "
    "English, on-prem only, air-gapped.",
    "Un outil de recherche pour nos juristes : 8 000 contrats et décisions de justice, "
    "questions ouvertes, analyse et comparaison des clauses entre fournisseurs. Pas de "
    "contrainte de temps réel, budget généreux. Données sur site, souverain.",
    "Customer support tickets (around 1.5k per month), real-time answers in the chat "
    "widget, HIPAA compliant, English and Spanish.",
    "A small wiki of 60 Notion pages for onboarding. Nothing special.",
    "Chatbot interne pour un cabinet comptable, 2 000 PDF en français, questions sur les "
    "procédures et les liens entre clients et dossiers, données sensibles.",
    "FAQ chatbot for an e-commerce site, 300 help pages in English, customers ask simple "
    "questions like return policy or shipping times, must answer in under 2 seconds.",
    "Assistant de recherche pour une équipe juridique : 50 000 contrats et décisions, les "
    "avocats posent des questions ouvertes qui demandent de croiser plusieurs documents, la "
    "latence n'est pas un problème, hébergement en France obligatoire.",
    "Assistant pour les commerciaux : catalogue produits, prix, fiches techniques. Combien "
    "coûte le modèle X ? Quelle est la garantie ? Mise à jour chaque mois.",
    "Scientific papers archive (12 000 publications), researchers ask why a method fails "
    "and want a synthesis of the literature; the archive rarely changes. Latency 30 s ok.",
    "Plateforme de veille réglementaire : lois, réglementation, rapports. Tendances et "
    "liens entre textes. Mise à jour quotidienne, hébergement en Europe, HDS.",
    "Emails and invoices of 3 m customers, CRM and ERP exports; finance asks which customer "
    "is linked to which invoice and supplier. Low cost, 800 ms latency.",
    "",
]

#: (description index, overrides) pairs run in both languages.
OVERRIDE_CASES: list[tuple[int, dict[str, Any]]] = [
    (0, {"corpus_size_docs": 300, "compliance": ["EU only", "GDPR"]}),
    (1, {"latency_budget_ms": 60_000, "cost_sensitivity": "low",
         "question_mix": {"single_fact": 10, "relational_multi_hop": 20,
                          "exploratory_multi_step": 70}}),
    (2, {"languages": ["FR", " en "], "entity_richness": "high", "data_freshness": "static"}),
    (3, {"question_examples": ["Why did margins drop?", "Compare A and B"],
         "compliance": []}),
    (7, {"question_mix": {"single_fact": 0, "relational_multi_hop": 0,
                          "exploratory_multi_step": 0}}),
    (8, {"question_mix": {"single_fact": 1, "relational_multi_hop": 1,
                          "exploratory_multi_step": 1}, "corpus_size_docs": 99_999}),
]


def _defaults() -> dict[str, Any]:
    fields = Settings.model_fields
    return {name: fields[name].default for name in ENGINE_SETTINGS}


def _use_defaults() -> None:
    """Point the shared settings at the values the fixtures must reflect."""
    for name, value in _defaults().items():
        setattr(settings, name, value)
    settings.anthropic_api_key = ""  # heuristic extraction only: no model call
    settings.require_api_key = False
    settings.telemetry_mode = "off"
    settings.metrics_enabled = False
    settings.wandb_enabled = False


def _cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for lang in ("en", "fr"):
        for d in DESCRIPTIONS:
            if d:
                cases.append({"description": d, "overrides": {}, "language": lang})
        for i, ov in OVERRIDE_CASES:
            cases.append({"description": DESCRIPTIONS[i] or "x", "overrides": ov, "language": lang})
    return cases


def main() -> None:
    _use_defaults()
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    def advise(req: dict[str, Any]) -> dict[str, Any]:
        r = client.post("/advise", json=req)
        r.raise_for_status()
        return r.json()

    fixtures = []
    for req in _cases():
        fixtures.append({"request": req, "response": advise(req)})

    # Follow-up rounds, like the UI: answers accumulate and are re-sent.
    for d in (DESCRIPTIONS[0], DESCRIPTIONS[1], DESCRIPTIONS[7], DESCRIPTIONS[3]):
        for lang, pick in (("en", 0), ("fr", -1)):
            answers: dict[str, Any] = {}
            for _ in range(2):
                req = {"description": d, "overrides": dict(answers), "language": lang}
                resp = advise(req)
                fixtures.append({"request": req, "response": resp})
                questions = resp["follow_up_questions"]
                if not questions:
                    break
                for q in questions:
                    answers.update(q["options"][pick]["overrides"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "messages.json").write_text(
        json.dumps(MESSAGES, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    engine = {
        "settings": _defaults(),
        "embedding_is_multilingual": is_multilingual_model(_defaults()["embedding_model"]),
        "byok": {
            "model": BYOK_MODEL,
            "tool": PROFILE_TOOL,
            "tool_name": TOOL_NAME,
            "system": SYSTEM_PROMPT,
            "request_options": _request_options(BYOK_MODEL),
            "max_tokens": 8000,
        },
    }
    (OUT / "engine.json").write_text(
        json.dumps(engine, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (OUT / "fixtures.json").write_text(
        json.dumps(fixtures, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(fixtures)} fixtures and the catalog to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
