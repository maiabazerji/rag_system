"""Turn a free-text project description into a ``ProjectProfile``.

Two extractors produce the same shape:

- ``extract_with_llm``: Claude reads the description (any language) and fills a
  ``record_project_profile`` tool call. Models that still accept sampling
  parameters and forced ``tool_choice`` get both (``temperature=0`` and the tool
  forced); newer models reject those parameters with a 400, so they get
  ``tool_choice=auto`` plus an explicit instruction to call the tool.
- ``extract_heuristic``: English and French keyword rules. Used when no API key
  is configured or the provider call fails, so /advise always answers.

``apply_overrides`` then lets explicit caller-supplied facts win.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from app.advisor.schemas import (
    AdviseOverrides,
    ProjectProfile,
    QuestionMix,
    corpus_bucket,
)
from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.providers.anthropic_provider import _create_message

logger = get_structured_logger(__name__)

TOOL_NAME = "record_project_profile"

_LEVEL = {"type": "string", "enum": ["low", "medium", "high"]}

PROFILE_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Record the structured profile of a retrieval-augmented generation (RAG) project "
        "described by a customer. Call this exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "corpus_size_docs": {
                "type": ["integer", "null"],
                "description": "Approximate number of documents, or null if not stated.",
            },
            "corpus_size": {
                "type": "string",
                "enum": ["tiny", "small", "medium", "large", "xlarge"],
                "description": "tiny <100 docs, small <1k, medium <10k, large <100k, "
                "xlarge >=100k. Best guess when not stated.",
            },
            "document_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short English labels, e.g. contracts, support tickets, manuals.",
            },
            "languages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ISO 639-1 codes of the document and question languages.",
            },
            "question_mix": {
                "type": "object",
                "properties": {
                    "single_fact": {"type": "number"},
                    "relational_multi_hop": {"type": "number"},
                    "exploratory_multi_step": {"type": "number"},
                },
                "required": ["single_fact", "relational_multi_hop", "exploratory_multi_step"],
                "description": "Estimated percentages summing to 100. single_fact: one "
                "lookup answers it. relational_multi_hop: connects entities across "
                "documents. exploratory_multi_step: open-ended research needing several "
                "searches.",
            },
            "latency_budget_ms": {
                "type": ["integer", "null"],
                "description": "Acceptable time to answer in ms, or null if not stated.",
            },
            "cost_sensitivity": _LEVEL,
            "data_freshness": {
                "type": "string",
                "enum": ["static", "monthly", "weekly", "daily", "realtime"],
            },
            "compliance": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Stated compliance or residency needs, e.g. 'EU only', "
                "'GDPR', 'on-prem'. Empty if none stated. Do not invent any.",
            },
            "data_residency": {
                "type": ["string", "null"],
                "description": "Required hosting region, e.g. 'EU', or null.",
            },
            "entity_richness": {
                **_LEVEL,
                "description": "How much the documents revolve around named entities "
                "(people, companies, products, clauses) and their relationships.",
            },
            "example_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Example user questions, quoted or inferred from the text, "
                "in their original language. At most 10.",
            },
        },
        "required": [
            "corpus_size",
            "document_types",
            "languages",
            "question_mix",
            "cost_sensitivity",
            "data_freshness",
            "compliance",
            "entity_richness",
            "example_questions",
        ],
    },
}

SYSTEM_PROMPT = (
    "You analyse descriptions of retrieval-augmented generation projects, written in any "
    "language, and record a structured profile by calling the record_project_profile tool. "
    "Base every field on the description; where it is silent, choose the most plausible "
    "default rather than an extreme. Never invent compliance requirements."
)

# Models that reject sampling parameters and/or forced tool_choice. Everything
# else (Haiku 4.5, Sonnet/Opus 4.6 and older) accepts temperature=0 and a forced tool.
_MODERN_MODEL = re.compile(r"(sonnet-5|opus-5|opus-4-[78]|fable|mythos)")


def _request_options(model: str) -> dict[str, Any]:
    """Determinism controls the model accepts.

    Newer models return 400 for ``temperature`` and for forced ``tool_choice``
    (forced tool use is also incompatible with the adaptive thinking they run by
    default), so those get ``auto`` and rely on the system instruction.
    """
    if _MODERN_MODEL.search(model):
        return {"tool_choice": {"type": "auto"}}
    return {"tool_choice": {"type": "tool", "name": TOOL_NAME}, "temperature": 0}


async def extract_with_llm(description: str) -> tuple[ProjectProfile, int, int]:
    """Extract a profile with Claude.

    Returns:
        The profile (``source="llm"``) plus input and output token counts.

    Raises:
        Exception: Any provider error, or ``ValueError`` when no usable tool call
            comes back. Callers fall back to the heuristic.
    """
    model = settings.generator_model
    resp = await _create_message(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        tools=[PROFILE_TOOL],
        messages=[
            {
                "role": "user",
                "content": (
                    "Project description:\n<description>\n"
                    f"{description}\n</description>\n\n"
                    f"Call {TOOL_NAME} with the profile."
                ),
            }
        ],
        _operation="advisor_profile",
        **_request_options(model),
    )
    in_tok = int(getattr(resp.usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(resp.usage, "output_tokens", 0) or 0)
    block = next(
        (
            b
            for b in resp.content
            if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == TOOL_NAME
        ),
        None,
    )
    if block is None or not isinstance(block.input, dict):
        raise ValueError(f"model did not call {TOOL_NAME} (stop_reason={resp.stop_reason})")

    data = {k: v for k, v in dict(block.input).items() if v is not None}
    data["example_questions"] = list(data.get("example_questions", []))[:10]
    if isinstance(data.get("corpus_size_docs"), int):
        data["corpus_size"] = corpus_bucket(data["corpus_size_docs"])
    try:
        profile = ProjectProfile(**data, source="llm")
    except ValidationError as e:
        raise ValueError(f"profile tool input did not validate: {e.error_count()} errors") from e
    return profile, in_tok, out_tok


# --- heuristic fallback -------------------------------------------------------

_NUM_UNIT = re.compile(
    r"(\d[\d\s.,]*)\s*(k|m|million|millions|mille|thousand)?\s*"
    r"(?:de\s+|d')?(documents?|docs?|fichiers?|files?|pages?|pdfs?|articles?|tickets?|"
    r"contrats?|contracts?|records?|emails?|mails?)\b",
    re.IGNORECASE,
)
_LATENCY = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(ms|milliseconds?|millisecondes?|s|secs?|seconds?|secondes?)\b",
    re.IGNORECASE,
)

_LANGS = {
    "fr": r"\b(french|fran[cç]ais|francophone)\b",
    "en": r"\b(english|anglais)\b",
    "de": r"\b(german|allemand|deutsch)\b",
    "es": r"\b(spanish|espagnol|espa[nñ]ol)\b",
    "it": r"\b(italian|italien)\b",
    "nl": r"\b(dutch|n[eé]erlandais)\b",
    "pt": r"\b(portuguese|portugais)\b",
}
_FRENCH_STOPWORDS = re.compile(
    r"\b(nous|notre|nos|les|des|est|une|pour|avec|dans|sont|qui)\b", re.IGNORECASE
)

_SINGLE_FACT = (
    "faq", "lookup", "what is", "quick answer", "support", "policy", "policies", "definition",
    "procedure", "how do i", "hr", "helpdesk", "quel est", "quelle est", "combien", "prix",
    "price", "horaires", "politique", "comment faire",
)
_RELATIONAL = (
    "relationship", "relation", "connected", "between", "link", "lien", "entre", "dependenc",
    "supplier", "fournisseur", "graph", "entity", "entities", "entité", "multi-hop",
    "who works", "subsidiar", "filiale", "ownership", "cross-reference", "croiser",
)
_EXPLORATORY = (
    "research", "analysis", "analyse", "investigat", "explore", "synthes", "report on",
    "rapport", "why", "pourquoi", "open-ended", "due diligence", "recherche", "compare",
    "comparer", "trend", "tendance", "summari", "résum",
)
_ENTITIES = (
    "contract", "contrat", "legal", "juridique", "compan", "entreprise", "société",
    "supplier", "fournisseur", "people", "personne", "customer", "client", "product",
    "produit", "org chart", "organigramme", "crm", "erp", "part number", "patent", "brevet",
    "drug", "médicament", "gene", "clause",
)
_DOC_TYPES = {
    "contracts": ("contract", "contrat"),
    "support tickets": ("ticket",),
    "faq": ("faq",),
    "manuals": ("manual", "manuel", "documentation technique"),
    "emails": ("email", "e-mail", "courriel"),
    "reports": ("report", "rapport"),
    "policies": ("policy", "policies", "politique", "procédure", "procedure"),
    "wiki pages": ("wiki", "confluence", "notion"),
    "pdf documents": ("pdf",),
    "scientific papers": ("paper", "publication", "article scientifique"),
    "invoices": ("invoice", "facture"),
    "legal texts": ("regulation", "réglementation", "loi", "law"),
}


def _count(text: str, words: tuple[str, ...]) -> int:
    return sum(text.count(w) for w in words)


def _parse_number(raw: str, scale: str | None) -> int | None:
    digits = re.sub(r"[\s.,]", "", raw)
    if not digits.isdigit():
        return None
    n = int(digits)
    s = (scale or "").lower()
    if s in ("k", "mille", "thousand"):
        n *= 1_000
    elif s in ("m", "million", "millions"):
        n *= 1_000_000
    return n


def extract_heuristic(description: str) -> ProjectProfile:
    """Keyword-rule extraction (English and French). Always succeeds."""
    text = description.lower()

    docs: int | None = None
    for m in _NUM_UNIT.finditer(description):
        n = _parse_number(m.group(1), m.group(2))
        if n is not None:
            if m.group(3).lower().startswith("page"):
                n = max(1, n // 10)
            docs = max(docs or 0, n)

    langs = [code for code, pat in _LANGS.items() if re.search(pat, text)]
    if len(_FRENCH_STOPWORDS.findall(description)) >= 3 and "fr" not in langs:
        langs.append("fr")
    if not langs:
        langs = ["en"]

    sf = 1.5 + _count(text, _SINGLE_FACT)
    rel = 0.5 + _count(text, _RELATIONAL)
    exp = 0.5 + _count(text, _EXPLORATORY)
    mix = QuestionMix(single_fact=sf, relational_multi_hop=rel, exploratory_multi_step=exp)

    latency: int | None = None
    lat = _LATENCY.search(text)
    if lat:
        value = float(lat.group(1).replace(",", "."))
        latency = int(value if lat.group(2).startswith("m") else value * 1000)
        latency = max(100, latency)
    elif re.search(r"real[- ]time|temps r[ée]el|instant", text):
        latency = 2_000

    cost = "medium"
    if re.search(r"cost is not|budget is not|no budget constraint|budget g[ée]n[ée]reux|"
                 r"money is not", text):
        cost = "low"
    elif re.search(r"cheap|low cost|low-cost|tight budget|limited budget|pas cher|"
                   r"budget (serr|limit)|co[uû]ts? (bas|r[ée]duit|ma[iî]tris)|économ|minimi[sz]e cost",
                   text):
        cost = "high"

    freshness = "weekly"
    if re.search(r"real[- ]time|temps r[ée]el|\blive\b|continuous", text):
        freshness = "realtime"
    elif re.search(r"daily|every day|quotidien|chaque jour|tous les jours", text):
        freshness = "daily"
    elif re.search(r"monthly|mensuel|chaque mois", text):
        freshness = "monthly"
    elif re.search(r"static|archive|rarely|rarement|never change", text):
        freshness = "static"

    compliance: list[str] = []
    residency: str | None = None
    if re.search(r"\bgdpr\b|\brgpd\b", text):
        compliance.append("GDPR")
    if re.search(r"\beu\b|\bue\b|europe|union europ|h[ée]berg\w* en france|hosted in (the )?eu",
                 text):
        compliance.append("EU data residency")
        residency = "EU"
    if re.search(r"on[- ]?prem|on[- ]premise|sur site|self[- ]host|air[- ]?gap|souverain|"
                 r"sovereign", text):
        compliance.append("on-prem")
    if re.search(r"\bhipaa\b", text):
        compliance.append("HIPAA")
    if re.search(r"\bhds\b", text):
        compliance.append("HDS")

    ent_hits = _count(text, _ENTITIES)
    entity = "high" if ent_hits >= 3 else "medium" if ent_hits >= 1 else "low"

    doc_types = [label for label, kws in _DOC_TYPES.items() if any(k in text for k in kws)]
    questions = [
        q.strip(" -*•\"'«»“”")
        for q in re.findall(r"[^.!?\n]*\?", description)
        if len(q.strip()) > 8
    ][:10]

    return ProjectProfile(
        corpus_size=corpus_bucket(docs) if docs is not None else "small",
        corpus_size_docs=docs,
        document_types=doc_types,
        languages=langs,
        question_mix=mix,
        latency_budget_ms=latency,
        cost_sensitivity=cost,  # type: ignore[arg-type]
        data_freshness=freshness,  # type: ignore[arg-type]
        compliance=compliance,
        data_residency=residency,
        entity_richness=entity,  # type: ignore[arg-type]
        example_questions=questions,
        source="heuristic",
    )


def apply_overrides(profile: ProjectProfile, overrides: AdviseOverrides) -> ProjectProfile:
    """Return a copy of ``profile`` where every explicit override wins."""
    updates: dict[str, Any] = {}
    fields: list[str] = []

    if overrides.corpus_size_docs is not None:
        updates["corpus_size_docs"] = overrides.corpus_size_docs
        updates["corpus_size"] = corpus_bucket(overrides.corpus_size_docs)
        fields.append("corpus_size_docs")
    if overrides.languages:
        updates["languages"] = [lang.strip().lower() for lang in overrides.languages if lang]
        fields.append("languages")
    if overrides.latency_budget_ms is not None:
        updates["latency_budget_ms"] = overrides.latency_budget_ms
        fields.append("latency_budget_ms")
    if overrides.cost_sensitivity is not None:
        updates["cost_sensitivity"] = overrides.cost_sensitivity
        fields.append("cost_sensitivity")
    if overrides.data_freshness is not None:
        updates["data_freshness"] = overrides.data_freshness
        fields.append("data_freshness")
    if overrides.question_examples:
        updates["example_questions"] = overrides.question_examples[:20]
        fields.append("question_examples")
    if overrides.compliance is not None:
        updates["compliance"] = overrides.compliance
        joined = " ".join(overrides.compliance).lower()
        if re.search(r"\beu\b|\bue\b|europe|gdpr|rgpd", joined):
            updates["data_residency"] = "EU"
        fields.append("compliance")
    if overrides.entity_richness is not None:
        updates["entity_richness"] = overrides.entity_richness
        fields.append("entity_richness")
    if overrides.question_mix is not None:
        updates["question_mix"] = overrides.question_mix
        fields.append("question_mix")

    updates["overridden_fields"] = fields
    return profile.model_copy(update=updates)


async def build_profile(
    description: str, overrides: AdviseOverrides
) -> tuple[ProjectProfile, int, int]:
    """Extract a profile (LLM, else heuristic) and apply overrides.

    Returns:
        The profile plus the input and output tokens spent on extraction.
    """
    tokens_in = tokens_out = 0
    if settings.anthropic_api_key:
        try:
            profile, tokens_in, tokens_out = await extract_with_llm(description)
        except Exception as e:
            logger.warning(
                f"Advisor profile extraction failed, using heuristic: {type(e).__name__}: {e}",
                extra_fields={"error_type": type(e).__name__},
            )
            profile = extract_heuristic(description)
    else:
        profile = extract_heuristic(description)
    return apply_overrides(profile, overrides), tokens_in, tokens_out
