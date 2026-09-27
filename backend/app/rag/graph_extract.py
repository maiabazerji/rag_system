"""Extract (subject, predicate, object) triples from a chunk of text.

We ask Claude Haiku (cheap, fast) to return JSON triples. This is the *only* part
of the system where the LLM rewrites the corpus into a knowledge graph- once
extracted, retrieval is purely structural.
"""
from __future__ import annotations

import json
import re

from app.config import settings
from app.rag.graph_store import Triple
from app.rag.providers.anthropic_provider import generate_with_usage

_SYSTEM = (
    "You extract structured knowledge from text. "
    "Return ONLY a JSON array of triples, no prose. "
    "Each triple is an object {subject, predicate, object}. "
    "Use short canonical predicates like 'is_a', 'part_of', 'uses', 'depends_on', "
    "'introduced', 'compared_to', 'measured_by', 'located_in'. "
    "Keep entities short noun phrases. Skip vague or unfounded relations."
)

_USER_TEMPLATE = """Extract up to {max_triples} factual triples from this passage.
Only include relations the passage actually states.

Passage:
\"\"\"{text}\"\"\"

Return JSON only, e.g.:
[{{"subject":"BM25","predicate":"is_a","object":"sparse retrieval algorithm"}}]"""


_JSON_BLOCK = re.compile(r"\[\s*{.*}\s*]", re.DOTALL)


def _parse_triples(raw: str) -> list[dict]:
    if not raw:
        return []
    m = _JSON_BLOCK.search(raw)
    payload = m.group(0) if m else raw
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and {"subject", "predicate", "object"} <= d.keys()]


async def extract_triples(
    *,
    chunk_id: str,
    doc_id: str,
    text: str,
    max_triples: int = 8,
) -> list[Triple]:
    user = _USER_TEMPLATE.format(text=text[:3500], max_triples=max_triples)
    out = await generate_with_usage(
        model=settings.graph_extraction_model,
        prompt=user,
        system=_SYSTEM,
        max_tokens=600,
    )
    triples = _parse_triples(out["text"])
    return [
        Triple(
            subject=str(t["subject"]).strip(),
            predicate=str(t["predicate"]).strip(),
            object=str(t["object"]).strip(),
            chunk_id=chunk_id,
            doc_id=doc_id,
        )
        for t in triples
        if all(str(t[k]).strip() for k in ("subject", "predicate", "object"))
    ]


class QuestionEntities(list[str]):
    """Entities pulled from a question, plus the tokens spent extracting them.

    A plain list of entity strings, so existing callers are unaffected; the
    usage rides along as attributes for callers that account for cost.
    """

    input_tokens: int = 0
    output_tokens: int = 0


async def extract_question_entities(question: str) -> QuestionEntities:
    """Pull out candidate entities from a question- what to walk the graph from.

    Returns:
        The entities, carrying ``input_tokens`` / ``output_tokens`` for the
        extraction call even when nothing could be parsed.
    """
    out = await generate_with_usage(
        model=settings.graph_extraction_model,
        prompt=(
            f'List the key noun-phrase entities in this question as a JSON array of strings. '
            "If the question is not in English, also list each entity's English form, "
            "since the documents may be in English. "
            f'No prose. Question: "{question}"'
        ),
        system="You return only valid JSON.",
        max_tokens=200,
    )
    result = QuestionEntities()
    result.input_tokens = int(out.get("input_tokens", 0) or 0)
    result.output_tokens = int(out.get("output_tokens", 0) or 0)
    raw = out["text"].strip()
    m = re.search(r"\[.*]", raw, re.DOTALL)
    if not m:
        return result
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return result
    if isinstance(data, list):
        result.extend(s.strip() for s in data if isinstance(s, str) and s.strip())
    return result
