"""Lightweight language support: detection, text folding and localized messages.

Deliberately dependency-free. Detection is a stopword-ratio heuristic over the
five languages the project targets (English, French, German, Spanish, Italian):
it is right on ordinary questions and sentences, which is all it is used for
(picking the language of a refusal message). It is not a general-purpose
language identifier and says so by falling back to a default when unsure.

The request's ``Accept-Language`` header, when present, supplies that default
(see :class:`AcceptLanguageMiddleware`), so a question too short to classify,
such as "BM25 ?", still gets a message in the user's UI language.
"""
from __future__ import annotations

import re
import unicodedata
from contextvars import ContextVar

from starlette.types import ASGIApp, Receive, Scope, Send

SUPPORTED_LANGUAGES = ("en", "fr", "de", "es", "it")
DEFAULT_LANGUAGE = "en"

# Function words only: frequent, topic-independent, and distinctive enough
# between these five languages. Stored accent-folded (see ``fold``).
_STOPWORDS_RAW: dict[str, str] = {
    "en": (
        "the a an and or of to in on for with from by at as is are was were be been "
        "it its this that these those what which who whom whose how why when where "
        "does do did can could should would will not no than then there their they "
        "you your we our i my he she his her them if into about between vs"
    ),
    "fr": (
        "le la les l un une des du de d au aux et ou mais donc car ni que qu qui quoi "
        "quel quelle quels quelles comment pourquoi quand où est sont était être avoir "
        "a ont ce cet cette ces c il elle ils elles on nous vous je j tu me te se s "
        "son sa ses leur leurs mon ma mes notre nos votre vos dans sur pour par avec "
        "sans sous entre chez ne n pas plus moins y en faut peut doit lorsque"
    ),
    "de": (
        "der die das den dem des ein eine einer eines einem einen und oder aber ist "
        "sind war waren sein wie was warum wann wo welche welcher welches wer nicht "
        "kein keine mit für auf aus bei nach von zu zum zur im ins ich du er sie es "
        "wir ihr man sich auch noch nur kann können soll sollte wird werden"
    ),
    "es": (
        "el la los las un una unos unas y o pero de del al que qué quien quién cual "
        "cuál cuáles como cómo por porque para con sin sobre entre es son era está "
        "están ser estar se su sus lo le les no más muy también cuando cuándo donde "
        "dónde hay puede este esta estos estas ese esa mi tu nos"
    ),
    "it": (
        "il lo la i gli le un uno una di del dello della dei degli delle da dal "
        "dalla in nel nella con su sul sulla per tra fra e ed o ma che chi cosa "
        "come perché quando dove quale quali è sono era essere ha hanno non si ci "
        "ne questo questa questi quello quella anche più molto può"
    ),
}

_TOKEN = re.compile(r"\w+", re.UNICODE)


def fold(text: str) -> str:
    """Lowercase and strip accents: "Élève" -> "eleve", "Straße" -> "strasse".

    NFKD splits a letter from its combining accents, which are then dropped;
    ``casefold`` rather than ``lower`` also maps "ß" to "ss".
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Accent-folded Unicode word tokens. Apostrophes split: "l'index" -> l, index."""
    return _TOKEN.findall(fold(text))


STOPWORDS: dict[str, frozenset[str]] = {
    lang: frozenset(fold(words).split()) for lang, words in _STOPWORDS_RAW.items()
}

# Letters that exist in one or two of the languages only. Each occurrence adds
# a fraction of a stopword's weight: enough to break ties ("que" is French and
# Spanish), not enough to outvote real evidence.
_MARKERS: dict[str, str] = {
    "fr": "çœêâîôûëïùàè",
    "de": "äöüß",
    "es": "ñ¿¡áíóú",
    "it": "òìàè",
}
_MARKER_WEIGHT = 0.5


def detect_language(text: str, default: str | None = None) -> str:
    """Guess the language of ``text`` among :data:`SUPPORTED_LANGUAGES`.

    Scores each language by the fraction of tokens that are its stopwords
    (plus a small bonus for language-specific letters) and returns the best.
    Ties go to the language listed first in ``SUPPORTED_LANGUAGES``.

    Args:
        text: The text to classify, typically a question.
        default: Returned when there is no evidence either way. Defaults to the
            request's preferred language (from ``Accept-Language``), else "en".

    Returns:
        An ISO 639-1 code.
    """
    fallback = default or preferred_language()
    tokens = tokenize(text)
    if not tokens:
        return fallback
    lowered = text.casefold()
    scores: dict[str, float] = {}
    for lang in SUPPORTED_LANGUAGES:
        hits = sum(1 for tok in tokens if tok in STOPWORDS[lang])
        markers = sum(lowered.count(ch) for ch in _MARKERS.get(lang, ""))
        scores[lang] = (hits + _MARKER_WEIGHT * markers) / len(tokens)
    best = max(SUPPORTED_LANGUAGES, key=lambda lang: scores[lang])
    if scores[best] == 0:
        return fallback
    return best


# --- request language preference ---------------------------------------------

_preferred: ContextVar[str | None] = ContextVar("preferred_language", default=None)


def parse_accept_language(header: str | None) -> str | None:
    """First supported language in an ``Accept-Language`` header, by q-value.

    ``"fr-FR,fr;q=0.9,en;q=0.8"`` -> ``"fr"``. Unsupported or malformed entries
    are skipped; returns None when nothing usable is present.
    """
    if not header:
        return None
    ranked: list[tuple[float, int, str]] = []
    for i, part in enumerate(header.split(",")):
        tag, _, params = part.strip().partition(";")
        lang = tag.strip().split("-")[0].lower()
        if lang not in SUPPORTED_LANGUAGES:
            continue
        q = 1.0
        for param in params.split(";"):
            key, _, value = param.strip().partition("=")
            if key == "q":
                try:
                    q = float(value)
                except ValueError:
                    q = 0.0
        if q > 0:
            ranked.append((-q, i, lang))
    return min(ranked)[2] if ranked else None


def preferred_language() -> str:
    """The current request's preferred language, or :data:`DEFAULT_LANGUAGE`."""
    return _preferred.get() or DEFAULT_LANGUAGE


class AcceptLanguageMiddleware:
    """Expose the request's ``Accept-Language`` preference to :func:`preferred_language`.

    Pure ASGI (not ``BaseHTTPMiddleware``) so the context variable is visible to
    the route handler that runs inside it.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        header = next(
            (v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"accept-language"),
            None,
        )
        token = _preferred.set(parse_accept_language(header))
        try:
            await self.app(scope, receive, send)
        finally:
            _preferred.reset(token)


# --- localized messages --------------------------------------------------------

MESSAGES: dict[str, dict[str, str]] = {
    "no_documents": {
        "en": "No documents uploaded yet. Go to Ingest to upload files, "
        "then I can answer your questions.",
        "fr": "Aucun document n'a encore été importé. Allez sur la page Ingestion "
        "pour ajouter des fichiers, puis je pourrai répondre à vos questions.",
    },
    "no_context": {
        "en": "No relevant context found in the indexed documents.",
        "fr": "Aucun passage pertinent n'a été trouvé dans les documents indexés.",
    },
    "insufficient_context": {
        "en": "I cannot answer this from the retrieved documents: they do not "
        "contain the information needed.",
        "fr": "Je ne peux pas répondre à partir des documents récupérés : ils ne "
        "contiennent pas l'information nécessaire.",
    },
    "graph_not_built": {
        "en": "Graph RAG needs setup first. Go to Compare page and click 'Build graph' "
        "to extract connections between concepts in your documents.",
        "fr": "Le Graph RAG doit d'abord être configuré. Allez sur la page Comparer et "
        "cliquez sur « Construire le graphe » pour extraire les liens entre les "
        "concepts de vos documents.",
    },
    "graph_no_match": {
        "en": "Graph RAG found neither matching entities nor similar chunks.",
        "fr": "Le Graph RAG n'a trouvé ni entité correspondante ni passage similaire.",
    },
    "provider_unavailable": {
        "en": "The language model provider is unavailable right now. Try again shortly; "
        "the details are in the backend logs.",
        "fr": "Le fournisseur du modèle de langage est indisponible pour le moment. "
        "Réessayez dans un instant ; les détails sont dans les journaux du backend.",
    },
    "internal_error": {
        "en": "Something went wrong answering this question. "
        "The details are in the backend logs.",
        "fr": "Une erreur s'est produite en répondant à cette question. "
        "Les détails sont dans les journaux du backend.",
    },
    "agent_provider_failed": {
        "en": "The agent could not finish: the model provider failed mid-run. "
        "Try again shortly.",
        "fr": "L'agent n'a pas pu terminer : le fournisseur du modèle a échoué en cours "
        "d'exécution. Réessayez dans un instant.",
    },
    "agent_step_limit": {
        "en": "The agent reached its step limit without finding a grounded answer.",
        "fr": "L'agent a atteint sa limite d'étapes sans trouver de réponse étayée "
        "par les documents.",
    },
    "agent_no_answer": {
        "en": "(agent stopped without finishing)",
        "fr": "(l'agent s'est arrêté sans terminer)",
    },
}


def message(key: str, lang: str) -> str:
    """The message ``key`` in ``lang``, falling back to English.

    Raises:
        KeyError: If ``key`` is not a known message.
    """
    variants = MESSAGES[key]
    return variants.get(lang) or variants[DEFAULT_LANGUAGE]


def localized(key: str, question: str) -> str:
    """The message ``key`` in the language ``question`` is written in."""
    return message(key, detect_language(question))

