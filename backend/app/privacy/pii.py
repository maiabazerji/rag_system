"""Detect and redact personal data (PII) in free text.

Every detector pairs a pattern with a validity check, so a random run of digits
is not mistaken for an identifier just because it has the right length:

======== ================================================================
Type     Validation
======== ================================================================
EMAIL    RFC-ish shape; no leading, trailing or doubled dot in the local part
PHONE    French numbers, ``0X XX XX XX XX`` or ``+33 X XX XX XX XX``
NIR      French social security number; key = 97 - (first 13 digits mod 97),
         with Corsica's ``2A``/``2B`` counted as 19/18
IBAN     Any country; ISO 13616 length for known countries, mod-97 == 1
SIRET    14 digits, Luhn (or La Poste's digit-sum rule)
SIREN    9 digits, Luhn
CARD     13-19 digits, card-network prefix, Luhn
IPV4     Four octets, each 0-255
======== ================================================================

Person names are deliberately not detected: no regular expression tells
"Claire Martin" from "Claude Monet" or "Pont Neuf". A named-entity model (e.g.
spaCy's ``fr_core_news_md``) can be plugged in by implementing
:class:`PIIDetector` and passing it to :func:`register_detector`.

Validation keeps false positives low, not absent. SIREN in particular is a
bare nine-digit number with a 1-in-10 Luhn check, so ``mask`` mode will
occasionally mask an unrelated number. That is the conservative failure.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from app.config import settings

PIIMode = Literal["off", "mask", "reject"]

EMAIL = "EMAIL"
PHONE = "PHONE"
NIR = "NIR"
IBAN = "IBAN"
SIRET = "SIRET"
SIREN = "SIREN"
CARD = "CARD"
IPV4 = "IPV4"


@dataclass(frozen=True)
class PIISpan:
    """One detected piece of personal data: ``text[start:end]`` is of ``type``."""

    start: int
    end: int
    type: str

    def __len__(self) -> int:
        return self.end - self.start


@runtime_checkable
class PIIDetector(Protocol):
    """Anything that can find personal data in text.

    Implement this to add a detector the built-in patterns cannot express, such
    as a named-entity model for person names, and register it with
    :func:`register_detector`. Spans may overlap other detectors' spans;
    :func:`detect` resolves overlaps.
    """

    def detect(self, text: str) -> Iterable[PIISpan]:
        """Yield every span of ``text`` this detector recognises."""
        ...


class PIIRejected(ValueError):
    """Raised by :func:`redact` in ``reject`` mode when personal data is found.

    Only the *types* found are carried, never the values: the exception text
    ends up in logs and HTTP responses.
    """

    def __init__(self, types: Iterable[str]):
        self.types = sorted(set(types))
        super().__init__(
            "Document contains personal data (" + ", ".join(self.types) + ") and "
            "PII_MODE_INGEST is 'reject'. Remove it, or ingest with 'mask'."
        )


@dataclass(frozen=True)
class Redaction:
    """The outcome of :func:`redact`: the (possibly masked) text and per-type counts."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def luhn_valid(number: str) -> bool:
    """Luhn (mod 10) check over a string of digits."""
    if not number.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def nir_valid(raw: str) -> bool:
    """Validate a French NIR (numéro de sécurité sociale), spaces allowed.

    The key is ``97 - (n mod 97)`` where ``n`` is the first 13 characters read
    as a number, with Corsica's department codes ``2A`` and ``2B`` read as
    ``19`` and ``18``.
    """
    compact = re.sub(r"\s", "", raw).upper()
    if len(compact) != 15:
        return False
    body, key = compact[:13], compact[13:]
    dept = body[5:7]
    if dept == "2A":
        body = body[:5] + "19" + body[7:]
    elif dept == "2B":
        body = body[:5] + "18" + body[7:]
    if not (body.isdigit() and key.isdigit()):
        return False
    return 97 - int(body) % 97 == int(key)


# ISO 13616 IBAN lengths. Countries missing here are still detected: the
# longest candidate that passes mod-97 wins.
IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28,
    "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24,
    "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IQ": 23,
    "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32,
    "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22,
    "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24,
    "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24,
    "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}  # fmt: skip
_IBAN_MIN, _IBAN_MAX = 15, 34


def iban_valid(raw: str) -> bool:
    """ISO 7064 mod-97 check of an IBAN, spaces allowed."""
    compact = raw.replace(" ", "").upper()
    if not (_IBAN_MIN <= len(compact) <= _IBAN_MAX) or not compact.isalnum():
        return False
    expected = IBAN_LENGTHS.get(compact[:2])
    if expected is not None and len(compact) != expected:
        return False
    rearranged = compact[4:] + compact[:4]
    try:
        n = int("".join(str(int(ch, 36)) for ch in rearranged))
    except ValueError:
        return False
    return n % 97 == 1


# La Poste's establishments share one SIREN and do not follow Luhn; INSEE
# validates them by requiring the digit sum to be a multiple of 5.
_LA_POSTE_SIREN = "356000000"


def siren_valid(raw: str) -> bool:
    digits = _digits(raw)
    return len(digits) == 9 and luhn_valid(digits)


def siret_valid(raw: str) -> bool:
    digits = _digits(raw)
    if len(digits) != 14:
        return False
    if digits.startswith(_LA_POSTE_SIREN) and digits != _LA_POSTE_SIREN + "00000":
        return sum(int(d) for d in digits) % 5 == 0
    return luhn_valid(digits)


_CARD_PREFIX = re.compile(r"^(?:4|5[1-5]|2[2-7]|3[47]|3(?:0[0-5]|[68])|35|6(?:011|2|4[4-9]|5))")


def card_valid(raw: str) -> bool:
    """Payment card number: known network prefix, 13-19 digits, Luhn."""
    digits = _digits(raw)
    return 13 <= len(digits) <= 19 and bool(_CARD_PREFIX.match(digits)) and luhn_valid(digits)


def phone_fr_valid(raw: str) -> bool:
    """A French number normalises to exactly nine national digits."""
    digits = _digits(raw.replace("(0)", ""))
    if digits.startswith("0033"):
        national = digits[4:]
    elif raw.lstrip().startswith("+33"):
        national = digits[2:]
    elif digits.startswith("0"):
        national = digits[1:]
    else:
        return False
    return len(national) == 9 and national[0] != "0"


def email_valid(raw: str) -> bool:
    local, _, domain = raw.rpartition("@")
    if not local or len(local) > 64 or len(domain) > 253:
        return False
    return not (local.startswith(".") or local.endswith(".") or ".." in local)


def ipv4_valid(raw: str) -> bool:
    parts = raw.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 for p in parts)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegexDetector:
    """Find candidates with a pattern, keep those that pass ``validate``."""

    type: str
    pattern: re.Pattern[str]
    validate: Callable[[str], bool]

    def detect(self, text: str) -> Iterator[PIISpan]:
        # Not finditer: a candidate that fails validation must not consume the
        # text a valid one starts in ("67 89 1 84 03 ..." before a NIR).
        pos = 0
        while (m := self.pattern.search(text, pos)) is not None:
            if self.validate(m.group(0)):
                yield PIISpan(m.start(), m.end(), self.type)
                pos = m.end()
            else:
                pos = m.start() + 1


class IBANDetector:
    """IBANs, which may be written in groups of four and run into following text.

    A greedy pattern happily swallows an uppercase word after the IBAN, so the
    candidate is cut to its country's registered length (or, for a country not
    in :data:`IBAN_LENGTHS`, to the longest prefix that passes mod-97).
    """

    type = IBAN
    _candidate = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,32}")

    def detect(self, text: str) -> Iterator[PIISpan]:
        pos = 0
        while (m := self._candidate.search(text, pos)) is not None:
            span = self._fit(text, m.start(), m.group(0))
            if span is not None:
                yield span
                pos = span.end
            else:
                pos = m.start() + 1

    def _fit(self, text: str, start: int, candidate: str) -> PIISpan | None:
        compact = candidate.replace(" ", "")
        known = IBAN_LENGTHS.get(compact[:2])
        lengths = [known] if known else range(min(len(compact), _IBAN_MAX), _IBAN_MIN - 1, -1)
        for length in lengths:
            if length > len(compact) or not iban_valid(compact[:length]):
                continue
            end = _end_after_n_alnum(text, start, length)
            # The IBAN must end at a token boundary, not in the middle of one.
            if end < len(text) and text[end].isalnum():
                continue
            return PIISpan(start, end, IBAN)
        return None


def _end_after_n_alnum(text: str, start: int, n: int) -> int:
    seen = 0
    for i in range(start, len(text)):
        if text[i].isalnum():
            seen += 1
            if seen == n:
                return i + 1
    return len(text)


_SEP = r"[ .\-]?"

_EMAIL_RE = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}(?![\w-])"
)
_PHONE_RE = re.compile(
    rf"(?<![\w+])(?:(?:\+|00)33{_SEP}(?:\(0\){_SEP})?|0)[1-9](?:{_SEP}\d{{2}}){{4}}(?!\d)"
)
_NIR_RE = re.compile(
    r"(?<![\w])[1-478](?: ?\d){4} ?(?:\d ?\d|2 ?[ABab])(?: ?\d){8}(?!\d)"
)
_SIRET_RE = re.compile(r"(?<![\w])\d{3} ?\d{3} ?\d{3} ?\d{5}(?!\d)")
_SIREN_RE = re.compile(r"(?<![\w])\d{3}[ .]?\d{3}[ .]?\d{3}(?!\d)")
_CARD_RE = re.compile(r"(?<![\w])\d(?:[ -]?\d){12,18}(?!\d)")
_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)")

# Order breaks ties between overlapping spans of equal length: earlier wins.
DEFAULT_DETECTORS: tuple[PIIDetector, ...] = (
    RegexDetector(EMAIL, _EMAIL_RE, email_valid),
    IBANDetector(),
    RegexDetector(NIR, _NIR_RE, nir_valid),
    RegexDetector(SIRET, _SIRET_RE, siret_valid),
    RegexDetector(CARD, _CARD_RE, card_valid),
    RegexDetector(SIREN, _SIREN_RE, siren_valid),
    RegexDetector(PHONE, _PHONE_RE, phone_fr_valid),
    RegexDetector(IPV4, _IPV4_RE, ipv4_valid),
)

_detectors: list[PIIDetector] = list(DEFAULT_DETECTORS)


def register_detector(detector: PIIDetector) -> None:
    """Add a detector (e.g. a NER model for names) to every later :func:`detect` call.

    Registered detectors rank after the built-ins when two spans of equal
    length overlap.
    """
    _detectors.append(detector)


def reset_detectors() -> None:
    """Restore the built-in detector set. Intended for tests."""
    _detectors[:] = list(DEFAULT_DETECTORS)


def detect(text: str, detectors: Iterable[PIIDetector] | None = None) -> list[PIISpan]:
    """Find personal data in ``text``.

    Overlapping candidates are resolved longest-first, then by detector order,
    so an IBAN is not also reported as the SIREN-shaped digits inside it.

    Args:
        text: Text to scan.
        detectors: Detectors to run. Defaults to the registered set.

    Returns:
        Non-overlapping spans, in text order.
    """
    if not text:
        return []
    active = list(_detectors if detectors is None else detectors)
    candidates: list[tuple[int, int, PIISpan]] = []
    for rank, detector in enumerate(active):
        for span in detector.detect(text):
            if span.end > span.start:
                candidates.append((-len(span), rank, span))
    candidates.sort(key=lambda c: (c[0], c[1], c[2].start))

    taken: list[PIISpan] = []
    for _, _, span in candidates:
        if all(span.end <= t.start or span.start >= t.end for t in taken):
            taken.append(span)
    return sorted(taken, key=lambda s: s.start)


def placeholder(pii_type: str) -> str:
    """The token a span of ``pii_type`` is replaced with, e.g. ``[EMAIL]``."""
    return f"[{pii_type}]"


def redact(
    text: str,
    mode: PIIMode | str = "mask",
    detectors: Iterable[PIIDetector] | None = None,
) -> Redaction:
    """Apply a PII policy to ``text``.

    Args:
        text: Text to scan.
        mode: ``off`` returns the text untouched without scanning it; ``mask``
            replaces each span with its :func:`placeholder`; ``reject`` raises
            if anything is found and otherwise returns the text unchanged.
        detectors: Detectors to run. Defaults to the registered set.

    Returns:
        The resulting text and how many spans of each type were found. Counts
        never include the values themselves.

    Raises:
        PIIRejected: In ``reject`` mode, when any personal data is found.
        ValueError: For an unknown mode.
    """
    if mode == "off":
        return Redaction(text)
    if mode not in ("mask", "reject"):
        raise ValueError(f"Unknown PII mode '{mode}'; expected off, mask or reject")

    spans = detect(text, detectors)
    counts = dict(Counter(s.type for s in spans))
    if not spans:
        return Redaction(text)
    if mode == "reject":
        raise PIIRejected(counts)

    parts: list[str] = []
    cursor = 0
    for s in spans:
        parts.append(text[cursor : s.start])
        parts.append(placeholder(s.type))
        cursor = s.end
    parts.append(text[cursor:])
    return Redaction("".join(parts), counts)


def mask(text: str) -> str:
    """Shorthand for ``redact(text, "mask").text``."""
    return redact(text, "mask").text


def mask_value(value: Any) -> Any:
    """Mask every string inside a JSON-like value (dicts, lists, tuples, sets).

    Dict keys are left alone: they are field names, not data. Other scalars
    pass through unchanged.
    """
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, dict):
        return {k: mask_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return type(value)(mask_value(v) for v in value)
    return value


def redact_for_telemetry(text: str) -> str:
    """Mask ``text`` before it leaves for a trace store or telemetry backend.

    Honours ``PII_REDACT_TRACES``, so exporters (in-memory traces, Langfuse,
    W&B) share one switch.
    """
    return mask(text) if settings.pii_redact_traces else text


__all__ = [
    "CARD",
    "DEFAULT_DETECTORS",
    "EMAIL",
    "IBAN",
    "IPV4",
    "NIR",
    "PHONE",
    "SIREN",
    "SIRET",
    "IBANDetector",
    "PIIDetector",
    "PIIMode",
    "PIIRejected",
    "PIISpan",
    "Redaction",
    "RegexDetector",
    "card_valid",
    "detect",
    "iban_valid",
    "luhn_valid",
    "mask",
    "mask_value",
    "nir_valid",
    "placeholder",
    "redact",
    "redact_for_telemetry",
    "register_detector",
    "reset_detectors",
    "siren_valid",
    "siret_valid",
]
