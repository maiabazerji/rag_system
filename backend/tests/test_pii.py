"""Tests for PII detection and redaction.

Every detector is checked both ways: real identifiers are found, and numbers of
the right shape but a wrong check digit are not. The second half matters as
much as the first -- a detector that masks every nine-digit number would wreck
the corpus it is meant to protect.
"""
from dataclasses import dataclass

import pytest

from app.privacy import pii
from app.privacy.pii import (
    PIIDetector,
    PIIRejected,
    PIISpan,
    card_valid,
    detect,
    iban_valid,
    luhn_valid,
    mask,
    mask_value,
    nir_valid,
    redact,
    redact_for_telemetry,
    siren_valid,
    siret_valid,
)


def _types(text):
    return [s.type for s in detect(text)]


def _iban_with_check(country: str, bban: str) -> str:
    """Build a valid IBAN for any country code by computing its check digits."""
    rearranged = bban + country + "00"
    n = int("".join(str(int(ch, 36)) for ch in rearranged))
    return f"{country}{98 - n % 97:02d}{bban}"


@pytest.fixture(autouse=True)
def _default_detectors():
    pii.reset_detectors()
    yield
    pii.reset_detectors()


class TestLuhn:
    @pytest.mark.parametrize("n", ["4111111111111111", "79927398713", "0"])
    def test_valid(self, n):
        assert luhn_valid(n)

    @pytest.mark.parametrize("n", ["4111111111111112", "79927398710", "", "12a4"])
    def test_invalid(self, n):
        assert not luhn_valid(n)


class TestEmail:
    @pytest.mark.parametrize(
        "email",
        ["jean.dupont@example.fr", "a+tag@sub.domain.co.uk", "x_y%z@mail-host.io"],
    )
    def test_detected(self, email):
        assert _types(f"Contact: {email}.") == ["EMAIL"]

    @pytest.mark.parametrize(
        "text",
        ["not an email @ all", "a@b", ".dot@example.com", "dot.@example.com", "a..b@example.com"],
    )
    def test_rejected(self, text):
        assert "EMAIL" not in _types(text)

    def test_trailing_punctuation_is_not_part_of_the_span(self):
        text = "Write to marie@example.org."
        (span,) = detect(text)
        assert text[span.start : span.end] == "marie@example.org"


class TestFrenchPhone:
    @pytest.mark.parametrize(
        "phone",
        [
            "06 12 34 56 78",
            "0612345678",
            "01.23.45.67.89",
            "09-87-65-43-21",
            "+33 6 12 34 56 78",
            "+33612345678",
            "0033 1 23 45 67 89",
            "+33 (0)1 23 45 67 89",
        ],
    )
    def test_detected(self, phone):
        text = f"Appelez le {phone} demain"
        (span,) = detect(text)
        assert span.type == "PHONE"
        assert text[span.start : span.end] == phone

    @pytest.mark.parametrize(
        "text",
        [
            "00 12 34 56 78",  # no area digit
            "12 34 56 78 90",  # no leading 0
            "06 12 34 56",  # too short
            "+44 20 7946 0958",  # not French
            "061234567890",  # too long
        ],
    )
    def test_rejected(self, text):
        assert "PHONE" not in _types(text)


class TestNIR:
    @pytest.mark.parametrize(
        "nir",
        [
            "1 84 03 75 115 001 09",
            "184037511500109",
            "2 89 05 2A 123 456 78",  # Corse-du-Sud counts as 19
            "178112B04501282",  # Haute-Corse counts as 18
        ],
    )
    def test_valid(self, nir):
        assert nir_valid(nir)
        assert _types(f"NIR : {nir}") == ["NIR"]

    @pytest.mark.parametrize(
        "nir",
        ["1 84 03 75 115 001 10", "2 89 05 2B 123 456 78", "184037511500"],
    )
    def test_wrong_key_or_length_is_rejected(self, nir):
        assert not nir_valid(nir)
        assert "NIR" not in _types(nir)

    def test_found_right_after_a_near_miss(self):
        """A failed candidate must not swallow the start of a real NIR."""
        text = "ref 45 67 89 1 84 03 75 115 001 09"
        assert _types(text) == ["NIR"]


class TestIBAN:
    @pytest.mark.parametrize(
        "iban",
        [
            "FR76 3000 6000 0112 3456 7890 189",
            "FR7630006000011234567890189",
            "GB82 WEST 1234 5698 7654 32",
            "DE89370400440532013000",
            "BE68539007547034",
        ],
    )
    def test_detected(self, iban):
        assert iban_valid(iban)
        text = f"IBAN: {iban}"
        (span,) = detect(text)
        assert span.type == "IBAN"
        assert text[span.start : span.end] == iban

    def test_wrong_check_digits_are_rejected(self):
        assert not iban_valid("FR7730006000011234567890189")
        assert "IBAN" not in _types("FR77 3000 6000 0112 3456 7890 189")

    def test_wrong_length_for_the_country_is_rejected(self):
        assert not iban_valid("FR763000600001123456789018")

    def test_does_not_swallow_a_following_uppercase_word(self):
        text = "FR76 3000 6000 0112 3456 7890 189 SARL DUPONT"
        assert redact(text).text == "[IBAN] SARL DUPONT"

    def test_country_missing_from_the_length_table_still_detected(self):
        iban = _iban_with_check("ZZ", "12345678901234")
        assert "ZZ" not in pii.IBAN_LENGTHS
        assert _types(f"account {iban} ok") == ["IBAN"]


class TestSirenSiret:
    def test_siren(self):
        assert siren_valid("732 829 320")
        assert _types("SIREN 732 829 320") == ["SIREN"]
        assert _types("SIREN 732829320") == ["SIREN"]

    def test_siren_bad_luhn(self):
        assert not siren_valid("732829321")
        assert _types("SIREN 732829321") == []

    def test_siret(self):
        assert siret_valid("73282932000074")
        assert _types("SIRET 732 829 320 00074") == ["SIRET"]

    def test_siret_bad_luhn_leaves_its_valid_siren_masked(self):
        """The first nine digits of a SIRET are the company's SIREN."""
        assert not siret_valid("73282932000075")
        assert _types("SIRET 732 829 320 00075") == ["SIREN"]

    def test_la_poste_exception(self):
        """La Poste establishments use a digit-sum rule instead of Luhn."""
        number = "35600000000001"
        assert not luhn_valid(number)
        assert siret_valid(number)
        assert not siret_valid("35600000000002")


class TestCards:
    @pytest.mark.parametrize(
        "card",
        ["4111 1111 1111 1111", "5555-5555-5555-4444", "378282246310005", "2221000000000009"],
    )
    def test_detected(self, card):
        assert card_valid(card)
        assert _types(f"card {card} exp") == ["CARD"]

    def test_bad_luhn(self):
        assert not card_valid("4111111111111112")

    def test_unknown_network_prefix(self):
        """Luhn-valid, but no card network issues numbers starting with 9."""
        number = next(
            f"900000000000000{d}" for d in "0123456789" if luhn_valid(f"900000000000000{d}")
        )
        assert not card_valid(number)


class TestIPv4:
    def test_detected(self):
        assert _types("from 192.168.1.10 at noon") == ["IPV4"]

    @pytest.mark.parametrize("text", ["10.0.0.256", "1.2.3", "v1.2.3.4.5", "1.2.3.4a"])
    def test_rejected(self, text):
        assert "IPV4" not in _types(text)


class TestDetect:
    def test_spans_are_ordered_and_disjoint(self):
        text = (
            "Jean (jean@example.fr, 06 12 34 56 78) paie par FR76 3000 6000 0112 "
            "3456 7890 189 ; NIR 1 84 03 75 115 001 09."
        )
        spans = detect(text)
        assert [s.type for s in spans] == ["EMAIL", "PHONE", "IBAN", "NIR"]
        for a, b in zip(spans, spans[1:], strict=False):
            assert a.end <= b.start

    def test_longest_span_wins_an_overlap(self):
        """The digits inside an IBAN must not be reported separately."""
        spans = detect("FR76 3000 6000 0112 3456 7890 189")
        assert [s.type for s in spans] == ["IBAN"]

    def test_plain_prose_has_no_pii(self):
        text = (
            "Retrieval-augmented generation (RAG) indexes 600-word chunks, "
            "reranks the top 50 in 2024 and costs 1 234 EUR per month."
        )
        assert detect(text) == []

    def test_empty(self):
        assert detect("") == []


class TestRedact:
    TEXT = "Mail jean@example.fr or call 06 12 34 56 78 / 06 98 76 54 32."

    def test_mask_replaces_with_placeholders_and_counts(self):
        result = redact(self.TEXT, "mask")
        assert result.text == "Mail [EMAIL] or call [PHONE] / [PHONE]."
        assert result.counts == {"EMAIL": 1, "PHONE": 2}

    def test_off_leaves_text_untouched(self):
        result = redact(self.TEXT, "off")
        assert result.text == self.TEXT
        assert result.counts == {}

    def test_reject_raises_with_types_but_not_values(self):
        with pytest.raises(PIIRejected) as exc:
            redact(self.TEXT, "reject")
        assert exc.value.types == ["EMAIL", "PHONE"]
        assert "jean@example.fr" not in str(exc.value)
        assert isinstance(exc.value, ValueError)

    def test_reject_passes_clean_text_through(self):
        assert redact("nothing personal here", "reject").text == "nothing personal here"

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown PII mode"):
            redact("x", "shred")

    def test_mask_shorthand(self):
        assert mask("a@b.fr") == "[EMAIL]"

    def test_mask_value_walks_containers_but_not_keys(self):
        value = {
            "q": "mail a@b.fr",
            "a@b.fr": 1,
            "items": ["06 12 34 56 78", 3, None],
            "pair": ("x@y.fr", True),
        }
        assert mask_value(value) == {
            "q": "mail [EMAIL]",
            "a@b.fr": 1,
            "items": ["[PHONE]", 3, None],
            "pair": ("[EMAIL]", True),
        }


class TestPluggableDetector:
    def test_a_registered_detector_runs_alongside_the_builtins(self):
        @dataclass
        class NameDetector:
            """Stand-in for a NER model."""

            names: tuple[str, ...]

            def detect(self, text):
                for name in self.names:
                    start = text.find(name)
                    if start >= 0:
                        yield PIISpan(start, start + len(name), "PERSON")

        detector = NameDetector(("Claire Martin",))
        assert isinstance(detector, PIIDetector)
        pii.register_detector(detector)

        result = redact("Claire Martin <claire@example.fr>")
        assert result.text == "[PERSON] <[EMAIL]>"
        assert result.counts == {"PERSON": 1, "EMAIL": 1}

    def test_explicit_detector_list_overrides_the_registry(self):
        only_email = [d for d in pii.DEFAULT_DETECTORS if getattr(d, "type", "") == "EMAIL"]
        assert _types("a@b.fr 06 12 34 56 78") == ["EMAIL", "PHONE"]
        assert [s.type for s in detect("a@b.fr 06 12 34 56 78", only_email)] == ["EMAIL"]


class TestRedactForTelemetry:
    def test_masks_when_enabled(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_redact_traces", True)
        assert redact_for_telemetry("a@b.fr") == "[EMAIL]"

    def test_passes_through_when_disabled(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "pii_redact_traces", False)
        assert redact_for_telemetry("a@b.fr") == "a@b.fr"


class TestSettings:
    def test_pii_mode_is_validated(self):
        from app.config import Settings

        assert Settings(pii_mode_ingest="REJECT").pii_mode_ingest == "reject"
        with pytest.raises(ValueError, match="PII_MODE_INGEST"):
            Settings(pii_mode_ingest="shred")

    def test_defaults(self):
        from app.config import Settings

        s = Settings(_env_file=None)
        assert s.pii_mode_ingest == "mask"
        assert s.pii_redact_logs is True
        assert s.pii_redact_traces is True
        assert (s.retention_traces_days, s.retention_eval_runs_days, s.retention_audit_days) == (
            7,
            365,
            365,
        )
