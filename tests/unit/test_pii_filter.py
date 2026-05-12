"""Tests for regex-based PII masking."""

from llm_firewall.filters.pii import mask_pii


def test_mask_pii_masks_multiple_entity_types():
    result = mask_pii(
        "Email me at jane@example.com or call 555-123-4567 using key sk-123456789012345678901234."
    )

    assert "jane@example.com" not in result.text
    assert "555-123-4567" not in result.text
    assert "sk-123456789012345678901234" not in result.text
    assert "EMAIL_ADDRESS" in result.masked_entities
    assert "PHONE_NUMBER" in result.masked_entities
    assert "API_KEY" in result.masked_entities


def test_mask_pii_reports_no_matches_for_clean_text():
    result = mask_pii("The capital of France is Paris.")

    assert result.text == "The capital of France is Paris."
    assert result.masked is False
    assert result.masked_entities == ()


# Numbers below come from each country's official "reserved for fictional use"
# range; none route to a real subscriber. Covers the categories an assistant
# is likely to volunteer when asked for sample phone numbers — the case that
# motivated broadening the phone regex past US-only formats.
def test_mask_pii_masks_international_phone_formats():
    samples = [
        "+1 (555) 555-0123",       # US NANP with country code
        "(555) 555-0123",           # US local
        "5555550123",               # bare 10-digit NANP
        "+44 7700 900123",          # UK mobile
        "+44 20 7946 0123",         # UK landline grouped
        "01632 960123",             # UK domestic trunk
        "+61 491 570 123",          # Australia mobile
        "(02) 9901 1234",           # Australia landline
        "+91 98765 43210",          # India mobile
        "+33 1 23 45 67 89",        # France
        "+49 30 12345678",          # Germany
        "+81 90 1234 5678",         # Japan
        "+15555550123",             # E.164 US
        "+447700900123",            # E.164 UK
        "+919876543210",            # E.164 India
    ]
    for sample in samples:
        result = mask_pii(sample)
        assert result.masked, f"expected {sample!r} to be masked"
        assert "PHONE_NUMBER" in result.masked_entities
        # Every digit should be redacted — the rendered output must not
        # contain a 7-or-more-digit substring leaking the original number.
        import re
        assert not re.search(r"\d{7}", result.text), (
            f"phone digits leaked through for {sample!r}: {result.text!r}"
        )


def test_mask_pii_does_not_mask_non_phone_digit_strings():
    # Guards against the broader phone regex turning generic numeric tokens
    # into false positives.
    for text in ["order #12345", "price was 1234.56", "ID:1234567", "year 2024"]:
        result = mask_pii(text)
        assert "PHONE_NUMBER" not in result.masked_entities, (
            f"false positive PHONE_NUMBER match in {text!r} -> {result.text!r}"
        )
