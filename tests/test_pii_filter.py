"""Tests for regex-based PII masking."""

from llm_firewall.pii_filter import mask_pii


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
