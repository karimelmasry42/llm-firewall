"""Tests for the input language router."""

from llm_firewall.language_router import detect_language, route_input_text


def test_detect_language_uses_lingua_for_short_spanish(monkeypatch):
    monkeypatch.setattr(
        "llm_firewall.language_router.detect_with_lingua",
        lambda _text: ("es", 0.92),
    )

    result = detect_language("gracias")

    assert result == {"lang": "es", "confidence": 0.92, "method": "lingua"}


def test_detect_language_uses_fasttext_for_long_english(monkeypatch):
    monkeypatch.setattr(
        "llm_firewall.language_router.detect_with_fasttext",
        lambda _text: ("en", 0.88),
    )

    result = detect_language("This product is absolutely terrible and I want a refund.")

    assert result == {"lang": "en", "confidence": 0.88, "method": "fasttext"}


def test_detect_language_falls_back_when_input_has_no_alpha():
    result = detect_language("12345 😊", user_preferred_lang="en", conversation_history_lang="es")

    assert result == {"lang": "es", "confidence": 0.0, "method": "fallback_no_alpha"}


def test_route_input_text_uses_spanish_filter(monkeypatch):
    monkeypatch.setattr(
        "llm_firewall.language_router.detect_language",
        lambda _text, **_kwargs: {"lang": "es", "confidence": 0.99, "method": "lingua"},
    )

    result = route_input_text(
        "hola",
        english_filter_name="linear_svm_input_classifier",
        spanish_filter_name="linear_svm_spanish",
    )

    assert result.lang == "es"
    assert result.target == "linear_svm_spanish"


def test_route_input_text_keeps_english_filter(monkeypatch):
    monkeypatch.setattr(
        "llm_firewall.language_router.detect_language",
        lambda _text, **_kwargs: {"lang": "en", "confidence": 0.85, "method": "fasttext"},
    )

    result = route_input_text(
        "Say hi",
        english_filter_name="linear_svm_input_classifier",
        spanish_filter_name="linear_svm_spanish",
    )

    assert result.lang == "en"
    assert result.target == "linear_svm_input_classifier"
