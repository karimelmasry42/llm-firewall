"""
Tiered English/Spanish language routing for input handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import os
import re

_ALPHA_TEXT_RE = re.compile(r"[^a-zA-Z\u00C0-\u024F]")
_SPANISH_HINT_WORDS = {
    "ayuda",
    "bueno",
    "buenos",
    "esta",
    "este",
    "gracias",
    "gusta",
    "hola",
    "mucho",
    "no",
    "producto",
    "recomiendo",
    "respuesta",
    "soporte",
}
_SPANISH_MARKERS = set("áéíóúñ¿¡")
_DEFAULT_FASTTEXT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "lid.176.bin"
)
_FASTTEXT_MODEL_PATH_ENV = "LLM_FIREWALL_LANGUAGE_ROUTER_FASTTEXT_PATH"


@dataclass(frozen=True)
class InputRouteDecision:
    """The selected route for one prompt."""

    lang: str
    confidence: float
    method: str
    target: str
    latency_ms: float


def _configured_fasttext_model_path() -> Path:
    configured_path = os.getenv(_FASTTEXT_MODEL_PATH_ENV, "").strip()
    if configured_path:
        return Path(configured_path).expanduser()
    return _DEFAULT_FASTTEXT_MODEL_PATH


@lru_cache(maxsize=1)
def _get_fasttext_model():
    try:
        import fasttext
    except ImportError:
        return None

    model_path = _configured_fasttext_model_path()
    if not model_path.exists():
        return None

    return fasttext.load_model(str(model_path))


@lru_cache(maxsize=1)
def _get_lingua_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        return None

    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.SPANISH,
    ).with_minimum_relative_distance(0.25).build()


def _detect_with_heuristic(text: str) -> tuple[str, float]:
    lowered = text.lower()
    words = {
        word
        for word in re.findall(r"[a-zA-Z\u00C0-\u024F]+", lowered)
        if word
    }
    if any(marker in lowered for marker in _SPANISH_MARKERS):
        return "es", 0.95
    if words & _SPANISH_HINT_WORDS:
        return "es", 0.75
    return "en", 0.75 if words else 0.0


def detect_with_fasttext(text: str) -> tuple[str, float]:
    """Detect a language code with fasttext when the model is available."""
    clean = text.replace("\n", " ").strip()
    if not clean:
        return "en", 0.0

    model = _get_fasttext_model()
    if model is None:
        return _detect_with_heuristic(clean)

    labels, scores = model.predict(clean, k=1)
    lang_code = labels[0].replace("__label__", "")
    return lang_code, float(scores[0])


def detect_with_lingua(text: str) -> tuple[str, float]:
    """Detect a language code with lingua when available."""
    detector = _get_lingua_detector()
    if detector is None:
        return _detect_with_heuristic(text)

    confidences = detector.compute_language_confidence_values(text)
    if not confidences:
        return "en", 0.0

    from lingua import Language

    lang_to_code = {
        Language.ENGLISH: "en",
        Language.SPANISH: "es",
    }
    top = confidences[0]
    return lang_to_code.get(top.language, "en"), float(top.value)


def detect_language(
    text: str,
    user_preferred_lang: str = "en",
    conversation_history_lang: str | None = None,
    confidence_threshold: float = 0.6,
) -> dict:
    """
    Detect English vs Spanish using fasttext, lingua, then fallback state.
    """
    cleaned = text.strip()
    alpha_text = _ALPHA_TEXT_RE.sub("", cleaned)
    if len(alpha_text) < 2:
        fallback = conversation_history_lang or user_preferred_lang
        return {
            "lang": fallback,
            "confidence": 0.0,
            "method": "fallback_no_alpha",
        }

    word_count = len(cleaned.split())
    if word_count > 5:
        lang, conf = detect_with_fasttext(cleaned)
        method = "fasttext"
    else:
        lang, conf = detect_with_lingua(cleaned)
        method = "lingua"

    if lang != "es":
        lang = "en"

    if conf < confidence_threshold:
        fallback = conversation_history_lang or user_preferred_lang
        return {
            "lang": fallback,
            "confidence": round(conf, 4),
            "method": f"fallback_{method}",
        }

    return {
        "lang": lang,
        "confidence": round(conf, 4),
        "method": method,
    }


def route_input_text(
    text: str,
    english_filter_name: str,
    spanish_filter_name: str,
    **kwargs,
) -> InputRouteDecision:
    """Route a prompt to the English or Spanish input filter."""
    started_at = perf_counter()
    detection = detect_language(text, **kwargs)
    lang = detection["lang"]
    target = english_filter_name
    if lang == "es":
        target = spanish_filter_name

    return InputRouteDecision(
        lang=lang,
        confidence=float(detection["confidence"]),
        method=str(detection["method"]),
        target=target,
        latency_ms=round((perf_counter() - started_at) * 1000, 3),
    )
