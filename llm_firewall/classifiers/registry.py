"""
Hard-coded classifier registry for the LLM firewall.

Each classifier is explicitly configured with a source and a preprocessing
function instead of being auto-discovered from a directory.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re

TextPreprocessor = Callable[[str], str]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_models_dir() -> Path:
    """Resolve the models directory, honoring `LLM_FIREWALL_MODELS_DIR` if set."""
    override = os.environ.get("LLM_FIREWALL_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "data" / "models"


MODELS_DIR = _resolve_models_dir()


def identity_preprocessor(text: str) -> str:
    """Return text unchanged."""
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip surrounding spaces."""
    return " ".join(text.split())


_OVERRIDE_PAT = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|skip|discard|dismiss)\b",
    re.I,
)
_SYSTEM_PAT = re.compile(
    r"\b(system|instruction|prompt|directive|guideline|rule|constraint)\b",
    re.I,
)
_ROLEPLAY_PAT = re.compile(
    r"\b(pretend|imagine|act\s+as|roleplay|role-play|simulate|hypothetically|as\s+if|dan|jailbreak)\b",
    re.I,
)
_NONASCII_PAT = re.compile(r"[^\x00-\x7F]+")


def preprocess_injection_text(text: str) -> str:
    """Normalize text for prompt-injection detection before classification."""
    text = str(text).strip()
    text = _NONASCII_PAT.sub(" ", text)
    text = _OVERRIDE_PAT.sub(" __OVERRIDE__ ", text)
    text = _SYSTEM_PAT.sub(" __SYSTEM__ ", text)
    text = _ROLEPLAY_PAT.sub(" __ROLEPLAY__ ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


@dataclass(frozen=True)
class ClassifierSpec:
    """Configuration for one firewall classifier artifact."""

    name: str
    display_name: str | None = None
    path: Path | None = None
    preprocess: TextPreprocessor = identity_preprocessor
    backend: str = "pickle"
    model_id: str | None = None
    threshold: float = 0.5
    max_length: int = 128
    # Used by `HFSequenceClassifier` to locate the "blocking" label in
    # `model.config.id2label`. Set at most one. If both are None we fall back
    # to index 1 for binary models, or the first label whose name contains a
    # known blocking keyword (`injection`, `jailbreak`, etc.).
    injection_label_id: int | None = None
    injection_label_name: str | None = None


# v1 semantic input classifier. Replaces the old per-language Linear SVM
# specs (`linear_svm_input_classifier.pkl`, `linear_svm_spanish.pkl`) which
# operated on regex tag-substituted text and had no semantic understanding.
# See docs/input_classifier/ for the dataset/eval/model bake-off that
# justified this choice. To swap models (e.g. once
# meta-llama/Prompt-Guard-2-86M is approved + authenticated), replace the
# spec below — the runtime is generic.
INPUT_CLASSIFIER_SPECS = [
    ClassifierSpec(
        name="protectai_deberta_v3_prompt_injection_v2",
        display_name="protectai/deberta-v3-base-prompt-injection-v2",
        backend="huggingface_sequence",
        model_id="protectai/deberta-v3-base-prompt-injection-v2",
        preprocess=normalize_whitespace,
        injection_label_name="INJECTION",
        threshold=0.5,
        max_length=512,
    ),
]


# A single multilingual classifier handles every language. The router can
# still send `language` for logging/output-side use, but every key returns
# the same spec list here.
INPUT_CLASSIFIER_SPECS_BY_LANGUAGE = {
    "en": list(INPUT_CLASSIFIER_SPECS),
    "es": list(INPUT_CLASSIFIER_SPECS),
}


OUTPUT_CLASSIFIER_SPECS = [
    ClassifierSpec(
        name="tiny_toxic_detector",
        display_name="Tiny-Toxic-Detector",
        preprocess=normalize_whitespace,
        backend="huggingface_tiny_toxic_detector",
        model_id="AssistantsLab/Tiny-Toxic-Detector",
        threshold=0.5,
        max_length=128,
    ),
]


def get_input_classifier_specs() -> list[ClassifierSpec]:
    """Return the hard-coded input classifier registry."""
    return list(INPUT_CLASSIFIER_SPECS)


def get_input_classifier_specs_by_language() -> dict[str, list[ClassifierSpec]]:
    """Return the hard-coded input classifiers grouped by routed language."""
    return {
        language: list(specs)
        for language, specs in INPUT_CLASSIFIER_SPECS_BY_LANGUAGE.items()
    }


def get_output_classifier_specs() -> list[ClassifierSpec]:
    """Return the hard-coded output classifier registry."""
    return list(OUTPUT_CLASSIFIER_SPECS)
