"""
Classifier loading for firewall input and output checks.
"""
from __future__ import annotations

import asyncio
import pickle
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from llm_firewall.filters import FilterResult
from llm_firewall.huggingface_toxicity import TinyToxicDetectorClassifier
from llm_firewall.model_registry import ClassifierSpec

_BLOCKING_LABELS = {
    "1",
    "block",
    "blocked",
    "deny",
    "denied",
    "harmful",
    "malicious",
    "nsfw",
    "refuse",
    "reject",
    "toxic",
    "unsafe",
}
_SAFE_LABELS = {
    "0",
    "allow",
    "allowed",
    "benign",
    "clean",
    "pass",
    "passed",
    "safe",
}


@dataclass
class EnsembleValidationResult:
    """Aggregated result from a classifier ensemble."""

    passed: bool
    results: list[FilterResult]

    @property
    def failed_filters(self) -> list[FilterResult]:
        """Return only the classifiers that blocked the text."""
        return [result for result in self.results if not result.passed]

    @property
    def scores_summary(self) -> dict[str, float]:
        """Return classifier_name -> blocking confidence."""
        return {result.filter_name: result.confidence for result in self.results}

    @property
    def latencies_summary(self) -> dict[str, float]:
        """Return classifier_name -> evaluation latency in milliseconds."""
        return {result.filter_name: result.latency_ms for result in self.results}


class PickleClassifier:
    """Runs a single pickle-backed classifier bundle."""

    def __init__(self, spec: ClassifierSpec):
        if spec.path is None:
            raise ValueError(f"{spec.name} is configured without a pickle path")

        self.path = spec.path
        self.name = spec.display_name or spec.name
        self._preprocess = spec.preprocess

        if not self.path.exists():
            raise FileNotFoundError(f"Classifier file not found: {self.path}")

        bundle = self._load_bundle()

        self._vectorizer, self._model = self._extract_components(bundle)

        if not hasattr(self._model, "predict"):
            raise ValueError(f"{self.path} does not contain a compatible classifier")

    def evaluate(self, text: str) -> FilterResult:
        """Classify text and return a firewall-friendly result."""
        started_at = perf_counter()
        normalized_text = self._preprocess(text)
        features = self._transform(normalized_text)
        prediction = self._model.predict(features)[0]
        blocked = self._is_blocking_label(prediction)
        confidence = self._blocking_confidence(features, prediction, blocked)
        latency_ms = round((perf_counter() - started_at) * 1000, 3)

        if blocked:
            return FilterResult(
                passed=False,
                filter_name=self.name,
                confidence=confidence,
                latency_ms=latency_ms,
                detail=f"[BLOCKED] {self.name} | Confidence: {confidence:.1%}",
            )

        return FilterResult(
            passed=True,
            filter_name=self.name,
            confidence=confidence,
            latency_ms=latency_ms,
            detail=f"[PASSED] {self.name} | Confidence: {1 - confidence:.1%}",
        )

    def _load_bundle(self) -> Any:
        """Load a classifier bundle from pickle or joblib serialization."""
        _patch_sklearn_pickle_compatibility()

        try:
            with self.path.open("rb") as handle:
                return pickle.load(handle)
        except (pickle.UnpicklingError, AttributeError, EOFError, ValueError):
            pass

        try:
            import joblib
        except ImportError as exc:
            raise ValueError(
                f"{self.path} requires joblib-compatible loading, but joblib is not installed"
            ) from exc

        return joblib.load(self.path)

    def _transform(self, text: str) -> Any:
        if self._vectorizer is None:
            return [text]
        return self._vectorizer.transform([text])

    def _extract_components(self, bundle: Any) -> tuple[Any | None, Any]:
        if not isinstance(bundle, dict):
            return None, bundle

        if "model" in bundle:
            return bundle.get("vectorizer"), bundle["model"]

        if "pipeline" in bundle and hasattr(bundle["pipeline"], "predict"):
            return None, bundle["pipeline"]

        for key in ("classifier", "estimator", "predictor"):
            candidate = bundle.get(key)
            if hasattr(candidate, "predict"):
                return bundle.get("vectorizer"), candidate

        for value in bundle.values():
            if hasattr(value, "predict"):
                return bundle.get("vectorizer"), value

        return bundle.get("vectorizer"), bundle

    def _blocking_confidence(self, features: Any, prediction: Any, blocked: bool) -> float:
        if not hasattr(self._model, "predict_proba"):
            return 1.0 if blocked else 0.0

        probabilities = self._model.predict_proba(features)[0]
        blocking_index = self._blocking_class_index()
        if blocking_index is not None and blocking_index < len(probabilities):
            return float(probabilities[blocking_index])

        classes = list(getattr(self._model, "classes_", []))
        if prediction in classes:
            return float(probabilities[classes.index(prediction)])

        if len(probabilities) == 2:
            return float(probabilities[1])

        return float(max(probabilities)) if blocked else 0.0

    def _blocking_class_index(self) -> int | None:
        classes = list(getattr(self._model, "classes_", []))
        for index, label in enumerate(classes):
            if self._is_blocking_label(label):
                return index

        if len(classes) == 2:
            return 1

        return None

    def _is_blocking_label(self, label: Any) -> bool:
        if isinstance(label, bool):
            return label

        if isinstance(label, (int, float)):
            return int(label) == 1

        normalized = str(label).strip().lower()
        if normalized in _BLOCKING_LABELS:
            return True
        if normalized in _SAFE_LABELS:
            return False

        try:
            return int(normalized) == 1
        except ValueError:
            return False


class ClassifierEnsemble:
    """Loads explicitly configured classifiers and evaluates them all."""

    def __init__(self, classifier_specs: Sequence[ClassifierSpec]):
        self._classifier_specs = list(classifier_specs)
        self._classifiers = [
            _build_classifier(spec)
            for spec in self._classifier_specs
        ]
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(self._classifiers))
        )

    @property
    def model_names(self) -> list[str]:
        return [classifier.name for classifier in self._classifiers]

    def validate(self, text: str) -> EnsembleValidationResult:
        """Evaluate text against every classifier in the ensemble."""
        results = [classifier.evaluate(text) for classifier in self._classifiers]
        return EnsembleValidationResult(
            passed=all(result.passed for result in results),
            results=results,
        )

    async def validate_async(self, text: str) -> EnsembleValidationResult:
        """Evaluate text against every classifier concurrently."""
        if not self._classifiers:
            return EnsembleValidationResult(passed=True, results=[])

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self._executor, classifier.evaluate, text)
            for classifier in self._classifiers
        ]
        results = list(await asyncio.gather(*tasks))
        return EnsembleValidationResult(
            passed=all(result.passed for result in results),
            results=results,
        )


def _patch_sklearn_pickle_compatibility() -> None:
    """Patch missing sklearn symbols needed to unpickle older classifier files."""
    try:
        import sklearn.linear_model._sgd_fast as sgd_fast
        import sklearn._loss._loss as losses
    except Exception:
        return

    if hasattr(sgd_fast, "Log"):
        return

    # Older SGDClassifier pickles reference sklearn.linear_model._sgd_fast.Log.
    # Newer sklearn versions expose the equivalent implementation as CyHalfBinomialLoss.
    sgd_fast.Log = losses.CyHalfBinomialLoss


def _build_classifier(spec: ClassifierSpec):
    """Instantiate the classifier backend configured for one registry entry."""
    if spec.backend == "pickle":
        return PickleClassifier(spec)

    if spec.backend == "huggingface_tiny_toxic_detector":
        return TinyToxicDetectorClassifier(spec)

    raise ValueError(f"Unsupported classifier backend '{spec.backend}' for {spec.name}")
