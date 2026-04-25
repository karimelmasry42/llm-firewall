"""
Classifier ensemble — runs every configured classifier and aggregates results.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from llm_firewall.classifiers.huggingface import TinyToxicDetectorClassifier
from llm_firewall.classifiers.pickle_classifier import PickleClassifier
from llm_firewall.classifiers.registry import ClassifierSpec
from llm_firewall.filters import FilterResult


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


class ClassifierEnsemble:
    """Loads explicitly configured classifiers and evaluates them all."""

    def __init__(self, classifier_specs: Sequence[ClassifierSpec]):
        self._classifier_specs = list(classifier_specs)
        self._classifiers = [
            _build_classifier(spec) for spec in self._classifier_specs
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


def _build_classifier(spec: ClassifierSpec):
    """Instantiate the classifier backend configured for one registry entry."""
    if spec.backend == "pickle":
        return PickleClassifier(spec)

    if spec.backend == "huggingface_tiny_toxic_detector":
        return TinyToxicDetectorClassifier(spec)

    raise ValueError(f"Unsupported classifier backend '{spec.backend}' for {spec.name}")
