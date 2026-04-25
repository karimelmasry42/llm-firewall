"""
Output Validator — runs every explicitly configured output classifier.
"""
from collections.abc import Sequence

from llm_firewall.classifiers.ensemble import ClassifierEnsemble, EnsembleValidationResult
from llm_firewall.classifiers.registry import ClassifierSpec, get_output_classifier_specs

ValidationResult = EnsembleValidationResult


class OutputValidator:
    """Validates LLM output using every configured output classifier."""

    def __init__(self, classifier_specs: Sequence[ClassifierSpec] | None = None):
        self._ensemble = ClassifierEnsemble(
            classifier_specs or get_output_classifier_specs()
        )

    async def validate(self, text: str) -> ValidationResult:
        """Run all configured output classifiers."""
        return await self._ensemble.validate_async(text)

    def warmup(self, text: str) -> None:
        """Run a non-logged warm-up inference to stabilize first-request latency."""
        self._ensemble.validate(text)

    @property
    def model_names(self) -> list[str]:
        """Return the configured output classifier names."""
        return self._ensemble.model_names
