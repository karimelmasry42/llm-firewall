"""
Input Validator — runs every explicitly configured input classifier.
"""
from collections.abc import Sequence

from llm_firewall.classifiers.ensemble import ClassifierEnsemble, EnsembleValidationResult
from llm_firewall.classifiers.registry import ClassifierSpec, get_input_classifier_specs


class InputValidator:
    """Validates user prompts using every configured input classifier."""

    def __init__(self, classifier_specs: Sequence[ClassifierSpec] | None = None):
        """
        Load all configured input classifiers.

        Args:
            classifier_specs: Explicit classifier configuration entries.
        """
        self._ensemble = ClassifierEnsemble(
            classifier_specs or get_input_classifier_specs()
        )

    def is_malicious(self, prompt: str) -> bool:
        """
        Check if any input classifier blocks a prompt.

        Returns:
            True if one or more classifiers classify the prompt as malicious.
        """
        return not self.validate(prompt).passed

    def validate(self, prompt: str) -> EnsembleValidationResult:
        """Return the full ensemble result for a prompt."""
        return self._ensemble.validate(prompt)

    def warmup(self, prompt: str) -> None:
        """Run a non-logged warm-up inference to stabilize first-request latency."""
        self._ensemble.validate(prompt)

    @property
    def model_names(self) -> list[str]:
        """Return the configured input classifier names."""
        return self._ensemble.model_names

    def get_score(self, prompt: str) -> dict:
        """
        Get the aggregate input decision with per-model scores.

        Returns:
            Dict with 'is_malicious', aggregate confidence, and per-model details.
        """
        validation = self.validate(prompt)
        confidence = max(validation.scores_summary.values(), default=0.0)
        blocked_by = [result.filter_name for result in validation.failed_filters]

        return {
            "is_malicious": not validation.passed,
            "confidence": confidence,
            "blocked_by": blocked_by,
            "scores": validation.scores_summary,
            "detail": (
                f"Blocked by: {', '.join(blocked_by)}"
                if blocked_by
                else "All input classifiers passed"
            ),
        }
