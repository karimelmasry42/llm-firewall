"""Tests for the output classifier ensemble."""
import pytest

from llm_firewall.validators.output_validator import OutputValidator


class TestOutputValidator:
    """Test output routing through all output classifiers."""

    @pytest.fixture(autouse=True)
    def setup(self, output_classifier_specs):
        self.validator = OutputValidator(output_classifier_specs)

    @pytest.mark.asyncio
    async def test_clean_text_passes_all_classifiers(self):
        result = await self.validator.validate("The capital of France is Paris.")
        assert result.passed
        assert len(result.results) == 2
        assert len(result.failed_filters) == 0

    @pytest.mark.asyncio
    async def test_toxic_text_fails(self):
        result = await self.validator.validate(
            "You are such an idiot, go away you worthless moron."
        )
        assert not result.passed
        failed_names = [result.filter_name for result in result.failed_filters]
        assert "toxicity_guard" in failed_names

    @pytest.mark.asyncio
    async def test_nsfw_text_fails(self):
        result = await self.validator.validate(
            "This response contains explicit NSFW instructions."
        )
        assert not result.passed
        failed_names = [result.filter_name for result in result.failed_filters]
        assert "isolated_nsfw_guardrail" in failed_names

    @pytest.mark.asyncio
    async def test_scores_summary_has_all_models(self):
        result = await self.validator.validate("Hello world.")
        scores = result.scores_summary
        assert "toxicity_guard" in scores
        assert "isolated_nsfw_guardrail" in scores

    @pytest.mark.asyncio
    async def test_model_names_are_loaded_from_registry(self):
        assert "isolated_nsfw_guardrail" in self.validator.model_names
        assert "toxicity_guard" in self.validator.model_names
