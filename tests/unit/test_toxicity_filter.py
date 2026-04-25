"""Tests for the Toxicity ML filter."""
import pytest
from llm_firewall.filters.toxicity_filter import ToxicityFilter


class TestToxicityFilter:
    """Test toxicity detection using the ML model."""

    @pytest.fixture(autouse=True)
    def setup(self, toxicity_model_path):
        self.filter = ToxicityFilter(toxicity_model_path)

    def test_detects_toxic_insult(self):
        result = self.filter.scan("You are such an idiot, go away.")
        assert not result.passed
        assert result.filter_name == "Toxicity"
        assert result.confidence > 0.5
        assert "ALERT" in result.detail

    def test_detects_toxic_hate(self):
        result = self.filter.scan("I hate you and everything you stand for.")
        assert not result.passed

    def test_detects_toxic_threat(self):
        result = self.filter.scan("Drop dead, you piece of garbage.")
        assert not result.passed

    def test_clean_text_passes(self):
        result = self.filter.scan("Thank you for your help!")
        assert result.passed
        assert "CLEAN" in result.detail

    def test_clean_text_low_confidence(self):
        result = self.filter.scan("This is a wonderful day.")
        assert result.passed
        assert result.confidence < 0.5

    def test_clean_factual_text(self):
        result = self.filter.scan("The capital of France is Paris.")
        assert result.passed

    def test_returns_confidence_score(self):
        result = self.filter.scan("You are worthless and disgusting.")
        assert 0.0 <= result.confidence <= 1.0
        assert not result.passed

    def test_filter_name(self):
        result = self.filter.scan("hello world")
        assert result.filter_name == "Toxicity"
