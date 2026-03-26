"""
Toxicity Filter — Detects toxic content using a pickle bundle ML model.
"""
import pickle

from llm_firewall.filters import FilterResult


class ToxicityFilter:
    """Scans text for toxic content using a trained sklearn bundle."""

    FILTER_NAME = "Toxicity"

    def __init__(self, model_path: str):
        """
        Load the toxicity model bundle.

        Args:
            model_path: Path to toxicity_classifier.pkl bundle.
        """
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self._vectorizer = bundle["vectorizer"]
        self._model = bundle["model"]

    def scan(self, text: str) -> FilterResult:
        """
        Scan text for toxic content.

        Returns FilterResult with passed=False if toxicity is detected (label 1).
        Includes confidence score from predict_proba.
        """
        X = self._vectorizer.transform([text])
        prediction = self._model.predict(X)[0]
        score = self._model.predict_proba(X)[0][1]

        if prediction == 1:
            return FilterResult(
                passed=False,
                filter_name=self.FILTER_NAME,
                confidence=score,
                detail=f"🚩 [ALERT] TOXIC CONTENT | Confidence: {score:.1%}",
            )

        return FilterResult(
            passed=True,
            filter_name=self.FILTER_NAME,
            confidence=score,
            detail=f"✅ [CLEAN] SAFE | Confidence: {1 - score:.1%}",
        )
