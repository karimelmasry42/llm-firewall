"""Small pickle-friendly fake models for firewall tests."""


class KeywordVectorizer:
    """Lowercase pass-through vectorizer."""

    def transform(self, texts):
        return [text.lower() for text in texts]


class KeywordClassifier:
    """Binary classifier that blocks when a keyword is present."""

    def __init__(self, blocked_keywords):
        self.blocked_keywords = tuple(keyword.lower() for keyword in blocked_keywords)
        self.classes_ = [0, 1]

    def predict(self, texts):
        return [1 if self._is_blocked(text) else 0 for text in texts]

    def predict_proba(self, texts):
        return [
            [0.08, 0.92] if self._is_blocked(text) else [0.94, 0.06]
            for text in texts
        ]

    def _is_blocked(self, text):
        normalized = text.lower()
        return any(keyword in normalized for keyword in self.blocked_keywords)


class KeywordPipeline:
    """Pipeline-shaped classifier bundle for pickle loader tests."""

    def __init__(self, blocked_keywords):
        self.vectorizer = KeywordVectorizer()
        self.classifier = KeywordClassifier(blocked_keywords)
        self.classes_ = self.classifier.classes_

    def predict(self, texts):
        return self.classifier.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts):
        return self.classifier.predict_proba(self.vectorizer.transform(texts))
