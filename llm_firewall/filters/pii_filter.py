"""
PII Filter — Detects personally identifiable information using regex patterns.
"""
import re

from llm_firewall.filters import FilterResult


# Compiled regex patterns for PII detection
_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "Email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "Phone": re.compile(
        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "CreditCard": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "IPAddress": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}


class PiiFilter:
    """Scans text for PII using regex patterns."""

    FILTER_NAME = "PII"

    def scan(self, text: str) -> FilterResult:
        """
        Scan text for PII patterns.

        Returns FilterResult with passed=False if any PII is found.
        PII is binary (regex match), so confidence is 1.0 or 0.0.
        """
        detected = []
        for pii_type, pattern in _PATTERNS.items():
            if pattern.search(text):
                detected.append(pii_type)

        if detected:
            return FilterResult(
                passed=False,
                filter_name=self.FILTER_NAME,
                confidence=1.0,
                detail=f"🚩 [ALERT] PII DETECTED: {', '.join(detected)}",
            )

        return FilterResult(
            passed=True,
            filter_name=self.FILTER_NAME,
            confidence=0.0,
            detail="✅ [CLEAN] No PII detected",
        )
