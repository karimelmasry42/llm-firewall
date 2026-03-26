"""
Filters package — shared types.
"""
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Result of a filter scan."""
    passed: bool
    filter_name: str
    confidence: float = 0.0
    latency_ms: float = 0.0
    detail: str = ""
