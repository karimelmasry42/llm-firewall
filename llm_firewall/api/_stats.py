"""
Aggregate stats over the decision log.

Lives in its own module because both `dashboard.py` (the HTTP routes)
and `_processing.py` (the SSE publisher) need to compute identical
totals — putting it in either of those files creates an import cycle,
since `dashboard.py` already imports from `_processing.py`.
"""
from __future__ import annotations


def _iter_total_latency_values(log: list[dict]) -> list[float]:
    """Extract valid per-request total latency values from the decision log."""
    return [
        float(entry["total_latency_ms"])
        for entry in log
        if isinstance(entry.get("total_latency_ms"), (int, float))
    ]


def _classifier_latency_ms(entry: dict) -> float | None:
    """Sum latencies of every classifier (input + output) for one log entry.

    The decision log stores per-classifier timings under `latencies_ms` with
    keys like `input:<name>` and `output:<name>`. Summing those gives the
    cost the firewall *itself* added to a request, exclusive of the
    upstream LLM call (which dominates `total_latency_ms` and isn't a
    fair representation of how fast our screening is).
    """
    latencies = entry.get("latencies_ms")
    if not isinstance(latencies, dict):
        return None
    total = 0.0
    counted = False
    for value in latencies.values():
        if isinstance(value, (int, float)):
            total += float(value)
            counted = True
    return total if counted else None


def compute_average_total_latency_ms(log: list[dict]) -> float:
    """Mean end-to-end latency across the current log (kept for compatibility)."""
    latencies = _iter_total_latency_values(log)
    if not latencies:
        return 0.0
    return round(sum(latencies) / len(latencies), 3)


def compute_average_classifier_latency_ms(log: list[dict]) -> float:
    """Mean of (sum of input+output classifier latencies) across the log.

    This is what the dashboard displays as "average latency" — it answers
    "how fast does the firewall make decisions?" rather than "how fast is
    the upstream LLM?"
    """
    latencies = [v for v in (_classifier_latency_ms(entry) for entry in log)
                 if v is not None]
    if not latencies:
        return 0.0
    return round(sum(latencies) / len(latencies), 3)


def compute_stats(log: list[dict]) -> dict:
    """Aggregate stats over the full decision log.

    Used by `/api/stats`, the SSE `snapshot` event, and each `decision`
    event so every consumer sees identical, internally-consistent numbers
    (totals match the per-decision counts; latency averages are over the
    same population).
    """
    return {
        "total": len(log),
        "blocked": sum(1 for entry in log if entry["decision"] == "BLOCKED"),
        "dropped": sum(1 for entry in log if entry["decision"] == "DROPPED"),
        "allowed": sum(1 for entry in log if entry["decision"] == "ALLOWED"),
        "errors": sum(1 for entry in log if entry["decision"] == "ERROR"),
        # Classifier-only mean (input + output classifiers summed). This is
        # what the dashboard surfaces as "average latency".
        "average_classifier_latency_ms": compute_average_classifier_latency_ms(log),
        # End-to-end including the upstream LLM call. Kept for API compatibility
        # and for anyone who actually wants the round-trip number.
        "average_total_latency_ms": compute_average_total_latency_ms(log),
    }
