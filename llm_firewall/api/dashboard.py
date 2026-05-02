"""
Dashboard, decision-log, stats, config, and health routes.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from llm_firewall.api._processing import MAX_LOG_SIZE, list_classifier_names
from llm_firewall.api.events import get_broadcaster

DASHBOARD_HTML_PATH = (
    Path(__file__).resolve().parents[2] / "dashboard" / "index.html"
)

router = APIRouter()


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


def _compute_average_total_latency_ms(log: list[dict]) -> float:
    """Mean end-to-end latency across the current log (kept for compatibility)."""
    latencies = _iter_total_latency_values(log)
    if not latencies:
        return 0.0
    return round(sum(latencies) / len(latencies), 3)


def _compute_average_classifier_latency_ms(log: list[dict]) -> float:
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


@router.get("/api/logs")
async def get_logs(request: Request, limit: int = 50):
    """Return the most recent decision log entries."""
    if limit < 1 or limit > MAX_LOG_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"`limit` must be between 1 and {MAX_LOG_SIZE}.",
        )
    return request.app.state.decision_log[:limit]


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
        "average_classifier_latency_ms": _compute_average_classifier_latency_ms(log),
        # End-to-end including the upstream LLM call. Kept for API compatibility
        # and for anyone who actually wants the round-trip number.
        "average_total_latency_ms": _compute_average_total_latency_ms(log),
    }


@router.get("/api/stats")
async def get_stats(request: Request):
    """Return aggregate stats for the dashboard."""
    return compute_stats(request.app.state.decision_log)


@router.get("/api/config")
async def get_config(request: Request):
    """Expose dashboard-safe runtime configuration."""
    state = request.app.state
    return {
        "upstream_chat_completions_url": state.settings.upstream_chat_completions_url,
        "default_model_id": state.settings.default_model_id,
        "input_models": list_classifier_names(state.input_classifier_specs),
        "output_models": list_classifier_names(state.output_classifier_specs),
        "enable_output_classifiers": state.settings.enable_output_classifiers,
        "refusal_message": state.settings.refusal_message,
        "conversation_cumulative_threshold": state.settings.conversation_cumulative_threshold,
        "conversation_max_tracked": state.settings.conversation_max_tracked,
    }


def _sse(event_name: str, payload: dict | str) -> bytes:
    """Encode one Server-Sent Event frame."""
    body = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {body}\n\n".encode("utf-8")


@router.get("/api/stream")
async def stream(request: Request):
    """Push decision-log updates to the dashboard over Server-Sent Events.

    On connect: emit one `snapshot` event carrying the current logs (up to
    100) plus stats so a fresh tab paints immediately. After that, only push
    a `decision` event when `log_decision` actually appends a new entry —
    no polling, no idle traffic. A 25-second `heartbeat` keeps the
    connection alive through proxies that close idle TCP.

    The decision log itself remains the source of truth at
    `app.state.decision_log`, exposed via `/api/logs` for clients that need
    to re-snapshot after a network blip.
    """
    broadcaster = get_broadcaster(request.app)
    queue = await broadcaster.subscribe()

    async def event_generator() -> AsyncIterator[bytes]:
        try:
            # Snapshot first so the dashboard can render before any new
            # decision arrives. Re-using the same shape as /api/logs +
            # /api/stats keeps the client code simple.
            full_log = request.app.state.decision_log
            # Cap the rendered list (fresh tabs need bounded HTML), but
            # always compute stats from the full log so totals reconcile
            # with the per-decision counts.
            snapshot = {
                "logs": list(full_log[:100]),
                "stats": compute_stats(full_log),
            }
            yield _sse("snapshot", snapshot)

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                except asyncio.TimeoutError:
                    yield _sse("heartbeat", {})
                    continue
                yield _sse("decision", event)
        finally:
            broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx-style buffering if proxied
        },
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "promptshield"}
