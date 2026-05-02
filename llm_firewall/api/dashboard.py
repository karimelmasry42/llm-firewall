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
from llm_firewall.api._stats import compute_stats
from llm_firewall.api.events import get_broadcaster

DASHBOARD_HTML_PATH = (
    Path(__file__).resolve().parents[2] / "dashboard" / "index.html"
)

router = APIRouter()


@router.get("/api/logs")
async def get_logs(request: Request, limit: int = 50):
    """Return the most recent decision log entries."""
    if limit < 1 or limit > MAX_LOG_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"`limit` must be between 1 and {MAX_LOG_SIZE}.",
        )
    return request.app.state.decision_log[:limit]


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
    queue = broadcaster.subscribe()

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
        except (asyncio.CancelledError, ConnectionError, GeneratorExit):
            # Client closed the SSE stream mid-iteration. Without this
            # except, an orphan subscriber lives until the next heartbeat
            # boundary (up to 25s). The `finally` below still runs, so
            # the queue is unsubscribed promptly.
            pass
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
