"""
Dashboard, decision-log, stats, config, and health routes.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from llm_firewall.api._processing import list_classifier_names

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


def _compute_average_total_latency_ms(log: list[dict]) -> float:
    """Compute the mean end-to-end latency across the current in-memory log."""
    latencies = _iter_total_latency_values(log)
    if not latencies:
        return 0.0
    return round(sum(latencies) / len(latencies), 3)


@router.get("/api/logs")
async def get_logs(request: Request, limit: int = 50):
    """Return the most recent decision log entries."""
    return request.app.state.decision_log[:limit]


@router.get("/api/stats")
async def get_stats(request: Request):
    """Return aggregate stats for the dashboard."""
    log = request.app.state.decision_log
    return {
        "total": len(log),
        "blocked": sum(1 for entry in log if entry["decision"] == "BLOCKED"),
        "dropped": sum(1 for entry in log if entry["decision"] == "DROPPED"),
        "allowed": sum(1 for entry in log if entry["decision"] == "ALLOWED"),
        "errors": sum(1 for entry in log if entry["decision"] == "ERROR"),
        "average_total_latency_ms": _compute_average_total_latency_ms(log),
    }


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
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "promptshield"}
