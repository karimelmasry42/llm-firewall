"""
Dummy LLM — OpenAI-compatible upstream used for local firewall testing.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings


class DummyLLMSettings(BaseSettings):
    """Settings for the local dummy LLM server."""

    api_key: str = ""
    response_text: str = "This is a dummy response."

    model_config = {
        "env_prefix": "DUMMY_LLM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def _build_openai_response(content: str, model: str) -> dict:
    """Build a minimal OpenAI-compatible completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def create_dummy_llm_app(settings: DummyLLMSettings | None = None) -> FastAPI:
    """Create the dummy upstream app."""
    app = FastAPI(
        title="Dummy LLM",
        description="OpenAI-compatible upstream that always returns a fixed response.",
        version="0.1.0",
    )
    app.state.settings = settings or DummyLLMSettings()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Return the fixed dummy response, optionally enforcing a bearer token."""
        settings = request.app.state.settings
        auth_header = request.headers.get("Authorization", "")

        if settings.api_key:
            expected = f"Bearer {settings.api_key}"
            if auth_header != expected:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid or missing API key."}},
                )

        return JSONResponse(
            content=_build_openai_response(
                settings.response_text,
                "dummy-llm",
            )
        )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "dummy-llm"}

    return app


app = create_dummy_llm_app()


def run() -> None:
    """Console-script entry point for `llm-firewall-dummy`."""
    import uvicorn

    uvicorn.run(
        "llm_firewall.api.dummy_llm:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )
