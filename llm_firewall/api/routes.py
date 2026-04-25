"""
OpenAI-compatible chat completion routes (single + batch) and model listing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm_firewall.api._processing import (
    process_chat_completion,
    resolve_upstream_api_key,
)
from llm_firewall.core.proxy import (
    list_upstream_models,
    retrieve_upstream_model,
)

logger = logging.getLogger("llm_firewall")
MAX_BATCH_SIZE = 1000
DEFAULT_BATCH_CONCURRENCY = 20
MAX_BATCH_CONCURRENCY = 100

router = APIRouter()


async def _read_json_object(request: Request) -> tuple[dict | None, JSONResponse | None]:
    """Parse a JSON object body or return a 400 response."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        return None, JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON body: {exc.msg}"}},
        )
    if not isinstance(body, dict):
        return None, JSONResponse(
            status_code=400,
            content={"error": {"message": "Request body must be a JSON object."}},
        )
    return body, None


def _build_model_object(model_id: str) -> dict:
    """Build a minimal OpenAI-compatible model object."""
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "promptshield",
    }


def _build_fallback_models_list(model_ids: list[str] | None = None) -> dict:
    """Build a fallback models list when the upstream model endpoint is unavailable."""
    effective_model_ids = model_ids or ["firewall-demo"]
    return {
        "object": "list",
        "data": [_build_model_object(model_id) for model_id in effective_model_ids],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Flow:
    1. Route the prompt to the matching input classifier.
    2. If all pass, forward to the configured upstream LLM URL.
    3. Run the assistant response through every output classifier.
    4. Return the response, or a refusal if any classifier blocks.
    """
    body, error = await _read_json_object(request)
    if error is not None:
        return error
    auth_header = request.headers.get("Authorization", "")
    result = await process_chat_completion(request.app, body, auth_header)
    return JSONResponse(
        status_code=result["status_code"],
        content=result["payload"],
    )


@router.get("/v1/models")
async def list_models(request: Request):
    """OpenAI-compatible model listing endpoint for SDK compatibility."""
    api_key = resolve_upstream_api_key(
        request.app.state.settings,
        request.headers.get("Authorization", ""),
    )

    try:
        payload = await list_upstream_models(
            llm_api_url=request.app.state.settings.upstream_chat_completions_url,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning("Falling back to local model list: %s", exc)
        payload = _build_fallback_models_list(
            [request.app.state.settings.default_model_id]
        )

    return JSONResponse(content=payload)


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, request: Request):
    """OpenAI-compatible model retrieval endpoint for SDK compatibility."""
    api_key = resolve_upstream_api_key(
        request.app.state.settings,
        request.headers.get("Authorization", ""),
    )

    try:
        payload = await retrieve_upstream_model(
            model_id=model_id,
            llm_api_url=request.app.state.settings.upstream_chat_completions_url,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning("Falling back to local model object for %s: %s", model_id, exc)
        payload = _build_model_object(model_id)

    return JSONResponse(content=payload)


@router.post("/v1/chat/completions/batch")
async def chat_completions_batch(request: Request):
    """
    Batch testing endpoint for sending up to 1000 prompts in one request.

    Each prompt is wrapped into its own OpenAI-style request body and routed
    through the same firewall path as `/v1/chat/completions`.
    """
    body, error = await _read_json_object(request)
    if error is not None:
        return error
    prompts = body.get("prompts")
    model = body.get("model", request.app.state.settings.default_model_id)
    system_message = body.get("system_message")
    concurrency = body.get("concurrency", DEFAULT_BATCH_CONCURRENCY)

    if not isinstance(prompts, list) or not prompts:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "`prompts` must be a non-empty array of strings."}},
        )

    if len(prompts) > MAX_BATCH_SIZE:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Batch requests are limited to {MAX_BATCH_SIZE} prompts."}},
        )

    if not isinstance(model, str) or not model.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "`model` must be a non-empty string."}},
        )

    if system_message is not None and not isinstance(system_message, str):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "`system_message` must be a string when provided."}},
        )

    if not all(isinstance(prompt, str) for prompt in prompts):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Every item in `prompts` must be a string."}},
        )

    if not isinstance(concurrency, int) or concurrency < 1:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "`concurrency` must be an integer greater than 0."}},
        )

    if concurrency > MAX_BATCH_CONCURRENCY:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"`concurrency` must not exceed {MAX_BATCH_CONCURRENCY}."}},
        )

    auth_header = request.headers.get("Authorization", "")
    semaphore = asyncio.Semaphore(min(concurrency, len(prompts)))

    async def _process_batch_item(index: int, prompt: str) -> dict:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        async with semaphore:
            result = await process_chat_completion(
                request.app,
                {"model": model, "messages": messages},
                auth_header,
            )

        return {
            "index": index,
            "prompt": prompt,
            "http_status": result["status_code"],
            "decision": result["decision"],
            "content": result["content"],
            "scores": result["scores"],
            "latencies_ms": result["latencies_ms"],
            "total_latency_ms": result["total_latency_ms"],
            "detail": result["detail"],
            "failed_filters": result["failed_filters"],
        }

    results = list(
        await asyncio.gather(
            *[
                _process_batch_item(index, prompt)
                for index, prompt in enumerate(prompts)
            ]
        )
    )

    summary = {
        "total": len(results),
        "allowed": sum(1 for result in results if result["decision"] == "ALLOWED"),
        "blocked": sum(1 for result in results if result["decision"] == "BLOCKED"),
        "dropped": sum(1 for result in results if result["decision"] == "DROPPED"),
        "errors": sum(1 for result in results if result["decision"] == "ERROR"),
    }

    return JSONResponse(
        content={
            "id": f"batch-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.batch",
            "created": int(time.time()),
            "model": model,
            "system_message": system_message,
            "concurrency": min(concurrency, len(prompts)),
            "summary": summary,
            "results": results,
        }
    )
