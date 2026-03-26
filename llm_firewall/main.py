"""
PromptShield — FastAPI application.

Routes prompts through the selected input classifier, forwards allowed prompts to the
configured LLM URL, and validates the response with every output classifier.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from llm_firewall.config import Settings
from llm_firewall.language_router import InputRouteDecision, route_input_text
from llm_firewall.model_registry import (
    ClassifierSpec,
    get_input_classifier_specs_by_language,
    get_output_classifier_specs,
)
from llm_firewall.proxy import (
    forward_to_llm,
    list_upstream_models,
    retrieve_upstream_model,
)
from llm_firewall.pii_filter import mask_pii
from llm_firewall.validators.input_validator import InputValidator
from llm_firewall.validators.output_validator import OutputValidator

logger = logging.getLogger("llm_firewall")
MAX_LOG_SIZE = 500
MAX_BATCH_SIZE = 1000
DEFAULT_BATCH_CONCURRENCY = 20
INPUT_WARMUP_TEXT_BY_LANGUAGE = {
    "en": "What is the capital of France?",
    "es": "Hola, necesito ayuda con mi cuenta.",
}
OUTPUT_WARMUP_TEXT = "This is a routine warmup response."


def _build_openai_response(content: str, model: str = "firewall") -> dict:
    """Build a response matching OpenAI's chat completion format."""
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


def _extract_bearer_token(auth_header: str) -> str:
    """Extract a bearer token value from an Authorization header."""
    if not auth_header:
        return ""
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return auth_header.strip()


def _resolve_upstream_api_key(settings: Settings, auth_header: str = "") -> str:
    """Prefer server-side upstream auth, otherwise forward the caller token."""
    return settings.upstream_api_key or _extract_bearer_token(auth_header)


def _extract_prompt(messages: list[dict]) -> str:
    """Extract the last user message from an OpenAI-compatible messages list."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _extract_assistant_content(response: dict) -> str:
    """Extract assistant text from an OpenAI-compatible response."""
    choices = response.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def _set_assistant_content(response: dict, content: str) -> None:
    """Replace assistant text in an OpenAI-compatible response in place."""
    choices = response.get("choices", [])
    if not choices:
        return
    message = choices[0].setdefault("message", {})
    message["content"] = content


def _prefixed_scores(prefix: str, scores: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}:{name}": score for name, score in scores.items()}


def _prefixed_latencies(prefix: str, latencies_ms: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}:{name}": latency for name, latency in latencies_ms.items()}


def _list_classifier_names(classifier_specs: list[ClassifierSpec]) -> list[str]:
    return [spec.display_name or spec.name for spec in classifier_specs]


def _flatten_classifier_specs(
    classifier_specs_by_language: dict[str, list[ClassifierSpec]],
) -> list[ClassifierSpec]:
    """Flatten routed classifier specs while preserving order and de-duplicating paths."""
    flattened: list[ClassifierSpec] = []
    seen_keys: set[tuple[str, str | None]] = set()
    for specs in classifier_specs_by_language.values():
        for spec in specs:
            spec_key = (spec.name, str(spec.path) if spec.path is not None else None)
            if spec_key in seen_keys:
                continue
            seen_keys.add(spec_key)
            flattened.append(spec)
    return flattened


def _format_input_route_detail(route: InputRouteDecision) -> str:
    """Return a concise string describing one input route decision."""
    return (
        f"Input language router: {route.lang} "
        f"({route.confidence:.4f}, {route.method}) -> {route.target}"
    )


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


def _log_decision(app: FastAPI, entry: dict) -> None:
    """Add a decision entry to the in-memory log."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    entry["id"] = str(uuid.uuid4())[:8]
    app.state.decision_log.insert(0, entry)
    if len(app.state.decision_log) > MAX_LOG_SIZE:
        app.state.decision_log.pop()


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


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


def _build_model_error_response(
    prompt: str,
    stage: str,
    exc: Exception,
    total_latency_ms: float,
) -> dict:
    """Build a structured response when a classifier ensemble cannot be loaded."""
    detail = f"{stage} model error: {exc}"
    return {
        "status_code": 503,
        "payload": {"error": {"message": detail}},
        "prompt": prompt,
        "decision": "ERROR",
        "content": str(exc),
        "scores": {},
        "latencies_ms": {},
        "total_latency_ms": total_latency_ms,
        "detail": detail,
        "failed_filters": [],
    }


def _resolve_input_route_language(app: FastAPI, language: str) -> str:
    """Resolve an input route language to a configured validator bucket."""
    if language in app.state.input_classifier_specs_by_language:
        return language
    if "en" in app.state.input_classifier_specs_by_language:
        return "en"
    return next(iter(app.state.input_classifier_specs_by_language), language)


def _get_input_validator(app: FastAPI, language: str = "en") -> InputValidator:
    validators = getattr(app.state, "input_validators", None)
    if validators is None:
        validators = {}
        app.state.input_validators = validators

    resolved_language = _resolve_input_route_language(app, language)
    validator = validators.get(resolved_language)
    if validator is None:
        validator = InputValidator(
            app.state.input_classifier_specs_by_language[resolved_language]
        )
        validators[resolved_language] = validator
    return validator


def _get_input_route_target_name(app: FastAPI, language: str) -> str:
    """Return a human-readable label for one routed input branch."""
    resolved_language = _resolve_input_route_language(app, language)
    names = _list_classifier_names(
        app.state.input_classifier_specs_by_language[resolved_language]
    )
    return ", ".join(names) if names else "configured input filter"


def _get_output_validator(app: FastAPI) -> OutputValidator:
    validator = getattr(app.state, "output_validator", None)
    if validator is None:
        validator = OutputValidator(app.state.output_classifier_specs)
        app.state.output_validator = validator
    return validator


def _preload_validators(app: FastAPI) -> None:
    """Warm validator ensembles so first-request latency reflects steady state."""
    for language in app.state.input_classifier_specs_by_language:
        try:
            validator = _get_input_validator(app, language)
            validator.warmup(
                INPUT_WARMUP_TEXT_BY_LANGUAGE.get(language, INPUT_WARMUP_TEXT_BY_LANGUAGE["en"])
            )
        except Exception as exc:
            logger.warning("Failed to preload input validators for %s: %s", language, exc)

    if app.state.settings.enable_output_classifiers:
        try:
            _get_output_validator(app).warmup(OUTPUT_WARMUP_TEXT)
        except Exception as exc:
            logger.warning("Failed to preload output validators: %s", exc)


async def _process_chat_completion(
    app: FastAPI,
    body: dict,
    auth_header: str = "",
) -> dict:
    """Run a single request through input checks, upstream, and output checks."""
    settings = app.state.settings
    request_started_at = time.perf_counter()
    prompt = _extract_prompt(body.get("messages", []))

    if not prompt:
        return {
            "status_code": 400,
            "payload": {"error": {"message": "No user message found in request."}},
            "prompt": "",
            "decision": "ERROR",
            "content": "",
            "scores": {},
            "latencies_ms": {},
            "total_latency_ms": _elapsed_ms(request_started_at),
            "detail": "No user message found in request.",
            "failed_filters": [],
        }

    route_decision = route_input_text(
        prompt,
        english_filter_name=_get_input_route_target_name(app, "en"),
        spanish_filter_name=_get_input_route_target_name(app, "es"),
    )
    route_detail = _format_input_route_detail(route_decision)
    route_language = _resolve_input_route_language(app, route_decision.lang)

    try:
        input_validator = _get_input_validator(app, route_language)
    except Exception as exc:
        logger.error("Failed to load input validators: %s", exc)
        _log_decision(
            app,
            {
                "type": "INPUT_MODEL_ERROR",
                "prompt": prompt,
                "response": str(exc),
                "decision": "ERROR",
                "scores": {"input:Language Router": route_decision.confidence},
                "latencies_ms": {"input:Language Router": route_decision.latency_ms},
                "total_latency_ms": _elapsed_ms(request_started_at),
                "detail": f"Input model error: {exc} | {route_detail}",
            },
        )
        result = _build_model_error_response(
            prompt,
            "Input",
            exc,
            _elapsed_ms(request_started_at),
        )
        result["scores"] = {"input:Language Router": route_decision.confidence}
        result["latencies_ms"] = {"input:Language Router": route_decision.latency_ms}
        result["detail"] = f"{result['detail']} | {route_detail}"
        return result

    output_validator = None
    if settings.enable_output_classifiers:
        try:
            output_validator = _get_output_validator(app)
        except Exception as exc:
            logger.error("Failed to load output validators: %s", exc)
            _log_decision(
                app,
                {
                    "type": "OUTPUT_MODEL_ERROR",
                    "prompt": prompt,
                    "response": str(exc),
                    "decision": "ERROR",
                    "scores": {},
                    "latencies_ms": {},
                    "total_latency_ms": _elapsed_ms(request_started_at),
                    "detail": f"Output model error: {exc}",
                },
            )
            return _build_model_error_response(
                prompt,
                "Output",
                exc,
                _elapsed_ms(request_started_at),
            )

    input_result = input_validator.validate(prompt)
    input_scores = _prefixed_scores("input", input_result.scores_summary)
    input_latencies = _prefixed_latencies("input", input_result.latencies_summary)
    input_scores["input:Language Router"] = route_decision.confidence
    input_latencies["input:Language Router"] = route_decision.latency_ms

    if not input_result.passed:
        blocked_by = [result.filter_name for result in input_result.failed_filters]
        logger.warning("Blocked prompt with input classifiers: %s", ", ".join(blocked_by))
        detail = f"Blocked by input classifiers: {', '.join(blocked_by)} | {route_detail}"
        refusal_response = _build_openai_response(settings.refusal_message)
        total_latency_ms = _elapsed_ms(request_started_at)
        _log_decision(
            app,
            {
                "type": "INPUT_BLOCKED",
                "prompt": prompt,
                "response": settings.refusal_message,
                "decision": "BLOCKED",
                "scores": input_scores,
                "latencies_ms": input_latencies,
                "total_latency_ms": total_latency_ms,
                "detail": detail,
                "failed_filters": blocked_by,
            },
        )
        return {
            "status_code": 200,
            "payload": refusal_response,
            "prompt": prompt,
            "decision": "BLOCKED",
            "content": settings.refusal_message,
            "scores": input_scores,
            "latencies_ms": input_latencies,
            "total_latency_ms": total_latency_ms,
            "detail": detail,
            "failed_filters": blocked_by,
        }

    scores = dict(input_scores)
    latencies_ms = dict(input_latencies)
    api_key = _resolve_upstream_api_key(settings, auth_header)
    try:
        upstream_response = await forward_to_llm(
            request_body=body,
            llm_api_url=settings.upstream_chat_completions_url,
            api_key=api_key,
        )
    except Exception as exc:
        logger.error("Upstream LLM error: %s", exc)
        detail = f"Upstream error: {exc} | {route_detail}"
        total_latency_ms = _elapsed_ms(request_started_at)
        _log_decision(
            app,
            {
                "type": "UPSTREAM_ERROR",
                "prompt": prompt,
                "response": str(exc),
                "decision": "ERROR",
                "scores": scores,
                "latencies_ms": latencies_ms,
                "total_latency_ms": total_latency_ms,
                "detail": detail,
            },
        )
        return {
            "status_code": 502,
            "payload": {"error": {"message": f"Upstream LLM error: {exc}"}},
            "prompt": prompt,
            "decision": "ERROR",
            "content": str(exc),
            "scores": scores,
            "latencies_ms": latencies_ms,
            "total_latency_ms": total_latency_ms,
            "detail": detail,
            "failed_filters": [],
        }

    assistant_content = _extract_assistant_content(upstream_response)
    pii_result = mask_pii(assistant_content)
    assistant_content = pii_result.text
    if pii_result.masked:
        _set_assistant_content(upstream_response, assistant_content)

    output_result = None

    pii_detail = ""
    if pii_result.masked:
        pii_detail = f"PII masked: {', '.join(pii_result.masked_entities)}"

    if output_validator is not None:
        output_result = await output_validator.validate(assistant_content)
        output_scores = _prefixed_scores("output", output_result.scores_summary)
        output_latencies = _prefixed_latencies("output", output_result.latencies_summary)
        scores.update(output_scores)
        latencies_ms.update(output_latencies)

        if not output_result.passed:
            blocked_by = [result.filter_name for result in output_result.failed_filters]
            logger.warning("Blocked response with output classifiers: %s", ", ".join(blocked_by))
            detail = f"Blocked by output classifiers: {', '.join(blocked_by)}"
            detail = f"{detail} | {route_detail}"
            if pii_detail:
                detail = f"{detail} | {pii_detail}"
            refusal_response = _build_openai_response(settings.refusal_message)
            total_latency_ms = _elapsed_ms(request_started_at)
            _log_decision(
                app,
                {
                    "type": "OUTPUT_BLOCKED",
                    "prompt": prompt,
                    "response": assistant_content or settings.refusal_message,
                    "decision": "DROPPED",
                    "scores": scores,
                    "latencies_ms": latencies_ms,
                    "total_latency_ms": total_latency_ms,
                    "detail": detail,
                    "failed_filters": blocked_by,
                },
            )
            return {
                "status_code": 200,
                "payload": refusal_response,
                "prompt": prompt,
                "decision": "DROPPED",
                "content": settings.refusal_message,
                "scores": scores,
                "latencies_ms": latencies_ms,
                "total_latency_ms": total_latency_ms,
                "detail": detail,
                "failed_filters": blocked_by,
            }

    detail = (
        "Output classifiers disabled; upstream response returned without output validation"
        if not settings.enable_output_classifiers
        else "All classifiers passed"
    )
    detail = f"{detail} | {route_detail}"
    if pii_detail:
        detail = f"{detail} | {pii_detail}"
    total_latency_ms = _elapsed_ms(request_started_at)
    _log_decision(
        app,
        {
            "type": "PASSED",
            "prompt": prompt,
            "response": assistant_content or "(empty response)",
            "decision": "ALLOWED",
            "scores": scores,
            "latencies_ms": latencies_ms,
            "total_latency_ms": total_latency_ms,
            "detail": detail,
        },
    )
    return {
        "status_code": 200,
        "payload": upstream_response,
        "prompt": prompt,
        "decision": "ALLOWED",
        "content": assistant_content,
        "scores": scores,
        "latencies_ms": latencies_ms,
        "total_latency_ms": total_latency_ms,
        "detail": detail,
        "failed_filters": [],
    }


def create_app(
    settings: Settings | None = None,
    input_classifier_specs: list[ClassifierSpec] | None = None,
    input_classifier_specs_by_language: dict[str, list[ClassifierSpec]] | None = None,
    output_classifier_specs: list[ClassifierSpec] | None = None,
) -> FastAPI:
    """Create the FastAPI app with lazy-loaded validators."""
    app = FastAPI(
        title="PromptShield",
        description="Routes prompts through input/output classifier ensembles.",
        version="0.2.0",
    )

    app.state.settings = settings or Settings()
    app.state.decision_log = []
    routed_input_specs = (
        {
            language: list(specs)
            for language, specs in input_classifier_specs_by_language.items()
        }
        if input_classifier_specs_by_language is not None
        else (
            {"en": list(input_classifier_specs)}
            if input_classifier_specs is not None
            else get_input_classifier_specs_by_language()
        )
    )
    app.state.input_validators = {}
    app.state.output_validator = None
    app.state.input_classifier_specs_by_language = routed_input_specs
    app.state.input_classifier_specs = _flatten_classifier_specs(routed_input_specs)
    app.state.output_classifier_specs = list(
        output_classifier_specs or get_output_classifier_specs()
    )
    _preload_validators(app)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """
        OpenAI-compatible chat completions endpoint.

        Flow:
        1. Route the prompt to the matching input classifier.
        2. If all pass, forward to the configured upstream LLM URL.
        3. Run the assistant response through every output classifier.
        4. Return the response, or a refusal if any classifier blocks.
        """
        body = await request.json()
        auth_header = request.headers.get("Authorization", "")
        result = await _process_chat_completion(request.app, body, auth_header)
        return JSONResponse(
            status_code=result["status_code"],
            content=result["payload"],
        )

    @app.get("/v1/models")
    async def models(request: Request):
        """OpenAI-compatible model listing endpoint for SDK compatibility."""
        api_key = _resolve_upstream_api_key(
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

    @app.get("/v1/models/{model_id}")
    async def model_retrieve(model_id: str, request: Request):
        """OpenAI-compatible model retrieval endpoint for SDK compatibility."""
        api_key = _resolve_upstream_api_key(
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

    @app.post("/v1/chat/completions/batch")
    async def chat_completions_batch(request: Request):
        """
        Batch testing endpoint for sending up to 1000 prompts in one request.

        Each prompt is wrapped into its own OpenAI-style request body and routed
        through the same firewall path as `/v1/chat/completions`.
        """
        body = await request.json()
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

        auth_header = request.headers.get("Authorization", "")
        semaphore = asyncio.Semaphore(min(concurrency, len(prompts)))

        async def _process_batch_item(index: int, prompt: str) -> dict:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            async with semaphore:
                result = await _process_chat_completion(
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

    @app.get("/api/logs")
    async def get_logs(limit: int = 50):
        """Return the most recent decision log entries."""
        return app.state.decision_log[:limit]

    @app.get("/api/stats")
    async def get_stats():
        """Return aggregate stats for the dashboard."""
        log = app.state.decision_log
        return {
            "total": len(log),
            "blocked": sum(1 for entry in log if entry["decision"] == "BLOCKED"),
            "dropped": sum(1 for entry in log if entry["decision"] == "DROPPED"),
            "allowed": sum(1 for entry in log if entry["decision"] == "ALLOWED"),
            "errors": sum(1 for entry in log if entry["decision"] == "ERROR"),
            "average_total_latency_ms": _compute_average_total_latency_ms(log),
        }

    @app.get("/api/config")
    async def get_config():
        """Expose dashboard-safe runtime configuration."""
        return {
            "upstream_chat_completions_url": app.state.settings.upstream_chat_completions_url,
            "default_model_id": app.state.settings.default_model_id,
            "input_models": _list_classifier_names(app.state.input_classifier_specs),
            "output_models": _list_classifier_names(app.state.output_classifier_specs),
            "enable_output_classifiers": app.state.settings.enable_output_classifiers,
            "refusal_message": app.state.settings.refusal_message,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve the monitoring dashboard."""
        dashboard_path = os.path.join(
            os.path.dirname(__file__), "..", "dashboard", "index.html"
        )
        with open(dashboard_path, "r", encoding="utf-8") as handle:
            return HTMLResponse(content=handle.read())

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "promptshield"}

    return app


app = create_app()
