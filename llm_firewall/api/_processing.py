"""
Internal request-processing helpers shared by the chat and batch routes.

Keeps the heavy single-prompt firewall flow (input checks → upstream call →
PII mask → output checks → decision log) out of the routing modules.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI

from llm_firewall.classifiers.language_router import (
    InputRouteDecision,
    route_input_text,
)
from llm_firewall.classifiers.registry import ClassifierSpec
from llm_firewall.core.config import Settings
from llm_firewall.core.proxy import forward_to_llm
from llm_firewall.filters.pii import mask
from llm_firewall.api import conversations as conv_state
from llm_firewall.api.events import get_broadcaster
from llm_firewall.validators.input import InputValidator
from llm_firewall.validators.output import OutputValidator

logger = logging.getLogger("llm_firewall")

MAX_LOG_SIZE = 500
INPUT_WARMUP_TEXT_BY_LANGUAGE = {
    "en": "What is the capital of France?",
    "es": "Hola, necesito ayuda con mi cuenta.",
}
OUTPUT_WARMUP_TEXT = "This is a routine warmup response."


def build_openai_response(content: str, model: str = "firewall") -> dict:
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


def extract_bearer_token(auth_header: str) -> str:
    """Extract a bearer token value from an Authorization header."""
    if not auth_header:
        return ""
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return auth_header.strip()


def resolve_upstream_api_key(settings: Settings, auth_header: str = "") -> str:
    """Prefer server-side upstream auth, otherwise forward the caller token."""
    return settings.upstream_api_key or extract_bearer_token(auth_header)


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


def list_classifier_names(classifier_specs: list[ClassifierSpec]) -> list[str]:
    return [spec.display_name or spec.name for spec in classifier_specs]


def flatten_classifier_specs(
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


def log_decision(app: FastAPI, entry: dict) -> None:
    """Add a decision entry to the in-memory log and broadcast it.

    Subscribers (the dashboard's `/api/stream` SSE clients) receive the
    fully-stamped entry as a `decision` event — no polling needed.
    Publishing failures are best-effort; the in-memory log is the source
    of truth and a refreshed client can always re-snapshot it.
    """
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    entry["id"] = str(uuid.uuid4())[:8]
    app.state.decision_log.insert(0, entry)
    if len(app.state.decision_log) > MAX_LOG_SIZE:
        app.state.decision_log.pop()
    # Pass authoritative aggregate stats with every event so SSE clients
    # don't have to maintain a running tally locally (which drifts once
    # the bounded decision_log starts evicting old entries).
    from llm_firewall.api.dashboard import compute_stats  # local to avoid cycle
    get_broadcaster(app).publish(
        {
            "type": "decision",
            "entry": entry,
            "stats": compute_stats(app.state.decision_log),
        }
    )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _primary_input_score(input_result) -> float:
    """Pick the per-prompt P(injection) score we feed into the conversation gate.

    The input ensemble may run multiple classifiers; we take the maximum of
    their `confidence` values (each is P(injection) per the contract in
    HFSequenceClassifier and PickleClassifier). The language router doesn't
    contribute — it isn't a security signal.
    """
    if not input_result.results:
        return 0.0
    return max(float(r.confidence) for r in input_result.results)


def _attach_conversation(payload: dict, conversation) -> None:
    """Inject conversation_id + a small summary into a chat-completion payload.

    OpenAI clients tolerate unknown top-level fields, so this is the
    cheapest way to give back the running conversation state without
    forcing callers onto a custom envelope. The dashboard reads these
    fields directly.
    """
    if not isinstance(payload, dict):
        return
    payload["conversation_id"] = conversation.id
    payload["conversation"] = conversation.to_summary()


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


def get_input_validator(app: FastAPI, language: str = "en") -> InputValidator:
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
    names = list_classifier_names(
        app.state.input_classifier_specs_by_language[resolved_language]
    )
    return ", ".join(names) if names else "configured input filter"


def get_output_validator(app: FastAPI) -> OutputValidator:
    validator = getattr(app.state, "output_validator", None)
    if validator is None:
        validator = OutputValidator(app.state.output_classifier_specs)
        app.state.output_validator = validator
    return validator


def preload_validators(app: FastAPI) -> None:
    """Warm validator ensembles so first-request latency reflects steady state."""
    for language in app.state.input_classifier_specs_by_language:
        try:
            validator = get_input_validator(app, language)
            validator.warmup(
                INPUT_WARMUP_TEXT_BY_LANGUAGE.get(
                    language, INPUT_WARMUP_TEXT_BY_LANGUAGE["en"]
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to preload input validators for %s: %s", language, exc
            )

    if app.state.settings.enable_output_classifiers:
        try:
            get_output_validator(app).warmup(OUTPUT_WARMUP_TEXT)
        except Exception as exc:
            logger.warning("Failed to preload output validators: %s", exc)


async def process_chat_completion(
    app: FastAPI,
    body: dict,
    auth_header: str = "",
) -> dict:
    """Run a single request through input checks, upstream, and output checks."""
    settings = app.state.settings
    request_started_at = time.perf_counter()
    prompt = _extract_prompt(body.get("messages", []))

    # Refuse malformed requests before allocating any conversation state.
    # Otherwise repeated empty-message POSTs would churn the LRU store.
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

    conversation = conv_state.get_or_create(
        app, conv_state.extract_conversation_id(body)
    )

    # If this conversation already tripped the cumulative gate, refuse
    # immediately — no need to score the new prompt or hit the upstream.
    if conv_state.is_blocked_by_cumulative(conversation):
        refusal_response = build_openai_response(settings.refusal_message)
        refusal_response["conversation_id"] = conversation.id
        refusal_response["conversation"] = conversation.to_summary()
        detail = (
            f"Conversation {conversation.id} previously blocked: "
            f"{conversation.blocked_reason}"
        )
        log_decision(
            app,
            {
                "type": "CONVERSATION_BLOCKED",
                "prompt": prompt,
                "response": settings.refusal_message,
                "decision": "BLOCKED",
                "scores": {},
                "latencies_ms": {},
                "total_latency_ms": _elapsed_ms(request_started_at),
                "detail": detail,
                "conversation_id": conversation.id,
                "cumulative_score": conversation.cumulative_score,
            },
        )
        return {
            "status_code": 200,
            "payload": refusal_response,
            "prompt": prompt,
            "decision": "BLOCKED",
            "content": settings.refusal_message,
            "scores": {},
            "latencies_ms": {},
            "total_latency_ms": _elapsed_ms(request_started_at),
            "detail": detail,
            "failed_filters": ["conversation_cumulative"],
            "conversation_id": conversation.id,
        }

    route_decision = route_input_text(
        prompt,
        english_filter_name=_get_input_route_target_name(app, "en"),
        spanish_filter_name=_get_input_route_target_name(app, "es"),
    )
    route_detail = _format_input_route_detail(route_decision)
    route_language = _resolve_input_route_language(app, route_decision.lang)

    try:
        input_validator = get_input_validator(app, route_language)
    except Exception as exc:
        logger.error("Failed to load input validators: %s", exc)
        log_decision(
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
            output_validator = get_output_validator(app)
        except Exception as exc:
            logger.error("Failed to load output validators: %s", exc)
            log_decision(
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

    # Primary per-prompt score: max P(injection) across the input classifiers
    # (excluding the language router which isn't a security signal). This
    # is what feeds the conversation-level cumulative gate.
    primary_score = _primary_input_score(input_result)

    if not input_result.passed:
        blocked_by = [result.filter_name for result in input_result.failed_filters]
        logger.warning(
            "Blocked prompt with input classifiers: %s", ", ".join(blocked_by)
        )
        detail = (
            f"Blocked by input classifiers: {', '.join(blocked_by)} | {route_detail}"
        )
        refusal_response = build_openai_response(settings.refusal_message)
        total_latency_ms = _elapsed_ms(request_started_at)
        conv_state.record_turn(
            app, conversation, prompt=prompt, score=primary_score, decision="BLOCKED"
        )
        _attach_conversation(refusal_response, conversation)
        log_decision(
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
                "conversation_id": conversation.id,
                "cumulative_score": conversation.cumulative_score,
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
            "conversation_id": conversation.id,
        }

    # The per-prompt classifier passed, but adding this prompt's score might
    # tip the conversation's running total over the cumulative threshold.
    # Probe by tentatively adding the score and checking, without mutating
    # state until we know the final decision.
    cum_threshold = float(settings.conversation_cumulative_threshold)
    if (conversation.cumulative_score + primary_score) >= cum_threshold:
        detail = (
            f"Blocked by conversation cumulative score "
            f"({conversation.cumulative_score + primary_score:.4f} ≥ "
            f"{cum_threshold:.4f}) after {len(conversation.turns) + 1} turn(s) | "
            f"{route_detail}"
        )
        logger.warning("Blocked prompt by conversation gate: %s", detail)
        refusal_response = build_openai_response(settings.refusal_message)
        total_latency_ms = _elapsed_ms(request_started_at)
        conv_state.record_turn(
            app, conversation, prompt=prompt, score=primary_score, decision="BLOCKED"
        )
        _attach_conversation(refusal_response, conversation)
        log_decision(
            app,
            {
                "type": "CONVERSATION_BLOCKED",
                "prompt": prompt,
                "response": settings.refusal_message,
                "decision": "BLOCKED",
                "scores": input_scores,
                "latencies_ms": input_latencies,
                "total_latency_ms": total_latency_ms,
                "detail": detail,
                "failed_filters": ["conversation_cumulative"],
                "conversation_id": conversation.id,
                "cumulative_score": conversation.cumulative_score,
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
            "failed_filters": ["conversation_cumulative"],
            "conversation_id": conversation.id,
        }

    scores = dict(input_scores)
    latencies_ms = dict(input_latencies)
    api_key = resolve_upstream_api_key(settings, auth_header)
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
        # Don't penalize the conversation for an upstream failure — record the
        # turn with score=0 so cumulative isn't poisoned by infrastructure noise.
        conv_state.record_turn(
            app, conversation, prompt=prompt, score=0.0, decision="ERROR"
        )
        log_decision(
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
                "conversation_id": conversation.id,
                "cumulative_score": conversation.cumulative_score,
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
            "conversation_id": conversation.id,
        }

    assistant_content = _extract_assistant_content(upstream_response)
    pii_result = mask(assistant_content)
    assistant_content = pii_result.text
    if pii_result.masked:
        _set_assistant_content(upstream_response, assistant_content)

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
            logger.warning(
                "Blocked response with output classifiers: %s", ", ".join(blocked_by)
            )
            detail = (
                f"Blocked by output classifiers: {', '.join(blocked_by)} | {route_detail}"
            )
            if pii_detail:
                detail = f"{detail} | {pii_detail}"
            refusal_response = build_openai_response(settings.refusal_message)
            total_latency_ms = _elapsed_ms(request_started_at)
            # Output was blocked, but the input passed cleanly. Record the
            # turn against the input score (which the user actually sent),
            # not against the model's response.
            conv_state.record_turn(
                app, conversation, prompt=prompt, score=primary_score, decision="DROPPED"
            )
            _attach_conversation(refusal_response, conversation)
            log_decision(
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
                    "conversation_id": conversation.id,
                    "cumulative_score": conversation.cumulative_score,
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
                "conversation_id": conversation.id,
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
    conv_state.record_turn(
        app, conversation, prompt=prompt, score=primary_score, decision="ALLOWED"
    )
    _attach_conversation(upstream_response, conversation)
    log_decision(
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
            "conversation_id": conversation.id,
            "cumulative_score": conversation.cumulative_score,
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
        "conversation_id": conversation.id,
    }
