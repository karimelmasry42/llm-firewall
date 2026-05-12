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

from llm_firewall.classifiers.registry import ClassifierSpec
from llm_firewall.core.config import Settings
from llm_firewall.core.proxy import forward_to_llm
from llm_firewall.filters.pii import mask
from llm_firewall.api import conversations as conv_state
from llm_firewall.api._stats import compute_stats
from llm_firewall.api.events import get_broadcaster
from llm_firewall.validators.input import InputValidator
from llm_firewall.validators.output import OutputValidator

logger = logging.getLogger("llm_firewall")

MAX_LOG_SIZE = 500
# The shipped classifier (Llama-Prompt-Guard-2-86M) is multilingual, so
# warming it on a single English prompt exercises the same code paths as
# any other language would.
INPUT_WARMUP_TEXT = "What is the capital of France?"
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
    # the bounded decision_log starts evicting old entries). `compute_stats`
    # lives in `_stats.py` to avoid the import cycle that arose when
    # `dashboard.py` (which imports from `_processing.py`) owned it.
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
    HFSequenceClassifier and PickleClassifier).
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


def get_input_validator(app: FastAPI) -> InputValidator:
    validator = getattr(app.state, "input_validator", None)
    if validator is None:
        validator = InputValidator(app.state.input_classifier_specs)
        app.state.input_validator = validator
    return validator


def get_output_validator(app: FastAPI) -> OutputValidator:
    validator = getattr(app.state, "output_validator", None)
    if validator is None:
        validator = OutputValidator(app.state.output_classifier_specs)
        app.state.output_validator = validator
    return validator


def preload_validators(app: FastAPI) -> None:
    """Warm validator ensembles so first-request latency reflects steady state."""
    try:
        get_input_validator(app).warmup(INPUT_WARMUP_TEXT)
    except Exception as exc:
        logger.warning("Failed to preload input validator: %s", exc)

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

    try:
        input_validator = get_input_validator(app)
    except Exception as exc:
        logger.error("Failed to load input validators: %s", exc)
        log_decision(
            app,
            {
                "type": "INPUT_MODEL_ERROR",
                "prompt": prompt,
                "response": str(exc),
                "decision": "ERROR",
                "scores": {},
                "latencies_ms": {},
                "total_latency_ms": _elapsed_ms(request_started_at),
                "detail": f"Input model error: {exc}",
            },
        )
        return _build_model_error_response(
            prompt,
            "Input",
            exc,
            _elapsed_ms(request_started_at),
        )

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

    # CRITICAL SECTION: serialize the predict→record window per
    # conversation. Concurrent requests sharing a `conversation_id` queue
    # here so two parallel attackers can't both pass the cumulative gate
    # before either appends its turn. The lock is per-conversation, not
    # global — independent conversations still process in parallel.
    async with conversation.lock:
        # Re-check under the lock — another request for this same
        # conversation may have just tripped the gate while we were
        # waiting for the lock.
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

        input_result = input_validator.validate(prompt)
        input_scores = _prefixed_scores("input", input_result.scores_summary)
        input_latencies = _prefixed_latencies("input", input_result.latencies_summary)

        # Primary per-prompt score: max P(injection) across the input
        # classifiers. This is what feeds the conversation-level cumulative gate.
        primary_score = _primary_input_score(input_result)

        if not input_result.passed:
            blocked_by = [result.filter_name for result in input_result.failed_filters]
            logger.warning(
                "Blocked prompt with input classifiers: %s", ", ".join(blocked_by)
            )
            detail = f"Blocked by input classifiers: {', '.join(blocked_by)}"
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

        # The per-prompt classifier passed, but adding this prompt's score
        # might tip the conversation's *windowed* cumulative over threshold.
        # `predict_windowed_cumulative` simulates the append (so we can refuse
        # before mutating state); `record_turn` below applies the same math
        # for real once we've decided the turn's outcome.
        cum_threshold = float(settings.conversation_cumulative_threshold)
        pending_cumulative = conv_state.predict_windowed_cumulative(
            app, conversation, primary_score
        )
        if pending_cumulative >= cum_threshold:
            # The "last N turns" the predictor actually summed: the new
            # turn plus the most recent (window_size - 1) existing turns,
            # capped by how many turns we have.
            window = int(settings.conversation_window_size)
            summed_turns = min(len(conversation.turns), max(0, window - 1)) + 1
            detail = (
                f"Blocked by conversation cumulative score "
                f"({pending_cumulative:.4f} ≥ {cum_threshold:.4f}) over the "
                f"last {summed_turns} turn(s)"
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
        # Strip firewall-private fields before forwarding. OpenAI silently
        # accepts unknown top-level fields, but stricter OpenAI-compatible
        # endpoints (e.g. Gemini's) 400 on them.
        upstream_body = {
            k: v for k, v in body.items() if k not in ("conversation_id", "firewall")
        }
        try:
            upstream_response = await forward_to_llm(
                request_body=upstream_body,
                llm_api_url=settings.upstream_chat_completions_url,
                api_key=api_key,
            )
        except Exception as exc:
            logger.error("Upstream LLM error: %s", exc)
            detail = f"Upstream error: {exc}"
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
                detail = f"Blocked by output classifiers: {', '.join(blocked_by)}"
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
