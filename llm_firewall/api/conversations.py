"""
Conversational-awareness state for the firewall.

A *conversation* is a sequence of user prompts the firewall sees that share
a `conversation_id`. Each prompt produces a per-prompt P(injection) score
from the input classifier; this module sums those scores across the whole
conversation and surfaces a single block decision when the total crosses
`Settings.conversation_cumulative_threshold`.

This catches multi-turn social-engineering attacks where every individual
prompt sits below the per-prompt threshold but the trajectory of the
conversation is clearly adversarial — three or four borderline prompts in
a row will trip the cumulative gate even if each would pass alone.

State lives in `app.state.conversations`. The store is in-memory and capped
at `Settings.conversation_max_tracked` entries — oldest-touched
conversations are evicted on overflow. Restarting the server clears it.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI


@dataclass
class ConversationTurn:
    """One user prompt + the firewall's verdict on it."""

    timestamp: float
    prompt: str
    score: float
    decision: str  # ALLOWED / BLOCKED / DROPPED / ERROR
    cumulative_score_after: float


@dataclass
class Conversation:
    """In-memory state for one conversation_id."""

    id: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    cumulative_score: float = 0.0
    blocked: bool = False
    blocked_reason: str | None = None
    turns: list[ConversationTurn] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Lightweight JSON-friendly view (no per-turn prompt text)."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "cumulative_score": round(self.cumulative_score, 6),
            "turn_count": len(self.turns),
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
        }

    def to_full(self) -> dict[str, Any]:
        """Full JSON view including each turn — for inspection / dashboard."""
        return {
            **self.to_summary(),
            "turns": [
                {
                    "timestamp": t.timestamp,
                    "prompt": t.prompt,
                    "score": round(t.score, 6),
                    "decision": t.decision,
                    "cumulative_score_after": round(t.cumulative_score_after, 6),
                }
                for t in self.turns
            ],
        }


def _store(app: FastAPI) -> "OrderedDict[str, Conversation]":
    """Lazy-init the per-app conversation store."""
    store = getattr(app.state, "conversations", None)
    if store is None:
        store = OrderedDict()
        app.state.conversations = store
    return store


def _max_tracked(app: FastAPI) -> int:
    return int(getattr(app.state.settings, "conversation_max_tracked", 1000))


def _evict_if_full(store: "OrderedDict[str, Conversation]", cap: int) -> None:
    while len(store) > cap:
        store.popitem(last=False)


def get_or_create(app: FastAPI, conversation_id: str | None) -> Conversation:
    """Look up a conversation by id, creating one if needed.

    Pass `None` (or empty string) to start a new conversation; the function
    generates a fresh id. Otherwise the caller's id is honored — even if
    we've never seen it, we register it so the second prompt with the same
    id continues the thread.

    Touching a conversation also moves it to the most-recent end of the
    LRU so it survives eviction.
    """
    store = _store(app)
    cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    conv = store.get(cid)
    if conv is None:
        conv = Conversation(id=cid)
        store[cid] = conv
        _evict_if_full(store, _max_tracked(app))
    else:
        store.move_to_end(cid)
    conv.last_used_at = time.time()
    return conv


def get(app: FastAPI, conversation_id: str) -> Conversation | None:
    """Look up a conversation without creating one. Returns None if absent."""
    return _store(app).get(conversation_id)


def reset(app: FastAPI, conversation_id: str) -> bool:
    """Drop a conversation from the store. Returns True if it existed."""
    return _store(app).pop(conversation_id, None) is not None


def list_summaries(app: FastAPI, limit: int = 50) -> list[dict[str, Any]]:
    """Return summaries of the most-recently-used conversations, newest first."""
    store = _store(app)
    items = list(store.values())[-limit:]
    return [c.to_summary() for c in reversed(items)]


def record_turn(
    app: FastAPI,
    conversation: Conversation,
    *,
    prompt: str,
    score: float,
    decision: str,
) -> None:
    """Append a turn and update the conversation's cumulative score."""
    conversation.cumulative_score += float(score)
    threshold = float(
        getattr(app.state.settings, "conversation_cumulative_threshold", 1.5)
    )
    if conversation.cumulative_score >= threshold and not conversation.blocked:
        conversation.blocked = True
        conversation.blocked_reason = (
            f"cumulative score {conversation.cumulative_score:.4f} ≥ threshold "
            f"{threshold:.4f} after {len(conversation.turns) + 1} turn(s)"
        )
    conversation.turns.append(
        ConversationTurn(
            timestamp=time.time(),
            # Truncate long prompts to keep memory bounded; the firewall
            # log already keeps a fuller copy under decision_log.
            prompt=prompt[:500],
            score=float(score),
            decision=decision,
            cumulative_score_after=conversation.cumulative_score,
        )
    )


def is_blocked_by_cumulative(conversation: Conversation) -> bool:
    """Has this conversation tripped the cumulative-score gate?"""
    return conversation.blocked


def extract_conversation_id(body: dict) -> str | None:
    """Pull a conversation_id out of an OpenAI-compatible request body.

    OpenAI's chat-completions schema doesn't define this field, but it
    accepts unknown top-level fields silently — we use that as our
    side-channel. We also accept it nested under a `firewall` namespace
    for clients that prefer to keep custom fields grouped.
    """
    raw = body.get("conversation_id")
    if isinstance(raw, str) and raw:
        return raw
    nested = body.get("firewall")
    if isinstance(nested, dict):
        nested_id = nested.get("conversation_id")
        if isinstance(nested_id, str) and nested_id:
            return nested_id
    return None
