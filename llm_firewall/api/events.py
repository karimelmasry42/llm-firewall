"""
In-process pub/sub for live decision-log updates.

The dashboard previously polled `/api/logs` and `/api/stats` every 3 seconds,
which spammed the uvicorn access log and caused the UI to repaint constantly.
This module replaces that with a simple Server-Sent Events broadcaster:
each dashboard tab opens one long-lived `GET /api/stream` connection, the
firewall pushes a `decision` event when (and only when) a new entry is
appended to the in-memory decision log, and a low-rate heartbeat keeps the
TCP connection alive through proxies.

State lives on `app.state.event_broadcaster` (lazy-init in `subscribe()`).
Subscribers are bounded `asyncio.Queue` instances; if a slow subscriber
fills its queue, individual events are dropped silently rather than blocking
the publisher. The decision log itself is the source of truth — a refreshed
client can always pull a snapshot via the existing `/api/logs` endpoint.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Per-subscriber queue depth. Small on purpose — if a tab can't keep up with
# 32 pending events we'd rather drop than buffer megabytes per stale client.
_QUEUE_MAX = 32


class DecisionBroadcaster:
    """Fan-out pub/sub for decision-log events."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Return a fresh queue scoped to one subscriber."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber. Safe to call from a finally block."""
        self._subscribers.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        """Best-effort fan-out; drop events for subscribers that can't keep up."""
        # Snapshot the set so unsubscribe() during iteration is safe.
        for q in tuple(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("dropping event for slow subscriber (queue full)")


def get_broadcaster(app) -> DecisionBroadcaster:
    """Lazy-init the broadcaster on `app.state` so existing tests don't need fixturing."""
    bc = getattr(app.state, "event_broadcaster", None)
    if bc is None:
        bc = DecisionBroadcaster()
        app.state.event_broadcaster = bc
    return bc
