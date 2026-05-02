"""Smoke tests for the Server-Sent Events live decision feed."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm_firewall.api.events import DecisionBroadcaster, get_broadcaster


def test_broadcaster_publishes_to_all_subscribers():
    async def run():
        bc = DecisionBroadcaster()
        a = await bc.subscribe()
        b = await bc.subscribe()
        bc.publish({"hello": "world"})
        assert await asyncio.wait_for(a.get(), 0.5) == {"hello": "world"}
        assert await asyncio.wait_for(b.get(), 0.5) == {"hello": "world"}
        assert bc.subscriber_count == 2

    asyncio.run(run())


def test_broadcaster_unsubscribe_stops_delivery():
    async def run():
        bc = DecisionBroadcaster()
        a = await bc.subscribe()
        bc.unsubscribe(a)
        bc.publish({"hello": "world"})
        # Either nothing is delivered (correct) or the queue stays empty.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(a.get(), 0.1)
        assert bc.subscriber_count == 0

    asyncio.run(run())


def test_broadcaster_drops_for_full_subscriber():
    async def run():
        bc = DecisionBroadcaster()
        slow = await bc.subscribe()
        # Saturate the queue (depth is 32).
        for i in range(32):
            bc.publish({"i": i})
        # This one should be dropped silently rather than blocking the publisher.
        bc.publish({"i": "overflow"})
        # The first 32 are still there.
        assert (await slow.get())["i"] == 0

    asyncio.run(run())


def test_get_broadcaster_lazy_init_on_app_state():
    app = SimpleNamespace(state=SimpleNamespace())
    bc1 = get_broadcaster(app)
    bc2 = get_broadcaster(app)
    assert bc1 is bc2
    assert isinstance(bc1, DecisionBroadcaster)


def test_log_decision_publishes_to_subscribers():
    """Integration: writing to the decision log should reach SSE subscribers."""
    async def run():
        from llm_firewall.api._processing import log_decision

        app = SimpleNamespace(state=SimpleNamespace(decision_log=[]))
        bc = get_broadcaster(app)
        sub = await bc.subscribe()

        log_decision(app, {
            "type": "PASSED",
            "prompt": "hi",
            "response": "ok",
            "decision": "ALLOWED",
            "scores": {},
            "latencies_ms": {"input:foo": 12.0},
            "total_latency_ms": 50.0,
            "detail": "",
        })
        event = await asyncio.wait_for(sub.get(), 0.5)
        assert event["type"] == "decision"
        assert event["entry"]["decision"] == "ALLOWED"
        assert "id" in event["entry"]
        assert "timestamp" in event["entry"]

    asyncio.run(run())


def test_decision_event_carries_authoritative_stats():
    """The published decision event must include aggregate stats so SSE
    clients don't have to maintain a local running tally (which drifts
    once the bounded decision_log starts evicting entries)."""
    async def run():
        from llm_firewall.api._processing import log_decision

        app = SimpleNamespace(state=SimpleNamespace(decision_log=[]))
        bc = get_broadcaster(app)
        sub = await bc.subscribe()

        for decision in ("ALLOWED", "BLOCKED", "DROPPED"):
            log_decision(app, {
                "type": "PASSED",
                "prompt": "x",
                "response": "y",
                "decision": decision,
                "scores": {},
                "latencies_ms": {"input:foo": 5.0},
                "total_latency_ms": 10.0,
                "detail": "",
            })

        events = [await asyncio.wait_for(sub.get(), 0.5) for _ in range(3)]
        for ev in events:
            assert "stats" in ev, "every decision event must carry aggregate stats"
            for key in ("total", "blocked", "dropped", "allowed", "errors",
                        "average_classifier_latency_ms"):
                assert key in ev["stats"]

        # The last event reflects the full state.
        final = events[-1]["stats"]
        assert final["total"] == 3
        assert final["allowed"] == 1
        assert final["blocked"] == 1
        assert final["dropped"] == 1
        # Counts must reconcile with total — the bug being prevented.
        assert (final["allowed"] + final["blocked"] + final["dropped"]
                + final["errors"]) == final["total"]

    asyncio.run(run())
