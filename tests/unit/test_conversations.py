"""Tests for conversational awareness (cumulative-score per-conversation gate)."""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from llm_firewall.api import conversations as conv_state


def _fake_app(threshold: float = 1.5, max_tracked: int = 1000):
    return SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(
                conversation_cumulative_threshold=threshold,
                conversation_max_tracked=max_tracked,
            )
        )
    )


def test_get_or_create_generates_id_when_none():
    app = _fake_app()
    conv = conv_state.get_or_create(app, None)
    assert conv.id.startswith("conv_")
    # Same call returns the same conversation object via lookup.
    again = conv_state.get_or_create(app, conv.id)
    assert again is conv


def test_get_or_create_honors_caller_id():
    app = _fake_app()
    conv = conv_state.get_or_create(app, "my-conv-id")
    assert conv.id == "my-conv-id"


def test_record_turn_accumulates_and_blocks():
    app = _fake_app(threshold=0.6)
    conv = conv_state.get_or_create(app, "block-test")
    # Three borderline-injection prompts (each below per-prompt threshold but
    # together cross the cumulative gate at 0.6).
    conv_state.record_turn(app, conv, prompt="hi", score=0.25, decision="ALLOWED")
    assert conv.blocked is False
    conv_state.record_turn(app, conv, prompt="hi", score=0.25, decision="ALLOWED")
    assert conv.blocked is False
    assert abs(conv.cumulative_score - 0.5) < 1e-9
    conv_state.record_turn(app, conv, prompt="hi", score=0.2, decision="ALLOWED")
    assert conv.blocked is True
    assert conv.cumulative_score >= 0.6
    assert "cumulative" in (conv.blocked_reason or "")
    assert len(conv.turns) == 3


def test_reset_drops_conversation():
    app = _fake_app()
    conv = conv_state.get_or_create(app, "reset-me")
    conv_state.record_turn(app, conv, prompt="hi", score=0.5, decision="ALLOWED")
    assert conv_state.reset(app, "reset-me") is True
    assert conv_state.reset(app, "reset-me") is False  # already gone
    new_conv = conv_state.get_or_create(app, "reset-me")
    assert new_conv.cumulative_score == 0.0
    assert new_conv.turns == []


def test_eviction_when_max_tracked_exceeded():
    app = _fake_app(max_tracked=3)
    for i in range(5):
        conv_state.get_or_create(app, f"c{i}")
    store = app.state.conversations
    assert isinstance(store, OrderedDict)
    assert len(store) == 3
    # The two oldest were evicted.
    assert "c0" not in store
    assert "c1" not in store
    assert "c4" in store


def test_extract_conversation_id_top_level_then_nested():
    assert conv_state.extract_conversation_id({"conversation_id": "abc"}) == "abc"
    assert (
        conv_state.extract_conversation_id({"firewall": {"conversation_id": "xyz"}})
        == "xyz"
    )
    assert conv_state.extract_conversation_id({}) is None
    # Empty string is treated as missing.
    assert conv_state.extract_conversation_id({"conversation_id": ""}) is None


def test_summary_and_full_views():
    app = _fake_app()
    conv = conv_state.get_or_create(app, "view-test")
    conv_state.record_turn(app, conv, prompt="hello there", score=0.42, decision="ALLOWED")
    summary = conv.to_summary()
    assert summary["id"] == "view-test"
    assert summary["turn_count"] == 1
    assert summary["cumulative_score"] == 0.42
    full = conv.to_full()
    assert full["turns"][0]["prompt"] == "hello there"
    assert full["turns"][0]["score"] == 0.42
    assert full["turns"][0]["decision"] == "ALLOWED"
