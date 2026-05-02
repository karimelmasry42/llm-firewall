"""Tests for conversational awareness (cumulative-score per-conversation gate)."""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from llm_firewall.api import conversations as conv_state


def _fake_app(
    threshold: float = 0.01,
    max_tracked: int = 1000,
    window_size: int = 30,
):
    return SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(
                conversation_cumulative_threshold=threshold,
                conversation_max_tracked=max_tracked,
                conversation_window_size=window_size,
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
    app = _fake_app(threshold=0.6, window_size=30)
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


def test_cumulative_score_uses_sliding_window_not_all_time():
    """A small window must drop old scores out of the cumulative once
    enough new turns have arrived. Critical so benign long conversations
    don't trip the gate just because they're long."""
    app = _fake_app(threshold=10.0, window_size=3)
    conv = conv_state.get_or_create(app, "window-test")
    # Five turns with score=1 each. Window=3 → cumulative should reflect
    # only the last 3 turns at all times.
    for _ in range(5):
        conv_state.record_turn(app, conv, prompt="x", score=1.0, decision="ALLOWED")
    assert len(conv.turns) == 5
    # Last 3 turns sum to 3.0, NOT 5.0 (which would be all-time sum).
    assert abs(conv.cumulative_score - 3.0) < 1e-9
    # cumulative_score_after on the latest turn matches the windowed total.
    assert abs(conv.turns[-1].cumulative_score_after - 3.0) < 1e-9


def test_blocked_flag_is_sticky_after_window_recovery():
    """Once the cumulative trips the threshold the conversation must stay
    locked even if the windowed sum later drops back below threshold —
    attackers cannot dilute their way out by appending benign turns."""
    app = _fake_app(threshold=1.5, window_size=2)
    conv = conv_state.get_or_create(app, "sticky-test")
    # Two suspicious turns trip the gate (window=2 → sum=2.0).
    conv_state.record_turn(app, conv, prompt="x", score=1.0, decision="ALLOWED")
    conv_state.record_turn(app, conv, prompt="x", score=1.0, decision="ALLOWED")
    assert conv.blocked is True
    blocked_reason = conv.blocked_reason
    # Two benign turns drop the windowed sum back to 0.0.
    conv_state.record_turn(app, conv, prompt="x", score=0.0, decision="ALLOWED")
    conv_state.record_turn(app, conv, prompt="x", score=0.0, decision="ALLOWED")
    assert conv.cumulative_score == 0.0
    # But blocked stays True. Reason string is preserved (not overwritten).
    assert conv.blocked is True
    assert conv.blocked_reason == blocked_reason


def test_truncated_prompt_is_marked_in_full_view():
    """`to_full()` must surface the truncation marker so dashboards don't
    silently render a cut-off payload as if it were the whole prompt."""
    app = _fake_app(threshold=10.0, window_size=30)
    conv = conv_state.get_or_create(app, "trunc-test")
    long_prompt = "X" * 1200
    conv_state.record_turn(app, conv, prompt=long_prompt, score=0.0, decision="ALLOWED")

    turn = conv.to_full()["turns"][0]
    assert turn["prompt_truncated"] is True
    assert turn["prompt_original_length"] == 1200
    # Stored payload has the elision marker so a reader can tell it's not
    # the original.
    assert turn["prompt"].endswith("…")
    assert len(turn["prompt"]) == conv_state.PROMPT_STORAGE_LIMIT + 1  # 500 + ellipsis


def test_short_prompt_is_not_marked_as_truncated():
    """Round-trip case: short prompts come out untouched and unmarked."""
    app = _fake_app()
    conv = conv_state.get_or_create(app, "short-test")
    conv_state.record_turn(app, conv, prompt="hi there", score=0.0, decision="ALLOWED")
    turn = conv.to_full()["turns"][0]
    assert turn["prompt"] == "hi there"
    assert turn["prompt_truncated"] is False
    assert turn["prompt_original_length"] == 8


def test_conversation_has_async_lock():
    """The per-conversation lock is what serializes the predict→record
    critical section; verify it's a real asyncio.Lock and is released
    when the `async with` block exits cleanly."""
    import asyncio

    app = _fake_app()
    conv = conv_state.get_or_create(app, "lock-test")
    assert isinstance(conv.lock, asyncio.Lock)

    async def acquire_release():
        async with conv.lock:
            assert conv.lock.locked()
        assert not conv.lock.locked()

    asyncio.run(acquire_release())


def test_predict_windowed_cumulative_simulates_append():
    """The predictor returns the windowed sum as-if the new score were
    appended, without mutating any state."""
    app = _fake_app(threshold=10.0, window_size=3)
    conv = conv_state.get_or_create(app, "predict-test")
    conv_state.record_turn(app, conv, prompt="x", score=1.0, decision="ALLOWED")
    conv_state.record_turn(app, conv, prompt="x", score=1.0, decision="ALLOWED")
    conv_state.record_turn(app, conv, prompt="x", score=1.0, decision="ALLOWED")
    # Window=3, current cumulative=3.0. Predicting score=2.0 should drop
    # the oldest turn (1.0) and add 2.0 → predicted=4.0.
    predicted = conv_state.predict_windowed_cumulative(app, conv, 2.0)
    assert abs(predicted - 4.0) < 1e-9
    # The conversation state didn't change.
    assert len(conv.turns) == 3
    assert abs(conv.cumulative_score - 3.0) < 1e-9


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


def test_get_returns_existing_without_creating():
    app = _fake_app()
    # Empty store: get() must not allocate.
    assert conv_state.get(app, "missing") is None
    store = getattr(app.state, "conversations", None)
    assert store is None or "missing" not in store
    # Existing entry round-trips identity.
    conv = conv_state.get_or_create(app, "exists")
    assert conv_state.get(app, "exists") is conv


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
