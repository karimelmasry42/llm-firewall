"""Tests for shared environment configuration loading."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_firewall.core.config import Settings


def test_firewall_settings_ignore_dummy_keys_in_shared_env_file(tmp_path):
    env_file = Path(tmp_path) / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL=http://127.0.0.1:9000/v1/chat/completions",
                "LLM_FIREWALL_UPSTREAM_API_KEY=firewall-key",
                "LLM_FIREWALL_INPUT_MODELS_DIR=./models/input",
                "LLM_FIREWALL_OUTPUT_MODELS_DIR=./models/output",
                "LLM_FIREWALL_REFUSAL_MESSAGE=Sorry, I cannot answer this prompt",
                "DUMMY_LLM_API_KEY=local-dummy-key",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.upstream_api_key == "firewall-key"
    assert settings.upstream_chat_completions_url == "http://127.0.0.1:9000/v1/chat/completions"


def test_conversation_max_tracked_must_be_positive():
    # 0 would evict the conversation we just inserted; negative values
    # would pop until empty and then raise.
    with pytest.raises(ValidationError):
        Settings(conversation_max_tracked=0)
    with pytest.raises(ValidationError):
        Settings(conversation_max_tracked=-3)
    Settings(conversation_max_tracked=1)  # boundary OK


def test_conversation_cumulative_threshold_must_be_positive():
    # 0 (or negative) would block the very first turn before any signal.
    with pytest.raises(ValidationError):
        Settings(conversation_cumulative_threshold=0.0)
    with pytest.raises(ValidationError):
        Settings(conversation_cumulative_threshold=-0.5)
    Settings(conversation_cumulative_threshold=0.0001)  # any positive value OK
