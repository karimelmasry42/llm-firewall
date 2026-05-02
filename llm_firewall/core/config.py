"""
Configuration for the PromptShield proxy.
"""
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment or defaults."""

    upstream_chat_completions_url: str = "https://api.openai.com/v1/chat/completions"
    upstream_api_key: str = ""
    default_model_id: str = "firewall-demo"
    enable_output_classifiers: bool = True

    refusal_message: str = "Sorry, I cannot answer this prompt"

    # Conversational awareness. The firewall sums per-prompt P(injection)
    # scores over the most recent N turns of a conversation (windowed sum,
    # not all-time sum — see conversations.py). When the windowed total
    # crosses this threshold the conversation is marked blocked and every
    # subsequent prompt is refused until the caller starts a new
    # conversation. Catches multi-turn social engineering where each
    # individual prompt is borderline but the recent pattern is adversarial.
    #
    # Calibration with default per-prompt threshold of 0.001: a windowed
    # threshold of 0.01 trips when the recent window contains roughly 10
    # prompts at the per-prompt threshold (or fewer + higher-scoring ones).
    # Must be > 0 — a non-positive threshold blocks the very first turn.
    conversation_cumulative_threshold: float = Field(default=0.01, gt=0.0)
    # Number of most-recent turns the cumulative sum reaches back over. A
    # bounded window stops benign long conversations from spuriously trip-
    # ping the gate purely because they're long. Must be >= 1.
    conversation_window_size: int = Field(default=30, ge=1)
    # Soft cap on tracked conversations to bound memory in long-running
    # processes. Oldest-touched conversations are evicted when exceeded.
    # Must be >= 1 — 0 would evict every new conversation immediately, and
    # negative values would pop until the store is empty and then raise.
    conversation_max_tracked: int = Field(default=1000, ge=1)

    model_config = {
        "env_prefix": "LLM_FIREWALL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
