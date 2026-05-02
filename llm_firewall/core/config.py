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
    # scores across every prompt in one conversation; if the cumulative
    # score exceeds this threshold the conversation is marked blocked and
    # all future prompts in it are refused (until the caller starts a new
    # conversation). Catches multi-turn social engineering where each
    # prompt is borderline but the trajectory is adversarial.
    # Must be > 0 — a non-positive threshold blocks the very first turn.
    conversation_cumulative_threshold: float = Field(default=1.5, gt=0.0)
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
