"""
Configuration for the PromptShield proxy.
"""
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
    conversation_cumulative_threshold: float = 1.5
    # Soft cap on tracked conversations to bound memory in long-running
    # processes. Oldest-touched conversations are evicted when exceeded.
    conversation_max_tracked: int = 1000

    model_config = {
        "env_prefix": "LLM_FIREWALL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
