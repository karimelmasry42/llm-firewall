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

    model_config = {
        "env_prefix": "LLM_FIREWALL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
