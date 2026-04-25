"""HTTP entry points: the firewall app, the dummy upstream, and routes."""
from llm_firewall.api.app import app, create_app

__all__ = ["app", "create_app"]
