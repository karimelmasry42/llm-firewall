"""Tests for the local dummy upstream LLM."""
from pathlib import Path

from fastapi.testclient import TestClient

from llm_firewall.api.dummy_llm import DummyLLMSettings, create_dummy_llm_app


class TestDummyLLM:
    """Test the dummy upstream LLM app."""

    def test_returns_fixed_response_without_api_key(self):
        client = TestClient(create_dummy_llm_app())

        response = client.post(
            "/v1/chat/completions",
            json={"model": "firewall-demo", "messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "This is a dummy response."
        assert data["model"] == "dummy-llm"

    def test_rejects_missing_api_key_when_auth_is_enabled(self):
        client = TestClient(
            create_dummy_llm_app(DummyLLMSettings(api_key="local-dummy-key"))
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "firewall-demo", "messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "Invalid or missing API key."

    def test_accepts_matching_api_key_when_auth_is_enabled(self):
        client = TestClient(
            create_dummy_llm_app(DummyLLMSettings(api_key="local-dummy-key"))
        )

        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-dummy-key"},
            json={"model": "firewall-demo", "messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "This is a dummy response."

    def test_uses_configured_response_text(self):
        client = TestClient(
            create_dummy_llm_app(
                DummyLLMSettings(response_text="Your prompt passed the firewall!")
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "firewall-demo", "messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == (
            "Your prompt passed the firewall!"
        )

    def test_health_endpoint(self):
        client = TestClient(create_dummy_llm_app())
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["service"] == "dummy-llm"

    def test_settings_ignore_firewall_keys_in_shared_env_file(self, tmp_path):
        env_file = Path(tmp_path) / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL=https://api.openai.com/v1/chat/completions",
                    "LLM_FIREWALL_UPSTREAM_API_KEY=test-key",
                    "LLM_FIREWALL_INPUT_MODELS_DIR=./models/input",
                    "LLM_FIREWALL_OUTPUT_MODELS_DIR=./models/output",
                    "LLM_FIREWALL_REFUSAL_MESSAGE=Sorry, I cannot answer this prompt",
                    "DUMMY_LLM_API_KEY=local-dummy-key",
                    "DUMMY_LLM_RESPONSE_TEXT=Custom dummy text",
                ]
            ),
            encoding="utf-8",
        )

        settings = DummyLLMSettings(_env_file=env_file)
        assert settings.api_key == "local-dummy-key"
        assert settings.response_text == "Custom dummy text"
