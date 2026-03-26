"""Tests for the proxy module."""
import pytest
import httpx
import respx
from llm_firewall.proxy import (
    _derive_upstream_api_base,
    forward_to_llm,
    list_upstream_models,
    retrieve_upstream_model,
)


class TestProxy:
    """Test request forwarding to upstream."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_forwards_request_to_upstream(self, sample_openai_request, sample_openai_response):
        """Verify the request is forwarded and response is returned."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_openai_response)
        )

        result = await forward_to_llm(
            request_body=sample_openai_request,
            llm_api_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
        )

        assert route.called
        assert result["id"] == "chatcmpl-abc123"
        assert result["choices"][0]["message"]["content"] == "The capital of France is Paris."

    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_authorization_header(self, sample_openai_request, sample_openai_response):
        """Verify the API key is sent as Bearer token."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_openai_response)
        )

        await forward_to_llm(
            request_body=sample_openai_request,
            llm_api_url="https://api.openai.com/v1/chat/completions",
            api_key="sk-test123",
        )

        assert route.called
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer sk-test123"

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_upstream_error(self, sample_openai_request):
        """Verify HTTPStatusError is raised on upstream failure."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await forward_to_llm(
                request_body=sample_openai_request,
                llm_api_url="https://api.openai.com/v1/chat/completions",
                api_key="test-key",
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_uses_explicit_llm_url(self, sample_openai_request, sample_openai_response):
        """Verify the provided upstream URL is used as-is."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_openai_response)
        )

        await forward_to_llm(
            request_body=sample_openai_request,
            llm_api_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
        )

        assert route.called

    def test_derives_upstream_api_base_from_chat_url(self):
        assert _derive_upstream_api_base(
            llm_api_url="https://api.openai.com/v1/chat/completions"
        ) == "https://api.openai.com/v1"

    def test_derives_upstream_api_base_from_base_url(self):
        assert _derive_upstream_api_base(
            llm_api_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ) == "https://generativelanguage.googleapis.com/v1beta/openai"

    @pytest.mark.asyncio
    @respx.mock
    async def test_accepts_base_url_for_chat_completions(self, sample_openai_request, sample_openai_response):
        route = respx.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_openai_response)
        )

        result = await forward_to_llm(
            request_body=sample_openai_request,
            llm_api_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="test-key",
        )

        assert route.called
        assert result["id"] == "chatcmpl-abc123"

    @pytest.mark.asyncio
    @respx.mock
    async def test_lists_models_from_upstream(self):
        route = respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-4o-mini",
                            "object": "model",
                            "created": 1700000000,
                            "owned_by": "openai",
                        }
                    ],
                },
            )
        )

        result = await list_upstream_models(
            llm_api_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
        )

        assert route.called
        assert result["data"][0]["id"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    @respx.mock
    async def test_retrieves_one_model_from_upstream(self):
        route = respx.get("https://api.openai.com/v1/models/gpt-4o-mini").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "gpt-4o-mini",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "openai",
                },
            )
        )

        result = await retrieve_upstream_model(
            model_id="gpt-4o-mini",
            llm_api_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
        )

        assert route.called
        assert result["id"] == "gpt-4o-mini"
