"""
Proxy — Forwards requests to the upstream OpenAI-compatible API.
"""
from typing import Any, Dict, Optional


def _normalize_upstream_base_url(url: str) -> str:
    """Normalize an upstream OpenAI-compatible base URL."""
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    return normalized


def _normalize_chat_completions_url(url: str) -> str:
    """Normalize an upstream chat completions URL from either a base or full path."""
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _derive_upstream_api_base(
    llm_api_url: Optional[str] = None,
    upstream_base_url: Optional[str] = None,
) -> str:
    """Derive the upstream API base from a base URL or chat-completions URL."""
    if upstream_base_url:
        return _normalize_upstream_base_url(upstream_base_url)

    if not llm_api_url:
        raise ValueError("Either llm_api_url or upstream_base_url must be provided.")

    return _normalize_upstream_base_url(llm_api_url)


async def _request_upstream_json(
    method: str,
    url: str,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    json_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method,
            url,
            json=json_body,
            headers=_build_headers(api_key),
        )
        response.raise_for_status()
        return response.json()


async def forward_to_llm(
    request_body: Dict[str, Any],
    llm_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    upstream_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Forward a chat completion request to the upstream LLM API.

    Args:
        request_body: The original request body (OpenAI format).
        llm_api_url: Full upstream chat completions URL.
        api_key: Optional API key for the upstream service.
        timeout: Request timeout in seconds.
        upstream_base_url: Backward-compatible base URL for the upstream API.

    Returns:
        The JSON response from the upstream API.

    Raises:
        httpx.HTTPStatusError: If the upstream returns an error status.
    """
    if not llm_api_url:
        llm_api_url = f"{_derive_upstream_api_base(upstream_base_url=upstream_base_url)}/chat/completions"
    else:
        llm_api_url = _normalize_chat_completions_url(llm_api_url)

    return await _request_upstream_json(
        method="POST",
        url=llm_api_url,
        api_key=api_key,
        timeout=timeout,
        json_body=request_body,
    )


async def list_upstream_models(
    llm_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    upstream_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """List models from the upstream OpenAI-compatible API."""
    base_url = _derive_upstream_api_base(
        llm_api_url=llm_api_url,
        upstream_base_url=upstream_base_url,
    )
    return await _request_upstream_json(
        method="GET",
        url=f"{base_url}/models",
        api_key=api_key,
        timeout=timeout,
    )


async def retrieve_upstream_model(
    model_id: str,
    llm_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    upstream_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve one model object from the upstream OpenAI-compatible API."""
    base_url = _derive_upstream_api_base(
        llm_api_url=llm_api_url,
        upstream_base_url=upstream_base_url,
    )
    return await _request_upstream_json(
        method="GET",
        url=f"{base_url}/models/{model_id}",
        api_key=api_key,
        timeout=timeout,
    )
