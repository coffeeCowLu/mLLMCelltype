"""Anthropic provider module for LLMCellType."""

from __future__ import annotations

from typing import Any

import requests

from ..logger import write_log
from .common import (
    call_http_api_with_retry,
    ensure_api_key,
    resolve_endpoint_url,
)

# Model alias mapping: user-friendly names -> official API model IDs
MODEL_ALIASES = {
    # Claude 4.6 series (latest - Feb 2026)
    "claude-opus-4.6": "claude-opus-4-6-20260205",
    "claude-opus-latest": "claude-opus-4-6-20260205",
    # Claude 4.5 series (Nov 2025)
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-sonnet-latest": "claude-sonnet-4-5-20250929",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-haiku-latest": "claude-haiku-4-5-20251001",
    # Claude 4.1 series (Aug 2025)
    "claude-opus-4.1": "claude-opus-4-1-20250805",
    # Claude 4 series (May 2025)
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    # Claude 3.7 series (Feb 2025)
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    # Claude 3.5 series (2024)
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-new": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-old": "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    # Claude 3 series (2024)
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


def _resolve_model_name(model: str) -> str:
    """Resolve model alias to official API model ID."""
    return MODEL_ALIASES.get(model, model)


def _parse_anthropic_response(content: dict[str, Any]) -> list[str]:
    """Parse Anthropic response payload into clean lines."""
    try:
        text = content["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected response format from Anthropic: {content}") from e

    if not isinstance(text, str):
        write_log("Unexpected non-string response content from Anthropic, coercing to string", level="warning")
        text = str(text)

    lines = text.strip().split("\n")
    return [line.rstrip(",") for line in lines]


def process_anthropic(
    prompt: str, model: str, api_key: str, base_url: str | None = None
) -> list[str]:
    """Process request using Anthropic Claude models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'claude-3-opus', 'claude-sonnet-4-5-20250929')
        api_key: Anthropic API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting Anthropic API request with model: {model}")

    api_key = ensure_api_key(api_key, "Anthropic")

    # Resolve model aliases (e.g., claude-opus-4.5 -> claude-opus-4-5-20251101)
    model = _resolve_model_name(model)
    write_log(f"Using model: {model}")

    url = resolve_endpoint_url("anthropic", "Anthropic", base_url)

    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }

    return call_http_api_with_retry(
        provider_name="Anthropic",
        url=url,
        body=body,
        headers=headers,
        post_func=requests.post,
        response_parser=_parse_anthropic_response,
        max_retries=3,
        retry_delay=2,
        timeout=30,
        request_json=False,
    )
