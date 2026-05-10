"""Grok provider module for LLMCellType."""

from __future__ import annotations

import requests

from ..logger import write_log
from .common import (
    build_chat_completions_body,
    call_openai_compatible_api,
    ensure_api_key,
    resolve_endpoint_url,
)


def process_grok(
    prompt: str, model: str, api_key: str, base_url: str | None = None
) -> list[str]:
    """Process request using Grok models from xAI.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'grok-4.3', 'grok-4.3-latest')
        api_key: xAI API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting Grok API request with model: {model}")

    api_key = ensure_api_key(api_key, "Grok")
    url = resolve_endpoint_url("grok", "Grok", base_url)

    write_log(f"Using model: {model}")

    body = build_chat_completions_body(model=model, prompt=prompt)
    return call_openai_compatible_api(
        provider_name="Grok",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
    )
