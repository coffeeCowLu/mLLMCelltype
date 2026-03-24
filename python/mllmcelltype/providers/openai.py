"""OpenAI provider module for LLMCellType."""

from __future__ import annotations

import requests

from ..logger import write_log
from .common import (
    build_chat_completions_body,
    call_openai_compatible_api,
    ensure_api_key,
    resolve_endpoint_url,
)


def process_openai(
    prompt: str, model: str, api_key: str, base_url: str | None = None
) -> list[str]:
    """Process request using OpenAI models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'gpt-5.2', 'o1')
        api_key: OpenAI API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting OpenAI API request with model: {model}")

    api_key = ensure_api_key(api_key, "OpenAI")
    url = resolve_endpoint_url("openai", "OpenAI", base_url)

    write_log(f"Using model: {model}")

    body = build_chat_completions_body(model=model, prompt=prompt)
    return call_openai_compatible_api(
        provider_name="OpenAI",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
    )
