"""OpenAI provider module for LLMCellType."""

from __future__ import annotations

import requests

from ..logger import write_log
from .common import (
    UsageSink,
    build_chat_completions_body,
    call_openai_compatible_api,
    ensure_api_key,
    resolve_endpoint_url,
)


def process_openai(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Process request using OpenAI models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'gpt-5.5', 'gpt-5.4-mini')
        api_key: OpenAI API key
        base_url: Optional custom base URL
        usage_sink: Optional dict populated in place with token usage.

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
        usage_sink=usage_sink,
    )
