"""OpenRouter provider module for LLMCellType."""

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


def process_openrouter(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Process request using OpenRouter API, which provides access to various LLM models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'openai/gpt-5.5', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-opus-4.7')
        api_key: OpenRouter API key
        base_url: Optional custom base URL
        usage_sink: Optional dict populated in place with token usage. When
            provided, the request opts in to OpenRouter's ``usage: {include: true}``
            so the response carries an accounted ``cost`` (USD) field.

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting OpenRouter API request with model: {model}")

    api_key = ensure_api_key(api_key, "OpenRouter")
    url = resolve_endpoint_url("openrouter", "OpenRouter", base_url)

    write_log(f"Using model: {model}")

    # Ensure model ID is in the correct format for OpenRouter (provider/model)
    if "/" not in model:
        write_log(
            f"Model ID '{model}' may not be in the correct format for OpenRouter. "
            "Expected format: 'provider/model'",
            level="warning",
        )

    body = build_chat_completions_body(model=model, prompt=prompt)
    if usage_sink is not None:
        # Opt in to OpenRouter accounting only when the caller wants usage back,
        # so the default request shape is unchanged.
        body["usage"] = {"include": True}

    return call_openai_compatible_api(
        provider_name="OpenRouter",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
        extra_headers={
            "HTTP-Referer": "https://github.com/cafferychen777/mLLMCelltype",
            "X-Title": "mLLMCelltype",
        },
        usage_sink=usage_sink,
    )
