"""Qwen provider module for LLMCellType."""

from __future__ import annotations

import requests

from ..logger import write_log
from ..url_utils import get_working_qwen_endpoint
from .common import (
    UsageSink,
    build_chat_completions_body,
    call_openai_compatible_api,
    ensure_api_key,
    resolve_endpoint_url,
)


def process_qwen(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
    normalize_response: bool = True,
) -> list[str] | str:
    """Process request using Alibaba Qwen models with smart endpoint selection.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'qwen3.6-plus', 'qwen3.6-flash', 'qwen3.6-max-preview')
        api_key: DashScope API key
        base_url: Optional custom base URL (overrides smart selection)
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting Qwen API request with model: {model}")

    api_key = ensure_api_key(api_key, "DashScope")

    # Use custom URL or smart selection
    if base_url:
        url = resolve_endpoint_url("qwen", "Qwen", base_url)
    else:
        url = get_working_qwen_endpoint(api_key)
        write_log(f"Using smart-selected endpoint: {url}")

    write_log(f"Using model: {model}")

    body = build_chat_completions_body(
        model=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=4096,
    )

    return call_openai_compatible_api(
        provider_name="Qwen",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
        usage_sink=usage_sink,
        normalize_response=normalize_response,
    )
