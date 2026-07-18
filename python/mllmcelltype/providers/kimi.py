"""Kimi provider module for LLMCellType.

By default this targets the Moonshot AI Open Platform (api.moonshot.cn) over
the OpenAI-compatible Chat Completions protocol; Kimi k2 thinking mode is
explicitly disabled so cell-type annotations stay deterministic.

A custom ``base_url`` may instead point at the Kimi Code platform
(api.kimi.com/coding), which speaks both protocols. The protocol is inferred
from the effective endpoint URL:

- ``.../v1/messages`` -> Anthropic Messages protocol (x-api-key +
  anthropic-version headers)
- ``.../v1/chat/completions``, the base ``https://api.kimi.com/coding``,
  or any other URL -> OpenAI-compatible Chat Completions protocol
"""

from __future__ import annotations

from typing import Any

import requests

from ..logger import write_log
from .anthropic import extract_anthropic_usage
from .common import (
    UsageSink,
    build_chat_completions_body,
    call_http_api_with_retry,
    call_openai_compatible_api,
    ensure_api_key,
    extract_messages_response_text,
    normalize_response_lines,
    resolve_endpoint_url,
)

# Thinking policy: Kimi k2 models enable reasoning by default; annotation
# prompts expect plain per-cluster lines, so thinking is always disabled.
KIMI_THINKING_POLICY = {"type": "disabled"}

_KIMI_ANTHROPIC_VERSION = "2023-06-01"


def _resolve_kimi_protocol(url: str) -> tuple[str, str]:
    """Classify a Kimi endpoint URL into ``(final_url, protocol)``.

    Kimi Code base URLs are completed to their canonical endpoints; every
    other URL is used as-is. Protocol is ``"anthropic"`` for Messages
    endpoints and ``"openai"`` otherwise.
    """
    lower = url.lower()
    if lower.endswith("/messages"):
        return url, "anthropic"
    if lower.endswith("/chat/completions"):
        return url, "openai"
    if lower.endswith("/coding/v1"):
        completed = f"{url}/chat/completions"
        write_log(f"Using Kimi Code OpenAI-compatible endpoint: {completed}")
        return completed, "openai"
    if lower.endswith("/coding"):
        completed = f"{url}/v1/chat/completions"
        write_log(f"Using Kimi Code OpenAI-compatible endpoint: {completed}")
        return completed, "openai"
    return url, "openai"


def _parse_kimi_messages_response(content: dict[str, Any]) -> list[str]:
    """Parse an Anthropic-compatible Kimi Messages payload into clean lines."""
    text = extract_messages_response_text(content, "Kimi")
    return normalize_response_lines(text, "Kimi")


def process_kimi(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Process request using Kimi models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'kimi-k2.6', 'moonshot-v1-8k')
        api_key: Moonshot API key
        base_url: Optional custom base URL. Kimi Code endpoints are detected
            automatically: URLs ending in '/messages' use the Anthropic Messages
            protocol; the Kimi Code base 'https://api.kimi.com/coding' and URLs
            ending in '/chat/completions' use OpenAI-compatible Chat Completions.
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting Kimi API request with model: {model}")

    api_key = ensure_api_key(api_key, "Kimi")
    url = resolve_endpoint_url("kimi", "Kimi", base_url)
    url, protocol = _resolve_kimi_protocol(url)

    write_log(f"Using model: {model}")

    if protocol == "anthropic":
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.6,
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": _KIMI_ANTHROPIC_VERSION,
        }
        return call_http_api_with_retry(
            provider_name="Kimi",
            url=url,
            body=body,
            headers=headers,
            post_func=requests.post,
            response_parser=_parse_kimi_messages_response,
            usage_sink=usage_sink,
            usage_parser=extract_anthropic_usage,
        )

    body = build_chat_completions_body(
        model=model,
        prompt=prompt,
        temperature=0.6,
        max_tokens=4096,
    )
    body["thinking"] = dict(KIMI_THINKING_POLICY)

    return call_openai_compatible_api(
        provider_name="Kimi",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
        usage_sink=usage_sink,
    )
