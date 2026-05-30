"""Shared helpers for OpenAI-compatible provider implementations."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, MutableMapping
from typing import Any

import requests

from ..logger import write_log
from ..url_utils import get_default_api_url, validate_base_url

ResponseParser = Callable[[dict[str, Any]], list[str]]

# A caller-supplied dict that, when passed, is populated in place with token
# usage for the call: prompt_tokens / completion_tokens / total_tokens, plus an
# optional native `cost` (USD) when the provider returns one. None = no capture.
UsageSink = MutableMapping[str, Any]

# Maps a raw response payload to the normalized usage schema (or None if absent).
UsageParser = Callable[[dict[str, Any]], dict[str, Any] | None]


def extract_chat_completions_usage(content: dict[str, Any]) -> dict[str, Any] | None:
    """Extract token usage from an OpenAI-compatible chat completions payload.

    Returns a normalized ``{prompt_tokens, completion_tokens, total_tokens}`` dict
    (plus ``cost`` when the provider includes a native USD cost, e.g. OpenRouter
    when ``usage: {include: true}`` was requested), or ``None`` when no usage block
    is present. Never raises on a malformed/absent ``usage`` block.

    Resilient to provider format drift by design: only the three canonical keys
    (plus ``cost``) are read; any extra/unknown fields are ignored. Missing keys
    are reported as ``None`` rather than fabricated, so callers can tell "not
    reported" from a real zero. Validated live against OpenRouter; the other
    OpenAI-compatible providers (OpenAI, DeepSeek, Qwen, Grok, StepFun, Zhipu,
    MiniMax) share this exact path but were not run against their native endpoints.
    """
    if not isinstance(content, dict):
        return None
    usage = content.get("usage")
    if not isinstance(usage, dict):
        return None

    normalized: dict[str, Any] = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    if usage.get("cost") is not None:
        normalized["cost"] = usage["cost"]
    return normalized


class NonRetryableProviderError(ValueError):
    """Provider error that should fail fast without retry."""


def build_chat_completions_body(
    *,
    model: str,
    prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    user_name: str | None = None,
) -> dict[str, Any]:
    """Build a normalized OpenAI-compatible chat completions request body."""
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if user_name:
        message["name"] = user_name

    body: dict[str, Any] = {
        "model": model,
        "messages": [message],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def parse_chat_completions_response(content: dict[str, Any], provider_name: str) -> list[str]:
    """Parse OpenAI-compatible chat completions response into clean lines."""
    try:
        text = content["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected response format from {provider_name}: {content}") from e

    if not isinstance(text, str):
        write_log(
            f"Unexpected non-string response content from {provider_name}, coercing to string",
            level="warning",
        )
        text = str(text)

    lines = text.strip().split("\n")
    return [line.rstrip(",") for line in lines]


def _extract_error_message(response: requests.Response) -> str | None:
    """Extract provider error message from a non-200 response."""
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError, TypeError):
        return None

    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if isinstance(message, str) and message.strip():
                return message
        if isinstance(error_obj, str) and error_obj.strip():
            return error_obj

        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message

    return None


def ensure_api_key(api_key: str | None, provider_name: str) -> str:
    """Ensure API key is present and return it."""
    if isinstance(api_key, str):
        normalized = api_key.strip()
        if normalized:
            return normalized

    error_msg = f"{provider_name} API key is missing or empty"
    write_log(error_msg, level="error")
    raise ValueError(error_msg)


def resolve_endpoint_url(
    provider_key: str,
    provider_name: str,
    base_url: str | None,
) -> str:
    """Resolve endpoint URL with optional custom override and validation."""
    if base_url is not None:
        if not isinstance(base_url, str):
            raise ValueError(
                f"Invalid base URL type for {provider_name}: "
                f"expected str or None, got {type(base_url).__name__}"
            )
        normalized_base_url = base_url.strip()
        if normalized_base_url:
            if not validate_base_url(normalized_base_url):
                raise ValueError(f"Invalid base URL: {base_url}")
            write_log(f"Using custom base URL: {normalized_base_url}")
            return normalized_base_url

    default_url = get_default_api_url(provider_key)
    if not default_url:
        raise ValueError(
            f"No default API URL configured for {provider_name} "
            f"(provider key: {provider_key})"
        )
    write_log(f"Using default URL: {default_url}")
    return default_url


def call_http_api_with_retry(
    *,
    provider_name: str,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    post_func: Callable[..., requests.Response],
    response_parser: ResponseParser,
    max_retries: int = 3,
    retry_delay: int = 2,
    timeout: int = 30,
    request_json: bool = False,
    non_retry_exceptions: tuple[type[Exception], ...] = (),
    usage_sink: UsageSink | None = None,
    usage_parser: UsageParser = extract_chat_completions_usage,
) -> list[str]:
    """Execute an HTTP API request with retry and unified error handling.

    When ``usage_sink`` is provided, it is populated in place with token usage
    parsed from the successful response via ``usage_parser`` (default: the
    OpenAI-compatible extractor). Leaving it ``None`` is byte-identical to the
    prior behavior.
    """
    write_log("Sending API request...")

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
            if request_json:
                kwargs["json"] = body
            else:
                kwargs["data"] = json.dumps(body)

            response = post_func(**kwargs)

            if response.status_code != 200:
                error_detail = _extract_error_message(response)
                if error_detail:
                    write_log(f"{provider_name} API request failed: {error_detail}", level="error")
                else:
                    write_log(
                        f"{provider_name} API request failed with status {response.status_code}",
                        level="error",
                    )

                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

            content = response.json()
            try:
                res = response_parser(content)
            except NonRetryableProviderError:
                raise
            except (ValueError, KeyError, TypeError, IndexError) as e:
                # Parsing/shape errors are deterministic for a given payload:
                # fail fast instead of retrying the same request.
                raise NonRetryableProviderError(
                    f"Failed to parse {provider_name} response: {e!s}"
                ) from e

            if not isinstance(res, list):
                raise NonRetryableProviderError(
                    f"{provider_name} response parser returned {type(res).__name__}, expected list"
                )
            if usage_sink is not None:
                usage = usage_parser(content)
                if usage is not None:
                    usage_sink.update(usage)

            normalized_res = [str(line) for line in res]
            write_log(f"Got response with {len(normalized_res)} lines")
            write_log(f"Raw response from {provider_name}:\n{normalized_res}", level="debug")
            return normalized_res

        except Exception as e:
            if isinstance(e, NonRetryableProviderError):
                raise
            if non_retry_exceptions and isinstance(e, non_retry_exceptions):
                raise
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response is not None
                and e.response.status_code < 500
            ):
                raise

            write_log(
                f"Error during API call (attempt {attempt + 1}/{max_retries}): {e!s}",
                level="error",
            )
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...", level="warning")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"{provider_name} API request failed after {max_retries} attempts")


def call_openai_compatible_api(
    *,
    provider_name: str,
    api_key: str,
    url: str,
    body: dict[str, Any],
    post_func: Callable[..., requests.Response],
    response_parser: ResponseParser | None = None,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = 3,
    retry_delay: int = 2,
    timeout: int = 30,
    request_json: bool = False,
    non_retry_exceptions: tuple[type[Exception], ...] = (),
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Execute a request against an OpenAI-compatible endpoint with retries.

    When ``usage_sink`` is provided, it is populated in place with the call's
    token usage (see :func:`extract_chat_completions_usage`). Default ``None``
    preserves prior behavior exactly.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    parser = response_parser or (lambda content: parse_chat_completions_response(content, provider_name))

    return call_http_api_with_retry(
        provider_name=provider_name,
        url=url,
        body=body,
        headers=headers,
        post_func=post_func,
        response_parser=parser,
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=timeout,
        request_json=request_json,
        non_retry_exceptions=non_retry_exceptions,
        usage_sink=usage_sink,
    )
