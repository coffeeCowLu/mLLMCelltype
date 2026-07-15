"""URL utilities for base URL resolution."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

import requests

from .config import get_default_api_url
from .logger import write_log

# Re-export get_default_api_url for backward compatibility
# The actual implementation is in config.py (Single Source of Truth)
__all__ = [
    "get_default_api_url",
    "get_working_minimax_endpoint",
    "get_working_qwen_endpoint",
    "resolve_provider_base_url",
    "validate_base_url",
]

# In-memory cache for smart endpoint probing results.
# Keyed by provider + API key fingerprint to avoid repeated probe calls.
_SMART_ENDPOINT_CACHE: dict[str, str] = {}


def _endpoint_cache_key(provider: str, api_key: str) -> str:
    """Build a deterministic cache key for provider+api_key."""
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"{provider}:{key_hash}"


def _normalize_base_url_value(value: object, source: str) -> str | None:
    """Normalize and validate one optional base URL value."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"base_urls{source} must be a string URL, got {type(value).__name__}")
    normalized = value.strip().rstrip("/")
    if normalized and not validate_base_url(normalized):
        raise ValueError(f"Invalid base URL in base_urls{source}: {value}")
    return normalized or None


def _normalize_base_url_mapping(base_urls: dict) -> tuple[dict[str, object], dict[str, object]]:
    """Normalize provider keys while rejecting ambiguous duplicates."""
    normalized: dict[str, object] = {}
    original_keys: dict[str, object] = {}
    for raw_provider, value in base_urls.items():
        if not isinstance(raw_provider, str):
            raise ValueError(
                f"base_urls provider keys must be strings, got {type(raw_provider).__name__}"
            )
        provider = raw_provider.strip().lower()
        if not provider:
            raise ValueError("base_urls provider keys must be non-empty strings")
        if provider in normalized:
            raise ValueError(
                "Ambiguous base_urls provider keys after case/whitespace normalization: "
                f"{original_keys[provider]!r} and {raw_provider!r}"
            )
        normalized[provider] = value
        original_keys[provider] = raw_provider
    return normalized, original_keys


def resolve_provider_base_url(provider: str, base_urls: str | dict | None) -> str | None:
    """Resolve provider-specific base URL.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        base_urls: User-provided base URLs (string or dict)

    Returns:
        Resolved base URL or None
    """
    if base_urls is None or provider is None:
        return None

    provider_normalized = str(provider).strip().lower()
    if not provider_normalized:
        return None

    if isinstance(base_urls, str):
        return _normalize_base_url_value(base_urls, "")

    if isinstance(base_urls, dict):
        normalized_base_urls, original_keys = _normalize_base_url_mapping(base_urls)
        raw_key = original_keys.get(provider_normalized, provider_normalized)
        return _normalize_base_url_value(
            normalized_base_urls.get(provider_normalized),
            f"[{raw_key!r}]",
        )

    raise ValueError(f"base_urls must be a string, dict, or None, got {type(base_urls).__name__}")


def validate_base_url(url: str | None) -> bool:
    """Validate base URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(url, str):
        return False

    normalized = url.strip()
    if not normalized or any(character.isspace() for character in normalized):
        return False

    try:
        parsed = urlsplit(normalized)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError:
        return False

    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc and hostname)
        and not parsed.netloc.endswith(":")
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and (port is None or 1 <= port <= 65535)
    )


def get_working_minimax_endpoint(api_key: str) -> str:
    """Smart endpoint selection for MiniMax.

    MiniMax documents region-based API hosts:
    - International: api.minimax.io
    - China mainland: api.minimaxi.com
    We test both hosts and pick one that accepts the key.

    Args:
        api_key: MiniMax API key

    Returns:
        Working endpoint URL
    """
    cache_key = _endpoint_cache_key("minimax", api_key)
    cached = _SMART_ENDPOINT_CACHE.get(cache_key)
    if cached:
        write_log(f"Using cached MiniMax endpoint: {cached}", level="debug")
        return cached

    endpoints = [
        "https://api.minimax.io/v1/chat/completions",  # International
        "https://api.minimaxi.com/v1/chat/completions",  # China mainland
    ]

    write_log("Testing MiniMax endpoint compatibility...", level="debug")

    for endpoint in endpoints:
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            test_body = {
                "model": "MiniMax-M2.7",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }
            response = requests.post(endpoint, headers=headers, json=test_body, timeout=5)

            # MiniMax returns HTTP 200 even for auth errors; check base_resp
            if response.status_code == 200:
                data = response.json()
                base_resp = data.get("base_resp", {})
                status_code = base_resp.get("status_code", 0)
                if status_code == 0:
                    write_log(f"MiniMax endpoint accepted key: {endpoint}", level="debug")
                    _SMART_ENDPOINT_CACHE[cache_key] = endpoint
                    return endpoint
                write_log(
                    f"MiniMax endpoint rejected key ({base_resp.get('status_msg', '')}): {endpoint}",
                    level="debug",
                )
            else:
                write_log(
                    f"MiniMax endpoint returned HTTP {response.status_code}: {endpoint}",
                    level="debug",
                )
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            write_log(f"MiniMax endpoint unreachable: {endpoint}", level="debug")

    # Fallback to international endpoint
    write_log("No MiniMax endpoint accepted the key, using international endpoint as fallback")
    return endpoints[0]


def get_working_qwen_endpoint(api_key: str) -> str:
    """Smart endpoint selection for Qwen.

    Args:
        api_key: Qwen API key

    Returns:
        Working endpoint URL
    """
    cache_key = _endpoint_cache_key("qwen", api_key)
    cached = _SMART_ENDPOINT_CACHE.get(cache_key)
    if cached:
        write_log(f"Using cached Qwen endpoint: {cached}", level="debug")
        return cached

    def test_endpoint_connectivity(endpoint: str, api_key: str, timeout: int = 5) -> bool:
        """Test endpoint connectivity.

        Args:
            endpoint: API endpoint URL
            api_key: API key for authentication
            timeout: Timeout in seconds

        Returns:
            True if endpoint is accessible, False otherwise
        """
        try:
            # Send a simple test request
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            test_body = {
                "model": "qwen-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }

            response = requests.post(endpoint, headers=headers, json=test_body, timeout=timeout)

            # Choose an endpoint likely to accept this key. Region-specific
            # DashScope keys can return 401/403 on the wrong regional host.
            status_code = response.status_code
            if status_code in {401, 403, 404}:
                return False
            return status_code is not None and status_code < 500

        except requests.RequestException:
            return False

    endpoints = [
        (
            "international",
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        ),
        ("domestic", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        (
            "legacy international",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        ),
    ]

    write_log("Testing Qwen endpoint connectivity...", level="debug")

    for endpoint_type, endpoint in endpoints:
        write_log(f"Testing {endpoint_type} endpoint: {endpoint}", level="debug")

        if test_endpoint_connectivity(endpoint, api_key):
            write_log(f"{endpoint_type} endpoint is accessible", level="debug")
            _SMART_ENDPOINT_CACHE[cache_key] = endpoint
            return endpoint
        write_log(f"{endpoint_type} endpoint is not accessible", level="warning")

    # If none are reachable, return international endpoint as fallback
    write_log("No endpoints accessible, using international endpoint as fallback")
    return endpoints[0][1]
