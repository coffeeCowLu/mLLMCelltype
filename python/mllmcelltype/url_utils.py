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


def resolve_provider_base_url(provider: str, base_urls: str | dict | None) -> str | None:
    """Resolve provider-specific base URL.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        base_urls: User-provided base URLs (string or dict)

    Returns:
        Resolved base URL or None
    """
    def normalize_value(value: object, source: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"base_urls{source} must be a string URL, got {type(value).__name__}")
        normalized = value.strip()
        if normalized and not validate_base_url(normalized):
            raise ValueError(f"Invalid base URL in base_urls{source}: {value}")
        return normalized or None

    if base_urls is None or provider is None:
        return None

    provider_normalized = str(provider).strip().lower()
    if not provider_normalized:
        return None

    if isinstance(base_urls, str):
        return normalize_value(base_urls, "")

    if isinstance(base_urls, dict):
        # Case-insensitive lookup
        if provider in base_urls:
            return normalize_value(base_urls[provider], f"[{provider!r}]")
        if provider_normalized in base_urls:
            return normalize_value(base_urls[provider_normalized], f"[{provider_normalized!r}]")
        normalized_base_urls = {str(k).strip().lower(): v for k, v in base_urls.items()}
        return normalize_value(normalized_base_urls.get(provider_normalized), f"[{provider_normalized!r}]")

    raise ValueError(
        f"base_urls must be a string, dict, or None, got {type(base_urls).__name__}"
    )


def validate_base_url(url: str | None) -> bool:
    """Validate base URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(url, str) or not url:
        return False

    normalized = url.strip()
    if not normalized:
        return False

    # Disallow whitespace and URL fragments in endpoint strings.
    if any(ch.isspace() for ch in normalized):
        return False

    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    if parsed.query:
        return False
    return not parsed.fragment


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
        "https://api.minimax.io/v1/chat/completions",   # International
        "https://api.minimaxi.com/v1/chat/completions",  # China mainland
    ]

    write_log("Testing MiniMax endpoint compatibility...", level="debug")

    for endpoint in endpoints:
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            test_body = {
                "model": "MiniMax-M2.1",
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
    _SMART_ENDPOINT_CACHE[cache_key] = endpoints[0]
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

            # Any HTTP response indicates the endpoint is network-reachable
            # 200: success, 400: bad request, 401/403: auth error - all mean endpoint works
            # Only connection failures (timeout, DNS error, etc.) indicate unreachable endpoint
            return response.status_code is not None

        except requests.RequestException:
            return False

    endpoints = [
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",  # International
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  # Domestic (China)
    ]

    write_log("Testing Qwen endpoint connectivity...", level="debug")

    for i, endpoint in enumerate(endpoints):
        endpoint_type = "international" if i == 0 else "domestic"
        write_log(f"Testing {endpoint_type} endpoint: {endpoint}", level="debug")

        if test_endpoint_connectivity(endpoint, api_key):
            write_log(f"{endpoint_type} endpoint is accessible", level="debug")
            _SMART_ENDPOINT_CACHE[cache_key] = endpoint
            return endpoint
        else:
            write_log(f"{endpoint_type} endpoint is not accessible", level="warning")

    # If none are reachable, return international endpoint as fallback
    write_log("No endpoints accessible, using international endpoint as fallback")
    _SMART_ENDPOINT_CACHE[cache_key] = endpoints[0]
    return endpoints[0]
