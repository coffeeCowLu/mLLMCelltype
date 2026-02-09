"""URL utilities for base URL resolution."""

from __future__ import annotations

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

    if isinstance(base_urls, str):
        return base_urls  # Single URL applies to all providers

    if isinstance(base_urls, dict):
        # Case-insensitive lookup: try original, then lowercase
        if provider in base_urls:
            return base_urls[provider]
        if provider.lower() in base_urls:
            return base_urls[provider.lower()]

    return None


def validate_base_url(url: str) -> bool:
    """Validate base URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False

    # Basic URL format check
    return url.startswith("http://") or url.startswith("https://")


def get_working_minimax_endpoint(api_key: str) -> str:
    """Smart endpoint selection for MiniMax.

    MiniMax has region-specific endpoints that only accept keys issued for
    that region (e.g. Coding Plan ``sk-cp-`` keys work on the domestic
    ``.com`` endpoint but not on the international ``.chat`` endpoint).
    Unlike Qwen where any key works on any reachable endpoint, MiniMax
    requires matching the key to its endpoint, so we test authentication
    rather than mere connectivity.

    Args:
        api_key: MiniMax API key

    Returns:
        Working endpoint URL
    """
    endpoints = [
        "https://api.minimaxi.com/v1/chat/completions",   # Domestic (China)
        "https://api.minimaxi.chat/v1/chat/completions",  # International
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
        except Exception:
            write_log(f"MiniMax endpoint unreachable: {endpoint}", level="debug")

    # Fallback to domestic endpoint
    write_log("No MiniMax endpoint accepted the key, using domestic endpoint as fallback")
    return endpoints[0]


def get_working_qwen_endpoint(api_key: str) -> str:
    """Smart endpoint selection for Qwen.

    Args:
        api_key: Qwen API key

    Returns:
        Working endpoint URL
    """

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

        except Exception:
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
            return endpoint
        else:
            write_log(f"{endpoint_type} endpoint is not accessible", level="warning")

    # If none are reachable, return international endpoint as fallback
    write_log("No endpoints accessible, using international endpoint as fallback")
    return endpoints[0]
