"""Tests for the Kimi provider (Moonshot AI Open Platform)."""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from mllmcelltype.config import (
    PROVIDER_CONFIGS,
    get_api_key_env_var,
    get_default_api_url,
    get_default_model,
)
from mllmcelltype.functions import (
    PROVIDER_FUNCTIONS,
    PROVIDER_MODEL_PREFIXES,
    get_provider,
)
from mllmcelltype.providers.common import NonRetryableProviderError
from mllmcelltype.providers.kimi import (
    KIMI_THINKING_POLICY,
    _resolve_kimi_protocol,
    process_kimi,
)

KIMI_DEFAULT_URL = "https://api.moonshot.cn/v1/chat/completions"


def _ok_response(payload: dict) -> MagicMock:
    """Build a 200 OK mock requests.Response returning ``payload`` from .json()."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def _error_response(status_code: int, payload: dict) -> MagicMock:
    """Build a non-200 mock requests.Response carrying a JSON error payload."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.side_effect = requests.HTTPError(
        f"HTTP {status_code}", response=response
    )
    return response


def _annotation_payload(
    content: str = "Cluster 1: T cells\nCluster 2: B cells",
    usage: dict | None = None,
) -> dict:
    """Build an OpenAI-compatible chat completions payload for Kimi."""
    payload: dict = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


# --- Provider detection ---


def test_get_provider_detects_kimi_open_platform_prefixes():
    """Test kimi- and moonshot- model names resolve to the kimi provider."""
    assert get_provider("kimi-k2.6") == "kimi"
    assert get_provider("moonshot-v1-8k") == "kimi"


def test_get_provider_kimi_matching_is_case_insensitive_and_trimmed():
    """Test detection normalizes case and surrounding whitespace."""
    assert get_provider("KIMI-K2.6") == "kimi"
    assert get_provider("  Moonshot-v1-8k  ") == "kimi"


# --- Configuration contract ---


def test_kimi_config_defaults_target_open_platform():
    """Test default model, endpoint, and API-key environment variable."""
    assert get_default_model("kimi") == "kimi-k2.6"
    assert get_default_api_url("kimi") == KIMI_DEFAULT_URL
    assert get_api_key_env_var("kimi") == "MOONSHOT_API_KEY"
    assert PROVIDER_CONFIGS["kimi"].model_prefixes == ("kimi-", "moonshot-")


# --- Registry consistency ---


def test_kimi_runtime_registry_is_built_from_config():
    """Test the auto-built dispatch map wires kimi without manual edits."""
    assert PROVIDER_FUNCTIONS["kimi"] is process_kimi
    assert PROVIDER_MODEL_PREFIXES["kimi"] == ("kimi-", "moonshot-")
    parameters = tuple(inspect.signature(process_kimi).parameters)
    assert parameters == ("prompt", "model", "api_key", "base_url", "usage_sink", "normalize_response")


# --- Request shape ---


@patch("mllmcelltype.providers.kimi.call_openai_compatible_api")
@patch("mllmcelltype.providers.kimi.resolve_endpoint_url")
def test_process_kimi_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test Kimi wrapper builds the OpenAI-compatible body with thinking disabled."""
    mock_resolve_endpoint.return_value = KIMI_DEFAULT_URL
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_kimi("genes", "kimi-k2.6", "test-key")

    assert result == ["Cluster 1: T cells"]
    mock_resolve_endpoint.assert_called_once_with("kimi", "Kimi", None)
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "Kimi"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["url"] == KIMI_DEFAULT_URL
    assert kwargs["body"]["model"] == "kimi-k2.6"
    assert kwargs["body"]["messages"] == [{"role": "user", "content": "genes"}]
    assert kwargs["body"]["temperature"] == 0.7
    assert kwargs["body"]["max_tokens"] == 4096
    assert kwargs["body"]["thinking"] == {"type": "disabled"}


@patch("mllmcelltype.providers.kimi.call_openai_compatible_api")
@patch("mllmcelltype.providers.kimi.resolve_endpoint_url")
def test_process_kimi_thinking_body_does_not_alias_module_constant(
    mock_resolve_endpoint,
    mock_call_api,
):
    """Test the per-request thinking policy cannot mutate the shared constant."""
    mock_resolve_endpoint.return_value = KIMI_DEFAULT_URL
    mock_call_api.return_value = ["ok"]

    process_kimi("genes", "kimi-k2.6", "test-key")

    thinking = mock_call_api.call_args.kwargs["body"]["thinking"]
    assert thinking == KIMI_THINKING_POLICY
    assert thinking is not KIMI_THINKING_POLICY


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_sends_bearer_header_to_open_platform(mock_post):
    """Test the HTTP request targets api.moonshot.cn with Bearer auth."""
    mock_post.return_value = _ok_response(_annotation_payload("Cluster 1: T cells"))

    result = process_kimi("genes", "kimi-k2.6", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_post.call_args.kwargs
    assert kwargs["url"] == KIMI_DEFAULT_URL
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    body = json.loads(kwargs["data"])
    assert body["model"] == "kimi-k2.6"
    assert body["messages"] == [{"role": "user", "content": "genes"}]
    assert body["thinking"] == {"type": "disabled"}


# --- Response parsing and usage ---


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_parses_lines_and_normalizes_usage(mock_post):
    """Test multi-line content is cleaned and usage fills the caller's sink."""
    mock_post.return_value = _ok_response(
        _annotation_payload(
            "  Cluster 1: T cells,  \n\nCluster 2: B cells,\n",
            usage={"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36},
        )
    )
    sink: dict = {}

    result = process_kimi("genes", "kimi-k2.6", "test-key", usage_sink=sink)

    assert result == ["Cluster 1: T cells", "Cluster 2: B cells"]
    assert sink == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_missing_usage_block_leaves_sink_empty(mock_post):
    """Test a response without usage telemetry does not crash or touch the sink."""
    mock_post.return_value = _ok_response(_annotation_payload())
    sink: dict = {}

    result = process_kimi("genes", "kimi-k2.6", "test-key", usage_sink=sink)

    assert result == ["Cluster 1: T cells", "Cluster 2: B cells"]
    assert sink == {}


# --- Custom endpoint handling ---


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_honors_custom_base_url(mock_post):
    """Test a custom endpoint overrides the default Open Platform URL."""
    mock_post.return_value = _ok_response(_annotation_payload("Cluster 1: T cells"))

    result = process_kimi(
        "genes",
        "kimi-k2.6",
        "test-key",
        base_url="https://proxy.example/v1/chat/completions/",
    )

    assert result == ["Cluster 1: T cells"]
    assert mock_post.call_args.kwargs["url"] == "https://proxy.example/v1/chat/completions"


def test_process_kimi_rejects_invalid_base_url_before_request():
    """Test invalid custom endpoints fail fast during URL resolution."""
    with pytest.raises(ValueError, match="Invalid base URL"):
        process_kimi("genes", "kimi-k2.6", "test-key", base_url="not-a-url")


# --- Malformed and error responses ---


def test_process_kimi_missing_api_key_rejected():
    """Test an absent API key fails before any HTTP request."""
    with pytest.raises(ValueError, match="Kimi API key is missing or empty"):
        process_kimi("genes", "kimi-k2.6", "   ")


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_non_200_surfaces_provider_error_without_retry(mock_post):
    """Test a 401 error raises immediately with the provider message."""
    mock_post.return_value = _error_response(
        401, {"error": {"message": "Invalid authentication credentials"}}
    )

    with pytest.raises(requests.HTTPError):
        process_kimi("genes", "kimi-k2.6", "test-key")

    assert mock_post.call_count == 1


@patch("mllmcelltype.providers.common.time.sleep")
@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_retries_transient_status_then_raises(mock_post, mock_sleep):
    """Test transient 5xx responses use the shared retry policy."""
    mock_post.return_value = _error_response(500, {"error": {"message": "server exploded"}})

    with pytest.raises(requests.HTTPError):
        process_kimi("genes", "kimi-k2.6", "test-key")

    assert mock_post.call_count == 3
    assert [call.args[0] for call in mock_sleep.call_args_list] == [2, 4]


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_malformed_payload_fails_fast(mock_post):
    """Test a payload without choices is non-retryable."""
    mock_post.return_value = _ok_response({"invalid": "payload"})

    with pytest.raises(NonRetryableProviderError, match="Failed to parse Kimi response"):
        process_kimi("genes", "kimi-k2.6", "test-key")

    assert mock_post.call_count == 1


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_empty_content_rejected(mock_post):
    """Test blank message content is rejected as an empty response."""
    mock_post.return_value = _ok_response(_annotation_payload("   \n  "))

    with pytest.raises(NonRetryableProviderError, match="Empty response content"):
        process_kimi("genes", "kimi-k2.6", "test-key")

    assert mock_post.call_count == 1


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_error_sentinel_content_rejected(mock_post):
    """Test error-sentinel text is treated as a failed provider response."""
    mock_post.return_value = _ok_response(_annotation_payload("Error: quota exceeded"))

    with pytest.raises(NonRetryableProviderError, match="Error response content"):
        process_kimi("genes", "kimi-k2.6", "test-key")

    assert mock_post.call_count == 1


# --- Dual-protocol support (Kimi Code endpoints) ---


def test_resolve_kimi_protocol_classifies_endpoints():
    """Test protocol classification for Open Platform, Kimi Code, and custom URLs."""
    assert _resolve_kimi_protocol("https://api.moonshot.cn/v1/chat/completions") == (
        "https://api.moonshot.cn/v1/chat/completions",
        "openai",
    )
    assert _resolve_kimi_protocol("https://api.kimi.com/coding/v1/messages") == (
        "https://api.kimi.com/coding/v1/messages",
        "anthropic",
    )
    assert _resolve_kimi_protocol("https://api.kimi.com/coding/v1/chat/completions") == (
        "https://api.kimi.com/coding/v1/chat/completions",
        "openai",
    )
    # Kimi Code base URLs are completed to their canonical endpoints
    assert _resolve_kimi_protocol("https://api.kimi.com/coding/v1") == (
        "https://api.kimi.com/coding/v1/chat/completions",
        "openai",
    )
    assert _resolve_kimi_protocol("https://api.kimi.com/coding") == (
        "https://api.kimi.com/coding/v1/chat/completions",
        "openai",
    )
    # Generic custom endpoints default to OpenAI-compatible mode as-is
    assert _resolve_kimi_protocol("https://proxy.example/v1") == (
        "https://proxy.example/v1",
        "openai",
    )


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_anthropic_mode_request_shape(mock_post):
    """Test a Messages endpoint switches Kimi to the Anthropic protocol."""
    mock_post.return_value = _ok_response(
        {
            "content": [{"text": "Cluster 1: T cells"}],
            "usage": {"input_tokens": 20, "output_tokens": 4},
        }
    )
    sink: dict = {}

    result = process_kimi(
        "genes",
        "kimi-for-coding",
        "test-key",
        base_url="https://api.kimi.com/coding/v1/messages",
        usage_sink=sink,
    )

    assert result == ["Cluster 1: T cells"]
    assert sink == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    kwargs = mock_post.call_args.kwargs
    assert kwargs["url"] == "https://api.kimi.com/coding/v1/messages"
    assert kwargs["headers"]["x-api-key"] == "test-key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in kwargs["headers"]
    body = json.loads(kwargs["data"])
    assert body["model"] == "kimi-for-coding"
    assert body["messages"] == [{"role": "user", "content": "genes"}]
    assert body["max_tokens"] == 4096
    assert "thinking" not in body


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_openai_mode_from_coding_base_url(mock_post):
    """Test the bare Kimi Code base resolves to OpenAI chat/completions."""
    mock_post.return_value = _ok_response(_annotation_payload("Cluster 1: T cells"))

    result = process_kimi(
        "genes",
        "kimi-for-coding",
        "test-key",
        base_url="https://api.kimi.com/coding/",
    )

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_post.call_args.kwargs
    assert kwargs["url"] == "https://api.kimi.com/coding/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    body = json.loads(kwargs["data"])
    assert body["thinking"] == {"type": "disabled"}


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_openai_mode_from_coding_v1_base_url(mock_post):
    """Test the Kimi Code OpenAI base resolves to chat/completions."""
    mock_post.return_value = _ok_response(_annotation_payload("Cluster 1: T cells"))

    result = process_kimi(
        "genes",
        "kimi-for-coding",
        "test-key",
        base_url="https://api.kimi.com/coding/v1",
    )

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_post.call_args.kwargs
    assert kwargs["url"] == "https://api.kimi.com/coding/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    body = json.loads(kwargs["data"])
    assert body["thinking"] == {"type": "disabled"}


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_anthropic_mode_malformed_payload_fails_fast(mock_post):
    """Test a malformed Messages payload is non-retryable."""
    mock_post.return_value = _ok_response({"invalid": "payload"})

    with pytest.raises(NonRetryableProviderError, match="Failed to parse Kimi response"):
        process_kimi(
            "genes",
            "kimi-for-coding",
            "test-key",
            base_url="https://api.kimi.com/coding/v1/messages",
        )

    assert mock_post.call_count == 1


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_anthropic_mode_non_200_surfaces_error(mock_post):
    """Test Anthropic-mode HTTP errors surface the provider message."""
    mock_post.return_value = _error_response(401, {"error": {"message": "invalid api key"}})

    with pytest.raises(requests.HTTPError):
        process_kimi(
            "genes",
            "kimi-for-coding",
            "test-key",
            base_url="https://api.kimi.com/coding/v1/messages",
        )

    assert mock_post.call_count == 1


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_anthropic_mode_parses_multi_block_response(mock_post):
    """Test Messages responses with thinking + text blocks return text content."""
    mock_post.return_value = _ok_response({
        "content": [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "Cluster 1: T cells"},
        ],
        "usage": {"input_tokens": 20, "output_tokens": 4},
    })

    result = process_kimi(
        "genes",
        "kimi-for-coding",
        "test-key",
        base_url="https://api.kimi.com/coding/v1/messages",
    )

    assert result == ["Cluster 1: T cells"]


@patch("mllmcelltype.providers.kimi.requests.post")
def test_process_kimi_anthropic_mode_rejects_thinking_only_response(mock_post):
    """Test Messages responses with only thinking blocks are rejected."""
    mock_post.return_value = _ok_response({
        "content": [{"type": "thinking", "thinking": "reasoning..."}]
    })

    with pytest.raises(NonRetryableProviderError, match="Failed to parse Kimi response"):
        process_kimi(
            "genes",
            "kimi-for-coding",
            "test-key",
            base_url="https://api.kimi.com/coding/v1/messages",
        )
