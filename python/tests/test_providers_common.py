"""High-value tests for shared provider HTTP execution logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from mllmcelltype.providers.common import (
    NonRetryableProviderError,
    _extract_error_message,
    build_chat_completions_body,
    call_http_api_with_retry,
    call_openai_compatible_api,
    ensure_api_key,
    extract_chat_completions_usage,
    is_retryable_http_status,
    normalize_optional_base_url,
    normalize_usage,
    parse_chat_completions_response,
    resolve_endpoint_url,
)


@pytest.mark.parametrize("status_code", [408, 425, 429, 500, 503])
def test_retryable_http_status_accepts_only_transient_codes(status_code):
    """Test the shared retry policy covers timeouts, rate limits, and servers."""
    assert is_retryable_http_status(status_code)


@pytest.mark.parametrize("status_code", [None, True, 200, 302, 400, 401, 404])
def test_retryable_http_status_rejects_non_transient_values(status_code):
    """Test successful, client, and malformed statuses fail fast."""
    assert not is_retryable_http_status(status_code)


def test_normalize_optional_base_url_strips_and_validates_once():
    """Test provider wrappers share one endpoint override normalization policy."""
    assert (
        normalize_optional_base_url(
            "  https://proxy.example.com/v1/  ",
            "Gemini",
        )
        == "https://proxy.example.com/v1"
    )
    assert normalize_optional_base_url("   ", "Gemini") is None


def test_parse_chat_completions_response_non_string_content():
    """Test parser rejects non-string content instead of inventing annotation text."""
    payload = {"choices": [{"message": {"content": 123}}]}
    with pytest.raises(NonRetryableProviderError, match="non-string"):
        parse_chat_completions_response(payload, "OpenAI")


def test_build_chat_completions_body_minimal_shape():
    """Test shared body builder returns normalized minimal request payload."""
    body = build_chat_completions_body(model="gpt-5.5", prompt="genes")
    assert body == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "genes"}],
    }


def test_build_chat_completions_body_with_optional_controls():
    """Test shared body builder includes optional generation controls."""
    body = build_chat_completions_body(
        model="MiniMax-M2.7",
        prompt="genes",
        temperature=0.7,
        max_tokens=4096,
        user_name="user",
    )
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 4096
    assert body["messages"][0]["name"] == "user"


def test_parse_chat_completions_response_invalid_shape_raises():
    """Test parser returns clear error on invalid response schema."""
    with pytest.raises(ValueError, match="Unexpected response format"):
        parse_chat_completions_response({"not_choices": []}, "OpenAI")


def test_extract_error_message_handles_multiple_payload_shapes():
    """Test provider error extraction from common API payload formats."""
    response = MagicMock()

    response.json.return_value = {"error": {"message": "nested error"}}
    assert _extract_error_message(response) == "nested error"

    response.json.return_value = {"error": "flat error"}
    assert _extract_error_message(response) == "flat error"

    response.json.return_value = {"message": "top-level message"}
    assert _extract_error_message(response) == "top-level message"

    response.json.side_effect = ValueError("bad json")
    assert _extract_error_message(response) is None


@patch("mllmcelltype.providers.common.time.sleep")
def test_call_http_api_with_retry_handles_429_then_success(mock_sleep):
    """Test 429 rate-limit is retried and eventually succeeds."""
    response_429 = MagicMock()
    response_429.status_code = 429
    response_429.json.return_value = {"error": {"message": "rate limit"}}

    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}

    post_func = MagicMock(side_effect=[response_429, response_200])
    parser = MagicMock(return_value=["Cluster 1: T cells"])

    result = call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=parser,
        max_retries=2,
        retry_delay=1,
    )

    assert result == ["Cluster 1: T cells"]
    assert post_func.call_count == 2
    parser.assert_called_once_with({"ok": True})
    mock_sleep.assert_called_once_with(1)


def test_call_http_api_with_retry_client_error_does_not_retry():
    """Test HTTP 4xx errors fail fast (no pointless retries)."""
    response_400 = MagicMock()
    response_400.status_code = 400
    response_400.json.return_value = {"error": {"message": "bad request"}}
    response_400.raise_for_status.side_effect = requests.HTTPError(
        "400 Client Error", response=response_400
    )

    post_func = MagicMock(return_value=response_400)

    with pytest.raises(requests.HTTPError):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            max_retries=3,
        )

    assert post_func.call_count == 1


def test_call_http_api_with_retry_redirect_does_not_parse_as_success():
    """Test non-error redirect statuses are still rejected by the API contract."""
    response_302 = MagicMock()
    response_302.status_code = 302
    response_302.json.return_value = {"choices": []}
    post_func = MagicMock(return_value=response_302)

    with pytest.raises(requests.HTTPError, match="Unexpected HTTP status 302"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            max_retries=3,
        )

    assert post_func.call_count == 1


@patch("mllmcelltype.providers.common.time.sleep")
def test_call_http_api_with_retry_retries_request_timeout(mock_sleep):
    """Test HTTP 408 follows the same transient-status policy as rate limits."""
    response_408 = MagicMock()
    response_408.status_code = 408
    response_408.json.return_value = {"error": {"message": "request timeout"}}
    response_408.raise_for_status.side_effect = requests.HTTPError(
        "408 Request Timeout", response=response_408
    )

    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}
    post_func = MagicMock(side_effect=[response_408, response_200])

    result = call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda _payload: ["ok"],
        max_retries=2,
        retry_delay=1,
    )

    assert result == ["ok"]
    assert post_func.call_count == 2
    mock_sleep.assert_called_once_with(1)


@patch("mllmcelltype.providers.common.time.sleep")
def test_call_http_api_with_retry_retries_connection_failure(mock_sleep):
    """Test transport failures retry because a later connection can succeed."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}
    post_func = MagicMock(side_effect=[requests.ConnectionError("offline"), response_200])

    result = call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda _payload: ["ok"],
        max_retries=2,
        retry_delay=1,
    )

    assert result == ["ok"]
    assert post_func.call_count == 2
    mock_sleep.assert_called_once_with(1)


def test_call_http_api_with_retry_programming_error_fails_fast():
    """Test local programming errors are not amplified into repeated requests."""
    post_func = MagicMock(side_effect=RuntimeError("broken adapter"))

    with pytest.raises(RuntimeError, match="broken adapter"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            max_retries=3,
        )

    assert post_func.call_count == 1


def test_call_http_api_with_retry_invalid_json_fails_fast():
    """Test a successful HTTP response with invalid JSON is not blindly replayed."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.side_effect = ValueError("bad json")
    post_func = MagicMock(return_value=response_200)

    with pytest.raises(NonRetryableProviderError, match="Failed to decode"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            max_retries=3,
        )

    assert post_func.call_count == 1


def test_call_http_api_with_retry_non_object_json_fails_fast():
    """Test provider payloads must honor the shared top-level object contract."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = ["unexpected"]
    post_func = MagicMock(return_value=response_200)

    with pytest.raises(NonRetryableProviderError, match="expected object"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            max_retries=3,
        )

    assert post_func.call_count == 1


def test_call_http_api_with_retry_non_retry_exception_fails_fast():
    """Test configured non-retry exception exits immediately."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}

    post_func = MagicMock(return_value=response_200)

    with pytest.raises(NonRetryableProviderError):
        call_http_api_with_retry(
            provider_name="MiniMax",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: (_ for _ in ()).throw(
                NonRetryableProviderError("business error")
            ),
            non_retry_exceptions=(NonRetryableProviderError,),
        )

    assert post_func.call_count == 1


def test_call_http_api_with_retry_parser_value_error_fails_fast():
    """Test response parsing schema errors fail fast without retrying."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"unexpected": "shape"}
    post_func = MagicMock(return_value=response_200)

    with pytest.raises(NonRetryableProviderError, match="Failed to parse"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: (_ for _ in ()).throw(ValueError("bad schema")),
            max_retries=3,
        )

    assert post_func.call_count == 1


def test_call_http_api_with_retry_parser_non_list_fails_fast():
    """Test parser must return list and non-list result is rejected immediately."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}
    post_func = MagicMock(return_value=response_200)

    with pytest.raises(NonRetryableProviderError, match="expected list"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: "not-a-list",  # type: ignore[return-value]
            max_retries=3,
        )

    assert post_func.call_count == 1


def test_ensure_api_key_rejects_whitespace_only():
    """Test blank/whitespace API keys are treated as missing."""
    with pytest.raises(ValueError, match="missing or empty"):
        ensure_api_key("   ", "OpenAI")


def test_ensure_api_key_trims_and_returns_value():
    """Test API key normalization trims surrounding whitespace."""
    assert ensure_api_key("  real-key  ", "OpenAI") == "real-key"


def test_call_http_api_with_retry_rejects_non_string_parser_items():
    """Test parser items must already satisfy the shared text response contract."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}
    post_func = MagicMock(return_value=response_200)

    with pytest.raises(NonRetryableProviderError, match="non-string"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: [1, 2],  # type: ignore[list-item]
            max_retries=1,
        )

    assert post_func.call_count == 1


def test_call_http_api_with_retry_request_json_mode_uses_json_kwarg():
    """Test request_json mode uses requests json= parameter (not raw data)."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}
    post_func = MagicMock(return_value=response_200)

    call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda _payload: ["ok"],
        request_json=True,
        max_retries=1,
    )

    called_kwargs = post_func.call_args.kwargs
    assert "json" in called_kwargs
    assert "data" not in called_kwargs


@patch("mllmcelltype.providers.common.time.sleep")
def test_call_http_api_with_retry_server_error_logs_and_raises(mock_sleep):
    """Test HTTP 5xx errors do not fail-fast as 4xx and raise after retries."""
    response_500 = MagicMock()
    response_500.status_code = 500
    response_500.json.return_value = {}
    response_500.raise_for_status.side_effect = requests.HTTPError(
        "500 Server Error", response=response_500
    )
    post_func = MagicMock(return_value=response_500)

    with pytest.raises(requests.HTTPError):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            max_retries=1,
        )

    assert post_func.call_count == 1
    mock_sleep.assert_not_called()


def test_call_openai_compatible_api_sets_auth_header_and_parses():
    """Test OpenAI-compatible wrapper applies headers and default parser."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"choices": [{"message": {"content": "Cluster 1: T cells"}}]}
    post_func = MagicMock(return_value=response_200)

    result = call_openai_compatible_api(
        provider_name="OpenAI",
        api_key="secret-key",
        url="https://api.example.com/v1/chat/completions",
        body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "test"}]},
        post_func=post_func,
    )

    assert result == ["Cluster 1: T cells"]
    called_kwargs = post_func.call_args.kwargs
    assert called_kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert "data" in called_kwargs


def test_call_openai_compatible_api_merges_extra_headers():
    """Test OpenAI-compatible wrapper merges caller-provided extra headers."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"choices": [{"message": {"content": "Cluster 1: T cells"}}]}
    post_func = MagicMock(return_value=response_200)

    call_openai_compatible_api(
        provider_name="OpenRouter",
        api_key="secret-key",
        url="https://api.example.com/v1/chat/completions",
        body={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "test"}]},
        post_func=post_func,
        extra_headers={"X-Title": "mLLMCelltype"},
    )

    called_headers = post_func.call_args.kwargs["headers"]
    assert called_headers["Authorization"] == "Bearer secret-key"
    assert called_headers["X-Title"] == "mLLMCelltype"


def test_resolve_endpoint_url_trims_custom_base_url():
    """Test custom endpoint URL is stripped before validation/use."""
    result = resolve_endpoint_url(
        "openai",
        "OpenAI",
        "  https://proxy.example.com/v1/chat/completions/  ",
    )
    assert result == "https://proxy.example.com/v1/chat/completions"


def test_resolve_endpoint_url_blank_custom_base_url_uses_default():
    """Test blank custom URL is treated as unset and falls back to default endpoint."""
    result = resolve_endpoint_url("openai", "OpenAI", "   ")
    assert result == "https://api.openai.com/v1/chat/completions"


def test_resolve_endpoint_url_rejects_invalid_base_url_type():
    """Test non-string base URL type fails fast with clear error."""
    with pytest.raises(ValueError, match="Invalid base URL type"):
        resolve_endpoint_url("openai", "OpenAI", 123)  # type: ignore[arg-type]


def test_resolve_endpoint_url_unknown_provider_without_default_fails_fast():
    """Test missing provider default URL raises clear error before HTTP layer."""
    with pytest.raises(ValueError, match="No default API URL configured"):
        resolve_endpoint_url("unknown_provider", "Unknown Provider", None)


def test_extract_chat_completions_usage_normalizes_block():
    """Test usage extractor maps the OpenAI usage block to the shared schema."""
    usage = extract_chat_completions_usage(
        {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )
    assert usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_extract_chat_completions_usage_includes_native_cost():
    """Test native USD cost (e.g. OpenRouter) is surfaced when present."""
    usage = extract_chat_completions_usage(
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.0042}}
    )
    assert usage["cost"] == 0.0042


def test_extract_chat_completions_usage_absent_returns_none():
    """Test extractor returns None (does not raise) when no usage block exists."""
    assert extract_chat_completions_usage({"choices": []}) is None


# --- Format-drift resilience: pin behavior so a provider changing shape is caught ---


def test_extract_chat_completions_usage_ignores_unknown_fields():
    """Extra/unknown usage fields are dropped; only canonical keys are read."""
    usage = extract_chat_completions_usage(
        {
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 4},  # newer OpenAI field
                "reasoning_tokens": 2,  # future/unknown field
            }
        }
    )
    assert usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_extract_chat_completions_usage_partial_reports_none_not_zero():
    """Missing token keys are reported as None (not fabricated to 0)."""
    usage = extract_chat_completions_usage({"usage": {"prompt_tokens": 5}})
    assert usage == {"prompt_tokens": 5, "completion_tokens": None, "total_tokens": None}


def test_extract_chat_completions_usage_no_cost_key_when_absent():
    """`cost` is only present in output when the provider actually returned one."""
    usage = extract_chat_completions_usage(
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    )
    assert "cost" not in usage


def test_extract_chat_completions_usage_malformed_block_returns_none():
    """A non-dict `usage` (e.g. provider sends a string/null) yields None, not a crash."""
    assert extract_chat_completions_usage({"usage": "unexpected"}) is None
    assert extract_chat_completions_usage({"usage": None}) is None


def test_normalize_usage_rejects_invalid_token_and_cost_values():
    """Test provider telemetry cannot inject negative, boolean, or non-finite values."""
    usage = normalize_usage(
        {
            "prompt_tokens": -1,
            "completion_tokens": True,
            "total_tokens": "3",
            "cost": float("nan"),
            "unknown": 99,
        }
    )

    assert usage == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_normalize_usage_canonicalizes_supported_numeric_values():
    """Test valid counts remain integers and valid cost becomes a finite float."""
    usage = normalize_usage(
        {
            "prompt_tokens": 0,
            "completion_tokens": 2,
            "total_tokens": 2,
            "cost": 1,
        }
    )

    assert usage == {
        "prompt_tokens": 0,
        "completion_tokens": 2,
        "total_tokens": 2,
        "cost": 1.0,
    }


def test_call_http_api_with_retry_populates_usage_sink_when_provided():
    """Test usage sink is filled in place from a successful response's usage block."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {
        "choices": [{"message": {"content": "Cluster 1: T cells"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50},
    }
    post_func = MagicMock(return_value=response_200)
    sink: dict = {}

    result = call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda payload: parse_chat_completions_response(payload, "OpenAI"),
        max_retries=1,
        usage_sink=sink,
    )

    assert result == ["Cluster 1: T cells"]
    assert sink == {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50}


def test_call_http_api_with_retry_default_sink_none_leaves_no_trace():
    """Test default path (no sink) returns exactly today's value and captures nothing."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {
        "choices": [{"message": {"content": "Cluster 1: T cells"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50},
    }
    post_func = MagicMock(return_value=response_200)

    result = call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda payload: parse_chat_completions_response(payload, "OpenAI"),
        max_retries=1,
    )

    assert result == ["Cluster 1: T cells"]


def test_call_http_api_with_retry_sink_untouched_when_usage_absent():
    """Test a missing usage block leaves the caller's sink empty (no KeyError)."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    post_func = MagicMock(return_value=response_200)
    sink: dict = {}

    call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda payload: parse_chat_completions_response(payload, "OpenAI"),
        max_retries=1,
        usage_sink=sink,
    )

    assert sink == {}


def test_call_http_api_with_retry_replaces_stale_usage_sink_values():
    """Test each call reports only current usage when callers reuse a sink."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {
        "choices": [{"message": {"content": "x"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
    }
    post_func = MagicMock(return_value=response_200)
    sink: dict = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.1}

    call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda payload: parse_chat_completions_response(payload, "OpenAI"),
        max_retries=1,
        usage_sink=sink,
    )

    assert sink == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}


def test_call_http_api_with_retry_clears_stale_sink_when_usage_absent():
    """Test absent usage means no current-call usage, not stale previous usage."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    post_func = MagicMock(return_value=response_200)
    sink: dict = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda payload: parse_chat_completions_response(payload, "OpenAI"),
        max_retries=1,
        usage_sink=sink,
    )

    assert sink == {}


def test_call_http_api_with_retry_clears_stale_sink_when_request_fails():
    """Test a failed current call cannot expose usage from an earlier call."""
    sink = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    post_func = MagicMock(side_effect=requests.ConnectionError("offline"))

    with pytest.raises(requests.ConnectionError, match="offline"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            max_retries=1,
            usage_sink=sink,
        )

    assert sink == {}


def test_call_http_api_with_retry_usage_parser_failure_is_non_fatal():
    """Test optional usage telemetry cannot invalidate a successful model response."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}
    post_func = MagicMock(return_value=response_200)
    usage_parser = MagicMock(side_effect=ValueError("unsupported usage shape"))
    sink = {"total_tokens": 99}

    result = call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda _payload: ["ok"],
        max_retries=3,
        usage_sink=sink,
        usage_parser=usage_parser,
    )

    assert result == ["ok"]
    assert sink == {}
    assert post_func.call_count == 1


@pytest.mark.parametrize("usage_sink", [[], set(), "usage", 1])
def test_call_http_api_with_retry_rejects_invalid_usage_sink_before_request(usage_sink):
    """Test malformed telemetry destinations cannot fail after an API side effect."""
    post_func = MagicMock()

    with pytest.raises(ValueError, match="usage_sink must be a mutable mapping"):
        call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=post_func,
            response_parser=lambda _payload: ["unused"],
            usage_sink=usage_sink,  # type: ignore[arg-type]
        )

    post_func.assert_not_called()


def test_call_http_api_with_retry_usage_sink_update_failure_is_non_fatal():
    """Test a mutable sink implementation cannot invalidate a valid response."""

    class RejectingSink(dict):
        def update(self, *args, **kwargs):
            raise RuntimeError("read-only sink")

    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {
        "ok": True,
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    sink = RejectingSink()

    with patch("mllmcelltype.providers.common.write_log") as mock_log:
        result = call_http_api_with_retry(
            provider_name="OpenAI",
            url="https://example.com",
            body={"messages": []},
            headers={"Authorization": "Bearer test"},
            post_func=MagicMock(return_value=response_200),
            response_parser=lambda _payload: ["ok"],
            usage_sink=sink,
        )

    assert result == ["ok"]
    assert sink == {}
    assert any("Failed to store token usage" in call.args[0] for call in mock_log.call_args_list)


def test_call_openai_compatible_api_surfaces_usage_to_sink():
    """Test the OpenAI-compatible wrapper threads usage through to the caller's sink."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {
        "choices": [{"message": {"content": "Cluster 1: T cells"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    post_func = MagicMock(return_value=response_200)
    sink: dict = {}

    result = call_openai_compatible_api(
        provider_name="OpenAI",
        api_key="secret-key",
        url="https://api.example.com/v1/chat/completions",
        body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "test"}]},
        post_func=post_func,
        usage_sink=sink,
    )

    assert result == ["Cluster 1: T cells"]
    assert sink["total_tokens"] == 8
