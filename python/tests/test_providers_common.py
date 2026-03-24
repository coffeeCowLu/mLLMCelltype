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
    parse_chat_completions_response,
    resolve_endpoint_url,
)


def test_parse_chat_completions_response_non_string_content():
    """Test parser coerces non-string content and still returns line list."""
    payload = {"choices": [{"message": {"content": 123}}]}
    result = parse_chat_completions_response(payload, "OpenAI")
    assert result == ["123"]


def test_build_chat_completions_body_minimal_shape():
    """Test shared body builder returns normalized minimal request payload."""
    body = build_chat_completions_body(model="gpt-5.2", prompt="genes")
    assert body == {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "genes"}],
    }


def test_build_chat_completions_body_with_optional_controls():
    """Test shared body builder includes optional generation controls."""
    body = build_chat_completions_body(
        model="MiniMax-M2.1",
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


def test_call_http_api_with_retry_normalizes_parser_output_to_strings():
    """Test parser list output is normalized to string list for downstream consistency."""
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {"ok": True}
    post_func = MagicMock(return_value=response_200)

    result = call_http_api_with_retry(
        provider_name="OpenAI",
        url="https://example.com",
        body={"messages": []},
        headers={"Authorization": "Bearer test"},
        post_func=post_func,
        response_parser=lambda _payload: [1, 2],  # type: ignore[list-item]
        max_retries=1,
    )

    assert result == ["1", "2"]


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
    response_200.json.return_value = {
        "choices": [{"message": {"content": "Cluster 1: T cells"}}]
    }
    post_func = MagicMock(return_value=response_200)

    result = call_openai_compatible_api(
        provider_name="OpenAI",
        api_key="secret-key",
        url="https://api.example.com/v1/chat/completions",
        body={"model": "gpt-5.2", "messages": [{"role": "user", "content": "test"}]},
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
    response_200.json.return_value = {
        "choices": [{"message": {"content": "Cluster 1: T cells"}}]
    }
    post_func = MagicMock(return_value=response_200)

    call_openai_compatible_api(
        provider_name="OpenRouter",
        api_key="secret-key",
        url="https://api.example.com/v1/chat/completions",
        body={"model": "openai/gpt-5.2", "messages": [{"role": "user", "content": "test"}]},
        post_func=post_func,
        extra_headers={"X-Title": "mLLMCelltype"},
    )

    called_headers = post_func.call_args.kwargs["headers"]
    assert called_headers["Authorization"] == "Bearer secret-key"
    assert called_headers["X-Title"] == "mLLMCelltype"


def test_resolve_endpoint_url_trims_custom_base_url():
    """Test custom endpoint URL is stripped before validation/use."""
    result = resolve_endpoint_url("openai", "OpenAI", "  https://proxy.example.com/v1/chat/completions  ")
    assert result == "https://proxy.example.com/v1/chat/completions"


def test_resolve_endpoint_url_blank_custom_base_url_uses_default():
    """Test blank custom URL is treated as unset and falls back to default endpoint."""
    result = resolve_endpoint_url("openai", "OpenAI", "   ")
    assert result == "https://api.openai.com/v1/chat/completions"


def test_resolve_endpoint_url_rejects_invalid_base_url_type():
    """Test non-string base URL type fails fast with clear error."""
    with pytest.raises(ValueError, match="Invalid base URL type"):
        resolve_endpoint_url("openai", "OpenAI", 123)  # type: ignore[arg-type]
