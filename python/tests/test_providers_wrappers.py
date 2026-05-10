"""Tests for provider wrapper functions (request construction and routing)."""

from __future__ import annotations

import builtins
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mllmcelltype.providers.anthropic import _parse_anthropic_response, process_anthropic
from mllmcelltype.providers.common import NonRetryableProviderError
from mllmcelltype.providers.deepseek import process_deepseek
from mllmcelltype.providers.gemini import process_gemini
from mllmcelltype.providers.grok import process_grok
from mllmcelltype.providers.minimax import (
    _parse_minimax_response,
    process_minimax,
)
from mllmcelltype.providers.openai import process_openai
from mllmcelltype.providers.openrouter import process_openrouter
from mllmcelltype.providers.stepfun import process_stepfun
from mllmcelltype.providers.zhipu import process_zhipu


def _build_fake_google_modules(
    client: object,
    config_factory: MagicMock | None = None,
) -> dict[str, ModuleType]:
    """Build fake google-genai modules for deterministic Gemini tests."""
    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    types_module = ModuleType("google.genai.types")

    if config_factory is None:
        config_factory = MagicMock(return_value={"temperature": 0.7, "max_output_tokens": 4096})

    genai_module.Client = MagicMock(return_value=client)  # type: ignore[attr-defined]
    types_module.GenerateContentConfig = config_factory  # type: ignore[attr-defined]
    genai_module.types = types_module  # type: ignore[attr-defined]
    google_module.genai = genai_module  # type: ignore[attr-defined]

    return {
        "google": google_module,
        "google.genai": genai_module,
        "google.genai.types": types_module,
    }


@patch("mllmcelltype.providers.deepseek.call_openai_compatible_api")
@patch("mllmcelltype.providers.deepseek.resolve_endpoint_url")
def test_process_deepseek_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test DeepSeek wrapper sends expected body and retry parameters."""
    mock_resolve_endpoint.return_value = "https://deepseek.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_deepseek("genes", "deepseek-v4-flash", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "DeepSeek"
    assert kwargs["url"] == "https://deepseek.example/v1"
    assert kwargs["request_json"] is True
    assert kwargs["max_retries"] == 5
    assert kwargs["retry_delay"] == 3
    assert kwargs["timeout"] == 90
    assert kwargs["body"]["temperature"] == 0.7
    assert kwargs["body"]["max_tokens"] == 4096


@patch("mllmcelltype.providers.grok.call_openai_compatible_api")
@patch("mllmcelltype.providers.grok.resolve_endpoint_url")
def test_process_grok_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test Grok wrapper builds OpenAI-compatible body."""
    mock_resolve_endpoint.return_value = "https://grok.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_grok("genes", "grok-4.3", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "Grok"
    assert kwargs["body"]["model"] == "grok-4.3"
    assert kwargs["body"]["messages"][0]["content"] == "genes"


@patch("mllmcelltype.providers.openai.call_openai_compatible_api")
@patch("mllmcelltype.providers.openai.resolve_endpoint_url")
def test_process_openai_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test OpenAI wrapper uses normalized shared chat payload format."""
    mock_resolve_endpoint.return_value = "https://openai.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_openai("genes", "gpt-5.5", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "OpenAI"
    assert kwargs["body"]["model"] == "gpt-5.5"
    assert kwargs["body"]["messages"] == [{"role": "user", "content": "genes"}]


@patch("mllmcelltype.providers.openrouter.write_log")
@patch("mllmcelltype.providers.openrouter.call_openai_compatible_api")
@patch("mllmcelltype.providers.openrouter.resolve_endpoint_url")
def test_process_openrouter_warns_for_non_provider_model(
    mock_resolve_endpoint,
    mock_call_api,
    mock_write_log,
):
    """Test OpenRouter warns when model lacks provider/model format."""
    mock_resolve_endpoint.return_value = "https://openrouter.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_openrouter("genes", "gpt-5.5", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["extra_headers"]["X-Title"] == "mLLMCelltype"
    assert any(
        "may not be in the correct format" in call.args[0] and call.kwargs.get("level") == "warning"
        for call in mock_write_log.call_args_list
    )


@patch("mllmcelltype.providers.stepfun.call_openai_compatible_api")
@patch("mllmcelltype.providers.stepfun.resolve_endpoint_url")
def test_process_stepfun_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test StepFun wrapper includes expected generation controls."""
    mock_resolve_endpoint.return_value = "https://stepfun.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_stepfun("genes", "step-3.5-flash", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "StepFun"
    assert kwargs["body"]["temperature"] == 0.7
    assert kwargs["body"]["max_tokens"] == 4096


@patch("mllmcelltype.providers.zhipu.call_openai_compatible_api")
@patch("mllmcelltype.providers.zhipu.resolve_endpoint_url")
def test_process_zhipu_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test Zhipu wrapper includes expected generation controls."""
    mock_resolve_endpoint.return_value = "https://zhipu.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_zhipu("genes", "glm-5.1", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "Zhipu AI"
    assert kwargs["body"]["temperature"] == 0.7
    assert kwargs["body"]["max_tokens"] == 4096


@patch("mllmcelltype.providers.anthropic.call_http_api_with_retry")
@patch("mllmcelltype.providers.anthropic.resolve_endpoint_url")
def test_process_anthropic_alias_resolution_and_headers(
    mock_resolve_endpoint,
    mock_call_http,
):
    """Test Anthropic wrapper resolves aliases and sends expected headers."""
    mock_resolve_endpoint.return_value = "https://anthropic.example/v1/messages"
    mock_call_http.return_value = ["Cluster 1: T cells"]

    result = process_anthropic("genes", "claude-opus-latest", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_http.call_args.kwargs
    assert kwargs["body"]["model"] == "claude-opus-4-7"
    assert kwargs["headers"]["x-api-key"] == "test-key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"


def test_parse_anthropic_response_coerces_non_string():
    """Test Anthropic parser coerces non-string text payload."""
    payload = {"content": [{"text": 123}]}
    assert _parse_anthropic_response(payload) == ["123"]


@patch("mllmcelltype.providers.minimax.call_openai_compatible_api")
@patch("mllmcelltype.providers.minimax.get_working_minimax_endpoint")
def test_process_minimax_uses_smart_endpoint_by_default(
    mock_get_working_endpoint,
    mock_call_api,
):
    """Test MiniMax wrapper uses smart endpoint selection when base_url is absent."""
    mock_get_working_endpoint.return_value = "https://api.minimax.io/v1/chat/completions"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_minimax("genes", "MiniMax-M2.7", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["url"] == "https://api.minimax.io/v1/chat/completions"
    assert kwargs["non_retry_exceptions"] == (NonRetryableProviderError,)
    assert kwargs["body"]["messages"] == [{"role": "user", "name": "user", "content": "genes"}]


@patch("mllmcelltype.providers.minimax.call_openai_compatible_api")
@patch("mllmcelltype.providers.minimax.resolve_endpoint_url")
def test_process_minimax_uses_custom_base_url(
    mock_resolve_endpoint,
    mock_call_api,
):
    """Test MiniMax wrapper uses explicit base_url when provided."""
    mock_resolve_endpoint.return_value = "https://custom.example/v1/chat/completions"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_minimax(
        "genes",
        "MiniMax-M2.7",
        "test-key",
        base_url="https://custom.example/v1/chat/completions",
    )

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["url"] == "https://custom.example/v1/chat/completions"


def test_parse_minimax_response_strips_think_block():
    """Test MiniMax parser removes reasoning block and trailing commas."""
    payload = {
        "choices": [{"message": {"content": "<think>reasoning</think>\nCluster 1: T cells,\n"}}]
    }
    assert _parse_minimax_response(payload) == ["Cluster 1: T cells"]


def test_parse_minimax_response_coerces_non_string_content():
    """Test MiniMax parser coerces non-string message content."""
    payload = {
        "choices": [{"message": {"content": 123}}],
    }
    assert _parse_minimax_response(payload) == ["123"]


def test_parse_minimax_response_invalid_shape_raises_value_error():
    """Test MiniMax parser raises clear error for invalid payload shape."""
    with pytest.raises(ValueError, match="Unexpected response format from MiniMax"):
        _parse_minimax_response({"invalid": "payload"})


def test_parse_minimax_response_business_error_is_non_retryable():
    """Test MiniMax business-level error is raised as non-retryable."""
    payload = {
        "base_resp": {"status_code": 1001, "status_msg": "invalid key"},
        "choices": [{"message": {"content": "ignored"}}],
    }
    with pytest.raises(NonRetryableProviderError, match="MiniMax API error"):
        _parse_minimax_response(payload)


@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_retries_then_succeeds(mock_sleep):
    """Test Gemini retries transient errors and succeeds on a later attempt."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells,\nCluster 2: B cells,")
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_client.models.generate_content.side_effect = [RuntimeError("transient"), fake_response]

    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert result == ["Cluster 1: T cells", "Cluster 2: B cells"]
    assert fake_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(2)


def test_process_gemini_non_string_text_is_coerced():
    """Test Gemini coerces non-string response text content."""
    fake_response = SimpleNamespace(text=123)
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response)))

    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert result == ["123"]


@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_missing_text_fails_fast_without_retry(mock_sleep):
    """Test missing response text raises ValueError immediately (no useless retries)."""
    fake_response = SimpleNamespace(text=None)
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response)))

    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False), pytest.raises(
        ValueError, match="missing text content"
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert fake_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()


def test_process_gemini_import_error_message_is_clear():
    """Test missing google-genai dependency raises actionable guidance."""
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" or name.startswith("google.genai"):
            raise ImportError("no module named google")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_fake_import), pytest.raises(
        ImportError, match="google-genai"
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")


@patch("mllmcelltype.providers.gemini.write_log")
def test_process_gemini_base_url_is_ignored_with_warning(mock_write_log):
    """Test Gemini logs warning when base_url is provided (SDK path ignores it)."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells")
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response)))
    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini(
            "genes",
            "gemini-3.1-pro-preview",
            "test-key",
            base_url="https://proxy.example/v1",
        )

    assert result == ["Cluster 1: T cells"]
    assert any(
        "base_url parameter is ignored for Gemini" in call.args[0]
        and call.kwargs.get("level") == "warning"
        for call in mock_write_log.call_args_list
    )


@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_retries_exhausted_raises(mock_sleep):
    """Test Gemini raises final error after exhausting retries."""
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_client.models.generate_content.side_effect = RuntimeError("persistent failure")
    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False), pytest.raises(
        RuntimeError, match="persistent failure"
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert fake_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2
