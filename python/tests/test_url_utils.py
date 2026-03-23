"""Tests for URL utilities module."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from mllmcelltype.url_utils import (
    get_default_api_url,
    get_working_qwen_endpoint,
    resolve_provider_base_url,
    validate_base_url,
)


class TestResolveProviderBaseUrl:
    """Test resolve_provider_base_url function."""

    def test_resolve_with_none(self):
        """Test resolving with None base_urls."""
        result = resolve_provider_base_url("openai", None)
        assert result is None

    def test_resolve_with_string(self):
        """Test resolving with string base_urls."""
        base_url = "https://proxy.example.com/v1"
        result = resolve_provider_base_url("openai", base_url)
        assert result == base_url

    def test_resolve_with_dict_existing_provider(self):
        """Test resolving with dict base_urls for existing provider."""
        base_urls = {
            "openai": "https://openai-proxy.com/v1",
            "anthropic": "https://anthropic-proxy.com/v1",
        }
        result = resolve_provider_base_url("openai", base_urls)
        assert result == "https://openai-proxy.com/v1"

    def test_resolve_with_dict_missing_provider(self):
        """Test resolving with dict base_urls for missing provider."""
        base_urls = {"anthropic": "https://anthropic-proxy.com/v1"}
        result = resolve_provider_base_url("openai", base_urls)
        assert result is None

    def test_resolve_with_empty_dict(self):
        """Test resolving with empty dict base_urls."""
        result = resolve_provider_base_url("openai", {})
        assert result is None


class TestGetDefaultApiUrl:
    """Test get_default_api_url function."""

    def test_get_openai_url(self):
        """Test getting OpenAI default URL."""
        result = get_default_api_url("openai")
        assert result == "https://api.openai.com/v1/chat/completions"

    def test_get_anthropic_url(self):
        """Test getting Anthropic default URL."""
        result = get_default_api_url("anthropic")
        assert result == "https://api.anthropic.com/v1/messages"

    def test_get_qwen_url(self):
        """Test getting Qwen default URL."""
        result = get_default_api_url("qwen")
        assert result == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

    def test_get_deepseek_url(self):
        """Test getting DeepSeek default URL."""
        result = get_default_api_url("deepseek")
        assert result == "https://api.deepseek.com/v1/chat/completions"

    def test_get_gemini_url(self):
        """Test getting Gemini default URL."""
        result = get_default_api_url("gemini")
        assert result == "https://generativelanguage.googleapis.com/v1beta/models"

    def test_get_zhipu_url(self):
        """Test getting Zhipu default URL."""
        result = get_default_api_url("zhipu")
        assert result == "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def test_get_grok_url(self):
        """Test getting Grok default URL."""
        result = get_default_api_url("grok")
        assert result == "https://api.x.ai/v1/chat/completions"

    def test_get_openrouter_url(self):
        """Test getting OpenRouter default URL."""
        result = get_default_api_url("openrouter")
        assert result == "https://openrouter.ai/api/v1/chat/completions"

    def test_get_stepfun_url(self):
        """Test getting StepFun default URL."""
        result = get_default_api_url("stepfun")
        assert result == "https://api.stepfun.com/v1/chat/completions"

    def test_get_minimax_url(self):
        """Test getting MiniMax default URL."""
        result = get_default_api_url("minimax")
        assert result == "https://api.minimaxi.com/v1/chat/completions"

    def test_get_unknown_provider(self):
        """Test getting URL for unknown provider."""
        result = get_default_api_url("unknown_provider")
        assert result == ""


class TestValidateBaseUrl:
    """Test validate_base_url function."""

    def test_validate_https_url(self):
        """Test validating HTTPS URL."""
        assert validate_base_url("https://api.example.com/v1") is True

    def test_validate_http_url(self):
        """Test validating HTTP URL."""
        assert validate_base_url("http://localhost:8080") is True

    def test_validate_empty_string(self):
        """Test validating empty string."""
        assert validate_base_url("") is False

    def test_validate_none(self):
        """Test validating None."""
        assert validate_base_url(None) is False

    def test_validate_invalid_protocol(self):
        """Test validating URL with invalid protocol."""
        assert validate_base_url("ftp://example.com") is False

    def test_validate_no_protocol(self):
        """Test validating URL without protocol."""
        assert validate_base_url("api.example.com") is False


class TestGetWorkingQwenEndpoint:
    """Test get_working_qwen_endpoint function."""

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_international_endpoint_works(self, mock_log, mock_post):
        """Test when international endpoint is accessible."""
        # Mock successful response for international endpoint
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = get_working_qwen_endpoint("test-api-key")

        assert result == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        mock_post.assert_called_once()

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_domestic_endpoint_fallback(self, mock_log, mock_post):
        """Test fallback to domestic endpoint when international fails."""

        # Mock failed response for international, success for domestic
        def side_effect(*args, **kwargs):
            if "dashscope-intl" in args[0]:
                raise requests.RequestException("Connection failed")
            else:
                mock_response = MagicMock()
                mock_response.status_code = 200
                return mock_response

        mock_post.side_effect = side_effect

        result = get_working_qwen_endpoint("test-api-key")

        assert result == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert mock_post.call_count == 2

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_both_endpoints_fail(self, mock_log, mock_post):
        """Test when both endpoints fail."""
        # Mock failed responses for both endpoints
        mock_post.side_effect = requests.RequestException("Connection failed")

        result = get_working_qwen_endpoint("test-api-key")

        # Should return international endpoint as fallback
        assert result == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert mock_post.call_count == 2

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_auth_error_considered_accessible(self, mock_log, mock_post):
        """Test that authentication errors are considered as accessible endpoints."""
        # Mock 401 response (auth error) for international endpoint
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        result = get_working_qwen_endpoint("test-api-key")

        assert result == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        mock_post.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
