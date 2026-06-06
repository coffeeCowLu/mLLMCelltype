"""Tests for URL utilities module."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import mllmcelltype.url_utils as url_utils
from mllmcelltype.url_utils import (
    get_default_api_url,
    get_working_minimax_endpoint,
    get_working_qwen_endpoint,
    resolve_provider_base_url,
    validate_base_url,
)


@pytest.fixture(autouse=True)
def clear_smart_endpoint_cache():
    """Ensure smart endpoint cache does not leak across tests."""
    url_utils._SMART_ENDPOINT_CACHE.clear()
    yield
    url_utils._SMART_ENDPOINT_CACHE.clear()


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

    def test_resolve_with_string_trims_whitespace(self):
        """Test resolving with string base_urls trims spaces and trailing slashes."""
        result = resolve_provider_base_url("openai", "  https://proxy.example.com/v1/  ")
        assert result == "https://proxy.example.com/v1"

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

    def test_resolve_with_dict_mixed_case_provider_key(self):
        """Test case-insensitive matching for mixed-case dict keys."""
        base_urls = {"OpenAI": "https://openai-proxy.com/v1/chat/completions"}
        result = resolve_provider_base_url("openai", base_urls)
        assert result == "https://openai-proxy.com/v1/chat/completions"

    def test_resolve_with_whitespace_provider_name(self):
        """Test provider name with whitespace is normalized."""
        base_urls = {"openai": "https://openai-proxy.com/v1/chat/completions"}
        result = resolve_provider_base_url("  OPENAI  ", base_urls)
        assert result == "https://openai-proxy.com/v1/chat/completions"

    def test_resolve_with_whitespace_provider_key(self):
        """Test provider key with whitespace in dict is normalized."""
        base_urls = {"  openai  ": "https://openai-proxy.com/v1/chat/completions/"}
        result = resolve_provider_base_url("openai", base_urls)
        assert result == "https://openai-proxy.com/v1/chat/completions"

    def test_resolve_with_empty_dict(self):
        """Test resolving with empty dict base_urls."""
        result = resolve_provider_base_url("openai", {})
        assert result is None

    def test_resolve_with_non_string_value_raises(self):
        """Test resolving with non-string base URL value raises clear error."""
        with pytest.raises(ValueError, match="must be a string URL"):
            resolve_provider_base_url("openai", {"openai": 123})  # type: ignore[dict-item]

    def test_resolve_with_invalid_base_url_string_raises(self):
        """Test invalid URL string is rejected early with clear error."""
        with pytest.raises(ValueError, match="Invalid base URL"):
            resolve_provider_base_url("openai", "not-a-url")

    def test_resolve_with_invalid_target_provider_url_in_dict_raises(self):
        """Test provider-specific invalid URL is rejected early."""
        with pytest.raises(ValueError, match="Invalid base URL"):
            resolve_provider_base_url("openai", {"openai": "notaurl"})

    def test_resolve_with_invalid_base_urls_type_raises(self):
        """Test unsupported base_urls container types fail fast."""
        with pytest.raises(ValueError, match="must be a string, dict, or None"):
            resolve_provider_base_url("openai", ["https://proxy.example.com"])  # type: ignore[arg-type]


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
        assert result == "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"

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
        assert result == "https://api.z.ai/api/paas/v4/chat/completions"

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
        assert result == "https://api.minimax.io/v1/chat/completions"

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

    def test_validate_missing_hostname(self):
        """Test validating URL with scheme but no hostname."""
        assert validate_base_url("https:///v1/chat/completions") is False

    def test_validate_url_with_credentials(self):
        """Test rejecting URLs embedding credentials."""
        assert validate_base_url("https://user:pass@example.com/v1/chat/completions") is False

    def test_validate_url_with_fragment(self):
        """Test rejecting URLs containing fragments."""
        assert validate_base_url("https://api.example.com/v1#fragment") is False

    def test_validate_url_with_query(self):
        """Test rejecting URLs containing query parameters."""
        assert validate_base_url("https://api.example.com/v1?token=abc") is False


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

        assert result == "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
        mock_post.assert_called_once()

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_domestic_endpoint_fallback(self, mock_log, mock_post):
        """Test fallback to domestic endpoint when international fails."""

        # Mock failed response for international, success for domestic
        def side_effect(*args, **kwargs):
            if "dashscope-us" in args[0]:
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
        assert result == "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert mock_post.call_count == 3

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_auth_error_tries_next_region(self, mock_log, mock_post):
        """Test that regional auth errors do not lock endpoint selection."""

        def side_effect(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 401 if "dashscope-us" in args[0] else 200
            return mock_response

        mock_post.side_effect = side_effect

        result = get_working_qwen_endpoint("test-api-key")

        assert result == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert mock_post.call_count == 2

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_endpoint_selection_is_cached_per_api_key(self, mock_log, mock_post):
        """Test smart endpoint probing is cached for repeated calls with same key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        first = get_working_qwen_endpoint("test-api-key")
        second = get_working_qwen_endpoint("test-api-key")

        assert first == second
        assert mock_post.call_count == 1


class TestGetWorkingMiniMaxEndpoint:
    """Test get_working_minimax_endpoint function."""

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_international_endpoint_works(self, mock_log, mock_post):
        """Test when international endpoint is accepted."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"base_resp": {"status_code": 0, "status_msg": ""}}
        mock_post.return_value = mock_response

        result = get_working_minimax_endpoint("test-api-key")

        assert result == "https://api.minimax.io/v1/chat/completions"
        mock_post.assert_called_once()

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_fallback_to_domestic_when_international_rejected(self, mock_log, mock_post):
        """Test fallback to domestic endpoint when international rejects key."""
        intl_response = MagicMock()
        intl_response.status_code = 200
        intl_response.json.return_value = {
            "base_resp": {"status_code": 1001, "status_msg": "Invalid API key"}
        }

        domestic_response = MagicMock()
        domestic_response.status_code = 200
        domestic_response.json.return_value = {"base_resp": {"status_code": 0, "status_msg": ""}}

        mock_post.side_effect = [intl_response, domestic_response]

        result = get_working_minimax_endpoint("test-api-key")

        assert result == "https://api.minimaxi.com/v1/chat/completions"
        assert mock_post.call_count == 2

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_both_endpoints_fail(self, mock_log, mock_post):
        """Test fallback when both endpoints are unreachable."""
        mock_post.side_effect = requests.RequestException("Connection failed")

        result = get_working_minimax_endpoint("test-api-key")

        assert result == "https://api.minimax.io/v1/chat/completions"
        assert mock_post.call_count == 2

    @patch("mllmcelltype.url_utils.requests.post")
    @patch("mllmcelltype.url_utils.write_log")
    def test_endpoint_selection_is_cached_per_api_key(self, mock_log, mock_post):
        """Test MiniMax endpoint probing is cached for repeated calls with same key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"base_resp": {"status_code": 0, "status_msg": ""}}
        mock_post.return_value = mock_response

        first = get_working_minimax_endpoint("test-api-key")
        second = get_working_minimax_endpoint("test-api-key")

        assert first == second
        assert mock_post.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__])
