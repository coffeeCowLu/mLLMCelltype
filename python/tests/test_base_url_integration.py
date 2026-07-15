"""Integration tests for base_url functionality."""

from unittest.mock import MagicMock, patch

import pytest

from mllmcelltype import annotate_clusters, get_model_response, interactive_consensus_annotation


class TestAnnotateClustersBaseUrl:
    """Test base_url functionality in annotate_clusters function."""

    @patch("mllmcelltype.providers.openai.requests.post")
    def test_annotate_clusters_with_string_base_url(self, mock_post):
        """Test annotate_clusters with string base_url."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Cluster 1: T cells\nCluster 2: B cells"}}]
        }
        mock_post.return_value = mock_response

        marker_genes = {"1": ["CD3D", "CD3E"], "2": ["CD19", "MS4A1"]}

        _ = annotate_clusters(
            marker_genes=marker_genes,
            species="human",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            base_urls="https://custom-proxy.com/v1/chat/completions",
            use_cache=False,
        )

        # Verify the custom URL was used
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "https://custom-proxy.com/v1/chat/completions"

    @patch("mllmcelltype.providers.openai.requests.post")
    @patch("mllmcelltype.providers.anthropic.requests.post")
    def test_annotate_clusters_with_dict_base_urls(self, mock_anthropic_post, mock_openai_post):
        """Test annotate_clusters with dict base_urls."""
        # Mock successful response for OpenAI
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Cluster 1: T cells\nCluster 2: B cells"}}]
        }
        mock_openai_post.return_value = mock_response

        marker_genes = {"1": ["CD3D", "CD3E"], "2": ["CD19", "MS4A1"]}

        base_urls = {
            "openai": "https://openai-proxy.com/v1/chat/completions",
            "anthropic": "https://anthropic-proxy.com/v1/messages",
        }

        _ = annotate_clusters(
            marker_genes=marker_genes,
            species="human",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            base_urls=base_urls,
            use_cache=False,
        )

        # Verify the custom OpenAI URL was used
        mock_openai_post.assert_called_once()
        call_args = mock_openai_post.call_args
        assert call_args[1]["url"] == "https://openai-proxy.com/v1/chat/completions"

    @patch("mllmcelltype.providers.openai.requests.post")
    def test_annotate_clusters_with_invalid_base_url(self, mock_post):
        """Test annotate_clusters with invalid base_url."""
        marker_genes = {"1": ["CD3D", "CD3E"]}

        with pytest.raises(ValueError, match="Invalid base URL"):
            annotate_clusters(
                marker_genes=marker_genes,
                species="human",
                provider="openai",
                model="gpt-4o",
                api_key="test-key",
                base_urls="invalid-url",
                use_cache=False,
            )

    @patch("mllmcelltype.providers.openai.requests.post")
    def test_annotate_clusters_without_base_url(self, mock_post):
        """Test annotate_clusters without base_url uses default."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Cluster 1: T cells"}}]
        }
        mock_post.return_value = mock_response

        marker_genes = {"1": ["CD3D", "CD3E"]}

        _ = annotate_clusters(
            marker_genes=marker_genes,
            species="human",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            use_cache=False,
        )

        # Verify the default URL was used
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "https://api.openai.com/v1/chat/completions"


class TestGetModelResponseBaseUrl:
    """Test base_url functionality in get_model_response function."""

    @patch("mllmcelltype.providers.openai.requests.post")
    def test_get_model_response_with_base_url(self, mock_post):
        """Test get_model_response with custom base_url."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_post.return_value = mock_response

        _ = get_model_response(
            prompt="Test prompt",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            base_url="https://custom-proxy.com/v1/chat/completions",
            use_cache=False,
        )

        # Verify the custom URL was used
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "https://custom-proxy.com/v1/chat/completions"

    @patch("mllmcelltype.annotate.load_api_key")
    def test_get_model_response_rejects_invalid_base_url_before_credentials(self, mock_load_key):
        """Test invalid endpoints fail before cache or credential resolution."""
        with pytest.raises(ValueError, match="Invalid base URL"):
            get_model_response(
                prompt="Test prompt",
                provider="openai",
                model="gpt-5.5",
                api_key=None,
                base_url="invalid-url",
            )

        mock_load_key.assert_not_called()


class TestInteractiveConsensusAnnotationBaseUrl:
    """Test base_url functionality in interactive_consensus_annotation function."""

    @patch("mllmcelltype.consensus.annotate_clusters")
    def test_consensus_annotation_with_base_urls(self, mock_annotate):
        """Test interactive_consensus_annotation with base_urls."""
        # Mock annotate_clusters responses
        mock_annotate.side_effect = [
            {"1": "T cells", "2": "B cells"},  # First model
            {"1": "T cells", "2": "B cells"},  # Second model
        ]

        marker_genes = {"1": ["CD3D", "CD3E"], "2": ["CD19", "MS4A1"]}

        base_urls = {
            "openai": "https://openai-proxy.com/v1/chat/completions",
            "anthropic": "https://anthropic-proxy.com/v1/messages",
        }

        _ = interactive_consensus_annotation(
            marker_genes=marker_genes,
            species="human",
            models=["gpt-4o", "claude-3-opus"],
            api_keys={"openai": "openai-key", "anthropic": "anthropic-key"},
            base_urls=base_urls,
        )

        # Verify annotate_clusters was called with base_urls
        assert mock_annotate.call_count == 2
        for call in mock_annotate.call_args_list:
            assert "base_urls" in call[1]
            assert call[1]["base_urls"] == base_urls


class TestQwenSmartEndpointSelection:
    """Test Qwen smart endpoint selection functionality."""

    @patch("mllmcelltype.url_utils.requests.post")
    def test_qwen_smart_endpoint_selection(self, mock_post):
        """Test Qwen smart endpoint selection when no base_url is provided."""
        mock_test_response = MagicMock()
        mock_test_response.status_code = 200
        mock_qwen_response = MagicMock()
        mock_qwen_response.status_code = 200
        mock_qwen_response.json.return_value = {
            "choices": [{"message": {"content": "Cluster 1: T cells"}}]
        }

        def route_request(*args, **kwargs):
            return mock_test_response if args else mock_qwen_response

        mock_post.side_effect = route_request

        marker_genes = {"1": ["CD3D", "CD3E"]}

        _ = annotate_clusters(
            marker_genes=marker_genes,
            species="human",
            provider="qwen",
            model="qwen-max",
            api_key="test-key",
            use_cache=False,
        )

        assert mock_post.call_count == 2
        first_call_args = mock_post.call_args_list[0]
        assert "dashscope-us.aliyuncs.com" in first_call_args[0][0]
        actual_call = mock_post.call_args_list[1]
        assert (
            actual_call.kwargs["url"]
            == "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
        )

    @patch("mllmcelltype.providers.qwen.requests.post")
    def test_qwen_with_custom_base_url_skips_smart_selection(self, mock_post):
        """Test that providing custom base_url skips smart endpoint selection."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Cluster 1: T cells"}}]
        }
        mock_post.return_value = mock_response

        marker_genes = {"1": ["CD3D", "CD3E"]}

        _ = annotate_clusters(
            marker_genes=marker_genes,
            species="human",
            provider="qwen",
            model="qwen-max",
            api_key="test-key",
            base_urls="https://custom-qwen-proxy.com/v1/chat/completions",
            use_cache=False,
        )

        # Verify the custom URL was used directly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "https://custom-qwen-proxy.com/v1/chat/completions"


if __name__ == "__main__":
    pytest.main([__file__])
