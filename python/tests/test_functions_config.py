"""Tests for provider/model resolution helpers in functions.py and config.py."""

import pytest

from mllmcelltype.config import (
    get_api_key_env_var,
    get_default_api_url,
    get_default_model,
)
from mllmcelltype.functions import get_provider


def test_get_provider_trims_whitespace():
    """Test get_provider handles model names with leading/trailing spaces."""
    assert get_provider("  gpt-5.5  ") == "openai"
    assert get_provider("  anthropic/claude-opus-4.7  ") == "openrouter"


def test_get_provider_invalid_type():
    """Test get_provider validates input type with clear error."""
    with pytest.raises(ValueError, match="Model name must be a string"):
        get_provider(123)  # type: ignore[arg-type]


def test_get_provider_empty_string():
    """Test get_provider rejects whitespace-only model names."""
    with pytest.raises(ValueError, match="cannot be empty"):
        get_provider("   ")


def test_get_provider_unknown_model_prefix():
    """Test get_provider gives clear error for unsupported model naming."""
    with pytest.raises(ValueError, match="Cannot determine provider for model"):
        get_provider("mystery-model-v1")


def test_config_provider_normalization_with_whitespace():
    """Test config lookups normalize provider names with whitespace/case."""
    assert get_default_model("  OPENAI  ") == "gpt-5.5"
    assert get_default_model("gemini") == "gemini-3.1-pro-preview"
    assert get_default_api_url("  OpenRouter ") == "https://openrouter.ai/api/v1/chat/completions"
    assert get_api_key_env_var("  openai  ") == "OPENAI_API_KEY"


def test_config_unknown_provider_fallback_is_robust():
    """Test config fallback paths handle unknown/non-string providers safely."""
    assert get_default_model("unknown_provider") == "unknown"
    assert get_default_api_url(None) == ""  # type: ignore[arg-type]
    assert get_api_key_env_var(None) == "UNKNOWN_API_KEY"  # type: ignore[arg-type]
