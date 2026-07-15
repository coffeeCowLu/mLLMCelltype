"""Tests for provider/model resolution helpers in functions.py and config.py."""

import inspect

import pytest

from mllmcelltype.config import (
    PROVIDER_CONFIGS,
    get_api_key_env_var,
    get_default_api_url,
    get_default_model,
)
from mllmcelltype.functions import PROVIDER_FUNCTIONS, PROVIDER_MODEL_PREFIXES, get_provider


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


def test_provider_runtime_registry_covers_every_configured_provider():
    """Test configuration and callable provider registries cannot silently drift."""
    assert set(PROVIDER_FUNCTIONS) == set(PROVIDER_CONFIGS)


def test_provider_model_prefixes_are_derived_from_config():
    """Test model detection uses the same provider metadata as defaults and URLs."""
    expected = {
        provider: config.model_prefixes
        for provider, config in PROVIDER_CONFIGS.items()
        if config.model_prefixes
    }

    assert expected == PROVIDER_MODEL_PREFIXES


def test_provider_callables_share_one_runtime_signature():
    """Test every provider supports the same orchestration arguments."""
    expected_parameters = ["prompt", "model", "api_key", "base_url", "usage_sink"]

    for provider, provider_func in PROVIDER_FUNCTIONS.items():
        parameters = list(inspect.signature(provider_func).parameters)
        assert parameters == expected_parameters, provider
