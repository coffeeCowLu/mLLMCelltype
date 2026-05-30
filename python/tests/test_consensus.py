#!/usr/bin/env python

"""
Tests for consensus and comparison functionality in mLLMCelltype.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from mllmcelltype.config import (
    DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
    DEFAULT_FALLBACK_ENTROPY,
)
from mllmcelltype.consensus import (
    _build_interactive_result,
    _call_llm_with_retry,
    _extract_cell_type_via_llm,
    _extract_metrics_from_text,
    _merge_consensus_and_resolved,
    _normalize_api_keys,
    _normalize_consensus_model_spec,
    _run_initial_annotations,
    check_consensus,
    check_consensus_for_discussion_round,
    interactive_consensus_annotation,
    process_controversial_clusters,
)


class TestConsensus:
    """Test class for consensus functions."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_marker_genes_df, sample_marker_genes_dict):
        """Set up test fixtures."""
        self.marker_genes_df = sample_marker_genes_df
        self.marker_genes_dict = sample_marker_genes_dict

        # Sample annotations from different models
        self.model_annotations = {
            "gpt-5": {
                "1": "T cells",
                "2": "B cells",
                "3": "NK cells",
            },
            "claude-sonnet-4-6": {
                "1": "T lymphocytes",
                "2": "B lymphocytes",
                "3": "Natural killer cells",
            },
            "gemini-3.1-pro-preview": {
                "1": "CD4+ T cells",
                "2": "Plasma B cells",
                "3": "NK cells",
            },
        }

    def test_check_consensus_basic(self):
        """Test basic check_consensus function behavior."""
        # Create a simple prediction dictionary where all models have the same prediction for cluster 3
        simple_predictions = {
            "model1": {"3": "NK cells"},
            "model2": {"3": "NK cells"},
            "model3": {"3": "NK cells"},
        }

        # Test function without LLM (should fall back to simple consensus)
        consensus, consensus_proportion, entropy, controversial = check_consensus(
            predictions=simple_predictions,
            api_keys={},  # No API keys to force fallback
        )

        # Verify results
        assert isinstance(consensus, dict)
        assert isinstance(consensus_proportion, dict)
        assert isinstance(entropy, dict)
        assert isinstance(controversial, list)
        assert "3" in consensus
        assert consensus["3"] == "NK cells"
        assert consensus_proportion["3"] == 1.0  # Complete agreement, should be 1.0
        assert entropy["3"] == 0.0  # Complete agreement, entropy should be 0

    def test_build_interactive_result_includes_discussion_round_counts(self):
        """Test interactive result includes per-cluster discussion round counts."""
        result = _build_interactive_result(
            metadata={"timestamp": "2026-03-24 00:00:00"},
            discussion_logs={
                1: [{"model_a": "round 1"}, {"model_a": "round 2"}],  # type: ignore[dict-item]
                "2": [],
            },
        )

        assert result["discussion_round_counts"] == {"1": 2, "2": 0}

    def test_build_interactive_result_round_counts_include_controversial_without_logs(self):
        """Test round-count map includes explicit 0 for controversial clusters lacking logs."""
        result = _build_interactive_result(
            metadata={"timestamp": "2026-03-24 00:00:00"},
            controversial_clusters=["1", 2],  # type: ignore[list-item]
            discussion_logs={"1": [{"model_a": "round 1"}]},
        )

        assert result["controversial_clusters"] == ["1", "2"]
        assert result["discussion_round_counts"] == {"1": 1, "2": 0}

    def test_merge_consensus_and_resolved_does_not_downgrade_known_label(self):
        """Test Unknown from discussion does not overwrite a known base consensus label."""
        merged = _merge_consensus_and_resolved(
            consensus={"1": "T cells", "2": "Unknown"},
            resolved={"1": "Unknown", "2": "Unknown", "3": "Unknown", "4": "B cells"},
        )

        assert merged["1"] == "T cells"
        assert merged["2"] == "Unknown"
        assert merged["3"] == "Unknown"
        assert merged["4"] == "B cells"

    @staticmethod
    def _capture_initial_prompts(prompt_template):
        """Run the initial-annotation phase with the network stubbed; return prompts sent.

        Patches the provider registry (not the annotation function), so the real
        ``create_prompt`` runs and we observe the actual outgoing prompt text.
        """
        captured: list[str] = []

        def fake_provider_func(prompt, model, api_key, base_url):
            captured.append(prompt)
            return ["Cluster 1: T cells"]

        with patch.dict(
            "mllmcelltype.annotate.PROVIDER_FUNCTIONS",
            {"openai": fake_provider_func},
        ):
            _run_initial_annotations(
                marker_genes={"1": ["CD3D", "CD3E"]},
                species="human",
                models=[{"provider": "openai", "model": "gpt-4o"}],
                api_keys={"openai": "sk-test"},
                tissue="blood",
                additional_context=None,
                prompt_template=prompt_template,
                use_cache=False,
                force_rerun=False,
                cache_dir=None,
                base_urls=None,
                verbose=False,
            )
        return captured

    def test_custom_prompt_template_reaches_outgoing_prompt(self):
        """A custom prompt_template's text actually lands in the prompt sent to the model."""
        sentinel = "QQQ_CUSTOM_STATE_TASK_QQQ"
        custom_template = f"{sentinel}\nSpecies {{species}}, tissue {{tissue}}.\n{{markers}}"

        prompts = self._capture_initial_prompts(custom_template)

        assert prompts, "provider was never called"
        assert sentinel in prompts[0]
        # The default template's instruction must NOT survive a full override.
        assert "IN NUMERICAL ORDER" not in prompts[0]

    def test_default_prompt_template_used_when_none(self):
        """With prompt_template=None the built-in default template is used (no regression)."""
        prompts = self._capture_initial_prompts(None)

        assert prompts, "provider was never called"
        # A stable phrase from DEFAULT_PROMPT_TEMPLATE.
        assert "IN NUMERICAL ORDER" in prompts[0]

    def test_normalize_api_keys_strips_and_drops_blank(self):
        """Test api key normalization trims and removes blank entries."""
        normalized = _normalize_api_keys(
            {
                " OPENAI ": "  key-a  ",
                "anthropic": "   ",
                "qwen": None,  # type: ignore[dict-item]
            }
        )
        assert normalized == {"openai": "key-a"}

    def test_normalize_api_keys_non_string_value_raises(self):
        """Test api key values must be strings when provided."""
        with pytest.raises(ValueError, match="must be a string"):
            _normalize_api_keys({"openai": 123})  # type: ignore[dict-item]

    def test_normalize_api_keys_non_string_provider_raises(self):
        """Test api key provider names must be strings."""
        with pytest.raises(ValueError, match="provider names must be strings"):
            _normalize_api_keys({1: "key"})  # type: ignore[dict-item]

    def test_normalize_consensus_model_spec_from_string(self):
        """Test string consensus model resolves to provider+model dict."""
        result = _normalize_consensus_model_spec("  gpt-5.5  ")
        assert result == {"provider": "openai", "model": "gpt-5.5"}

    def test_normalize_consensus_model_spec_provider_only_gets_default_model(self):
        """Test provider-only dict consensus model picks default model."""
        result = _normalize_consensus_model_spec({"provider": "anthropic"})
        assert result is not None
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-opus-4-7"

    def test_normalize_consensus_model_spec_invalid_type_raises(self):
        """Test invalid consensus_model type fails fast."""
        with pytest.raises(ValueError, match="must be a string or dict"):
            _normalize_consensus_model_spec(123)  # type: ignore[arg-type]

    def test_normalize_consensus_model_spec_empty_dict_raises(self):
        """Test empty consensus_model dict fails with clear guidance."""
        with pytest.raises(ValueError, match="must include at least one of 'provider' or 'model'"):
            _normalize_consensus_model_spec({})

    def test_normalize_consensus_model_spec_unknown_provider_raises(self):
        """Test unsupported consensus_model provider fails fast."""
        with pytest.raises(ValueError, match=r"Unsupported consensus_model\.provider"):
            _normalize_consensus_model_spec({"provider": "unknown"})

    def test_normalize_consensus_model_spec_provider_model_mismatch_raises(self):
        """Test explicit provider/model mismatch is rejected."""
        with pytest.raises(ValueError, match="provider/model mismatch"):
            _normalize_consensus_model_spec(
                {
                    "provider": "openai",
                    "model": "claude-sonnet-4-6",
                }
            )

    def test_normalize_consensus_model_spec_openrouter_skips_mismatch_guard(self):
        """Test OpenRouter provider allows upstream provider model names."""
        result = _normalize_consensus_model_spec(
            {
                "provider": "openrouter",
                "model": "gpt-5.5",
            }
        )
        assert result == {"provider": "openrouter", "model": "gpt-5.5"}

    def test_check_consensus_fallback(self):
        """Test check_consensus function fallback behavior."""
        # Test with disagreement to trigger fallback logic
        disagreement_predictions = {
            "model1": {"1": "T cells", "2": "B cells"},
            "model2": {"1": "T lymphocytes", "2": "Plasma cells"},
            "model3": {"1": "CD4+ T cells", "2": "Memory B cells"},
        }

        # Test function (should use fallback consensus since no working API keys)
        consensus, consensus_proportion, entropy, controversial = check_consensus(
            predictions=disagreement_predictions,
            consensus_threshold=0.7,
            entropy_threshold=0.6,
            api_keys={},  # No API keys to force fallback
        )

        # Verify results
        assert isinstance(consensus, dict)
        assert isinstance(consensus_proportion, dict)
        assert isinstance(entropy, dict)
        assert isinstance(controversial, list)
        assert "1" in consensus
        assert "2" in consensus
        # Should have some consensus values
        assert len(consensus) == 2
        assert len(controversial) >= 0  # May or may not have controversial clusters

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_without_api_keys_skips_llm_call(self, mock_call_llm):
        """Test disagreement path skips LLM call when no API key is available."""
        predictions = {
            "model1": {"1": "T cells"},
            "model2": {"1": "T lymphocytes"},
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        mock_call_llm.assert_not_called()
        assert consensus["1"] == "T cells"
        assert cp["1"] == 0.5
        assert entropy["1"] == 1.0

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_invalid_llm_payload_falls_back(self, mock_call_llm):
        """Test non-parseable LLM consensus output falls back to deterministic majority."""
        mock_call_llm.return_value = "nonsense output without CP/H"
        predictions = {
            "model1": {"1": "T cells"},
            "model2": {"1": "B cells"},
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={"openai": "test-key"},
        )

        assert consensus["1"] == "T cells"
        assert cp["1"] == 0.5
        assert entropy["1"] == 1.0

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_llm_metrics_unknown_label_uses_majority_fallback(self, mock_call_llm):
        """Test parsed CP/H with Unknown label falls back to deterministic majority label."""
        mock_call_llm.return_value = "1\n0.88\n0.42\nUnknown"
        predictions = {
            "model1": {"1": "T cells"},
            "model2": {"1": "T cells"},
            "model3": {"1": "B cells"},
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={"openai": "test-key"},
        )

        assert consensus["1"] == "T cells"
        assert cp["1"] == 0.88
        assert entropy["1"] == 0.42

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_only_unsupported_provider_keys_skip_llm(self, mock_call_llm):
        """Test unknown provider keys do not trigger LLM calls in consensus flow."""
        predictions = {
            "model1": {"1": "T cells"},
            "model2": {"1": "B cells"},
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={"custom": "custom-key"},
        )

        mock_call_llm.assert_not_called()
        assert consensus["1"] == "T cells"
        assert cp["1"] == 0.5
        assert entropy["1"] == 1.0

    def test_check_consensus_invalid_predictions_type_raises_clear_error(self):
        """Test non-dict predictions fail fast with a clear ValueError."""
        with pytest.raises(ValueError, match="predictions must be a dict"):
            check_consensus(predictions=["not-a-dict"], api_keys={})  # type: ignore[arg-type]

    @patch("mllmcelltype.consensus.get_model_response")
    def test_call_llm_with_retry_primary_missing_key_uses_supported_fallback(self, mock_get_model):
        """Test retry helper ignores unsupported keys and uses first supported fallback."""
        mock_get_model.return_value = "fallback-response"

        response = _call_llm_with_retry(
            prompt="test prompt",
            provider="openai",
            model="gpt-5.5",
            api_key=None,
            max_retries=1,
            api_keys={"custom": "custom-key", "anthropic": "anth-key"},
        )

        assert response == "fallback-response"
        assert mock_get_model.call_count == 1
        assert mock_get_model.call_args.kwargs["provider"] == "anthropic"

    @patch("mllmcelltype.consensus.get_model_response")
    def test_call_llm_with_retry_import_error_on_primary_uses_fallback(self, mock_get_model):
        """Test ImportError on primary provider short-circuits retries and tries fallback."""

        def side_effect(**kwargs):
            if kwargs["provider"] == "openai":
                raise ImportError("openai sdk missing")
            return "fallback-ok"

        mock_get_model.side_effect = side_effect

        response = _call_llm_with_retry(
            prompt="test prompt",
            provider="openai",
            model="gpt-5.5",
            api_key="openai-key",
            max_retries=3,
            api_keys={"openai": "openai-key", "anthropic": "anth-key"},
        )

        assert response == "fallback-ok"
        assert mock_get_model.call_count == 2

    @patch("mllmcelltype.consensus.get_model_response")
    def test_call_llm_with_retry_ignores_explicit_unsupported_fallback_provider(
        self, mock_get_model
    ):
        """Test explicit unsupported fallback provider is ignored instead of called."""

        def side_effect(**kwargs):
            if kwargs["provider"] == "openai":
                raise RuntimeError("primary timeout")
            return "should-not-be-used"

        mock_get_model.side_effect = side_effect

        response = _call_llm_with_retry(
            prompt="test prompt",
            provider="openai",
            model="gpt-5.5",
            api_key="openai-key",
            max_retries=1,
            fallback_provider="custom",
            fallback_model="custom-model",
            api_keys={"openai": "openai-key", "custom": "custom-key"},
        )

        assert response is None
        assert mock_get_model.call_count == 1

    def test_check_consensus_all_annotations_empty_returns_unknown_zero_metrics(self):
        """Test clusters with only blank/None annotations become Unknown with zero metrics."""
        predictions = {
            "model1": {"1": "   "},
            "model2": {"1": None},  # type: ignore[dict-item]
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        assert consensus["1"] == "Unknown"
        assert cp["1"] == 0.0
        assert entropy["1"] == 0.0

    def test_check_consensus_blank_annotations_with_zero_thresholds_not_controversial(self):
        """Test blank-only clusters are not marked controversial when thresholds are both zero."""
        predictions = {
            "model1": {"1": "   "},
            "model2": {"1": None},  # type: ignore[dict-item]
        }
        consensus, cp, entropy, controversial = check_consensus(
            predictions=predictions,
            api_keys={},
            consensus_threshold=0.0,
            entropy_threshold=0.0,
        )
        assert consensus["1"] == "Unknown"
        assert cp["1"] == 0.0
        assert entropy["1"] == 0.0
        assert controversial == []

    def test_check_consensus_skips_invalid_model_prediction_payloads(self):
        """Test malformed model payloads are skipped instead of crashing."""
        predictions = {
            "model1": {"1": "T cells"},
            "model2": ["not", "a", "dict"],  # type: ignore[dict-item]
            "model3": None,  # type: ignore[dict-item]
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        assert consensus["1"] == "T cells"
        assert cp["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY

    def test_check_consensus_accepts_string_consensus_model_spec(self):
        """Test check_consensus accepts string consensus_model for API consistency."""
        predictions = {
            "model1": {"1": "T cells"},
            "model2": {"1": "T cells"},
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
            consensus_model="gpt-5.5",
        )

        assert consensus["1"] == "T cells"
        assert cp["1"] == 1.0
        assert entropy["1"] == 0.0

    def test_check_consensus_for_discussion_round_empty_input(self):
        """Test empty discussion round returns default consensus payload."""
        result = check_consensus_for_discussion_round(round_responses={}, api_keys={})
        assert result["reached"] is False
        assert result["consensus_proportion"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert result["entropy"] == DEFAULT_FALLBACK_ENTROPY
        assert result["majority_prediction"] == "Unknown"

    def test_check_consensus_for_discussion_round_invalid_input_type_raises(self):
        """Test non-dict discussion responses fail fast with clear error."""
        with pytest.raises(ValueError, match="round_responses must be a dict"):
            check_consensus_for_discussion_round(round_responses=["invalid"], api_keys={})  # type: ignore[arg-type]

    @patch("mllmcelltype.consensus._extract_cell_type_via_llm")
    def test_check_consensus_for_discussion_round_single_response(self, mock_extract):
        """Test single-response round uses extraction path."""
        mock_extract.return_value = "CD4+ T cells"
        result = check_consensus_for_discussion_round(
            round_responses={"modelA": "Likely CD4+ T cells"},
            api_keys={"openai": "test-key"},
        )
        assert result["reached"] is False
        assert result["majority_prediction"] == "CD4+ T cells"

    @patch("mllmcelltype.consensus._extract_cell_type_via_llm")
    def test_check_consensus_for_discussion_round_single_response_uses_structured_label_without_llm(
        self, mock_extract
    ):
        """Test single structured response avoids unnecessary LLM extraction."""
        result = check_consensus_for_discussion_round(
            round_responses={"modelA": "CELL TYPE: T cells\nGROUNDS: CD3D"},
            api_keys={},
        )
        assert result["reached"] is False
        assert result["majority_prediction"] == "T cells"
        assert result["consensus_proportion"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert result["entropy"] == DEFAULT_FALLBACK_ENTROPY
        mock_extract.assert_not_called()

    @patch("mllmcelltype.consensus._extract_cell_type_via_llm")
    def test_check_consensus_for_discussion_round_filters_error_and_blank(self, mock_extract):
        """Test discussion round ignores Error/blank responses before consensus."""
        mock_extract.return_value = "T cells"

        result = check_consensus_for_discussion_round(
            round_responses={
                "modelA": "T cells candidate",
                "modelB": "Error: upstream timeout",
                "modelC": "   ",
            },
            api_keys={"openai": "test-key"},
        )

        assert result["reached"] is False
        assert result["majority_prediction"] == "T cells"
        mock_extract.assert_called_once()

    @patch("mllmcelltype.consensus._extract_cell_type_via_llm")
    def test_check_consensus_for_discussion_round_filters_error_prefix_case_insensitively(
        self, mock_extract
    ):
        """Test discussion round ignores provider error responses regardless of case."""
        mock_extract.return_value = "T cells"
        result = check_consensus_for_discussion_round(
            round_responses={
                "modelA": "T cells candidate",
                "modelB": "error: upstream timeout",
                "modelC": "ERROR: provider unavailable",
            },
            api_keys={"openai": "test-key"},
        )
        assert result["reached"] is False
        assert result["majority_prediction"] == "T cells"
        mock_extract.assert_called_once()

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_multi_response_success(self, mock_call_llm):
        """Test multi-response round parses LLM consensus metrics."""
        mock_call_llm.return_value = "1\n0.90\n0.20\nT cells"
        result = check_consensus_for_discussion_round(
            round_responses={"modelA": "T cells", "modelB": "CD4 T cells"},
            api_keys={"openai": "test-key"},
            consensus_threshold=0.7,
            entropy_threshold=1.0,
        )
        assert result["reached"] is True
        assert result["consensus_proportion"] == 0.9
        assert result["entropy"] == 0.2
        assert result["majority_prediction"] == "T cells"

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_llm_failure_returns_default(self, mock_call_llm):
        """Test multi-response round falls back cleanly when LLM fails."""
        mock_call_llm.return_value = None
        result = check_consensus_for_discussion_round(
            round_responses={
                "modelA": "Given marker ambiguity, this could be activated lymphocytes.",
                "modelB": "Insufficient evidence for a confident subtype assignment.",
            },
            api_keys={"openai": "test-key"},
        )
        assert result["reached"] is False
        assert result["majority_prediction"] == "Unknown"

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_llm_failure_plain_label_lines_use_fallback_majority(
        self, mock_call_llm
    ):
        """Test fallback can recover consensus from plain single-line label responses."""
        mock_call_llm.return_value = None
        result = check_consensus_for_discussion_round(
            round_responses={"m1": "T cells", "m2": "B cells", "m3": "T cells"},
            api_keys={"openai": "test-key"},
            consensus_threshold=0.7,
            entropy_threshold=1.0,
        )
        assert result["majority_prediction"] == "T cells"
        assert result["consensus_proportion"] == pytest.approx(2 / 3)
        assert result["reached"] is False

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_llm_failure_multiline_plain_label_uses_fallback(
        self, mock_call_llm
    ):
        """Test fallback extracts first-line labels from multiline free-text responses."""
        mock_call_llm.return_value = None
        result = check_consensus_for_discussion_round(
            round_responses={
                "m1": "T cells\nReasoning: CD3D, IL7R",
                "m2": "- T cells\nevidence discussed above",
                "m3": "B cells\nReasoning: MS4A1, CD79A",
            },
            api_keys={"openai": "test-key"},
            consensus_threshold=0.7,
            entropy_threshold=1.0,
        )
        assert result["majority_prediction"] == "T cells"
        assert result["consensus_proportion"] == pytest.approx(2 / 3)
        assert result["reached"] is False

    @patch("mllmcelltype.consensus._extract_cell_type_via_llm")
    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_filters_unknown_like_responses(
        self, mock_call_llm, mock_extract
    ):
        """Test unknown-like responses are removed before single-response decision path."""
        mock_call_llm.return_value = "1\n0.90\n0.10\nT cells"
        mock_extract.return_value = "T cells"
        result = check_consensus_for_discussion_round(
            round_responses={"m1": "Unknown", "m2": "CELL TYPE: T cells"},
            api_keys={"openai": "test-key"},
        )
        assert result["majority_prediction"] == "T cells"
        mock_call_llm.assert_not_called()
        mock_extract.assert_not_called()

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_llm_failure_uses_structured_label_fallback(
        self, mock_call_llm
    ):
        """Test discussion fallback extracts structured CELL TYPE labels."""
        mock_call_llm.return_value = None
        result = check_consensus_for_discussion_round(
            round_responses={
                "m1": "CELL TYPE: T cells\nGROUNDS: CD3D",
                "m2": "CELL TYPE: T cells\nGROUNDS: IL7R",
                "m3": "CELL TYPE: B cells\nGROUNDS: MS4A1",
            },
            api_keys={"openai": "test-key"},
            consensus_threshold=0.7,
            entropy_threshold=1.0,
        )
        assert result["majority_prediction"] == "T cells"
        assert result["consensus_proportion"] == pytest.approx(2 / 3)
        assert result["reached"] is False

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_llm_failure_claim_fullwidth_colon(
        self, mock_call_llm
    ):
        """Test fallback supports CLAIM with fullwidth colon and bracket wrapper."""
        mock_call_llm.return_value = None
        result = check_consensus_for_discussion_round(
            round_responses={
                "m1": "CLAIM\uFF1A[T cells]\nGROUNDS: CD3D",
                "m2": "CLAIM\uFF1AT cells\nGROUNDS: IL7R",
            },
            api_keys={"openai": "test-key"},
            consensus_threshold=0.7,
            entropy_threshold=1.0,
        )
        assert result["majority_prediction"] == "T cells"
        assert result["consensus_proportion"] == 1.0
        assert result["entropy"] == 0.0
        assert result["reached"] is True

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_llm_failure_single_extracted_label(
        self, mock_call_llm
    ):
        """Test fallback returns extracted label when only one structured label is found."""
        mock_call_llm.return_value = None
        result = check_consensus_for_discussion_round(
            round_responses={
                "m1": "CELL TYPE: T cells",
                "m2": "unstructured free text without label",
            },
            api_keys={"openai": "test-key"},
        )
        assert result["majority_prediction"] == "T cells"
        assert result["consensus_proportion"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert result["entropy"] == DEFAULT_FALLBACK_ENTROPY

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_check_consensus_for_discussion_round_llm_metrics_unknown_label_uses_fallback_label(
        self, mock_call_llm
    ):
        """Test parsed CP/H with Unknown label uses structured fallback label."""
        mock_call_llm.return_value = "1\n0.80\n0.20\nUnknown"
        result = check_consensus_for_discussion_round(
            round_responses={
                "m1": "CELL TYPE: T cells\nGROUNDS: CD3D",
                "m2": "CELL TYPE: T cells\nGROUNDS: IL7R",
                "m3": "CELL TYPE: B cells\nGROUNDS: MS4A1",
            },
            api_keys={"openai": "test-key"},
        )
        assert result["majority_prediction"] == "T cells"
        assert result["consensus_proportion"] == 0.8
        assert result["entropy"] == 0.2

    def test_extract_metrics_from_text_standard_format(self):
        """Test standard 4-line format extraction."""
        text = "1\n0.80\n0.20\nT cells"
        cp, h, label = _extract_metrics_from_text(text)
        assert cp == 0.8
        assert h == 0.2
        assert label == "T cells"

    def test_extract_metrics_from_text_standard_format_unknown_label_is_dropped(self):
        """Test unknown-like labels in 4-line format are treated as missing labels."""
        text = "1\n0.80\n0.20\nunknown"
        cp, h, label = _extract_metrics_from_text(text)
        assert cp == 0.8
        assert h == 0.2
        assert label is None

    def test_extract_metrics_from_text_flexible_format(self):
        """Test flexible labeled extraction format."""
        text = (
            "Consensus reached = 1\n"
            "Consensus Proportion = 0.75\n"
            "Shannon Entropy = 0.55\n"
            "Majority cell type: CD8+ T cells"
        )
        cp, h, label = _extract_metrics_from_text(text)
        assert cp == 0.75
        assert h == 0.55
        assert label == "CD8+ T cells"

    def test_extract_metrics_from_text_json_format(self):
        """Test JSON format extraction for CP/H/majority prediction."""
        text = (
            "```json\n"
            '{"consensus_proportion": 0.91, "entropy": 0.12, "majority_prediction": "T cells"}\n'
            "```"
        )
        cp, h, label = _extract_metrics_from_text(text)
        assert cp == 0.91
        assert h == 0.12
        assert label == "T cells"

    def test_extract_metrics_from_text_json_alias_keys_and_bracket_label(self):
        """Test alias metric keys and bracket-wrapped labels in JSON are normalized."""
        cp, h, label = _extract_metrics_from_text(
            '{"cp": "0.60", "h": "0.90", "annotation": "[B cells]"}'
        )
        assert cp == 0.6
        assert h == 0.9
        assert label == "B cells"

    def test_extract_metrics_from_text_cp_h_lines_do_not_pollute_annotation(self):
        """Test CP/H labeled lines are not mis-parsed as cell-type annotations."""
        cp, h, label = _extract_metrics_from_text("CP: 0.80\nH: 0.20")
        assert cp == 0.8
        assert h == 0.2
        assert label is None

    def test_extract_metrics_from_text_claim_fullwidth_colon(self):
        """Test CLAIM fullwidth-colon format extracts a clean label."""
        cp, h, label = _extract_metrics_from_text(
            "Consensus Proportion: 0.70\nEntropy: 0.20\nCLAIM\uFF1A[Myeloid cells]"
        )
        assert cp == 0.7
        assert h == 0.2
        assert label == "Myeloid cells"

    def test_extract_metrics_from_text_blank(self):
        """Test blank text returns no parsed metrics."""
        cp, h, label = _extract_metrics_from_text("   ")
        assert cp is None
        assert h is None
        assert label is None

    @patch("mllmcelltype.consensus._call_llm_with_retry")
    def test_extract_cell_type_via_llm_rejects_overlong_label(self, mock_call_llm):
        """Test extraction rejects implausibly long labels."""
        mock_call_llm.return_value = "A" * 120
        label = _extract_cell_type_via_llm(
            text="some response",
            provider="openai",
            model="gpt-5.5",
            api_key="test-key",
            api_keys={"openai": "test-key"},
        )
        assert label is None

    def test_check_consensus_handles_non_string_annotations(self):
        """Test check_consensus tolerates non-string/None annotation values."""
        predictions = {
            "model1": {"1": None},  # type: ignore[dict-item]
            "model2": {"1": "T cells"},
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        assert consensus["1"] == "T cells"
        assert cp["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY

    def test_check_consensus_unknown_sentinel_annotations_are_excluded_from_votes(self):
        """Test unknown-like annotations are treated as missing, not valid votes."""
        predictions = {
            "model1": {"1": "unknown"},
            "model2": {"1": "Unknown (low confidence)"},
            "model3": {"1": "T cells"},
        }

        consensus, cp, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        assert consensus["1"] == "T cells"
        assert cp["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY

    @patch("mllmcelltype.consensus.get_provider")
    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.process_controversial_clusters")
    def test_interactive_consensus_annotation(
        self,
        mock_process_controversial,
        mock_check_consensus,
        mock_annotate_clusters,
        mock_get_provider,
    ):
        """Test interactive_consensus_annotation function."""
        # Setup mocks
        mock_get_provider.return_value = "openai"  # Ensure get_provider returns a valid provider
        mock_annotate_clusters.side_effect = [
            {"1": "T cells", "2": "B cells", "3": "NK cells"},
            {"1": "T lymphocytes", "2": "B lymphocytes", "3": "Natural killer cells"},
            {"1": "CD4+ T cells", "2": "Plasma B cells", "3": "NK cells"},
        ]
        mock_check_consensus.return_value = (
            {
                "1": "T cells",
                "3": "NK cells",
            },  # Consensus for non-controversial clusters
            {"1": 0.85, "2": 0.60, "3": 0.95},  # Consensus proportions
            {"1": 0.45, "2": 0.70, "3": 0.20},  # Entropy values
            ["2"],  # Controversial clusters
        )
        mock_process_controversial.return_value = (
            {"2": "B cells"},  # Resolved annotations
            {"2": ["Discussion round 1", "Discussion round 2"]},  # Discussion history
            {"2": 0.85},  # Updated consensus proportions
            {"2": 0.40},  # Updated entropy values
        )

        # Test function
        result = interactive_consensus_annotation(
            marker_genes=self.marker_genes_dict,
            species="human",
            models=[
                "gpt-5",
                "claude-sonnet-4-6",
                "gemini-3.1-pro-preview",
            ],
            api_keys={
                "openai": "test-key",
                "anthropic": "test-key",
                "gemini": "test-key",
            },
            tissue="blood",
            consensus_threshold=0.7,
            entropy_threshold=0.6,
            max_discussion_rounds=2,
            use_cache=False,
        )

        # Verify results
        assert isinstance(result, dict)
        assert "consensus" in result
        assert "consensus_proportion" in result
        assert "entropy" in result
        assert "model_annotations" in result
        assert "controversial_clusters" in result
        assert (
            "discussion_logs" in result
        )  # Note: field name is discussion_logs not discussion_history
        assert result["consensus"]["1"] == "T cells"
        assert result["consensus"]["2"] == "B cells"
        assert result["consensus"]["3"] == "NK cells"
        assert result["controversial_clusters"] == ["2"]
        assert len(result["model_annotations"]) == 3
        assert result["consensus_proportion"]["2"] == 0.85
        assert result["entropy"]["2"] == 0.40

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.process_controversial_clusters")
    def test_interactive_consensus_annotation_none_resolved_label_becomes_unknown(
        self,
        mock_process_controversial,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test None/blank resolved labels are normalized to Unknown."""
        mock_annotate_clusters.return_value = {"1": "T cells", "2": "B cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0, "2": 0.5},
            {"1": 0.0, "2": 0.8},
            ["2"],
        )
        mock_process_controversial.return_value = (
            {"2": None},
            {"2": ["round 1"]},
            {"2": 0.75},
            {"2": 0.3},
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"], "2": ["MS4A1"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            use_cache=False,
        )

        assert result["consensus"]["1"] == "T cells"
        assert result["consensus"]["2"] == "Unknown"

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.process_controversial_clusters")
    def test_interactive_consensus_annotation_inconclusive_or_error_label_becomes_unknown(
        self,
        mock_process_controversial,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test inconclusive/error resolved labels are normalized to Unknown."""
        mock_annotate_clusters.return_value = {"1": "T cells", "2": "B cells", "3": "NK cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0, "2": 0.5, "3": 0.5},
            {"1": 0.0, "2": 0.8, "3": 0.8},
            ["2", "3"],
        )
        mock_process_controversial.return_value = (
            {"2": "Inconclusive", "3": "Error: upstream timeout"},
            {"2": ["round 1"], "3": ["round 1"]},
            {"2": 0.75, "3": 0.25},
            {"2": 0.3, "3": 2.0},
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"], "2": ["MS4A1"], "3": ["NKG7"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            use_cache=False,
        )

        assert result["consensus"]["1"] == "T cells"
        assert result["consensus"]["2"] == "Unknown"
        assert result["consensus"]["3"] == "Unknown"

    @patch("mllmcelltype.consensus._extract_cell_type_via_llm")
    @patch("mllmcelltype.consensus.get_model_response")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.annotate_clusters")
    def test_interactive_consensus_annotation_single_valid_discussion_response_uses_structured_label(
        self,
        mock_annotate_clusters,
        mock_check_consensus,
        mock_get_model_response,
        mock_extract_cell_type,
    ):
        """Test full interactive chain resolves single valid structured response without LLM extraction."""
        mock_annotate_clusters.side_effect = [
            {"1": "T cells"},
            {"1": "B cells"},
        ]
        mock_check_consensus.return_value = (
            {"1": "Unknown"},
            {"1": 0.4},
            {"1": 1.3},
            ["1"],
        )

        def response_side_effect(**kwargs):
            if kwargs["provider"] == "openai":
                return "CELL TYPE: T cells\nGROUNDS: CD3D"
            raise RuntimeError("provider timeout")

        mock_get_model_response.side_effect = response_side_effect
        mock_extract_cell_type.side_effect = AssertionError("LLM extraction should not be called")

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert result["consensus"]["1"] == "T cells"
        assert result["resolved"]["1"] == "T cells"
        assert result["consensus_proportion"]["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert result["entropy"]["1"] == DEFAULT_FALLBACK_ENTROPY
        assert result["discussion_round_counts"]["1"] == 1
        mock_extract_cell_type.assert_not_called()

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.process_controversial_clusters")
    def test_interactive_consensus_annotation_unknown_resolution_does_not_overwrite_known_base(
        self,
        mock_process_controversial,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test discussion Unknown fallback does not degrade known base consensus labels."""
        mock_annotate_clusters.return_value = {"1": "T cells", "2": "B cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells", "2": "B cells"},
            {"1": 1.0, "2": 0.55},
            {"1": 0.0, "2": 0.9},
            ["2"],
        )
        mock_process_controversial.return_value = (
            {"2": "Unknown"},
            {"2": ["round 1"]},
            {"2": 0.25},
            {"2": 2.0},
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"], "2": ["MS4A1"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            use_cache=False,
        )

        assert result["consensus"]["1"] == "T cells"
        assert result["consensus"]["2"] == "B cells"

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.process_controversial_clusters")
    def test_interactive_consensus_annotation_discussion_failure_keeps_base_consensus(
        self,
        mock_process_controversial,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test discussion failure does not discard initial consensus outputs."""
        mock_annotate_clusters.return_value = {"1": "T cells", "2": "B cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells", "2": "Unknown"},
            {"1": 1.0, "2": 0.4},
            {"1": 0.0, "2": 1.2},
            ["2"],
        )
        mock_process_controversial.side_effect = RuntimeError("discussion engine timeout")

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"], "2": ["MS4A1"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            use_cache=False,
        )

        assert result["consensus"]["1"] == "T cells"
        assert result["consensus"]["2"] == "Unknown"
        assert result["resolved"] == {}
        assert result["discussion_logs"] == {}
        assert result["consensus_proportion"]["2"] == 0.4
        assert result["entropy"]["2"] == 1.2

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.load_api_key")
    def test_interactive_consensus_annotation_partial_api_keys_loads_missing_from_env(
        self,
        mock_load_api_key,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test partial api_keys are completed from env/.env for missing providers."""
        mock_load_api_key.side_effect = (
            lambda provider: "anthropic-key" if provider == "anthropic" else None
        )
        mock_annotate_clusters.side_effect = [
            {"1": "T cells"},
            {"1": "T lymphocytes"},
        ]
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 0.8},
            {"1": 0.4},
            [],
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "openai-key"},
            use_cache=False,
        )

        assert len(result["model_annotations"]) == 2
        called_providers = {call.kwargs["provider"] for call in mock_annotate_clusters.call_args_list}
        assert called_providers == {"openai", "anthropic"}
        assert any(call.args[0] == "anthropic" for call in mock_load_api_key.call_args_list)

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_interactive_consensus_annotation_recovers_from_runtime_error(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test runtime failures from one model do not crash full consensus flow."""
        mock_annotate_clusters.side_effect = [
            RuntimeError("provider failed after retries"),
            {"1": "T cells"},
        ]
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            use_cache=False,
        )

        assert len(result["model_annotations"]) == 1
        assert "anthropic:claude-sonnet-4-6" in result["model_annotations"]

    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.annotate_clusters")
    def test_interactive_consensus_annotation_all_models_fail_returns_error_payload(
        self,
        mock_annotate_clusters,
        mock_check_consensus,
    ):
        """Test all-model failure returns structured error payload."""
        mock_annotate_clusters.side_effect = RuntimeError("provider failure")

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            use_cache=False,
        )

        mock_check_consensus.assert_not_called()
        assert result["error"] == "No annotations were successful"
        assert result["consensus"] == {}
        assert result["model_annotations"] == {}

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_provider_name_is_case_insensitive(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test that mixed-case provider names still match api_keys correctly."""
        mock_annotate_clusters.side_effect = [
            {"1": "T cells"},
            {"1": "T cells"},
        ]
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[
                {"provider": "OpenAI", "model": "gpt-5.5"},
                {"provider": "ANTHROPIC", "model": "claude-sonnet-4-6"},
            ],
            api_keys={
                "openai": "test-key",
                "anthropic": "test-key",
            },
            use_cache=False,
        )

        assert len(result["model_annotations"]) == 2
        called_providers = {call.kwargs["provider"] for call in mock_annotate_clusters.call_args_list}
        assert called_providers == {"openai", "anthropic"}

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_api_key_provider_key_with_whitespace_is_normalized(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test api_keys provider keys are normalized with trim+lower."""
        mock_annotate_clusters.return_value = {"1": "T cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"  OPENAI  ": "test-key"},
            use_cache=False,
        )

        assert len(result["model_annotations"]) == 1
        assert "openai:gpt-5.5" in result["model_annotations"]

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_api_key_value_with_whitespace_is_stripped(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test api_keys values are stripped to avoid accidental auth failures."""
        mock_annotate_clusters.return_value = {"1": "T cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "  test-key  "},
            use_cache=False,
        )

        assert mock_annotate_clusters.call_args_list[0].kwargs["api_key"] == "test-key"

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_same_model_name_different_providers_not_overwritten(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test that same model name from different providers produces distinct keys."""
        mock_annotate_clusters.side_effect = [
            {"1": "T cells"},
            {"1": "B cells"},
        ]
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 0.5},
            {"1": 0.7},
            [],
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "openrouter", "model": "gpt-5.5"},
            ],
            api_keys={
                "openai": "test-key",
                "openrouter": "test-key",
            },
            use_cache=False,
        )

        # Both models must be preserved — no silent overwrite
        assert len(result["model_annotations"]) == 2
        assert "openai:gpt-5.5" in result["model_annotations"]
        assert "openrouter:gpt-5.5" in result["model_annotations"]
        # Results should differ (first call → T cells, second → B cells)
        assert (
            result["model_annotations"]["openai:gpt-5.5"]
            != result["model_annotations"]["openrouter:gpt-5.5"]
        )

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_interactive_consensus_annotation_duplicate_model_skipped(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test duplicate provider:model specs are skipped to avoid silent overwrite."""
        mock_annotate_clusters.return_value = {"1": "T cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        result = interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": " OPENAI ", "model": "gpt-5.5"},
            ],
            api_keys={"openai": "test-key"},
            use_cache=False,
        )

        assert mock_annotate_clusters.call_count == 1
        assert len(result["model_annotations"]) == 1
        assert "openai:gpt-5.5" in result["model_annotations"]

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_duplicate_participant_skipped(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test duplicate discussion participants are deduplicated before round calls."""
        mock_get_model_response.return_value = "T cells"
        mock_check_round_consensus.return_value = {
            "reached": True,
            "consensus_proportion": 0.9,
            "entropy": 0.1,
            "majority_prediction": "T cells",
        }

        results, _history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": "T cells"},
                "anthropic:claude-sonnet-4-6": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": " OPENAI ", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert mock_get_model_response.call_count == 2
        assert results["1"] == "T cells"
        assert cp["1"] == 0.9
        assert entropy["1"] == 0.1

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_normalizes_whitespace_cluster_ids(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test process_controversial_clusters trims cluster IDs for lookups and outputs."""
        mock_get_model_response.return_value = "T cells"
        mock_check_round_consensus.return_value = {
            "reached": True,
            "consensus_proportion": 0.9,
            "entropy": 0.1,
            "majority_prediction": "T cells",
        }

        results, _history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters=[" 1 "],
            model_predictions={
                "openai:gpt-5.5": {" 1 ": "T cells"},
                "anthropic:claude-sonnet-4-6": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert set(results.keys()) == {"1"}
        assert results["1"] == "T cells"
        assert cp["1"] == 0.9
        assert entropy["1"] == 0.1

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_skips_invalid_model_prediction_payloads(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test malformed model_predictions entries are skipped instead of crashing."""
        mock_get_model_response.return_value = "T cells"
        mock_check_round_consensus.return_value = {
            "reached": True,
            "consensus_proportion": 0.9,
            "entropy": 0.1,
            "majority_prediction": "T cells",
        }

        results, _history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": ["not-a-dict"],  # type: ignore[dict-item]
                "anthropic:claude-sonnet-4-6": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert results["1"] == "T cells"
        assert cp["1"] == 0.9
        assert entropy["1"] == 0.1

    @patch("mllmcelltype.consensus.load_api_key")
    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_consensus_provider_key_autoloaded(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
        mock_load_api_key,
    ):
        """Test direct discussion API autoloads consensus-model provider key from environment."""
        mock_get_model_response.return_value = "CELL TYPE: T cells\nGROUNDS: CD3D"

        captured: dict[str, object] = {}

        def check_round_side_effect(**kwargs):
            captured["api_keys"] = kwargs["api_keys"]
            captured["consensus_model"] = kwargs["consensus_model"]
            return {
                "reached": True,
                "consensus_proportion": 0.9,
                "entropy": 0.1,
                "majority_prediction": "T cells",
            }

        mock_check_round_consensus.side_effect = check_round_side_effect
        mock_load_api_key.side_effect = lambda provider: "anth-key" if provider == "anthropic" else None

        results, _history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": "T cells"},
                "zhipu:glm-5.1": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "zhipu", "model": "glm-5.1"},
            ],
            api_keys={"openai": "key-a", "zhipu": "key-z"},
            consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-6"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert results["1"] == "T cells"
        assert cp["1"] == 0.9
        assert entropy["1"] == 0.1
        assert captured["consensus_model"] == {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }
        assert isinstance(captured["api_keys"], dict)
        assert captured["api_keys"]["anthropic"] == "anth-key"  # type: ignore[index]

    def test_process_controversial_clusters_missing_markers_returns_unknown(self):
        """Test controversial cluster without marker genes returns canonical Unknown."""
        results, history, cp, entropy = process_controversial_clusters(
            marker_genes={"2": ["MS4A1"]},
            controversial_clusters=["1"],
            model_predictions={},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            use_cache=False,
        )

        assert results["1"] == "Unknown"
        assert history["1"] == []
        assert cp["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY

    def test_process_controversial_clusters_set_input_is_deterministic(self):
        """Test set-form controversial cluster IDs are processed in stable natural order."""
        results, history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters={"10", "2"},
            model_predictions={},
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            use_cache=False,
        )

        assert list(results.keys()) == ["2", "10"]
        assert list(history.keys()) == ["2", "10"]
        assert list(cp.keys()) == ["2", "10"]
        assert list(entropy.keys()) == ["2", "10"]

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_single_valid_last_round_uses_extraction(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test single valid responder path uses round-consensus extraction output."""
        def response_side_effect(**kwargs):
            if kwargs["provider"] == "openai":
                return "Likely CD4+ T cells"
            raise RuntimeError("provider timeout")

        mock_get_model_response.side_effect = response_side_effect
        mock_check_round_consensus.return_value = {
            "reached": False,
            "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
            "entropy": DEFAULT_FALLBACK_ENTROPY,
            "majority_prediction": "CD4+ T cells",
        }

        results, _history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D", "IL7R"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": "T cells"},
                "anthropic:claude-sonnet-4-6": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert mock_check_round_consensus.call_count == 1
        assert results["1"] == "CD4+ T cells"
        assert cp["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY

    @patch("mllmcelltype.consensus._extract_cell_type_via_llm")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_single_valid_structured_response_skips_llm_extraction(
        self,
        mock_get_model_response,
        mock_extract_cell_type,
    ):
        """Test single valid structured response resolves without LLM extraction call."""

        def response_side_effect(**kwargs):
            if kwargs["provider"] == "openai":
                return "CELL TYPE: T cells\nGROUNDS: CD3D, IL7R"
            raise RuntimeError("provider timeout")

        mock_get_model_response.side_effect = response_side_effect
        mock_extract_cell_type.side_effect = AssertionError("LLM extraction should not be called")

        results, history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D", "IL7R"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": "T cells"},
                "anthropic:claude-sonnet-4-6": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert results["1"] == "T cells"
        assert cp["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY
        assert len(history["1"]) == 1
        mock_extract_cell_type.assert_not_called()

    def test_process_controversial_clusters_requires_at_least_two_models(self):
        """Test discussion flow exits early when fewer than 2 runnable models are available."""
        results, history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters=["1"],
            model_predictions={"openai:gpt-5.5": {"1": "T cells"}},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "key-a"},
            use_cache=False,
        )

        assert results == {}
        assert history == {}
        assert cp == {}
        assert entropy == {}

    def test_process_controversial_clusters_invalid_controversial_clusters_string_raises(self):
        """Test string controversial_clusters is rejected to avoid char-wise splitting."""
        with pytest.raises(ValueError, match="controversial_clusters must be a list/tuple/set"):
            process_controversial_clusters(
                marker_genes={"1": ["CD3D"]},
                controversial_clusters="1",  # type: ignore[arg-type]
                model_predictions={"openai:gpt-5.5": {"1": "T cells"}},
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "key-a"},
                use_cache=False,
            )

    def test_process_controversial_clusters_invalid_model_predictions_type_raises(self):
        """Test malformed top-level model_predictions fails fast with clear error."""
        with pytest.raises(ValueError, match="model_predictions must be a dict"):
            process_controversial_clusters(
                marker_genes={"1": ["CD3D"]},
                controversial_clusters=["1"],
                model_predictions=["bad-payload"],  # type: ignore[arg-type]
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "key-a"},
                use_cache=False,
            )

    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_all_round_responses_fail_returns_unknown(
        self,
        mock_get_model_response,
    ):
        """Test discussion flow returns Unknown when all models fail every round."""
        mock_get_model_response.side_effect = RuntimeError("upstream timeout")

        results, history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D", "IL7R"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": "T cells"},
                "anthropic:claude-sonnet-4-6": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=2,
            use_cache=False,
        )

        assert mock_get_model_response.call_count == 4
        assert results["1"] == "Unknown"
        assert cp["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY
        assert len(history["1"]) == 2

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_last_round_unknown_stays_unknown(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test last-round Unknown majority remains canonical Unknown."""
        mock_get_model_response.return_value = "ambiguous response"
        mock_check_round_consensus.return_value = {
            "reached": False,
            "consensus_proportion": 0.51,
            "entropy": 1.2,
            "majority_prediction": "Unknown",
        }

        results, _history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D", "MS4A1"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": "T cells"},
                "anthropic:claude-sonnet-4-6": {"1": "B cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert results["1"] == "Unknown"
        assert cp["1"] == 0.51
        assert entropy["1"] == 1.2

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_last_round_unknown_with_context_stays_unknown(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test unknown-with-context majority remains canonical Unknown."""
        mock_get_model_response.return_value = "ambiguous response"
        mock_check_round_consensus.return_value = {
            "reached": False,
            "consensus_proportion": 0.51,
            "entropy": 1.2,
            "majority_prediction": "unknown (low confidence)",
        }

        results, _history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D", "MS4A1"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": "T cells"},
                "anthropic:claude-sonnet-4-6": {"1": "B cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert results["1"] == "Unknown"
        assert cp["1"] == 0.51
        assert entropy["1"] == 1.2

    def test_check_consensus_mixed_key_types(self):
        """Test that mixed int/str cluster keys are normalized to str."""
        # Model A returns int keys, Model B returns str keys
        mixed_predictions = {
            "model_a": {0: "T cells", 1: "B cells"},
            "model_b": {"0": "T cells", "1": "B cells"},
            "model_c": {0: "T cells", 1: "B cells"},
        }

        consensus, consensus_proportion, _entropy, _controversial = check_consensus(
            predictions=mixed_predictions,
            api_keys={},
        )

        # Should have exactly 2 clusters, not 4
        assert len(consensus) == 2
        assert "0" in consensus
        assert "1" in consensus
        # int keys should NOT appear
        assert 0 not in consensus
        assert 1 not in consensus
        # Full agreement across all 3 models
        assert consensus_proportion["0"] == 1.0

    def test_check_consensus_skips_missing_like_cluster_ids(self):
        """Test consensus normalization skips NaN/pandas.NA cluster IDs."""
        predictions = {
            "model_a": {float("nan"): "T cells", "1": "T cells"},
            "model_b": {"1": "T cells", pd.NA: "B cells"},  # type: ignore[dict-item]
        }

        consensus, consensus_proportion, _entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        assert list(consensus.keys()) == ["1"]
        assert consensus["1"] == "T cells"
        assert consensus_proportion["1"] == 1.0

    def test_check_consensus_skips_missing_like_annotation_values(self):
        """Test consensus normalization ignores NaN/pandas.NA annotation values."""
        predictions = {
            "model_a": {"1": pd.NA},  # type: ignore[dict-item]
            "model_b": {"1": "T cells"},
        }

        consensus, consensus_proportion, entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        assert consensus["1"] == "T cells"
        assert consensus_proportion["1"] == DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        assert entropy["1"] == DEFAULT_FALLBACK_ENTROPY

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_skips_missing_like_cluster_ids(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test discussion flow ignores NaN/pandas.NA cluster IDs instead of crashing."""
        mock_get_model_response.return_value = "T cells"
        mock_check_round_consensus.return_value = {
            "reached": True,
            "consensus_proportion": 0.9,
            "entropy": 0.1,
            "majority_prediction": "T cells",
        }

        results, history, cp, entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters=[pd.NA, "1"],  # type: ignore[list-item]
            model_predictions={
                "openai:gpt-5.5": {pd.NA: "Noise", "1": "T cells"},  # type: ignore[dict-item]
                "anthropic:claude-sonnet-4-6": {"1": "T cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert set(results.keys()) == {"1"}
        assert set(history.keys()) == {"1"}
        assert set(cp.keys()) == {"1"}
        assert set(entropy.keys()) == {"1"}
        assert results["1"] == "T cells"

    @patch("mllmcelltype.consensus.check_consensus_for_discussion_round")
    @patch("mllmcelltype.consensus.get_model_response")
    def test_process_controversial_clusters_omits_missing_annotations_from_initial_predictions(
        self,
        mock_get_model_response,
        mock_check_round_consensus,
    ):
        """Test missing discussion annotations do not leak as '<NA>'/None into prompts."""
        captured_prompts: list[str] = []

        def response_side_effect(**kwargs):
            captured_prompts.append(kwargs["prompt"])
            return "CELL TYPE: T cells\nGROUNDS: CD3D"

        mock_get_model_response.side_effect = response_side_effect
        mock_check_round_consensus.return_value = {
            "reached": True,
            "consensus_proportion": 0.9,
            "entropy": 0.1,
            "majority_prediction": "T cells",
        }

        results, _history, _cp, _entropy = process_controversial_clusters(
            marker_genes={"1": ["CD3D"]},
            controversial_clusters=["1"],
            model_predictions={
                "openai:gpt-5.5": {"1": pd.NA},  # type: ignore[dict-item]
                "grok:grok-4.3": {"1": "Unknown (low confidence)"},
                "anthropic:claude-sonnet-4-6": {"1": "B cells"},
            },
            species="human",
            models=[
                {"provider": "openai", "model": "gpt-5.5"},
                {"provider": "grok", "model": "grok-4.3"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
            api_keys={"openai": "key-a", "grok": "key-g", "anthropic": "key-b"},
            max_discussion_rounds=1,
            use_cache=False,
        )

        assert results["1"] == "T cells"
        assert captured_prompts
        assert "<NA>" not in captured_prompts[0]
        assert "None" not in captured_prompts[0]
        assert "openai:gpt-5.5: Unknown" not in captured_prompts[0]
        assert "grok:grok-4.3: Unknown" not in captured_prompts[0]
        assert "anthropic:claude-sonnet-4-6" in captured_prompts[0]

    def test_check_consensus_whitespace_cluster_keys_are_normalized(self):
        """Test whitespace-padded cluster keys are trimmed and merged."""
        predictions = {
            "model_a": {" 1 ": "T cells", "2": "B cells"},
            "model_b": {"1": "T cells", "2": "B cells"},
            "model_c": {"\t1\t": "T cells", "2": "B cells"},
        }

        consensus, consensus_proportion, _entropy, _controversial = check_consensus(
            predictions=predictions,
            api_keys={},
        )

        assert set(consensus.keys()) == {"1", "2"}
        assert consensus["1"] == "T cells"
        assert consensus["2"] == "B cells"
        assert consensus_proportion["1"] == 1.0
        assert consensus_proportion["2"] == 1.0

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_interactive_consensus_annotation_merges_mixed_marker_cluster_keys(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test mixed int/str marker cluster keys are merged before annotation."""
        mock_annotate_clusters.return_value = {"1": "T cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        result = interactive_consensus_annotation(
            marker_genes={1: ["CD3D"], "1": ["IL7R"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            use_cache=False,
        )

        annotate_call = mock_annotate_clusters.call_args_list[0]
        merged_marker_genes = annotate_call.kwargs["marker_genes"]
        assert list(merged_marker_genes.keys()) == ["1"]
        assert set(merged_marker_genes["1"]) == {"CD3D", "IL7R"}
        assert result["consensus"]["1"] == "T cells"

    def test_interactive_consensus_annotation_invalid_model_spec_type(self):
        """Test invalid model spec type fails with clear message."""
        with pytest.raises(ValueError, match="expected string or dict"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[123],  # type: ignore[list-item]
                api_keys={"openai": "test-key"},
                use_cache=False,
            )

    def test_interactive_consensus_annotation_invalid_provider_type_in_dict(self):
        """Test invalid provider type in model dict fails early."""
        with pytest.raises(ValueError, match="'provider' must be a string"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": 1, "model": "gpt-5.5"}],  # type: ignore[dict-item]
                api_keys={"openai": "test-key"},
                use_cache=False,
            )

    def test_interactive_consensus_annotation_invalid_provider_name_in_dict(self):
        """Test unsupported provider name fails early with clear guidance."""
        with pytest.raises(ValueError, match="unsupported provider"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "not-a-provider", "model": "foo-model"}],
                api_keys={"openai": "test-key"},
                use_cache=False,
            )

    def test_interactive_consensus_annotation_invalid_whitespace_provider_in_dict(self):
        """Test whitespace-only provider in model dict fails early."""
        with pytest.raises(ValueError, match="'provider' must be a non-empty string"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "   ", "model": "gpt-5.5"}],
                api_keys={"openai": "test-key"},
                use_cache=False,
            )

    def test_interactive_consensus_annotation_invalid_marker_genes_type(self):
        """Test invalid marker_genes type fails with clear error."""
        with pytest.raises(ValueError, match="marker_genes must be a dict"):
            interactive_consensus_annotation(
                marker_genes=["CD3D", "IL7R"],  # type: ignore[arg-type]
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "test-key"},
                use_cache=False,
            )

    def test_interactive_consensus_annotation_invalid_clusters_to_analyze_string(self):
        """Test string clusters_to_analyze is rejected to avoid char-wise splitting."""
        with pytest.raises(ValueError, match="clusters_to_analyze must be a list/tuple/set"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "test-key"},
                clusters_to_analyze="1",  # type: ignore[arg-type]
                use_cache=False,
            )

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_interactive_consensus_annotation_clusters_to_analyze_dedup_preserves_order(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test clusters_to_analyze is deduplicated while preserving user-specified order."""
        mock_annotate_clusters.return_value = {"2": "B cells", "1": "T cells"}
        mock_check_consensus.return_value = (
            {"2": "B cells", "1": "T cells"},
            {"2": 1.0, "1": 1.0},
            {"2": 0.0, "1": 0.0},
            [],
        )

        interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"], "2": ["MS4A1"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            clusters_to_analyze=["2", "2", "1"],
            use_cache=False,
        )

        filtered_marker_genes = mock_annotate_clusters.call_args.kwargs["marker_genes"]
        assert list(filtered_marker_genes.keys()) == ["2", "1"]

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_interactive_consensus_annotation_clusters_to_analyze_set_is_deterministic(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test set-form clusters_to_analyze is sorted to deterministic cluster order."""
        mock_annotate_clusters.return_value = {"2": "B cells", "10": "T cells"}
        mock_check_consensus.return_value = (
            {"2": "B cells", "10": "T cells"},
            {"2": 1.0, "10": 1.0},
            {"2": 0.0, "10": 0.0},
            [],
        )

        interactive_consensus_annotation(
            marker_genes={"10": ["CD3D"], "2": ["MS4A1"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            clusters_to_analyze={"10", "2"},
            use_cache=False,
        )

        filtered_marker_genes = mock_annotate_clusters.call_args.kwargs["marker_genes"]
        assert list(filtered_marker_genes.keys()) == ["2", "10"]

    def test_interactive_consensus_annotation_clusters_to_analyze_all_invalid_raises(self):
        """Test requesting only missing cluster IDs raises a clear validation error."""
        with pytest.raises(ValueError, match="None of the specified clusters exist"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "test-key"},
                clusters_to_analyze=["9", "10"],
                use_cache=False,
            )

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    def test_interactive_consensus_annotation_clusters_to_analyze_strips_whitespace(
        self,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test clusters_to_analyze values are stripped before matching keys."""
        mock_annotate_clusters.return_value = {"1": "T cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"], "2": ["MS4A1"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "test-key"},
            clusters_to_analyze=[" 1 "],
            use_cache=False,
        )

        filtered_marker_genes = mock_annotate_clusters.call_args.kwargs["marker_genes"]
        assert list(filtered_marker_genes.keys()) == ["1"]

    def test_interactive_consensus_annotation_clusters_to_analyze_only_empty_ids_raises(self):
        """Test empty/whitespace cluster IDs are ignored and can trigger clear validation error."""
        with pytest.raises(ValueError, match="None of the specified clusters exist"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "test-key"},
                clusters_to_analyze=[" ", "\t", ""],
                use_cache=False,
            )

    @patch("mllmcelltype.consensus.annotate_clusters")
    @patch("mllmcelltype.consensus.check_consensus")
    @patch("mllmcelltype.consensus.load_api_key")
    def test_interactive_consensus_annotation_consensus_provider_key_autoloaded(
        self,
        mock_load_api_key,
        mock_check_consensus,
        mock_annotate_clusters,
    ):
        """Test consensus model provider key is auto-loaded when missing from api_keys."""
        mock_load_api_key.side_effect = lambda provider: "anth-key" if provider == "anthropic" else None
        mock_annotate_clusters.return_value = {"1": "T cells"}
        mock_check_consensus.return_value = (
            {"1": "T cells"},
            {"1": 1.0},
            {"1": 0.0},
            [],
        )

        interactive_consensus_annotation(
            marker_genes={"1": ["CD3D"]},
            species="human",
            models=[{"provider": "openai", "model": "gpt-5.5"}],
            api_keys={"openai": "openai-key"},
            consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-6"},
            use_cache=False,
        )

        check_args = mock_check_consensus.call_args.kwargs
        assert check_args["api_keys"]["anthropic"] == "anth-key"
        assert any(call.args[0] == "anthropic" for call in mock_load_api_key.call_args_list)

    def test_interactive_consensus_annotation_invalid_api_key_value_type(self):
        """Test api_keys values must be strings to avoid silent misconfiguration."""
        with pytest.raises(ValueError, match="API key for provider 'openai' must be a string"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": 123},  # type: ignore[dict-item]
                use_cache=False,
            )

    def test_interactive_consensus_annotation_invalid_consensus_model_type(self):
        """Test consensus_model must be string or dict."""
        with pytest.raises(ValueError, match="consensus_model must be a string or dict"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "test-key"},
                consensus_model=123,  # type: ignore[arg-type]
                use_cache=False,
            )

    def test_interactive_consensus_annotation_invalid_empty_consensus_model_string(self):
        """Test empty consensus_model string fails fast."""
        with pytest.raises(ValueError, match="consensus_model string cannot be empty"):
            interactive_consensus_annotation(
                marker_genes={"1": ["CD3D"]},
                species="human",
                models=[{"provider": "openai", "model": "gpt-5.5"}],
                api_keys={"openai": "test-key"},
                consensus_model="   ",
                use_cache=False,
            )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
