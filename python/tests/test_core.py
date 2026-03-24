#!/usr/bin/env python

"""
Core functionality tests for mLLMCelltype.
Tests for utility functions and core features that don't depend on external APIs.
"""

import os
import tempfile
import time
from unittest.mock import patch

import pandas as pd
import pytest

# Import utility functions
from mllmcelltype.utils import (
    clear_cache,
    create_cache_key,
    format_results,
    get_cache_stats,
    is_unknown_annotation,
    load_api_key,
    load_from_cache,
    normalize_annotation,
    normalize_marker_genes_keys,
    parse_marker_genes,
    save_to_cache,
)


# Test parse_marker_genes function
def test_parse_marker_genes(sample_marker_genes_df):
    """Test parsing marker genes from DataFrame to dictionary."""
    parsed = parse_marker_genes(sample_marker_genes_df)

    assert isinstance(parsed, dict)
    assert "1" in parsed
    assert "2" in parsed
    assert len(parsed["1"]) == 3
    assert len(parsed["2"]) == 3
    assert "CD3D" in parsed["1"]
    assert "CD19" in parsed["2"]


def test_parse_marker_genes_empty():
    """Test parsing empty marker genes DataFrame."""
    empty_df = pd.DataFrame()
    parsed = parse_marker_genes(empty_df)

    assert isinstance(parsed, dict)
    assert len(parsed) == 0


def test_parse_marker_genes_missing_columns():
    """Test parsing marker genes DataFrame with missing columns."""
    # Missing 'gene' column
    df = pd.DataFrame({"cluster": [1, 2, 3]})

    with pytest.raises(ValueError, match="'gene' column not found"):
        parse_marker_genes(df)

    # Missing 'cluster' column
    df = pd.DataFrame({"gene": ["CD3D", "CD19"]})

    with pytest.raises(ValueError, match="'cluster' column not found"):
        parse_marker_genes(df)


def test_parse_marker_genes_column_names_strip_whitespace():
    """Test parse_marker_genes tolerates common CSV header whitespace."""
    df = pd.DataFrame(
        {
            " cluster ": [1, 1, 2],
            " gene ": ["CD3D", "IL7R", "MS4A1"],
        }
    )

    parsed = parse_marker_genes(df)

    assert parsed == {"1": ["CD3D", "IL7R"], "2": ["MS4A1"]}


def test_parse_marker_genes_mixed_cluster_key_types_are_merged():
    """Test mixed int/str cluster IDs are merged after string normalization."""
    df = pd.DataFrame(
        {
            "cluster": [1, "1"],
            "gene": ["CD3D", "IL7R"],
        }
    )

    parsed = parse_marker_genes(df)

    assert list(parsed.keys()) == ["1"]
    assert parsed["1"] == ["CD3D", "IL7R"]


def test_parse_marker_genes_strips_cluster_id_whitespace():
    """Test parse_marker_genes trims cluster IDs and merges collisions."""
    df = pd.DataFrame(
        {
            "cluster": [" 1 ", "1"],
            "gene": ["CD3D", "IL7R"],
        }
    )

    parsed = parse_marker_genes(df)

    assert list(parsed.keys()) == ["1"]
    assert parsed["1"] == ["CD3D", "IL7R"]


def test_parse_marker_genes_mixed_incomparable_cluster_types():
    """Test parse_marker_genes handles incomparable cluster types without sorting crash."""
    df = pd.DataFrame(
        {
            "cluster": [1, (2,), "3"],
            "gene": ["CD3D", "IL7R", "MS4A1"],
        }
    )

    parsed = parse_marker_genes(df)

    assert parsed["1"] == ["CD3D"]
    assert parsed["(2,)"] == ["IL7R"]
    assert parsed["3"] == ["MS4A1"]


def test_parse_marker_genes_unhashable_cluster_values_do_not_crash():
    """Test parse_marker_genes handles unhashable cluster values by robust row-wise parsing."""
    df = pd.DataFrame(
        {
            "cluster": [[1], [1], "2"],
            "gene": ["CD3D", "IL7R", "MS4A1"],
        }
    )

    parsed = parse_marker_genes(df)

    assert parsed["[1]"] == ["CD3D", "IL7R"]
    assert parsed["2"] == ["MS4A1"]


def test_parse_marker_genes_invalid_input_type_raises():
    """Test parse_marker_genes rejects non-DataFrame input with clear message."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        parse_marker_genes({"cluster": ["1"], "gene": ["CD3D"]})  # type: ignore[arg-type]


def test_parse_marker_genes_preserves_cluster_with_empty_gene_rows():
    """Test clusters with explicit rows but no valid genes are preserved as empty lists."""
    df = pd.DataFrame(
        {
            "cluster": ["1", "2", "2"],
            "gene": ["CD3D", None, ""],
        }
    )

    parsed = parse_marker_genes(df)

    assert parsed["1"] == ["CD3D"]
    assert parsed["2"] == []


def test_parse_marker_genes_skips_invalid_cluster_ids():
    """Test parse_marker_genes skips rows with invalid/blank cluster IDs."""
    df = pd.DataFrame(
        {
            "cluster": ["1", "   ", None],  # type: ignore[list-item]
            "gene": ["CD3D", "IL7R", "MS4A1"],
        }
    )

    parsed = parse_marker_genes(df)

    assert parsed == {"1": ["CD3D"]}


def test_parse_marker_genes_skips_pandas_na_cluster_and_gene_values():
    """Test parse_marker_genes treats pandas.NA in cluster/gene columns as missing."""
    df = pd.DataFrame(
        {
            "cluster": [pd.NA, "1", "1"],
            "gene": ["CD3D", pd.NA, "IL7R"],
        }
    )

    parsed = parse_marker_genes(df)

    assert parsed == {"1": ["IL7R"]}


def test_normalize_marker_genes_keys_strips_and_skips_invalid_cluster_ids():
    """Test dict marker keys are normalized and invalid IDs are skipped."""
    parsed = normalize_marker_genes_keys(
        {
            " 1 ": ["CD3D"],
            1: ["IL7R"],
            "   ": ["MS4A1"],
            None: ["NKG7"],  # type: ignore[dict-item]
        }
    )

    assert parsed == {"1": ["CD3D", "IL7R"]}


def test_normalize_marker_genes_keys_skips_pandas_na_cluster_and_gene_values():
    """Test dict marker normalization skips pandas.NA cluster keys and gene entries."""
    parsed = normalize_marker_genes_keys(
        {
            pd.NA: ["CD3D"],  # type: ignore[dict-item]
            "1": [pd.NA, "IL7R"],
        }
    )

    assert parsed == {"1": ["IL7R"]}


# Test load_api_key function
def test_load_api_key_from_env(mock_env_with_api_keys):
    """Test loading API key from environment variables."""
    key = load_api_key("openai")
    assert key == os.environ["OPENAI_API_KEY"]

    key = load_api_key("anthropic")
    assert key == os.environ["ANTHROPIC_API_KEY"]


def test_load_api_key_from_env_trims_whitespace():
    """Test loading API key trims accidental surrounding whitespace."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "  test-key-123  "}, clear=True):
        key = load_api_key("openai")
        assert key == "test-key-123"


def test_load_api_key_from_dotenv_trims_whitespace_and_quotes():
    """Test .env-loaded keys are normalized to avoid auth failures."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = os.path.join(temp_dir, ".env")
        with open(env_file, "w") as f:
            f.write('OPENAI_API_KEY="  test-dotenv-key  "\n')

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            with patch.dict(os.environ, {}, clear=True):
                key = load_api_key("openai")
                assert key == "test-dotenv-key"
        finally:
            os.chdir(original_cwd)


def test_load_api_key_gemini_supports_google_api_key_alias():
    """Test Gemini provider accepts legacy GOOGLE_API_KEY env var."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "legacy-google-key"}, clear=True):
        key = load_api_key("gemini")
        assert key == "legacy-google-key"


def test_load_api_key_missing():
    """Test loading missing API key."""
    # Ensure the environment variable doesn't exist
    with patch.dict(os.environ, {}, clear=True):
        key = load_api_key("nonexistent")
        assert key is None or key == ""


def test_load_api_key_unknown_provider():
    """Test loading API key for unknown provider."""
    with patch.dict(os.environ, {"CUSTOM_API_KEY": "test-key-456"}):
        key = load_api_key("custom")
        # For unknown providers, it tries to find PROVIDER_API_KEY in environment
        assert key == "test-key-456"


# Test cache functions
def test_create_cache_key():
    """Test creating cache key."""
    key1 = create_cache_key("test prompt", "gpt-5.2", "openai")
    key2 = create_cache_key("test prompt", "gpt-5.2", "openai")
    key3 = create_cache_key("different prompt", "gpt-5.2", "openai")

    assert key1 == key2  # Same inputs should produce same key
    assert key1 != key3  # Different inputs should produce different keys
    assert isinstance(key1, str)
    assert len(key1) > 0


def test_create_cache_key_with_non_string_base_url():
    """Test create_cache_key handles non-string base_url robustly."""
    key = create_cache_key("test prompt", "gpt-5.2", "openai", base_url=123)  # type: ignore[arg-type]
    assert isinstance(key, str)
    assert len(key) > 0


def test_save_and_load_from_cache():
    """Test saving to and loading from cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with dictionary
        data_dict = {"1": "T cells", "2": "B cells"}
        key = create_cache_key("test prompt", "gpt-5.2", "openai")

        save_to_cache(key, data_dict, cache_dir=temp_dir)
        loaded = load_from_cache(key, cache_dir=temp_dir)

        assert loaded == data_dict

        # Test with list
        data_list = ["T cells", "B cells"]
        key = create_cache_key("test prompt 2", "gpt-5.2", "openai")

        save_to_cache(key, data_list, cache_dir=temp_dir)
        loaded = load_from_cache(key, cache_dir=temp_dir)

        assert loaded == data_list


def test_load_from_nonexistent_cache():
    """Test loading from nonexistent cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        key = create_cache_key("nonexistent", "gpt-5.2", "openai")
        loaded = load_from_cache(key, cache_dir=temp_dir)

        assert loaded is None


def test_load_from_cache_legacy_format_payload():
    """Test legacy cache files (raw payload without metadata) still load."""
    with tempfile.TemporaryDirectory() as temp_dir:
        key = create_cache_key("legacy", "gpt-5.2", "openai")
        cache_file = os.path.join(temp_dir, f"{key}.json")
        with open(cache_file, "w") as f:
            f.write('{"1": "T cells"}')

        loaded = load_from_cache(key, cache_dir=temp_dir)
        assert loaded == {"1": "T cells"}


def test_clear_cache_older_than_handles_invalid_json():
    """Test clear_cache(older_than=...) gracefully handles corrupted cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        bad_cache_file = os.path.join(temp_dir, "bad.json")
        with open(bad_cache_file, "w") as f:
            f.write("{invalid json")
        os.utime(bad_cache_file, (time.time() - 10, time.time() - 10))

        removed = clear_cache(cache_dir=temp_dir, older_than=1)
        assert removed == 1
        assert not os.path.exists(bad_cache_file)


def test_clear_cache_no_directory_returns_zero():
    """Test clear_cache returns 0 when cache directory does not exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_dir = os.path.join(temp_dir, "missing-cache")
        assert clear_cache(cache_dir=missing_dir) == 0


def test_clear_cache_remove_all_json_only():
    """Test clear_cache removes only JSON cache files in remove-all mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        json_file = os.path.join(temp_dir, "cache-a.json")
        txt_file = os.path.join(temp_dir, "readme.txt")
        with open(json_file, "w") as f:
            f.write("{}")
        with open(txt_file, "w") as f:
            f.write("keep me")

        removed = clear_cache(cache_dir=temp_dir)

        assert removed == 1
        assert not os.path.exists(json_file)
        assert os.path.exists(txt_file)


def test_clear_cache_older_than_keeps_recent_entries():
    """Test age-based cleanup removes old cache but keeps recent cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_file = os.path.join(temp_dir, "old.json")
        new_file = os.path.join(temp_dir, "new.json")
        now = time.time()

        with open(old_file, "w") as f:
            f.write(f'{{"version":"1.0","timestamp":{now - 1000},"data":"old"}}')
        with open(new_file, "w") as f:
            f.write(f'{{"version":"1.0","timestamp":{now - 5},"data":"new"}}')

        removed = clear_cache(cache_dir=temp_dir, older_than=60)

        assert removed == 1
        assert not os.path.exists(old_file)
        assert os.path.exists(new_file)


def test_get_cache_stats_nonexistent_directory():
    """Test cache stats for a missing directory returns clear no-cache payload."""
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_dir = os.path.join(temp_dir, "missing-cache")
        stats = get_cache_stats(cache_dir=missing_dir)

        assert stats["exists"] is False
        assert stats["count"] == 0
        assert stats["status"] == "No cache directory"


def test_get_cache_stats_empty_directory():
    """Test empty cache directory detailed stats are stable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        stats = get_cache_stats(cache_dir=temp_dir, detailed=True)
        assert stats["exists"] is True
        assert stats["count"] == 0
        assert stats["status"] == "Empty cache"
        assert stats["valid_files"] == 0
        assert stats["invalid_files"] == 0


def test_get_cache_stats_mixed_formats_and_invalid_file():
    """Test detailed cache stats classify legacy/new/unknown/invalid cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        now = time.time()
        with open(os.path.join(temp_dir, "new.json"), "w") as f:
            f.write(f'{{"version":"1.0","timestamp":{now - 10},"data":{{"1":"T cells"}}}}')
        with open(os.path.join(temp_dir, "legacy.json"), "w") as f:
            f.write('{"1":"B cells"}')
        with open(os.path.join(temp_dir, "unknown.json"), "w") as f:
            f.write('["unexpected","list"]')
        with open(os.path.join(temp_dir, "broken.json"), "w") as f:
            f.write("{broken")

        stats = get_cache_stats(cache_dir=temp_dir, detailed=True)

        assert stats["exists"] is True
        assert stats["count"] == 4
        assert stats["valid_files"] == 3
        assert stats["invalid_files"] == 1
        assert stats["format_counts"]["1.0"] == 1
        assert stats["format_counts"]["legacy"] == 1
        assert stats["format_counts"]["unknown"] == 1
        assert stats["oldest"] is not None
        assert stats["newest"] is not None


# Test format_results function
def test_format_results_simple():
    """Test formatting simple results."""
    results = ["Cluster 1: T cells", "Cluster 2: B cells"]
    clusters = ["1", "2"]

    formatted = format_results(results, clusters)

    assert isinstance(formatted, dict)
    assert "1" in formatted
    assert "2" in formatted
    assert formatted["1"] == "T cells"
    assert formatted["2"] == "B cells"


def test_format_results_complex():
    """Test formatting complex results with different formats."""
    results = ["1. T cells", "Cluster 2 - B cells", "3: NK cells"]
    clusters = ["1", "2", "3"]

    formatted = format_results(results, clusters)

    assert isinstance(formatted, dict)
    assert "1" in formatted
    assert "2" in formatted
    assert "3" in formatted
    assert formatted["1"] == "T cells"
    assert formatted["2"] == "B cells"
    assert formatted["3"] == "NK cells"


def test_format_results_mismatched():
    """Test formatting results with mismatched clusters."""
    results = ["Cluster 1: T cells", "Cluster 2: B cells"]
    clusters = ["1", "2", "3"]

    formatted = format_results(results, clusters)

    assert isinstance(formatted, dict)
    assert "1" in formatted
    assert "2" in formatted
    # The function adds "Unknown" for missing clusters
    assert "3" in formatted
    # Line-by-line mapping strips "Cluster X:" prefix when present
    assert formatted["1"] == "T cells"
    assert formatted["2"] == "B cells"
    assert formatted["3"] == "Unknown"


# Test format_results with JSON responses
def test_format_results_json_with_markers():
    """Test parsing JSON response with code block markers."""
    json_response = [
        "```json",
        "{",
        '  "annotations": [',
        '    {"cluster": "1", "cell_type": "T cells"},',
        '    {"cluster": "2", "cell_type": "B cells"},',
        '    {"cluster": "3", "cell_type": "Monocytes"}',
        "  ]",
        "}",
        "```",
    ]
    clusters = ["1", "2", "3"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells", "3": "Monocytes"}


def test_format_results_json_without_markers():
    """Test parsing JSON response without code block markers."""
    json_response = [
        "{",
        '  "annotations": [',
        '    {"cluster": "1", "cell_type": "T cells"},',
        '    {"cluster": "2", "cell_type": "B cells"}',
        "  ]",
        "}",
    ]
    clusters = ["1", "2"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells"}


def test_format_results_dict_input():
    """Test formatting results when cache contains direct dictionary output."""
    results = {"1": "T cells", "2": "B cells"}
    clusters = ["1", "2", "3"]

    formatted = format_results(results, clusters)

    assert formatted == {"1": "T cells", "2": "B cells", "3": "Unknown"}


def test_format_results_dict_input_supports_cluster_alias_mapping():
    """Test dict results map to prefixed cluster IDs via alias candidates."""
    results = {"1": "T cells"}
    clusters = ["Cluster_1"]

    formatted = format_results(results, clusters)

    assert formatted == {"Cluster_1": "T cells"}


def test_format_results_dict_input_supports_reverse_cluster_alias_mapping():
    """Test dict key 'Cluster_1' maps back to requested cluster '1'."""
    results = {"Cluster_1": "T cells"}
    clusters = ["1"]

    formatted = format_results(results, clusters)

    assert formatted == {"1": "T cells"}


def test_format_results_preserves_exact_cluster_id_over_alias_collision_dict():
    """Test exact cluster IDs are not shadowed by alias expansion collisions."""
    results = {"Cluster_1": "B cells", "1": "T cells"}
    clusters = ["1", "Cluster_1"]

    formatted = format_results(results, clusters)

    assert formatted == {"1": "T cells", "Cluster_1": "B cells"}


def test_format_results_preserves_exact_cluster_id_over_alias_collision_labeled():
    """Test labeled lines keep exact cluster IDs when alias names overlap."""
    results = ["Cluster_1: B cells", "1: T cells"]
    clusters = ["1", "Cluster_1"]

    formatted = format_results(results, clusters)

    assert formatted == {"1": "T cells", "Cluster_1": "B cells"}


def test_format_results_case_variant_cluster_ids_preserve_exact_dict_mapping():
    """Test exact IDs remain stable when requested clusters differ only by case."""
    results = {"A": "T cells", "a": "B cells"}
    clusters = ["A", "a"]

    formatted = format_results(results, clusters)

    assert formatted == {"A": "T cells", "a": "B cells"}


def test_format_results_case_variant_cluster_ids_preserve_exact_labeled_mapping():
    """Test labeled parsing keeps exact case-variant cluster IDs distinct."""
    results = ["A: T cells", "a: B cells"]
    clusters = ["A", "a"]

    formatted = format_results(results, clusters)

    assert formatted == {"A": "T cells", "a": "B cells"}


def test_format_results_dict_input_strips_whitespace_keys():
    """Test dict keys with surrounding whitespace are normalized."""
    results = {" 1 ": "T cells"}
    clusters = ["1"]

    formatted = format_results(results, clusters)

    assert formatted == {"1": "T cells"}


def test_format_results_dict_input_normalized_key_collision_prefers_known_over_unknown():
    """Test normalized dict-key collisions keep known labels over Unknown sentinels."""
    clusters = ["1"]

    formatted_a = format_results({1: "T cells", "1": "Unknown"}, clusters)
    formatted_b = format_results({"1": "Unknown", 1: "T cells"}, clusters)
    formatted_c = format_results({" 1 ": "Unknown", "1": "T cells"}, clusters)
    formatted_d = format_results({" 1 ": "T cells", "1": "Unknown"}, clusters)

    assert formatted_a == {"1": "T cells"}
    assert formatted_b == {"1": "T cells"}
    assert formatted_c == {"1": "T cells"}
    assert formatted_d == {"1": "T cells"}


def test_format_results_json_direct_mapping():
    """Test parsing direct JSON mapping format."""
    json_response = ['{"1": "T cells", "2": "B cells"}']
    clusters = ["1", "2", "3"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells", "3": "Unknown"}


def test_format_results_json_direct_mapping_supports_cluster_alias_keys():
    """Test JSON direct mapping accepts Cluster_*/cluster * keys for numeric cluster ids."""
    json_response = ['{"Cluster_1": "T cells", "cluster 2": "B cells"}']
    clusters = ["1", "2"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells"}


def test_format_results_json_annotations_payload_supports_cluster_alias_ids():
    """Test JSON annotations list maps cluster aliases to requested ids."""
    json_response = [
        "{",
        '  "annotations": [',
        '    {"cluster": "Cluster_1", "cell_type": "T cells"},',
        '    {"cluster": "cluster 2", "annotation": "B cells"}',
        "  ]",
        "}",
    ]
    clusters = ["1", "2"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells"}


def test_format_results_json_list_of_objects():
    """Test parsing JSON list format with cluster/cell_type objects."""
    json_response = [
        "[",
        '  {"cluster": "1", "cell_type": "T cells"},',
        '  {"cluster": "2", "cell_type": "B cells"}',
        "]",
    ]
    clusters = ["1", "2"]
    result = format_results(json_response, clusters)
    assert result == {"1": "T cells", "2": "B cells"}


def test_format_results_labeled_duplicate_unknown_then_valid_prefers_valid():
    """Test duplicate labeled lines allow later valid annotation to replace Unknown sentinel."""
    results = ["1: unknown", "1: T cells"]
    clusters = ["1"]
    formatted = format_results(results, clusters)
    assert formatted == {"1": "T cells"}


def test_format_results_unknown_sentinel_variants_normalized():
    """Test unknown-like sentinel variants are normalized to Unknown."""
    results = ["Cluster 1: n/a", "Cluster 2: Unknown (low confidence)"]
    clusters = ["1", "2"]
    formatted = format_results(results, clusters)
    assert formatted == {"1": "Unknown", "2": "Unknown"}


def test_format_results_cluster_format_blank_annotation_becomes_unknown():
    """Test blank 'Cluster X:' annotation is normalized to Unknown."""
    results = ["Cluster 1:   ", "Cluster 2: B cells"]
    clusters = ["1", "2"]

    formatted = format_results(results, clusters)

    assert formatted["1"] == "Unknown"
    assert formatted["2"] == "B cells"


def test_format_results_labeled_partial_prefers_cluster_id_mapping():
    """Test partial labeled parsing avoids positional misalignment side-effects."""
    results = [
        "Some explanation header",
        "Cluster 2: B cells",
        "Footer",
    ]
    clusters = ["1", "2"]

    formatted = format_results(results, clusters)

    assert formatted["1"] == "Unknown"
    assert formatted["2"] == "B cells"


def test_format_results_numeric_colon_format():
    """Test plain numeric label format like '1: T cells' is parsed correctly."""
    results = ["1: T cells", "2: B cells"]
    clusters = ["1", "2"]
    formatted = format_results(results, clusters)
    assert formatted == {"1": "T cells", "2": "B cells"}


def test_format_results_fullwidth_colon_and_cluster_prefix_alias():
    """Test fullwidth colon and Cluster_ prefixed ids are parsed by id-aware mapping."""
    results = ["1\uff1a T cells"]
    clusters = ["Cluster_1"]
    formatted = format_results(results, clusters)
    assert formatted == {"Cluster_1": "T cells"}


def test_format_results_parenthesis_label_format():
    """Test parenthesis numeric labels like '1) T cells' are parsed."""
    results = ["1) T cells", "2) B cells"]
    clusters = ["1", "2"]
    formatted = format_results(results, clusters)
    assert formatted == {"1": "T cells", "2": "B cells"}


def test_format_results_duplicate_label_uses_first_and_ignores_unknown_cluster():
    """Test duplicate cluster labels keep first value and unknown labels are ignored."""
    results = [
        "1: T cells",
        "1: B cells",
        "999: Noise cluster",
    ]
    clusters = ["1"]
    formatted = format_results(results, clusters)
    assert formatted == {"1": "T cells"}


def test_format_results_scalar_fallback_does_not_crash():
    """Test unexpected scalar provider output is handled as line-by-line fallback."""
    formatted = format_results(12345, ["1", "2"])  # type: ignore[arg-type]
    assert formatted["1"] == "12345"
    assert formatted["2"] == "Unknown"


def test_format_results_none_fallback_does_not_crash():
    """Test None provider output is tolerated and mapped to Unknown."""
    formatted = format_results(None, ["1"])  # type: ignore[arg-type]
    assert formatted == {"1": "Unknown"}


def test_is_unknown_annotation_supports_common_unknown_variants():
    """Test unknown sentinel detector handles wrapper/context variants."""
    assert is_unknown_annotation("unknown")
    assert is_unknown_annotation("Unknown (low confidence)")
    assert is_unknown_annotation("Inconclusive")
    assert is_unknown_annotation("Inconclusive (no consensus)")
    assert is_unknown_annotation("Error: provider timeout")
    assert is_unknown_annotation("Error(timeout)")
    assert not is_unknown_annotation("Error-prone T cells")
    assert is_unknown_annotation("[N/A]")
    assert is_unknown_annotation("  none  ")
    assert not is_unknown_annotation("T cells")


def test_normalize_annotation_maps_unknown_like_values_to_unknown():
    """Test annotation normalization maps unknown-like values to canonical Unknown."""
    assert normalize_annotation("unknown") == "Unknown"
    assert normalize_annotation(" [unknown] ") == "Unknown"
    assert normalize_annotation("inconclusive") == "Unknown"
    assert normalize_annotation("Inconclusive (no consensus)") == "Unknown"
    assert normalize_annotation(pd.NA) == "Unknown"
    assert normalize_annotation("Error: temporary outage") == "Unknown"
    assert normalize_annotation("Error-prone T cells") == "Error-prone T cells"
    assert normalize_annotation("CD4+ T cells") == "CD4+ T cells"


# Test format_discussion_report function
def test_format_discussion_report_basic():
    """Test basic functionality of format_discussion_report."""
    from mllmcelltype.consensus import format_discussion_report

    # Mock results structure
    mock_results = {
        "consensus": {"0": "T cells", "1": "B cells"},
        "consensus_proportion": {"0": 0.8, "1": 1.0},
        "entropy": {"0": 0.5, "1": 0.0},
        "controversial_clusters": ["0"],
        "resolved": {"0": "T cells"},
        "model_annotations": {
            "gpt-5": {"0": "T cells", "1": "B cells"},
            "claude": {"0": "CD4+ T cells", "1": "B cells"},
        },
        "discussion_logs": {
            "0": [
                {
                    "gpt-5": "CELL TYPE: T cells\nGROUNDS: CD3D, CD4",
                    "claude": "CELL TYPE: CD4+ T cells\nGROUNDS: CD3D, CD4, IL7R",
                }
            ]
        },
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": ["gpt-5", "claude"],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    # Test basic report generation
    report = format_discussion_report(mock_results)

    # Verify report contains expected sections
    assert "MULTI-LLM CONSENSUS DISCUSSION REPORT" in report
    assert "CLUSTER 0" in report
    assert "CLUSTER 1" in report
    assert "INITIAL PREDICTIONS" in report
    assert "ROUND 1 DISCUSSION" in report
    assert "FINAL RESULT" in report
    assert "T cells" in report
    assert "B cells" in report
    assert "gpt-5" in report
    assert "claude" in report


def test_format_discussion_report_single_cluster():
    """Test format_discussion_report with cluster_id filter."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells", "1": "B cells"},
        "consensus_proportion": {"0": 0.8, "1": 1.0},
        "entropy": {"0": 0.5, "1": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {
            "gpt-5": {"0": "T cells", "1": "B cells"},
        },
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": ["gpt-5"],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    # Test single cluster report
    report = format_discussion_report(mock_results, cluster_id="0")

    assert "CLUSTER 0" in report
    assert "CLUSTER 1" not in report
    assert "T cells" in report


def test_format_discussion_report_single_cluster_strips_cluster_id_whitespace():
    """Test report cluster filter accepts whitespace-padded IDs."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells", "1": "B cells"},
        "consensus_proportion": {"0": 0.8, "1": 1.0},
        "entropy": {"0": 0.5, "1": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {"gpt-5": {"0": "T cells", "1": "B cells"}},
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": ["gpt-5"],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    report = format_discussion_report(mock_results, cluster_id=" 0 ")

    assert "CLUSTER 0" in report
    assert "CLUSTER 1" not in report


def test_format_discussion_report_single_cluster_accepts_numeric_cluster_id():
    """Test report cluster filter tolerates numeric cluster IDs."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells", "1": "B cells"},
        "consensus_proportion": {"0": 0.8, "1": 1.0},
        "entropy": {"0": 0.5, "1": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {"gpt-5": {"0": "T cells", "1": "B cells"}},
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": ["gpt-5"],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    report = format_discussion_report(mock_results, cluster_id=0)  # type: ignore[arg-type]

    assert "CLUSTER 0" in report
    assert "CLUSTER 1" not in report


def test_format_discussion_report_no_discussion():
    """Test format_discussion_report when no discussion was needed."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells"},
        "consensus_proportion": {"0": 1.0},
        "entropy": {"0": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {
            "gpt-5": {"0": "T cells"},
            "claude": {"0": "T cells"},
        },
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": ["gpt-5", "claude"],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    report = format_discussion_report(mock_results)

    assert "NO DISCUSSION NEEDED" in report
    assert "Consensus reached with initial predictions" in report


def test_format_discussion_report_metadata_formats_model_dicts_and_none_fields():
    """Test metadata formatting avoids malformed model labels and None display."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells"},
        "consensus_proportion": {"0": 1.0},
        "entropy": {"0": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {"openai:gpt-5.2": {"0": "T cells"}},
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": None,
            "models": [{"model": "gpt-5.2"}],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    report = format_discussion_report(mock_results)

    assert "Models: gpt-5.2" in report
    assert "Models: :gpt-5.2" not in report
    assert "Tissue: N/A" in report


def test_format_discussion_report_metadata_models_string_not_split_by_character():
    """Test string metadata models value is treated as one model entry."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells"},
        "consensus_proportion": {"0": 1.0},
        "entropy": {"0": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {"openai:gpt-5.2": {"0": "T cells"}},
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": "gpt-5.2",
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    report = format_discussion_report(mock_results)

    assert "Models: gpt-5.2" in report
    assert "Models: g, p, t" not in report


def test_format_discussion_report_metadata_models_set_is_deterministic():
    """Test set-form metadata models are rendered in deterministic sorted order."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells"},
        "consensus_proportion": {"0": 1.0},
        "entropy": {"0": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {"openai:gpt-5.2": {"0": "T cells"}},
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": {"zeta-model", "alpha-model"},
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    report = format_discussion_report(mock_results)

    assert "Models: alpha-model, zeta-model" in report


def test_format_discussion_report_save_to_file():
    """Test format_discussion_report saving to file."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells"},
        "consensus_proportion": {"0": 1.0},
        "entropy": {"0": 0.0},
        "controversial_clusters": [],
        "resolved": {},
        "model_annotations": {"gpt-5": {"0": "T cells"}},
        "discussion_logs": {},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": ["gpt-5"],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    # Test saving to file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        output_path = f.name

    try:
        report = format_discussion_report(mock_results, output_file=output_path)

        # Verify file was created and contains the report
        assert os.path.exists(output_path)
        with open(output_path) as f:
            file_content = f.read()
        assert file_content == report
        assert "MULTI-LLM CONSENSUS DISCUSSION REPORT" in file_content
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_format_discussion_report_non_string_round_response():
    """Test report generation tolerates non-string discussion response payloads."""
    from mllmcelltype.consensus import format_discussion_report

    mock_results = {
        "consensus": {"0": "T cells"},
        "consensus_proportion": {"0": 0.8},
        "entropy": {"0": 0.5},
        "controversial_clusters": ["0"],
        "resolved": {"0": "T cells"},
        "model_annotations": {"gpt-5": {"0": "T cells"}},
        "discussion_logs": {"0": [{"gpt-5": {"raw": "object-response"}}]},
        "metadata": {
            "timestamp": "2026-01-26 12:00:00",
            "species": "human",
            "tissue": "blood",
            "models": ["gpt-5"],
            "consensus_threshold": 0.7,
            "max_discussion_rounds": 3,
        },
    }

    report = format_discussion_report(mock_results)
    assert "object-response" in report


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
