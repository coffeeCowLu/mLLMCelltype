"""Main annotation module for LLMCellType."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from .config import get_default_model
from .functions import PROVIDER_FUNCTIONS
from .logger import setup_logging, write_log
from .prompts import create_prompt
from .url_utils import resolve_provider_base_url
from .utils import (
    cluster_sort_key,
    create_cache_key,
    format_results,
    load_api_key,
    load_from_cache,
    normalize_marker_genes_keys,
    parse_marker_genes,
    save_to_cache,
)


def _resolve_provider(provider: str) -> tuple[str, Callable[..., Any]]:
    """Normalize provider name and resolve provider callable."""
    if not isinstance(provider, str):
        raise ValueError(f"Provider name must be a string, got {type(provider).__name__}")
    if not provider:
        raise ValueError("Provider name is required")

    normalized_provider = provider.strip().lower()
    if not normalized_provider:
        raise ValueError("Provider name is required")
    provider_func = PROVIDER_FUNCTIONS.get(normalized_provider)
    if not provider_func:
        error_msg = f"Unknown provider: {normalized_provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    return normalized_provider, provider_func


def _resolve_model_and_api_key(
    provider: str,
    model: str | None,
    api_key: str | None,
) -> tuple[str, str]:
    """Resolve model and API key with consistent fallback behavior."""
    def _normalize_optional_text(value: str | None, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
        normalized = value.strip()
        return normalized or None

    resolved_model = _normalize_optional_text(model, "model")
    if not resolved_model:
        resolved_model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {resolved_model}")

    resolved_api_key = _normalize_optional_text(api_key, "api_key")
    if not resolved_api_key:
        resolved_api_key = _normalize_optional_text(load_api_key(provider), "api_key")
    if not resolved_api_key:
        error_msg = f"API key not found for provider: {provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    return resolved_model, resolved_api_key


def _to_text_response(result: Any) -> str:
    """Normalize provider response to text."""
    if isinstance(result, list):
        return "\n".join(result)
    if isinstance(result, str):
        return result
    return str(result)


def _split_analyzable_clusters(
    marker_genes: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Split marker genes into analyzable and empty-marker cluster IDs."""
    analyzable: dict[str, list[str]] = {}
    empty_clusters: list[str] = []
    for cluster_id, genes in marker_genes.items():
        if genes:
            analyzable[cluster_id] = genes
        else:
            empty_clusters.append(cluster_id)
    return analyzable, empty_clusters


def _merge_annotation_results(
    *,
    all_clusters: list[str],
    analyzed_results: dict[str, str],
    empty_clusters: list[str],
) -> dict[str, str]:
    """Merge model annotations with deterministic Unknown for empty-marker clusters."""
    empty_cluster_set = set(empty_clusters)
    return {
        cluster_id: (
            "Unknown" if cluster_id in empty_cluster_set else analyzed_results.get(cluster_id, "Unknown")
        )
        for cluster_id in all_clusters
    }


def annotate_clusters(
    marker_genes: dict[str, list[str]] | pd.DataFrame,
    species: str,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    tissue: str | None = None,
    additional_context: str | None = None,
    prompt_template: str | None = None,
    use_cache: bool = True,
    cache_dir: str | None = None,
    log_dir: str | None = None,
    log_level: str = "INFO",
    base_urls: str | dict[str, str] | None = None,
) -> dict[str, str]:
    """Annotate cell clusters using LLM.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes,
                     or DataFrame with 'cluster' and 'gene' columns
        species: Species name (e.g., 'human', 'mouse')
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-5.5', 'claude-sonnet-4-6')
        api_key: API key for the provider
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        log_dir: Directory to store log files
        log_level: Logging level
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs (e.g., {'openai': 'https://proxy.com/v1'})

    Returns:
        Dict[str, str]: Dictionary mapping cluster names to annotations

    """
    # Setup logging
    setup_logging(log_dir=log_dir, log_level=log_level)

    # Normalize provider to lowercase early so all downstream code
    # (base_urls lookup, cache key, API key resolution) is consistent.
    provider, provider_func = _resolve_provider(provider)
    write_log(f"Starting annotation with provider: {provider}")

    # Parse marker genes if DataFrame
    if isinstance(marker_genes, pd.DataFrame):
        marker_genes = parse_marker_genes(marker_genes)
    else:
        marker_genes = normalize_marker_genes_keys(marker_genes)

    # Get clusters in natural numerical order (consistent with prompt)
    all_clusters = sorted(marker_genes.keys(), key=cluster_sort_key)
    write_log(f"Found {len(all_clusters)} clusters")

    if not all_clusters:
        write_log("No clusters provided; skipping annotation request", level="warning")
        return {}

    analyzable_marker_genes, empty_clusters = _split_analyzable_clusters(marker_genes)
    analyzable_clusters = sorted(analyzable_marker_genes.keys(), key=cluster_sort_key)

    if empty_clusters:
        write_log(
            f"Skipping {len(empty_clusters)} cluster(s) with empty marker genes: {', '.join(sorted(empty_clusters, key=cluster_sort_key))}",
            level="warning",
        )

    if not analyzable_clusters:
        write_log(
            "No clusters have non-empty marker genes; returning Unknown for all clusters",
            level="warning",
        )
        return {cluster_id: "Unknown" for cluster_id in all_clusters}

    model, api_key = _resolve_model_and_api_key(provider, model, api_key)

    # Create prompt
    prompt = create_prompt(
        marker_genes=analyzable_marker_genes,
        species=species,
        tissue=tissue,
        additional_context=additional_context,
        prompt_template=prompt_template,
    )

    # Resolve base URL (before cache check — base_url is part of the cache key)
    base_url = resolve_provider_base_url(provider, base_urls)

    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider, base_url)
        cached_results = load_from_cache(cache_key, cache_dir)
        if cached_results is not None:
            write_log("Using cached results")
            analyzed_results = format_results(cached_results, analyzable_clusters)
            return _merge_annotation_results(
                all_clusters=all_clusters,
                analyzed_results=analyzed_results,
                empty_clusters=empty_clusters,
            )

    # Process request
    try:
        write_log(f"Processing request with {provider} using model {model}")
        start_time = time.time()

        # Call provider function with base_url
        results = provider_func(prompt, model, api_key, base_url)

        end_time = time.time()
        write_log(f"Request processed in {end_time - start_time:.2f} seconds")

        # Save to cache
        if use_cache:
            save_to_cache(cache_key, results, cache_dir)

        # Format results
        analyzed_results = format_results(results, analyzable_clusters)
        return _merge_annotation_results(
            all_clusters=all_clusters,
            analyzed_results=analyzed_results,
            empty_clusters=empty_clusters,
        )

    except Exception as e:
        error_msg = f"Error during annotation: {e!s}"
        write_log(error_msg, level="error")
        raise


def get_model_response(
    prompt: str,
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
    cache_dir: str | None = None,
    base_url: str | None = None,
) -> str:
    """Get response from a model for a given prompt.

    Args:
        prompt: The prompt to send to the model
        provider: The provider name (e.g., 'openai', 'anthropic')
        model: The model name. If None, uses the default model for the provider.
        api_key: The API key for the provider. If None, loads from environment.
        use_cache: Whether to use cache
        cache_dir: The cache directory
        base_url: Optional custom base URL

    Returns:
        str: The model response

    """

    provider, provider_func = _resolve_provider(provider)
    model, api_key = _resolve_model_and_api_key(provider, model, api_key)

    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider, base_url)
        cached_result = load_from_cache(cache_key, cache_dir)
        if cached_result is not None:
            write_log(f"Using cached result for {model}")
            return _to_text_response(cached_result)

    # Call provider function
    try:
        write_log(f"Requesting response from {provider} ({model})")
        result = provider_func(prompt, model, api_key, base_url)

        # Save to cache
        if use_cache:
            save_to_cache(cache_key, result, cache_dir)

        # Convert list to string if needed
        return _to_text_response(result)
    except Exception as e:
        error_msg = f"Error getting model response: {e!s}"
        write_log(error_msg, level="error")
        raise
