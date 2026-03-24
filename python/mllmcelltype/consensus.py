"""Module for consensus annotation of cell types from multiple LLM predictions."""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from typing import Any

import requests

from .annotate import annotate_clusters, get_model_response
from .config import (
    DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
    DEFAULT_FALLBACK_ENTROPY,
    get_default_model,
    get_supported_providers,
)
from .functions import get_provider
from .logger import write_log
from .prompts import (
    create_cell_type_extraction_prompt,
    create_consensus_check_prompt,
    create_discussion_consensus_prompt,
    create_discussion_prompt,
    create_initial_discussion_prompt,
)
from .url_utils import resolve_provider_base_url
from .utils import (
    cluster_sort_key,
    is_unknown_annotation,
    load_api_key,
    normalize_annotation,
    normalize_marker_genes_keys,
)

# Default result structure for discussion round consensus check
# Used when consensus check fails or has insufficient data
DEFAULT_CONSENSUS_RESULT = {
    "reached": False,
    "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
    "entropy": DEFAULT_FALLBACK_ENTROPY,
    "majority_prediction": "Unknown",
}

# Recoverable errors from provider calls and response parsing.
# These should not crash full consensus workflows.
RECOVERABLE_LLM_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    json.JSONDecodeError,
    AttributeError,
    ImportError,
    TypeError,
    RuntimeError,
)


def _normalize_api_keys(api_keys: dict[str, str] | None) -> dict[str, str]:
    """Normalize and validate provider API keys.

    Rules:
    - provider keys are normalized to lowercase strings
    - key values must be strings
    - blank keys are treated as missing and dropped
    """
    if not api_keys:
        return {}

    normalized: dict[str, str] = {}
    for raw_provider, raw_key in api_keys.items():
        if not isinstance(raw_provider, str):
            raise ValueError(
                f"API key provider names must be strings, got {type(raw_provider).__name__}"
            )
        provider = raw_provider.strip().lower()
        if not provider:
            continue
        if raw_key is None:
            continue
        if not isinstance(raw_key, str):
            raise ValueError(
                f"API key for provider '{provider}' must be a string, got {type(raw_key).__name__}"
            )
        key = raw_key.strip()
        if key:
            normalized[provider] = key

    return normalized


def _normalize_consensus_model_spec(
    consensus_model: str | dict[str, str] | None,
) -> dict[str, str] | None:
    """Normalize and validate consensus_model specification."""
    if consensus_model is None:
        return None

    if isinstance(consensus_model, str):
        model_name = consensus_model.strip()
        if not model_name:
            raise ValueError("consensus_model string cannot be empty")
        provider = get_provider(model_name)
        return {"provider": provider, "model": model_name}

    if not isinstance(consensus_model, dict):
        raise ValueError(
            f"consensus_model must be a string or dict, got {type(consensus_model).__name__}"
        )

    provider_normalized = _normalize_consensus_model_provider_field(consensus_model.get("provider"))
    model_normalized = _normalize_consensus_model_name_field(consensus_model.get("model"))

    if provider_normalized is None and model_normalized is None:
        raise ValueError("consensus_model must include at least one of 'provider' or 'model'")

    if provider_normalized is None and model_normalized is not None:
        provider_normalized = get_provider(model_normalized)
    if provider_normalized is not None and model_normalized is None:
        model_normalized = get_default_model(provider_normalized)

    _validate_consensus_model_provider_match(provider_normalized, model_normalized)

    return {"provider": provider_normalized, "model": model_normalized}


def _normalize_consensus_model_provider_field(provider: Any) -> str | None:
    """Normalize optional consensus provider field."""
    if provider is None:
        return None
    if not isinstance(provider, str):
        raise ValueError(
            f"consensus_model.provider must be a string when provided, got {type(provider).__name__}"
        )
    normalized = provider.strip().lower()
    if not normalized:
        raise ValueError("consensus_model.provider cannot be empty when provided")
    if normalized not in get_supported_providers():
        raise ValueError(f"Unsupported consensus_model.provider: {normalized}")
    return normalized


def _normalize_consensus_model_name_field(model: Any) -> str | None:
    """Normalize optional consensus model field."""
    if model is None:
        return None
    if not isinstance(model, str):
        raise ValueError(
            f"consensus_model.model must be a string when provided, got {type(model).__name__}"
        )
    normalized = model.strip()
    if not normalized:
        raise ValueError("consensus_model.model cannot be empty when provided")
    return normalized


def _validate_consensus_model_provider_match(
    provider: str | None,
    model: str | None,
) -> None:
    """Validate provider/model consistency when both are specified."""
    if not provider or not model or provider == "openrouter":
        return
    inferred_provider = None
    with contextlib.suppress(ValueError):
        inferred_provider = get_provider(model)
    if inferred_provider and inferred_provider != provider:
        raise ValueError(
            "consensus_model provider/model mismatch: "
            f"provider='{provider}', model='{model}' "
            f"(inferred provider '{inferred_provider}')"
        )


def _normalize_single_prediction_map(model_name: str, raw_results: Any) -> dict[str, str]:
    """Normalize one model's cluster->annotation payload."""
    if raw_results is None:
        write_log(f"Model '{model_name}' returned no predictions, skipping", level="warning")
        return {}

    if not isinstance(raw_results, dict):
        write_log(
            f"Model '{model_name}' predictions must be a dict, got {type(raw_results).__name__}; skipping",
            level="warning",
        )
        return {}

    cluster_map: dict[str, str] = {}
    for raw_cluster, raw_annotation in raw_results.items():
        if raw_cluster is None or raw_annotation is None:
            continue
        cluster_id = str(raw_cluster).strip()
        if not cluster_id:
            continue
        annotation = str(raw_annotation).strip()
        if annotation and not is_unknown_annotation(annotation):
            existing = cluster_map.get(cluster_id)
            if existing is None or existing == annotation:
                cluster_map[cluster_id] = annotation
            else:
                write_log(
                    f"Model '{model_name}' has conflicting annotations for normalized cluster "
                    f"'{cluster_id}', keeping first value '{existing}' and ignoring '{annotation}'",
                    level="warning",
                )
    return cluster_map


def _normalize_predictions(
    predictions: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Normalize consensus prediction payload with defensive validation."""
    if not isinstance(predictions, dict):
        raise ValueError(
            f"predictions must be a dict mapping model->cluster annotations, got {type(predictions).__name__}"
        )

    normalized: dict[str, dict[str, str]] = {}
    for raw_model, raw_results in predictions.items():
        model_name = str(raw_model).strip() or str(raw_model)
        cluster_map = _normalize_single_prediction_map(model_name, raw_results)

        if cluster_map:
            if model_name in normalized:
                write_log(
                    f"Duplicate model name after normalization: '{model_name}', merging predictions",
                    level="warning",
                )
                normalized[model_name].update(cluster_map)
            else:
                normalized[model_name] = cluster_map
        else:
            write_log(f"Model '{model_name}' had no valid annotations, skipping", level="warning")

    return normalized


def _collect_valid_round_responses(round_responses: dict[str, Any]) -> dict[str, str]:
    """Normalize round responses and drop provider error outputs."""
    valid_responses: dict[str, str] = {}
    for model_key, response in round_responses.items():
        response_text = response if isinstance(response, str) else str(response)
        response_text = response_text.strip()
        if not response_text or response_text.casefold().startswith("error:"):
            continue
        valid_responses[model_key] = response_text
    return valid_responses


def _build_metadata(
    *,
    models: list[str | dict[str, str]],
    species: str,
    tissue: str | None,
    consensus_threshold: float,
    entropy_threshold: float,
    max_discussion_rounds: int,
) -> dict[str, Any]:
    """Build metadata payload for consensus outputs."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "species": species,
        "tissue": tissue,
        "consensus_threshold": consensus_threshold,
        "entropy_threshold": entropy_threshold,
        "max_discussion_rounds": max_discussion_rounds,
    }


def _build_interactive_result(
    *,
    metadata: dict[str, Any],
    consensus: dict[str, str] | None = None,
    consensus_proportion: dict[str, float] | None = None,
    entropy: dict[str, float] | None = None,
    controversial_clusters: list[str] | None = None,
    resolved: dict[str, str] | None = None,
    model_annotations: dict[str, dict[str, str]] | None = None,
    discussion_logs: dict[str, list[dict]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build normalized return payload for interactive consensus."""
    normalized_discussion_logs = discussion_logs or {}
    normalized_controversial_clusters = [str(cluster_id) for cluster_id in (controversial_clusters or [])]
    discussion_round_counts = {
        cluster_id: 0 for cluster_id in normalized_controversial_clusters
    }
    discussion_round_counts.update({
        str(cluster_id): len(rounds) if isinstance(rounds, list) else 0
        for cluster_id, rounds in normalized_discussion_logs.items()
    })

    result = {
        "consensus": consensus or {},
        "consensus_proportion": consensus_proportion or {},
        "entropy": entropy or {},
        "controversial_clusters": normalized_controversial_clusters,
        "resolved": resolved or {},
        "model_annotations": model_annotations or {},
        "discussion_logs": normalized_discussion_logs,
        "discussion_round_counts": discussion_round_counts,
        "metadata": metadata,
    }
    if error:
        result["error"] = error
    return result


def _resolve_api_keys_for_models(
    models: list[str | dict[str, str]],
    api_keys: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve API keys for all model providers.

    Priority per provider:
    1) caller-provided api_keys entry
    2) environment/.env via load_api_key(provider)
    """
    resolved_keys = _normalize_api_keys(api_keys)

    for model_item in models:
        provider, _model_name = _resolve_model_spec(model_item)
        if not provider:
            continue
        if resolved_keys.get(provider):
            continue
        env_key = load_api_key(provider)
        if env_key:
            resolved_keys[provider] = env_key

    return resolved_keys


def _validate_models_spec(models: list[str | dict[str, str]]) -> None:
    """Validate model specification list."""
    supported_providers = set(get_supported_providers())

    for i, item in enumerate(models):
        if isinstance(item, dict):
            _validate_model_spec_dict(item, i, supported_providers)
            continue
        _validate_model_spec_scalar(item, i)


def _validate_model_spec_dict(
    item: dict[str, str],
    index: int,
    supported_providers: set[str],
) -> None:
    """Validate dict-form model specification."""
    model_name = item.get("model")
    provider_name = item.get("provider")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            f"Invalid model specification at index {index}: "
            f"dict must include a non-empty string 'model' key, got {item}"
        )
    if provider_name is None:
        return
    if not isinstance(provider_name, str):
        raise ValueError(
            f"Invalid model specification at index {index}: "
            f"'provider' must be a string when provided, got {item}"
        )
    provider_normalized = provider_name.strip().lower()
    if not provider_normalized:
        raise ValueError(
            f"Invalid model specification at index {index}: "
            f"'provider' must be a non-empty string when provided, got {item}"
        )
    if provider_normalized not in supported_providers:
        raise ValueError(
            f"Invalid model specification at index {index}: "
            f"unsupported provider '{provider_name}'. "
            f"Supported providers: {sorted(supported_providers)}"
        )


def _validate_model_spec_scalar(item: Any, index: int) -> None:
    """Validate scalar-form model specification."""
    if isinstance(item, str):
        if item.strip():
            return
        raise ValueError(
            f"Invalid model specification at index {index}: "
            "model name must be a non-empty string"
        )
    if not item:
        raise ValueError(
            f"Invalid model specification at index {index}: "
            "model name must be a non-empty string"
        )
    raise ValueError(
        f"Invalid model specification at index {index}: "
        f"expected string or dict, got {type(item).__name__}"
    )


def _resolve_model_spec(model_item: str | dict[str, str]) -> tuple[str | None, str]:
    """Resolve provider and model name from a model specification."""
    if isinstance(model_item, dict):
        provider = model_item.get("provider")
        model_name = model_item.get("model")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"Model spec must include 'model': {model_item}")
        model_name = model_name.strip()
        if isinstance(provider, str):
            provider = provider.strip()
            if not provider:
                provider = None
        if not provider:
            provider = get_provider(model_name)
    else:
        model_name = model_item.strip()
        provider = get_provider(model_name)

    return str(provider).strip().lower() if provider else None, model_name


def _iter_supported_provider_keys(api_keys: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (provider, default_model, api_key) for supported configured providers only."""
    supported: list[tuple[str, str, str]] = []
    supported_providers = set(get_supported_providers())
    for provider, key in _normalize_api_keys(api_keys).items():
        if not key:
            continue
        if provider not in supported_providers:
            write_log(
                f"Ignoring unsupported provider key in consensus flow: {provider}",
                level="debug",
            )
            continue
        default_model = get_default_model(provider)
        supported.append((provider, default_model, key))
    return supported


def _has_supported_api_key(api_keys: dict[str, str]) -> bool:
    """Whether at least one supported provider key is configured."""
    return bool(_iter_supported_provider_keys(api_keys))


def _resolve_consensus_provider(
    consensus_model: dict[str, str] | None,
    api_keys: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Resolve provider, model, and API key for consensus checking.

    Resolution order: explicit ``consensus_model`` dict → first available
    key in ``api_keys``.

    Args:
        consensus_model: Optional dict with 'provider' and/or 'model' keys
        api_keys: Dictionary mapping provider names to API keys

    Returns:
        tuple of (provider, model, api_key) — any element may be None
    """
    normalized_api_keys = _normalize_api_keys(api_keys)
    supported_keys = _iter_supported_provider_keys(normalized_api_keys)

    if consensus_model:
        if not isinstance(consensus_model, dict):
            raise ValueError(
                f"consensus_model must be a dict when passed to _resolve_consensus_provider, got {type(consensus_model).__name__}"
            )
        provider = consensus_model.get("provider")
        if isinstance(provider, str):
            provider = provider.strip().lower()
            if not provider:
                provider = None
        model = consensus_model.get("model")
        if isinstance(model, str):
            model = model.strip()
            if not model:
                model = None
        if not provider and model:
            provider = get_provider(model)
        if provider and not model:
            model = get_default_model(provider)
    else:
        provider = None
        model = None
        if supported_keys:
            provider, model, _key = supported_keys[0]

    api_key = normalized_api_keys.get(provider) if provider else None
    return provider, model, api_key


def _resolve_fallback_target(
    *,
    provider: str | None,
    fallback_provider: str | None,
    fallback_model: str | None,
    supported_keys: list[tuple[str, str, str]],
) -> tuple[str | None, str | None]:
    """Resolve fallback provider/model target from explicit args or configured keys."""
    if fallback_provider is None:
        for candidate_provider, default_model, _key in supported_keys:
            if candidate_provider != provider:
                return candidate_provider, default_model
        return None, None

    supported_provider_names = {provider_name for provider_name, _model, _key in supported_keys}
    if fallback_provider not in supported_provider_names:
        write_log(
            f"Ignoring unsupported fallback provider in consensus flow: {fallback_provider}",
            level="debug",
        )
        return None, None

    if fallback_model:
        return fallback_provider, fallback_model

    with contextlib.suppress(ValueError):
        return fallback_provider, get_default_model(fallback_provider)
    return fallback_provider, fallback_model


def _call_primary_provider_with_retries(
    *,
    prompt: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    max_retries: int,
    base_urls: str | dict[str, str] | None,
) -> str | None:
    """Call primary provider with retry semantics."""
    if not api_key:
        write_log(f"No API key for {provider}, trying fallback", level="debug")
        return None

    primary_base_url = resolve_provider_base_url(provider, base_urls)
    for attempt in range(max_retries):
        try:
            response = get_model_response(
                prompt=prompt,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=primary_base_url,
            )
            write_log(f"Successfully got response from {provider} on attempt {attempt + 1}")
            return response
        except RECOVERABLE_LLM_EXCEPTIONS as e:
            if isinstance(e, ImportError):
                write_log(
                    f"Provider {provider} unavailable (missing dependency): {e!s}",
                    level="warning",
                )
                return None
            if attempt < max_retries - 1:
                write_log(
                    f"Error on {provider} attempt {attempt + 1}/{max_retries}: {e!s}",
                    level="warning",
                )
                time.sleep(5 * (2**attempt))
                continue
            write_log(f"All {provider} retry attempts failed: {e!s}", level="warning")
    return None


def _call_fallback_provider_once(
    *,
    prompt: str,
    fallback_provider: str | None,
    fallback_model: str | None,
    normalized_api_keys: dict[str, str],
    base_urls: str | dict[str, str] | None,
) -> str | None:
    """Call fallback provider once when configured."""
    if not fallback_provider or not fallback_model:
        return None

    fallback_api_key = normalized_api_keys.get(fallback_provider)
    if not fallback_api_key:
        return None

    fallback_base_url = resolve_provider_base_url(fallback_provider, base_urls)
    try:
        response = get_model_response(
            prompt=prompt,
            provider=fallback_provider,
            model=fallback_model,
            api_key=fallback_api_key,
            base_url=fallback_base_url,
        )
        write_log(f"Successfully got response from {fallback_provider} as fallback")
        return response
    except RECOVERABLE_LLM_EXCEPTIONS as e:
        write_log(f"Error on {fallback_provider} fallback: {e!s}", level="warning")
        return None


def _call_llm_with_retry(
    prompt: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    max_retries: int = 3,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    api_keys: dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> str | None:
    """Call LLM with retry logic and fallback provider.

    API key resolution: explicit ``api_key`` arg → ``api_keys`` dict.
    This function does NOT load keys from environment variables or
    ``.env`` files; the caller (or the entry-point function) is
    responsible for populating ``api_keys`` with all available keys.

    Args:
        prompt: The prompt to send
        provider: Primary provider to use
        model: Primary model to use
        api_key: API key for primary provider (falls back to api_keys dict)
        max_retries: Maximum retry attempts
        fallback_provider: Fallback provider if primary fails
        fallback_model: Fallback model if primary fails
        api_keys: Dictionary of API keys keyed by provider name
        base_urls: Custom base URLs for API endpoints

    Returns:
        str | None: LLM response or None if all attempts failed
    """
    normalized_api_keys = _normalize_api_keys(api_keys)
    supported_keys = _iter_supported_provider_keys(normalized_api_keys)
    provider = provider.lower() if provider else None
    fallback_provider = fallback_provider.lower() if fallback_provider else None

    # Resolve primary API key from explicit arg or api_keys dict.
    # We intentionally do NOT call load_api_key() here: if the caller
    # didn't provide the key, it means the provider is not configured.
    if not api_key:
        api_key = normalized_api_keys.get(provider) if provider else None

    fallback_provider, fallback_model = _resolve_fallback_target(
        provider=provider,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        supported_keys=supported_keys,
    )

    primary_response = _call_primary_provider_with_retries(
        prompt=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        base_urls=base_urls,
    )
    if primary_response is not None:
        return primary_response

    fallback_response = _call_fallback_provider_once(
        prompt=prompt,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        normalized_api_keys=normalized_api_keys,
        base_urls=base_urls,
    )
    if fallback_response is not None:
        return fallback_response

    write_log(f"All LLM attempts failed for provider {provider}", level="warning")

    return None


def _normalize_predicted_label(label: str | None) -> str | None:
    """Normalize parsed label and drop empty/Unknown sentinels."""
    if label is None:
        return None
    normalized = str(label).strip()
    if len(normalized) >= 2 and normalized[0] == "[" and normalized[-1] == "]":
        normalized = normalized[1:-1].strip()
    return normalized if normalized and not is_unknown_annotation(normalized) else None


def _parse_standard_metrics_lines(lines: list[str]) -> tuple[float | None, float | None, str | None]:
    """Parse strict 4-line [0/1, CP, H, annotation] format."""
    if len(lines) < 4:
        return None, None, None

    result_lines = lines[-4:]
    if not re.match(r"^\s*[01]\s*$", result_lines[0]):
        return None, None, None
    if not re.match(r"^\s*(0\.\d+|1\.0*|[01])\s*$", result_lines[1]):
        return None, None, None
    if not re.match(r"^\s*(\d+\.\d+|\d+)\s*$", result_lines[2]):
        return None, None, None

    with contextlib.suppress(ValueError, IndexError):
        cp_value = float(result_lines[1].strip())
        h_value = float(result_lines[2].strip())
        annotation = _normalize_predicted_label(result_lines[3])
        write_log("Detected standard 4-line format", level="debug")
        return cp_value, h_value, annotation

    return None, None, None


def _extract_json_block_from_text(text: str) -> str | None:
    """Extract likely JSON payload from free-text response."""
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        return code_block_match.group(1)

    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        return brace_match.group(1)
    return None


def _extract_first_valid_json_number(
    payload: dict[str, Any],
    keys: list[str],
    validator: Any,
) -> float | None:
    """Extract first numeric JSON field that satisfies validator."""
    for key in keys:
        value = payload.get(key)
        with contextlib.suppress(TypeError, ValueError):
            candidate = float(value)
            if validator(candidate):
                return candidate
    return None


def _extract_first_valid_json_label(payload: dict[str, Any], keys: list[str]) -> str | None:
    """Extract first non-empty label field from JSON payload."""
    for key in keys:
        if key not in payload:
            continue
        normalized = _normalize_predicted_label(str(payload.get(key)))
        if normalized:
            return normalized
    return None


def _parse_metrics_from_json_text(text: str) -> tuple[float | None, float | None, str | None]:
    """Parse CP/H/label from JSON-style payload when present."""
    json_text = _extract_json_block_from_text(text)
    if not json_text:
        return None, None, None

    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, None, None

    if not isinstance(payload, dict):
        return None, None, None

    cp_candidates = ["consensus_proportion", "cp", "consensusProportion"]
    entropy_candidates = ["entropy", "h", "shannon_entropy", "shannonEntropy"]
    label_candidates = [
        "majority_prediction",
        "majorityPrediction",
        "majority",
        "annotation",
        "cell_type",
        "label",
    ]

    cp_value = _extract_first_valid_json_number(payload, cp_candidates, lambda v: 0 <= v <= 1)
    entropy_value = _extract_first_valid_json_number(payload, entropy_candidates, lambda v: v >= 0)
    label = _extract_first_valid_json_label(payload, label_candidates)

    if cp_value is not None or entropy_value is not None or label is not None:
        write_log("Parsed consensus metrics from JSON payload", level="debug")
    return cp_value, entropy_value, label


def _extract_numeric_metric(
    *,
    lines: list[str],
    full_text: str,
    line_keywords: list[str],
    inline_patterns: list[str],
    full_patterns: list[str],
    validator: Any,
) -> float | None:
    """Extract numeric metric from either keyworded lines or full-text regex patterns."""
    for line in lines:
        lower_line = line.lower()
        if not any(keyword in lower_line for keyword in line_keywords):
            continue
        for pattern in inline_patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            with contextlib.suppress(ValueError, TypeError):
                value = float(match.group(1))
                if validator(value):
                    return value

    for pattern in full_patterns:
        match = re.search(pattern, full_text)
        if not match:
            continue
        with contextlib.suppress(ValueError, TypeError):
            value = float(match.group(1))
            if validator(value):
                return value
    return None


def _extract_structured_annotation_from_lines(lines: list[str]) -> str | None:
    """Extract annotation from explicitly labeled lines."""
    structured_patterns = [
        r"(?i)majority(?:\s+prediction|\s+cell\s*type)?\s*(?:[:=]|\uFF1A)\s*(.+?)\s*$",
        r"(?i)cell\s*type\s*(?:[:=]|\uFF1A)\s*(.+?)\s*$",
        r"(?i)claim\s*(?:[:=]|\uFF1A)\s*(.+?)\s*$",
    ]
    for line in lines:
        for pattern in structured_patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            normalized = _normalize_predicted_label(match.group(1))
            if normalized:
                return normalized
    return None


def _is_metric_or_meta_line(line_clean: str) -> bool:
    """Return True if a line looks like metric/meta content instead of annotation."""
    metric_patterns = [
        r"^\s*\d+(\.\d+)?\s*$",
        r"^\s*[01]\s*$",
        r"^\s*(0\.\d+|1\.0*|[01])\s*$",
        r"^\s*(\d+\.\d+|\d+)\s*$",
    ]
    label_patterns = ["consensus", "proportion", "entropy", "cp", "majority", "cell type", "claim"]

    if not line_clean:
        return True
    matches_metric_pattern = any(re.match(pattern, line_clean) for pattern in metric_patterns)
    has_metric_prefix = bool(re.match(r"(?i)^\s*(?:cp|h)\s*[:=]", line_clean))
    has_meta_label = any(label in line_clean.lower() for label in label_patterns)
    return matches_metric_pattern or has_metric_prefix or has_meta_label


def _extract_annotation_from_lines(lines: list[str]) -> str | None:
    """Extract annotation from structured lines or unlabeled majority line."""
    structured = _extract_structured_annotation_from_lines(lines)
    if structured:
        return structured

    for line in lines:
        line_clean = line.strip()
        if _is_metric_or_meta_line(line_clean):
            continue
        normalized = _normalize_predicted_label(line_clean)
        if normalized:
            return normalized
    return None


def _extract_metrics_from_text(
    text: str,
) -> tuple[float | None, float | None, str | None]:
    """Extract consensus metrics (CP, H) and optional annotation from text."""
    if not text or not text.strip():
        return None, None, None

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return None, None, None

    cp_value, entropy_value, annotation = _parse_standard_metrics_lines(lines)
    if cp_value is not None and entropy_value is not None:
        return cp_value, entropy_value, annotation

    write_log(f"Using flexible format parsing for {len(lines)} line(s)", level="debug")

    json_cp, json_entropy, json_annotation = _parse_metrics_from_json_text(text)
    cp_value = json_cp
    entropy_value = json_entropy
    annotation = json_annotation

    if cp_value is None:
        cp_value = _extract_numeric_metric(
            lines=lines,
            full_text=text,
            line_keywords=["consensus", "proportion", "cp"],
            inline_patterns=[
                r"(?i)consensus\s+proportion\s*(?:\(CP\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bcp\s*[:=]\s*([0-9.]+)",
            ],
            full_patterns=[
                r"(?i)consensus\s+proportion\s*(?:\(CP\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bCP\s*[:=]\s*([0-9.]+)",
            ],
            validator=lambda v: 0 <= v <= 1,
        )

    if entropy_value is None:
        entropy_value = _extract_numeric_metric(
            lines=lines,
            full_text=text,
            line_keywords=["entropy", "shannon", " h "],
            inline_patterns=[
                r"(?i)(?:shannon\s+)?entropy\s*(?:\(H\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bH\s*[:=]\s*([0-9.]+)",
            ],
            full_patterns=[
                r"(?i)(?:shannon\s+)?entropy\s*(?:\(H\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bH\s*[:=]\s*([0-9.]+)",
            ],
            validator=lambda v: v >= 0,
        )

    if annotation is None:
        annotation = _extract_annotation_from_lines(lines)

    return cp_value, entropy_value, annotation


def _collect_cluster_annotations(
    predictions: dict[str, dict[str, str]],
    cluster: str,
) -> list[str]:
    """Collect normalized non-empty annotations for a single cluster."""
    cluster_annotations: list[str] = []
    for results in predictions.values():
        raw_annotation = results.get(cluster)
        if raw_annotation is None:
            continue
        annotation = str(raw_annotation).strip()
        if annotation and not is_unknown_annotation(annotation):
            cluster_annotations.append(annotation)
    return cluster_annotations


def _collect_candidate_clusters_from_raw_predictions(predictions: dict[str, Any]) -> list[str]:
    """Collect all cluster ids present in raw prediction payloads."""
    cluster_ids: set[str] = set()
    for raw_results in predictions.values():
        if not isinstance(raw_results, dict):
            continue
        for raw_cluster in raw_results:
            if raw_cluster is None:
                continue
            cluster_id = str(raw_cluster).strip()
            if cluster_id:
                cluster_ids.add(cluster_id)
    return sorted(cluster_ids, key=cluster_sort_key)


def _resolve_simple_cluster_consensus(
    cluster_annotations: list[str],
) -> tuple[str, float, float] | None:
    """Resolve deterministic consensus cases that do not require an LLM call."""
    if len(cluster_annotations) < 2:
        if cluster_annotations:
            return (
                cluster_annotations[0],
                DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
                DEFAULT_FALLBACK_ENTROPY,
            )
        return "Unknown", 0.0, 0.0

    # Trivial case: all annotations are identical strings — no LLM needed.
    if len(set(cluster_annotations)) == 1:
        return cluster_annotations[0], 1.0, 0.0

    return None


def _resolve_llm_cluster_consensus(
    *,
    cluster: str,
    cluster_annotations: list[str],
    has_any_api_key: bool,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    api_keys: dict[str, str],
    base_urls: str | dict[str, str] | None,
) -> tuple[str, float, float]:
    """Resolve non-trivial cluster consensus using LLM check and fallback policy."""
    if not has_any_api_key:
        return "Unknown", DEFAULT_FALLBACK_CONSENSUS_PROPORTION, DEFAULT_FALLBACK_ENTROPY

    prompt = create_consensus_check_prompt(cluster_annotations)
    llm_response = _call_llm_with_retry(
        prompt=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        max_retries=3,
        api_keys=api_keys,
        base_urls=base_urls,
    )

    if llm_response:
        llm_cp, llm_entropy, llm_prediction = _extract_metrics_from_text(llm_response)
        if llm_cp is not None and llm_entropy is not None:
            resolved_prediction = _normalize_predicted_label(llm_prediction)
            if resolved_prediction is None:
                resolved_prediction, _cp_unused, _entropy_unused = _deterministic_consensus_metrics(
                    cluster_annotations
                )
                write_log(
                    f"LLM metrics parsed but label missing for cluster {cluster}; "
                    f"falling back to deterministic majority '{resolved_prediction}'",
                    level="warning",
                )
            write_log(
                f"LLM consensus check for cluster {cluster}: "
                f"CP={llm_cp:.2f}, H={llm_entropy:.2f}",
                level="info",
            )
            return resolved_prediction, llm_cp, llm_entropy

    write_log(
        f"LLM consensus check failed for cluster {cluster}, returning Unknown",
        level="warning",
    )
    majority, cp_value, entropy_value = _deterministic_consensus_metrics(cluster_annotations)
    write_log(
        f"Using deterministic fallback for cluster {cluster}: "
        f"majority={majority}, CP={cp_value:.2f}, H={entropy_value:.2f}",
        level="warning",
    )
    return majority, cp_value, entropy_value


def _deterministic_consensus_metrics(labels: list[str]) -> tuple[str, float, float]:
    """Compute majority label, consensus proportion, and Shannon entropy from raw labels."""
    if not labels:
        return "Unknown", DEFAULT_FALLBACK_CONSENSUS_PROPORTION, DEFAULT_FALLBACK_ENTROPY

    counts: dict[str, int] = {}
    ordered_labels: list[str] = []
    for label in labels:
        if label not in counts:
            ordered_labels.append(label)
            counts[label] = 0
        counts[label] += 1

    majority = ordered_labels[0]
    majority_count = counts[majority]
    for label in ordered_labels[1:]:
        count = counts[label]
        if count > majority_count:
            majority = label
            majority_count = count

    n = len(labels)
    cp_value = majority_count / n

    entropy_value = 0.0
    for count in counts.values():
        p = count / n
        entropy_value -= p * math.log2(p)

    return majority, cp_value, entropy_value


def _extract_structured_cell_type_label(response_text: str) -> str | None:
    """Extract cell-type label from structured discussion response text."""
    patterns = [
        r"(?im)^\s*cell\s*type\s*(?:[:\-]|\uFF1A)\s*(.+?)\s*$",
        r"(?im)^\s*claim\s*(?:[:\-]|\uFF1A)\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if not match:
            continue
        label = _normalize_predicted_label(match.group(1))
        if label:
            return label
    return None


def _extract_conservative_free_text_label(response_text: str) -> str | None:
    """Extract a label from simple free-text responses using conservative heuristics."""
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None

    single_line = re.sub(r"^\s*(?:[-*]\s*)", "", lines[0]).strip()
    if not single_line:
        return None

    # Strip common uncertainty prefixes while keeping the candidate label.
    single_line = re.sub(
        r"(?i)^(?:likely|probably|possibly|maybe|it\s+is|it's|seems\s+to\s+be|appears\s+to\s+be)\s+",
        "",
        single_line,
    ).strip()
    if not single_line:
        return None

    # Avoid treating full narrative sentences as labels.
    if any(punct in single_line for punct in ".!?;"):
        return None
    if len(single_line) > 80 or len(single_line.split()) > 10:
        return None

    # Reject narrative/uncertainty phrasing that commonly appears in explanations.
    lowered_tokens = {
        token
        for token in re.split(r"[^A-Za-z0-9+]+", single_line.lower())
        if token
    }
    if lowered_tokens & {
        "candidate",
        "likely",
        "probably",
        "possibly",
        "maybe",
        "uncertain",
        "insufficient",
        "evidence",
        "suggest",
        "suggests",
        "given",
        "without",
        "label",
        "free",
        "text",
        "could",
        "might",
    }:
        return None

    # Keep this heuristic strict: only accept obvious cell-type-like labels.
    lower_line = single_line.lower()
    has_cell_word = re.search(r"\bcell(?:s)?\b", lower_line) is not None
    ends_with_known_cell_type = re.search(
        r"\b("
        r"lymphocyte|lymphocytes|monocyte|monocytes|macrophage|macrophages|"
        r"neutrophil|neutrophils|fibroblast|fibroblasts|astrocyte|astrocytes|"
        r"neuron|neurons|plasmablast|plasmablasts|megakaryocyte|megakaryocytes|"
        r"erythrocyte|erythrocytes|endothelial|epithelial|myeloid|treg|tregs|nk"
        r")\b$",
        lower_line,
    ) is not None
    if not (has_cell_word or ends_with_known_cell_type):
        return None

    return _normalize_predicted_label(single_line)


def _fallback_discussion_consensus_from_responses(
    *,
    valid_round_responses: dict[str, str],
    consensus_threshold: float,
    entropy_threshold: float,
) -> dict[str, Any]:
    """Fallback consensus for discussion round when LLM parsing/call fails."""
    extracted_labels = []
    for response in valid_round_responses.values():
        label = _extract_structured_cell_type_label(response)
        if label is None:
            label = _extract_conservative_free_text_label(response)
        if label:
            extracted_labels.append(label)

    if not extracted_labels:
        return DEFAULT_CONSENSUS_RESULT.copy()

    if len(extracted_labels) == 1:
        return {
            "reached": False,
            "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
            "entropy": DEFAULT_FALLBACK_ENTROPY,
            "majority_prediction": extracted_labels[0],
        }

    majority, cp_value, entropy_value = _deterministic_consensus_metrics(extracted_labels)
    reached = cp_value >= consensus_threshold and entropy_value <= entropy_threshold
    return {
        "reached": reached,
        "consensus_proportion": cp_value,
        "entropy": entropy_value,
        "majority_prediction": majority,
    }


def check_consensus(
    predictions: dict[str, Any],
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    api_keys: dict[str, str] | None = None,
    consensus_model: str | dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, float], dict[str, float], list[str]]:
    """Check consensus among different model predictions using LLM assistance.

    Args:
        predictions: Dictionary mapping model names to dictionaries of cluster annotations
        consensus_threshold: Agreement threshold below which a cluster is considered controversial
        entropy_threshold: Entropy threshold above which a cluster is considered controversial
        api_keys: Dictionary mapping provider names to API keys
        consensus_model: Optional model specification for consensus checking.
            Can be a string model name or dict with 'provider'/'model' keys.
            If not provided, picks from api_keys.
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs

    Returns:
        Tuple of (consensus, consensus_proportion, entropy, controversial_clusters)
    """
    consensus = {}
    consensus_proportion = {}
    entropy = {}

    if not isinstance(predictions, dict):
        raise ValueError(
            f"predictions must be a dict mapping model->cluster annotations, got {type(predictions).__name__}"
        )

    candidate_clusters = _collect_candidate_clusters_from_raw_predictions(predictions)
    api_keys = _normalize_api_keys(api_keys)
    consensus_model_dict = _normalize_consensus_model_spec(consensus_model)
    predictions = _normalize_predictions(predictions)

    # Filter out models with empty results, but keep models with valid annotations.
    # This allows consensus calculation even if some models failed.
    valid_predictions = {model: results for model, results in predictions.items() if results}

    # Check if we have enough valid predictions
    if not valid_predictions:
        write_log("No valid predictions from any model", level="warning")
        if not candidate_clusters:
            return {}, {}, {}, []
        consensus = {cluster: "Unknown" for cluster in candidate_clusters}
        consensus_proportion = {cluster: 0.0 for cluster in candidate_clusters}
        entropy = {cluster: 0.0 for cluster in candidate_clusters}
        is_controversial = consensus_threshold > 0.0 or entropy_threshold < 0.0
        controversial = candidate_clusters.copy() if is_controversial else []
        return consensus, consensus_proportion, entropy, controversial

    if len(valid_predictions) < len(predictions):
        failed_models = set(predictions.keys()) - set(valid_predictions.keys())
        write_log(
            f"Some models returned empty results, excluding: {failed_models}",
            level="warning",
        )

    # Use filtered predictions for the rest of the function
    predictions = valid_predictions

    # Get all clusters (sorted for deterministic output order)
    # Include raw candidate clusters so blank/None-only clusters are not dropped.
    all_clusters_set: set[str] = set(candidate_clusters)
    for model_results in predictions.values():
        all_clusters_set.update(model_results.keys())
    all_clusters = sorted(all_clusters_set, key=cluster_sort_key)

    primary_provider, primary_model, primary_api_key = _resolve_consensus_provider(
        consensus_model_dict, api_keys
    )
    has_any_api_key = bool(primary_api_key) or _has_supported_api_key(api_keys)

    # Process each cluster
    for cluster in all_clusters:
        cluster_annotations = _collect_cluster_annotations(predictions, cluster)

        simple_result = _resolve_simple_cluster_consensus(cluster_annotations)
        if simple_result is not None:
            label, cp_value, entropy_value = simple_result
            consensus[cluster] = label
            consensus_proportion[cluster] = cp_value
            entropy[cluster] = entropy_value
            continue

        label, cp_value, entropy_value = _resolve_llm_cluster_consensus(
            cluster=cluster,
            cluster_annotations=cluster_annotations,
            has_any_api_key=has_any_api_key,
            provider=primary_provider,
            model=primary_model,
            api_key=primary_api_key,
            api_keys=api_keys,
            base_urls=base_urls,
        )
        consensus[cluster] = label
        consensus_proportion[cluster] = cp_value
        entropy[cluster] = entropy_value

    # Find controversial clusters based on both consensus proportion and entropy
    # (sorted for deterministic output order)
    controversial = sorted(
        (
            cluster
            for cluster, score in consensus_proportion.items()
            if score < consensus_threshold or entropy.get(cluster, 0) > entropy_threshold
        ),
        key=cluster_sort_key,
    )
    return consensus, consensus_proportion, entropy, controversial


def _extract_cell_type_via_llm(
    text: str,
    provider: str,
    model: str,
    api_key: str | None,
    api_keys: dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> str | None:
    """Use LLM to extract a concise cell-type label from free-text.

    Used for single-response extraction where the discussion consensus
    prompt is not applicable.  The prompt explicitly asks for only the
    cell-type name, and the result is validated with a length bound as
    a sanity check (legitimate cell-type labels are ≤60 characters).

    Args:
        text: Free-text response containing a cell-type reference
        provider: LLM provider to use
        model: LLM model to use
        api_key: API key for the provider
        api_keys: Dictionary of API keys (for fallback)
        base_urls: Custom base URLs for API endpoints

    Returns:
        str | None: Clean cell-type label or None if extraction failed
    """
    prompt = create_cell_type_extraction_prompt(text)
    response = _call_llm_with_retry(
        prompt=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        max_retries=2,
        api_keys=api_keys,
        base_urls=base_urls,
    )
    if response:
        cell_type = _normalize_predicted_label(response)
        if cell_type and len(cell_type) <= 60:
            write_log(f"LLM extracted cell type: '{cell_type}'", level="debug")
            return cell_type
        write_log(
            f"LLM extraction returned unsuitable result: '{response.strip()}'",
            level="warning",
        )
    return None


def check_consensus_for_discussion_round(
    round_responses: dict[str, Any],
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    api_keys: dict[str, str] | None = None,
    consensus_model: str | dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check consensus among model responses for a single discussion round.

    LLM-native approach mirroring R's check_consensus (check_consensus.R:488):
    the LLM receives raw discussion responses and performs both cell-type
    extraction and consensus checking in a single call, avoiding the
    sentence-as-label problem that arises from regex-based extraction.

    1. Empty input: returns default Unknown result immediately (no LLM needed)
    2. Single response: LLM extraction to produce a clean label
    3. Multiple responses: ONE LLM call with raw responses

    Args:
        round_responses: Dictionary mapping model names to their responses
        consensus_threshold: Agreement threshold (default: 0.7)
        entropy_threshold: Entropy threshold (default: 1.0)
        api_keys: Dictionary mapping provider names to API keys
        consensus_model: Optional model specification for consensus checking.
            Can be a string model name or dict with 'provider'/'model' keys.
        base_urls: Custom base URLs for API endpoints

    Returns:
        dict containing:
            - reached: bool, whether consensus was reached
            - consensus_proportion: float, the consensus proportion
            - entropy: float, the Shannon entropy
            - majority_prediction: str, the majority cell type prediction
    """
    valid_round_responses = _collect_valid_round_responses(round_responses)

    # Empty — nothing to check, no LLM needed
    if not valid_round_responses:
        write_log("No responses to check consensus", level="warning")
        return DEFAULT_CONSENSUS_RESULT.copy()

    api_keys = _normalize_api_keys(api_keys)
    consensus_model_dict = _normalize_consensus_model_spec(consensus_model)

    primary_provider, primary_model, primary_api_key = _resolve_consensus_provider(
        consensus_model_dict, api_keys
    )

    # Single response — first try deterministic structured extraction,
    # then optionally use LLM extraction as a fallback enhancer.
    if len(valid_round_responses) == 1:
        deterministic_result = _fallback_discussion_consensus_from_responses(
            valid_round_responses=valid_round_responses,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
        )
        deterministic_prediction = _normalize_predicted_label(
            deterministic_result.get("majority_prediction")
        )
        if deterministic_prediction:
            write_log(
                "Only 1 response, using structured label extraction without LLM",
                level="info",
            )
            deterministic_result["majority_prediction"] = deterministic_prediction
            return deterministic_result

        write_log("Only 1 response, extracting label via LLM", level="warning")
        single_response = next(iter(valid_round_responses.values()))
        cell_type = _extract_cell_type_via_llm(
            single_response,
            primary_provider,
            primary_model,
            primary_api_key,
            api_keys=api_keys,
            base_urls=base_urls,
        )
        return {
            "reached": False,
            "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
            "entropy": DEFAULT_FALLBACK_ENTROPY,
            "majority_prediction": cell_type or "Unknown",
        }

    # --- Primary path: LLM-native consensus check ---
    # Send raw responses to LLM for simultaneous extraction + consensus.
    # This mirrors R's approach (check_consensus.R:488) where the LLM
    # receives full discussion responses, extracts clean cell-type labels,
    # and calculates consensus metrics — all in one call.
    prompt = create_discussion_consensus_prompt(
        valid_round_responses,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
    )

    llm_response = _call_llm_with_retry(
        prompt=prompt,
        provider=primary_provider,
        model=primary_model,
        api_key=primary_api_key,
        max_retries=3,
        api_keys=api_keys,
        base_urls=base_urls,
    )

    if llm_response:
        cp_value, entropy_value, llm_prediction = _extract_metrics_from_text(llm_response)

        if cp_value is not None and entropy_value is not None:
            resolved_prediction = _normalize_predicted_label(llm_prediction)
            if resolved_prediction is None:
                fallback_result = _fallback_discussion_consensus_from_responses(
                    valid_round_responses=valid_round_responses,
                    consensus_threshold=consensus_threshold,
                    entropy_threshold=entropy_threshold,
                )
                fallback_prediction = _normalize_predicted_label(
                    fallback_result.get("majority_prediction")
                )
                if fallback_prediction:
                    resolved_prediction = fallback_prediction
                    write_log(
                        "LLM metrics parsed but label missing; using structured-response label fallback",
                        level="warning",
                    )

            reached = cp_value >= consensus_threshold and entropy_value <= entropy_threshold

            write_log(
                f"LLM consensus check: CP={cp_value:.2f}, H={entropy_value:.2f}, "
                f"reached={reached}, prediction={resolved_prediction}",
                level="info",
            )

            return {
                "reached": reached,
                "consensus_proportion": cp_value,
                "entropy": entropy_value,
                "majority_prediction": resolved_prediction or "Unknown",
            }

    # LLM call failed/returned unparseable output.
    # Fall back to structured text extraction from discussion responses.
    fallback_result = _fallback_discussion_consensus_from_responses(
        valid_round_responses=valid_round_responses,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
    )
    write_log(
        "LLM consensus check failed; using structured-response fallback",
        level="warning",
    )
    return fallback_result


def _build_discussion_model_info(
    models: list[str | dict[str, str]],
    api_keys: dict[str, str],
    base_urls: str | dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Build runnable discussion model descriptors with de-dup and key resolution."""
    model_info_list: list[dict[str, Any]] = []
    seen_model_keys: set[str] = set()

    for model_item in models:
        provider, model_name = _resolve_model_spec(model_item)
        if not provider:
            write_log(f"Could not determine provider for model {model_name}, skipping", level="warning")
            continue

        api_key = api_keys.get(provider) or load_api_key(provider)
        if not api_key:
            write_log(f"No API key for {provider}, skipping {model_name}", level="warning")
            continue

        model_key = f"{provider}:{model_name}"
        if model_key in seen_model_keys:
            write_log(f"Duplicate discussion model '{model_key}' detected, skipping", level="warning")
            continue
        seen_model_keys.add(model_key)

        model_info_list.append(
            {
                "key": model_key,
                "name": model_name,
                "provider": provider,
                "api_key": api_key,
                "base_url": resolve_provider_base_url(provider, base_urls),
            }
        )

    return model_info_list


def _build_discussion_round_prompt(
    *,
    cluster_id: str,
    marker_genes: list[str],
    rounds_history: list[dict[str, str]],
    initial_predictions: dict[str, str],
    current_round: int,
    species: str,
    tissue: str | None,
) -> str:
    """Build prompt for a single discussion round."""
    if current_round == 1:
        return create_initial_discussion_prompt(
            cluster_id=cluster_id,
            marker_genes=marker_genes,
            initial_predictions=initial_predictions,
            species=species,
            tissue=tissue,
        )

    return create_discussion_prompt(
        cluster_id=cluster_id,
        marker_genes=marker_genes,
        previous_rounds=rounds_history,
        round_number=current_round,
        species=species,
        tissue=tissue,
    )


def _run_discussion_round(
    *,
    model_info_list: list[dict[str, Any]],
    prompt: str,
    current_round: int,
    use_cache: bool,
    force_rerun: bool,
    cache_dir: str | None,
) -> dict[str, str]:
    """Execute one discussion round and collect responses from all models."""
    round_responses: dict[str, str] = {}
    for model_info in model_info_list:
        model_key = model_info["key"]
        try:
            response = get_model_response(
                prompt=prompt,
                provider=model_info["provider"],
                model=model_info["name"],
                api_key=model_info["api_key"],
                use_cache=use_cache and not force_rerun,
                cache_dir=cache_dir,
                base_url=model_info["base_url"],
            )
            round_responses[model_key] = response
            write_log(f"Got response from {model_key} in round {current_round}")
        except RECOVERABLE_LLM_EXCEPTIONS as e:
            write_log(
                f"Error getting response from {model_key}: {e!s}",
                level="warning",
            )
            round_responses[model_key] = f"Error: {e!s}"
    return round_responses


def _resolve_last_round_decision(
    *,
    rounds_history: list[dict[str, str]],
    consensus_threshold: float,
    entropy_threshold: float,
    api_keys: dict[str, str],
    consensus_model: str | dict[str, str] | None,
    base_urls: str | dict[str, str] | None,
) -> tuple[str | None, float, float]:
    """Resolve final decision from last discussion round when no early consensus was reached."""
    if not rounds_history:
        return None, DEFAULT_FALLBACK_CONSENSUS_PROPORTION, DEFAULT_FALLBACK_ENTROPY

    last_round = rounds_history[-1]
    valid_last = _collect_valid_round_responses(last_round)
    if not valid_last:
        return None, DEFAULT_FALLBACK_CONSENSUS_PROPORTION, DEFAULT_FALLBACK_ENTROPY

    last_consensus = check_consensus_for_discussion_round(
        round_responses=valid_last,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        api_keys=api_keys,
        consensus_model=consensus_model,
        base_urls=base_urls,
    )
    decision = last_consensus["majority_prediction"]
    if not isinstance(decision, str) or is_unknown_annotation(decision):
        decision = None

    return (
        decision,
        last_consensus["consensus_proportion"],
        last_consensus["entropy"],
    )


def _build_cluster_initial_predictions(
    model_predictions: dict[str, dict[str, str]],
    cluster_id: str,
) -> dict[str, str]:
    """Build initial per-model annotations for a target cluster."""
    return {
        model_name: str(predictions.get(cluster_id, "Unknown"))
        for model_name, predictions in model_predictions.items()
        if cluster_id in predictions
    }


def _run_cluster_discussion_rounds(
    *,
    cluster_id: str,
    marker_genes: list[str],
    initial_predictions: dict[str, str],
    model_info_list: list[dict[str, Any]],
    species: str,
    tissue: str | None,
    max_discussion_rounds: int,
    consensus_threshold: float,
    entropy_threshold: float,
    api_keys: dict[str, str],
    consensus_model: str | dict[str, str] | None,
    use_cache: bool,
    force_rerun: bool,
    cache_dir: str | None,
    base_urls: str | dict[str, str] | None,
) -> tuple[str | None, list[dict[str, str]], float, float]:
    """Execute discussion rounds for a cluster and return final decision + metrics."""
    rounds_history: list[dict[str, str]] = []
    final_decision: str | None = None
    current_cp = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
    current_h = DEFAULT_FALLBACK_ENTROPY

    for current_round in range(1, max_discussion_rounds + 1):
        write_log(f"Starting round {current_round} for cluster {cluster_id}")

        prompt = _build_discussion_round_prompt(
            cluster_id=cluster_id,
            marker_genes=marker_genes,
            rounds_history=rounds_history,
            initial_predictions=initial_predictions,
            current_round=current_round,
            species=species,
            tissue=tissue,
        )

        round_responses = _run_discussion_round(
            model_info_list=model_info_list,
            prompt=prompt,
            current_round=current_round,
            use_cache=use_cache,
            force_rerun=force_rerun,
            cache_dir=cache_dir,
        )
        rounds_history.append(round_responses)

        valid_responses = _collect_valid_round_responses(round_responses)
        if len(valid_responses) < 2:
            write_log(
                f"Only {len(valid_responses)} valid responses in round {current_round}",
                level="warning",
            )
            continue

        consensus_result = check_consensus_for_discussion_round(
            round_responses=valid_responses,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            api_keys=api_keys,
            consensus_model=consensus_model,
            base_urls=base_urls,
        )

        current_cp = consensus_result["consensus_proportion"]
        current_h = consensus_result["entropy"]
        majority = consensus_result["majority_prediction"]

        write_log(
            f"Round {current_round} consensus: CP={current_cp:.2f}, H={current_h:.2f}, "
            f"majority={majority}, reached={consensus_result['reached']}",
            level="info",
        )

        if consensus_result["reached"]:
            final_decision = majority
            write_log(
                f"Consensus reached in round {current_round} for cluster {cluster_id}: {final_decision}",
                level="info",
            )
            break

    if not final_decision and rounds_history:
        (
            last_round_decision,
            last_round_cp,
            last_round_h,
        ) = _resolve_last_round_decision(
            rounds_history=rounds_history,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            api_keys=api_keys,
            consensus_model=consensus_model,
            base_urls=base_urls,
        )
        final_decision = last_round_decision
        current_cp = last_round_cp
        current_h = last_round_h
        if last_round_decision:
            write_log(
                f"Using last round majority for cluster {cluster_id}: {last_round_decision} "
                f"(CP={current_cp:.2f}, H={current_h:.2f})",
                level="info",
            )

    return final_decision, rounds_history, current_cp, current_h


def _store_cluster_discussion_outcome(
    *,
    cluster_id: str,
    final_decision: str | None,
    rounds_history: list[dict[str, str]],
    current_cp: float,
    current_h: float,
    results: dict[str, str],
    discussion_history: dict[str, list[dict]],
    updated_consensus_proportion: dict[str, float],
    updated_entropy: dict[str, float],
) -> None:
    """Store final cluster result and metrics into aggregate output dictionaries."""
    if final_decision and not is_unknown_annotation(final_decision):
        results[cluster_id] = normalize_annotation(final_decision)
    else:
        results[cluster_id] = "Unknown"

    discussion_history[cluster_id] = rounds_history
    updated_consensus_proportion[cluster_id] = current_cp
    updated_entropy[cluster_id] = current_h


def _normalize_discussion_model_predictions(
    model_predictions: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Normalize discussion model predictions to trimmed model/cluster keys."""
    normalized_predictions: dict[str, dict[str, Any]] = {}
    for raw_model_name, raw_predictions in model_predictions.items():
        model_name = str(raw_model_name).strip() or str(raw_model_name)
        if not isinstance(raw_predictions, dict):
            write_log(
                f"Model '{model_name}' discussion predictions must be a dict, "
                f"got {type(raw_predictions).__name__}; skipping",
                level="warning",
            )
            continue

        normalized_cluster_map: dict[str, Any] = {}
        for raw_cluster_id, annotation in raw_predictions.items():
            if raw_cluster_id is None or raw_cluster_id != raw_cluster_id:
                continue
            cluster_id = str(raw_cluster_id).strip()
            if cluster_id:
                normalized_cluster_map[cluster_id] = annotation

        if model_name in normalized_predictions:
            write_log(
                f"Duplicate discussion model name after normalization: '{model_name}', "
                "merging cluster predictions",
                level="warning",
            )
            normalized_predictions[model_name].update(normalized_cluster_map)
        else:
            normalized_predictions[model_name] = normalized_cluster_map

    return normalized_predictions


def _normalize_controversial_cluster_ids(controversial_clusters: list[Any]) -> list[str]:
    """Normalize controversial cluster IDs with trim/filter/dedup semantics."""
    normalized_clusters: list[str] = []
    seen: set[str] = set()
    skipped_invalid = 0
    for raw_cluster_id in controversial_clusters:
        if raw_cluster_id is None or raw_cluster_id != raw_cluster_id:
            skipped_invalid += 1
            continue
        cluster_id = str(raw_cluster_id).strip()
        if not cluster_id:
            skipped_invalid += 1
            continue
        if cluster_id not in seen:
            seen.add(cluster_id)
            normalized_clusters.append(cluster_id)

    if skipped_invalid:
        write_log(
            f"Ignored {skipped_invalid} invalid controversial cluster IDs after normalization",
            level="warning",
        )

    return normalized_clusters


def process_controversial_clusters(
    marker_genes: dict[str, list[str]],
    controversial_clusters: list[str],
    model_predictions: dict[str, dict[str, str]],
    species: str,
    tissue: str | None = None,
    models: list[str | dict[str, str]] | None = None,
    api_keys: dict[str, str] | None = None,
    max_discussion_rounds: int = 3,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    use_cache: bool = True,
    cache_dir: str | None = None,
    base_urls: str | dict[str, str] | None = None,
    force_rerun: bool = False,
    consensus_model: str | dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, float], dict[str, float]]:
    """Process controversial clusters through multi-model discussion.

    This function facilitates a real discussion between multiple LLMs, where each
    model can see and respond to other models' arguments. This mirrors the R
    implementation where all models participate in each round of discussion.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        controversial_clusters: List of controversial cluster IDs
        model_predictions: Dictionary mapping model names to dictionaries of
            cluster annotations (from initial annotation phase)
        species: Species name (e.g., 'human', 'mouse')
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        models: List of models to participate in the discussion. Each item can be
            a string (model name) or dict with 'provider' and 'model' keys.
        api_keys: Dictionary mapping provider names to API keys
        max_discussion_rounds: Maximum number of discussion rounds
        consensus_threshold: Consensus proportion threshold for agreement
        entropy_threshold: Entropy threshold for agreement
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        base_urls: Custom base URLs for API endpoints
        force_rerun: If True, ignore cached results
        consensus_model: Optional model specification for consensus checking.
            Can be a string model name or dict with 'provider'/'model' keys.
            If not provided, picks from the caller's api_keys.

    Returns:
        tuple containing:
            - results: Dict mapping cluster IDs to resolved annotations
            - discussion_history: Dict mapping cluster IDs to list of round responses
            - updated_consensus_proportion: Dict mapping cluster IDs to CP scores
            - updated_entropy: Dict mapping cluster IDs to entropy scores
    """
    if not models:
        write_log("No models provided for discussion", level="error")
        return {}, {}, {}, {}

    if isinstance(controversial_clusters, (str, bytes)) or not isinstance(
        controversial_clusters, (list, tuple, set)
    ):
        raise ValueError(
            "controversial_clusters must be a list/tuple/set of cluster IDs, not a string"
        )

    if not isinstance(model_predictions, dict):
        raise ValueError(
            "model_predictions must be a dict mapping model name to cluster predictions"
        )

    _validate_models_spec(models)

    api_keys = _normalize_api_keys(api_keys)

    # Normalize cluster keys for consistent lookups.
    marker_genes = normalize_marker_genes_keys(marker_genes)
    model_predictions = _normalize_discussion_model_predictions(model_predictions)
    controversial_clusters = _normalize_controversial_cluster_ids(controversial_clusters)

    results = {}
    discussion_history = {}
    updated_consensus_proportion = {}
    updated_entropy = {}

    model_info_list = _build_discussion_model_info(models, api_keys, base_urls)

    if len(model_info_list) < 2:
        write_log(
            f"Need at least 2 models for discussion, only have {len(model_info_list)}",
            level="error",
        )
        return {}, {}, {}, {}

    write_log(f"Starting multi-model discussion with {len(model_info_list)} models")

    for cluster_id in controversial_clusters:
        write_log(f"Processing controversial cluster {cluster_id}")

        current_marker_genes = marker_genes.get(cluster_id, [])
        if not current_marker_genes:
            write_log(f"No marker genes found for cluster {cluster_id}", level="warning")
            results[cluster_id] = "Unknown"
            discussion_history[cluster_id] = []
            updated_consensus_proportion[cluster_id] = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
            updated_entropy[cluster_id] = DEFAULT_FALLBACK_ENTROPY
            continue

        initial_predictions = _build_cluster_initial_predictions(model_predictions, cluster_id)

        try:
            final_decision, rounds_history, current_cp, current_h = _run_cluster_discussion_rounds(
                cluster_id=cluster_id,
                marker_genes=current_marker_genes,
                initial_predictions=initial_predictions,
                model_info_list=model_info_list,
                species=species,
                tissue=tissue,
                max_discussion_rounds=max_discussion_rounds,
                consensus_threshold=consensus_threshold,
                entropy_threshold=entropy_threshold,
                api_keys=api_keys,
                consensus_model=consensus_model,
                use_cache=use_cache,
                force_rerun=force_rerun,
                cache_dir=cache_dir,
                base_urls=base_urls,
            )
            _store_cluster_discussion_outcome(
                cluster_id=cluster_id,
                final_decision=final_decision,
                rounds_history=rounds_history,
                current_cp=current_cp,
                current_h=current_h,
                results=results,
                discussion_history=discussion_history,
                updated_consensus_proportion=updated_consensus_proportion,
                updated_entropy=updated_entropy,
            )

        except RECOVERABLE_LLM_EXCEPTIONS as e:
            write_log(f"Error during discussion for cluster {cluster_id}: {e!s}", level="error")
            results[cluster_id] = "Unknown"
            discussion_history[cluster_id] = []
            updated_consensus_proportion[cluster_id] = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
            updated_entropy[cluster_id] = DEFAULT_FALLBACK_ENTROPY

    return results, discussion_history, updated_consensus_proportion, updated_entropy


def _filter_marker_genes_for_clusters(
    marker_genes: dict[str, list[str]],
    clusters_to_analyze: list[str] | None,
) -> dict[str, list[str]]:
    """Filter marker genes to requested clusters with validation and stable order."""
    if clusters_to_analyze is None:
        return marker_genes

    if isinstance(clusters_to_analyze, (str, bytes)) or not isinstance(
        clusters_to_analyze, (list, tuple, set)
    ):
        raise ValueError(
            "clusters_to_analyze must be a list/tuple/set of cluster IDs, not a string"
        )

    requested_clusters: list[str] = []
    skipped_empty: list[str] = []
    for cluster_id in clusters_to_analyze:
        normalized = str(cluster_id).strip()
        if not normalized:
            skipped_empty.append(str(cluster_id))
            continue
        requested_clusters.append(normalized)

    # Deduplicate while preserving caller-specified order.
    requested_clusters = list(dict.fromkeys(requested_clusters))

    if skipped_empty:
        write_log(
            "Ignored empty cluster IDs in clusters_to_analyze",
            level="warning",
        )

    available_clusters = list(marker_genes.keys())
    valid_clusters = [cluster_id for cluster_id in requested_clusters if cluster_id in available_clusters]
    invalid_clusters = [cluster_id for cluster_id in requested_clusters if cluster_id not in available_clusters]

    if invalid_clusters:
        warning_msg = f"The following cluster IDs were not found in the input: {', '.join(invalid_clusters)}"
        write_log(warning_msg, level="warning")

    if not valid_clusters:
        error_msg = "None of the specified clusters exist in the input data."
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    filtered = {cluster_id: marker_genes[cluster_id] for cluster_id in valid_clusters}
    log_msg = f"Filtered to analyze {len(valid_clusters)} clusters: {', '.join(valid_clusters)}"
    write_log(log_msg)
    return filtered


def _ensure_consensus_model_api_key(
    api_keys: dict[str, str],
    consensus_model_dict: dict[str, str] | None,
) -> dict[str, str]:
    """Ensure consensus-model provider key is available when provider is specified."""
    if not consensus_model_dict:
        return api_keys

    consensus_provider = consensus_model_dict.get("provider")
    if consensus_provider and consensus_provider not in api_keys:
        api_key = load_api_key(consensus_provider)
        if api_key:
            api_keys[consensus_provider] = api_key
    return api_keys


def _run_initial_annotations(
    *,
    marker_genes: dict[str, list[str]],
    species: str,
    models: list[str | dict[str, str]],
    api_keys: dict[str, str],
    tissue: str | None,
    additional_context: str | None,
    use_cache: bool,
    force_rerun: bool,
    cache_dir: str | None,
    base_urls: str | dict[str, str] | None,
    verbose: bool,
) -> dict[str, dict[str, str]]:
    """Run initial annotation phase across models with de-dup and error isolation."""
    model_results: dict[str, dict[str, str]] = {}
    seen_model_keys: set[str] = set()

    for model_item in models:
        provider, model_name = _resolve_model_spec(model_item)
        if not provider:
            write_log(
                f"Warning: Could not determine provider for model {model_name}, skipping",
                level="warning",
            )
            continue

        api_key = api_keys.get(provider)
        if not api_key:
            write_log(
                f"Warning: No API key found for {provider}, skipping {model_name}",
                level="warning",
            )
            continue

        model_key = f"{provider}:{model_name}"
        if model_key in seen_model_keys:
            write_log(f"Warning: Duplicate model '{model_key}' detected, skipping", level="warning")
            continue
        seen_model_keys.add(model_key)

        if verbose:
            write_log(f"Annotating with {model_key}")

        try:
            results = annotate_clusters(
                marker_genes=marker_genes,
                species=species,
                provider=provider,
                model=model_name,
                api_key=api_key,
                tissue=tissue,
                additional_context=additional_context,
                use_cache=use_cache and not force_rerun,
                cache_dir=cache_dir,
                base_urls=base_urls,
            )
            model_results[model_key] = results
            if verbose:
                write_log(f"Successfully annotated with {model_key}")
        except RECOVERABLE_LLM_EXCEPTIONS as e:
            write_log(f"Error annotating with {model_key}: {e!s}", level="error")

    return model_results


def _clean_annotations(annotations: dict[str, Any]) -> dict[str, str]:
    """Normalize annotation dictionary to non-empty strings."""
    cleaned: dict[str, str] = {}
    for cid, ann in annotations.items():
        cleaned[cid] = "Unknown" if is_unknown_annotation(ann) else str(ann).strip()
    return cleaned


def _prepare_interactive_annotation_context(
    *,
    marker_genes: dict[str, list[str]],
    species: str,
    tissue: str | None,
    models: list[str | dict[str, str]] | None,
    api_keys: dict[str, str] | None,
    consensus_model: str | dict[str, str] | None,
    clusters_to_analyze: list[str] | None,
    consensus_threshold: float,
    entropy_threshold: float,
    max_discussion_rounds: int,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, list[str]], list[str | dict[str, str]], dict[str, str], dict[str, str] | None]:
    """Prepare normalized context for interactive consensus orchestration."""
    if not models:
        raise ValueError("models must be a non-empty list of model specifications")

    _validate_models_spec(models)

    metadata = _build_metadata(
        models=models,
        species=species,
        tissue=tissue,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        max_discussion_rounds=max_discussion_rounds,
    )

    marker_genes_normalized = normalize_marker_genes_keys(marker_genes)
    if verbose:
        write_log("Starting interactive consensus annotation")
    marker_genes_filtered = _filter_marker_genes_for_clusters(
        marker_genes_normalized,
        clusters_to_analyze,
    )

    resolved_api_keys = _resolve_api_keys_for_models(models, api_keys)
    consensus_model_dict = _normalize_consensus_model_spec(consensus_model)
    resolved_api_keys = _ensure_consensus_model_api_key(resolved_api_keys, consensus_model_dict)

    return metadata, marker_genes_filtered, models, resolved_api_keys, consensus_model_dict


def _update_metrics_for_resolved_clusters(
    consensus_proportion: dict[str, float],
    entropy: dict[str, float],
    updated_cp: dict[str, float],
    updated_h: dict[str, float],
) -> None:
    """Merge updated metrics from discussion resolution into consensus metrics."""
    for cluster_id, cp_value in updated_cp.items():
        consensus_proportion[cluster_id] = cp_value
    for cluster_id, h_value in updated_h.items():
        entropy[cluster_id] = h_value


def _resolve_controversial_clusters_if_needed(
    *,
    controversial: list[str],
    marker_genes: dict[str, list[str]],
    model_results: dict[str, dict[str, str]],
    species: str,
    tissue: str | None,
    models: list[str | dict[str, str]],
    api_keys: dict[str, str],
    max_discussion_rounds: int,
    consensus_threshold: float,
    entropy_threshold: float,
    use_cache: bool,
    cache_dir: str | None,
    base_urls: str | dict[str, str] | None,
    force_rerun: bool,
    consensus_model: dict[str, str] | None,
    verbose: bool,
) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, float], dict[str, float]]:
    """Resolve controversial clusters via discussion, returning resolution + updated metrics."""
    if not controversial:
        return {}, {}, {}, {}

    if verbose:
        write_log(
            f"Resolving {len(controversial)} controversial clusters through multi-model discussion"
        )

    try:
        return process_controversial_clusters(
            marker_genes=marker_genes,
            controversial_clusters=controversial,
            model_predictions=model_results,
            species=species,
            tissue=tissue,
            models=models,
            api_keys=api_keys,
            max_discussion_rounds=max_discussion_rounds,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            use_cache=use_cache,
            cache_dir=cache_dir,
            base_urls=base_urls,
            force_rerun=force_rerun,
            consensus_model=consensus_model,
        )
    except RECOVERABLE_LLM_EXCEPTIONS as e:
        write_log(f"Error resolving controversial clusters: {e!s}", level="error")
        return {}, {}, {}, {}


def _merge_consensus_and_resolved(
    consensus: dict[str, str],
    resolved: dict[str, str],
) -> dict[str, str]:
    """Merge resolved controversial results into base consensus annotations.

    Principle:
    - Never downgrade a known base consensus label to Unknown.
    - Still allow Unknown to fill missing clusters when base consensus lacks them.
    """
    final_annotations = consensus.copy()
    for cluster_id, resolved_label in resolved.items():
        normalized_resolved = normalize_annotation(resolved_label)
        base_label = final_annotations.get(cluster_id)

        if (
            normalized_resolved == "Unknown"
            and base_label is not None
            and not is_unknown_annotation(base_label)
        ):
            # Keep known base consensus when discussion adds no better signal.
            continue

        final_annotations[cluster_id] = normalized_resolved
    return final_annotations


def interactive_consensus_annotation(
    marker_genes: dict[str, list[str]],
    species: str,
    models: list[str | dict[str, str]] | None = None,
    api_keys: dict[str, str] | None = None,
    tissue: str | None = None,
    additional_context: str | None = None,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    max_discussion_rounds: int = 3,
    use_cache: bool = True,
    cache_dir: str | None = None,
    verbose: bool = False,
    consensus_model: str | dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
    clusters_to_analyze: list[str] | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Perform consensus annotation of cell types using multiple LLMs and interactive resolution.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        models: List of models to use for annotation
        api_keys: Dictionary mapping provider names to API keys
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        consensus_threshold: Agreement threshold below which a cluster is considered controversial
        entropy_threshold: Entropy threshold above which a cluster is considered controversial
        max_discussion_rounds: Maximum number of discussion rounds for controversial clusters
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        verbose: Whether to print detailed logs
        consensus_model: Optional model specification for consensus checking and discussion.
            Can be a string (model name) or dict with 'provider' and 'model' keys.
            If not provided, picks from the user's available api_keys.
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs
        clusters_to_analyze: Optional list of cluster IDs to analyze. If provided,
            only the specified clusters will be processed. Cluster IDs must exist
            in the marker_genes dictionary. Non-existent cluster IDs will be
            ignored with a warning. If None (default), all clusters will be analyzed.
        force_rerun: If True, ignore cached results and force re-analysis of all
            specified clusters, including both the initial annotation phase and the
            discussion phase for controversial clusters. Useful when you want to
            re-analyze clusters with different context or for subtype identification.
            Default is False. Only effective when use_cache is True.

    Returns:
        dict[str, Any]: Dictionary containing consensus results and metadata

    """
    metadata, marker_genes, models, api_keys, consensus_model_dict = (
        _prepare_interactive_annotation_context(
            marker_genes=marker_genes,
            species=species,
            tissue=tissue,
            models=models,
            api_keys=api_keys,
            consensus_model=consensus_model,
            clusters_to_analyze=clusters_to_analyze,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            max_discussion_rounds=max_discussion_rounds,
            verbose=verbose,
        )
    )

    model_results = _run_initial_annotations(
        marker_genes=marker_genes,
        species=species,
        models=models,
        api_keys=api_keys,
        tissue=tissue,
        additional_context=additional_context,
        use_cache=use_cache,
        force_rerun=force_rerun,
        cache_dir=cache_dir,
        base_urls=base_urls,
        verbose=verbose,
    )

    # Check if we have any results
    if not model_results:
        write_log("No annotations were successful", level="error")
        return _build_interactive_result(
            metadata=metadata,
            error="No annotations were successful",
        )

    # Check consensus
    consensus, consensus_proportion, entropy, controversial = check_consensus(
        model_results,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        api_keys=api_keys,
        consensus_model=consensus_model_dict,
        base_urls=base_urls,
    )

    if verbose:
        write_log(f"Found {len(controversial)} controversial clusters out of {len(consensus)}")

    resolved, discussion_logs, updated_cp, updated_h = _resolve_controversial_clusters_if_needed(
        controversial=controversial,
        marker_genes=marker_genes,
        model_results=model_results,
        species=species,
        tissue=tissue,
        models=models,
        api_keys=api_keys,
        max_discussion_rounds=max_discussion_rounds,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        use_cache=use_cache,
        cache_dir=cache_dir,
        base_urls=base_urls,
        force_rerun=force_rerun,
        consensus_model=consensus_model_dict,
        verbose=verbose,
    )
    _update_metrics_for_resolved_clusters(consensus_proportion, entropy, updated_cp, updated_h)
    if verbose and resolved:
        write_log(f"Successfully resolved {len(resolved)} controversial clusters")

    cleaned_annotations = _clean_annotations(_merge_consensus_and_resolved(consensus, resolved))

    return _build_interactive_result(
        metadata=metadata,
        consensus=cleaned_annotations,
        consensus_proportion=consensus_proportion,
        entropy=entropy,
        controversial_clusters=controversial,
        resolved=resolved,
        model_annotations=model_results,
        discussion_logs=discussion_logs,
    )


def _format_metadata_model_names(raw_models: list[Any]) -> list[str]:
    """Format model descriptors from metadata to human-readable names."""
    return [
        model if isinstance(model, str) else f"{model.get('provider', '')}:{model.get('model', '')}"
        for model in raw_models
    ]


def _resolve_clusters_to_report(
    *,
    consensus: dict[str, Any],
    cluster_id: str | None,
) -> list[str]:
    """Resolve report target clusters with deterministic ordering."""
    if cluster_id is not None:
        return [cluster_id] if cluster_id in consensus else []
    return sorted(consensus.keys(), key=cluster_sort_key)


def _append_report_header(
    *,
    lines: list[str],
    separator: str,
    metadata: dict[str, Any],
) -> None:
    """Append report header section."""
    lines.append(separator)
    lines.append("MULTI-LLM CONSENSUS DISCUSSION REPORT")
    lines.append(separator)
    lines.append(f"Generated: {metadata.get('timestamp', 'N/A')}")
    lines.append(f"Species: {metadata.get('species', 'N/A')}")
    lines.append(f"Tissue: {metadata.get('tissue', 'N/A')}")
    model_names = _format_metadata_model_names(metadata.get("models", []))
    lines.append(f"Models: {', '.join(model_names)}")
    lines.append(f"Consensus Threshold: {metadata.get('consensus_threshold', 'N/A')}")
    lines.append(f"Max Discussion Rounds: {metadata.get('max_discussion_rounds', 'N/A')}")
    lines.append("")


def _append_initial_predictions_section(
    *,
    lines: list[str],
    model_annotations: dict[str, dict[str, Any]],
    cluster_id: str,
) -> None:
    """Append initial prediction section for one cluster."""
    lines.append("")
    lines.append("-" * 40)
    lines.append("INITIAL PREDICTIONS")
    lines.append("-" * 40)

    for model_name, predictions in model_annotations.items():
        if cluster_id not in predictions:
            continue
        lines.append(f"\n[{model_name}]")
        lines.append(f"  {predictions[cluster_id]}")


def _append_discussion_section(
    *,
    lines: list[str],
    discussion_logs: dict[str, list[dict[str, Any]]],
    cluster_id: str,
) -> None:
    """Append discussion rounds section for one cluster."""
    rounds = discussion_logs.get(cluster_id)
    if not rounds:
        lines.append("")
        lines.append("-" * 40)
        lines.append("NO DISCUSSION NEEDED")
        lines.append("-" * 40)
        lines.append("  Consensus reached with initial predictions.")
        return

    for round_idx, round_responses in enumerate(rounds, start=1):
        lines.append("")
        lines.append("-" * 40)
        lines.append(f"ROUND {round_idx} DISCUSSION")
        lines.append("-" * 40)

        for model_name, response in round_responses.items():
            lines.append(f"\n[{model_name}]")
            response_text = response if isinstance(response, str) else str(response)
            response_text = response_text.strip() or "Unknown"
            for line in response_text.split("\n"):
                lines.append(f"  {line}")


def _append_final_result_section(
    *,
    lines: list[str],
    cluster_id: str,
    consensus: dict[str, Any],
    consensus_proportion: dict[str, Any],
    entropy: dict[str, Any],
    controversial_clusters: list[str],
) -> None:
    """Append final result section for one cluster."""
    lines.append("")
    lines.append("-" * 40)
    lines.append("FINAL RESULT")
    lines.append("-" * 40)
    lines.append(f"  Final Annotation: {consensus.get(cluster_id, 'N/A')}")
    lines.append(f"  Consensus Proportion: {consensus_proportion.get(cluster_id, 'N/A')}")
    lines.append(f"  Entropy: {entropy.get(cluster_id, 'N/A')}")
    lines.append(
        f"  Was Controversial: {'Yes' if cluster_id in controversial_clusters else 'No'}"
    )
    lines.append("")


def format_discussion_report(
    results: dict[str, Any],
    cluster_id: str | None = None,
    output_file: str | None = None,
) -> str:
    """Format discussion results into a clean, readable report.

    This function generates a structured report showing:
    1. Initial predictions from each model
    2. Full discussion for each round
    3. Final consensus result

    Args:
        results: The results dictionary from interactive_consensus_annotation
        cluster_id: Optional cluster ID to filter. If None, reports all clusters.
        output_file: Optional file path to save the report. If None, only returns string.

    Returns:
        str: Formatted discussion report

    Example:
        >>> results = interactive_consensus_annotation(...)
        >>> report = format_discussion_report(results, cluster_id="0")
        >>> print(report)

        # Or save to file:
        >>> format_discussion_report(results, output_file="discussion_report.txt")
    """
    lines: list[str] = []
    sep = "=" * 80

    model_annotations = results.get("model_annotations", {})
    discussion_logs = results.get("discussion_logs", {})
    consensus = results.get("consensus", {})
    consensus_proportion = results.get("consensus_proportion", {})
    entropy = results.get("entropy", {})
    controversial_clusters = results.get("controversial_clusters", [])
    metadata = results.get("metadata", {})

    _append_report_header(lines=lines, separator=sep, metadata=metadata)

    clusters_to_report = _resolve_clusters_to_report(consensus=consensus, cluster_id=cluster_id)

    for cid in clusters_to_report:
        lines.append(sep)
        lines.append(f"CLUSTER {cid}")
        lines.append(sep)
        _append_initial_predictions_section(
            lines=lines,
            model_annotations=model_annotations,
            cluster_id=cid,
        )
        _append_discussion_section(
            lines=lines,
            discussion_logs=discussion_logs,
            cluster_id=cid,
        )
        _append_final_result_section(
            lines=lines,
            cluster_id=cid,
            consensus=consensus,
            consensus_proportion=consensus_proportion,
            entropy=entropy,
            controversial_clusters=controversial_clusters,
        )

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    report = "\n".join(lines)

    # Save to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        write_log(f"Discussion report saved to: {output_file}")

    return report
