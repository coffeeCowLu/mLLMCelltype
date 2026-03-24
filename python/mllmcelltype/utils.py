"""Utility functions for LLMCellType."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .config import get_api_key_env_var
from .logger import write_log

UNKNOWN_ANNOTATION_TOKENS = {
    "unknown",
    "unk",
    "n/a",
    "na",
    "none",
    "null",
    "nan",
    "-",
    "--",
    "inconclusive",
}
UNKNOWN_WITH_CONTEXT_PATTERN = re.compile(r"(?i)^unknown(?:\s*[\(\[\{].*[\)\]\}])?$")
ERROR_ANNOTATION_PATTERN = re.compile(r"(?i)^error(?:\s*[:\-\(].*)?$")


def _unwrap_balanced_wrappers(text: str) -> str:
    """Strip one or more balanced wrapper pairs around text."""
    wrapper_pairs = {"[": "]", "(": ")", "{": "}", '"': '"', "'": "'"}
    normalized = text.strip()
    while (
        len(normalized) >= 2
        and normalized[0] in wrapper_pairs
        and normalized[-1] == wrapper_pairs[normalized[0]]
    ):
        normalized = normalized[1:-1].strip()
    return normalized


def is_unknown_annotation(value: Any) -> bool:
    """Whether a value should be treated as missing/unknown annotation."""
    if value is None:
        return True

    normalized = _unwrap_balanced_wrappers(str(value))
    if not normalized:
        return True

    lowered = normalized.casefold()
    if lowered in UNKNOWN_ANNOTATION_TOKENS:
        return True
    if UNKNOWN_WITH_CONTEXT_PATTERN.match(normalized):
        return True
    return bool(ERROR_ANNOTATION_PATTERN.match(normalized))


def normalize_annotation(value: Any) -> str:
    """Normalize raw annotation into clean text with deterministic Unknown sentinel."""
    if is_unknown_annotation(value):
        return "Unknown"
    return str(value).strip()


def cluster_sort_key(cluster_id: str) -> tuple[int, int, str]:
    """Sort key for deterministic, natural ordering of cluster IDs.

    Numeric IDs sort first by value (0, 1, 2, …, 10, 11);
    non-numeric IDs follow in lexicographic order.
    """
    cluster_str = str(cluster_id)
    try:
        return (0, int(cluster_str), cluster_str)
    except ValueError:
        return (1, 0, cluster_str)


def _get_cache_dir(cache_dir: str | None = None) -> str:
    """Get cache directory path with consistent handling.

    Args:
        cache_dir: Custom cache directory or None for default

    Returns:
        str: Cache directory path
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".mllmcelltype", "cache")
    return cache_dir


def load_api_key(provider: str) -> str | None:
    """Load API key for a specific provider from environment variables or .env file.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')

    Returns:
        str | None: The API key, or None if not found

    """
    # Get environment variable names from centralized config.
    # Gemini keeps a compatibility alias for older docs/user setups.
    provider_normalized = str(provider).strip().lower()
    env_vars = [get_api_key_env_var(provider)]
    if provider_normalized == "gemini" and "GOOGLE_API_KEY" not in env_vars:
        env_vars.append("GOOGLE_API_KEY")

    # Get API key from environment variables in order of priority
    api_key = None
    selected_env_var = env_vars[0]
    for env_var in env_vars:
        candidate = os.getenv(env_var)
        if isinstance(candidate, str):
            candidate = candidate.strip() or None
        if candidate:
            api_key = candidate
            selected_env_var = env_var
            break

    # If not found in environment, try to load from .env file
    if not api_key:
        try:
            # Try to find .env file in project root (current directory or parent directories)
            env_path = None
            current_dir = os.path.abspath(os.getcwd())

            # Check current directory and up to 3 parent directories
            for _ in range(4):
                potential_path = os.path.join(current_dir, ".env")
                if os.path.isfile(potential_path):
                    env_path = potential_path
                    break
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Reached root directory
                    break
                current_dir = parent_dir

            # Fall back to a .env in the package directory only if no
            # project-level .env was found (project takes priority).
            if env_path is None:
                package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                package_env_path = os.path.join(package_dir, ".env")
                if os.path.isfile(package_env_path):
                    env_path = package_env_path

            if env_path:
                write_log(f"Found .env file at {env_path}")
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        if key.strip() in env_vars:
                            # Strip surrounding quotes (common .env convention)
                            api_key = value.strip().strip("\"'").strip() or None
                            selected_env_var = key.strip()
                            write_log(f"Loaded API key for {provider} from .env file")
                            break
        except OSError as e:
            write_log(f"Error loading .env file: {e!s}", level="warning")

    if not api_key:
        write_log(f"API key not found for provider: {env_vars[0]}", level="debug")
    else:
        write_log(f"Using API key from environment variable: {selected_env_var}", level="debug")

    return api_key


def _coerce_marker_gene_list(genes: Any) -> list[str]:
    """Coerce marker genes into a cleaned list of strings."""
    if genes is None:
        return []

    if isinstance(genes, Mapping):
        raise ValueError("marker_genes values must be gene lists, not mapping objects")

    if isinstance(genes, str):
        values = [genes]
    elif isinstance(genes, Iterable):
        values = genes
    else:
        values = [genes]

    return [str(g).strip() for g in values if g is not None and g == g and str(g).strip()]


def _merge_marker_genes(
    target: dict[str, list[str]],
    normalized_cluster: str,
    genes: list[str],
    raw_cluster: Any,
) -> None:
    """Merge marker genes for the same normalized cluster key."""
    if normalized_cluster not in target:
        target[normalized_cluster] = genes
        return

    existing = target[normalized_cluster]
    existing_set = set(existing)
    appended = 0
    for gene in genes:
        if gene not in existing_set:
            existing.append(gene)
            existing_set.add(gene)
            appended += 1

    write_log(
        f"Cluster key collision after normalization: {raw_cluster!r} -> '{normalized_cluster}', "
        f"merged {appended} marker genes",
        level="warning",
    )


def normalize_marker_genes_keys(
    marker_genes: dict[Any, list[str]],
) -> dict[str, list[str]]:
    """Normalize marker gene dict keys to strings and merge key collisions."""
    if not isinstance(marker_genes, dict):
        raise ValueError(
            f"marker_genes must be a dict, got {type(marker_genes).__name__}"
        )

    normalized: dict[str, list[str]] = {}
    for raw_cluster, genes in marker_genes.items():
        cluster_id = str(raw_cluster)
        cleaned_genes = _coerce_marker_gene_list(genes)
        _merge_marker_genes(normalized, cluster_id, cleaned_genes, raw_cluster)

    return normalized


def create_cache_key(
    prompt: str, model: str, provider: str, base_url: str | None = None
) -> str:
    """Create a cache key for a specific request.

    Args:
        prompt: The prompt text
        model: The model name
        provider: The provider name
        base_url: Optional custom base URL. When provided, it becomes part of
                  the cache key so that different endpoints produce distinct
                  cache entries.

    Returns:
        str: The cache key

    """
    # Normalize inputs to ensure consistent keys
    normalized_provider = str(provider).lower().strip()
    normalized_model = str(model).lower().strip()
    normalized_prompt = str(prompt).strip()

    # For OpenRouter models (containing '/'), ensure provider is 'openrouter'
    # This prevents cache key collisions between different providers
    if "/" in normalized_model:
        normalized_provider = "openrouter"

    # Create a string to hash with clear separators to avoid collisions
    hash_string = (
        f"provider:{normalized_provider}||model:{normalized_model}||prompt:{normalized_prompt}"
    )

    # Include base_url when explicitly set (None = default endpoint, no change)
    if base_url:
        normalized_base_url = str(base_url).strip().rstrip("/")
        if normalized_base_url:
            hash_string += f"||base_url:{normalized_base_url}"

    # Create hash
    hash_object = hashlib.sha256(hash_string.encode("utf-8"))
    return hash_object.hexdigest()


def save_to_cache(
    cache_key: str,
    results: list[str] | dict[str, Any] | str,
    cache_dir: str | None = None,
) -> None:
    """Save results to cache.

    Args:
        cache_key: The cache key
        results: The results to cache (list, dictionary, or string)
        cache_dir: The cache directory. If None, uses default directory.

    """
    cache_dir = _get_cache_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Ensure results are in a consistent format
    cache_data = {"version": "1.0", "timestamp": time.time(), "data": results}

    # Save results to cache
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        write_log(f"Saved results to cache: {cache_file}")
    except (OSError, TypeError, ValueError) as e:
        write_log(f"Error saving cache for {cache_file}: {e}", level="error")


def load_from_cache(
    cache_key: str, cache_dir: str | None = None
) -> list[str] | dict[str, Any] | str | None:
    """Load results from cache.

    Args:
        cache_key: The cache key
        cache_dir: The cache directory. If None, uses default directory.

    Returns:
        list[str] | dict[str, Any] | str | None: Cached results, or None if not found

    """
    cache_dir = _get_cache_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # Check if cache file exists
    if not os.path.exists(cache_file):
        return None

    # Load results from cache
    try:
        with open(cache_file) as f:
            cache_content = json.load(f)

        # Handle different cache formats
        if isinstance(cache_content, dict) and "data" in cache_content:
            # New format with metadata
            results = cache_content["data"]
            write_log(
                f"Loaded results from cache (version {cache_content.get('version', 'unknown')}): {cache_file}"
            )
        else:
            # Old format (direct data)
            results = cache_content
            write_log(f"Loaded results from cache (legacy format): {cache_file}")

        return results
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        write_log(f"Error loading cache for {cache_file}: {e}", level="warning")
        return None


def parse_marker_genes(marker_genes_df: pd.DataFrame) -> dict[str, list[str]]:
    """Parse marker genes dataframe into a dictionary.

    Args:
        marker_genes_df: DataFrame containing marker genes

    Returns:
        dict[str, list[str]]: Dictionary mapping cluster names to lists of marker genes

    """
    if not isinstance(marker_genes_df, pd.DataFrame):
        raise ValueError(
            f"marker_genes_df must be a pandas DataFrame, got {type(marker_genes_df).__name__}"
        )

    result = {}

    # Check if dataframe is empty
    if marker_genes_df.empty:
        write_log("Empty marker genes dataframe", level="warning")
        return result

    # Resolve column names case-insensitively
    col_lower_map = {str(col).lower(): col for col in marker_genes_df.columns}

    cluster_col = col_lower_map.get("cluster")
    if cluster_col is None:
        write_log("'cluster' column not found in marker genes dataframe", level="error")
        raise ValueError(
            "'cluster' column not found in marker genes dataframe. "
            f"Available columns: {list(marker_genes_df.columns)}"
        )

    gene_col = col_lower_map.get("gene")
    if gene_col is None:
        write_log("'gene' column not found in marker genes dataframe", level="error")
        raise ValueError(
            "'gene' column not found in marker genes dataframe. "
            f"Available columns: {list(marker_genes_df.columns)}"
        )

    # Group by cluster and get list of genes (drop None/NaN values)
    for cluster, group in marker_genes_df.groupby(cluster_col, sort=False):
        cluster_id = str(cluster)
        genes = [
            str(g).strip()
            for g in group[gene_col]
            if g is not None and g == g and str(g).strip()
        ]
        if genes:
            _merge_marker_genes(result, cluster_id, genes, cluster)
        elif cluster_id not in result:
            # Preserve explicit cluster rows even when no marker genes are present.
            # Downstream annotation treats these as Unknown without issuing API calls.
            result[cluster_id] = []

    return result


def format_results(
    results: list[str] | dict[str, Any] | str,
    clusters: list[str],
) -> dict[str, str]:
    """Format results into a dictionary mapping cluster names to annotations.

    Args:
        results: Annotation results (list, dict, or string)
        clusters: List of cluster names

    Returns:
        dict[str, str]: Dictionary mapping cluster names to annotations

    """
    def _coerce_annotation(value: Any) -> str:
        return normalize_annotation(value)

    def _cluster_candidates(cluster: str) -> set[str]:
        candidates = {cluster}
        stripped = re.sub(r"^[Cc]luster[_\s]", "", cluster)
        if stripped != cluster:
            candidates.add(stripped)
        return {candidate.strip() for candidate in candidates if candidate.strip()}

    def _parse_labeled_line(line: str) -> tuple[str, str] | None:
        patterns = [
            r"(?i)^\s*(?:[-*]\s*)?(?:cluster[_\s]*)?([A-Za-z0-9_.-]+)\s*(?:[:\-]|\uFF1A)\s*(.+?)\s*$",
            r"(?i)^\s*(?:[-*]\s*)?(?:cluster[_\s]*)?([A-Za-z0-9_.-]+)\)\s*(.+?)\s*$",
            r"(?i)^\s*(?:[-*]\s*)?(?:cluster[_\s]*)?([A-Za-z0-9_-]+)\.\s*(.+?)\s*$",
        ]
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        return None

    def _build_cluster_aliases(cluster_id: str) -> set[str]:
        aliases = _cluster_candidates(cluster_id)
        expanded = set(aliases)
        for alias in aliases:
            stripped = re.sub(r"^[Cc]luster[_\s]", "", alias)
            if stripped != alias and stripped:
                expanded.add(stripped)
            if stripped:
                expanded.add(f"Cluster_{stripped}")
                expanded.add(f"cluster_{stripped}")
                expanded.add(f"Cluster {stripped}")
                expanded.add(f"cluster {stripped}")
        return {alias.strip() for alias in expanded if alias and alias.strip()}

    def _build_requested_cluster_lookup(target_clusters: list[str]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for cluster in target_clusters:
            cluster_str = str(cluster)
            for candidate in _build_cluster_aliases(cluster_str):
                lookup.setdefault(candidate, cluster_str)
                lookup.setdefault(candidate.lower(), cluster_str)
        return lookup

    def _resolve_target_cluster(raw_cluster_id: Any, lookup: dict[str, str]) -> str | None:
        if raw_cluster_id is None:
            return None
        raw_cluster_str = str(raw_cluster_id).strip()
        if not raw_cluster_str:
            return None

        target = lookup.get(raw_cluster_str) or lookup.get(raw_cluster_str.lower())
        if target:
            return target

        for alias in _build_cluster_aliases(raw_cluster_str):
            target = lookup.get(alias) or lookup.get(alias.lower())
            if target:
                return target
        return None

    def _store_annotation_if_better(
        target: dict[str, str],
        cluster_id: str,
        raw_annotation: Any,
    ) -> None:
        annotation = _coerce_annotation(raw_annotation)
        existing = target.get(cluster_id)
        if existing is None or (existing == "Unknown" and annotation != "Unknown"):
            target[cluster_id] = annotation

    # Build candidate->cluster mapping once so all parsing branches use identical matching rules.
    candidate_to_cluster = _build_requested_cluster_lookup(clusters)

    # Fast path: dictionary results (legacy/new cache compatibility).
    if isinstance(results, dict):
        normalized = {str(k).strip(): _coerce_annotation(v) for k, v in results.items()}
        resolved_annotations: dict[str, str] = {}
        for raw_cluster_id, annotation in normalized.items():
            target_cluster = _resolve_target_cluster(raw_cluster_id, candidate_to_cluster)
            if not target_cluster:
                continue
            _store_annotation_if_better(resolved_annotations, target_cluster, annotation)

        formatted = {}
        for cluster in clusters:
            cluster_str = str(cluster)
            annotation = resolved_annotations.get(cluster_str)
            formatted[cluster_str] = annotation if annotation is not None else "Unknown"
        write_log("Parsed response from dictionary format", level="info")
        return formatted

    # Normalize response into lines.
    if isinstance(results, str):
        lines = results.splitlines()
    elif isinstance(results, (list, tuple, set)):
        lines = [str(line) for line in results]
    elif results is None:
        lines = []
    else:
        # Defensive fallback for unexpected scalar payloads from provider wrappers/cache.
        lines = [str(results)]

    # Clean up results (remove empty lines and whitespace)
    clean_results = [line.strip() for line in lines if line and line.strip()]

    # Case 1: Try to parse the format "Cluster X: Annotation" (most common format from our prompts)
    result = {}
    cluster_pattern = r"(?i)Cluster\s+(.+?):\s*(.*)"

    # First pass: parse labeled lines in one pass (supports more real-world formats).
    for line in clean_results:
        parsed = _parse_labeled_line(line)
        if not parsed:
            continue
        raw_cluster_id, annotation = parsed
        target_cluster = _resolve_target_cluster(raw_cluster_id, candidate_to_cluster)
        if not target_cluster:
            continue
        _store_annotation_if_better(result, target_cluster, annotation)

    # If we found annotations for all clusters, return the result
    if len(result) == len(clusters):
        write_log(
            "Successfully parsed response in 'Cluster X: Annotation' format",
            level="info",
        )
        return result
    labeled_partial = result.copy()

    # Case 2: Try to parse JSON response
    try:
        # Join all lines and try to find JSON content
        full_text = "\n".join(clean_results)

        # Extract JSON content if it's wrapped in ```json and ``` markers
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", full_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code blocks, try to find JSON array/object directly
            json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", full_text)
            # Extract JSON content or use full text
            json_str = json_match.group(1) if json_match else full_text

        # Fix common JSON formatting issues
        # Add missing commas between JSON objects
        json_str = re.sub(r'("[^"]+")\s*\n\s*("[^"]+")', r"\1,\n\2", json_str)
        # Add missing commas after closing brackets
        json_str = re.sub(r'(\])\s*\n\s*("[^"]+")', r"\1,\n\2", json_str)
        # Add missing commas after closing braces
        json_str = re.sub(r'(\})\s*\n\s*("[^"]+")', r"\1,\n\2", json_str)
        # Add missing commas after closing braces before opening braces
        json_str = re.sub(r"(\})\s*\n\s*(\{)", r"\1,\n\2", json_str)

        # Parse JSON
        data = json.loads(json_str)

        json_result: dict[str, str] = {}

        # Case 2a: {"annotations": [{"cluster": "...", "cell_type": "..."}]}
        if isinstance(data, dict) and "annotations" in data and isinstance(data["annotations"], list):
            for annotation in data["annotations"]:
                if not isinstance(annotation, dict):
                    continue
                if "cluster" not in annotation:
                    continue
                target_cluster = _resolve_target_cluster(annotation.get("cluster"), candidate_to_cluster)
                if not target_cluster:
                    continue
                cell_type = (
                    annotation.get("cell_type")
                    or annotation.get("annotation")
                    or annotation.get("label")
                )
                _store_annotation_if_better(json_result, target_cluster, cell_type)

        # Case 2b: direct mapping {"1": "T cells", "2": "B cells"}
        elif isinstance(data, dict):
            for cluster_id, annotation in data.items():
                target_cluster = _resolve_target_cluster(cluster_id, candidate_to_cluster)
                if target_cluster:
                    _store_annotation_if_better(json_result, target_cluster, annotation)

        # Case 2c: list of objects [{"cluster": "...", "cell_type": "..."}]
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "cluster" not in item:
                    continue
                target_cluster = _resolve_target_cluster(item.get("cluster"), candidate_to_cluster)
                if not target_cluster:
                    continue
                cell_type = item.get("cell_type") or item.get("annotation") or item.get("label")
                _store_annotation_if_better(json_result, target_cluster, cell_type)

        if json_result:
            formatted = {str(cluster): json_result.get(str(cluster), "Unknown") for cluster in clusters}
            if len(json_result) == len(clusters):
                write_log("Successfully parsed JSON response", level="info")
            else:
                write_log(
                    f"Parsed partial JSON response ({len(json_result)}/{len(clusters)} clusters), "
                    "missing clusters marked Unknown",
                    level="warning",
                )
            return formatted
    except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:
        write_log(f"Failed to parse JSON response: {e!s}", level="debug")

    # Prefer partial labeled mapping over positional mapping because it is
    # cluster-id aware and avoids line-index misalignment.
    if labeled_partial:
        write_log(
            f"Parsed partial labeled response ({len(labeled_partial)}/{len(clusters)} clusters), "
            "missing clusters marked Unknown",
            level="warning",
        )
        return {str(cluster): labeled_partial.get(str(cluster), "Unknown") for cluster in clusters}

    # Case 3: Line-by-line mapping — each line corresponds to a cluster
    if len(clean_results) < len(clusters):
        write_log(
            f"Fewer result lines ({len(clean_results)}) than clusters ({len(clusters)}), "
            "remaining clusters will be marked Unknown",
            level="warning",
        )

    result = {}
    for i, cluster in enumerate(clusters):
        if i < len(clean_results):
            line = clean_results[i]
            match = re.match(cluster_pattern, line)
            if match:
                result[str(cluster)] = _coerce_annotation(match.group(2))
            else:
                result[str(cluster)] = _coerce_annotation(line)
        else:
            result[str(cluster)] = "Unknown"

    write_log("Parsed response as line-by-line format", level="info")
    return result


def clear_cache(cache_dir: str | None = None, older_than: int | None = None) -> int:
    """Clear cache.

    Args:
        cache_dir: Cache directory
        older_than: Only clear items older than this many seconds.
                   If None, clear all cache.

    Returns:
        int: Number of cache files removed

    """
    cache_dir = _get_cache_dir(cache_dir)

    if not os.path.exists(cache_dir):
        return 0

    # Get all cache files
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]

    if not older_than:
        # Remove all cache files
        count = 0
        for f in cache_files:
            try:
                os.remove(os.path.join(cache_dir, f))
                count += 1
            except OSError as e:
                write_log(f"Error removing cache file {f}: {e}", level="warning")
        return count
    # Remove only older files
    now = time.time()
    count = 0
    for f in cache_files:
        file_path = os.path.join(cache_dir, f)
        try:
            # Check file age using metadata first, then fall back to mtime.
            # Corrupted cache entries should not crash cleanup.
            try:
                with open(file_path) as file:
                    cache_data = json.load(file)
                if isinstance(cache_data, dict) and "timestamp" in cache_data:
                    file_age = now - cache_data.get("timestamp", 0)
                else:
                    file_age = now - os.path.getmtime(file_path)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                file_age = now - os.path.getmtime(file_path)
                write_log(
                    f"Invalid cache metadata in {f}; falling back to file mtime",
                    level="warning",
                )
            if file_age > older_than:
                os.remove(file_path)
                count += 1
        except OSError as e:
            write_log(f"Error processing cache file {f}: {e}", level="warning")

    return count


def get_cache_stats(cache_dir: str | None = None, detailed: bool = True) -> dict[str, Any]:
    """Get cache statistics.

    Args:
        cache_dir: The cache directory (None for default)
        detailed: If True, read file contents for detailed stats (slower).
                  If False, only return basic file counts and sizes (faster).

    Returns:
        dict[str, Any]: Cache statistics

    """
    cache_dir = _get_cache_dir(cache_dir)

    if not os.path.exists(cache_dir):
        result = {
            "exists": False,
            "path": cache_dir,
            "count": 0,
            "size": 0,
            "size_mb": 0.0,
        }
        if detailed:
            result.update(
                {
                    "status": "No cache directory",
                    "oldest": None,
                    "newest": None,
                }
            )
        return result

    # Get all cache files
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)

    # Basic stats (fast path)
    result = {
        "exists": True,
        "path": cache_dir,
        "count": len(cache_files),
        "size": total_size,
        "size_mb": total_size / (1024 * 1024),
    }

    if not detailed:
        return result

    # Detailed stats (slow path - reads file contents)
    if not cache_files:
        result.update(
            {
                "status": "Empty cache",
                "oldest": None,
                "newest": None,
                "format_counts": {"legacy": 0, "1.0": 0, "unknown": 0},
                "valid_files": 0,
                "invalid_files": 0,
            }
        )
        return result

    oldest = float("inf")
    newest = 0
    format_counts = {"legacy": 0, "1.0": 0, "unknown": 0}
    valid_files = 0
    invalid_files = 0

    for f in cache_files:
        file_path = os.path.join(cache_dir, f)
        try:
            # Load cache data
            with open(file_path) as file:
                cache_data = json.load(file)

            valid_files += 1

            # Check cache format
            if isinstance(cache_data, dict):
                if "version" in cache_data and "data" in cache_data:
                    # New format with metadata
                    version = cache_data.get("version", "unknown")
                    format_counts[version if version in format_counts else "unknown"] += 1

                    # Get timestamp
                    timestamp = cache_data.get("timestamp", 0)
                    oldest = min(oldest, timestamp)
                    newest = max(newest, timestamp)
                else:
                    # Legacy format or other dict format
                    format_counts["legacy"] += 1
            else:
                # Unknown format
                format_counts["unknown"] += 1

        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
        ) as e:
            invalid_files += 1
            write_log(f"Error processing cache file {f}: {e}", level="warning")

    # Convert timestamps to readable format
    oldest_str = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(oldest))
        if oldest != float("inf")
        else None
    )
    newest_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)) if newest != 0 else None

    result.update(
        {
            "status": "Cache available",
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "oldest": oldest_str,
            "newest": newest_str,
            "format_counts": format_counts,
        }
    )

    return result
