"""mLLMCelltype: A Python module for cell type annotation using various LLMs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# --- Public API (user-facing) ---
from .annotate import annotate_clusters, get_model_response
from .config import get_supported_providers
from .consensus import (
    format_discussion_report,
    interactive_consensus_annotation,
)
from .functions import get_provider
from .logger import get_current_log_file, setup_logging
from .utils import clear_cache, get_cache_stats, parse_marker_genes

try:
    __version__ = version("mllmcelltype")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "annotate_clusters",
    "clear_cache",
    "format_discussion_report",
    "get_cache_stats",
    "get_current_log_file",
    "get_model_response",
    "get_provider",
    "get_supported_providers",
    "interactive_consensus_annotation",
    "parse_marker_genes",
    "setup_logging",
]
