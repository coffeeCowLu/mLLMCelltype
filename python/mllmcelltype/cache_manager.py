"""Internal cache maintenance tool.

This module is intentionally non-interactive so it can be safely used in scripts/CI.
It is not part of the core annotation user flow.
"""

from __future__ import annotations

import argparse
import sys

from .utils import clear_cache, get_cache_stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mllmcelltype.cache_manager",
        description="Internal cache maintenance utility for mLLMCelltype",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached responses",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show cache directory, file count, and size",
    )
    return parser


def _print_cache_info() -> None:
    info = get_cache_stats(detailed=False)
    print(f"Cache directory: {info['path']}")
    print(f"Number of cache files: {info['count']}")
    print(f"Total cache size: {info['size_mb']:.2f} MB")


def main(argv: list[str] | None = None) -> int:
    """Run cache maintenance tool in non-interactive mode."""
    args = _build_parser().parse_args(argv)

    print("mLLMCelltype Internal Cache Tool")
    print("-" * 34)

    if args.clear:
        removed = clear_cache()
        print(f"Cleared {removed} cache files.")
        return 0

    # `--info` and no-arg mode both print status. No interactive prompts.
    _print_cache_info()
    if not args.info:
        print("\nTip: use --clear to remove all cache files.")
    return 0


def clear_cache_cli() -> int:
    """Backward-compatible wrapper for older imports."""
    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
