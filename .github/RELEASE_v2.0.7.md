# mLLMCelltype 2.0.7

## Summary

- Added built-in Kimi provider supporting the Moonshot AI Open Platform and Kimi Code endpoints (`/chat/completions` and `/messages`).
- Added `return_reasoning` parameter to `annotate_cell_types()` (R) and `annotate_clusters()` (Python) for structured per-cluster output (`cell_type`, `marker_genes`, `gene_expression`).
- Fixed intermittent `Unknown` results when `return_reasoning` is enabled by preserving the raw response text for JSON parsing.
- Reverted Kimi default temperature to `0.6` (API rejects `0.7` for these models).

## Verification

- Python: `python -m pytest` — 548 passed, 1 skipped
- R: `devtools::test()` — 637 passed, 4 skipped
