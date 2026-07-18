# mLLMCelltype v2.0.7 - Kimi Provider & Reasoning Mode

## 🎯 Release Highlights

### 🌟 **Key Features**
- **Built-in Kimi provider**: Added native support for the Moonshot AI Open Platform (`https://api.moonshot.cn/v1/chat/completions`) and the Kimi Code platform (`https://api.kimi.com/coding`). Models prefixed with `kimi-` or `moonshot-` are detected automatically (default `kimi-k2.6`).
- **`return_reasoning` parameter**: `annotate_cell_types()` (R) and `annotate_clusters()` (Python) now support `return_reasoning=TRUE/True`. When enabled, each cluster returns a structured record with `cell_type`, `marker_genes`, and `gene_expression` instead of a plain label, making it easier to inspect the evidence behind each annotation.

### 🐛 **Bug Fixes**
- Fixed intermittent `Unknown` results when `return_reasoning` is enabled. The provider pipeline now preserves the raw response text for JSON parsing, preventing line normalizers from stripping trailing commas inside JSON arrays/objects.

## 📊 **Consensus Framework Enhancements**

### 🧠 **LLM Provider Updates**
- Added `KimiProcessor` (R) and `process_kimi` (Python) with dual-protocol support:
  - OpenAI-compatible Chat Completions for the Moonshot Open Platform and Kimi Code `/chat/completions` endpoints.
  - Anthropic Messages protocol for Kimi Code `/messages` endpoints.
- Kimi k2 thinking mode is explicitly disabled for deterministic annotation output.
- Authentication supports `MOONSHOT_API_KEY` (and `KIMI_API_KEY` as an alias).

## 📚 **Documentation & Community**

### 📝 **Documentation**
- Added `return_reasoning` documentation to both `R/NEWS.md` and `python/CHANGELOG.md`.
- Updated release notes and changelogs with the Kimi provider details and reasoning-mode fixes.

## 🔧 **Technical Changes**

### 🏗️ **Internal Improvements**
- R: Added a `normalize` parameter to the model-request pipeline (`get_model_response`, `BaseAPIProcessor`) so reasoning mode receives the raw response string while plain-text mode keeps line normalization.
- Python: Added `normalize_response` to all built-in provider functions and the shared API helpers (`call_http_api_with_retry`, `call_openai_compatible_api`) for the same purpose.
- Added regression tests for markdown-fenced, multi-line JSON reasoning responses in both R and Python.

### ⚠️ **Note on Kimi Temperature**
- A brief trial to raise Kimi's default temperature from `0.6` to `0.7` was reverted because the Kimi API rejects `0.7` for the affected models (`"only 0.6 is allowed for this model"`). The default remains `0.6`.

## 🧪 **Testing & Validation**

### ✅ **New Tests**
- R: Added reasoning-mode parsing tests, including markdown-fenced multi-line JSON.
- Python: Added `test_annotate_clusters_return_reasoning_parses_multiline_json_with_commas` and updated provider-signature validation to include `normalize_response`.

## 📥 **Installation & Upgrade**

### **New Installation**
```r
# R from GitHub (latest)
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

```bash
# Python from GitHub (latest)
pip install "git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python"
```

### **Upgrade Instructions**
```r
# R
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)
packageVersion("mLLMCelltype")
```

```bash
# Python
pip install --force-reinstall --no-deps -e .
python -c "import mllmcelltype; print(mllmcelltype.__version__)"
```

## 🔗 **Links**

- **Documentation**: https://cafferychen777.github.io/mLLMCelltype/
- **GitHub**: https://github.com/cafferychen777/mLLMCelltype
- **Paper**: https://doi.org/10.1038/s42003-026-10420-8

---

**Full Changelog**: [v2.0.6...v2.0.7](https://github.com/cafferychen777/mLLMCelltype/compare/v2.0.6...v2.0.7)
