# Core Cache Logic Tests

test_that("Cache directory resolution works correctly", {
  # Test NULL (system cache)
  cache_manager <- CacheManager$new(cache_dir = NULL)
  actual_dir <- cache_manager$get_cache_dir()
  expected_pattern <- "mLLMCelltype.*consensus_cache"
  expect_true(grepl(expected_pattern, actual_dir))
  
  # Test "local" option
  cache_manager <- CacheManager$new(cache_dir = "local")
  expect_equal(cache_manager$get_cache_dir(), file.path(".", ".mllmcelltype_cache"))
  
  # Test "temp" option
  cache_manager <- CacheManager$new(cache_dir = "temp")
  expect_true(grepl("mllmcelltype_cache", cache_manager$get_cache_dir()))
  
  # Test custom path
  custom_path <- file.path(tempdir(), "custom_cache_test")
  cache_manager <- CacheManager$new(cache_dir = custom_path)
  expect_equal(cache_manager$get_cache_dir(), custom_path)
})

test_that("CacheManager rejects ambiguous cache directory inputs", {
  invalid_dirs <- list(character(0), c("one", "two"), NA_character_, "")

  for (cache_dir in invalid_dirs) {
    expect_error(
      CacheManager$new(cache_dir = cache_dir),
      "cache_dir must be NULL or a non-empty character scalar"
    )
  }
})

test_that("Cache directory creation follows CRAN policies", {
  # Save current working directory
  original_wd <- getwd()
  
  # Create a temporary directory to test in
  test_dir <- tempfile()
  dir.create(test_dir)
  setwd(test_dir)
  
  # Verify no directory created in current working directory by default
  initial_files <- list.files(".", all.files = TRUE)
  cache_manager <- CacheManager$new()
  after_files <- list.files(".", all.files = TRUE)
  
  expect_false("consensus_cache" %in% after_files)
  expect_equal(length(initial_files), length(after_files))
  
  # Restore working directory
  setwd(original_wd)
  unlink(test_dir, recursive = TRUE)
})

test_that("Utility functions work correctly", {
  # Test mllmcelltype_cache_dir()
  system_cache <- mllmcelltype_cache_dir()
  expect_true(dir.exists(system_cache))
  
  local_cache <- mllmcelltype_cache_dir("local")
  expect_equal(local_cache, file.path(".", ".mllmcelltype_cache"))
  
  # Test with temp option
  temp_cache <- mllmcelltype_cache_dir("temp")
  expect_true(grepl("mllmcelltype_cache", temp_cache))
})

test_that("Cache key generation works", {
  cache_manager <- CacheManager$new(cache_dir = tempdir())
  
  # Test with simple data
  test_data <- data.frame(
    cluster = c("cluster_0", "cluster_0", "cluster_1"),
    gene = c("CD3D", "CD8A", "IL7R"),
    avg_log2FC = c(2.1, 1.5, 0.8)
  )
  
  key1 <- cache_manager$generate_key(test_data, "gpt-5.5", "cluster_0")
  key2 <- cache_manager$generate_key(test_data, "gpt-5.5", "cluster_1")
  key3 <- cache_manager$generate_key(test_data, "claude-sonnet-4-6", "cluster_0")

  # Keys should be different for different inputs
  expect_true(nchar(key1) > 0)
  expect_false(key1 == key2)  # Different cluster
  expect_false(key1 == key3)  # Different model
})

test_that("Cache key varies with tissue_name and top_gene_count", {
  cache_manager <- CacheManager$new(cache_dir = tempdir())

  test_data <- data.frame(
    cluster = c(0, 0, 1, 1),
    gene = c("CD3D", "CD8A", "CD14", "LYZ"),
    avg_log2FC = c(2.5, 2.1, 1.8, 1.6)
  )

  base_key <- cache_manager$generate_key(test_data, "gpt-5.5", "0",
                                          tissue_name = "human PBMC", top_gene_count = 10)
  diff_tissue <- cache_manager$generate_key(test_data, "gpt-5.5", "0",
                                             tissue_name = "mouse brain", top_gene_count = 10)
  diff_topn <- cache_manager$generate_key(test_data, "gpt-5.5", "0",
                                           tissue_name = "human PBMC", top_gene_count = 5)
  same_key <- cache_manager$generate_key(test_data, "gpt-5.5", "0",
                                          tissue_name = "human PBMC", top_gene_count = 10)

  expect_false(base_key == diff_tissue)  # Different tissue
  expect_false(base_key == diff_topn)    # Different top_gene_count
  expect_equal(base_key, same_key)       # Same parameters = same key
})

test_that("Cache key preserves marker rank and model order", {
  cache_manager <- CacheManager$new(cache_dir = tempdir())
  ranked_markers <- data.frame(
    cluster = c(0, 0),
    gene = c("CD3D", "IL7R"),
    avg_log2FC = c(2, 1)
  )
  reversed_rank <- ranked_markers
  reversed_rank$avg_log2FC <- rev(reversed_rank$avg_log2FC)

  ranked_key <- cache_manager$generate_key(
    ranked_markers,
    c("gpt-5.5", "claude-sonnet-4-6"),
    "0"
  )
  reversed_genes_key <- cache_manager$generate_key(
    reversed_rank,
    c("gpt-5.5", "claude-sonnet-4-6"),
    "0"
  )
  reversed_models_key <- cache_manager$generate_key(
    ranked_markers,
    c("claude-sonnet-4-6", "gpt-5.5"),
    "0"
  )

  expect_false(ranked_key == reversed_genes_key)
  expect_false(ranked_key == reversed_models_key)
  expect_match(ranked_key, "^v_1\\.2_")
})

test_that("Cache key includes discussion inputs and rejects malformed markers", {
  cache_manager <- CacheManager$new(cache_dir = tempdir())
  markers <- list("0" = c("CD3D", "IL7R"))
  context <- list(
    initial_predictions = list(model_a = "T cell", model_b = "NK cell"),
    max_discussion_rounds = 3L,
    controversy_threshold = 0.7,
    entropy_threshold = 1
  )

  base_key <- cache_manager$generate_key(
    markers,
    c("gpt-5.5", "claude-sonnet-4-6"),
    "0",
    discussion_context = context
  )
  changed_context <- context
  changed_context$max_discussion_rounds <- 4L
  changed_key <- cache_manager$generate_key(
    markers,
    c("gpt-5.5", "claude-sonnet-4-6"),
    "0",
    discussion_context = changed_context
  )

  expect_false(base_key == changed_key)
  expect_error(
    cache_manager$generate_key(data.frame(CD3D = 1), "gpt-5.5", "0"),
    "must contain 'cluster' and 'gene'"
  )
})

test_that("Cache keys cannot escape the configured cache directory", {
  cache_dir <- tempfile("safe_cache_")
  cache_manager <- CacheManager$new(cache_dir = cache_dir)
  unsafe_keys <- c("../outside", "nested/key", "nested\\key", "", paste(rep("a", 241), collapse = ""))

  for (key in unsafe_keys) {
    expect_error(cache_manager$save_to_cache(key, list(value = 1)), "cache key must contain")
    expect_error(cache_manager$load_from_cache(key), "cache key must contain")
    expect_error(cache_manager$has_cache(key), "cache key must contain")
  }
})

test_that("Generated keys remain path-safe for arbitrary cluster IDs", {
  cache_dir <- tempfile("generated_key_cache_")
  cache_manager <- CacheManager$new(cache_dir = cache_dir)
  key <- cache_manager$generate_key(
    list("../x" = list(genes = c("CD3D"))),
    "gpt-5.5",
    "../x"
  )

  expect_false(grepl("/|\\\\", key))
  expect_true(cache_manager$save_to_cache(key, list(value = "safe")))
  expect_equal(cache_manager$load_from_cache(key)$value, "safe")
})

test_that("Failed cache writes preserve the previous value", {
  cache_dir <- tempfile("atomic_cache_")
  cache_manager <- CacheManager$new(cache_dir = cache_dir)
  key <- "atomic_key"
  expect_true(cache_manager$save_to_cache(key, list(value = "original")))

  FailingCacheManager <- R6::R6Class(
    "FailingCacheManager",
    inherit = CacheManager,
    private = list(
      write_cache_data = function(data, path) {
        stop("simulated serialization failure")
      }
    )
  )
  failing_manager <- FailingCacheManager$new(cache_dir = cache_dir)

  expect_warning(
    expect_false(failing_manager$save_to_cache(key, list(value = "replacement"))),
    "simulated serialization failure"
  )
  expect_equal(cache_manager$load_from_cache(key)$value, "original")
  expect_length(list.files(cache_dir, pattern = "\\.(tmp|bak)$"), 0)
})

test_that("Corrupted cache files are treated as misses", {
  cache_dir <- tempfile("corrupt_cache_")
  cache_manager <- CacheManager$new(cache_dir = cache_dir)
  key <- "corrupt_key"
  writeLines("not an RDS file", file.path(cache_dir, paste0(key, ".rds")))

  expect_warning(
    expect_null(cache_manager$load_from_cache(key)),
    "Failed to load cache file"
  )
})

test_that("Cache clearing requires an explicit logical confirmation", {
  cache_dir <- tempfile("clear_cache_")
  cache_manager <- CacheManager$new(cache_dir = cache_dir)
  expect_true(cache_manager$save_to_cache("keep_me", list(value = "safe")))

  for (confirm in list(1, 0, NA, NULL, "TRUE", logical(0), c(TRUE, FALSE))) {
    expect_error(cache_manager$clear_cache(confirm), "confirm must be TRUE or FALSE")
    expect_true(cache_manager$has_cache("keep_me"))
  }

  expect_message(cache_manager$clear_cache(FALSE), "Set confirm=TRUE")
  expect_true(cache_manager$has_cache("keep_me"))
  expect_message(cache_manager$clear_cache(TRUE), "Cleared 1 cache files")
  expect_false(cache_manager$has_cache("keep_me"))
})

test_that("Cache operations ignore directories with an RDS suffix", {
  cache_dir <- tempfile("regular_cache_files_")
  cache_manager <- CacheManager$new(cache_dir = cache_dir)
  fake_cache_dir <- file.path(cache_dir, "fake.rds")
  dir.create(fake_cache_dir)

  expect_false(cache_manager$has_cache("fake"))
  expect_null(cache_manager$load_from_cache("fake"))
  expect_equal(cache_manager$get_cache_stats()$cache_count, 0)
  expect_message(cache_manager$clear_cache(TRUE), "No cache files to clear")
  expect_true(dir.exists(fake_cache_dir))
})
