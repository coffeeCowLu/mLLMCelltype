# Real API Integration Tests

# Helper function to skip on CRAN
skip_on_cran <- function() {
  if (!identical(Sys.getenv("NOT_CRAN"), "true")) {
    testthat::skip("Skipping on CRAN")
  }
}

test_that("Real API integration with system cache", {
  skip_on_cran()
  
  # Check for OpenRouter API key
  has_openrouter <- nchar(Sys.getenv("OPENROUTER_API_KEY")) > 0
  
  if (!has_openrouter) {
    skip("No OpenRouter API key available")
  }
  
  # Create minimal test data - clear T cell signature
  test_data <- data.frame(
    CD3D = c(3.2, 0.1, 2.8),    # High T cell marker
    CD8A = c(2.1, 0.0, 1.9),    # CD8+ marker
    IL7R = c(1.2, 0.3, 1.1),    # Memory T cell
    CD14 = c(0.1, 2.8, 0.2),    # Monocyte marker (low in T cells)
    row.names = paste0("Cell_", 1:3)
  )
  
  clusters <- c(0, 1, 0)  # Two T cells, one other
  
  # Need to format data properly for the function
  # Convert to expected format with cluster and gene columns
  formatted_data <- data.frame(
    cluster = rep(clusters, each = ncol(test_data)),
    gene = rep(rownames(test_data), length(clusters)),
    avg_log2FC = as.vector(t(test_data))
  )
  
  # Test with system cache (NULL) - use a fast model
  result1 <- interactive_consensus_annotation(
    input = formatted_data,
    tissue_name = "human PBMC",
    cache_dir = NULL,  # Use system cache
    models = "openai/gpt-4o-mini",  # Fast model for testing
    api_keys = list(openrouter = Sys.getenv("OPENROUTER_API_KEY"))
  )
  
  expect_true(!is.null(result1))
  expect_true("final_annotations" %in% names(result1))
  
  # Test cache was created in system directory
  cache_dir <- mllmcelltype_cache_dir()
  expect_true(dir.exists(cache_dir))
  
  # Test with local cache
  result2 <- interactive_consensus_annotation(
    input = formatted_data,
    tissue_name = "human PBMC",
    cache_dir = "local",
    models = "openai/gpt-4o-mini",
    api_keys = list(openrouter = Sys.getenv("OPENROUTER_API_KEY"))
  )
  
  expect_true(!is.null(result2))
  expect_true("final_annotations" %in% names(result2))
  expect_true(dir.exists(file.path(".", ".mllmcelltype_cache")))
})

test_that("Cache reuse works with real API", {
  skip_on_cran()
  
  has_api_key <- nchar(Sys.getenv("OPENROUTER_API_KEY")) > 0
  if (!has_api_key) skip("No OpenRouter API key available")
  
  # Simple test data
  test_data <- data.frame(
    cluster = c(0, 1),
    gene = c("CD3D", "CD14"),
    avg_log2FC = c(2.1, 1.8)
  )
  
  temp_cache <- file.path(tempdir(), "cache_reuse_test")
  
  # First call - should hit API
  start_time1 <- Sys.time()
  result1 <- interactive_consensus_annotation(
    input = test_data,
    tissue_name = "human PBMC",
    cache_dir = temp_cache,
    models = "openai/gpt-4o-mini",
    api_keys = list(openrouter = Sys.getenv("OPENROUTER_API_KEY"))
  )
  time1 <- as.numeric(Sys.time() - start_time1)
  
  # Second call - should use cache
  start_time2 <- Sys.time()
  result2 <- interactive_consensus_annotation(
    input = test_data,
    tissue_name = "human PBMC",
    cache_dir = temp_cache,
    models = "openai/gpt-4o-mini",
    api_keys = list(openrouter = Sys.getenv("OPENROUTER_API_KEY"))
  )
  time2 <- as.numeric(Sys.time() - start_time2)
  
  # Cache call should be much faster
  expect_true(time2 < time1 * 0.5)  # At least 2x faster
  expect_equal(result1$final_annotations, result2$final_annotations)
})

test_that("Different cache directories work independently", {
  skip_on_cran()
  
  has_api_key <- nchar(Sys.getenv("OPENROUTER_API_KEY")) > 0
  if (!has_api_key) skip("No OpenRouter API key available")
  
  # Simple test data  
  test_data <- data.frame(
    cluster = c(0),
    gene = c("CD3D"),
    avg_log2FC = c(2.0)
  )
  
  # Test different cache strategies
  cache_dirs <- list(
    temp1 = file.path(tempdir(), "cache_test_1"),
    temp2 = file.path(tempdir(), "cache_test_2")
  )
  
  results <- list()
  
  # Run same query with different cache directories
  for (i in seq_along(cache_dirs)) {
    cache_name <- names(cache_dirs)[i]
    cache_dir <- cache_dirs[[i]]
    
    result <- interactive_consensus_annotation(
      input = test_data,
      tissue_name = "human PBMC",
      cache_dir = cache_dir,
      models = "openai/gpt-4o-mini",
      api_keys = list(openrouter = Sys.getenv("OPENROUTER_API_KEY"))
    )
    
    results[[cache_name]] <- result
    
    # Verify cache directory was created
    expect_true(dir.exists(cache_dir))
    
    # Verify cache files exist
    cache_files <- list.files(cache_dir, recursive = TRUE)
    expect_true(length(cache_files) > 0)
  }
  
  # Results should be consistent
  expect_equal(results$temp1$final_annotations, results$temp2$final_annotations)
})