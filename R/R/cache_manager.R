is_valid_consensus_cache_data <- function(data) {
  if (!is.list(data) ||
      !all(c("annotation", "discussion_log") %in% names(data))) {
    return(FALSE)
  }
  annotation_is_valid <- is.character(data$annotation) &&
    length(data$annotation) == 1 &&
    !is.na(data$annotation) &&
    nzchar(trimws(data$annotation))
  discussion_is_valid <- is.list(data$discussion_log) &&
    "rounds" %in% names(data$discussion_log) &&
    is.list(data$discussion_log$rounds)
  annotation_is_valid && discussion_is_valid
}

#' Cache Manager Class
#' @description Manages caching of consensus analysis results
#' @export
CacheManager <- R6::R6Class(
  "CacheManager",
  
  public = list(
    #' @field cache_dir Directory to store cache files. Options:
    #'   - NULL (default): Uses system cache directory
    #'   - "local": Uses .mllmcelltype_cache in current directory
    #'   - "temp": Uses temporary directory
    #'   - Custom path: Any other string is used as directory path
    cache_dir = NULL,
    
    #' @field cache_version Current cache version
    cache_version = "1.2",
    
    #' @description Initialize cache manager
    #
    #'   - NULL (default): Uses system cache directory via \code{tools::R_user_dir()}
    #'   - "local": Uses .mllmcelltype_cache in current directory  
    #'   - "temp": Uses temporary directory (cleared on R restart)
    #'   - Custom path: Any other string is used as directory path
    #' @param cache_dir Cache directory selector or custom path
    initialize = function(cache_dir = NULL) {
      if (is.null(cache_dir)) {
        # Default: use system cache directory
        cache_dir <- file.path(
          tools::R_user_dir("mLLMCelltype", which = "cache"),
          "consensus_cache"
        )
      } else {
        if (!is.character(cache_dir) || length(cache_dir) != 1 ||
            is.na(cache_dir) || !nzchar(trimws(cache_dir))) {
          stop("cache_dir must be NULL or a non-empty character scalar")
        }
      }

      if (identical(cache_dir, "local")) {
        # Special value: use project local cache
        cache_dir <- file.path(".", ".mllmcelltype_cache")
      } else if (identical(cache_dir, "temp")) {
        # Special value: use temporary directory
        cache_dir <- file.path(tempdir(), "mllmcelltype_cache")
      }
      # Other values: directly use user specified path
      
      self$cache_dir <- cache_dir
      
      if (!dir.exists(cache_dir)) {
        created <- dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
        if (!created && !dir.exists(cache_dir)) {
          stop("Failed to create cache directory: ", cache_dir)
        }
      }
    },
    
    #' @description Get actual cache directory path
    #
    get_cache_dir = function() {
      return(self$cache_dir)
    },
    
    #' @description Generate cache key from input parameters (improved version)
    #' @param input Marker gene input
    #' @param models Model identifiers contributing to the result
    #' @param cluster_id Cluster identifier
    #' @param tissue_name Tissue context included in the cache key
    #' @param top_gene_count Number of marker genes used
    #' @param discussion_context Optional normalized inputs that affect a discussion
    generate_key = function(input, models, cluster_id, tissue_name = "", top_gene_count = 10,
                            discussion_context = NULL) {
      models <- .normalize_model_vector(models)
      cluster_id <- .normalize_required_string(as.character(cluster_id), "cluster_id")
      top_gene_count <- .normalize_top_gene_count(top_gene_count)
      if (!is.character(tissue_name) || length(tissue_name) != 1 || is.na(tissue_name)) {
        stop("tissue_name must be a character scalar")
      }
      tissue_name <- trimws(tissue_name)
      if (!is.null(discussion_context) && !is.list(discussion_context)) {
        stop("discussion_context must be NULL or a list")
      }

      # Extract genes using a standardized approach
      genes <- private$extract_genes_standardized(
        input,
        cluster_id,
        top_gene_count
      )

      # Create standardized components with input context for empty genes
      genes_hash <- private$create_genes_hash(genes)
      models_hash <- private$create_models_hash(models)
      cluster_hash <- private$create_cluster_hash(cluster_id)
      context_hash <- private$create_context_hash(
        tissue_name,
        top_gene_count,
        discussion_context
      )

      # Combine into final key with version prefix
      key <- paste("v", self$cache_version, genes_hash, models_hash, cluster_hash, context_hash, sep = "_")

      return(key)
    },
    
    #' @description Save results to cache
    #' @param key Valid cache key
    #' @param data Result object to persist
    save_to_cache = function(key, data) {
      cache_file <- private$cache_file_path(key)

      # Ensure cache directory exists
      if (!dir.exists(self$cache_dir)) {
        created <- dir.create(self$cache_dir, recursive = TRUE, showWarnings = FALSE)
        if (!created && !dir.exists(self$cache_dir)) {
          warning("Failed to create cache directory: ", self$cache_dir, call. = FALSE)
          return(invisible(FALSE))
        }
      }

      saved <- tryCatch({
        private$atomic_save_rds(data, cache_file)
        get_logger()$log_cache_operation("store", key, file.size(cache_file))
        TRUE
      }, error = function(e) {
        get_logger()$log_cache_operation("store_failed", key, NULL)
        warning(paste("Failed to save cache file:", e$message), call. = FALSE)
        FALSE
      })
      invisible(saved)
    },
    
    #' @description Load results from cache
    #' @param key Valid cache key
    load_from_cache = function(key) {
      cache_file <- private$cache_file_path(key)
      if (!file.exists(cache_file) || dir.exists(cache_file)) {
        get_logger()$log_cache_operation("miss", key)
        return(NULL)
      }

      tryCatch({
        data <- readRDS(cache_file)
        get_logger()$log_cache_operation("hit", key, file.size(cache_file))
        data
      }, error = function(e) {
        get_logger()$log_cache_operation("load_failed", key, file.size(cache_file))
        warning(paste("Failed to load cache file:", e$message), call. = FALSE)
        NULL
      })
    },
    
    #' @description Check if results exist in cache
    #' @param key Valid cache key
    has_cache = function(key) {
      cache_file <- private$cache_file_path(key)
      return(file.exists(cache_file) && !dir.exists(cache_file))
    },
    
    #' @description Get cache statistics
    #
    get_cache_stats = function() {
      if (!dir.exists(self$cache_dir)) {
        return(list(
          cache_exists = FALSE,
          cache_count = 0,
          cache_size_mb = 0
        ))
      }
      
      cache_files <- private$list_cache_files()
      cache_sizes <- file.size(cache_files)
      
      return(list(
        cache_exists = TRUE,
        cache_count = length(cache_files),
        cache_size_mb = sum(cache_sizes, na.rm = TRUE) / (1024 * 1024),
        cache_files = cache_files
      ))
    },
    
    #' @description Clear all cache
    #' @param confirm Whether deletion is explicitly confirmed
    clear_cache = function(confirm = FALSE) {
      confirm <- .normalize_flag(confirm, "confirm")
      if (!dir.exists(self$cache_dir)) {
        message("Cache directory does not exist.")
        return(invisible(NULL))
      }
      
      cache_files <- private$list_cache_files()
      
      if (length(cache_files) == 0) {
        message("No cache files to clear.")
        return(invisible(NULL))
      }
      
      if (!confirm) {
        message(sprintf("This will delete %d cache files. Set confirm=TRUE to proceed.", length(cache_files)))
        return(invisible(NULL))
      }
      
      unlink(cache_files)
      removed_count <- sum(!file.exists(cache_files))
      get_logger()$log_cache_operation("clear", "all_cache", removed_count)
      message(sprintf("Cleared %d cache files.", removed_count))
      if (removed_count < length(cache_files)) {
        warning(sprintf("Failed to remove %d cache files.", length(cache_files) - removed_count),
                call. = FALSE)
      }
      
      return(invisible(NULL))
    },
    
    #' @description Validate cache content
    #' @param key Valid cache key
    validate_cache = function(key) {
      if (!self$has_cache(key)) {
        return(FALSE)
      }
      
      tryCatch({
        cache_data <- self$load_from_cache(key)
        is_valid_consensus_cache_data(cache_data)
      }, error = function(e) {
        message("Error validating cache: ", e$message)
        return(FALSE)
      })
    }
  ),
  
  private = list(
    #' Return deterministic paths for regular RDS cache files
    list_cache_files = function() {
      cache_files <- sort(list.files(
        self$cache_dir,
        pattern = "\\.rds$",
        full.names = TRUE
      ))
      if (length(cache_files) == 0) {
        return(character())
      }
      file_info <- file.info(cache_files)
      cache_files[!is.na(file_info$isdir) & !file_info$isdir]
    },

    #' Resolve a validated cache key to a path inside the cache directory
    cache_file_path = function(key) {
      valid_key <- is.character(key) && length(key) == 1 && !is.na(key) &&
        nzchar(key) && nchar(key, type = "bytes") <= 240 &&
        grepl("^[A-Za-z0-9._-]+$", key) && !key %in% c(".", "..")
      if (!valid_key) {
        stop("cache key must contain 1-240 ASCII letters, digits, dots, underscores, or hyphens")
      }
      file.path(self$cache_dir, paste0(key, ".rds"))
    },

    #' Write one R object to a path; separated for deterministic failure testing
    write_cache_data = function(data, path) {
      saveRDS(data, path)
    },

    #' Persist cache data through a same-directory temporary file
    atomic_save_rds = function(data, cache_file) {
      temp_file <- tempfile(
        pattern = paste0(".", basename(cache_file), "."),
        tmpdir = dirname(cache_file),
        fileext = ".tmp"
      )
      backup_file <- NULL
      committed <- FALSE

      on.exit({
        if (file.exists(temp_file)) {
          unlink(temp_file)
        }
        if (!is.null(backup_file) && file.exists(backup_file)) {
          if (committed) {
            unlink(backup_file)
          } else {
            if (file.exists(cache_file)) {
              unlink(cache_file)
            }
            if (!file.rename(backup_file, cache_file)) {
              warning("Failed to restore previous cache file after write failure", call. = FALSE)
            }
          }
        }
      }, add = TRUE)

      private$write_cache_data(data, temp_file)

      if (file.exists(cache_file)) {
        backup_file <- tempfile(
          pattern = paste0(".", basename(cache_file), "."),
          tmpdir = dirname(cache_file),
          fileext = ".bak"
        )
        if (!file.rename(cache_file, backup_file)) {
          stop("Failed to prepare existing cache file for replacement")
        }
      }

      if (!file.rename(temp_file, cache_file)) {
        stop("Failed to atomically replace cache file")
      }
      committed <- TRUE
      invisible(TRUE)
    },

    #' Extract genes from input in a standardized way
    #
    #
    #
    extract_genes_standardized = function(input, cluster_id, top_gene_count) {
      select_cluster_marker_genes(input, cluster_id, top_gene_count)
    },
    
    #' Create stable hash from genes list
    #
    #
    #
    #
    create_genes_hash = function(genes) {
      # Marker rank is part of the prompt, so preserve the selected gene order.
      digest::digest(genes, algo = "xxhash64")
    },
    
    #' Create stable hash from models list
    #
    #
    create_models_hash = function(models) {
      if (length(models) == 0) {
        return("no_models")
      }
      
      # Model order controls response ordering in later discussion prompts.
      digest::digest(as.character(models), algo = "xxhash64")
    },
    
    #' Create stable hash from tissue_name and top_gene_count
    create_context_hash = function(tissue_name, top_gene_count, discussion_context) {
      context <- list(
        tissue_name = tissue_name,
        top_gene_count = top_gene_count,
        discussion = discussion_context
      )
      digest::digest(context, algo = "xxhash64")
    },

    #' Create stable hash from cluster ID
    #
    #
    create_cluster_hash = function(cluster_id) {
      # Always convert to character for consistency
      cluster_str <- as.character(cluster_id)
      
      # Keep only short path-safe IDs readable; hash every other value.
      if (length(cluster_str) == 1 && !is.na(cluster_str) &&
          grepl("^[A-Za-z0-9_-]{1,8}$", cluster_str)) {
        return(paste0("c", cluster_str))
      } else {
        return(paste0("c", substr(digest::digest(cluster_str, algo = "xxhash64"), 1, 8)))
      }
    }
  )
)
