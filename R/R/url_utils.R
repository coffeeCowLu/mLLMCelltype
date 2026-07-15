#' URL Utilities for Base URL Resolution
#'
#' This file contains utility functions for resolving custom base URLs
#' for different API providers.

#' Validate base URL
#'
#' @param url URL string to validate
#' @return TRUE if the URL is a safe HTTP(S) endpoint, otherwise FALSE
#' @keywords internal
#' @noRd
validate_base_url <- function(url) {
  if (!is.character(url) || length(url) != 1 || is.na(url)) {
    return(FALSE)
  }

  normalized <- trimws(url)
  if (!nzchar(normalized) || grepl("\\s", normalized)) {
    return(FALSE)
  }

  parsed <- tryCatch(
    httr::parse_url(normalized),
    error = function(e) NULL
  )
  if (is.null(parsed)) {
    return(FALSE)
  }

  if (is.null(parsed$scheme) || !parsed$scheme %in% c("http", "https")) {
    return(FALSE)
  }
  if (is.null(parsed$hostname) || !nzchar(parsed$hostname)) {
    return(FALSE)
  }
  if (!is.null(parsed$username) || !is.null(parsed$password)) {
    return(FALSE)
  }
  if (!is.null(parsed$query) && length(parsed$query) > 0) {
    return(FALSE)
  }
  if (!is.null(parsed$fragment) && nzchar(parsed$fragment)) {
    return(FALSE)
  }

  authority <- sub("^[A-Za-z][A-Za-z0-9+.-]*://", "", normalized)
  authority <- sub("[/?#].*$", "", authority)
  if (endsWith(authority, ":")) {
    return(FALSE)
  }
  if (!is.null(parsed$port)) {
    if (length(parsed$port) != 1 || !grepl("^[0-9]+$", parsed$port)) {
      return(FALSE)
    }
    port <- suppressWarnings(as.numeric(parsed$port))
    if (is.na(port) || port < 1 || port > 65535) {
      return(FALSE)
    }
  }

  TRUE
}

#' Resolve provider-specific base URL
#'
#' This is the single entry point for all base URL resolution. It resolves
#' the appropriate URL and normalizes it (strips trailing slashes).
#'
#' @param provider Provider name (e.g., "openai", "anthropic")
#' @param base_urls User-provided base URLs: NULL, a single string, or a named list
#' @return Resolved and normalized base URL, or NULL if not specified
#' @keywords internal
resolve_provider_base_url <- function(provider, base_urls) {
  normalize_value <- function(value, source) {
    if (is.null(value)) {
      return(NULL)
    }
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("base_urls%s must be a string URL", source))
    }

    normalized <- sub("/+$", "", trimws(value))
    if (!nzchar(normalized)) {
      return(NULL)
    }
    if (!validate_base_url(normalized)) {
      stop(sprintf("Invalid base URL in base_urls%s: %s", source, value))
    }
    normalized
  }

  if (is.null(base_urls) || is.null(provider)) {
    return(NULL)
  }

  if (!is.character(provider) || length(provider) != 1 || is.na(provider)) {
    return(NULL)
  }

  provider_normalized <- tolower(trimws(provider))
  if (!nzchar(provider_normalized)) {
    return(NULL)
  }

  if (is.character(base_urls) && length(base_urls) == 1) {
    return(normalize_value(base_urls, ""))
  }

  if (is.list(base_urls)) {
    base_url_names <- names(base_urls)
    if (is.null(base_url_names)) {
      stop("base_urls must be a named list")
    }

    normalized_names <- tolower(trimws(base_url_names))
    if (any(is.na(base_url_names)) || any(!nzchar(normalized_names))) {
      stop("base_urls provider names must be non-empty strings")
    }
    duplicate_index <- anyDuplicated(normalized_names)
    if (duplicate_index > 0) {
      duplicate_name <- normalized_names[[duplicate_index]]
      stop("Ambiguous base_urls provider names after case/whitespace normalization: ",
           duplicate_name)
    }
    match_idx <- match(provider_normalized, normalized_names)
    if (is.na(match_idx)) {
      return(NULL)
    }
    return(normalize_value(
      base_urls[[match_idx]],
      sprintf("[['%s']]", base_url_names[[match_idx]])
    ))
  }

  stop("base_urls must be NULL, a single string URL, or a named list")
}
