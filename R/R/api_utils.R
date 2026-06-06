#' Utility functions for API key management
#' 
#' This file contains utility functions for managing API keys and related operations.

#' Get API key for a specific model
#' 
#' This function retrieves the appropriate API key for a given model by first checking
#' the provider name and then the model name in the provided API keys list.
#' 
#' @param model Model name to get API key for
#' @param api_keys Named list of API keys with provider or model names as keys
#'
#' @return API key string for the specified model
#' @export
get_api_key <- function(model, api_keys) {
  provider <- get_provider(model)
  model_normalized <- trimws(model)
  is_valid_key <- function(key) {
    is.character(key) && length(key) == 1 && !is.na(key) && nzchar(trimws(key))
  }
  get_named_key <- function(key_name) {
    api_key_names <- names(api_keys)
    if (is.null(api_key_names)) {
      return(NULL)
    }

    match_idx <- match(tolower(trimws(key_name)), tolower(trimws(api_key_names)))
    if (is.na(match_idx)) {
      return(NULL)
    }

    key <- api_keys[[match_idx]]
    if (is_valid_key(key)) {
      return(trimws(key))
    }
    NULL
  }

  # First try to get by provider name
  key <- get_named_key(provider)
  if (!is.null(key)) {
    return(key)
  }

  # If not found, try to get by model name
  key <- get_named_key(model_normalized)
  if (!is.null(key)) {
    return(key)
  }

  # If still not found or all keys empty, return NULL
  return(NULL)
}
