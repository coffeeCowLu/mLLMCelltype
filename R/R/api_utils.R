# Normalize API key mappings before provider- or model-specific lookup.

.normalize_api_keys <- function(api_keys, allow_null = FALSE, validate_values = TRUE) {
  if (is.null(api_keys) && allow_null) {
    return(NULL)
  }
  if (!is.list(api_keys) || length(api_keys) == 0 || is.null(names(api_keys))) {
    stop("api_keys must be a named, non-empty list")
  }

  normalized_names <- trimws(names(api_keys))
  if (anyNA(names(api_keys)) || any(!nzchar(normalized_names))) {
    stop("api_keys names must be non-empty strings")
  }
  if (anyDuplicated(normalized_names)) {
    stop("api_keys names must be unique after whitespace normalization")
  }

  registered_providers <- c(
    names(.BUILTIN_PROVIDER_SPECS),
    ls(custom_providers, all.names = TRUE)
  )
  normalized_provider_names <- tolower(normalized_names)
  provider_keys <- normalized_provider_names[
    normalized_provider_names %in% registered_providers
  ]
  if (anyDuplicated(provider_keys)) {
    stop("api_keys provider names must be unique after case/whitespace normalization")
  }

  names(api_keys) <- normalized_names
  if (validate_values) {
    for (index in seq_along(api_keys)) {
      value <- api_keys[[index]]
      is_missing <- is.null(value) || (
        length(value) == 1 && is.atomic(value) && is.na(value)
      )
      is_character_scalar <- is.character(value) && length(value) == 1
      if (!is_missing && !is_character_scalar) {
        stop(
          "api_keys value for '", normalized_names[[index]],
          "' must be a character scalar or missing"
        )
      }
      if (is_character_scalar && !is.na(value)) {
        api_keys[[index]] <- trimws(value)
      }
    }
  }
  api_keys
}

.get_api_key_from_mapping <- function(model, provider, api_keys) {
  is_valid_key <- function(key) {
    is.character(key) && length(key) == 1 && !is.na(key) && nzchar(trimws(key))
  }
  get_named_key <- function(key_name, ignore_case = FALSE) {
    api_key_names <- names(api_keys)
    if (is.null(api_key_names)) {
      return(NULL)
    }

    match_index <- if (ignore_case) {
      match(tolower(key_name), tolower(api_key_names))
    } else {
      match(key_name, api_key_names)
    }
    if (is.na(match_index)) {
      return(NULL)
    }

    key <- api_keys[[match_index]]
    if (is_valid_key(key)) {
      return(trimws(key))
    }
    NULL
  }

  provider_key <- get_named_key(provider, ignore_case = TRUE)
  if (!is.null(provider_key)) {
    return(provider_key)
  }
  get_named_key(model)
}

.api_key_environment_variables <- function(provider) {
  provider <- tolower(.normalize_required_string(provider, "provider"))
  primary_variable <- paste0(toupper(provider), "_API_KEY")
  provider_spec <- .BUILTIN_PROVIDER_SPECS[[provider]]
  aliases <- if (is.null(provider_spec)) {
    character(0)
  } else {
    provider_spec$api_key_env_aliases
  }
  unique(c(primary_variable, aliases))
}

.resolve_model_api_key <- function(model, api_keys) {
  model <- .normalize_required_string(model, "model")
  api_keys <- .normalize_api_keys(
    api_keys,
    allow_null = TRUE,
    validate_values = FALSE
  )
  provider <- get_provider(model)
  api_key <- .get_api_key_from_mapping(model, provider, api_keys)
  if (!is.null(api_key)) {
    return(list(
      api_key = api_key,
      provider = provider,
      source = "api_keys"
    ))
  }

  for (variable in .api_key_environment_variables(provider)) {
    api_key <- trimws(Sys.getenv(variable, unset = ""))
    if (nzchar(api_key)) {
      return(list(
        api_key = api_key,
        provider = provider,
        source = "environment"
      ))
    }
  }

  list(
    api_key = NULL,
    provider = provider,
    source = "none"
  )
}

#' Get an API key for a model
#'
#' Retrieves a configured API key by checking the model's provider name first,
#' followed by the exact model name.
#'
#' @param model Model name to get API key for
#' @param api_keys Named list of API keys with provider or model names as keys
#'
#' @return A trimmed API key string, or `NULL` when no valid key is configured.
#' @export
get_api_key <- function(model, api_keys) {
  model_normalized <- .normalize_required_string(model, "model")
  api_keys <- .normalize_api_keys(
    api_keys,
    allow_null = TRUE,
    validate_values = FALSE
  )
  provider <- get_provider(model_normalized)
  .get_api_key_from_mapping(model_normalized, provider, api_keys)
}
