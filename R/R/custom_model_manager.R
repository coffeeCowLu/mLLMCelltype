
# Custom model manager for user-defined providers and models.

.normalize_custom_identifier <- function(value, field_name) {
  normalized <- tolower(.normalize_required_string(value, field_name))
  if (!grepl("^[a-z0-9][a-z0-9._-]*$", normalized)) {
    stop(
      field_name,
      " must contain only letters, digits, dots, underscores, or hyphens"
    )
  }
  normalized
}

# Environments storing custom provider and model registrations.
custom_providers <- new.env(parent = emptyenv())
custom_models <- new.env(parent = emptyenv())

#' Register a custom LLM provider
#'
#' @param provider_name Unique name for the custom provider
#' @param process_fn Function that processes LLM requests. Must accept parameters:
#'   prompt, model, api_key; may optionally accept model_config and base_url
#' @param description Optional description of the provider
#'
#' @return Invisible NULL
#' @export
#'
#' @examples
#' \dontrun{
#' register_custom_provider(
#'   provider_name = "my_provider",
#'   process_fn = function(prompt, model, api_key) {
#'     # Custom implementation
#'     response <- httr::POST(
#'       url = "your_api_endpoint",
#'       body = list(prompt = prompt),
#'       encode = "json"
#'     )
#'     return(httr::content(response)$choices[[1]]$text)
#'   }
#' )
#' }
register_custom_provider <- function(provider_name, process_fn,
                                     description = NULL) {
  # Input validation
  if (!is.function(process_fn)) {
    stop("process_fn must be a function")
  }

  # Normalize provider name to lowercase for consistent lookup
  provider_name <- .normalize_custom_identifier(provider_name, "provider_name")
  if (provider_name %in% names(.BUILTIN_PROVIDER_SPECS)) {
    stop("Provider '", provider_name, "' is reserved for a built-in provider")
  }

  # Check if provider already exists
  if (exists(provider_name, envir = custom_providers)) {
    stop("Provider '", provider_name, "' already exists")
  }

  # Validate process_fn arguments
  fn_args <- names(formals(process_fn))
  required_args <- c("prompt", "model", "api_key")
  if (!all(required_args %in% fn_args)) {
    stop("process_fn must accept parameters: ",
         paste(required_args, collapse = ", "))
  }

  # Store provider configuration
  assign(provider_name,
         list(
           process_fn = process_fn,
           description = description,
           models = character(0)
         ),
         envir = custom_providers)

  get_logger()$info("Registered custom provider", list(provider = provider_name))
  invisible(TRUE)
}

#' Register a custom model for a provider
#'
#' @param model_name Unique name for the custom model
#' @param provider_name Name of the provider this model belongs to
#' @param model_config List of configuration parameters for the model (e.g., temperature, max_tokens)
#'
#' @return Invisible TRUE on success
#' @export
#'
#' @examples
#' \dontrun{
#' register_custom_model(
#'   model_name = "my_model",
#'   provider_name = "my_provider",
#'   model_config = list(
#'     temperature = 0.7,
#'     max_tokens = 2000
#'   )
#' )
#' }
register_custom_model <- function(model_name, provider_name,
                                  model_config = list()) {
  # Input validation
  if (!is.list(model_config)) {
    stop("model_config must be a list")
  }

  # Normalize names to lowercase for consistent lookup with get_provider()
  model_name <- .normalize_custom_identifier(model_name, "model_name")
  provider_name <- .normalize_custom_identifier(provider_name, "provider_name")

  # Check if provider exists
  if (!exists(provider_name, envir = custom_providers)) {
    stop("Provider '", provider_name, "' does not exist")
  }

  # Check if model already exists
  if (exists(model_name, envir = custom_models)) {
    stop("Model '", model_name, "' already exists")
  }

  # Store model configuration
  assign(model_name,
         list(
           provider = provider_name,
           config = model_config
         ),
         envir = custom_models)

  # Update provider's model list
  provider_data <- get(provider_name, envir = custom_providers)
  provider_data$models <- c(provider_data$models, model_name)
  assign(provider_name, provider_data, envir = custom_providers)

  get_logger()$info("Registered custom model for provider",
           list(model = model_name, provider = provider_name))
  invisible(TRUE)
}

#' Process request using custom provider
#' @keywords internal
process_custom <- function(prompt, model, api_key, base_url = NULL) {
  # Normalize to lowercase for consistent lookup with get_provider()
  model_lower <- .normalize_custom_identifier(model, "model")

  # Check if model exists
  if (!exists(model_lower, envir = custom_models)) {
    stop("Model '", model, "' not found")
  }

  # Get model and provider data
  model_data <- get(model_lower, envir = custom_models)
  provider_data <- get(model_data$provider, envir = custom_providers)

  # Call provider's process function
  get_logger()$info("Processing request with custom model", list(model = model))
  tryCatch({
    process_args <- names(formals(provider_data$process_fn))
    accepts_dots <- "..." %in% process_args
    call_args <- list(prompt = prompt, model = model, api_key = api_key)

    if ("model_config" %in% process_args || "..." %in% process_args) {
      call_args$model_config <- model_data$config
    }
    if (!is.null(base_url)) {
      if (!("base_url" %in% process_args || accepts_dots)) {
        stop("Custom provider process_fn does not accept base_url")
      }
      call_args$base_url <- base_url
    }

    response <- do.call(provider_data$process_fn, call_args)
    get_logger()$info("Custom model request processed successfully")
    return(response)
  }, error = function(e) {
    get_logger()$error("Error processing custom model request",
              list(error = e$message))
    if (inherits(e, "mllm_api_error")) {
      stop(e)
    }
    stop("Failed to process request with custom model: ", e$message)
  })
}

#' Get list of registered custom providers
#
#' @export
list_custom_providers <- function() {
  ls(envir = custom_providers)
}

#' Get list of registered custom models
#
#' @export
list_custom_models <- function() {
  ls(envir = custom_models)
}
