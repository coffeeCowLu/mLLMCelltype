# Package-level cache for Qwen endpoint (persists across QwenProcessor instances)
.qwen_endpoint_cache <- new.env(parent = emptyenv())

#' Qwen API Processor
#'
#' Concrete implementation of BaseAPIProcessor for Qwen models.
#' Handles Qwen-specific API calls, authentication, and response parsing.
#'
#' @export
QwenProcessor <- R6::R6Class("QwenProcessor",
  inherit = BaseAPIProcessor,

  private = list(

    #' @description
    #' Test if an endpoint is accessible
    #
    #
    #
    test_endpoint = function(url, api_key) {
      tryCatch({
        # Simple OpenAI-compatible test payload. Region-specific DashScope
        # keys can return 401/403 on the wrong regional host, so keep probing
        # alternatives for those statuses.
        test_payload <- list(
          model = "qwen-turbo",
          messages = list(
            list(role = "user", content = "test")
          ),
          max_tokens = 1
        )

        response <- httr::POST(
          url = url,
          httr::add_headers(
            "Authorization" = paste("Bearer", api_key),
            "Content-Type" = "application/json"
          ),
          body = test_payload,
          encode = "json",
          httr::timeout(10)  # 10 second timeout for quick test
        )

        status_code <- httr::status_code(response)
        return(!is.null(status_code) &&
                 !(status_code %in% c(401, 403, 404)) &&
                 status_code < 500)
      }, error = function(e) {
        return(FALSE)
      })
    }
  ),

  public = list(
    #' @description
    #' Initialize Qwen processor
    #' @param base_url Optional custom API endpoint
    initialize = function(base_url = NULL) {
      super$initialize("qwen", base_url)
    },

    #' @description
    #' Get default Qwen OpenAI-compatible chat completions API URL
    #
    #' @details Qwen has OpenAI-compatible chat completions endpoints:
    #'   - International (US): https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions
    #'   - Domestic (China): https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
    #'   - Legacy international: https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions
    #'   The processor automatically tries international first, then domestic, then legacy international.
    get_default_api_url = function() {
      return("https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions")
    },

    #' @description
    #' Get working Qwen API URL with automatic endpoint detection
    #' @param api_key Qwen API key used for regional endpoint probing
    get_working_api_url = function(api_key) {
      cache_key <- digest::digest(api_key, algo = "xxhash64")
      if (exists(cache_key, envir = .qwen_endpoint_cache, inherits = FALSE)) {
        return(get(cache_key, envir = .qwen_endpoint_cache, inherits = FALSE))
      }

      endpoints <- list(
        international = "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        domestic = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        legacy_international = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
      )

      self$logger$debug("Testing Qwen endpoints for accessibility")

      for (endpoint_name in names(endpoints)) {
        endpoint_url <- endpoints[[endpoint_name]]
        if (private$test_endpoint(endpoint_url, api_key)) {
          self$logger$info(
            sprintf("Using Qwen %s endpoint", endpoint_name),
            list(url = endpoint_url)
          )
          assign(cache_key, endpoint_url, envir = .qwen_endpoint_cache)
          return(endpoint_url)
        }
      }

      # If all probes fail, return international as default and let the main call
      # handle the error.
      self$logger$warn(
        "All Qwen endpoints failed during testing, using international as default"
      )
      return(endpoints$international)
    },
    
    #' @description
    #' Make API call to Qwen
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key Qwen API key
    make_api_call = function(chunk_content, model, api_key) {
      # Get API URL: custom base_url takes priority, otherwise auto-detect endpoint
      api_url <- if (!is.null(self$base_url)) {
        self$base_url
      } else {
        self$get_working_api_url(api_key)
      }

      private$post_chat_completions_request(
        chunk_content,
        model,
        api_key,
        api_url = api_url,
        body_extra = list(
          temperature = 0.7,
          max_tokens = 4096
        )
      )
    },
    
    #' @description
    #' Extract response content from Qwen API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing Qwen API response",
                       list(provider = self$provider_name, model = model))

      # Parse the response
      content <- httr::content(response, "parsed")

      # Extract from the current OpenAI-compatible format first, then retain
      # legacy DashScope format support for older cached/mocked responses.
      if (!is.null(content$choices) &&
          length(content$choices) > 0 &&
          !is.null(content$choices[[1]]$message$content)) {
        response_content <- content$choices[[1]]$message$content
      } else if (!is.null(content$output$text)) {
        response_content <- content$output$text
      } else if (!is.null(content$output$choices) &&
                 length(content$output$choices) > 0 &&
                 !is.null(content$output$choices[[1]]$message$content)) {
        response_content <- content$output$choices[[1]]$message$content
      } else {
        self$logger$error("Unexpected response format from Qwen API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              output_keys = if (!is.null(content$output)) names(content$output) else NULL))
        stop("Unexpected response format from Qwen API")
      }

      return(response_content)
    }
  )
)
