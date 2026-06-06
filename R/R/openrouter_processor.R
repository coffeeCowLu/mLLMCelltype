#' OpenRouter API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for OpenRouter models.
#' Handles OpenRouter-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
OpenRouterProcessor <- R6::R6Class("OpenRouterProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize OpenRouter processor
    #
    initialize = function(base_url = NULL) {
      super$initialize("openrouter", base_url)
    },

    #' @description
    #' Get default OpenRouter API URL
    #
    get_default_api_url = function() {
      return("https://openrouter.ai/api/v1/chat/completions")
    },
    
    #' @description
    #' Make API call to OpenRouter
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(chunk_content, model, api_key)
    },
    
    #' @description
    #' Extract response content from OpenRouter API response
    #
    #
    #
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)
