#' Grok API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Grok models.
#' Handles Grok-specific API calls, authentication, and response parsing.
#'
#' @export
GrokProcessor <- R6::R6Class("GrokProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Grok processor
    #' @param base_url Optional custom API endpoint
    initialize = function(base_url = NULL) {
      super$initialize("grok", base_url)
    },

    #' @description
    #' Get default Grok API URL
    #
    get_default_api_url = function() {
      return("https://api.x.ai/v1/chat/completions")
    },
    
    #' @description
    #' Make API call to Grok
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key Grok API key
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(chunk_content, model, api_key)
    },
    
    #' @description
    #' Extract response content from Grok API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)
