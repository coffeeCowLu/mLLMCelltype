#' OpenAI API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for OpenAI models.
#' Handles OpenAI-specific API calls, authentication, and response parsing.
#'
#' @export
OpenAIProcessor <- R6::R6Class("OpenAIProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize OpenAI processor
    #' @param base_url Optional custom API endpoint
    initialize = function(base_url = NULL) {
      super$initialize("openai", base_url)
    },

    #' @description
    #' Get default OpenAI API URL
    #
    get_default_api_url = function() {
      return("https://api.openai.com/v1/chat/completions")
    },
    
    #' @description
    #' Make API call to OpenAI
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key OpenAI API key
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(chunk_content, model, api_key)
    },
    
    #' @description
    #' Extract response content from OpenAI API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)
