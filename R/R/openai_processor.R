#' OpenAI API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for OpenAI models.
#' Handles OpenAI-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
OpenAIProcessor <- R6::R6Class("OpenAIProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize OpenAI processor
    #
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
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(chunk_content, model, api_key)
    },
    
    #' @description
    #' Extract response content from OpenAI API response
    #
    #
    #
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)
