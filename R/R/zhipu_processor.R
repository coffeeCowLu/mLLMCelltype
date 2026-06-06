#' Zhipu API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Zhipu models.
#' Handles Zhipu-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
ZhipuProcessor <- R6::R6Class("ZhipuProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Zhipu processor
    #
    initialize = function(base_url = NULL) {
      super$initialize("zhipu", base_url)
    },

    #' @description
    #' Get default Zhipu API URL
    #
    get_default_api_url = function() {
      return("https://api.z.ai/api/paas/v4/chat/completions")
    },
    
    #' @description
    #' Make API call to Zhipu
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(
        chunk_content,
        model,
        api_key,
        body_extra = list(
          temperature = 0.7,
          max_tokens = 4096
        )
      )
    },
    
    #' @description
    #' Extract response content from Zhipu API response
    #
    #
    #
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)
