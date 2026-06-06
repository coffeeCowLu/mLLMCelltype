#' StepFun API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for StepFun models.
#' Handles StepFun-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
StepFunProcessor <- R6::R6Class("StepFunProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize StepFun processor
    #
    initialize = function(base_url = NULL) {
      super$initialize("stepfun", base_url)
    },

    #' @description
    #' Get default StepFun API URL
    #
    get_default_api_url = function() {
      return("https://api.stepfun.com/v1/chat/completions")
    },
    
    #' @description
    #' Make API call to StepFun
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
    #' Extract response content from StepFun API response
    #
    #
    #
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)
