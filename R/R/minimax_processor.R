#' Minimax API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Minimax models.
#' Handles Minimax-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
MinimaxProcessor <- R6::R6Class("MinimaxProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Minimax processor
    #
    initialize = function(base_url = NULL) {
      super$initialize("minimax", base_url)
    },

    #' @description
    #' Get default MiniMax OpenAI-compatible chat completions API URL
    #
    get_default_api_url = function() {
      return("https://api.minimax.io/v1/chat/completions")
    },
    
    #' @description
    #' Make API call to Minimax
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(chunk_content, model, api_key)
    },
    
    #' @description
    #' Extract response content from Minimax API response
    #
    #
    #
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing MiniMax API response",
                       list(provider = self$provider_name, model = model))
      
      # Parse the response
      content <- httr::content(response, "parsed")
      
      base_resp <- content$base_resp
      if (is.list(base_resp) &&
          !is.null(base_resp$status_code) &&
          !identical(as.integer(base_resp$status_code), 0L)) {
        status_msg_available <- is.character(base_resp$status_msg) &&
          length(base_resp$status_msg) == 1 &&
          !is.na(base_resp$status_msg) &&
          nzchar(base_resp$status_msg)
        error_message <- if (status_msg_available) {
          base_resp$status_msg
        } else {
          "Unknown MiniMax API error"
        }
        stop(sprintf("MiniMax API error: %s", error_message))
      }

      if (!is.null(content$choices) &&
          length(content$choices) > 0 &&
          !is.null(content$choices[[1]]$message$content)) {
        response_content <- content$choices[[1]]$message$content
      } else if (!is.null(content$choices) &&
                 length(content$choices) > 0 &&
                 !is.null(content$choices[[1]]$messages) &&
                 length(content$choices[[1]]$messages) > 0 &&
                 !is.null(content$choices[[1]]$messages[[1]]$text)) {
        # Legacy MiniMax text/chatcompletion_v2 shape.
        response_content <- content$choices[[1]]$messages[[1]]$text
      } else {
        
        self$logger$error("Unexpected response format from MiniMax API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              choices_available = !is.null(content$choices),
                              choices_count = if (!is.null(content$choices)) {
                                length(content$choices)
                              } else {
                                0
                              }))
        
        stop("Unexpected response format from MiniMax API")
      }

      response_content <- gsub(
        "<think>[\\s\\S]*?</think>\\s*",
        "",
        response_content,
        perl = TRUE
      )

      return(response_content)
    }
  )
)
