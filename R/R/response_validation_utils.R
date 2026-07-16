# Internal failure markers that must never become user-facing annotations.
.SENTINEL_VALUES <- c(
  "Parsing_Failed",
  "Insufficient_Responses",
  "Prediction_Missing",
  "Annotation_Missing",
  "Unknown"
)

.UNKNOWN_ANNOTATION_TOKENS <- c(
  tolower(.SENTINEL_VALUES),
  "unk",
  "n/a",
  "na",
  "none",
  "null",
  "nan",
  "-",
  "--",
  "inconclusive"
)

.unwrap_annotation_wrappers <- function(annotation) {
  wrapper_pairs <- c(
    "[" = "]",
    "(" = ")",
    "{" = "}",
    "\"" = "\"",
    "'" = "'"
  )
  normalized <- trimws(annotation)

  while (nchar(normalized) >= 2) {
    opening <- substr(normalized, 1, 1)
    expected_closing <- unname(wrapper_pairs[opening])
    if (is.na(expected_closing) ||
        substr(normalized, nchar(normalized), nchar(normalized)) != expected_closing) {
      break
    }
    normalized <- trimws(substr(normalized, 2, nchar(normalized) - 1))
  }
  normalized
}

is_real_cell_type_annotation <- function(annotation) {
  is_text_scalar <- is.character(annotation) &&
    length(annotation) == 1 &&
    !is.na(annotation)
  if (!is_text_scalar) {
    return(FALSE)
  }

  normalized <- tolower(.unwrap_annotation_wrappers(annotation))
  if (!nzchar(normalized) || normalized %in% .UNKNOWN_ANNOTATION_TOKENS) {
    return(FALSE)
  }

  contextual_unknown <- grepl(
    "^(unknown|inconclusive)\\s*[\\(\\[\\{].*[\\)\\]\\}]$",
    normalized,
    perl = TRUE
  )
  error_sentinel <- grepl(
    "^error(\\s*:.*|\\s*\\(.*\\))?$",
    normalized,
    perl = TRUE
  )
  !contextual_unknown && !error_sentinel
}

is_error_response <- function(result) {
  is.character(result) &&
    length(result) > 0 &&
    any(grepl("^\\s*error\\s*:", result, ignore.case = TRUE), na.rm = TRUE)
}

normalize_model_response_lines <- function(result) {
  if (!is.character(result) || length(result) == 0 || anyNA(result)) {
    stop("Model response must be non-missing text")
  }

  lines <- unlist(strsplit(result, "\n", fixed = TRUE), use.names = FALSE)
  lines <- trimws(lines)
  lines <- sub(",+$", "", lines)
  lines <- trimws(lines)
  lines <- unname(lines[nzchar(lines)])

  if (length(lines) == 0) {
    stop("Model response must contain non-empty text")
  }
  if (is_error_response(lines)) {
    stop("Model response contains an error sentinel")
  }
  lines
}

normalize_token_usage <- function(usage) {
  if (!is.list(usage)) {
    return(NULL)
  }

  normalize_count <- function(value) {
    valid <- is.numeric(value) && length(value) == 1 &&
      !is.na(value) && is.finite(value) && value >= 0 &&
      value == floor(value)
    if (valid) as.numeric(value) else NULL
  }

  normalized <- list(
    prompt_tokens = normalize_count(usage$prompt_tokens),
    completion_tokens = normalize_count(usage$completion_tokens),
    total_tokens = normalize_count(usage$total_tokens)
  )

  cost <- usage$cost
  if (is.numeric(cost) && length(cost) == 1 &&
      !is.na(cost) && is.finite(cost) && cost >= 0) {
    normalized$cost <- as.numeric(cost)
  }
  normalized
}

is_valid_model_response <- function(result) {
  isTRUE(tryCatch({
    normalize_model_response_lines(result)
    TRUE
  }, error = function(e) FALSE))
}

is_retryable_http_status <- function(status_code) {
  is.numeric(status_code) &&
    length(status_code) == 1 &&
    !is.na(status_code) &&
    is.finite(status_code) &&
    status_code == floor(status_code) &&
    (status_code %in% c(408, 425, 429) || status_code >= 500)
}

stop_api_request_error <- function(message, status_code = NULL,
                                   retryable = NULL) {
  if (is.null(retryable)) {
    retryable <- is_retryable_http_status(status_code)
  }
  condition <- structure(
    list(
      message = message,
      call = NULL,
      status_code = status_code,
      retryable = isTRUE(retryable)
    ),
    class = c("mllm_api_error", "error", "condition")
  )
  stop(condition)
}
