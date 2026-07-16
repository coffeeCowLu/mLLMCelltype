
#' Normalize annotation for comparison
#
#
#' @keywords internal
normalize_annotation <- function(annotation) {
  # Convert to lowercase
  normalized <- tolower(trimws(annotation))
  
  # Replace punctuation with space to preserve word boundaries (e.g., "T-cell" -> "T cell")
  normalized <- gsub("[[:punct:]]", " ", normalized)
  # Normalize whitespace
  normalized <- gsub("\\s+", " ", normalized)
  normalized <- trimws(normalized)
  
  # Handle plurals (simple approach)
  normalized <- gsub("cells$", "cell", normalized)
  normalized <- gsub("cytes$", "cyte", normalized)
  
  # Common cell type variations
  normalized <- gsub("t lymphocyte", "t cell", normalized)
  normalized <- gsub("b lymphocyte", "b cell", normalized)
  normalized <- gsub("natural killer", "nk", normalized)
  
  return(normalized)
}

#' Extract discussion cell type label
#'
#' @keywords internal
#' @noRd
extract_discussion_cell_type <- function(response, default = "Unknown") {
  lines <- tryCatch(
    normalize_model_response_lines(response),
    error = function(e) NULL
  )
  if (is.null(lines)) {
    return(default)
  }
  cell_type_lines <- grep("^CELL TYPE:", trimws(lines), value = TRUE, ignore.case = TRUE)
  if (length(cell_type_lines) > 0) {
    annotation <- trimws(sub("^CELL TYPE:\\s*", "", cell_type_lines[1], ignore.case = TRUE))
    if (nzchar(annotation)) {
      return(annotation)
    }
    return(default)
  }
  if (length(lines) == 1) {
    return(lines)
  }

  default
}

#' Calculate simple consensus without LLM
#
#
#' @keywords internal
calculate_simple_consensus <- function(round_responses) {
  valid_responses <- vapply(
    as.list(round_responses),
    is_valid_model_response,
    logical(1)
  )
  round_responses <- round_responses[valid_responses]
  if (length(round_responses) == 0) {
    return(list(
      consensus_proportion = 0,
      entropy = 0,
      majority_prediction = "Insufficient_Responses",
      has_unique_majority = FALSE
    ))
  }

  # Normalize all annotations
  normalized_responses <- sapply(round_responses, normalize_annotation)
  
  # Count occurrences
  response_counts <- table(normalized_responses)
  total_responses <- length(round_responses)
  
  
  # Find most common response
  most_common_idx <- which.max(response_counts)
  most_common_count <- as.numeric(response_counts[most_common_idx])
  majority_groups <- names(response_counts)[response_counts == most_common_count]
  has_unique_majority <- length(majority_groups) == 1
  
  # Preserve model order when multiple groups are tied.
  matching_idx <- which(normalized_responses %in% majority_groups)[1]
  majority_prediction <- unname(round_responses[matching_idx])
  
  # Calculate consensus proportion
  consensus_proportion <- most_common_count / total_responses
  
  # Calculate Shannon entropy
  proportions <- as.numeric(response_counts) / total_responses
  entropy <- -sum(proportions * log2(proportions))
  
  
  return(list(
    consensus_proportion = consensus_proportion,
    entropy = entropy,
    majority_prediction = majority_prediction,
    has_unique_majority = has_unique_majority
  ))
}

simple_consensus_reached <- function(result, controversy_threshold,
                                     entropy_threshold) {
  isTRUE(result$has_unique_majority) &&
    result$consensus_proportion >= controversy_threshold &&
    result$entropy <= entropy_threshold
}

# Validate that consensus metrics can arise from a finite set of votes.
consensus_metrics_plausible <- function(result, vote_count, tolerance = 0.02) {
  consensus_proportion <- result$consensus_proportion
  entropy <- result$entropy

  if (length(vote_count) != 1 || !is.numeric(vote_count) ||
      is.na(vote_count) || !is.finite(vote_count) || vote_count < 1) {
    return(FALSE)
  }
  if (length(consensus_proportion) != 1 || !is.numeric(consensus_proportion) ||
      is.na(consensus_proportion) || !is.finite(consensus_proportion) ||
      consensus_proportion < 0 || consensus_proportion > 1) {
    return(FALSE)
  }
  if (length(entropy) != 1 || !is.numeric(entropy) ||
      is.na(entropy) || !is.finite(entropy) || entropy < 0) {
    return(FALSE)
  }

  supporting_votes <- consensus_proportion * vote_count
  nearest_vote_count <- round(supporting_votes)
  if (nearest_vote_count < 1 || nearest_vote_count > vote_count ||
      abs(supporting_votes - nearest_vote_count) > tolerance) {
    return(FALSE)
  }

  group_sizes <- nearest_vote_count
  remaining_votes <- vote_count - nearest_vote_count
  while (remaining_votes > 0) {
    next_group_size <- min(nearest_vote_count, remaining_votes)
    group_sizes <- c(group_sizes, next_group_size)
    remaining_votes <- remaining_votes - next_group_size
  }

  group_proportions <- group_sizes / vote_count
  minimum_entropy <- -sum(group_proportions * log2(group_proportions))
  majority_share <- nearest_vote_count / vote_count
  maximum_entropy <- -(majority_share * log2(majority_share))
  if (nearest_vote_count < vote_count) {
    singleton_share <- 1 / vote_count
    maximum_entropy <- maximum_entropy -
      (vote_count - nearest_vote_count) * singleton_share * log2(singleton_share)
  }

  entropy >= minimum_entropy - tolerance &&
    entropy <= maximum_entropy + tolerance
}

# Constants for consensus checking
.CONSENSUS_CONSTANTS <- list(
  DEFAULT_RESPONSE = "0\n0\n0\nUnknown",
  FALLBACK_MODELS = c("claude-sonnet-4-6", "gpt-5.5", "gemini-3-flash-preview", "qwen3.6-flash", "deepseek-v4-flash"),
  NUMBER_PATTERN = "^[+-]?([0-9]+(\\.[0-9]+)?|\\.[0-9]+)$",
  SEPARATOR_PATTERN = "(?:[:=]|\uFF1A)",
  SEARCH_PATTERNS = list(
    CONSENSUS_LABEL = "(?:consensus\\s+proportion(?:\\s*\\(\\s*cp\\s*\\))?|cp)",
    ENTROPY_LABEL = "(?:(?:shannon\\s+)?entropy(?:\\s*\\(\\s*h\\s*\\))?|h)"
  )
)

# Default values for failed parsing
.DEFAULT_CONSENSUS_RESULT <- list(
  reached = FALSE,
  consensus_proportion = 0,
  entropy = 0,
  majority_prediction = "Unknown"
)

parse_consensus_number <- function(value) {
  value <- trimws(value)
  if (length(value) != 1 || is.na(value) ||
      !grepl(.CONSENSUS_CONSTANTS$NUMBER_PATTERN, value)) {
    return(NULL)
  }

  number <- suppressWarnings(as.numeric(value))
  if (!is.finite(number)) NULL else number
}

#' Prepare list of models to try for consensus checking
#
#
#' @keywords internal
prepare_models_list <- function(consensus_check_model = NULL) {
  models_to_try <- c()
  
  if (!is.null(consensus_check_model)) {
    get_logger()$info("Using specified consensus check model", list(model = consensus_check_model))
    
    if (grepl("/", consensus_check_model)) {
      parts <- strsplit(consensus_check_model, "/")[[1]]
      if (length(parts) > 1) {
        base_model <- paste(parts[-1], collapse = "/")
        direct_provider <- if (grepl("/", base_model)) {
          NULL
        } else {
          tryCatch(get_provider(base_model), error = function(e) NULL)
        }
        if (!is.null(direct_provider)) {
          get_logger()$info("Detected OpenRouter model. Using base model name", list(base_model = base_model))
          models_to_try <- c(consensus_check_model, base_model)
        } else {
          models_to_try <- c(consensus_check_model)
        }
      } else {
        models_to_try <- c(consensus_check_model)
      }
    } else {
      models_to_try <- c(consensus_check_model)
    }
  }
  
  fallback_models <- .CONSENSUS_CONSTANTS$FALLBACK_MODELS[!.CONSENSUS_CONSTANTS$FALLBACK_MODELS %in% models_to_try]
  c(models_to_try, fallback_models)
}

#' Parse standard 4-line consensus response format
#
#
#' @keywords internal
parse_standard_format <- function(result_lines) {
  if (length(result_lines) != 4) return(NULL)

  indicator <- trimws(result_lines[[1]])
  consensus_proportion <- parse_consensus_number(result_lines[[2]])
  entropy <- parse_consensus_number(result_lines[[3]])
  valid_metrics <- !is.null(consensus_proportion) &&
    consensus_proportion >= 0 &&
    consensus_proportion <= 1 &&
    !is.null(entropy) &&
    entropy >= 0
  if (!indicator %in% c("0", "1") || !valid_metrics) {
    return(NULL)
  }

  get_logger()$debug("Detected standard 4-line format")

  list(
    reached = indicator == "1",
    consensus_proportion = consensus_proportion,
    entropy = entropy,
    majority_prediction = trimws(result_lines[[4]])
  )
}

#' Extract numeric value from line containing a label
#
#
#
#
#' @keywords internal
extract_labeled_value <- function(lines, label_pattern) {
  line_pattern <- paste0(
    "^\\s*",
    label_pattern,
    "\\s*",
    .CONSENSUS_CONSTANTS$SEPARATOR_PATTERN,
    "\\s*(.*?)\\s*$"
  )
  for (line in lines) {
    match <- regexec(line_pattern, line, ignore.case = TRUE, perl = TRUE)
    captures <- regmatches(line, match)[[1]]
    if (length(captures) != 2) {
      next
    }

    value <- parse_consensus_number(captures[[2]])
    if (!is.null(value)) {
      get_logger()$debug("Found value in line", list(value = value, line = line))
      return(value)
    }
  }
  NULL
}

extract_consensus_indicator <- function(lines) {
  first_line <- trimws(lines[[1]])
  if (first_line %in% c("0", "1")) {
    return(first_line == "1")
  }

  indicator_pattern <- paste0(
    "^\\s*consensus(?:\\s+reached)?\\s*",
    .CONSENSUS_CONSTANTS$SEPARATOR_PATTERN,
    "\\s*",
    "(0|1|true|false|yes|no)\\s*$"
  )
  for (line in lines) {
    match <- regexec(indicator_pattern, line, ignore.case = TRUE, perl = TRUE)
    captures <- regmatches(line, match)[[1]]
    if (length(captures) == 2) {
      return(tolower(captures[[2]]) %in% c("1", "true", "yes"))
    }
  }
  FALSE
}

#' Find majority prediction from response lines
#
#
#' @keywords internal
find_majority_prediction <- function(lines) {
  structured_labels <- c(
    "majority(?:\\s+prediction|\\s+cell\\s*type)?",
    "cell\\s*type",
    "claim"
  )
  separators <- paste0(
    "\\s*",
    .CONSENSUS_CONSTANTS$SEPARATOR_PATTERN,
    "\\s*"
  )

  for (line in lines) {
    for (label_pattern in structured_labels) {
      pattern <- paste0("^\\s*", label_pattern, separators, "(.*?)\\s*$")
      match <- regexec(pattern, line, ignore.case = TRUE, perl = TRUE)
      captures <- regmatches(line, match)[[1]]
      if (length(captures) == 2 && nzchar(trimws(captures[[2]]))) {
        return(trimws(captures[[2]]))
      }
    }
  }

  metadata_labels <- c(
    .CONSENSUS_CONSTANTS$SEARCH_PATTERNS$CONSENSUS_LABEL,
    .CONSENSUS_CONSTANTS$SEARCH_PATTERNS$ENTROPY_LABEL,
    "consensus(?:\\s+reached)?",
    structured_labels
  )
  for (line in lines) {
    line_clean <- trimws(line)
    if (!nzchar(line_clean) || !is.null(parse_consensus_number(line_clean))) {
      next
    }

    is_metadata <- any(vapply(metadata_labels, function(label_pattern) {
      grepl(
        paste0("^\\s*", label_pattern, separators),
        line_clean,
        ignore.case = TRUE,
        perl = TRUE
      )
    }, logical(1)))
    if (is_metadata) {
      next
    }

    return(line_clean)
  }

  "Parsing_Failed"
}

#' Parse flexible format consensus response
#
#
#' @keywords internal
parse_flexible_format <- function(lines) {
  result <- .DEFAULT_CONSENSUS_RESULT
  result$reached <- extract_consensus_indicator(lines)

  proportion_value <- extract_labeled_value(
    lines,
    .CONSENSUS_CONSTANTS$SEARCH_PATTERNS$CONSENSUS_LABEL
  )
  if (!is.null(proportion_value) && proportion_value >= 0 && proportion_value <= 1) {
    result$consensus_proportion <- proportion_value
  }

  entropy_value <- extract_labeled_value(
    lines,
    .CONSENSUS_CONSTANTS$SEARCH_PATTERNS$ENTROPY_LABEL
  )
  if (!is.null(entropy_value) && entropy_value >= 0) {
    result$entropy <- entropy_value
  }

  result$majority_prediction <- find_majority_prediction(lines)
  result
}

#' Parse consensus response from model
#
#
#' @keywords internal
parse_consensus_response <- function(response) {
  lines <- tryCatch(
    normalize_model_response_lines(response),
    error = function(e) {
      get_logger()$warn("Invalid consensus response", list(error = e$message))
      NULL
    }
  )
  if (is.null(lines)) {
    return(.DEFAULT_CONSENSUS_RESULT)
  }
  
  if (length(lines) < 4) {
    get_logger()$warn("Not enough lines for standard format, trying flexible parsing", list(line_count = length(lines)))
  } else {
    # Try standard format first
    result_lines <- tail(lines, 4)
    standard_result <- parse_standard_format(result_lines)
    
    if (!is.null(standard_result)) {
      get_logger()$info("Parsed standard format", list(
        consensus = standard_result$reached,
        proportion = standard_result$consensus_proportion,
        entropy = standard_result$entropy
      ))
      return(standard_result)
    }
  }

  # Fall back to flexible parsing
  get_logger()$debug("Using flexible format parsing")
  parse_flexible_format(lines)
}

#' Execute consensus check across ordered model candidates
#' @param formatted_responses Prompt containing model responses
#' @param api_keys Named API key list
#' @param models_to_try Ordered model candidates
#' @param base_urls Optional provider base URLs
#' @return A list containing success status and response text
#' @keywords internal
execute_consensus_check <- function(formatted_responses, api_keys, models_to_try, base_urls = NULL) {
  for (model_name in models_to_try) {
    get_logger()$info("Trying model for consensus check", list(model = model_name))
    
    credential <- .resolve_model_api_key(model_name, api_keys)
    if (is.null(credential$api_key)) {
      get_logger()$warn("No API key available for model, skipping", list(model = model_name))
      next
    }
    
    result <- tryCatch({
      temp_response <- get_model_response(
        formatted_responses,
        model_name,
        credential$api_key,
        base_urls
      )

      if (is.character(temp_response) && length(temp_response) > 1) {
        temp_response <- paste(temp_response, collapse = "\n")
      }

      get_logger()$info("Successfully got response from model", list(model = model_name))
      list(success = TRUE, response = temp_response)
    }, error = function(error) {
      get_logger()$error("Consensus model request failed", list(
        model = model_name,
        error = error$message
      ))
      list(success = FALSE, response = NULL)
    })

    if (result$success) {
      return(result)
    }
  }
  
  # All attempts failed
  get_logger()$warn("All model attempts failed, consensus check could not be performed")
  warning("All available models failed for consensus check. Please ensure at least one model API key is valid.")
  list(success = FALSE, response = .CONSENSUS_CONSTANTS$DEFAULT_RESPONSE)
}

#' Check if consensus is reached among models
#
#
#
#
#
#' @note This function uses create_consensus_check_prompt from prompt_templates.R
#' @importFrom utils tail
#' @keywords internal
check_consensus <- function(round_responses, api_keys = NULL, controversy_threshold = 2/3, entropy_threshold = 1.0, consensus_check_model = NULL, base_urls = NULL) {
  # Initialize logging
  get_logger()$info("Starting check_consensus function")
  get_logger()$debug("Input responses", list(responses = round_responses))

  controversy_threshold <- .normalize_probability(
    controversy_threshold,
    "controversy_threshold"
  )
  entropy_threshold <- .normalize_nonnegative_number(
    entropy_threshold,
    "entropy_threshold"
  )

  response_list <- as.list(round_responses)
  valid_response_mask <- vapply(
    response_list,
    is_valid_model_response,
    logical(1)
  )
  response_list <- response_list[valid_response_mask]

  if (length(response_list) < 2) {
    get_logger()$warn("Not enough responses to check consensus", list(response_count = length(response_list)))
    return(list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = "Insufficient_Responses"))
  }

  # OPTIMIZATION: First try simple consensus calculation
  get_logger()$info("Starting with simple consensus calculation")
  
  # Extract cell types from responses if they are discussion format
  extracted_cell_types <- vapply(
    response_list,
    extract_discussion_cell_type,
    character(1),
    default = NA_character_
  )
  real_annotation_mask <- vapply(
    as.list(extracted_cell_types),
    is_real_cell_type_annotation,
    logical(1)
  )
  extracted_cell_types <- extracted_cell_types[real_annotation_mask]
  if (length(extracted_cell_types) < 2) {
    get_logger()$warn("Not enough parseable cell type responses to check consensus", list(
      response_count = length(extracted_cell_types)
    ))
    return(list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = "Insufficient_Responses"))
  }
  
  
  # Calculate simple consensus
  simple_result <- calculate_simple_consensus(extracted_cell_types)
  deterministic_result <- list(
    reached = simple_consensus_reached(
      simple_result,
      controversy_threshold,
      entropy_threshold
    ),
    consensus_proportion = simple_result$consensus_proportion,
    entropy = simple_result$entropy,
    majority_prediction = simple_result$majority_prediction
  )

  if (deterministic_result$reached) {
    # Simple consensus is sufficient
    get_logger()$info("CONSENSUS ACHIEVED WITH SIMPLE CHECK - NO LLM NEEDED", list(
      proportion = simple_result$consensus_proportion,
      entropy = simple_result$entropy,
      controversy_threshold = controversy_threshold,
      entropy_threshold = entropy_threshold,
      majority = simple_result$majority_prediction
    ))
    
    message(sprintf("Cluster consensus achieved via SIMPLE CHECK: %s (CP=%.2f, H=%.2f)",
                    simple_result$majority_prediction,
                    simple_result$consensus_proportion,
                    simple_result$entropy))
    
    return(deterministic_result)
  }

  # Simple consensus didn't meet thresholds, use LLM for double-checking
  failure_reasons <- c(
    if (!isTRUE(simple_result$has_unique_majority)) "no unique majority",
    if (simple_result$consensus_proportion < controversy_threshold) {
      "consensus proportion below threshold"
    },
    if (simple_result$entropy > entropy_threshold) "entropy above threshold"
  )
  get_logger()$info("Simple consensus BELOW threshold, REQUIRING LLM double-check", list(
    proportion = simple_result$consensus_proportion,
    entropy = simple_result$entropy,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    reasons = failure_reasons
  ))

  message(sprintf("Simple check insufficient (%s; CP=%.2f, H=%.2f) - Using LLM",
                  paste(failure_reasons, collapse = "; "),
                  simple_result$consensus_proportion,
                  simple_result$entropy))

  # Get the formatted prompt from the dedicated function
  formatted_responses <- create_consensus_check_prompt(extracted_cell_types, controversy_threshold, entropy_threshold)

  # Prepare models and execute consensus check
  models_to_try <- prepare_models_list(consensus_check_model)
  execution_result <- execute_consensus_check(formatted_responses, api_keys, models_to_try, base_urls)

  # Handle execution failure - fall back to simple consensus
  if (!execution_result$success) {
    get_logger()$error("All model attempts failed, using simple consensus results")
    return(deterministic_result)
  }

  # Parse the response using the new modular approach
  result <- parse_consensus_response(execution_result$response)

  if (!consensus_metrics_plausible(result, length(extracted_cell_types))) {
    get_logger()$warn("Ignoring impossible LLM consensus metrics", list(
      consensus_proportion = result$consensus_proportion,
      entropy = result$entropy,
      vote_count = length(extracted_cell_types)
    ))
    return(deterministic_result)
  }

  reported_majority <- result$majority_prediction
  valid_majority <- is_real_cell_type_annotation(reported_majority)
  metrics_reach_consensus <- result$consensus_proportion >= controversy_threshold &&
    result$entropy <= entropy_threshold
  llm_claimed_consensus <- isTRUE(result$reached)
  result$reached <- llm_claimed_consensus && metrics_reach_consensus && valid_majority
  if (!valid_majority) {
    result$majority_prediction <- deterministic_result$majority_prediction
  }

  if (llm_claimed_consensus && !result$reached) {
    get_logger()$warn("Ignoring inconsistent LLM consensus indicator", list(
      consensus_proportion = result$consensus_proportion,
      entropy = result$entropy,
      majority_prediction = reported_majority
    ))
  }
  
  # Compare LLM result with simple consensus
  if (result$reached) {
    get_logger()$info("LLM confirmed consensus", list(
      majority_prediction = result$majority_prediction,
      consensus_proportion = result$consensus_proportion,
      entropy = result$entropy
    ))
    message(sprintf("LLM CONFIRMED consensus: %s (CP=%.2f, H=%.2f)",
                    result$majority_prediction,
                    result$consensus_proportion,
                    result$entropy))
  } else {
    get_logger()$info("LLM rejected consensus", list(
      simple_suggestion = simple_result$majority_prediction,
      llm_consensus_proportion = result$consensus_proportion,
      llm_entropy = result$entropy
    ))
    message(sprintf("LLM REJECTED consensus: Simple suggested %s but LLM found CP=%.2f, H=%.2f",
                    simple_result$majority_prediction,
                    result$consensus_proportion,
                    result$entropy))
  }
  

  return(result)
}
