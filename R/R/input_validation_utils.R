.normalize_required_string <- function(value, field_name) {
  if (!is.character(value) || length(value) != 1 || is.na(value) ||
      !nzchar(trimws(value))) {
    stop(field_name, " must be a non-empty character scalar")
  }
  trimws(value)
}

.normalize_positive_integer <- function(value, field_name) {
  valid <- is.numeric(value) && length(value) == 1 &&
    !is.na(value) && is.finite(value) &&
    value >= 1 && value == as.integer(value)
  if (!valid) {
    stop(field_name, " must be a positive integer")
  }
  as.integer(value)
}

.normalize_probability <- function(value, field_name) {
  valid <- is.numeric(value) && length(value) == 1 &&
    !is.na(value) && is.finite(value) &&
    value >= 0 && value <= 1
  if (!valid) {
    stop(field_name, " must be a finite number between 0 and 1")
  }
  as.numeric(value)
}

.normalize_nonnegative_number <- function(value, field_name) {
  valid <- is.numeric(value) && length(value) == 1 &&
    !is.na(value) && is.finite(value) && value >= 0
  if (!valid) {
    stop(field_name, " must be a finite non-negative number")
  }
  as.numeric(value)
}

.normalize_positive_number <- function(value, field_name) {
  valid <- is.numeric(value) && length(value) == 1 &&
    !is.na(value) && is.finite(value) && value > 0
  if (!valid) {
    stop(field_name, " must be a finite positive number")
  }
  as.numeric(value)
}

.normalize_log_level <- function(level) {
  normalized <- toupper(.normalize_required_string(level, "level"))
  valid_levels <- c("DEBUG", "INFO", "WARN", "ERROR")
  if (!normalized %in% valid_levels) {
    stop("level must be one of: ", paste(valid_levels, collapse = ", "))
  }
  normalized
}

.normalize_flag <- function(value, field_name) {
  if (!is.logical(value) || length(value) != 1 || is.na(value)) {
    stop(field_name, " must be TRUE or FALSE")
  }
  value
}

.normalize_model_vector <- function(models, minimum_count = 1L) {
  valid <- is.character(models) &&
    length(models) >= minimum_count &&
    !anyNA(models) &&
    all(nzchar(trimws(models)))
  if (!valid) {
    stop(sprintf(
      "models must contain at least %d non-empty model name%s",
      minimum_count,
      if (minimum_count == 1L) "" else "s"
    ))
  }

  normalized <- trimws(models)
  if (anyDuplicated(tolower(normalized))) {
    stop("models must not contain duplicate names")
  }
  normalized
}

.normalize_api_keys <- function(api_keys, allow_null = FALSE) {
  if (is.null(api_keys) && allow_null) {
    return(NULL)
  }
  if (!is.list(api_keys) || length(api_keys) == 0 || is.null(names(api_keys))) {
    stop("api_keys must be a named, non-empty list")
  }

  normalized_names <- tolower(trimws(names(api_keys)))
  if (anyNA(names(api_keys)) || any(!nzchar(normalized_names))) {
    stop("api_keys names must be non-empty strings")
  }
  if (anyDuplicated(normalized_names)) {
    stop("api_keys names must be unique after case/whitespace normalization")
  }

  names(api_keys) <- trimws(names(api_keys))
  api_keys
}
