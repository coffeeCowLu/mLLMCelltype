# Tests for the Kimi provider (Moonshot AI Open Platform)

make_kimi_json_response <- function(payload,
                                    status_code = 200L,
                                    url = "https://api.moonshot.cn/v1/chat/completions") {
  structure(
    list(
      url = url,
      status_code = status_code,
      headers = list("Content-Type" = "application/json"),
      all_headers = list(list(
        status = status_code,
        version = "HTTP/1.1",
        headers = list("Content-Type" = "application/json")
      )),
      cookies = data.frame(),
      content = charToRaw(jsonlite::toJSON(payload, auto_unbox = TRUE)),
      date = Sys.time(),
      times = c(),
      request = list(method = "POST")
    ),
    class = "response"
  )
}

preserve_kimi_env_variables <- function(variables) {
  previous <- Sys.getenv(variables, unset = NA_character_)

  function() {
    Sys.unsetenv(variables)
    configured <- !is.na(previous)
    if (any(configured)) {
      do.call(
        Sys.setenv,
        setNames(as.list(previous[configured]), variables[configured])
      )
    }
  }
}

quiet_kimi_logger <- function() {
  list(
    info = function(...) NULL,
    debug = function(...) NULL,
    warn = function(...) NULL,
    error = function(...) NULL,
    log_api_call = function(...) NULL,
    log_api_request_response = function(...) NULL
  )
}

test_that("get_provider detects Kimi Open Platform model prefixes", {
  expect_identical(get_provider("kimi-k2.6"), "kimi")
  expect_identical(get_provider("moonshot-v1-8k"), "kimi")
  expect_identical(get_provider("KIMI-K2.6"), "kimi")
  expect_identical(get_provider("  Moonshot-v1-8k  "), "kimi")
})

test_that("kimi registry entry owns pattern, processor, display name, and env aliases", {
  spec <- .BUILTIN_PROVIDER_SPECS$kimi
  expect_identical(spec$pattern, "^(kimi-|moonshot-)")
  expect_identical(spec$processor_class, "KimiProcessor")
  expect_identical(spec$display_name, "Kimi")
  expect_identical(spec$api_key_env_aliases, "MOONSHOT_API_KEY")
  expect_identical(.BUILTIN_PROVIDER_PATTERNS[["kimi"]], "^(kimi-|moonshot-)")
  expect_identical(get_builtin_provider_display_name("kimi"), "Kimi")
  expect_silent(validate_builtin_provider_registry())

  processor <- new_builtin_provider_processor("kimi")
  expect_s3_class(processor, "KimiProcessor")
  expect_identical(processor$provider_name, "kimi")
})

test_that("KimiProcessor targets the Moonshot Open Platform endpoint by default", {
  processor <- KimiProcessor$new()
  expect_identical(
    processor$get_default_api_url(),
    "https://api.moonshot.cn/v1/chat/completions"
  )
  expect_identical(
    processor$get_api_url(),
    "https://api.moonshot.cn/v1/chat/completions"
  )
})

test_that("KimiProcessor honors and normalizes a custom base URL", {
  processor <- KimiProcessor$new("  https://proxy.example.test/v1/chat/completions/  ")
  expect_identical(
    processor$get_api_url(),
    "https://proxy.example.test/v1/chat/completions"
  )
})

test_that("kimi credentials resolve KIMI_API_KEY before the MOONSHOT_API_KEY alias", {
  variables <- c("KIMI_API_KEY", "MOONSHOT_API_KEY")
  restore_environment <- preserve_kimi_env_variables(variables)
  on.exit(restore_environment(), add = TRUE)
  Sys.unsetenv(variables)

  Sys.setenv(MOONSHOT_API_KEY = "  alias-key  ")
  alias_credential <- .resolve_model_api_key("kimi-k2.6", api_keys = NULL)
  expect_identical(alias_credential$api_key, "alias-key")
  expect_identical(alias_credential$provider, "kimi")
  expect_identical(alias_credential$source, "environment")

  Sys.setenv(KIMI_API_KEY = " primary-key ")
  primary_credential <- .resolve_model_api_key("kimi-k2.6", api_keys = NULL)
  expect_identical(primary_credential$api_key, "primary-key")

  explicit_credential <- .resolve_model_api_key(
    "kimi-k2.6",
    api_keys = list(kimi = "  explicit-key  ")
  )
  expect_identical(explicit_credential$api_key, "explicit-key")
  expect_identical(explicit_credential$source, "api_keys")
})

test_that("KimiProcessor sends Bearer auth, model, messages, and disabled thinking", {
  captured <- new.env(parent = emptyenv())

  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      processor <- KimiProcessor$new()
      response <- processor$make_api_call("genes", "kimi-k2.6", "test-key")

      expect_identical(captured$url, "https://api.moonshot.cn/v1/chat/completions")
      expect_identical(captured$headers[["Authorization"]], "Bearer test-key")
      expect_identical(captured$headers[["Content-Type"]], "application/json")
      expect_identical(captured$body$model, "kimi-k2.6")
      expect_identical(
        captured$body$messages,
        list(list(role = "user", content = "genes"))
      )
      expect_identical(captured$body$temperature, 0.7)
      expect_identical(captured$body$max_tokens, 4096)
      expect_identical(captured$body$thinking, list(type = "disabled"))
      expect_s3_class(response, "response")
    },
    POST = function(url, config = NULL, body = NULL, ...) {
      captured$url <- url
      captured$headers <- config$headers
      captured$body <- body
      make_kimi_json_response(list(
        choices = list(list(message = list(content = "Cluster 1: T cells")))
      ))
    },
    .package = "httr")
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor parses OpenAI-compatible content and normalized usage", {
  payload <- list(
    choices = list(list(message = list(
      content = "Cluster 1: T cells\nCluster 2: B cells"
    ))),
    usage = list(prompt_tokens = 30, completion_tokens = 6, total_tokens = 36)
  )
  response <- make_kimi_json_response(payload)

  testthat::with_mocked_bindings({
    processor <- KimiProcessor$new()
    expect_identical(
      processor$extract_response_content(response, "kimi-k2.6"),
      "Cluster 1: T cells\nCluster 2: B cells"
    )
    expect_identical(
      processor$extract_usage(response),
      list(prompt_tokens = 30, completion_tokens = 6, total_tokens = 36)
    )
    expect_null(processor$extract_usage(make_kimi_json_response(list(choices = list()))))
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor normalizes response lines through process_request", {
  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      processor <- KimiProcessor$new()
      result <- processor$process_request("genes", "kimi-k2.6", "test-key")
      expect_identical(result, c("Cluster 1: T cells", "Cluster 2: B cells"))
    },
    POST = function(...) {
      make_kimi_json_response(list(
        choices = list(list(message = list(
          content = "  Cluster 1: T cells,  \n\nCluster 2: B cells,\n"
        ))),
        usage = list(prompt_tokens = 30, completion_tokens = 6, total_tokens = 36)
      ))
    },
    .package = "httr")
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor surfaces provider error messages for non-200 responses", {
  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      processor <- KimiProcessor$new()
      error <- tryCatch(
        processor$make_api_call("genes", "kimi-k2.6", "test-key"),
        error = identity
      )
      expect_s3_class(error, "mllm_api_error")
      expect_identical(error$status_code, 401L)
      expect_false(error$retryable)
      expect_identical(
        conditionMessage(error),
        "Kimi API request failed: Invalid authentication credentials"
      )
    },
    POST = function(...) {
      make_kimi_json_response(
        list(error = list(message = "Invalid authentication credentials")),
        status_code = 401L
      )
    },
    .package = "httr")
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor rejects malformed response payloads", {
  testthat::with_mocked_bindings({
    processor <- KimiProcessor$new()
    expect_error(
      processor$extract_response_content(
        make_kimi_json_response(list(invalid = "payload")),
        "kimi-k2.6"
      ),
      "Unexpected response format from Kimi API"
    )
    expect_error(
      processor$extract_response_content(
        make_kimi_json_response(list(choices = list(list()))),
        "kimi-k2.6"
      ),
      "Unexpected response format from Kimi API"
    )
  },
  get_logger = quiet_kimi_logger)
})

test_that("get_model_response dispatches kimi models through KimiProcessor", {
  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      result <- get_model_response("genes", "kimi-k2.6", "test-key")
      expect_identical(result, "Cluster 1: T cells")
    },
    POST = function(url, ...) {
      expect_identical(url, "https://api.moonshot.cn/v1/chat/completions")
      make_kimi_json_response(list(
        choices = list(list(message = list(content = "Cluster 1: T cells")))
      ))
    },
    .package = "httr")
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor switches to the Anthropic protocol for Messages endpoints", {
  captured <- new.env(parent = emptyenv())

  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      processor <- KimiProcessor$new("https://api.kimi.com/coding/v1/messages")
      response <- processor$make_api_call("genes", "kimi-for-coding", "test-key")

      expect_identical(captured$url, "https://api.kimi.com/coding/v1/messages")
      expect_identical(captured$headers[["x-api-key"]], "test-key")
      expect_identical(captured$headers[["anthropic-version"]], "2023-06-01")
      expect_false("Authorization" %in% names(captured$headers))
      expect_identical(captured$body$model, "kimi-for-coding")
      expect_identical(
        captured$body$messages,
        list(list(role = "user", content = "genes"))
      )
      expect_identical(captured$body$max_tokens, 4096)
      expect_null(captured$body$thinking)
      expect_s3_class(response, "response")
    },
    POST = function(url, config = NULL, body = NULL, ...) {
      captured$url <- url
      captured$headers <- config$headers
      captured$body <- body
      make_kimi_json_response(list(content = list(list(text = "Cluster 1: T cells"))))
    },
    .package = "httr")
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor completes Kimi Code base URLs to the OpenAI protocol by default", {
  captured <- new.env(parent = emptyenv())

  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      coding_processor <- KimiProcessor$new("https://api.kimi.com/coding/")
      coding_processor$make_api_call("genes", "kimi-for-coding", "test-key")
      expect_identical(captured$url, "https://api.kimi.com/coding/v1/chat/completions")
      expect_identical(captured$headers[["Authorization"]], "Bearer test-key")
      expect_identical(captured$body$thinking, list(type = "disabled"))

      openai_processor <- KimiProcessor$new("https://api.kimi.com/coding/v1")
      openai_processor$make_api_call("genes", "kimi-for-coding", "test-key")
      expect_identical(captured$url, "https://api.kimi.com/coding/v1/chat/completions")
      expect_identical(captured$headers[["Authorization"]], "Bearer test-key")
      expect_identical(captured$body$thinking, list(type = "disabled"))
    },
    POST = function(url, config = NULL, body = NULL, ...) {
      captured$url <- url
      captured$headers <- config$headers
      captured$body <- body
      make_kimi_json_response(list(
        choices = list(list(message = list(content = "Cluster 1: T cells")))
      ))
    },
    .package = "httr")
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor parses Anthropic-mode content and derives total usage", {
  processor <- KimiProcessor$new("https://api.kimi.com/coding/v1/messages")
  response <- make_kimi_json_response(list(
    content = list(list(text = "Cluster 1: T cells\nCluster 2: B cells")),
    usage = list(input_tokens = 20, output_tokens = 4)
  ))

  testthat::with_mocked_bindings({
    expect_identical(
      processor$extract_response_content(response, "kimi-for-coding"),
      "Cluster 1: T cells\nCluster 2: B cells"
    )
    expect_identical(
      processor$extract_usage(response),
      list(prompt_tokens = 20, completion_tokens = 4, total_tokens = 24)
    )
    expect_error(
      processor$extract_response_content(
        make_kimi_json_response(list(invalid = "payload")),
        "kimi-for-coding"
      ),
      "Unexpected response format from Kimi Messages API"
    )
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor parses multi-block Anthropic-mode responses", {
  processor <- KimiProcessor$new("https://api.kimi.com/coding/v1/messages")

  thinking_then_text <- make_kimi_json_response(list(
    content = list(
      list(type = "thinking", thinking = "reasoning..."),
      list(type = "text", text = "Cluster 1: T cells")
    )
  ))

  multiple_text <- make_kimi_json_response(list(
    content = list(
      list(type = "text", text = "Cluster 1: T cells"),
      list(type = "text", text = "Cluster 2: B cells")
    )
  ))

  only_thinking <- make_kimi_json_response(list(
    content = list(list(type = "thinking", thinking = "reasoning..."))
  ))

  testthat::with_mocked_bindings({
    expect_identical(
      processor$extract_response_content(thinking_then_text, "kimi-for-coding"),
      "Cluster 1: T cells"
    )
    expect_identical(
      processor$extract_response_content(multiple_text, "kimi-for-coding"),
      "Cluster 1: T cells\nCluster 2: B cells"
    )
    expect_error(
      processor$extract_response_content(only_thinking, "kimi-for-coding"),
      "Unexpected response format from Kimi Messages API"
    )
  },
  get_logger = quiet_kimi_logger)
})

test_that("KimiProcessor surfaces Anthropic-mode HTTP errors", {
  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      processor <- KimiProcessor$new("https://api.kimi.com/coding/v1/messages")
      error <- tryCatch(
        processor$make_api_call("genes", "kimi-for-coding", "test-key"),
        error = identity
      )
      expect_s3_class(error, "mllm_api_error")
      expect_identical(error$status_code, 401L)
      expect_false(error$retryable)
      expect_identical(
        conditionMessage(error),
        "Kimi API request failed: invalid api key"
      )
    },
    POST = function(...) {
      make_kimi_json_response(
        list(error = list(message = "invalid api key")),
        status_code = 401L
      )
    },
    .package = "httr")
  },
  get_logger = quiet_kimi_logger)
})
