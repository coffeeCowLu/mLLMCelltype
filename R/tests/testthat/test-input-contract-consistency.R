# Input contract consistency tests

test_that("normalize_cluster_gene_list canonicalizes list inputs", {
  unnamed <- normalize_cluster_gene_list(list(c("G1"), c("G2")))
  expect_identical(names(unnamed), c("0", "1"))

  numeric_named <- normalize_cluster_gene_list(
    list("1" = c("A"), "2" = list(genes = c("B")))
  )
  expect_identical(names(numeric_named), c("1", "2"))

  non_numeric_named <- normalize_cluster_gene_list(
    list("t_cells" = c("CD3D"), "b_cells" = list(genes = c("MS4A1")))
  )
  expect_identical(names(non_numeric_named), c("t_cells", "b_cells"))
})

test_that("create_annotation_prompt accepts mixed list element formats", {
  prompt_result <- create_annotation_prompt(
    input = list(
      gs1 = c("CD4", "CD3D"),
      gs2 = list(genes = c("CD14"))
    ),
    tissue_name = "PBMC"
  )

  expect_true(all(c("gs1", "gs2") %in% names(prompt_result$gene_lists)))
  expect_match(prompt_result$prompt, "gs1: CD4, CD3D")
  expect_match(prompt_result$prompt, "gs2: CD14")
})

test_that("CacheManager generate_key supports character-vector list clusters", {
  cache_manager <- CacheManager$new(cache_dir = "temp")
  input_list <- list(
    "0" = c("CD3D", "CD3E"),
    "1" = list(genes = c("MS4A1"))
  )

  key_0 <- cache_manager$generate_key(input_list, c("gpt-5.2"), "0")
  key_1 <- cache_manager$generate_key(input_list, c("gpt-5.2"), "1")

  expect_true(is.character(key_0))
  expect_gt(nchar(key_0), 10)
  expect_false(identical(key_0, key_1))
})

test_that("compare_model_predictions pads unequal lengths without recycling", {
  res <- testthat::with_mocked_bindings({
    capture.output({
      result <- suppressMessages(compare_model_predictions(
        input = list("0" = list(genes = c("G1"))),
        tissue_name = "PBMC",
        models = c("m1", "m2"),
        api_keys = list(openai = "k")
      ))
    })
    result
  },
  get_api_key = function(model, api_keys) "k",
  get_provider = function(model) "openai",
  annotate_cell_types = function(input, tissue_name, model, api_key, top_gene_count = 10, ...) {
    if (identical(model, "m1")) c("A", "B", "C") else c("A", "B")
  },
  standardize_cell_type_names = function(predictions, models, api_keys, standardization_model = "x", base_urls = NULL) {
    predictions
  })

  expect_identical(dim(res$comparison_matrix), c(3L, 2L))
  expect_true(is.na(res$comparison_matrix[3, "m2"]))
  expect_true(is.na(res$consensus_predictions[3]))
  expect_true(is.na(res$consensus_proportions[3]))
  expect_true(is.na(res$entropies[3]))
})

test_that("interactive_consensus_annotation supports mapped cluster selectors", {
  mocked_call <- function(selector) {
    testthat::with_mocked_bindings({
      suppressMessages(interactive_consensus_annotation(
        input = list(
          "1" = c("CD3D"),
          "2" = list(genes = c("MS4A1"))
        ),
        tissue_name = "PBMC",
        models = c("gpt-5.2", "grok-4.1"),
        api_keys = list(openai = "k", grok = "k"),
        use_cache = FALSE,
        clusters_to_analyze = selector
      ))
    },
    log_info = function(...) NULL,
    print_consensus_summary = function(...) NULL,
    get_initial_predictions = function(input, ...) {
      # Predictions keyed by original cluster names (no 0-based conversion)
      cluster_ids <- names(input)
      preds <- setNames(rep("T", length(cluster_ids)), cluster_ids)
      list(
        individual_predictions = list(m1 = as.list(preds), m2 = as.list(preds)),
        successful_models = c("m1", "m2")
      )
    },
    identify_controversial_clusters = function(input, ...) {
      list(
        consensus_results = list(),
        controversial_clusters = character(0),
        final_annotations = setNames(rep("X", length(input)), names(input))
      )
    },
    process_controversial_clusters = function(...) {
      list(discussion_logs = list(), final_annotations = list())
    },
    combine_results = function(initial_results, controversy_results, discussion_results) {
      list(final_annotations = controversy_results$final_annotations)
    })
  }

  # Cluster names are preserved as-is (no 0-based conversion)
  result_single <- mocked_call("1")
  expect_identical(names(result_single$final_annotations), "1")

  result_both <- mocked_call(c("1", "2"))
  expect_identical(sort(names(result_both$final_annotations)), c("1", "2"))
})

test_that("parse_consensus_response falls back to flexible parsing for short responses", {
  result <- testthat::with_mocked_bindings({
    parse_consensus_response(
      c(
        "Consensus Proportion = 0.75",
        "Entropy = 0.81",
        "B cell"
      )
    )
  },
  get_logger = function() {
    list(
      warn = function(...) NULL,
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_identical(result$reached, FALSE)
  expect_equal(result$consensus_proportion, 0.75)
  expect_equal(result$entropy, 0.81)
  expect_identical(result$majority_prediction, "B cell")
})

test_that("get_initial_predictions skips invalid model names without aborting", {
  result <- testthat::with_mocked_bindings({
    suppressWarnings(get_initial_predictions(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      models = c("bad-model", "gpt-5.2"),
      api_keys = list(openai = "k"),
      top_gene_count = 10
    ))
  },
  get_provider = function(model) {
    if (identical(model, "bad-model")) stop("unknown model")
    "openai"
  },
  get_api_key = function(model, api_keys) {
    provider <- get_provider(model)
    if (provider %in% names(api_keys)) api_keys[[provider]] else NULL
  },
  annotate_cell_types = function(...) c("T cell"),
  log_info = function(...) NULL,
  log_warn = function(...) NULL)

  expect_identical(result$successful_models, "gpt-5.2")
  expect_true("gpt-5.2" %in% names(result$individual_predictions))
})

test_that("interactive_consensus_annotation consumes log_dir via initialize_logger", {
  seen_log_dir <- NULL

  testthat::with_mocked_bindings({
    suppressMessages(interactive_consensus_annotation(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      models = c("gpt-5.2", "grok-4.1"),
      api_keys = list(openai = "k", grok = "k"),
      use_cache = FALSE,
      log_dir = "custom_logs_dir"
    ))
  },
  initialize_logger = function(log_dir = "logs") {
    seen_log_dir <<- log_dir
    invisible(NULL)
  },
  log_info = function(...) NULL,
  print_consensus_summary = function(...) NULL,
  get_initial_predictions = function(input, ...) {
    list(
      individual_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      successful_models = c("m1", "m2")
    )
  },
  identify_controversial_clusters = function(input, ...) {
    list(
      consensus_results = list(),
      controversial_clusters = character(0),
      final_annotations = setNames(rep("X", length(input)), names(input))
    )
  },
  process_controversial_clusters = function(...) {
    list(discussion_logs = list(), final_annotations = list())
  },
  combine_results = function(initial_results, controversy_results, discussion_results) {
    list(final_annotations = controversy_results$final_annotations)
  })

  expect_identical(seen_log_dir, "custom_logs_dir")
})

test_that("combine_results provides stable aliases for return fields", {
  result <- testthat::with_mocked_bindings({
    combine_results(
      initial_results = list(individual_predictions = list(m1 = "A")),
      controversy_results = list(
        consensus_results = list("0" = list(reached = TRUE)),
        controversial_clusters = c("1"),
        final_annotations = list("0" = "T cell")
      ),
      discussion_results = list(
        discussion_logs = list("1" = list(rounds = list())),
        final_annotations = list("1" = "B cell")
      )
    )
  },
  get_logger = function() list(session_id = "session-test"))

  expect_identical(result$voting_results, result$initial_results)
  expect_identical(result$discussion_results, result$discussion_logs)
  expect_identical(result$final_consensus, result$final_annotations)
})

test_that("facilitate_cluster_discussion keeps fallback cluster_genes on extraction error", {
  captured_cluster_genes <- NULL

  expect_warning(
    testthat::with_mocked_bindings({
      facilitate_cluster_discussion(
        cluster_id = "0",
        input = 42,
        tissue_name = "PBMC",
        models = c("m1", "m2"),
        api_keys = list(openai = "k"),
        initial_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
        top_gene_count = 5,
        max_rounds = 2
      )
    },
    create_initial_discussion_prompt = function(cluster_id, cluster_genes, tissue_name, initial_predictions) {
      captured_cluster_genes <<- cluster_genes
      "prompt"
    },
    get_model_response = function(...) "CELL TYPE: T cell",
    check_consensus = function(...) {
      list(reached = TRUE, consensus_proportion = 1, entropy = 0, majority_prediction = "T cell")
    },
    get_api_key = function(model, api_keys) "k",
    get_provider = function(model) "openai",
    get_logger = function() {
      list(log_discussion = function(...) NULL)
    },
    log_warn = function(...) NULL,
    log_info = function(...) NULL),
    regexp = "Error extracting genes"
  )

  expect_true(grepl("Error extracting genes", captured_cluster_genes, fixed = TRUE))
})

test_that("annotate_cell_types supports custom providers", {
  custom_env <- new.env(parent = emptyenv())
  assign("customx", TRUE, envir = custom_env)

  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list("0" = list(genes = c("CD3D", "CD3E"))),
      tissue_name = "PBMC",
      model = "custom-model",
      api_key = "k"
    )
  },
  get_provider = function(model) "customx",
  resolve_provider_base_url = function(provider, base_urls) NULL,
  create_annotation_prompt = function(input, tissue_name, top_gene_count = 10) {
    list(prompt = "prompt", expected_count = 1L, gene_lists = list("0" = "CD3D, CD3E"))
  },
  process_custom = function(prompt, model, api_key) "custom-ok",
  custom_providers = custom_env,
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_identical(result, "custom-ok")
})

test_that("parse_consensus_response handles NA_character_ safely", {
  result <- testthat::with_mocked_bindings({
    parse_consensus_response(NA_character_)
  },
  get_logger = function() {
    list(
      warn = function(...) NULL,
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_identical(result, list(
    reached = FALSE,
    consensus_proportion = 0,
    entropy = 0,
    majority_prediction = "Unknown"
  ))
})

test_that("print_consensus_summary handles vector predictions without crashing", {
  results <- list(
    final_annotations = list("0" = c("Consensus A", "Consensus B")),
    controversial_clusters = c("0"),
    discussion_logs = list(
      "0" = list(initial_predictions = list(m1 = c("Pred A", "Pred B")))
    )
  )

  expect_no_error(capture.output(print_consensus_summary(results)))
})

test_that("BaseAPIProcessor rejects non-scalar prompt cleanly", {
  TP <- R6::R6Class(
    "TP",
    inherit = BaseAPIProcessor,
    public = list(
      get_default_api_url = function() "x",
      make_api_call = function(chunk_content, model, api_key) list(),
      extract_response_content = function(response, model) "ok"
    )
  )

  p <- TP$new("tp")
  expect_error(
    p$process_request(c("a", "b"), "m", "k"),
    "Prompt is required but not provided"
  )
})

test_that("get_provider validates model as non-empty scalar", {
  expect_error(get_provider(character(0)), "model must be a non-empty character scalar")
  expect_error(get_provider(c("gpt-5.2", "grok-4.1")), "model must be a non-empty character scalar")
  expect_error(get_provider(NA_character_), "model must be a non-empty character scalar")
})

test_that("annotate_cell_types validates api_key scalar contract", {
  expect_error(
    annotate_cell_types(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      model = "gpt-5.2",
      api_key = c("a", "b")
    ),
    "api_key must be a non-empty character scalar, or NA to return prompt only"
  )
})

test_that("interactive_consensus_annotation validates api_keys list contract", {
  expect_error(
    interactive_consensus_annotation(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      models = c("gpt-5.2", "grok-4.1"),
      api_keys = "bad",
      use_cache = FALSE
    ),
    "api_keys must be a named, non-empty list"
  )
})

test_that("get_api_key ignores empty/NA/non-scalar keys and falls back correctly", {
  expect_null(get_api_key("gpt-5.2", list(openai = "")))
  expect_null(get_api_key("gpt-5.2", list(openai = "   ")))
  expect_null(get_api_key("gpt-5.2", list(openai = NA_character_)))
  expect_null(get_api_key("gpt-5.2", list(openai = c("a", "b"))))
  expect_identical(get_api_key("gpt-5.2", list(openai = "", "gpt-5.2" = "k2")), "k2")
  expect_identical(get_api_key("gpt-5.2", list(openai = "  k1  ")), "k1")
})

test_that("facilitate_cluster_discussion uses last available consensus on early break", {
  logger_env <- new.env(parent = emptyenv())
  logger_env$events <- list()
  logger <- list(
    log_discussion = function(cluster_id, event_type, data = NULL) {
      logger_env$events <- c(logger_env$events, list(list(event = event_type, data = data)))
    }
  )

  testthat::with_mocked_bindings({
    suppressMessages(facilitate_cluster_discussion(
      cluster_id = "0",
      input = list("0" = list(genes = c("G1"))),
      tissue_name = "PBMC",
      models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      initial_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      top_gene_count = 5,
      max_rounds = 3
    ))
  },
  create_initial_discussion_prompt = function(...) "p1",
  create_discussion_prompt = function(...) "p2",
  get_model_response = local({
    cnt <- 0
    function(...) {
      cnt <<- cnt + 1
      if (cnt <= 2) "CELL TYPE: T cell" else if (cnt == 3) "CELL TYPE: T cell" else "Error: failed"
    }
  }),
  check_consensus = local({
    k <- 0
    function(...) {
      k <<- k + 1
      list(reached = FALSE, consensus_proportion = 0.5, entropy = 1.2, majority_prediction = paste0("M", k))
    }
  }),
  get_api_key = function(...) "k",
  get_provider = function(...) "openai",
  get_logger = function() logger,
  log_warn = function(...) NULL,
  log_info = function(...) NULL)

  end_events <- Filter(function(x) identical(x$event, "end"), logger_env$events)
  expect_equal(length(end_events), 1)
  expect_identical(end_events[[1]]$data$final_result, "M1")
})

test_that("process_controversial_clusters backtracks final prediction when last round lacks consensus", {
  out <- testthat::with_mocked_bindings({
    process_controversial_clusters(
      controversial_clusters = c("0"),
      input = list("0" = list(genes = c("G1"))),
      tissue_name = "PBMC",
      successful_models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      individual_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      top_gene_count = 5,
      controversy_threshold = 0.7,
      entropy_threshold = 1.0,
      max_discussion_rounds = 3,
      cache_manager = list(
        generate_key = function(...) "k",
        save_to_cache = function(...) NULL,
        has_cache = function(...) FALSE
      ),
      use_cache = FALSE
    )
  },
  facilitate_cluster_discussion = function(...) {
    list(
      cluster_id = "0",
      rounds = list(
        list(consensus_result = list(majority_prediction = "T cell")),
        list(responses = list(m1 = "..."))
      )
    )
  },
  log_info = function(...) NULL,
  log_warn = function(...) NULL)

  expect_identical(out$final_annotations[["0"]], "T cell")
})

test_that("facilitate_cluster_discussion logs consensus and end on first-round early return", {
  logger_env <- new.env(parent = emptyenv())
  logger_env$events <- list()
  logger <- list(
    log_discussion = function(cluster_id, event_type, data = NULL) {
      logger_env$events <- c(logger_env$events, list(list(event = event_type, data = data)))
    }
  )

  testthat::with_mocked_bindings({
    facilitate_cluster_discussion(
      cluster_id = "0",
      input = list("0" = list(genes = c("G1"))),
      tissue_name = "PBMC",
      models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      initial_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      top_gene_count = 5,
      max_rounds = 3
    )
  },
  create_initial_discussion_prompt = function(...) "p1",
  create_discussion_prompt = function(...) "p2",
  get_model_response = function(...) "Error: failed",
  check_consensus = function(...) list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = "Unknown"),
  get_api_key = function(...) "k",
  get_provider = function(...) "openai",
  get_logger = function() logger,
  log_warn = function(...) NULL,
  log_info = function(...) NULL)

  event_names <- vapply(logger_env$events, function(x) x$event, character(1))
  expect_true("consensus" %in% event_names)
  expect_true("end" %in% event_names)
})

test_that("BaseAPIProcessor marks semantic error result as failed API call", {
  logger_env <- new.env(parent = emptyenv())
  logger_env$api_call_success <- logical(0)
  logger_env$failed_audit_logs <- 0L

  logger <- list(
    info = function(...) NULL,
    debug = function(...) NULL,
    error = function(...) NULL,
    log_api_call = function(provider, model, duration, success = TRUE, tokens = NULL) {
      logger_env$api_call_success <- c(logger_env$api_call_success, success)
    },
    log_api_request_response = function(...) {
      logger_env$failed_audit_logs <- logger_env$failed_audit_logs + 1L
    }
  )

  testthat::with_mocked_bindings({
    TestProcessor <- R6::R6Class("TestProcessor",
      inherit = BaseAPIProcessor,
      public = list(
        initialize = function() super$initialize("testp", NULL),
        get_default_api_url = function() "https://example.test",
        make_api_call = function(chunk_content, model, api_key) stop("simulated failure"),
        extract_response_content = function(response, model) "unused"
      )
    )

    proc <- TestProcessor$new()
    expect_error(proc$process_request("prompt", "test-model", "test-key"),
                 "simulated failure")
  },
  get_logger = function() logger)

  expect_true(length(logger_env$api_call_success) >= 1)
  expect_identical(tail(logger_env$api_call_success, 1), FALSE)
  # Failed requests should still produce audit logs for traceability
  expect_true(logger_env$failed_audit_logs >= 1L)
})

test_that("initialize_logger preserves logger configuration and regenerates unique session IDs", {
  initialize_logger("logs")
  configure_logger(level = "DEBUG", console_output = FALSE, json_format = FALSE)
  logger_cfg <- get_logger()
  logger_cfg$max_log_size <- 42
  logger_cfg$max_log_files <- 7

  initialize_logger("logs")
  logger_after_reinit <- get_logger()
  expect_identical(logger_after_reinit$log_level, "DEBUG")
  expect_identical(logger_after_reinit$enable_console, FALSE)
  expect_identical(logger_after_reinit$enable_json, FALSE)
  expect_identical(logger_after_reinit$max_log_size, 42)
  expect_identical(logger_after_reinit$max_log_files, 7)

  id1 <- logger_after_reinit$session_id
  initialize_logger("logs")
  id2 <- get_logger()$session_id
  expect_false(identical(id1, id2))
})
