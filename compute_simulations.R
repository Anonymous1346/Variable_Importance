DEBUG <- TRUE
N_SIMULATIONS <- `if`(!DEBUG, 1L:100L, 1L)
N_CPU <- ifelse(!DEBUG, 50L, 1L)

suppressMessages({
  require(data.table)
  if (!DEBUG) {
    require(snowfall)
    sfInit(parallel = TRUE, cpus = N_CPU, type = "SOCK")
    sfLibrary(snowfall)
    sfLibrary(doParallel)
    sfLibrary(grf)
    sfLibrary(party)
    sfLibrary(permimp)
    sfLibrary(ranger)
    sfLibrary(randomForest)
    sfLibrary(reticulate)
    sfLibrary(vita)
    sfLibrary(gtools)
    sfLibrary(MASS)
    sfLibrary(simstudy)
    sfSource("data/data_gen.R")
    sfSource("utils/compute_methods.R")
  } else {
    library(doParallel)
    library(grf)
    library("party", quietly = TRUE)
    library(permimp)
    library(ranger)
    library(randomForest)
    library(reticulate)
    library(vita)
    library(gtools)
    library(MASS)
    library(simstudy)
    source("data/data_gen.R")
    source("utils/compute_methods.R")
  }
})

my_apply <- lapply
if (!DEBUG) {
  my_apply <- sfLapply
}

##### Running Methods #####

methods <- c(
  # "marginal",
  "knockoff",
  # "shap",
  # "mdi",
  # "d0crt",
  # "bart",
  # "dnn_py",
  # "dnn_py_cond",
  # "rf_cond",
  "strobl"
)
list_models <- paste0("Best_model_1_", N_SIMULATIONS)

##### Configuration #####

param_grid <- expand.grid(
  # File, if given, for the real data
  # file = "data/ukbb_data",
  file = "",
  # The number of samples
  n_samples = ifelse(!DEBUG, 1000L, 250L), #8300
  # n_samples = `if`(!DEBUG, seq(100, 1000, by = 100), 10L),
  # The number of covariates
  n_features = ifelse(!DEBUG, 50L, 5L), #2300
  # Whether to use or not discrete variables
  discrete_bool = c(
    #  TRUE
    FALSE
  ),
  # The number of relevant covariates
  n_signal = ifelse(!DEBUG, 20L, 2L), #115
  # The mean for the simulation
  mean = c(0),
  # The correlation coefficient
  rho = c(
    # 0,
    # 0.2,
    # 0.5,
    0.8
  ),
  # The covariance matrix, if given
  sigma = "",
  # Number of blocks
  n_blocks = ifelse(!DEBUG, 10L, 1L),
  # Type of simulation
  # It can be ["blocks_toeplitz", "blocks_fixed",
  # "simple_toeplitz", "simple_fixed"]
  type_sim = c("blocks_fixed"),
  # Signal-to-Noise ratio
  snr = c(4),
  # The task (computation of the response vector)
  prob_sim_data = c(
    # "classification",
    "regression"
    # "regression_combine",
    # "regression_product",
    # "regression_relu"
    # "regression_perm"
  ),
  # The running methods implemented
  method = methods,
  # The d0crt method'statistic tests scaled or not
  scaled_statistics = c(
    # TRUE,
    FALSE
  ),
  # Refit parameter for the d0crt method
  refit = FALSE,
  # The holdout importance's implementation (ranger or original)
  with_ranger = FALSE,
  # The holdout importance measure to use (impurity corrected vs MDA)
  with_impurity = FALSE,
  # The holdout importance in python
  with_python = FALSE,
  # The statistic to use with the knockoff method
  stat_knockoff = c(
    "lasso_cv",
    "bart",
    "deep"
  ),
  # Type of forest for grf package
  type_forest = c(
    "regression",
    "quantile"
  ),
  # Depth for the Random Forest (Conditional Sampling)
  # depth = c(1:10)
  depth = c(2L)
)

param_grid <- param_grid[
  ((!param_grid$scaled_statistics) & # if scaled stats
     (param_grid$stat_knockoff %in% c("bart")) & # and defaults
     (!param_grid$refit) & # and refit
     (param_grid$type_forest == "regression") & # and type_forest
     (!param_grid$method %in% c(
       "d0crt", # but not ...
       "knockoff",
       "grf"
     ))) |
    ((!param_grid$scaled_statistics) & # or scaled
       (!param_grid$refit) &
       (param_grid$type_forest == "regression") &
       (param_grid$method == "knockoff")) |
    ((param_grid$stat_knockoff %in% c("bart")) &
       (param_grid$type_forest == "regression") &
       (param_grid$method == "d0crt")) |
    ((!param_grid$scaled_statistics) &
       (!param_grid$refit) &
       (param_grid$stat_knockoff %in% c("bart")) &
       (param_grid$method == "grf")),
]

param_grid$index_i <- 1:nrow(param_grid)
cat(sprintf("Number of rows: %i \n", nrow(param_grid)))

if (!DEBUG) {
  # Models names for saving DNNs
  sfExport("list_models")
  sfExport("param_grid")
}

compute_method <- function(method,
                           index_i,
                           n_simulations, ...) {
  print("Begin")
  cat(sprintf("%s: %i \n", method, index_i))
  
  compute_fun <- function(seed, ...) {
    # sfCat(paste("Iteration: ", seed), sep="\n")
    sim_data <- generate_data(
      seed,
      ...
    )
    timing <- system.time(
      out <- switch(as.character(method),
                    marginal = compute_marginal(
                      sim_data,
                      ...
                    ),
                    knockoff = compute_knockoff(sim_data,
                                                seed,
                                                list_models[[seed]],
                                                verbose = TRUE,
                                                ...
                    ),
                    bart = compute_bart(sim_data,
                                        ntree = 100L,
                                        ...
                    ),
                    mdi = compute_mdi(sim_data,
                                      ntree = 500L,
                                      ...
                    ),
                    shap = compute_shap(sim_data,
                                        ntree = 100L,
                                        ...
                    ),
                    strobl = compute_strobl(sim_data,
                                            ntree = 100L,
                                            conditional = TRUE,
                                            ...
                    ),
                    d0crt = compute_d0crt(sim_data,
                                          seed,
                                          loss = "least_square",
                                          statistic = "randomforest",
                                          ntree = 100L,
                                          verbose = TRUE,
                                          ...
                    ),
                    dnn_py = compute_dnn_py(
                      sim_data,
                      seed,
                      ...
                    ),
                    dnn_py_cond = compute_dnn_py_cond(
                      sim_data,
                      seed,
                      ...
                    ),
                    rf_cond = compute_rf_cond(
                      sim_data,
                      seed,
                      ...
                    )
      )
    )
    out <- data.frame(out)
    out$elapsed <- timing[[3]]
    out$correlation <- list(...)$rho
    out$n_samples <- list(...)$n
    out$prob_data <- list(...)$prob_sim_data
    
    # sfCat(paste("Done Iteration: ", seed), sep="\n")
    return(out)
  }
  sim_range <- n_simulations
  # compute results
  result <- my_apply(sim_range, compute_fun, ...)
  # postprocess and package outputs
  result <- do.call(rbind, lapply(sim_range, function(ii) {
    out <- result[[ii]]
    out$iteration <- ii
    out
  }))
  # print(result)
  # stop()
  res <- data.table(result)[,
                            mean(elapsed),
                            by = .(
                              n_samples,
                              correlation,
                              method,
                              iteration,
                              prob_data
                            )
  ]
  
  res <- res[,
             sum(V1) / (N_CPU * 60),
             by = .(
               n_samples,
               method,
               correlation,
               prob_data
             )
  ]
  
  print(res)
  print("Finish")
  return(result)
}


# if (DEBUG) {
#   set.seed(42)
#   param_grid <- param_grid[sample(1:nrow(param_grid), 5), ]
# }

results <-
  by(
    param_grid, 1:nrow(param_grid),
    function(x) {
      with(
        x,
        compute_method(
          file = file,
          n = n_samples,
          p = n_features,
          discrete_bool = discrete_bool,
          n_signal = n_signal,
          mean = mean,
          rho = rho,
          sigma = sigma,
          n_blocks = n_blocks,
          type_sim = type_sim,
          snr = snr,
          method = method,
          index_i = index_i,
          n_simulations = N_SIMULATIONS,
          stat_knockoff = stat_knockoff,
          with_ranger = with_ranger,
          with_impurity = with_impurity,
          with_python = with_python,
          refit = refit,
          scaled_statistics = scaled_statistics,
          type_forest = type_forest,
          prob_sim_data = prob_sim_data,
          prob_type = strsplit(as.character(prob_sim_data), "_")[[1]][1],
          depth = depth
        )
      )
    }
  )

results <- rbindlist(results, fill=TRUE)

out_fname <- "test.csv"

if (DEBUG) {
  out_fname <- gsub(".csv", "-debug.csv", out_fname)
}

fwrite(results, out_fname)

if (!DEBUG) {
  sfStop()
}