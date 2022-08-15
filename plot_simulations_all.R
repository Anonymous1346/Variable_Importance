suppressMessages({
  source("utils/plot_methods_all_increasing_combine.R")
  library("tools")
  library("data.table")
  library("ggpubr")
  library("scales")
})

file_path <- paste0("results/")
filename <- "simulation_results_blocks_100_dnn_dnn_py_perm_100--1000"

list_func <- c(
  "Marg",
  "Knockoff_bart",
  "Knockoff_lasso",
  "Shap",
  "MDI",
  "d0CRT",
  "BART",
  "Knockoff_deep",
  "Permfit-DNN",
  "CRF-DNN",
  "CRF-RF",
  "Strobl"
)

run_plot_auc <- FALSE
run_plot_type1error <- FALSE
run_plot_power <- FALSE
run_time <- FALSE
run_plot_pred <- FALSE
run_plot_fdr <- FALSE
run_plot_combine <- TRUE


if (run_plot_auc) {
  plot_method(paste0(filename, ".csv"),
              "AUC_blocks_100_allMethods_pred_imp_withoutPval",
              compute_auc,
              nb_relevant = 20,
              cor_coef = 0.8,
              title = "AUC",
              list_func = list_func,
              mediane_bool = TRUE
  )
}


if (run_plot_type1error) {
  plot_method(paste0(filename, ".csv"),
              "type1error_blocks_100_allMethods_pred_imp",
              compute_pval,
              nb_relevant = 20,
              upper_bound = 0.05,
              cor_coef = 0.8,
              title = "Type I Error",
              list_func = list_func
  )
}


if (run_plot_power) {
  plot_method(paste0(filename, ".csv"),
              "power_blocks_100_allMethods_pred_imp",
              compute_power,
              nb_relevant = 20,
              upper_bound = 0.05,
              cor_coef = 0.8,
              title = "Power",
              list_func = list_func
  )
}


if (run_time) {
  plot_time(paste0(filename, ".csv"),
            "time_bars_blocks_100_allMethods_pred_imp",
            list_func = list_func,
            N_CPU = 50
  )
}


if (run_plot_pred) {
  plot_method(paste0(filename, ".csv"),
              "pred_blocks_100_allMethods_pred_imp",
              compute_pred,
              nb_relevant = 20,
              upper_bound = 0.2,
              cor_coef = 0.8,
              title = "Prediction scores",
              list_func = list_func
  )
}


if (run_plot_combine) {
  plot_method(paste0(filename, ".csv"),
              "combine_blocks_100_dnn_dnn_py_perm_100--1000_test",
              nb_relevant = 20,
              cor_coef = 0.8,
              title = "AUC",
              list_func = list_func,
              mediane_bool = TRUE
  )
}


if (run_plot_fdr) {
  plot_method(paste0(filename, ".csv"),
              "fdr_blocks_10_knockoffDeep_single_orig_imp_n",
              compute_fdr,
              nb_relevant = 20,
              upper_bound = 0.2,
              cor_coef = 0.8,
              title = "FDR Control",
              list_func = list_func
  )
}
