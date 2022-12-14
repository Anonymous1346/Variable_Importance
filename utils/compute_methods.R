alib <- import("alibi.explainers", convert = FALSE)
np <- import("numpy", convert = FALSE)
sklearn <- import("sklearn", convert = FALSE)
shap <- import("shap", convert = FALSE)
sandbox <- import_from_path("sandbox",
    path = "sandbox"
)

compute_janitza <- function(sim_data,
                            cv,
                            ntree = 5000L,
                            ncores = 4L,
                            with_ranger = FALSE,
                            with_impurity = FALSE,
                            with_python = FALSE,
                            replace = FALSE,
                            prob_sim_data = "regression",
                            ...) {
    print("Applying HoldOut Method")
    prob_type <- strsplit(as.character(prob_sim_data), "_")[[1]][1]

    mtry <- ceiling(sqrt(ncol(sim_data[, -1])))

    if (prob_type == "classification") {
          sim_data$y <- as.factor(sim_data$y)
      }

    if (!with_ranger) {
        res <- CVPVI(sim_data[, -1],
            sim_data$y,
            k = cv,
            mtry = mtry,
            ntree = ntree,
            ncores = ncores
        )
        tryCatch(
            {
                return(data.frame(
                    method = "HoldOut_nr",
                    importance = as.numeric(res$cv_varim),
                    p_value = as.numeric(NTA(res$cv_varim)$pvalue)
                ))
            },
            finally = {
                return(data.frame(
                    method = "HoldOut_nr",
                    importance = as.numeric(res$cv_varim)
                ))
            }
        )
    }

    else {
        if (!with_impurity) {
            rf_sim <- holdoutRF(y ~ .,
                data = sim_data,
                mtry = mtry,
                num.trees = ntree
            )
            suffix <- "r"
        }
        else {
            rf_sim <- ranger(y ~ .,
                data = sim_data,
                importance = "impurity_corrected",
                mtry = mtry,
                replace = replace,
                num.trees = ntree
            )
            suffix <- "ri"
        }

        res <- importance_pvalues(rf_sim, method = "janitza")
        return(data.frame(
            method = paste0("HoldOut_", suffix),
            importance = as.numeric(res[, 1]),
            p_value = as.numeric(res[, 2])
        ))
    }
}


compute_altmann <- function(sim_data,
                            nper = 100L,
                            ntree = 500L,
                            replace = FALSE,
                            prob_sim_data = "regression",
                            ...) {
    print("Applying Altmann Method")
    prob_type <- strsplit(as.character(prob_sim_data), "_")[[1]][1]

    if (prob_type == "classification") {
          sim_data$y <- as.factor(sim_data$y)
      }

    rf_altmann <- ranger(y ~ .,
        data = sim_data,
        importance = "permutation",
        mtry = ceiling(sqrt(ncol(sim_data[, -1]))),
        num.trees = ntree,
        replace = replace
    )

    res <- data.frame(importance_pvalues(rf_altmann,
        method = "altmann",
        num.permutations = nper,
        formula = y ~ .,
        data = sim_data
    ))
    return(data.frame(
        method = "Altmann",
        importance = res[, 1],
        p_value = res[, 2]
    ))
}


compute_d0crt <- function(sim_data,
                          seed,
                          loss = "least_square",
                          statistic = "residual",
                          ntree = 100L,
                          prob_type = "regression",
                          verbose = FALSE,
                          scaled_statistics = FALSE,
                          refit = FALSE,
                          ...) {
    print("Applying d0CRT Method")

    d0crt_results <- sandbox$dcrt_zero(
        sim_data[, -1],
        as.numeric(sim_data$y),
        loss = loss,
        screening = FALSE,
        statistic = statistic,
        ntree = ntree,
        type_prob = prob_type,
        refit = refit,
        scaled_statistics = scaled_statistics,
        verbose = TRUE,
        random_state = seed
    )

    return(data.frame(
        method = ifelse(scaled_statistics,
            "d0CRT_scaled",
            "d0CRT"
        ),
        importance = d0crt_results[[3]],
        p_value = d0crt_results[[2]],
        score = d0crt_results[[4]]
    ))
}


compute_strobl <- function(sim_data,
                           ntree = 100L,
                           mtry = 5L,
                           conditional = TRUE,
                           prob_type = "regression",
                           ...) {
    print("Applying Strobl Method")
    train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)

    sim_data$y <- as.numeric(sim_data$y)

    f1 <- cforest(y ~ .,
                  data = sim_data[train_ind, ],
                  control = cforest_unbiased(ntree = ntree,
                                             mtry = mtry
                                            )
                 )

    result <- permimp(f1,
                      conditional = conditional,
                      nperm = 100L,
                      progressBar = FALSE
                    )

    if (prob_type == "classification") {
        pred <- predict(f1, newdata = sim_data[-train_ind, -1], type = "response")
        score <- sklearn$metrics$roc_auc_score(sim_data$y[-train_ind], pred)
    }
    else {
        pred <- predict(f1, newdata = sim_data[-train_ind, -1])
        score <- sklearn$metrics$r2_score(sim_data$y[-train_ind], pred)
      }

    return(data.frame(
        method = "Strobl",
        importance = as.numeric(result$values),
        p_value = ifelse(is.nan(result$p_val), 1.0, result$p_val),
        score = py_to_r(score)
    ))
}


compute_shap <- function(sim_data,
                         seed = 2021L,
                         ntree = 100L,
                         prob_type = "regression",
                         ...) {
    print("Applying SHAP Method")

    if (prob_type == "classification") {
          clf_rf <- sklearn$ensemble$
              RandomForestClassifier(n_estimators = ntree)
      }

    if (prob_type == "regression") {
          clf_rf <- sklearn$ensemble$
              RandomForestRegressor(n_estimators = ntree)
      }

    # Splitting train/test sets
    train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
    clf_rf$fit(sim_data[train_ind, -1], sim_data$y[train_ind])
    explainer <- shap$TreeExplainer(clf_rf)

    if (prob_type == "classification") {
          shap_values <- as.matrix(explainer$shap_values(sim_data[-train_ind, -1])[[1]])
      }
    if (prob_type == "regression") {
          shap_values <- as.matrix(explainer$shap_values(sim_data[-train_ind, -1]))
      }

    return(data.frame(
        method = "Shap",
        importance = colMeans(shap_values),
        p_value = NA,
        score = NA
    ))
}


compute_mdi <- function(sim_data,
                        ntree = 100L,
                        prob_type = "regression",
                        ...) {
    print("Applying MDI Method")

    # Splitting train/test sets
    train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)

    if (prob_type == "classification") {
        sim_data$y <- as.factor(sim_data$y)
        clf_rf <- sklearn$ensemble$
            RandomForestClassifier(n_estimators = ntree)
        clf_rf$fit(sim_data[train_ind, -1], sim_data$y[train_ind])
        pred <- py_to_r(clf_rf$predict_proba(sim_data[-train_ind, -1]))[, 2]
        score <- sklearn$metrics$roc_auc_score(sim_data$y[-train_ind], pred)
    }

    if (prob_type == "regression") {
        sim_data$y <- as.numeric(sim_data$y)
        clf_rf <- sklearn$ensemble$
            RandomForestRegressor(n_estimators = ntree)
        clf_rf$fit(sim_data[train_ind, -1], sim_data$y[train_ind])
        pred <- clf_rf$predict(sim_data[-train_ind, -1])
        score <- sklearn$metrics$r2_score(sim_data$y[-train_ind], pred)
    }

    # Compute p-values with permutation approach
    res = sklearn$inspection$permutation_importance(clf_rf, sim_data[-train_ind, -1],
                                                    sim_data$y[-train_ind], n_repeats=100L)
    imp_mean = py_to_r(res$importances_mean)
    imp_std = py_to_r(res$importances_std)
    z_test = imp_mean / imp_std
    p_val = 1 - stats::pnorm(z_test)

    return(data.frame(
        method = "MDI",
        importance = as.numeric(clf_rf$feature_importances_),
        p_value = ifelse(is.nan(p_val), 1.0, p_val),
        score = py_to_r(score))
    )
}


compute_marginal <- function(sim_data,
                             prob_type = "regression",
                             ...) {
    print("Applying Marginal Method")

    # Splitting train/test sets
    train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
    marginal_imp <- numeric()
    marginal_pval <- numeric()
    score_val <- 0

    if (prob_type == "classification") {
        sim_data$y <- as.factor(sim_data$y)
        for (i in 1:ncol(sim_data[, -1])) {
            fit <- glm(formula(paste0("y ~ x", i)),
                data = sim_data[train_ind, ],
                family = binomial()
            )
            sum_fit <- summary(fit)
            marginal_imp[i] <- coef(sum_fit)[, 1][[2]]
            marginal_pval[i] <- coef(sum_fit)[, 4][[2]]
            pred <- predict(fit, newdata = sim_data[-train_ind, -1], type="response")
            score_val <- score_val +
                py_to_r(sklearn$metrics$roc_auc_score(sim_data$y[-train_ind], pred))
          }
      }

    if (prob_type == "regression") {
        sim_data$y <- as.numeric(sim_data$y)
        for (i in 1:ncol(sim_data[, -1])) {
            fit <- glm(formula(paste0("y ~ x", i)),
                data = sim_data[train_ind, ]
            )
            sum_fit <- summary(fit)
            marginal_imp[i] <- coef(sum_fit)[, 1][[2]]
            marginal_pval[i] <- coef(sum_fit)[, 4][[2]]
            pred <- predict(fit, newdata = sim_data[-train_ind, -1])
            score_val <- score_val + py_to_r(sklearn$metrics$r2_score(sim_data$y[-train_ind], pred))
        }
      }

    return(data.frame(
        method = "Marg",
        importance = marginal_imp,
        p_value = marginal_pval,
        score = score_val / ncol(sim_data[, -1])
    ))
}


compute_bart <- function(sim_data,
                         ntree = 100L,
                         num_cores = 4,
                         prob_type = "regression",
                         ...) {
    print("Applying BART Method")

    if (prob_type == "classification") {
        sim_data$y <- as.factor(sim_data$y)
        score_fn <- sklearn$metrics$roc_auc_score
    }
    if (prob_type == "regression") {
        sim_data$y <- as.numeric(sim_data$y)
        score_fn <- sklearn$metrics$r2_score
    }
    options(java.parameters = "-Xmx10000m")
    library(bartMachine)
    
    # Splitting train/test sets
    train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)

    bart_machine <- bartMachine(
        X = sim_data[train_ind, -1],
        y = sim_data$y[train_ind],
        num_trees = ntree,
        mem_cache_for_speed = FALSE,
        verbose = FALSE
    )

    imp <- investigate_var_importance(bart_machine,
        plot = FALSE    
    )$avg_var_props
    imp <- imp[mixedsort(names(imp))]

#     p_val <- c()
#     for(i in colnames(sim_data[, -1]))
#         p_val <- c(p_val, cov_importance_test(bart_machine,
#                                               covariates = i,
#                                               plot = FALSE)$pval)

    pred = bart_predict_for_test_data(bart_machine,
                                      sim_data[-train_ind, -1],
                                      sim_data$y[-train_ind])

    if (prob_type == "classification") {
        pred = 1 - pred$p_hat
    }
    if (prob_type == "regression") {
        pred = pred$y_hat
    }
    return(data.frame(
        method = "BART",
        importance = as.numeric(imp),
        p_value = NA,
        score = py_to_r(score_fn(sim_data$y[-train_ind], pred))
    ))
}


compute_knockoff <- function(sim_data,
                             seed,
                             save_file,
                             stat_knockoff = NULL,
                             with_bart = TRUE,
                             verbose = TRUE,
                             prob_type = "regression",
                             ...) {
    print("Applying Knockoff Method")

    sim_data$y <- as.numeric(sim_data$y)
    if (prob_type == "classification") {
        if (stat_knockoff == "lasso_cv") {
            stat_knockoff = "logistic_l1"
        }
        else
            sim_data$y <- as.factor(sim_data$y)
    }

    res <- sandbox$model_x_knockoff(sim_data[, -1],
        sim_data$y,
        statistics = stat_knockoff,
        verbose = verbose,
        save_file = save_file,
        prob_type = prob_type,
        seed = seed
    )

    if (stat_knockoff == "l1_regu_path") {
        return(data.frame(
            method = "Knockoff_path",
            importance = res[[2]][1:as.integer(length(res[[2]]) / 2)],
            p_value = NA,
            score = NA
        ))
    } else if (stat_knockoff == "bart") {
        res_imp <- compute_bart(data.frame(y = sim_data$y,
                                           res[[1]]),
                                prob_type = prob_type)
        test_score <- res_imp$importance[1:ncol(sim_data[, -1])]
        -res_imp$importance[ncol(sim_data[, -1]):(2 * ncol(sim_data[, -1]))]

        return(data.frame(
            method = "Knockoff_bart",
            importance = test_score,
            p_value = NA,
            score = res_imp$score[1]
        ))
    } else if (stat_knockoff == "deep") {
        return(data.frame(
            method = "Knockoff_deep",
            importance = res[[1]],
            p_value = NA,
            score = res[[2]]
        ))
    }
    return(data.frame(
        method = "Knockoff_lasso",
        importance = res[[2]],
        p_value = NA,
        score = res[[3]]
    ))
}


compute_ale <- function(sim_data,
                        ntree = 100L,
                        prob_type = "regression",
                        ...) {
    print("Applying ALE Method")

    if (prob_type == "classification") {
        clf_rf <- sklearn$ensemble$
            RandomForestClassifier(n_estimators = ntree)
        clf_rf$fit(sim_data[, -1], sim_data$y)
        rf_ale <- alib$ALE(clf_rf$predict_proba)
    }
    if (prob_type == "regression") {
        clf_rf <- sklearn$ensemble$
            RandomForestRegressor(n_estimators = ntree)
        clf_rf$fit(sim_data[, -1], sim_data$y)
        rf_ale <- alib$ALE(clf_rf$predict)
    }
    rf_explain <- rf_ale$explain(as.matrix(sim_data[, -1]))
    imp <- c()
    for (i in 1:dim(sim_data[, -1])[[2]]) {
          imp <- c(
              imp,
              mean(as.vector(rf_explain$ale_values[[i - 1]]))
          )
      }
    return(data.frame(
        method = "Ale",
        importance = imp
    ))
}


compute_dnn <- function(sim_data,
                        n = 1000L,
                        ...) {
    print("Applying DNN Method")
    set.seed(NULL)
    ## 1.0 Hyper-parameters
    esCtrl <- list(
        n.hidden = c(50L, 40L, 30L, 20L),
        activate = "relu",
        l1.reg = 10**-4,
        early.stop.det = 1000L,
        n.batch = 50L,
        n.epoch = 200L,
        learning.rate.adaptive = "adam",
        plot = FALSE
    )
    n_ensemble <- 10L
    n_perm <- 100L
    dnn_obj <- importDnnet(
        x = sim_data[, -1],
        y = as.numeric(sim_data$y)
    )

    # PermFIT-DNN
    shuffle <- sample(n)

    dat_spl <- splitDnnet(dnn_obj, 0.8)
    permfit_dnn <- permfit(
        train = dat_spl$train,
        validate = dat_spl$valid,
        k_fold = 0,
        pathway_list = list(),
        n_perm = n_perm,
        method = "ensemble_dnnet",
        shuffle = shuffle,
        n.ensemble = n_ensemble,
        esCtrl = esCtrl
    )

    return(data.frame(
        method = "Permfit-DNN_old",
        importance = permfit_dnn@importance$importance,
        p_value = permfit_dnn@importance$importance_pval
    ))
}


compute_grf <- function(sim_data,
                        type_forest = "regression",
                        ...) {
    print("Applying GRF Method")

    if (type_forest == "regression") {
          forest <- regression_forest(sim_data[, -1],
              as.numeric(sim_data$y),
              tune.parameters = "all"
          )
      }
    if (type_forest == "quantile") {
          forest <- quantile_forest(sim_data[, -1],
              as.numeric(sim_data$y),
              quantiles = c(0.1, 0.3, 0.5, 0.7, 0.9)
          )
      }
    return(data.frame(
        method = paste0("GRF_", type_forest),
        importance = variable_importance(forest)[, 1]
    ))
}


compute_bart_py <- function(sim_data,
                            ntree = 100L,
                            ...) {
    print("Applying BART Python Method")
    bartpy <- import_from_path("utils_py",
        path = "utils"
    )
    imp <- bartpy$compute_bart_py(
        sim_data[, -1],
        np$array(as.numeric(sim_data$y))
    )

    return(data.frame(
        method = "Bart_py",
        importance = as.numeric(imp)
    ))
}


compute_dnn_py <- function(sim_data,
                           index_i,
                           n = 1000L,
                           prob_type = "regression",
                           ...) {
    print("Applying DNN Permfit Method")

    deep_py <- import_from_path("permfit_py",
        path = "permfit_python"
    )
    results <- deep_py$permfit(
        X_train = sim_data[, -1],
        y_train = as.numeric(sim_data$y),
        prob_type = prob_type,
        conditional = FALSE,
        index_i = index_i
    )

    return(data.frame(
        method = "Permfit-DNN",
        importance = results$importance,
        p_value = results$pval,
        score = results$score
    ))
}


compute_dnn_py_cond <- function(sim_data,
                                index_i,
                                n = 1000L,
                                prob_type = "regression",
                                depth = 2,
                                ...) {
    print("Applying DNN Conditional Method")

    deep_py <- import_from_path("permfit_py",
        path = "permfit_python"
    )
    results <- deep_py$permfit(
        X_train = sim_data[, -1],
        y_train = as.numeric(sim_data$y),
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        max_depth = depth
    )

    return(data.frame(
        method = "CRF-DNN",
        importance = results$importance,
        p_value = results$pval,
        score = results$score,
        depth = results$RF_depth
    ))
}


compute_rf_cond <- function(sim_data,
                            index_i,
                            n = 1000L,
                            prob_type = "regression",
                            depth = 2,
                            ...) {
    print("Applying RF Conditional Permutation Method")

    deep_py <- import_from_path("permfit_py_RF",
        path = "permfit_python"
    )
    results <- deep_py$permfit(
        X_train = sim_data[, -1],
        y_train = as.numeric(sim_data$y),
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        max_depth = depth
    )

    return(data.frame(
        method = "CRF-RF",
        importance = results$importance,
        p_value = results$pval,
        score = results$score
        ))
}