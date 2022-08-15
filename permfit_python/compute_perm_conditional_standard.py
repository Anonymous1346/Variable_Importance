import numpy as np
import numba as nb
import torch
from numba import cuda
import os
from deep_model_standard import relu, sigmoid
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def pred_avg(optimal_list, X_test, y_test, prob_type, device, batch_size, list_cont=[]):
    org_pred = np.zeros(len(y_test))
    n_layer = len(optimal_list[0][0][0]) - 1
    for mod in optimal_list:
        X_test_scaled = X_test.copy()
        X_test_scaled[:, list_cont] = mod[1][0].transform(X_test[:, list_cont])
        for j in range(n_layer):
            if j == 0:
                pred = relu(X_test_scaled.dot(mod[0][0][j]) + mod[0][1][j])
            else:
                pred = relu(pred.dot(mod[0][0][j]) + mod[0][1][j])
        
        pred = (pred.dot(mod[0][0][n_layer]) + mod[0][1][n_layer])[:, 0]
        # Original Predictions
        if prob_type == 'regression':
            org_pred += pred * mod[1][1].scale_ + mod[1][1].mean_
        else:
            org_pred += sigmoid(pred)
    org_pred /= len(optimal_list)
    return org_pred


def conf_binary(y, y_min, y_max):
    y_new = np.empty(len(y))
    for el in range(len(y)):
        if y[el] < y_min:
            y_new[el] = y_min
        elif y[el] > y_max:
            y_new[el] = y_max
        else:
            y_new[el] = y[el]
    return y_new


def joblib_compute_perm_conditional(p_col, n_sample, optimal_list, X_test,
                                    y_test, prob_type, device, batch_size,
                                    org_pred, seed=None, dict_cont={}, dict_nom={},
                                    X_nominal=None, encoder={}, max_depth=2):
    print(f"Processing col:{p_col}")
    res_ar = np.empty((n_sample, len(y_test)))
    y_test_new = np.ravel(y_test)

    if p_col in dict_nom.keys():
        output = np.ravel(X_nominal[p_col])
        p_col_n = dict_nom[p_col]
        regr = RandomForestClassifier(max_depth=max_depth)
        var_type = "classification"
        predict_func = regr.predict_proba
    else:
        p_col_n = [dict_cont[p_col]]
        regr = RandomForestRegressor(max_depth=max_depth)
        output = np.ravel(X_test[:, p_col_n])
        var_type = "regression"
        predict_func = regr.predict

    X_test_minus_idx = np.delete(np.copy(X_test), p_col_n, 1)
    regr.fit(X_test_minus_idx, output)
    X_col_pred = predict_func(X_test_minus_idx)
    if var_type != "classification":
        Res_col = output - X_col_pred
    
    current_X = X_test.copy()
    for sample in range(n_sample):
        if var_type == "classification":
            X_col_new = np.array([np.random.choice(np.unique(X_nominal[p_col]),
                size=1, p=X_col_pred[i]) for i in range(X_col_pred.shape[0])])
            X_col_new = encoder[p_col].transform(X_col_new).toarray().astype('int32')
            # Check if to remove the zero columns from the one-hot encoding
            current_X[:, p_col_n] = X_col_new
        else:
            np.random.shuffle(Res_col)
            X_col_new = X_col_pred + Res_col
            current_X[:, p_col_n[0]] = X_col_new

        pred_i = pred_avg(optimal_list, current_X, y_test,
                            prob_type, device, batch_size,
                            list_cont=list(dict_cont.values()))
        if prob_type == 'regression':
            res_ar[sample, :] = (
                y_test_new - pred_i) ** 2 - (y_test_new - org_pred) ** 2
        else:
            y_max = 1-1e-10
            y_min = 1e-10
            pred_i = conf_binary(pred_i, y_min, y_max)
            org_pred = conf_binary(org_pred, y_min, y_max)
            res_ar[sample, :] = -y_test_new*np.log(pred_i) - (1-y_test_new)*np.log(1-pred_i) \
                + y_test_new*np.log(org_pred) + \
                (1-y_test_new)*np.log(1-org_pred)
    return res_ar


def joblib_compute_perm(p_col, perm, optimal_list, X_test, y_test, prob_type,
                        device, batch_size, org_pred, seed=None, dict_cont={},
                        dict_nom={}):
    y_test_new = np.ravel(y_test)
    print(f"Processing col:{p_col}")
    current_X = X_test.copy()
    indices = np.arange(X_test.shape[0])
    np.random.shuffle(indices)

    p_col = dict_nom[p_col] if p_col in dict_nom.keys() else [dict_cont[p_col]]
    current_X[:, p_col] = current_X[:, p_col][indices, :]

    pred_i = pred_avg(optimal_list, current_X, y_test, prob_type, device, batch_size, list_cont=list(dict_cont.values()))
    if prob_type == 'regression':
        res = (y_test_new - pred_i) ** 2 - (y_test_new - org_pred) ** 2
    else:
        y_max = 1-1e-10
        y_min = 1e-10
        pred_i = conf_binary(pred_i, y_min, y_max)
        org_pred = conf_binary(org_pred, y_min, y_max)
        res = -y_test_new*np.log(pred_i) - (1-y_test_new)*np.log(1-pred_i) \
            + y_test_new*np.log(org_pred) + \
            (1-y_test_new)*np.log(1-org_pred)
    return res


def joblib_compute_perm_conditional_RF_old(p_col, n_sample, rf, X_train_scaled, y_train,
                                           X_test_scaled, y_test, scaler_x, scaler_y, prob_type,
                                           org_pred, seed=None, dict_cont={}, dict_nom={},
                                           X_nominal_train=None, X_nominal_test=None,
                                           encoder={}, max_depth=2):
    res_ar = np.empty((n_sample, len(y_test)))
    y_test_new = np.ravel(y_test)

    if p_col in dict_nom.keys():
        output = np.ravel(X_nominal_train[p_col])
        p_col_n = dict_nom[p_col]
        regr = RandomForestClassifier()
        var_type = "classification"
        predict_func = regr.predict_proba
    else:
        p_col_n = [dict_cont[p_col]]
        regr = RandomForestRegressor()
        output = np.ravel(X_train_scaled[:, p_col_n])
        output_test = np.ravel(X_test_scaled[:, p_col_n])
        var_type = "regression"
        predict_func = regr.predict

    X_train_minus_idx = np.delete(np.copy(X_train_scaled), p_col_n, 1)
    X_test_minus_idx = np.delete(np.copy(X_test_scaled), p_col_n, 1)
    regr.fit(X_train_minus_idx, output)
    X_col_pred = predict_func(X_test_minus_idx)
    if var_type != "classification":
        Res_col = output_test - X_col_pred

    current_X = X_test_scaled.copy()
    for sample in range(n_sample):
        if var_type == "classification":
            X_col_new = np.array([np.random.choice(np.unique(X_nominal_train[p_col]),
                size=1, p=X_col_pred[i]) for i in range(X_col_pred.shape[0])])
            X_col_new = encoder[p_col].transform(X_col_new).toarray().astype('int32')
            # Check if to remove the zero columns from the one-hot encoding
            current_X[:, p_col_n] = X_col_new
        else:
            np.random.shuffle(Res_col)
            X_col_new = X_col_pred + Res_col
            current_X[:, p_col_n[0]] = X_col_new

        if prob_type == 'regression':
            pred_i = rf.predict(current_X) * scaler_y.scale_ + scaler_y.mean_
            res_ar[sample, :] = (
                y_test_new - pred_i) ** 2 - (y_test_new - org_pred) ** 2
        else:
            pred_i = rf.predict_proba(current_X)[:, 1]
            y_max = 1-1e-10
            y_min = 1e-10
            pred_i = conf_binary(pred_i, y_min, y_max)
            org_pred = conf_binary(org_pred, y_min, y_max)
            res_ar[sample, :] = -y_test_new*np.log(pred_i) - (1-y_test_new)*np.log(1-pred_i) \
                + y_test_new*np.log(org_pred) + \
                (1-y_test_new)*np.log(1-org_pred)
    return res_ar


def joblib_compute_perm_conditional_RF_n(p_col, n_sample, rf, X_train_scaled, y_train,
                                       X_test_scaled, y_test, scaler_x, scaler_y, prob_type,
                                       org_pred, seed=None, dict_cont={}, dict_nom={},
                                       X_nominal_train=None, X_nominal_test=None,
                                       encoder={}, max_depth=2):
    res_ar = np.empty((n_sample, len(y_test)))
    y_test_new = np.ravel(y_test)
    X_test = X_test_scaled * scaler_x.scale_ + scaler_x.mean_

    if p_col in dict_nom.keys():
        output = np.ravel(X_nominal_train[p_col])
        p_col_n = dict_nom[p_col]
        regr = RandomForestClassifier(max_depth=max_depth)
        var_type = "classification"
        predict_func = regr.predict_proba
    else:
        p_col_n = [dict_cont[p_col]]
        regr = RandomForestRegressor(max_depth=max_depth)
        output = np.ravel(X_test[:, p_col_n])
        var_type = "regression"
        predict_func = regr.predict

    X_test_minus_idx = np.delete(np.copy(X_test), p_col_n, 1)
    regr.fit(X_test_minus_idx, output)
    X_col_pred = predict_func(X_test_minus_idx)
    if var_type != "classification":
        Res_col = output - X_col_pred

    current_X = X_test_scaled.copy()
    for sample in range(n_sample):
        if var_type == "classification":
            X_col_new = np.array([np.random.choice(np.unique(X_nominal_train[p_col]),
                size=1, p=X_col_pred[i]) for i in range(X_col_pred.shape[0])])
            X_col_new = encoder[p_col].transform(X_col_new).toarray().astype('int32')
            # Check if to remove the zero columns from the one-hot encoding
            current_X[:, p_col_n] = X_col_new
        else:
            current_X = X_test.copy()
            np.random.shuffle(Res_col)
            X_col_new = X_col_pred + Res_col
            current_X[:, p_col_n[0]] = X_col_new
            current_X = scaler_x.transform(current_X)

        if prob_type == 'regression':
            pred_i = rf.predict(current_X) * scaler_y.scale_ + scaler_y.mean_
            res_ar[sample, :] = (
                y_test_new - pred_i) ** 2 - (y_test_new - org_pred) ** 2
        else:
            pred_i = rf.predict_proba(current_X)[:, 1]
            y_max = 1-1e-10
            y_min = 1e-10
            pred_i = conf_binary(pred_i, y_min, y_max)
            org_pred = conf_binary(org_pred, y_min, y_max)
            res_ar[sample, :] = -y_test_new*np.log(pred_i) - (1-y_test_new)*np.log(1-pred_i) \
                + y_test_new*np.log(org_pred) + \
                (1-y_test_new)*np.log(1-org_pred)
    return res_ar
