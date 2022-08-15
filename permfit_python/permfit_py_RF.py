import os
import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from deep_model_standard import ensemble_dnnet
from compute_perm_conditional_standard import (pred_avg, joblib_compute_perm,
    joblib_compute_perm_conditional, joblib_compute_perm_conditional_RF_n)
import time


def permfit(X_train, y_train, X_test=None, y_test=None, prob_type='regression', k_fold=0,
            split_perc=0.2, random_state=2021, n_ensemble=10, batch_size=1024, n_perm=100,
            res=True, verbose=0, conditional=False, n_jobs=1,
            list_nominal = [], max_depth=2, index_i=None, save_file=None):
    '''
    X_train: The matrix of predictors to train/validate
    y_train: The response vector to train/validate
    X_test: If given, The matrix of predictors to test
    y_test: If given, The response vector to test
    prob_type: A classification or regression probem
    k_fold: The number of folds for the cross validaton
    split_perc: if k_fold==0, The percentage of the split to train/validate portions
    random_state: Fixing the seeds of the random generator
    n_ensemble: The number of DNNs to be fit
    batch_size: The number of samples per batch for test prediction
    n_perm: The number of permutations for each column
    res: If True, it will return the dictionary of results
    verbose: If > 1, the progress bar will be printed
    conditional: The permutation or the conditional sampling approach
    list_nominal: The list of categorical variables if exists
    '''
    if index_i != None:
        print(f"Processing iteration: {index_i}")
    results = {}
    n, p = X_train.shape
    list_cols = list(X_train.columns)
    # One-hot encoding of Nominal variables

    tmp_list = []
    dict_nom = {}
    X_nominal = None
    X_nominal_train = None
    X_nominal_test = None

    enc_dict = {}
    if len(list_nominal) > 0:
        X_nominal = X_train[list(set(list_nominal) & set(list_cols))]
        for col_encode in list(set(list_nominal) & set(list_cols)):
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(X_train[[col_encode]])
            labeled_cols = [enc.feature_names_in_[0] + '_' + str(enc.categories_[0][j]) 
            for j in range(len(enc.categories_[0]))]
            hot_cols = pd.DataFrame(enc.transform(X_train[[col_encode]]).toarray(),
            dtype='int32', columns=labeled_cols)
            X_train = X_train.drop(columns=[col_encode])
            X_train = pd.concat([X_train, hot_cols], axis=1)
            enc_dict[col_encode] = enc
    # Create a dictionary for the one-hot encoded variables with the indices of the corresponding categories
        for col_nom in list_nominal:
            current_list = [col for col in range(len(X_train.columns)) if X_train.columns[col].split('_')[0] == col_nom]
            if len(current_list) > 0:
                dict_nom[col_nom] = current_list
                tmp_list.extend(current_list)

    # Retrieve the list of continuous variables that will be scaled
    dict_cont = {}
    for col_cont in range(len(X_train.columns)):
        if col_cont not in tmp_list:
            dict_cont[X_train.columns[col_cont]] = col_cont
    X_train = X_train.to_numpy()

    if prob_type == 'classification':
        # Converting target values to corresponding integer values
        ind_unique = np.unique(y_train)
        dict_target = dict(zip(ind_unique, range(len(ind_unique))))
        y_train = np.array([dict_target[el]
                            for el in list(y_train)]).reshape(-1, 1)
        score = roc_auc_score
        rf = RandomForestClassifier()
    else:
        y_train = np.array(y_train).reshape(-1, 1)
        score = r2_score
        rf = RandomForestRegressor()
    # No cross validation, splitting the samples into train/validate portions
    if k_fold == 0:
        if X_test is None:
            indices = np.random.choice(X_train.shape[0], size=int(X_train.shape[0]*(1-split_perc)), replace=False)
            test_indices = np.array([ind for ind in range(n) if ind not in indices])
            X_train, X_test = X_train[indices], X_train[test_indices]
            y_train, y_test = y_train[indices], y_train[test_indices]
            if isinstance(X_nominal, pd.DataFrame):
                X_nominal_train = X_nominal.iloc[indices, :]
                X_nominal_test = X_nominal.iloc[test_indices, :]
        else:
            y_test = np.array(y_test).reshape(-1, 1)
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

        list_cont = list(dict_cont.values())
        scaler_x, scaler_y = StandardScaler(), StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[:, list_cont] = scaler_x.fit_transform(X_train[:, list_cont])
        X_test_scaled[:, list_cont] = scaler_x.transform(X_test[:, list_cont])

        if prob_type == 'regression':
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)
        else:
            y_train_scaled = y_train.copy()
            y_test_scaled = y_test.copy()

        rf.fit(X_train_scaled, y_train_scaled)

        if prob_type == 'regression':
            org_pred = rf.predict(X_test_scaled) * scaler_y.scale_ + scaler_y.mean_
        else:
            org_pred = rf.predict_proba(X_test_scaled)[:, 1]

        p2_score = np.array(parallel(delayed(joblib_compute_perm_conditional_RF_n)(p_col, n_perm, rf, X_train_scaled, y_train, 
                                                                                 X_test_scaled, y_test, scaler_x, scaler_y, prob_type,
                                                                                 org_pred, seed=random_state, dict_cont=dict_cont,
                                                                                 dict_nom=dict_nom, X_nominal_train=X_nominal_train,
                                                                                 X_nominal_test=X_nominal_test, encoder=enc_dict,
                                                                                 max_depth=max_depth)
                                        for p_col in list_cols))
        if not res:
            return p2_score
        
        # p2_score (nb_features (p) x nb_permutations x length_ytest)
        results['importance'] = np.mean(np.mean(p2_score, axis=2), axis=1)
        results['std'] = np.std(np.mean(p2_score, axis=1),
                                axis=1) / np.sqrt(len(y_test)-1)
        results['pval'] = norm.sf(results['importance'] / results['std'])
        results['score'] = score(y_test, org_pred)
        if index_i != None:
            print(f"Done processing iteration: {index_i}")
        return results
    else:
        valid_ind = []
        kf = KFold(n_splits=2, random_state=random_state, shuffle=True)
        p2_score = np.empty((n_perm, n, p))
        i = 0
        for train_index, test_index in kf.split(X):
            print(f"Fold: {i+1}")
            i += 1
            valid_ind.append((train_index, test_index))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            current_score = permfit(X_train, y_train, X_test, y_test, prob_type,
                                    k_fold=0, split_perc=split_perc, random_state=random_state,
                                    n_ensemble=n_ensemble, batch_size=batch_size, n_perm=n_perm,
                                    res=False)
            p2_score[:, test_index, :] = current_score

        results = {}
        results['importance'] = np.mean(np.mean(p2_score, axis=0), axis=0)
        results['std'] = np.std(np.mean(p2_score, axis=0),
                                axis=0) / np.sqrt(len(y_test)-1)
        results['pval'] = norm.sf(results['importance'] / results['std'])
        results['validation_ind'] = valid_ind
        return results
