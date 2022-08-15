import os
import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
import torch
from deep_model_standard import ensemble_dnnet
from compute_perm_conditional_standard import (pred_avg, joblib_compute_perm,
    joblib_compute_perm_conditional)
import time


def permfit(X_train, y_train, train_index=None, test_index=None, prob_type='regression', k_fold=0,
            split_perc=0.2, random_state=2021, n_ensemble=10, batch_size=1024, n_perm=100,
            res=True, verbose=10, conditional=False, n_jobs=1,
            list_nominal = [], max_depth=2, index_i=None, save_file=None):
    '''
    X_train: The matrix of predictors to train/validate
    y_train: The response vector to train/validate
    train_index: if given, the indices of the train matrix
    test_index: if given, the indices of the test matrix
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
    # No cross validation, splitting the samples into train/validate portions
    if k_fold == 0:
        # One-hot encoding of Nominal variables
        tmp_list = []
        dict_nom = {}
        X_nominal = None
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

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if prob_type == 'classification':
            # Converting target values to corresponding integer values
            ind_unique = np.unique(y_train)
            dict_target = dict(zip(ind_unique, range(len(ind_unique))))
            y_train = np.array([dict_target[el]
                                for el in list(y_train)]).reshape(-1, 1)
            score = roc_auc_score
        else:
            y_train = np.array(y_train).reshape(-1, 1)
            score = r2_score
        if train_index is None:
            train_index = np.random.choice(X_train.shape[0], size=int(X_train.shape[0]*(1-split_perc)), replace=False)
            test_index = np.array([ind for ind in range(n) if ind not in train_index])
        X_train, X_test = X_train[train_index, :], X_train[test_index, :]
        y_train, y_test = y_train[train_index], y_train[test_index]
        if isinstance(X_nominal, pd.DataFrame):
            X_nominal = X_nominal.iloc[test_index, :]

        parallel = Parallel(n_jobs=min(n_jobs, 10), verbose=verbose)
        optimal_list = ensemble_dnnet(X_train, y_train, n_ensemble=n_ensemble,
                                      prob_type=prob_type, save_file=save_file,
                                      verbose=verbose, random_state = random_state,
                                      parallel=parallel, list_cont=list(dict_cont.values()))
        org_pred = pred_avg(optimal_list, X_test, y_test, prob_type, device, batch_size,
                            list(dict_cont.values()))

        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        if not conditional:
            # print("Apply Permutation!")
            p2_score = np.array(parallel(delayed(joblib_compute_perm)(p_col, perm, optimal_list, X_test, y_test, prob_type, device,
                                                                      batch_size, org_pred, seed=random_state, dict_cont=dict_cont,
                                                                      dict_nom=dict_nom)
                                        for p_col in list_cols for perm in range(n_perm))).reshape((p, n_perm, len(y_test)))
        else:
            # print("Apply Conditional!")
            p2_score = np.array(parallel(delayed(joblib_compute_perm_conditional)(p_col, n_perm, optimal_list, X_test, y_test, prob_type, device,
                                                                     batch_size, org_pred, seed=random_state, dict_cont=dict_cont,
                                                                     dict_nom=dict_nom, X_nominal=X_nominal, encoder=enc_dict, max_depth=max_depth)
                                        for p_col in list_cols))
            results['RF_depth'] = max_depth
        if not res:
            return p2_score, org_pred, score(y_test, org_pred)
        
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
        score_l = []
        kf = KFold(n_splits=k_fold, random_state=random_state, shuffle=True)
        p2_score = np.empty((p, n_perm, n))
        i = 0
        for train_index, test_index in kf.split(X_train):
            print(f"Fold: {i+1}")
            i += 1
            valid_ind.append((train_index, test_index))
            p2_score[:, :, test_index], org_pred, score_val = permfit(X_train, y_train, train_index, test_index, prob_type,
                                                                      k_fold=0, split_perc=split_perc, random_state=random_state,
                                                                      n_ensemble=n_ensemble, batch_size=batch_size, n_perm=n_perm,
                                                                      res=False, conditional=conditional, n_jobs=n_jobs,
                                                                      list_nominal=list_nominal, max_depth=max_depth)
            print(f"Done Fold: {i+1}")
            score_l.append(score_val)

        results = {}
        results['importance'] = np.mean(np.mean(p2_score, axis=2), axis=1)
        results['std'] = np.std(np.mean(p2_score, axis=1),
                                axis=1) / np.sqrt(p2_score.shape[2]-1)
        results['pval'] = norm.sf(results['importance'] / results['std'])
        results['score'] = sum(score_l) / len(score_l)
        results['validation_ind'] = valid_ind
        return results

