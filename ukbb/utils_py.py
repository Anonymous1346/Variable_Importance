import sys
sys.path.insert(1, '../../tuan_binh_nguyen/dev')
sys.path.insert(1, '../permfit_python')
import sandbox
from permfit_py import permfit
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def compute_d0crt(x, y, loss = "least_square",
                  statistic = "residual", ntree = 100,
                  prob_type = "regression", verbose = False,
                  scaled_statistics = False, refit = False,
                  n_jobs=20):

    d0crt_results = sandbox.dcrt_zero(x,
                                      y,
                                      loss = "least_square",
                                      screening = False,
                                      statistic = "randomforest",
                                      ntree = 100,
                                      type_prob = "regression",
                                      refit = False,
                                      verbose = True,
                                      n_jobs=n_jobs)

    return pd.DataFrame({'variable':x.columns,
                        'importance':d0crt_results[2],
                        'p_value':d0crt_results[1],
                        'score':d0crt_results[3]})


def compute_RF(x, y, ntree = 100):
    print("Applying RF_MDI Method")

    rf = RandomForestRegressor(n_estimators = ntree)
    rf.fit(x, y)

    print("MAE:{}".format(mean_absolute_error(y, rf.predict(x))))

    return pd.DataFrame({'variable':x.columns,
                         'importance':rf.feature_importances_})


def compute_crf_pypermfitDnn(x, y, conditional=False, n_jobs=10, list_nominal=[]):
    
    results = permfit(X_train=x, y_train=y, prob_type="regression",
                      conditional=conditional, n_jobs=n_jobs,
                      list_nominal=list_nominal)

    return pd.DataFrame({'variable':x.columns,
                        'importance':results['importance'],
                        'p_value':results['pval'],
                        'score': results['score']})


def compute_marginal(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), y, test_size=0.2)
    marginal_imp = []
    marginal_pval = []
    score = 0
    for i in range(X_train.shape[1]):
        reg = LinearRegression().fit(X_train[:, i].reshape(-1, 1), y_train)
        result = f_regression(X_train[:, i].reshape(-1, 1), y_train)
        marginal_imp.append(result[0][0])
        marginal_pval.append(result[1][0])
        score += reg.score(X_test[:, i].reshape(-1, 1), y_test)
    return pd.DataFrame({'variable':X.columns,
                        'importance':marginal_imp,
                        'p_value':marginal_pval,
                        'score': score/len(X.columns)})