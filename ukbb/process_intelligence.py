import numpy as np
import pandas as pd
import time
from utils_py import compute_d0crt, compute_cpi_permfitDnn, compute_marginal


df = pd.read_csv("ukbb_data_intelligence_no_hot_encoding.csv")
X = df.loc[:, df.columns != "20016-2.0"]
y = df["20016-2.0"]
nominal_columns = ['31-0.0', '670-2.0', '680-2.0', '1647-2.0', '1677-2.0', 
                   '1707-2.0', '1767-2.0', '1777-2.0', '1787-2.0', '2040-2.0',
                   '2090-2.0', '2100-1.0', '2644-2.0', '2877-2.0', '2907-2.0',
                   '3446-2.0', '4598-2.0', '4631-2.0', '4642-2.0', '4653-2.0',
                   '5674-2.0', '5959-2.0', '6138-2.0', '6139-2.0', '6141-2.0',
                   '6142-2.0', '6143-2.0', '6145-2.0', '6156-2.0', '6157-2.0',
                   '6158-2.0', '6160-2.0']

marginal = False
permfit_dnn = True
cpi_dnn = False
d0crt = False

start_time = time.monotonic()
## Marginal
if marginal:
    print("Applying Marginal")
    res = compute_marginal(X, y)
    res['time'] = (time.monotonic() - start_time) / 60
    res = res.sort_values(by=["p_value"], ascending=True)
    res.to_csv("results_marginal_intelligence_withscore.csv", index=False)


## Permfit-DNN
if permfit_dnn:
    print("Applying Permfit-DNN")
    res = compute_cpi_permfitDnn(X, y, conditional=False, k_fold=5, random_state=2022,
                                    n_jobs=100, list_nominal=nominal_columns)
    res['time'] = (time.monotonic() - start_time) / 60
    res = res.sort_values(by=["p_value"], ascending=True)
    res.to_csv("results_permfitdnn_intelligence_withscore_cross_1.csv", index=False)


## CPI-DNN
if cpi_dnn:
    print("Applying CPI-DNN")
    res = compute_cpi_permfitDnn(X, y, conditional=True, k_fold=5, random_state=2022,
                                   n_jobs=100, list_nominal=nominal_columns)
    res['time'] = (time.monotonic() - start_time) / 60
    res = res.sort_values(by=["p_value"], ascending=True)
    res.to_csv("results_cpidnn_intelligence_withscore_cross.csv", index=False)


## d0crt method
if d0crt:
    print("Applying d0CRT")
    res = compute_d0crt(X, y, n_jobs=20)
    res['importance'] = np.abs(res["importance"])
    res['time'] = (time.monotonic() - start_time) / 60
    res = res.sort_values(by=["p_value"], ascending=True)
    res.to_csv("results_d0crt_intelligence.csv", index=False)

print('mins: ', (time.monotonic() - start_time) / 60)
