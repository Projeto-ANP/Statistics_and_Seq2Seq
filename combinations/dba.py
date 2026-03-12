from . import aux

import pandas as pd
import numpy as np
# tslearn is already listed in enviroment.yml
from tslearn.barycenters import dtw_barycenter_averaging
     
def dba_combination(models, dataset_name, dataset_index, val_type="final"):
    """Combine model forecasts using DTW‑Barycenter Averaging.

    The input predictions for each model form a univariate time series; DBA
    will compute a centroid sequence that minimises average DTW distance.

    Returns a pandas Series of the averaged predictions plus the shared test
    series and the associated date bounds (start_test/final_test).
    """

    # grab the full structure from the aux module
    model_preds = aux.get_predictions_models(models, dataset_index, dataset_name)

    # stack predictions into matrix shape (n_models, horizon)
    series_list = [model_preds[m][val_type]["predictions"] for m in models]
    X = np.vstack(series_list)  # shape: (n_models, horizon)

    # tslearn expects shape (n_ts, sz, d) where d=1 for univariate
    X3 = X.reshape(X.shape[0], X.shape[1], 1)

    # compute DBA centroid (returns shape (horizon, 1))
    centroid = dtw_barycenter_averaging(X3)
    centroid = centroid.ravel()  # flatten to 1‑D

    # convert back to pandas Series (index just range same as horizon)
    dba_preds = pd.Series(centroid)

    test_series = model_preds[models[0]][val_type]["test"]
    start_test = model_preds[models[0]][val_type]["start_test"]
    final_test = model_preds[models[0]][val_type]["final_test"]

    return dba_preds, test_series, start_test, final_test

#python -m combinations.dba
if __name__ == "__main__":
    models = [
        "ARIMA",
        "ETS",
        "THETA",
        "svr",
        "rf",
        "catboost",
        "CWT_svr",
        "DWT_svr",
        "FT_svr",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
        "ONLY_CWT_catboost",
        "ONLY_CWT_rf",
        "ONLY_CWT_svr",
        "ONLY_DWT_catboost",
        "ONLY_DWT_rf",
        "ONLY_DWT_svr",
        "ONLY_FT_catboost",
        "ONLY_FT_rf",
        "ONLY_FT_svr",
        "NaiveSeasonal",
        "NaiveMovingAverage",
    ]
    
    dataset_name = "ETTH1"
    
    len_datasets = aux.get_dataset_size(models[0], dataset_name=dataset_name)
    for dataset_index in range(len_datasets):
        mean_result, test_values, start_test, final_test = dba_combination(models, dataset_name=dataset_name, dataset_index=dataset_index)
        aux.save_to_csv(exp_name="dba", predictions=mean_result, test_values=test_values, dataset_name=dataset_name, dataset_index=dataset_index, horizon=12, start_test=start_test, final_test=final_test)