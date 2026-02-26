import pandas as pd
import re
import os
from . import metrics
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []

def read_model_preds(model_name, dataset_index, dataset_name):
    
    df = pd.read_csv(
        f"./timeseries/mestrado/resultados/{model_name}/normal/{dataset_name}.csv",
        sep=";",
    )
    df = df[df["dataset_index"] == dataset_index]

    # timestamps may include a time component (e.g. "2023-01-01 20:00:00")
    # so let pandas infer the format rather than enforcing one.
    df["start_test"] = pd.to_datetime(df["start_test"], errors="coerce")
    df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
    df = df.sort_values(by="start_test")

    return df

def get_dataset_size(model_name, dataset_name):
    """Return number of unique dataset_index values in the given results file.

    The CSV may contain many rows per index (e.g. validation folds), but we're
    interested only in how many distinct dataset_index samples are present.
    """
    df = pd.read_csv(
        f"./timeseries/mestrado/resultados/{model_name}/normal/{dataset_name}.csv",
        sep=";",
    )

    # count unique values in dataset_index column; treat missing as NaN
    if "dataset_index" not in df.columns:
        raise KeyError("dataset_index column not found in file")

    return int(df["dataset_index"].nunique())


def get_predictions_models(models, dataset_index, dataset_name):
    model_predictions = {}
    for model in models:
        df_model = read_model_preds(model, dataset_index, dataset_name)
        
        final = df_model.iloc[-1]
        val1 = df_model.iloc[-2]
        val2 = df_model.iloc[-3]
        val3 = df_model.iloc[-4]
        
        # predictions_final = extract_values(df_final_test["predictions"])
        final_predictions = extract_values(final["predictions"])
        val_predictions1 = extract_values(val1["predictions"])
        val_predictions2 = extract_values(val2["predictions"])
        val_predictions3 = extract_values(val3["predictions"])
        
        
        final_test = extract_values(final["test"])
        test_val1 = extract_values(val1["test"])
        test_val2 = extract_values(val2["test"])
        test_val3 = extract_values(val3["test"])
        #validation 1 is more recent than validation 2, which is more recent than validation 3
        model_predictions[model] = {
            "final": { "predictions": final_predictions, "test": final_test, "start_test": final["start_test"], "final_test": final["final_test"]},
            "val1": {   "predictions": val_predictions1, "test": test_val1, "start_test": val1["start_test"], "final_test": val1["final_test"]},
            "val2": {   "predictions": val_predictions2, "test": test_val2, "start_test": val2["start_test"], "final_test": val2["final_test"]},
            "val3": {   "predictions": val_predictions3, "test": test_val3, "start_test": val3["start_test"], "final_test": val3["final_test"]}
        }
            
    return model_predictions



def save_to_csv(exp_name, predictions, test_values, dataset_name, dataset_index, horizon, start_test, final_test):
    cols_serie = [
        "dataset_index",
        "horizon",
        "regressor",
        "mape",
        "pocid",
        "smape",
        "rmse",
        "msmape",
        "mae",
        "test",
        "predictions",
        "start_test",
        "final_test",
    ]
    
    path_experiments = f"./timeseries/mestrado/resultados/{exp_name}/"
    path_csv = f"{path_experiments}/{dataset_name}.csv"
    os.makedirs(path_experiments, exist_ok=True)
    
    
    test = np.asarray(test_values)
    preds_real_array = np.array(predictions.values)
    preds_real_reshaped = preds_real_array.reshape(1, -1)
    test_reshaped = test.reshape(1, -1)
    smape_result = metrics.calculate_smape(preds_real_reshaped, test_reshaped)
    # print(smape_result)
    rmse_result = metrics.calculate_rmse(preds_real_reshaped, test_reshaped)
    msmape_result = metrics.calculate_msmape(preds_real_reshaped, test_reshaped)
    # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
    mae_result = metrics.calculate_mae(preds_real_reshaped, test_reshaped)
    mape_result = mape(test, preds_real_array)
    pocid_result = metrics.pocid(test, preds_real_array)

    data_serie = {
        "dataset_index": f"{dataset_index}",
        "horizon": horizon,
        "regressor": exp_name,
        "mape": mape_result,
        "pocid": pocid_result,
        "smape": smape_result,
        "rmse": rmse_result,
        "msmape": msmape_result,
        "mae": mae_result,
        "test": [test.tolist()],
        "predictions": [predictions.tolist()],
        "start_test": start_test,
        "final_test": final_test,
        # 'training_time': times[0],
        # 'prediction_time': times[1],
    }

    # create file with column headers if it doesn't exist or is empty
    if (not os.path.exists(path_csv)) or os.path.getsize(path_csv) == 0:
        pd.DataFrame(columns=cols_serie).to_csv(
            path_csv, sep=";", index=False
        )

    df_new = pd.DataFrame(data_serie)
    df_new.to_csv(
        path_csv, sep=";", mode="a", header=False, index=False
    )
    