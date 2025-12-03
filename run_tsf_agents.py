import pandas as pd
import re
import numpy as np
from all_functions import *
import os
from sklearn.metrics import mean_absolute_percentage_error as mape
def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []

def read_model_preds(model_name, dataset_index):
    df = pd.read_csv(
        f"./Statistics_and_Seq2Seq/timeseries/mestrado/resultados/{model_name}/normal/ANP_MONTHLY.csv",
        sep=";",
    )
    df = df[df["dataset_index"] == dataset_index]

    df["start_test"] = pd.to_datetime(df["start_test"], format="%Y-%m-%d")
    df["final_test"] = pd.to_datetime(df["final_test"], format="%Y-%m-%d")
    df = df.sort_values(by="start_test")

    return df

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
def get_predictions_models(models, dataset_index, final_test):
    all_data = {}
    final_test_predictions = {}
    final_test_data = None

    final_test_date = pd.to_datetime(final_test, format="%Y-%m-%d")

    for model in models:
        df = read_model_preds(model, dataset_index)

        # train_df = df[df["final_test"] < final_test_date]
        # train_df = train_df[train_df["start_test"] >= "2010-12-31"]
        test_df = df[df["final_test"] == final_test_date]
        all_data[model] = []

        # for index, row in train_df.iterrows():
        #     preds_model = extract_values(row["predictions"])
        #     test = extract_values(row["test"])
        #     all_data[model].append(
        #         {"predictions": preds_model, "test": test, "date": row["start_test"]}
        #     )

        if not test_df.empty:
            final_row = test_df.iloc[0]
            final_test_predictions[model] = extract_values(final_row["predictions"])
            final_test_data = extract_values(final_row["test"])
        
    
    return final_test_predictions, final_test_data
def exec_dataset(models):
    dataset = "ANP_MONTHLY"
    exp_name = "simple_selective_agent_qwen3=14b"
    horizon = 12
    final_test = "2024-11-30"
    
    path_experiments = f"./Statistics_and_Seq2Seq/timeseries/mestrado/resultados/{exp_name}/"
    path_csv = f"{path_experiments}/{dataset}.csv"
    os.makedirs(path_experiments, exist_ok=True)
    for i in range (0, 182):
        
        val_predictions, val_test = get_predictions_models(models, dataset_index=i, final_test="2023-11-30")
        predictions, test = get_predictions_models(models, dataset_index=i, final_test=final_test)
        
        from timellm.combinator import simple_selective_agent
        
        description, preds_real = simple_selective_agent(val_test, val_predictions, predictions)
        print(f"----- DATASET INDEX: {i} -----")
        print("Description: ", description)
        print("Predictions: ", preds_real)
        test = np.array(test)
        preds_real_array = np.array(preds_real)
        preds_real_reshaped = preds_real_array.reshape(1, -1)
        test_reshaped = test.reshape(1, -1)
        smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
        # print(smape_result)
        rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
        msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
        # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
        mae_result = calculate_mae(preds_real_reshaped, test_reshaped)
        mape_result = mape(test, preds_real_array)
        pocid_result = pocid(test, preds_real_array)

        data_serie = {
            "dataset_index": f"{i}",
            "horizon": horizon,
            "regressor": exp_name,
            "mape": mape_result,
            "pocid": pocid_result,
            "smape": smape_result,
            "rmse": rmse_result,
            "msmape": msmape_result,
            "mae": mae_result,
            "test": [test.tolist()],
            "predictions": [preds_real],
            "start_test": "INICIO",
            "final_test": final_test,
            "description": description,
            # 'training_time': times[0],
            # 'prediction_time': times[1],
        }

        if not os.path.exists(path_csv):
            pd.DataFrame(columns=cols_serie).to_csv(
                path_csv, sep=";", index=False
            )

        print("Salvando resultados...\n")
        df_new = pd.DataFrame(data_serie)
        df_new.to_csv(
            path_csv, sep=";", mode="a", header=False, index=False
        )

    


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
        "NaiveMovingAverage"
    ]

    exec_dataset(models)
