import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from all_functions import *
import os
import time
import multiprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
warnings.filterwarnings("ignore", category=FutureWarning, module="aeon")
def ensure_even(n):
    return n + (n % 2)

# datasets = [
#             "monthly", 
#             "weekly", 
#             "yearly", 
#             "quarterly", 
#             "daily", 
#             "hourly"
#             ]

datasets = [
            # "kdd_cup",  #horizon 24
            # "saugeenday", 
            # "solar4", 
            # "sunspot", 
            # "us_births", #horizon 7
            # "vehicle_trips",
            # "wind4",
            # "cif2016",
            # "electricity_weekly_dataset",
            # "pedestrian_counts_dataset",
            # "dominick_dataset"
            "australian_electricity_demand_dataset" #horizon 48
            ]
cols_serie = ["dataset_index", "horizon","regressor", "mape", "pocid", "smape", "rmse", "msmape", "mae", "test", "predictions", "features_time", "training_time", "prediction_time"]
cols_dataset = ["dataset", "regressor", "mape_mean", "pocid_mean", "smape_mean", "rmse_mean", "msmape_mean", "mae_mean"]

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

def run_m4(args):
    dataset = args
    isNormal = False
    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
        f'../m4/timeseries/{dataset}.tsf',
        replace_missing_vals_with="NaN",
        value_column_name="series_value",
    )
    transform = "normal"
    regr = "rf"
    representation = ""

    wavelet = "bior2.2"
    level = 2
    exp_name = f"{representation}_{regr}_{transform}"
    path_experiments = f'../timeseries/{dataset}/{representation}_{regr}/{transform}/'
    path_csv = f"{path_experiments}/{dataset}.csv"

    os.makedirs(path_experiments, exist_ok=True)

    rmses, smapes, msmapes, maes, mapes, pocids = [], [], [], [], [], []
    if representation == "":
        isNormal = True
    for index in range(len(df)):
        try:
            times = []
            horizon = 48
            window = horizon
            series = df.iloc[index]['series_value'].tolist()
            train, test = train_test_stats(pd.Series(series), horizon)
        
            time_features_start = time.perf_counter()
            train_tf = transform_regressors(train, transform)
            if isNormal:
                data = rolling_window(pd.concat([train_tf, pd.Series([0] * horizon, index=test.index)]), window)
            else:
                data = rolling_window_image(pd.concat([train_tf, pd.Series([0] * horizon, index=test.index)]), window, representation, wavelet, level)
            data = data.dropna()

            time_features_end = time.perf_counter()
            times.append(time_features_end - time_features_start)

            X_train, X_test, y_train, _ = train_test_split(data, horizon) 
            if regr == "ridge":
                rg = RidgeCV(alphas=np.logspace(-3, 3, 10))
            elif regr == "rf":
                rg = RandomForestRegressor(random_state=42)
            elif regr == "catboost":
                rg = CatBoostRegressor(**{'iterations': 200, 'learning_rate': 0.18623693782832346, 'depth': 7, 'loss_function': 'RMSE', 'random_state': 42})

            time_training_start = time.perf_counter()
            rg.fit(X_train, y_train)
            time_training_end = time.perf_counter()
            times.append(time_training_end - time_training_start)

            time_preds_start = time.perf_counter()
            if isNormal:
                predictions = recursive_multistep_forecasting(X_test, rg, horizon)
                preds = pd.Series(predictions, index=test.index)
                preds_real = reverse_regressors(train, preds, window, format=transform)
            else:
                predictions = recursive_step(X_test, train, rg, horizon, window, transform, representation, wavelet, level)
                preds_real = pd.Series(predictions, index=test.index)

            time_preds_end = time.perf_counter()
            times.append(time_preds_end - time_preds_start)
            
            preds_real_array = np.array(preds_real)
            preds_real_reshaped = preds_real_array.reshape(1, -1)
            test_reshaped = test.values.reshape(1, -1)
            smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
            rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
            msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
            # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
            mae_result = calculate_mae(preds_real_reshaped, test_reshaped)
            mape_result = mape(test.values, preds_real_array)
            pocid_result = pocid(test.values, preds_real_array)

            data_serie = {
                'dataset_index': f'{index}',
                'horizon': horizon,
                'regressor': exp_name,
                'mape': mape_result,
                'pocid': pocid_result,
                'smape': smape_result,
                'rmse': rmse_result,
                'msmape': msmape_result,
                'mae': mae_result,
                'test': [test.tolist()],
                'predictions': [np.array(preds_real)],
                'features_time': times[0],
                'training_time': times[1],
                'prediction_time': times[2],
            }

            if not os.path.exists(path_csv):
                pd.DataFrame(columns=cols_serie).to_csv(path_csv, sep=';', index=False)

            df_new = pd.DataFrame(data_serie)
            df_new.to_csv(path_csv, sep=';', mode='a', header=False, index=False)

            maes.append(mae_result)
            rmses.append(rmse_result)
            msmapes.append(msmape_result)
            smapes.append(smape_result)
            mapes.append(mape_result)
            pocids.append(pocid_result)
        except Exception as e:
            print(f"Error, Crashou dataset: {index}, {e}")
            

    data_dataset = {
        "dataset": dataset, 
        "regressor": exp_name, 
        "mape_mean": np.nanmean(mapes),
        "pocid_mean": np.nanmean(pocids),
        "smape_mean": np.nanmean(smapes), 
        "rmse_mean": np.nanmean(rmses), 
        "msmape_mean": np.nanmean(msmapes), 
        "mae_mean": np.nanmean(maes)
    }
    path_dataset = path_experiments + "results.csv"
    if not os.path.exists(path_dataset):
        pd.DataFrame(columns=cols_dataset).to_csv(path_dataset, sep=';', index=False)

    df_final = pd.DataFrame([data_dataset])
    df_final.to_csv(path_dataset, sep=';', mode='a', header=False, index=False)

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
            tasks = [
                (dataset) 
                for dataset in datasets
            ]

            pool.map(run_m4, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")
    # run_m4()