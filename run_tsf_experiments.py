import pandas as pd
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import StatsForecastAutoETS
from darts.models import StatsForecastAutoARIMA
from darts.models import StatsForecastAutoTheta
from darts.models import TBATS
from darts.models import TransformerModel
from darts.models import NaiveSeasonal
from darts.models import NBEATSModel
from mlforecast import MLForecast
from streamfuels.datasets import DatasetLoader
from darts.models import TFTModel
from darts.models import NHiTSModel
from darts.models import NaiveMean
from darts.models import NaiveMovingAverage
from darts.models import RandomForest
from darts.models import NaiveDrift
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.models import CatBoostModel
from sklearn.linear_model import Ridge, RidgeCV
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
import time
from metaforecast.ensembles import MLForecastADE
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
import os
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pywt
import optuna
from catboost import CatBoostRegressor
from all_functions import *
import traceback
from functools import partial
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, MLP

from metaforecast.ensembles import ADE


def objective_optuna(trial, train_val_darts, train_val, transform, test_val):
    try:
        my_stopper = EarlyStopping(
            monitor="train_loss",
            patience=5,
            min_delta=0.05,
            mode="min",
        )
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": [1],
            "callbacks": [my_stopper],
        }
        param = {
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "num_stacks": trial.suggest_int("num_stacks", 5, 20),
            "layer_widths": trial.suggest_categorical(
                "layer_widths", [64, 128, 256, 512]
            ),
        }
        model = NBEATSModel(
            **param,
            input_chunk_length=12,
            output_chunk_length=12,
            random_state=42,
            n_epochs=100,
            pl_trainer_kwargs=pl_trainer_kwargs,
            activation="ReLU",
        )
        model.fit(train_val_darts)

        # transformados
        # predictions = recursive_step(X_test_v, train_original, model, horizon, window, format_v, representation, wavelet, level)
        # preds_real = pd.Series(predictions, index=test_val.index)

        result = model.predict(n=12)
        preds_norm = pd.Series(result.values().flatten().tolist(), index=test_val.index)

        # para modelos estatisticos
        preds_real = reverse_transform_norm_preds(preds_norm, train_val, transform)
        print_log("test_val")
        print_log(test_val)
        print_log("PREDS")
        print_log(preds_real)
        mape_result = mape(test_val, preds_real)
    except Exception as e:
        print(e)
        return float("inf")

    return mape_result


def find_best_parameter_optuna(train_val_darts, train_val, transform, test_val):
    objective_function = partial(
        objective_optuna,
        train_val_darts=train_val_darts,
        train_val=train_val,
        transform=transform,
        test_val=test_val,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_function, n_trials=35)

    return study.best_params


def custom_year_encoder(idx):
    return (idx.year - 1990) + (idx.month - 1) / 12


def generate_error_series(
    model, n_partes, train_tf, train, test, horizon, window, transform, regr="ridge"
):
    min_train_size = horizon * 2
    all_preds = pd.Series(dtype=float)
    all_test = pd.Series(dtype=float)
    all_errors = pd.Series(dtype=float)
    # from Statistics_and_Seq2Seq.all_functions import *
    representation, wavelet, level = "MTF", "bior2.2", 1

    for i in range(n_partes):
        val_end = len(train_tf) - i * horizon
        val_start = val_end - horizon
        train_end = val_start

        train_tf_val = train_tf.iloc[:train_end]
        train_val = train.iloc[:train_end]
        test_val = train.iloc[val_start:val_end]

        if len(train) < min_train_size:
            print(
                f"Iteração {i + 1}: Conjunto de treino está menor que o tamanho mínimo. Parando."
            )
            break

        train_tf_val_darts = TimeSeries.from_series(train_tf_val)

        model.fit(train_tf_val_darts)

        result_val = model.predict(n=horizon)
        preds_val_norm = pd.Series(
            result_val.values().flatten().tolist(), index=test_val.index
        )

        preds_val_real = reverse_transform_norm_preds(
            preds_val_norm, train_val, transform
        )
        all_preds = pd.concat([all_preds, preds_val_real])
        all_preds = all_preds.sort_index()

        all_test = pd.concat([all_test, test_val])
        all_test = all_test.sort_index()

        all_errors = all_test - all_preds

    all_errors.index = all_errors.index.to_period("M")
    # data = rolling_window_image(pd.concat([all_errors, pd.Series([0] * horizon, index=test_val.index)]), window, representation, wavelet, level)
    data = rolling_window(
        pd.concat([all_errors, pd.Series([0] * horizon, index=test_val.index)]), window
    )
    data = data.dropna()
    mean_data_error = np.mean(data)
    std_data_error = np.mean(np.std(data))
    if std_data_error == 0:
        return pd.Series([0] * horizon, index=test.index)

    X_train, X_test, y_train, _ = train_test_split(data, horizon)
    if regr == "ridge":
        results_rg = {"alphas": np.logspace(-3, 3, 10)}
        rg = RidgeCV(**results_rg)
        rg.fit(X_train, y_train)
    elif regr == "catboost":
        rg = CatBoostRegressor(
            **{
                "iterations": 200,
                "learning_rate": 0.01018185095858352,
                "depth": 8,
                "loss_function": "RMSE",
                "random_state": 42,
            }
        )
        rg.fit(X_train, y_train, verbose=False)

    # train_val2 = train_val
    # train_val2.index = train_val2.index.to_period('M')
    # predictions_error = recursive_2(X_test, all_errors, rg, horizon, window, transform, representation, wavelet, level)
    # preds_real_error = pd.Series(predictions_error, index=test.index)

    predictions_error = recursive_multistep_forecasting(X_test, rg, horizon)
    preds = pd.Series(predictions_error, index=test.index)
    preds_real_error = reverse_regressors(all_errors, preds, window, format=transform)

    return preds_real_error


def generate_weights_series(index, exp_name, model, train, test, horizon, transform):
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

    min_train_size = (horizon * 3) + (horizon * 15)
    train_test_splits = []
    aux_series = train
    path_experiments = f"./timeseries/mestrado/ponderacao/"
    path_csv = f"{path_experiments}/{exp_name}.csv"

    while len(aux_series) > horizon + min_train_size:
        train_val, test_val = aux_series[:-horizon], aux_series[-horizon:]
        train_test_splits.append((train_val, test_val))
        aux_series = train_val

    for train_val, test_val in train_test_splits:

        print(
            f"Iteração: Data range do conjunto de treino: {train_val.index[0]} - {train_val.index[-1]}"
        )
        train_tf_val, _, _ = rolling_window_series(train_val, horizon)

        train_tf_val_darts = TimeSeries.from_series(train_tf_val)
        model.fit(train_tf_val_darts)

        result_val = model.predict(n=horizon)
        preds_val_norm = pd.Series(
            result_val.values().flatten().tolist(), index=test_val.index
        )

        preds_val_real = reverse_transform_norm_preds(
            preds_val_norm, train_val, transform
        )

        preds_real_array = np.array(preds_val_real.values)
        preds_real_reshaped = preds_real_array.reshape(1, -1)
        test_reshaped = test_val.values.reshape(1, -1)
        smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
        # print(smape_result)
        rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
        msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
        # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
        mae_result = calculate_mae(preds_real_reshaped, test_reshaped)
        mape_result = mape(test_val.values, preds_real_array)
        pocid_result = pocid(test_val.values, preds_real_array)

        start_test = test_val.index.tolist()[0]
        final_test = test_val.index.tolist()[-1]

        data_serie = {
            "dataset_index": f"{index}",
            "horizon": horizon,
            "regressor": exp_name,
            "mape": mape_result,
            "pocid": pocid_result,
            "smape": smape_result,
            "rmse": rmse_result,
            "msmape": msmape_result,
            "mae": mae_result,
            "test": [test_val.tolist()],
            "predictions": [preds_val_real.values],
            "start_test": start_test,
            "final_test": final_test,
        }

        if not os.path.exists(path_csv):
            pd.DataFrame(columns=cols_serie).to_csv(path_csv, sep=";", index=False)

        df_new = pd.DataFrame(data_serie)
        df_new.to_csv(path_csv, sep=";", mode="a", header=False, index=False)


def run_tsf_file(tsf_file, dataset_name, regr):
    loader = DatasetLoader()

    # df, frequency, horizon, contain_missing_values, contain_equal_length
    df, metadata = loader.read_tsf(path_tsf=tsf_file)

    frequency = metadata["frequency"]
    horizon = metadata["horizon"]
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
        "training_time",
        "prediction_time",
    ]
    cols_dataset = [
        "dataset",
        "regressor",
        "mape_mean",
        "pocid_mean",
        "smape_mean",
        "rmse_mean",
        "msmape_mean",
        "mae_mean",
        "all_training_time",
        "all_prediction_time",
    ]

    transform = "normal"
    print(regr)
    # representation = ""
    # regr = "ARIMA"
    dataset = dataset_name
    exp_name = f"{regr}_{transform}"
    path_experiments = f"./timeseries/mestrado/models/{regr}/{transform}/"
    path_csv = f"{path_experiments}/{dataset}.csv"
    os.makedirs(path_experiments, exist_ok=True)

    rmses, smapes, msmapes, maes, mapes, pocids = [], [], [], [], [], []
    all_train_time = 0
    all_pred_time = 0
    for i in range(len(df)):
        isProb = False
        times = []
        series_value = df.iloc[i]["series_value"].tolist()
        start_timestamp = df.iloc[i]["start_timestamp"]

        freq = "ME" if frequency == "monthly" else "Y"

        index_series = pd.date_range(
            start=start_timestamp, periods=len(series_value), freq=freq
        )
        pd_series = pd.Series(series_value, index=index_series)
        # series = TimeSeries.from_series(pd_series)

        train, test = pd_series[:-horizon], pd_series[-horizon:]

        # normalizacao para estatisticos
        train_tf, _, _ = rolling_window_series(train, horizon)

        train_darts = TimeSeries.from_series(train_tf)

        add_encoders = {
            "cyclic": {"future": ["month"]},
            "datetime_attribute": {"future": ["year"]},
            "position": {"past": ["relative"], "future": ["relative"]},
            "custom": {"past": [custom_year_encoder]},
            # 'transformer': Scaler(),
            # 'tz': 'CET',
        }

        if regr == "ARIMA":
            model = StatsForecastAutoARIMA(season_length=12)
        elif regr == "ETS":
            model = StatsForecastAutoETS(season_length=12)
        elif regr == "THETA":
            model = StatsForecastAutoTheta(season_length=12)
        elif regr == "NaiveSeasonal":
            model = NaiveSeasonal(K=12)
        elif regr == "NaiveMean":
            model = NaiveMean()
        elif regr == "NaiveDrift":
            model = NaiveDrift()
        elif regr == "NaiveMovingAverage":
            model = NaiveMovingAverage(input_chunk_length=horizon)
        elif regr == "TBATS":
            model = TBATS(use_arma_errors=True, use_trend=True, seasonal_periods=[12])
        elif regr == "RandomForest":
            model = RandomForest(
                lags=horizon,
                use_static_covariates=False,
                n_estimators=200,
                output_chunk_length=horizon,
            )
        elif regr == "CatBoost":
            model = CatBoostModel(
                lags=horizon, output_chunk_length=horizon, use_static_covariates=False
            )
        elif regr == "NBEATS":

            my_stopper = EarlyStopping(
                monitor="train_loss",
                patience=5,
                min_delta=0.05,
                mode="min",
            )

            pl_trainer_kwargs = {
                "accelerator": "gpu",
                "devices": [0],
                "callbacks": [my_stopper],
            }
            model = NBEATSModel(
                input_chunk_length=horizon * 2,
                output_chunk_length=horizon,
                n_epochs=100,
                activation="ReLU",
                layer_widths=512,
                num_stacks=10,
                num_blocks=3,
                num_layers=4,
                random_state=42,
                nr_epochs_val_period=1,
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

        elif regr == "Transformer":
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
            model = TransformerModel(
                input_chunk_length=horizon,
                output_chunk_length=horizon,
                n_epochs=15,
                pl_trainer_kwargs=pl_trainer_kwargs,
            )
        elif regr == "TFT":
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
            model = TFTModel(
                input_chunk_length=horizon,
                output_chunk_length=horizon,
                n_epochs=15,
                pl_trainer_kwargs=pl_trainer_kwargs,
                add_encoders=add_encoders,
            )
            isProb = True
        elif regr == "NHiTS":
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
            model = NHiTSModel(
                input_chunk_length=horizon,
                output_chunk_length=horizon,
                num_blocks=2,
                n_epochs=15,
                pl_trainer_kwargs=pl_trainer_kwargs,
            )
        # future_cov = datetime_attribute_timeseries(train_darts, "month", cyclic=True, add_length=horizon)
        time_training_start = time.perf_counter()

        model.fit(train_darts)
        time_training_end = time.perf_counter()
        times.append(time_training_end - time_training_start)
        all_train_time = all_train_time + (time_training_end - time_training_start)

        time_preds_start = time.perf_counter()
        if isProb:
            result = model.predict(n=horizon, num_samples=100)
        else:
            result = model.predict(n=horizon)
        preds_norm = pd.Series(result.values().flatten().tolist(), index=test.index)

        # para modelos estatisticos
        preds_real = reverse_transform_norm_preds(preds_norm, train, transform)
        time_preds_end = time.perf_counter()
        times.append(time_preds_end - time_preds_start)
        all_pred_time = all_pred_time + (time_preds_end - time_preds_start)

        preds_real_array = np.array(preds_real.values)
        preds_real_reshaped = preds_real_array.reshape(1, -1)
        test_reshaped = test.values.reshape(1, -1)
        smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
        # print(smape_result)
        rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
        msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
        # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
        mae_result = calculate_mae(preds_real_reshaped, test_reshaped)
        mape_result = mape(test.values, preds_real_array)
        pocid_result = pocid(test.values, preds_real_array)

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
            "predictions": [preds_real.values],
            "training_time": times[0],
            "prediction_time": times[1],
        }

        if not os.path.exists(path_csv):
            pd.DataFrame(columns=cols_serie).to_csv(path_csv, sep=";", index=False)

        df_new = pd.DataFrame(data_serie)
        df_new.to_csv(path_csv, sep=";", mode="a", header=False, index=False)

        maes.append(mae_result)
        rmses.append(rmse_result)
        msmapes.append(msmape_result)
        smapes.append(smape_result)
        mapes.append(mape_result)
        pocids.append(pocid_result)

        generate_weights_series(
            i,
            exp_name,
            model=model,
            train=train,
            test=test,
            horizon=horizon,
            transform=transform,
        )

    data_dataset = {
        "dataset": dataset,
        "regressor": exp_name,
        "mape_mean": np.nanmean(mapes),
        "pocid_mean": np.nanmean(pocids),
        "smape_mean": np.nanmean(smapes),
        "rmse_mean": np.nanmean(rmses),
        "msmape_mean": np.nanmean(msmapes),
        "mae_mean": np.nanmean(maes),
        "all_training_time": all_train_time,
        "all_prediction_time": all_pred_time,
    }
    path_dataset = f"./timeseries/" + "results.csv"
    if not os.path.exists(path_dataset):
        pd.DataFrame(columns=cols_dataset).to_csv(path_dataset, sep=";", index=False)

    df_final = pd.DataFrame([data_dataset])
    df_final.to_csv(path_dataset, sep=";", mode="a", header=False, index=False)


def run_error_tsf_file(tsf_file, dataset_name):
    loader = DatasetLoader()

    # df, frequency, horizon, contain_missing_values, contain_equal_length
    df, metadata = loader.read_tsf(path_tsf=tsf_file)

    frequency = metadata["frequency"]
    horizon = metadata["horizon"]
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
        "training_time",
        "prediction_time",
    ]
    cols_dataset = [
        "dataset",
        "regressor",
        "mape_mean",
        "pocid_mean",
        "smape_mean",
        "rmse_mean",
        "msmape_mean",
        "mae_mean",
        "all_training_time",
        "all_prediction_time",
    ]

    transform = "normal"
    # representation = ""
    regr = "ETS"
    dataset = dataset_name
    exp_name = f"{regr}_{transform}"
    path_experiments = f"./timeseries/mestrado/combination/normal_mtf_{regr}_error/"
    path_csv = f"{path_experiments}/{dataset}.csv"
    os.makedirs(path_experiments, exist_ok=True)

    rmses, smapes, msmapes, maes, mapes, pocids = [], [], [], [], [], []
    all_train_time = 0
    all_pred_time = 0
    for i in range(len(df)):
        isProb = False
        times = []
        series_value = df.iloc[i]["series_value"].tolist()
        start_timestamp = df.iloc[i]["start_timestamp"]

        freq = "ME" if frequency == "monthly" else "Y"

        index_series = pd.date_range(
            start=start_timestamp, periods=len(series_value), freq=freq
        )
        pd_series = pd.Series(series_value, index=index_series)
        # series = TimeSeries.from_series(pd_series)

        train, test = pd_series[:-horizon], pd_series[-horizon:]

        # normalizacao para estatisticos
        train_tf, _, _ = rolling_window_series(train, horizon)

        train_darts = TimeSeries.from_series(train_tf)

        add_encoders = {
            "cyclic": {"future": ["month"]},
            "datetime_attribute": {"future": ["year"]},
            "position": {"past": ["relative"], "future": ["relative"]},
            "custom": {"past": [custom_year_encoder]},
            # 'transformer': Scaler(),
            # 'tz': 'CET',
        }

        if regr == "ARIMA":
            model = StatsForecastAutoARIMA(season_length=12)
        elif regr == "ETS":
            model = StatsForecastAutoETS(season_length=12)
        elif regr == "THETA":
            model = StatsForecastAutoTheta(season_length=12)
        elif regr == "NaiveSeasonal":
            model = NaiveSeasonal(K=12)
        elif regr == "NaiveMean":
            model = NaiveMean()
        elif regr == "NaiveDrift":
            model = NaiveDrift()
        elif regr == "NaiveMovingAverage":
            model = NaiveMovingAverage(input_chunk_length=horizon)
        elif regr == "TBATS":
            model = TBATS()

        elif regr == "CatBoost":
            model = CatBoostModel(
                lags=12, output_chunk_length=horizon, use_static_covariates=False
            )
        elif regr == "NBEATS":

            my_stopper = EarlyStopping(
                monitor="train_loss",
                patience=5,
                min_delta=0.05,
                mode="min",
            )

            pl_trainer_kwargs = {
                "accelerator": "gpu",
                "devices": [0],
                "callbacks": [my_stopper],
            }
            model = NBEATSModel(
                input_chunk_length=horizon * 2,
                output_chunk_length=horizon,
                n_epochs=100,
                activation="ReLU",
                layer_widths=512,
                num_stacks=10,
                num_blocks=3,
                num_layers=4,
                random_state=42,
                nr_epochs_val_period=1,
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

        elif regr == "Transformer":
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
            model = TransformerModel(
                input_chunk_length=horizon,
                output_chunk_length=horizon,
                n_epochs=15,
                pl_trainer_kwargs=pl_trainer_kwargs,
            )
        elif regr == "TFT":
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
            model = TFTModel(
                input_chunk_length=horizon,
                output_chunk_length=horizon,
                n_epochs=15,
                pl_trainer_kwargs=pl_trainer_kwargs,
                add_encoders=add_encoders,
            )
            isProb = True
        elif regr == "NHiTS":
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
            model = NHiTSModel(
                input_chunk_length=horizon,
                output_chunk_length=horizon,
                num_blocks=2,
                n_epochs=15,
                pl_trainer_kwargs=pl_trainer_kwargs,
            )
        # future_cov = datetime_attribute_timeseries(train_darts, "month", cyclic=True, add_length=horizon)
        time_training_start = time.perf_counter()

        # previsão do modelo com base nos anos anteriores
        preds_real_error = generate_error_series(
            regr="catboost",
            model=model,
            n_partes=10,
            train_tf=train_tf,
            train=train,
            test=test,
            horizon=horizon,
            window=12,
            transform=transform,
        )

        # fit no modelo normal para previsao ser combinada com preds_real_error
        model.fit(train_darts)
        time_training_end = time.perf_counter()
        times.append(time_training_end - time_training_start)
        all_train_time = all_train_time + (time_training_end - time_training_start)

        time_preds_start = time.perf_counter()
        if isProb:
            result_model_test = model.predict(n=horizon, num_samples=100)
        else:
            result_model_test = model.predict(n=horizon)
        preds_model_norm = pd.Series(
            result_model_test.values().flatten().tolist(), index=test.index
        )

        preds_model_real = reverse_transform_norm_preds(
            preds_model_norm, train, transform
        )
        preds_combined = preds_model_real + preds_real_error

        time_preds_end = time.perf_counter()
        times.append(time_preds_end - time_preds_start)
        all_pred_time = all_pred_time + (time_preds_end - time_preds_start)

        # predicao combinada para metrica
        preds_real_array = np.array(preds_combined.values)

        preds_real_reshaped = preds_real_array.reshape(1, -1)
        test_reshaped = test.values.reshape(1, -1)
        smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
        # print(smape_result)
        rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
        msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
        # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
        mae_result = calculate_mae(preds_real_reshaped, test_reshaped)
        mape_result = mape(test.values, preds_real_array)
        pocid_result = pocid(test.values, preds_real_array)

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
            "predictions": [preds_combined.values],
            "training_time": times[0],
            "prediction_time": times[1],
        }

        if not os.path.exists(path_csv):
            pd.DataFrame(columns=cols_serie).to_csv(path_csv, sep=";", index=False)

        df_new = pd.DataFrame(data_serie)
        df_new.to_csv(path_csv, sep=";", mode="a", header=False, index=False)

        maes.append(mae_result)
        rmses.append(rmse_result)
        msmapes.append(msmape_result)
        smapes.append(smape_result)
        mapes.append(mape_result)
        pocids.append(pocid_result)

    data_dataset = {
        "dataset": dataset,
        "regressor": exp_name,
        "mape_mean": np.nanmean(mapes),
        "pocid_mean": np.nanmean(pocids),
        "smape_mean": np.nanmean(smapes),
        "rmse_mean": np.nanmean(rmses),
        "msmape_mean": np.nanmean(msmapes),
        "mae_mean": np.nanmean(maes),
        "all_training_time": all_train_time,
        "all_prediction_time": all_pred_time,
    }
    path_dataset = f"./timeseries/" + "results.csv"
    if not os.path.exists(path_dataset):
        pd.DataFrame(columns=cols_dataset).to_csv(path_dataset, sep=";", index=False)

    df_final = pd.DataFrame([data_dataset])
    df_final.to_csv(path_dataset, sep=";", mode="a", header=False, index=False)


def run_darts_series(args):
    frequency, horizon, line, i, regressor, dataset = args
    global regr
    # horizon = 12
    window = horizon
    regr = regressor
    transformations = ["normal"]
    chave = ""

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
    # dataset = "ANP_MONTHLY"

    train_test_splits = []

    series_value = line["series_value"].tolist()
    start_timestamp = line["start_timestamp"]

    # freq = "M" if frequency == "monthly" else "Y"
    freq_map = {
        "yearly": "Y",
        "monthly": "M",
        "weekly": "W",
        "daily": "D",
        "hourly": "H",
        "half_hourly": "30min",  # 30T
    }

    freq = freq_map.get(frequency)
    if freq is None:
        raise ValueError(f"Frequência desconhecida: {frequency}")

    index_series = pd.date_range(
        start=start_timestamp, periods=len(series_value), freq=freq
    )
    series = pd.Series(series_value, index=index_series)
    isProb = False

    aux_series = series
    max_k = 4
    min_train_factor = 3

    qtd_series = horizon * max_k

    while qtd_series >= len(aux_series) or (len(aux_series) - qtd_series) < (
        horizon * min_train_factor
    ):
        qtd_series -= horizon
        if qtd_series < horizon:
            raise ValueError(
                "Série muito curta para satisfazer as condições com esse horizon."
            )

    min_train_size = len(aux_series) - qtd_series

    splits_realizados = 0
    while len(aux_series) >= min_train_size + horizon and splits_realizados < max_k:
        train, test = aux_series[:-horizon], aux_series[-horizon:]
        train_test_splits.append((train, test))
        aux_series = train
        splits_realizados += 1

    for train, test in train_test_splits:
        train_stl = train
        _, test_val = train_test_stats(train, horizon)
        if "noresid" in chave:
            print_log("----------- SEM RESIDUO NA SERIE ---------")
            transformer = STLTransformer(sp=12)
            stl = transformer.fit(train)
            train_stl = stl.seasonal_ + stl.trend_

        train_val, _ = train_test_stats(train_stl, horizon)
        start_test = test.index.tolist()[0]
        final_test = test.index.tolist()[-1]

        for transform in transformations:
            # path_derivado = f'{results_file}/{derivado}/{transform}'
            # flag = checkFolder(path_derivado, f"transform_{uf}.csv", test_range)
            exp_name = f"{regr}_{transform}"
            path_experiments = f"./timeseries/mestrado/resultados/{regr}/{transform}/"
            path_csv = f"{path_experiments}/{dataset}.csv"
            os.makedirs(path_experiments, exist_ok=True)
            flag = True
            start_exp = time.perf_counter()
            if flag:
                train_tf = transform_regressors(train_stl, transform)
                train_tf, _, _ = rolling_window_series(train_tf, horizon)
                train_val_tf = transform_regressors(train_val, transform)
                train_val_tf, _, _ = rolling_window_series(train_val_tf, horizon)
                # train_tf_val = transform_regressors(train_val, format=transform)

                train_darts = TimeSeries.from_series(train_tf)
                train_val_darts = TimeSeries.from_series(train_val_tf)

                try:
                    if regr == "ARIMA":
                        model = StatsForecastAutoARIMA(season_length=12)
                    elif regr == "ETS":
                        model = StatsForecastAutoETS(season_length=12)
                    elif regr == "THETA":
                        model = StatsForecastAutoTheta(season_length=12)
                    elif regr == "NaiveSeasonal":
                        model = NaiveSeasonal(K=12)
                    elif regr == "NaiveMean":
                        model = NaiveMean()
                    elif regr == "NaiveDrift":
                        model = NaiveDrift()
                    elif regr == "NaiveMovingAverage":
                        model = NaiveMovingAverage(input_chunk_length=horizon)
                    elif regr == "TBATS":
                        model = TBATS(
                            use_arma_errors=True, use_trend=True, seasonal_periods=[12]
                        )
                    elif regr == "RandomForest":
                        model = RandomForest(
                            lags=horizon,
                            use_static_covariates=False,
                            n_estimators=200,
                            output_chunk_length=horizon,
                        )
                    elif regr == "CatBoost":
                        model = CatBoostModel(
                            lags=horizon,
                            output_chunk_length=horizon,
                            use_static_covariates=False,
                        )
                    elif regr == "NBEATS":

                        my_stopper = EarlyStopping(
                            monitor="train_loss",
                            patience=5,
                            min_delta=0.05,
                            mode="min",
                        )

                        pl_trainer_kwargs = {
                            "accelerator": "gpu",
                            "devices": [0],
                            "callbacks": [my_stopper],
                        }

                        result_params = find_best_parameter_optuna(
                            train_val_darts, train_val, transform, test_val
                        )

                        model = NBEATSModel(
                            **result_params,
                            input_chunk_length=horizon,
                            output_chunk_length=horizon,
                            random_state=42,
                            n_epochs=100,
                            pl_trainer_kwargs=pl_trainer_kwargs,
                            activation="ReLU",
                        )

                    elif regr == "Transformer":
                        pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
                        model = TransformerModel(
                            input_chunk_length=horizon,
                            output_chunk_length=horizon,
                            n_epochs=15,
                            pl_trainer_kwargs=pl_trainer_kwargs,
                        )
                    elif regr == "TFT":
                        pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
                        model = TFTModel(
                            input_chunk_length=horizon,
                            output_chunk_length=horizon,
                            n_epochs=15,
                            pl_trainer_kwargs=pl_trainer_kwargs,
                            # add_encoders=add_encoders
                            use_static_covariates=False,
                        )
                        isProb = True
                    elif regr == "NHiTS":
                        pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
                        model = NHiTSModel(
                            input_chunk_length=horizon,
                            output_chunk_length=horizon,
                            num_blocks=2,
                            n_epochs=15,
                            pl_trainer_kwargs=pl_trainer_kwargs,
                        )
                    # future_cov = datetime_attribute_timeseries(train_darts, "month", cyclic=True, add_length=horizon)
                    model.fit(train_darts)

                    if isProb:
                        result = model.predict(n=horizon, num_samples=100)
                    else:
                        result = model.predict(n=horizon)
                    preds_norm = pd.Series(
                        result.values().flatten().tolist(), index=test.index
                    )

                    # para modelos estatisticos
                    preds_real = reverse_transform_norm_preds(
                        preds_norm, train, transform
                    )

                    preds_real_array = np.array(preds_real.values)
                    preds_real_reshaped = preds_real_array.reshape(1, -1)
                    test_reshaped = test.values.reshape(1, -1)
                    smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
                    # print(smape_result)
                    rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
                    msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
                    # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
                    mae_result = calculate_mae(preds_real_reshaped, test_reshaped)
                    mape_result = mape(test.values, preds_real.values)
                    pocid_result = pocid(test.values, preds_real_array)

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
                        "predictions": [preds_real.tolist()],
                        "start_test": start_test,
                        "final_test": final_test,
                        # 'training_time': times[0],
                        # 'prediction_time': times[1],
                    }

                    if not os.path.exists(path_csv):
                        pd.DataFrame(columns=cols_serie).to_csv(
                            path_csv, sep=";", index=False
                        )

                    df_new = pd.DataFrame(data_serie)
                    df_new.to_csv(
                        path_csv, sep=";", mode="a", header=False, index=False
                    )

                except Exception as e:
                    print_log(f" ==== DATASET {i}")
                    traceback.print_exc()


import time
import multiprocessing

if __name__ == "__main__":
    # start = time.perf_counter()
    # file_path = '../mes_11_venda_mensal.tsf'
    # df["series_value"] = df["series_value"].apply(np.array)
    # def should_remove(series, window_size=24):
    #     for i in range(0, len(series), window_size):
    #         window = series[i : i + window_size]
    #         if (window == 0).mean() > 0.5:
    #             return True

    #     return False

    # mask = df["series_value"].apply(should_remove)
    # df = df[~mask]
    files = [
        # "m4_daily_dataset.tsf",
        "m4_hourly_dataset.tsf",
        "m4_weekly_dataset.tsf",
        "nn5_daily_dataset_without_missing_values.tsf",
        "nn5_weekly_dataset.tsf",
        "pedestrian_counts_dataset.tsf",
        "us_births_dataset.tsf",
        "australian_electricity_demand_dataset.tsf",
        # "traffic_hourly_dataset.tsf",
        # "traffic_weekly_dataset.tsf",
    ]

    for tsf_file in files:
        dataset = tsf_file.split(".")[0].upper()
        file_path = f"../forecasting_datasets/{tsf_file}"
        loader = DatasetLoader()
        df, metadata = loader.read_tsf(path_tsf=file_path)

        if metadata["horizon"] == None:
            if metadata["frequency"] == "hourly":
                metadata["horizon"] = 24
            elif metadata["frequency"] == "daily":
                metadata["horizon"] = 14
            elif metadata["frequency"] == "half_hourly":
                metadata["horizon"] = 48

        df["series_value"] = df["series_value"].apply(np.array)

        frequency = metadata["frequency"]
        horizon = metadata["horizon"]
        regr = "THETA"

        # df.iloc[i]
        def run_wrapper(args):
            # frequency, horizon, line, i = args
            # run_ade_series(args)
            run_darts_series(args)

        tasks = [
            (frequency, horizon, df.iloc[i], i, regr, dataset) for i in range(len(df))
        ]

        with multiprocessing.Pool(processes=1) as pool:
            pool.map(run_wrapper, tasks)

    print_log(
        "--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------"
    )
