import numpy as np
from sklearn.metrics import mean_squared_error as mse
def mase(y_true, y_pred, y_baseline):
    # Calcula o MAE do modelo
    mae_pred = np.mean(np.abs(y_true - y_pred))
    # Calcula o MAE do modelo baseline Persistent Window (i.e., últimas h observações antes do teste)
    mae_naive = np.mean(np.abs(y_true - y_baseline))
    result = mae_pred / mae_naive
    return result


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    mean_y_true = sum(y_true) / len(y_true)
    ss_total = sum((y_true - mean_y_true) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2


def pocid(y_true, y_pred):
    n = len(y_true)
    D = [
        1 if (y_pred[i] - y_pred[i - 1]) * (y_true[i] - y_true[i - 1]) > 0 else 0
        for i in range(1, n)
    ]
    POCID = 100 * np.sum(D) / (n - 1)
    return POCID


def mcpm(rmse_result, mape_result, pocid_result):
    er_result = 100 - pocid_result

    A1 = (rmse_result * mape_result * np.sin((2 * np.pi) / 3)) / 2
    A2 = (mape_result * er_result * np.sin((2 * np.pi) / 3)) / 2
    A3 = (er_result * rmse_result * np.sin((2 * np.pi) / 3)) / 2
    total = A1 + A2 + A3
    return total


# Função para calcular os valores de SMAPE
def calculate_smape(forecasts, test_set):
    smape = 2 * np.abs(forecasts - test_set) / (np.abs(forecasts) + np.abs(test_set))
    smape_per_series = np.nanmean(smape, axis=1)  # Média por série
    return smape_per_series


# Função para calcular os valores de mSMAPE
def calculate_msmape(forecasts, test_set):
    epsilon = 0.1
    comparator = np.full(test_set.shape, 0.5 + epsilon)
    sum_values = np.maximum(
        comparator, (np.abs(forecasts) + np.abs(test_set) + epsilon)
    )
    smape = 2 * np.abs(forecasts - test_set) / sum_values
    msmape_per_series = np.nanmean(smape, axis=1)
    return msmape_per_series


# Função para calcular os valores de MASE
def calculate_mase(forecasts, test_set, training_set, seasonality):
    mase_per_series = []

    for k in range(forecasts.shape[0]):
        te = test_set[k, ~np.isnan(test_set[k, :])]
        tr = training_set[k][~np.isnan(training_set[k])]
        f = forecasts[k, ~np.isnan(forecasts[k, :])]

        # Cálculo de MASE
        mase = MASE(te, f, np.mean(np.abs(np.diff(tr, n=1, axis=0, prepend=tr[0]))))

        if np.isnan(mase):
            mase = MASE(te, f, np.mean(np.abs(np.diff(tr, n=1, axis=0, prepend=tr[0]))))

        mase_per_series.append(mase)

    return np.array(mase_per_series)


# Função para calcular os valores de MAE
def calculate_mae(forecasts, test_set):
    mae = np.abs(forecasts - test_set)
    mae_per_series = np.nanmean(mae, axis=1)
    return mae_per_series


# Função para calcular os valores de RMSE
def calculate_rmse(forecasts, test_set):
    squared_errors = (forecasts - test_set) ** 2
    rmse_per_series = np.sqrt(np.nanmean(squared_errors, axis=1))
    return rmse_per_series
