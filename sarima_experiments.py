#!pip install -U aeon
#!pip install aeon[all_extras]
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import warnings
import pandas as pd
import numpy as np
from aeon.datasets import load_airline
from aeon.forecasting.arima import ARIMA
from matplotlib import pyplot as plt
from aeon.forecasting.arima import AutoARIMA
from aeon.forecasting.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from aeon.visualisation import plot_series
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from aeon.transformations.detrend import ConditionalDeseasonalizer
from aeon.transformations.boxcox import BoxCoxTransformer
from scipy import stats
import os
import matplotlib.pyplot as plt
from mango import scheduler, Tuner
from all_functions import *
import multiprocessing
import traceback
import pickle

warnings.filterwarnings("ignore")

colunas = ['DATA', 'MCPM', 'UF', 'PRODUCT', 'MODEL', 'SAVED MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE', 'R2',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12',
           'Test Statistic', 'p-value', 'Lags Used', 'Observations Used', 'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)', 'Stationary'
           ]
df_result = pd.DataFrame(columns=colunas)
train_v_norm = pd.Series()
format_v = "UNKNOWN"
train_tf_v = pd.Series()
train_v_real = pd.Series()
test_v_real = pd.Series()
horizon = 12
window = 12
format_v = pd.Series()
sarima_order = (0,0,0)


def objfun(params):
    global train_tf_v
    global train_v_real
    global test_v_real
    global horizon
    global format_v
    global train_v_norm
    global sarima_order

    ps, ds, qs, ss = params['P'],params['D'], params['Q'], params['s']
    model = ARIMA(order=sarima_order, 
                seasonal_order=(ps,ds,qs,ss),
                # suppress_warnings=True
                )
    model.fit(train_tf_v)
    predictions = recursive_forecasting_stats(train_tf_v, model, horizon)
    preds_norm = transform_reverse_preds(predictions, train_v_norm, format=format_v)

    _, mean, std = rolling_window_series(train_v_real, window)

    preds_real = znorm_reverse(preds_norm, mean, std)

    rmse_result = rmse(test_v_real, preds_real)
    # mape_result = mape(test_v, preds_real)
    # pocid_result = pocid(test_v, preds_real)
    # mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
    # pbe_result = abs(pbe(test_v_real, preds_real))

    return rmse_result
# @scheduler.parallel(n_jobs=1)
def arima_objective_function(args_list):
    params_evaluated = []
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for params in args_list:
            future = (params,  executor.submit(objfun, params)) 
            futures.append(future)
        for params, future in futures:
            try:
                result = future.result()
                params_evaluated.append(params)
                results.append(result)
            except:
                # print_log(f"Exception raised for {params}")
                continue

    return params_evaluated, results


def find_best_parameter(train, test, train_val_real, train_val_norm, format, order):
    global train_tf_v
    global train_v_real
    global train_v_norm
    global test_v_real
    global format_v
    global sarima_order
    param_space_arima = dict(
                    P = range(2, 15),
                    D = range(0, 2),
                    Q = range(2, 15),
                    s = [12] #12 sazonalidade mensal
                    )
    conf_Dict = dict()
    conf_Dict['num_iteration'] = 30
    train_tf_v = train
    train_v_real = train_val_real
    train_v_norm = train_val_norm
    test_v_real = test
    format_v = format
    sarima_order = order

    tuner = Tuner(param_space_arima, arima_objective_function, conf_Dict)
    results_arima = tuner.minimize()

    return results_arima

def checkFolder(pasta, arquivo, tipo):
    if os.path.exists(pasta):
        caminho_arquivo = os.path.join(pasta, arquivo)
        if os.path.exists(caminho_arquivo):
            df = pd.read_csv(caminho_arquivo, sep=';')
            if 'DATA' in df.columns:
                if not tipo in df['DATA'].values:
                    print_log(f'Continuando... {tipo} em ${pasta}/{arquivo}".')
                    return True
    return False


dirs = [
    '../datasets/venda/mensal/uf/gasolinac/',
    '../datasets/venda/mensal/uf/etanolhidratado/',
    '../datasets/venda/mensal/uf/gasolinadeaviacao/',
    '../datasets/venda/mensal/uf/glp/',
    '../datasets/venda/mensal/uf/oleocombustivel/',
    '../datasets/venda/mensal/uf/oleodiesel/',
    '../datasets/venda/mensal/uf/querosenedeaviacao/',
    '../datasets/venda/mensal/uf/queroseneiluminante/',
]
pickle_file = './pickle/sarima/rolling'
results_file = './results/sarima/rolling'
path_results_arima = './results_arima_rolling'

def process_file(args):
    directory, file = args
    if file.endswith('.csv'):
        try:
            full_path = os.path.join(directory, file)
            df = pd.read_csv(full_path, header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
            df['timestamp']=pd.to_datetime(df['timestamp'], infer_datetime_format=True)
            df = df.set_index('timestamp',inplace=False)
            df.index = df.index.to_period('M')
            all_series_test = []
            series = df['m3']
            train, test = train_test_stats(series, horizon)
            train_val, test_val = train_test_stats(train, horizon)
            # train_val_norm = znorm(train_val)
            # train_norm = znorm(train)
            train_val_norm, _, _ = rolling_window_series(train_val, window)
            train_norm, mean, std = rolling_window_series(train, window)
            
            # test_val_norm = znorm_by(test_val, train_val)
            # test_norm = znorm_by(test, train)

            all_series_test.append(("normal", train_val_norm, train_norm))

            #series sem sazonalidade
            train_val_ds = transform_train(train_val_norm, format="deseasonal")
            train_tf_ds = transform_train(train_norm, format="deseasonal")
            all_series_test.append(("deseasonal", train_val_ds, train_tf_ds))

            #series deseasonal + log transform
            train_val_ds_log = transform_train(train_val_norm, format="deseasonal-log")
            train_tf_ds_log = transform_train(train_norm, format="deseasonal-log")
            all_series_test.append(("deseasonal-log", train_val_ds_log, train_tf_ds_log))

            #series sem sazonalidade e sem tendencia
            train_val_ds_diff = transform_train(train_val_norm, format="deseasonal-diff")
            train_tf_ds_diff = transform_train(train_norm, format="deseasonal-diff")
            all_series_test.append(("deseasonal-diff", train_val_ds_diff, train_tf_ds_diff))

            #series sem tendencia
            train_val_diff = transform_train(train_val_norm, format="diff")
            train_tf_diff = transform_train(train_norm, format="diff")
            all_series_test.append(("diff", train_val_diff, train_tf_diff))

            #series log transform
            train_val_log = transform_train(train_val_norm, format="log")
            train_tf_log = transform_train(train_norm, format="log")
            all_series_test.append(("log", train_val_log, train_tf_log))

            #series log transform + diff
            train_val_log_diff = transform_train(train_val_norm, format="log-diff")
            train_tf_log_diff = transform_train(train_norm, format="log-diff")
            all_series_test.append(("log-diff", train_val_log_diff, train_tf_log_diff))


            uf = file.split("_")[1].upper()
            derivado = file.split("_")[2].split(".")[0]
            path_derivado = f'{results_file}/{derivado}'
            os.makedirs(path_derivado, exist_ok=True)
            csv_path = f'{path_derivado}/transform_{uf}.csv'
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
            for k, train_tf_val, train_tf in all_series_test:
                derivado_result = results_file+"/"+derivado
                flag = checkFolder(derivado_result, f"transform_{uf}.csv", k)
                if flag:
                    print_log(f"{derivado} | {k} em {uf}")
                    arima_order = get_arima_param(path_results_arima, derivado, uf, k)
                    renamed_transform = k.replace("-", "_")
                    
                    results_arima = find_best_parameter(train_tf_val, test_val, train_val, train_val_norm, k, arima_order)
                    print_log(f'----------------------[VALIDACAO] ENCONTRADO PARAMETROS PARA {derivado} | {k} em {uf} ------------------------------')
                    seasonal_order = (results_arima['best_params']['P'], results_arima['best_params']['D'], results_arima['best_params']['Q'])

                    forecast, preds_norm, final_order = fit_sarima_train(train_tf, train_norm, arima_order, seasonal_order, horizon, format=k)

                    preds_real = znorm_reverse(preds_norm, mean, std)

                    rmse_result = rmse(test, preds_real)
                    mape_result = mape(test, preds_real)
                    pocid_result = pocid(test, preds_real)
                    pbe_result = pbe(test, preds_real)
                    r2_result = r2(test, preds_real)
                    mcpm_result = mcpm(rmse_result, mape_result, pocid_result)

                    all_params = final_order | seasonal_order
                    print_log('[RESULTADO EM TRAIN]')
                    print_log(f'PARAMS: {str(all_params)}')
                    print_log(f'MCPM: {mcpm_result}')
                    print_log(f'RMSE: {rmse_result}')
                    print_log(f'MAPE: {mape_result}')
                    print_log(f'POCID: {pocid_result}')
                    print_log(f'PBE: {pbe_result}')

                    print_log(f'---------------------- [FINALIZADO] {derivado} | {k} em {uf} ------------------------------')
                    adfuller_test = analyze_stationarity(train_tf[1:])
                    pkl_file = f"{pickle_file}/{derivado}/{renamed_transform}/{uf}_{renamed_transform}.pkl"
                    df_temp = pd.DataFrame({'DATA': k, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'SARIMA', 'SAVED MODEL': pkl_file, 'PARAMS': str(all_params  ), 'WINDOW': window, 'HORIZON': horizon,  
                                            'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 'R2': r2_result, 
                                            'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                            'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                            'P11': preds_real[10], 'P12': preds_real[11], 'Test Statistic': adfuller_test['Test Statistic'], 'p-value': adfuller_test['p-value'],
                                            'Lags Used': adfuller_test['Lags Used'],  'Observations Used': adfuller_test['Observations Used'], 'Critical Value (1%)': adfuller_test['Critical Value (1%)'],
                                            'Critical Value (5%)': adfuller_test['Critical Value (5%)'], 'Critical Value (10%)': adfuller_test['Critical Value (10%)'], 'Stationary': adfuller_test['Stationary']
                                            }, index=[0])
                    df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)
                    os.makedirs(f'{pickle_file}/{derivado}/{renamed_transform}', exist_ok=True)
                    with open(pkl_file, "wb") as f:
                        pickle.dump(forecast, f) 
        except Exception as e:
            print_log(f"Exception: {derivado} em {uf}\n", e)
            traceback.print_exc()


if __name__ == "__main__":
    with multiprocessing.Pool(processes=2) as pool:
        tasks = [
            (directory, file) 
            for directory in dirs 
            for file in os.listdir(directory)
        ]

        pool.map(process_file, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")