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

colunas = ['DATA', 'MCPM', 'UF', 'PRODUCT', 'MODEL', 'SAVED MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12'
           ]
df_result = pd.DataFrame(columns=colunas)
train_v_norm = pd.Series()
format_v = "UNKNOWN"
train_v = pd.Series()
test_v = pd.Series()
horizon = 12
format_v = pd.Series()


def objfun(params):
    p,d,q = params['p'],params['d'], params['q']
    model = ARIMA(order=(p,d,q), 
                #   seasonal_order=(pp,dd,qq,ss),
                suppress_warnings=True
                )
    model.fit(train_v)
    predictions = recursive_forecasting_stats(train_v, model, horizon)
    preds_norm = transform_reverse_preds(predictions, train_v_norm, format=format_v)
    rmse_result = rmse(test_v, preds_norm)
    mape_result = mape(test_v, preds_norm)
    pocid_result = pocid(test_v, preds_norm)
    mcpm_result = mcpm(rmse_result, mape_result, pocid_result)

    return mcpm_result
# @scheduler.parallel(n_jobs=1)
def arima_objective_function(args_list):
    global train_v
    global test_v
    global horizon
    global format_v
    global train_v_norm
    params_evaluated = []
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
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


def find_best_parameter(train, test, train_val_norm, format):
    global train_v
    global test_v
    global train_v_norm
    global format_v
    param_space_arima = dict(
                    p = range(2, 15),
                    d = range(0, 1),
                    q = range(2, 15))
    conf_Dict = dict()
    conf_Dict['num_iteration'] = 30
    train_v = train
    test_v = test
    format_v = format
    train_v_norm = train_val_norm
    tuner = Tuner(param_space_arima, arima_objective_function, conf_Dict)
    results_arima = tuner.minimize()

    return results_arima


dirs = [
    './datasets/venda/mensal/uf/gasolinac/',
    './datasets/venda/mensal/uf/etanolhidratado/',
    './datasets/venda/mensal/uf/gasolinadeaviacao/',
    './datasets/venda/mensal/uf/glp/',
    './datasets/venda/mensal/uf/oleocombustivel/',
    './datasets/venda/mensal/uf/oleodiesel/',
    './datasets/venda/mensal/uf/querosenedeaviacao/',
    './datasets/venda/mensal/uf/queroseneiluminante/',
]

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
            train_val_norm = znorm(train_val)
            train_norm = znorm(train)
            
            test_val_norm = znorm_by(test_val, train_val)
            test_norm = znorm_by(test, train)

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
            path_derivado = f'./results/{derivado}'
            os.makedirs(path_derivado, exist_ok=True)
            csv_path = f'{path_derivado}/transform_{uf}.csv'
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
            for k, train_val, train_tf in all_series_test:
                print_log(f"{derivado} | {k} em {uf}")
                renamed_transform = k.replace("-", "_")
                
                results_arima = find_best_parameter(train_val, test_val_norm, train_val_norm, k)
                print_log(f'----------------------[VALIDACAO] ENCONTRADO PARAMETROS PARA {derivado} | {k} em {uf} ------------------------------')
                initial_order = (results_arima['best_params']['p'], results_arima['best_params']['d'], results_arima['best_params']['q'])
                forecast, preds_norm, final_order = fit_arima_train(train_tf, train_norm, initial_order, horizon, format=k)
                # forecast = ARIMA(order_arima, suppress_warnings=True)
                # forecast.fit(train_tf)
                # preds = recursive_forecasting_stats(train_tf, forecast, horizon)
                # preds_norm = transform_reverse_preds(preds, train_norm, format=k)
                rmse_result = rmse(test_norm, preds_norm)
                mape_result = mape(test_norm, preds_norm)
                pocid_result = pocid(test_norm, preds_norm)
                pbe_result = pbe(test_norm, preds_norm)
                mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                print_log('[RESULTADO EM TRAIN]')
                print_log(f'PARAMS: {str(final_order)}')
                print_log(f'MCPM: {mcpm_result}')
                print_log(f'RMSE: {rmse_result}')
                print_log(f'MAPE: {mape_result}')
                print_log(f'POCID: {pocid_result}')
                print_log(f'PBE: {pbe_result}')

                print_log(f'---------------------- [FINALIZADO] {derivado} | {k} em {uf} ------------------------------')
                pkl_file = f"./pickle_arima/{derivado}/{renamed_transform}/{uf}_{renamed_transform}.pkl"
                df_temp = pd.DataFrame({'DATA': k, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ARIMA', 'SAVED MODEL': pkl_file, 'PARAMS': str(final_order), 'WINDOW': '-', 'HORIZON': horizon,  
                                        'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 
                                        'P1': preds_norm[0], 'P2': preds_norm[1], 'P3': preds_norm[2], 'P4': preds_norm[3], 'P5': preds_norm[4],
                                        'P6': preds_norm[5], 'P7': preds_norm[6], 'P8': preds_norm[7], 'P9': preds_norm[8], 'P10': preds_norm[9],
                                        'P11': preds_norm[10], 'P12': preds_norm[11]
                                        }, index=[0])
                df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)
                os.makedirs(f'./pickle_arima/{derivado}/{renamed_transform}', exist_ok=True)
                with open(pkl_file, "wb") as f:
                    pickle.dump(forecast, f) 

        except Exception as e:
            print_log(f"Exception: {derivado} em {uf}\n", e)
            traceback.print_exc()


if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        tasks = [
            (directory, file) 
            for directory in dirs 
            for file in os.listdir(directory) 
            if file.endswith('.csv')
        ]

        pool.map(process_file, tasks)
