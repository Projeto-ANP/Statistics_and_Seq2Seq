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
from scipy import stats
import os
import matplotlib.pyplot as plt
from mango import scheduler, Tuner
import statsmodels.api as sm
from all_functions import *
import multiprocessing
import traceback
import pickle

warnings.filterwarnings("ignore")

colunas = ['DATA', 'MCPM', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12',
           'Test Statistic', 'p-value', 'Lags Used', 'Observations Used', 'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)', 'Stationary'
           ]
df_result = pd.DataFrame(columns=colunas)
format_v = "UNKNOWN"
train_tf_v = pd.Series()
train_v_real = pd.Series()
test_v_real = pd.Series()
horizon = 12
window = 12
# format_v = pd.Series()


def checkFolder(pasta, arquivo, tipo):
    if not os.path.exists(pasta):
        return True
    
    caminho_arquivo = os.path.join(pasta, arquivo)
    try:
        if not os.path.exists(caminho_arquivo):
            return True
        
        df = pd.read_csv(caminho_arquivo, sep=';')
        
        if 'test_range' not in df.columns:
            return True
        
        if tipo not in df['test_range'].values:
            print_log(f'Continuando... {tipo} em "{pasta}/{arquivo}".')
            return True
    except Exception as e:
        print_log(f"Erro em: {caminho_arquivo} | {e}")
    return False



dirs = [
    '../datasets/venda/mensal/uf/gasolinac/',
    '../datasets/venda/mensal/uf/etanolhidratado/',
    # '../datasets/venda/mensal/uf/gasolinadeaviacao/',
    '../datasets/venda/mensal/uf/glp/',
    # '../datasets/venda/mensal/uf/oleocombustivel/',
    '../datasets/venda/mensal/uf/oleodiesel/',
    '../datasets/venda/mensal/uf/querosenedeaviacao/',
    # '../datasets/venda/mensal/uf/queroseneiluminante/',
]
# pickle_file = './pickle/arima/rolling'


def ets_error_series(args):
    directory, file = args
    chave = ''
    model_file = f'ETS{chave}'
    results_file = f'./paper_roma/{model_file}'
    transformations = ["normal", "deseasonal"]
    cols = ['train_range', 'test_range', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series',
           ]
    if file.endswith('.csv'):
        try:
            uf = file.split("_")[1].upper()
            derivado = file.split("_")[2].split(".")[0]

            full_path = os.path.join(directory, file)
            df = pd.read_csv(full_path, header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
            df['timestamp']=pd.to_datetime(df['timestamp'], infer_datetime_format=True)
            df = df.set_index('timestamp',inplace=False)
            df.index = df.index.to_period('M')
            all_series_test = []
            series = df['m3']
            # train, test = train_test_stats(series, horizon)
            # train_stl = train
            # if 'noresid' in chave:
            #     print_log('----------- SEM RESIDUO NA SERIE ---------')
            #     transformer = STLTransformer(sp=12) 
            #     stl = transformer.fit(train)
            #     train_stl = stl.seasonal_ + stl.trend_

            train_test_splits = []
            min_train_size = 36 + (12 * 25)

            aux_series = series
            while len(aux_series) > horizon + min_train_size:
                train, test = aux_series[:-horizon], aux_series[-horizon:]
                train_test_splits.append((train, test))
                aux_series = train
            for (train, test) in train_test_splits:
                train_stl = train
                _, test_val = train_test_stats(train, horizon) #para pegar o test_val real
                if 'noresid' in chave:
                    print_log('----------- SEM RESIDUO NA SERIE ---------')
                    transformer = STLTransformer(sp=12) 
                    stl = transformer.fit(train)
                    train_stl = stl.seasonal_ + stl.trend_
                train_val, _ = train_test_stats(train_stl, horizon) # pra pegar um train_val (sem/com residual)
                start_train = train.index.tolist()[0]
                final_train = train.index.tolist()[-1]

                start_test = test.index.tolist()[0]
                final_test = test.index.tolist()[-1]

                train_range = f"{start_train}_{final_train}"
                test_range = f"{start_test}_{final_test}"
                for transform in transformations:
                    path_derivado = f'{results_file}/{derivado}/{transform}'
                    flag = checkFolder(path_derivado, f"transform_{uf}.csv", test_range)
                    if flag:
                        train_tf = transform_train(train_stl, format=transform)
                        train_tf_val = transform_train(train_val, format=transform)
                    
                        # params = get_params_model(f'./results/{model_file}/{derivado}/transform_{uf}.csv', transform)
                        
                        model = sm.tsa.ExponentialSmoothing(train_tf, trend='add', seasonal='add', seasonal_periods=12).fit()


                        predictions = model.forecast(steps=horizon)
                        preds = pd.Series(predictions, index=test.index)
                        preds_real = reverse_transform_norm_preds(preds, train, transform)
                                         
                        # preds_real = znorm_reverse(preds_norm, mean, std)
                        error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
                        y_baseline = train[-horizon*1:].values
                        rmse_result = rmse(test, preds_real)
                        mape_result = mape(test, preds_real)
                        pocid_result = pocid(test, preds_real)
                        pbe_result = pbe(test, preds_real)
                        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                        mase_result = mase(test, preds_real, y_baseline)
                        print_log('[RESULTADO EM TRAIN]')
                        print_log(f'MCPM: {mcpm_result}')
                        print_log(f'RMSE: {rmse_result}')
                        print_log(f'MAPE: {mape_result}')
                        print_log(f'POCID: {pocid_result}')
                        print_log(f'PBE: {pbe_result}')
                        adfuller_test = analyze_stationarity(train_tf[1:])
                        os.makedirs(path_derivado, exist_ok=True)
                        csv_path = f'{path_derivado}/transform_{uf}.csv'

                        if not os.path.exists(csv_path):
                            pd.DataFrame(columns=cols).to_csv(csv_path, sep=';', index=False)

                        df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_range , 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ETS', 'PARAMS':str({'trend':'add', 'seasonal':'add', 'seasonal_periods':12}) , 'WINDOW': window, 'HORIZON': horizon,  
                                                'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result,'MCPM': mcpm_result,  'MASE': mase_result,
                                                'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                                'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                                'P11': preds_real[10], 'P12': preds_real[11], 
                                                'error_series': [error_series],
                                                }, index=[0])
                        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)
                   
        except Exception as e:
            print_log(f"Exception: {derivado} em {uf}\n {e}")
            traceback.print_exc()



if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        tasks = [
            (directory, file) 
            for directory in dirs 
            for file in os.listdir(directory) 
            if file.endswith('.csv')
        ]

        pool.map(ets_error_series, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")
