from all_functions import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from mango import scheduler, Tuner
import multiprocessing
from sklearn.metrics import mean_absolute_percentage_error as mape
import traceback
def objective(args_list):
    global train_v_data, train_v_real, test_v_data, test_v_real
    results = []
    params_evaluated = []
    for params in args_list:
        try:
            predictor = DeepAREstimator(
                prediction_length=12, freq="M", 
                context_length = int(params['context_length']),
                num_layers=params["num_layers"],
                # hidden_size=params["hidden_size"],
                batch_size= int(params['mini_batch_size']),
                dropout_rate = params['dropout_rate'],
                # num_cells = int(params['num_cells']),
                # num_layers = int(params['num_layers']),
                # embedding_dimension = int(params['embedding_dimension']),
                trainer_kwargs={
                    # "learning_rate": params['learning_rate'],
                    "accelerator": "gpu",
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "max_epochs": params['max_epochs']
                    # "mini_batch_size": int(params['mini_batch_size']),
                    }
            ).train(train_v_data)
            
            forecast_it = list(predictor.predict(test_v_data.input, num_samples=100))

            all_forecasts = []
            for forecast in forecast_it:
                all_forecasts.append(forecast.mean)
            
            _, mean, std = rolling_window_series(train_v_real, 12)
            preds_real = znorm_reverse(all_forecasts[0], mean, std)

            mape_result = mape(test_v_real, preds_real)
            params_evaluated.append(params)
            results.append(mape_result)
        except:
            continue
    return results

def find_best_parameter(train_norm, test_norm, train_real, test_real):
    global train_v_data, train_v_real, test_v_data, test_v_real

    params_space = {
        'context_length': range(1,200),
        "max_epochs": [50],
        # 'learning_rate': np.logspace(-5, -1, 10),
        'mini_batch_size': range(32, 1029, 32),
        # 'num_cells': range(30, 201, 10),
        'num_layers': range(1, 9),
        'dropout_rate': np.linspace(0.0, 0.2, 5),
        # 'embedding_dimension': range(1, 51)
    }
    conf_Dict = dict()
    conf_Dict['num_iteration'] = 15
    train_v_data = train_norm
    train_v_real = train_real
    test_v_data = test_norm
    test_v_real = test_real

    tuner = Tuner(params_space, objective, conf_Dict)
    results_arima = tuner.minimize()

    return results_arima

def get_train_test_deepar(train_norm, test_norm):
    concat_norm = pd.concat([train_norm, test_norm])
    dataset_norm = PandasDataset(concat_norm, target="value")
    # test_ds = PandasDataset(test_norm, target="value")
    training_data, test_gen = split(dataset_norm, offset=-12)
    test_data = test_gen.generate_instances(prediction_length=12, windows=1)
    return training_data, test_data


def deepar_train(args):
    directory, file = args
    chave = ''
    model_file = f'deepar{chave}'
    results_file = f'./results_hybrid/{model_file}'
    transformations = ["normal", "log", "deseasonal"]
    cols = ['train_range', 'test_range', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series',
           'Test Statistic', 'p-value', 'Lags Used', 'Observations Used', 'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)', 'Stationary'
           ]
    window = 12
    horizon = 12
    saved_params = None

    if file.endswith('.csv'):
        try:
            uf = file.split("_")[1].upper()
            derivado = file.split("_")[2].split(".")[0]

            full_path = os.path.join(directory, file)
            df = pd.read_csv(full_path, header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
            df['timestamp']=pd.to_datetime(df['timestamp'], infer_datetime_format=True)
            df = df.set_index('timestamp',inplace=False)
            df.index = df.index.to_period('M')
            series = df['m3']
            train_test_splits = []
            min_train_size = 36

            aux_series = series
            while len(aux_series) > horizon + min_train_size:
                train, test = aux_series[:-horizon], aux_series[-horizon:]
                train_test_splits.append((train, test))
                aux_series = train

            for i, (train, test) in enumerate(train_test_splits):
                train_stl = train
                _, test_val = train_test_stats(train, horizon) #para pegar o test_val real
                if 'noresid' in chave:
                    print_log('----------- SEM RESIDUO NA SERIE ---------')
                    transformer = STLTransformer(sp=12) 
                    stl = transformer.fit(train)
                    train_stl = stl.seasonal_ + stl.trend_
                train_val, _ = train_test_stats(train_stl, horizon) # pra pegar um train_val (sem/com residual)

                for transform in transformations:
                    train_tf, mean, std = transform_deep_train(train_stl, format=transform)
                    train_tf_val, mean_val, std_val = transform_deep_train(train_val, format=transform)

                    #treino
                    test_tf = znorm_mean_std(test, mean, std)
                    train_data, test_data = get_train_test_deepar(train_tf, test_tf)

                    if i == 0: #se é o primeiro pedaço (ultima parte da serie) encontra parametros para repassar para outras partes
                      #validacao                    
                      test_val_tf = znorm_mean_std(test_val, mean_val, std_val)
                      train_data_val, test_data_val = get_train_test_deepar(train_tf_val, test_val_tf)

                      saved_params = find_best_parameter(train_data_val, test_data_val, train_val, test_val)['best_params']

                    predictor = DeepAREstimator(
                            prediction_length=12,
                            context_length = int(saved_params['context_length']),
                            num_cells = int(saved_params['num_cells']),
                            num_layers = int(saved_params['num_layers']),
                            embedding_dimension = int(saved_params['embedding_dimension']),

                            prediction_length=12, freq="M", 
                            trainer_kwargs={
                                "learning_rate": saved_params['learning_rate'],
                                "max_epochs": saved_params['max_epochs'],
                                "batch_size": int(saved_params['mini_batch_size']),
                                "num_batches_per_epoch": saved_params['num_batches_per_epoch'],
                                "dropout_rate": saved_params['dropout_rate']
                                }
                        ).train(train_data)
                    

                    forecasts_it = list(predictor.predict(test_data.input, num_samples=100))

                    all_forecasts = []
                    for forecast in forecasts_it:
                        all_forecasts.append(forecast.mean)

                    preds_real = znorm_reverse(all_forecasts[0], mean, std)

                    start_train = train.index.tolist()[0]
                    final_train = train.index.tolist()[-1]

                    start_test = test.index.tolist()[0]
                    final_test = test.index.tolist()[-1]

                    train_range = f"{start_train}_{final_train}"
                    test_range = f"{start_test}_{final_test}"
                    
                    # preds_real = znorm_reverse(preds_norm, mean, std)
                    error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
                    y_baseline = series[-horizon*2:-horizon].values
                    rmse_result = rmse(test, preds_real)
                    mape_result = mape(test, preds_real)
                    pocid_result = pocid(test, preds_real)
                    pbe_result = pbe(test, preds_real)
                    mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                    mase_result = mase(test, preds_real, y_baseline)
                    print_log('[RESULTADO EM TRAIN]')
                    print_log(f'PARAMS: {saved_params}')
                    print_log(f'MCPM: {mcpm_result}')
                    print_log(f'RMSE: {rmse_result}')
                    print_log(f'MAPE: {mape_result}')
                    print_log(f'POCID: {pocid_result}')
                    print_log(f'PBE: {pbe_result}')
                    adfuller_test = analyze_stationarity(train_tf[1:])

                    path_derivado = f'{results_file}/{derivado}/{transform}'
                    os.makedirs(path_derivado, exist_ok=True)
                    csv_path = f'{path_derivado}/transform_{uf}.csv'

                    if not os.path.exists(csv_path):
                        pd.DataFrame(columns=cols).to_csv(csv_path, sep=';', index=False)

                    df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_range , 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'DeepAR', 'PARAMS': str(saved_params), 'WINDOW': window, 'HORIZON': horizon,  
                                            'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result,'MCPM': mcpm_result,  'MASE': mase_result,
                                            'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                            'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                            'P11': preds_real[10], 'P12': preds_real[11], 
                                            'error_series': [error_series],
                                            'Test Statistic': adfuller_test['Test Statistic'], 'p-value': adfuller_test['p-value'],
                                            'Lags Used': adfuller_test['Lags Used'],  'Observations Used': adfuller_test['Observations Used'], 'Critical Value (1%)': adfuller_test['Critical Value (1%)'],
                                            'Critical Value (5%)': adfuller_test['Critical Value (5%)'], 'Critical Value (10%)': adfuller_test['Critical Value (10%)'], 'Stationary': adfuller_test['Stationary']
                                            }, index=[0])
                    df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)

        except Exception as e:
            print_log(f"Exception: {derivado} em {uf}\n {e}")
            traceback.print_exc()

dirs = [
    '../datasets/venda/mensal/uf/gasolinac/',
    '../datasets/venda/mensal/uf/etanolhidratado/',
    # '../datasets/venda/mensal/uf/gasolinadeaviacao/',
    # '../datasets/venda/mensal/uf/glp/',
    # '../datasets/venda/mensal/uf/oleocombustivel/',
    # '../datasets/venda/mensal/uf/oleodiesel/',
    # '../datasets/venda/mensal/uf/querosenedeaviacao/',
    # '../datasets/venda/mensal/uf/queroseneiluminante/',
]

if __name__ == "__main__":
    with multiprocessing.Pool(processes=8) as pool:
        tasks = [
            (directory, file) 
            for directory in dirs 
            for file in os.listdir(directory) 
            if file.endswith('.csv')
        ]

        pool.map(deepar_train, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")
