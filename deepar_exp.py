from all_functions import *
import pandas as pd
import os
import csv
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from mango import scheduler, Tuner
import multiprocessing
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error as mape
import traceback
import optuna
import random
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

BATCH_SIZE = 32
MAX_EPOCHS = 15
NUM_BATCHES_PER_EPOCH = 20

def objective_optuna(trial):
    global train_v_data, train_v_real, test_v_data, test_v_real, format

    # Sugere valores para os hiperparâmetros
    context_length = trial.suggest_categorical('context_length', [12, 24, 36])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    # hidden_size = trial.suggest_categorical("hidden_size", [40, 50])
    # learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    # batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    batch_size = BATCH_SIZE
    # max_epochs = trial.suggest_int('max_epochs', 10, 50)
    max_epochs = MAX_EPOCHS
    # num_batches_per_epoch = trial.suggest_int('num_batches_per_epoch', 10, 50)
    num_batches_per_epoch = NUM_BATCHES_PER_EPOCH
    from pytorch_lightning.callbacks import ModelCheckpoint

    # Define a ModelCheckpoint callback but configure it to save only the latest checkpoint
    try:
        # Configura o DeepAREstimator com os parâmetros sugeridos
        predictor = DeepAREstimator(
                prediction_length=12, freq="M", 
                context_length=context_length,
                num_layers=num_layers,
                batch_size=batch_size,
                num_batches_per_epoch=num_batches_per_epoch,
                # hidden_size=hidden_size,
                # lr=learning_rate,
                trainer_kwargs={
                    "accelerator": "gpu",
                    "devices": [0],
                    "max_epochs": max_epochs,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "limit_test_batches": 0.25,
                    "callbacks": []
                    }
            ).train(train_v_data)
            
        
        forecast_it = list(predictor.predict(test_v_data.input, num_samples=100))

        all_forecasts = []
        for forecast in forecast_it:
            all_forecasts.append(forecast.mean)

        # _, mean, std = rolling_window_series(train_v_real, 12)
        # preds_real = znorm_reverse(all_forecasts[0], mean, std)
        preds = pd.Series(all_forecasts[0], index=test_v_real.index)
        preds_real = reverse_regressors(train_v_real, preds, format=format)

        mape_result = mape(test_v_real, preds_real)
        return mape_result
    except Exception as e:
        print(f"Error: {e}")
        return float('inf')  # Retorna um valor alto para indicar falha


def find_best_parameter_optuna(train_norm, test_norm, train_real, test_real, transform):
    global train_v_data, train_v_real, test_v_data, test_v_real, format

    train_v_data = train_norm
    train_v_real = train_real
    test_v_data = test_norm
    test_v_real = test_real
    format = transform

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(),
        pruner=MedianPruner()
    )
    study.optimize(objective_optuna, n_trials=35)
    
    return study.best_params, study.best_value
def get_train_test_deepar(train_norm, test_norm):
    concat_norm = pd.concat([train_norm, test_norm])
    dataset_norm = PandasDataset(concat_norm, target="value")
    # test_ds = PandasDataset(test_norm, target="value")
    training_data, test_gen = split(dataset_norm, offset=-12)
    test_data = test_gen.generate_instances(prediction_length=12, windows=1)
    return training_data, test_data

def transform_test_deepar(train_ref, test, format):
    if format == "log":
        constante = 10
        series_ts = np.log(test + constante)
        return series_ts
    elif format == "deseasonal":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train_ref)
        series_ts = transform.transform(test)
        return series_ts
    
    return test

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

            # for i, (train, test) in enumerate(train_test_splits):
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
                        train_tf, mean, std = transform_deep_train(train_stl, format=transform)
                        train_tf_val, mean_val, std_val = transform_deep_train(train_val, format=transform)

                        #treino
                        # test_tf = transform_test_deepar(train_stl, test, format=transform)
                        # test_tf_norm = znorm_mean_std(test_tf, mean, std)
                        train_data, test_data = get_train_test_deepar(train_tf, test)

                        # if i == 0: #se é o primeiro pedaço (ultima parte da serie) encontra parametros para repassar para outras partes
                        #   #validacao                    
                        #   test_tf_val = transform_test_deepar(train_val, test_val, format=transform)
                        #   test_tf_val_norm = znorm_mean_std(test_tf_val, mean_val, std_val)
                        #   train_data_val, test_data_val = get_train_test_deepar(train_tf_val, test_tf_val_norm)
                        #   print_log(f"------------ FINDING PARAMETERS FOR {derivado} in {uf}")
                        #   saved_params, _ = find_best_parameter_optuna(train_data_val, test_data_val, train_val, test_val)
                        #   print_log(f"\n ------------ FOUND BEST PARAMETERS FOR {derivado} in {uf} ---------------")
                        #   print_log(saved_params)

                        #validacao                    
                        # test_tf_val = transform_test_deepar(train_val, test_val, format=transform)
                        # test_tf_val_norm = znorm_mean_std(test_tf_val, mean_val, std_val)
                        train_data_val, test_data_val = get_train_test_deepar(train_tf_val, test_val)
                        print_log(f"------------ FINDING PARAMETERS FOR {derivado} in {uf}")
                        saved_params, _ = find_best_parameter_optuna(train_data_val, test_data_val, train_val, test_val, transform)
                        
                        saved_params['batch_size'] = BATCH_SIZE
                        # saved_params['max_epochs'] = MAX_EPOCHS
                        saved_params['max_epochs'] = 30
                        saved_params['num_batches_per_epoch'] = NUM_BATCHES_PER_EPOCH
                        predictor = DeepAREstimator(
                                prediction_length=12,
                                context_length = int(saved_params['context_length']),
                                num_layers = int(saved_params['num_layers']),
                                num_batches_per_epoch=saved_params['num_batches_per_epoch'],
                                batch_size= int(saved_params['batch_size']),
                                # hidden_size=saved_params['hidden_size'],
                                # lr=saved_params['lr'],
                                # dropout_rate = saved_params['dropout_rate'],
                                freq="M", 
                                trainer_kwargs={
                                    # "learning_rate":
                                    "max_epochs": saved_params['max_epochs'],
                                    # "strategy": saved_params["strategy"],
                                    "devices": [1],
                                    "enable_progress_bar": False,
                                    "enable_model_summary": False,
                                    "accelerator": "gpu",
                                     "callbacks": []
                                    }
                            ).train(train_data)
                        

                        forecasts_it = list(predictor.predict(test_data.input, num_samples=100))

                        all_forecasts = []
                        for forecast in forecasts_it:
                            all_forecasts.append(forecast.mean)
                        preds = pd.Series(all_forecasts[0], index=test.index)
                        preds_real = reverse_regressors(train, preds, format=transform)
                        # preds_real = znorm_reverse(all_forecasts[0], mean, std)

                        # preds_real = znorm_reverse(preds_norm, mean, std)
                        error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
                        y_baseline = series[-horizon*2:-horizon].values
                        try:
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

                            # path_derivado = f'{results_file}/{derivado}/{transform}'
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
                            print_log(f"Error: {train_range} {derivado} {transform} in {uf}: \n ... {e}")
                            print_log(preds_real)
        except Exception as e:
            print_log(f"Exception: {derivado} em {uf}\n {e}")
            traceback.print_exc()

dirs = [
    # '../datasets/venda/mensal/uf/gasolinac/',
    # '../datasets/venda/mensal/uf/etanolhidratado/',
    # '../datasets/venda/mensal/uf/gasolinadeaviacao/',
    # '../datasets/venda/mensal/uf/glp/',
    # '../datasets/venda/mensal/uf/oleocombustivel/',
    # '../datasets/venda/mensal/uf/oleodiesel/',
    # '../datasets/venda/mensal/uf/querosenedeaviacao/',
    # '../datasets/venda/mensal/uf/queroseneiluminante/',
]

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

#function to reproduce the same result always
def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    seed_value = 42
    set_seed(seed_value)
    # torch.multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=17) as pool:
        tasks = [
            (directory, file) 
            for directory in dirs 
            for file in os.listdir(directory) 
            if file.endswith('.csv')
        ]

        pool.map(deepar_train, tasks)
    
    # for directory in dirs:
    #     for file in os.listdir(directory):
    #         deepar_train(directory, file)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")
