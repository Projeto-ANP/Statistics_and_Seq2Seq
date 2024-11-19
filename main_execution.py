import os
import time
from all_functions import *
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
import multiprocessing
import traceback
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import optuna
from sklearn.linear_model import Ridge, RidgeCV
warnings.filterwarnings("ignore")

def print_log(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] {message}")

def objective_optuna(trial):
    global X_train_v, y_train_v, X_test_v 
    global train_original, test_val
    global regr, format_v, horizon
    global representation, wavelet, level

    if regr == 'xgb':
        param = {
            'learning_rate': trial.suggest_categorical('learning_rate', [0.2, 0.3, 0.4]),
            'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9]),
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200]),
            'random_state': 42
        }
        model = xgboost.XGBRegressor(**param)
        
    elif regr == 'rf':
        param = {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200]),
            'random_state': 42
        }
        model = RandomForestRegressor(**param)
        
    elif regr == 'knn':
        param = {
            'n_neighbors': trial.suggest_categorical('n_neighbors', [1, 3, 5, 7, 9])
        }
        model = KNeighborsRegressor(**param)
        
    elif regr == 'svr':
        param = {
            'kernel': trial.suggest_categorical('kernel', ['rbf']),
            'C': trial.suggest_categorical('C', [0.1, 1, 10, 100, 1000]),
            'gamma': trial.suggest_categorical('gamma', [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]),
            'epsilon': trial.suggest_categorical('epsilon', [0.1, 0.2, 0.5, 0.3]),
            # 'random_state': 42
        }
        model = SVR(**param)
        
    elif regr == 'catboost':
        param = {
            'iterations': trial.suggest_categorical('iterations', [100, 200]),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 9),
            'loss_function': trial.suggest_categorical('loss_function', ['RMSE']),
            'random_state': 42
        }
        model = CatBoostRegressor(**param)
        
    else:
        raise ValueError(f'MODELO {regr} nao existe')

    try:
        model.fit(X_train_v, y_train_v)

        predictions = recursive_step(X_test_v, train_original, model, horizon, window, format_v, representation, wavelet, level)
        preds_real = pd.Series(predictions, index=test_val.index)
        
        mape_result = mape(test_val, preds_real)
    except Exception as e:
        print_log(f'Error: {format_v}')
        print_log('X_TRAIN\n')
        print_log(X_train_v)
        print_log('y_TRAIN\n')
        print_log(y_train_v)
        print(e)
        return float('inf')

    return mape_result
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
def find_best_parameter_optuna(train_x, test_x, train_y, train_v, test_v, format):
    global X_train_v
    global y_train_v
    global X_test_v
    global train_original
    global regr
    global test_val
    global format_v

    X_train_v = train_x
    y_train_v = train_y
    X_test_v = test_x
    train_original = train_v
    test_val = test_v
    format_v = format
    if regr == "ridge":
        return {'alphas': np.logspace(-3, 3, 10)}
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_optuna, n_trials=35)

    return study.best_params

dirs = [
    '../datasets/venda/mensal/uf/gasolinac/',
    '../datasets/venda/mensal/uf/etanolhidratado/',
    '../datasets/venda/mensal/uf/glp/',
    '../datasets/venda/mensal/uf/oleodiesel/',
]
train_tf_v = pd.Series()
train_v_real = pd.Series()
test_v_real = pd.Series()
test_v = pd.Series()
horizon = 12
window = 12
transformacao = "normal"
format_v = "sem"
regr = 'SEM MODELO'
representation = 'NONE'
level = -1
wavelet = 'none'

def image_error_series(args):
    directory, file = args
    global regr 
    global representation
    global wavelet
    global level
    representation = "DWT"
    wavelet = "bior2.2"
    level = 2 #only DWT/SWT
    horizon = 12
    window = 12
    regr = 'svr'
    chave = ''
    model_file = f'{representation}_{regr}{chave}'
    results_file = f'./paper_roma/{model_file}'
    transformations = ["normal", "deseasonal"]
    cols = ['train_range', 'test_range', 'time','UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series'
           ]
    if file.endswith('.csv'):
        
        uf = file.split("_")[1].upper()
        derivado = file.split("_")[2].split(".")[0]

        full_path = os.path.join(directory, file)
        df = pd.read_csv(full_path, header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
        df['timestamp']=pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        df = df.set_index('timestamp',inplace=False)
        df.index = df.index.to_period('M')
        series = df['m3']
        train_test_splits = []
        min_train_size = 36 + (12 * 26)

        aux_series = series
        while len(aux_series) > horizon + min_train_size:
            train, test = aux_series[:-horizon], aux_series[-horizon:]
            train_test_splits.append((train, test))
            aux_series = train

        for (train, test) in train_test_splits:
            train_stl = train
            _, test_val = train_test_stats(train, horizon)
            if 'noresid' in chave:
                print_log('----------- SEM RESIDUO NA SERIE ---------')
                transformer = STLTransformer(sp=12) 
                stl = transformer.fit(train)
                train_stl = stl.seasonal_ + stl.trend_
            train_val, _ = train_test_stats(train_stl, horizon)
            start_train = train_stl.index.tolist()[0]
            final_train = train_stl.index.tolist()[-1]

            start_test = test.index.tolist()[0]
            final_test = test.index.tolist()[-1]

            train_range = f"{start_train}_{final_train}"
            test_range = f"{start_test}_{final_test}"
        
            for transform in transformations:
                path_derivado = f'{results_file}/{derivado}/{transform}'
                flag = checkFolder(path_derivado, f"transform_{uf}.csv", test_range)
                start_exp = time.perf_counter()
                if flag:
                    train_tf = transform_regressors(train_stl, transform)
                    train_tf_val = transform_regressors(train_val, format=transform)
                    try:
                        data = rolling_window_image(pd.concat([train_tf, pd.Series([1,2,3,4,5,6,7,8,9,10,11,12], index=test.index)]), window, representation, wavelet, level) 
                        data = data.dropna()
                        X_train, X_test, y_train, _ = train_test_split(data, horizon)
                        
                        results_rg = {'alphas': np.logspace(-3, 3, 10)}
                        #necessita fazer isso para nao implicar na sazonalidade do val com train
                        if regr != "ridge":
                            data_val = rolling_window_image(pd.concat([train_tf_val, pd.Series([0,0,0,0,0,0,0,0,0,0,0,0], index=test.index)]), window, representation, wavelet, level)
                            data_val = data_val.dropna()

                            X_train_v, X_test_v, y_train_v, _ = train_test_split(data_val, horizon)
                            results_rg = find_best_parameter_optuna(X_train_v, X_test_v, y_train_v, train_val, test_val, transform)
                        
                        if regr == 'rf':
                            results_rg['random_state'] = 42
                            rg = RandomForestRegressor(**results_rg)
                        elif regr == 'knn':
                            rg = KNeighborsRegressor(**results_rg)
                        elif regr == "catboost":
                            results_rg['random_state'] = 42
                            rg = CatBoostRegressor(**results_rg)
                        elif regr == "ridge":
                            rg = RidgeCV(**results_rg)
                        elif regr == "svr":
                            rg = SVR(**results_rg)
                        else:
                            raise ValueError('nao existe esse regressor')
                        rg.fit(X_train, y_train)

                        predictions = recursive_step(X_test, train_stl, rg, horizon, window, transform, representation, wavelet, level)
                        preds_real = pd.Series(predictions, index=test.index)
                        end_exp = time.perf_counter()

                        error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
                        y_baseline = train[-horizon*1:].values
                        rmse_result = rmse(test, preds_real)
                        mape_result = mape(test, preds_real)
                        pocid_result = pocid(test, preds_real)
                        pbe_result = pbe(test, preds_real)
                        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                        mase_result = mase(test, preds_real, y_baseline)

                        os.makedirs(path_derivado, exist_ok=True)
                        csv_path = f'{path_derivado}/transform_{uf}.csv'

                        if not os.path.exists(csv_path):
                            pd.DataFrame(columns=cols).to_csv(csv_path, sep=';', index=False)
                        final_exp = end_exp - start_exp
                        df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_range ,'time': final_exp ,'UF': uf, 'PRODUCT': derivado, 'MODEL': f'{representation}_{regr}_{wavelet}', 'PARAMS': str(results_rg), 'WINDOW': window, 'HORIZON': horizon,  
                                                'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result,'MCPM': mcpm_result,  'MASE': mase_result,
                                                'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                                'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                                'P11': preds_real[10], 'P12': preds_real[11], 
                                                'error_series': [error_series]
                                                }, index=[0])
                        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)
                    
                    except Exception as e:
                        print_log(f"Error: Not possible to train for {derivado}-{transform} in {uf}\n {e}")
                        traceback.print_exc()

if __name__ == '__main__':
    start = time.perf_counter()
    with multiprocessing.Pool(processes=1) as pool:
            tasks = [
                (directory, file) 
                for directory in dirs 
                for file in os.listdir(directory) 
            ]

            pool.map(image_error_series, tasks)
    end = time.perf_counter()
    finaltime = end - start
    print_log(f"EXECUTION TIME: {finaltime}")
    with open(f"./paper_roma/DWT_svr/execution_time.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write(str(finaltime))
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")