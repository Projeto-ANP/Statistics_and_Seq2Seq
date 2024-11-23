from all_functions import *
import os
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.ensemble import RandomForestRegressor
from aeon.transformations.detrend import STLTransformer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from mango import scheduler, Tuner
from sklearn.svm import SVR
import multiprocessing
import traceback
import xgboost
from tsml.feature_based import FPCARegressor
from catboost import CatBoostRegressor
import optuna

def generate_index(start_date, end_date):
    end_date_dt = pd.to_datetime(end_date)
    
    start_date_dt = pd.to_datetime(start_date)
    
    index = pd.period_range(start=start_date_dt, end=end_date_dt, freq='M')

    return index

def get_train_real(series, start_date):
    start_period = pd.to_datetime(start_date).to_period('M')
    
    filtered_series = series[series.index < start_period]

    return filtered_series

def get_preds_hybrid(path, test_date, start_index):
    df = pd.read_csv(path, sep=";")
    results = {}
    filtered_df = df[df['test_range'] == test_date]
    columns_p1_to_p12 = filtered_df.loc[:, 'P1':'P12']
    values_list = columns_p1_to_p12.values.flatten().tolist()     
    results = pd.Series(values_list, index=start_index)
    return results
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

def get_predictions_csv(path, data, start_index):
    df = pd.read_csv(path, sep=";")
    filtered_df = df[df['DATA'] == data]
    
    columns_p1_to_p12 = filtered_df.loc[:, 'P1':'P12']
    
    values_list = columns_p1_to_p12.values.flatten().tolist()        
    # params = filtered_df['PARAMS'].iloc[0]
            
    return pd.Series(values_list, index=start_index), ''


def get_arima_mean(path, data, start_index):
    df = pd.read_csv(path, sep=";")
    filtered_df = df[df['DATA'] == data]
    
    columns_p1_to_p12 = filtered_df.loc[:, 'P1':'P12']
    
    values_list = columns_p1_to_p12.values.flatten().tolist()        
            
    return pd.Series(values_list, index=start_index)



train_tf_v = pd.Series()
train_v_real = pd.Series()
test_v_real = pd.Series()
test_v = pd.Series()
horizon = 12
window = 12
transformacao = "normal"
format_v = "sem"
# @scheduler.parallel(n_jobs=1)


def objective_optuna(trial):
    global X_train_v, y_train_v, X_test_v 
    global train_original, test_val
    global regr, format_v, horizon

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
        # Treine o modelo
        model.fit(X_train_v, y_train_v)

        # Faça previsões
        predictions = recursive_multistep_forecasting(X_test_v, model, horizon)
        preds = pd.Series(predictions, index=test_val.index)
        preds_real = reverse_regressors(train_original, preds, window=12, format=format_v)
        
        # Calcule o MAPE
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

def get_params_model(caminho_arquivo, transformation):
    df = pd.read_csv(caminho_arquivo, sep=';')
    
    df_filtrado = df[df['DATA'] == transformation]
    params_dict = ast.literal_eval(df_filtrado['PARAMS'].iloc[0])
    
    return params_dict

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

horizon = 12
window = 12
colunas = ['DATA', 'MCPM', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 
            'RMSE', 'MAPE', 'POCID', 'PBE', 'MASE',
            'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12',
           ]
df_result = pd.DataFrame(columns=colunas)

#pkill -f -SIGINT "regressor_experiments.py"
regr = 'SEM MODELO'
def regressor_error_series(args):
    directory, file = args
    global regr 
    regr = 'catboost'
    chave = ''
    model_file = f'{regr}{chave}'
    window = 12
    results_file = f'./paper_roma/{model_file}'
    transformations = ["normal", "deseasonal"]
    cols = ['train_range', 'test_range', 'time', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series',
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
            _, test_val = train_test_stats(train, horizon) #para pegar o test_val real
            if 'noresid' in chave:
                print_log('----------- SEM RESIDUO NA SERIE ---------')
                transformer = STLTransformer(sp=12) 
                stl = transformer.fit(train)
                train_stl = stl.seasonal_ + stl.trend_
            train_val, _ = train_test_stats(train_stl, horizon) # pra pegar um train_val (sem/com residual)
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
                        data = rolling_window(pd.concat([train_tf, pd.Series([0,0,0,0,0,0,0,0,0,0,0,0], index=test.index)]), window)
                        data = data.dropna()
                        X_train, X_test, y_train, _ = train_test_split(data, horizon)

                        results_rg = {'alphas': np.logspace(-3, 3, 10)}
                        #necessita fazer isso para nao implicar na sazonalidade do val com train
                        if regr != "ridge" and regr != "fpca":
                            data_val = rolling_window(pd.concat([train_tf_val, pd.Series([0,0,0,0,0,0,0,0,0,0,0,0], index=test.index)]), window)
                            data_val = data_val.dropna()

                            X_train_v, X_test_v, y_train_v, _ = train_test_split(data_val, horizon)
                            results_rg = find_best_parameter_optuna(X_train_v, X_test_v, y_train_v, train_val, test_val, transform)

                        if regr == 'xgb':
                            results_rg['random_state'] = 42
                            rg = xgboost.XGBRegressor(**results_rg) 
                        elif regr == 'rf':
                            results_rg['random_state'] = 42
                            rg = RandomForestRegressor(**results_rg)
                        elif regr == 'knn':
                            rg = KNeighborsRegressor(**results_rg)
                        elif regr == "catboost":
                            results_rg['random_state'] = 42
                            rg = CatBoostRegressor(**results_rg)
                        elif regr == "svr":
                            # results_rg['random_state'] = 42
                            rg = SVR(**results_rg)
                        elif regr == "ridge":
                            rg = RidgeCV(**results_rg)
                            
                        elif regr == "fpca":
                            rg = FPCARegressor(
                                n_jobs=1,
                                bspline=True,
                                order=4,
                                # estimator=RidgeCV(**{'alphas': np.logspace(-3, 3, 10)}),
                                n_basis=10,
                                # n_basis=None
                            )
                        else:
                            raise ValueError('nao existe esse regressor')
                        rg.fit(X_train, y_train)

                        predictions = recursive_multistep_forecasting(X_test, rg, horizon)
                        preds = pd.Series(predictions, index=test.index)
                        preds_real = reverse_regressors(train_stl, preds, window,format=transform)
                        end_exp = time.perf_counter()
                        # preds_real = znorm_reverse(preds_norm, mean, std)
                        error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
                        y_baseline = train[-horizon*1:].values
                        rmse_result = rmse(test, preds_real)
                        mape_result = mape(test, preds_real)
                        pocid_result = pocid(test, preds_real)
                        pbe_result = pbe(test, preds_real)
                        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                        mase_result = mase(test, preds_real, y_baseline)
                    
                        path_derivado = f'{results_file}/{derivado}/{transform}'
                        os.makedirs(path_derivado, exist_ok=True)
                        csv_path = f'{path_derivado}/transform_{uf}.csv'

                        if not os.path.exists(csv_path):
                            pd.DataFrame(columns=cols).to_csv(csv_path, sep=';', index=False)
                        final_exp = end_exp - start_exp
                        df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_range, 'time': final_exp, 'UF': uf, 'PRODUCT': derivado, 'MODEL': f'{regr.upper()}', 'PARAMS': str(results_rg), 'WINDOW': window, 'HORIZON': horizon,  
                                                'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result,'MCPM': mcpm_result,  'MASE': mase_result,
                                                'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                                'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                                'P11': preds_real[10], 'P12': preds_real[11], 
                                                'error_series': [error_series],
                                                }, index=[0])
                        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)
                    
                    except Exception as e:
                        print_log(f"Error: Not possible to train for {derivado}-{transform} in {uf}\n {e}")
                        traceback.print_exc()

def combination_series(args):
    directory, file = args
    global regr 
    regr = 'rf'
    model_file = f'combination4_{regr}'
    window = 12
    results_file = f'./results/{model_file}'
    transformations = ["normal"]
    cols = ['train_range', 'test_range', 'time', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series',
           ]
    
    models_to_combine = ["arima", "catboost", "rf", "ETS"]
    dates_complete = [
        '1993-03_1994-02',
        '1994-03_1995-02', '1995-03_1996-02', '1996-03_1997-02', '1997-03_1998-02', '1998-03_1999-02',
        '1999-03_2000-02', '2000-03_2001-02', '2001-03_2002-02', '2002-03_2003-02', '2003-03_2004-02',
        '2004-03_2005-02', '2005-03_2006-02', '2006-03_2007-02', '2007-03_2008-02', '2008-03_2009-02',
        '2009-03_2010-02', '2010-03_2011-02', '2011-03_2012-02', '2012-03_2013-02', '2013-03_2014-02',
        '2014-03_2015-02', '2015-03_2016-02', '2016-03_2017-02', '2017-03_2018-02', '2018-03_2019-02',
        '2019-03_2020-02', '2020-03_2021-02', '2021-03_2022-02', '2022-03_2023-02', '2023-03_2024-02',
         ]
    last_years = 5
    for i in range(1, last_years + 1):
        test_target = dates_complete[-i]
        train_range = f"1993-03_{test_target.split('_')[1]}"

        if test_target in dates_complete:
            index = dates_complete.index(test_target) + 1
            dates = dates_complete[:index]

        if file.endswith('.csv'):
            
            uf = file.split("_")[1].upper()
            derivado = file.split("_")[2].split(".")[0]
            data_features = pd.DataFrame()

            full_path = os.path.join(directory, file)
            df = pd.read_csv(full_path, header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
            df['timestamp']=pd.to_datetime(df['timestamp'], infer_datetime_format=True)
            df = df.set_index('timestamp',inplace=False)
            df.index = df.index.to_period('M')
            series = df['m3']
            start_exp = time.perf_counter()
            for date in dates:
                start_date, end_date = date.split('_')
                test_index = generate_index(start_date, end_date)
                test_real = get_test_real(series, start_date, end_date)
                row_data = []
                for model in models_to_combine:
                    for transform in transformations:
                        serie_pred = get_preds_hybrid(f'./results/{model}/{derivado}/{transform}/transform_{uf}.csv', date, test_index)
                        row_data.extend(serie_pred.values)

                row_data.extend(test_real.values)

                new_row = pd.DataFrame([row_data])
                data_features = pd.concat([data_features, new_row], ignore_index=True)

            X = data_features.iloc[:, :-horizon].values
            y = data_features.iloc[:, -horizon:].values
            X_train = X[:-1]
            y_train = y[:-1]
            X_test = X[-1].reshape(1, -1)
            test = y[-1]

            if regr == "rf":
                model = RandomForestRegressor(random_state=42)

            model.fit(X_train, y_train)
            preds_real = model.predict(X_test)[0]
            end_exp = time.perf_counter()
            final_exp = end_exp - start_exp

            error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
            # y_baseline = train[-horizon*1:].values
            rmse_result = rmse(test, preds_real)
            mape_result = mape(test, preds_real)
            pocid_result = pocid(test, preds_real)
            pbe_result = pbe(test, preds_real)
            mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
            # mase_result = mase(test, preds_real, y_baseline)

            path_derivado = f'{results_file}/{derivado}/{transform}'
            os.makedirs(path_derivado, exist_ok=True)
            csv_path = f'{path_derivado}/transform_{uf}.csv'

            if not os.path.exists(csv_path):
                pd.DataFrame(columns=cols).to_csv(csv_path, sep=';', index=False)
            
            df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_target, 'time': final_exp, 'UF': uf, 'PRODUCT': derivado, 'MODEL':  f"{'_'.join(models_to_combine)}", 'PARAMS': str({}), 'WINDOW': window, 'HORIZON': horizon,  
                                    'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result,'MCPM': mcpm_result,  'MASE': 0.0,
                                    'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                    'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                    'P11': preds_real[10], 'P12': preds_real[11], 
                                    'error_series': [error_series],
                                    }, index=[0])
            df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)

def combination_mean(args):
    directory, file = args
    global regr 
    regr = 'rf'
    model_file = f'combination_mean'
    window = 12
    results_file = f'./results/{model_file}'
    transformations = ["normal"]
    cols = ['train_range', 'test_range', 'time', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series',
           ]
    
    models_to_combine = ["arima", "catboost", "rf", "ETS"]
    dates_complete = [
        '1993-03_1994-02',
        '1994-03_1995-02', '1995-03_1996-02', '1996-03_1997-02', '1997-03_1998-02', '1998-03_1999-02',
        '1999-03_2000-02', '2000-03_2001-02', '2001-03_2002-02', '2002-03_2003-02', '2003-03_2004-02',
        '2004-03_2005-02', '2005-03_2006-02', '2006-03_2007-02', '2007-03_2008-02', '2008-03_2009-02',
        '2009-03_2010-02', '2010-03_2011-02', '2011-03_2012-02', '2012-03_2013-02', '2013-03_2014-02',
        '2014-03_2015-02', '2015-03_2016-02', '2016-03_2017-02', '2017-03_2018-02', '2018-03_2019-02',
        '2019-03_2020-02', '2020-03_2021-02', '2021-03_2022-02', '2022-03_2023-02', '2023-03_2024-02',
         ]
    last_years = 5
    for i in range(1, last_years + 1):
        test_target = dates_complete[-i]
        train_range = f"1993-03_{test_target.split('_')[1]}"

        if file.endswith('.csv'):
            
            uf = file.split("_")[1].upper()
            derivado = file.split("_")[2].split(".")[0]

            full_path = os.path.join(directory, file)
            df = pd.read_csv(full_path, header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
            df['timestamp']=pd.to_datetime(df['timestamp'], infer_datetime_format=True)
            df = df.set_index('timestamp',inplace=False)
            df.index = df.index.to_period('M')
            series = df['m3']
            start_exp = time.perf_counter()

            start_date, end_date = test_target.split('_')
            test_index = generate_index(start_date, end_date)
            test = get_test_real(series, start_date, end_date)
            series_preds = []
            for model in models_to_combine:
                for transform in transformations:
                    serie_pred = get_preds_hybrid(f'./results/{model}/{derivado}/{transform}/transform_{uf}.csv', test_target, test_index)
                    series_preds.append(serie_pred.values)

            preds_real = [sum(ponto) / len(ponto) for ponto in zip(*series_preds)]
            
            end_exp = time.perf_counter()
            final_exp = end_exp - start_exp

            error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
            # y_baseline = train[-horizon*1:].values
            rmse_result = rmse(test, preds_real)
            mape_result = mape(test, preds_real)
            pocid_result = pocid(test, preds_real)
            pbe_result = pbe(test, preds_real)
            mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
            # mase_result = mase(test, preds_real, y_baseline)

            path_derivado = f'{results_file}/{derivado}/{transform}'
            os.makedirs(path_derivado, exist_ok=True)
            csv_path = f'{path_derivado}/transform_{uf}.csv'

            if not os.path.exists(csv_path):
                pd.DataFrame(columns=cols).to_csv(csv_path, sep=';', index=False)
            
            df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_target, 'time': final_exp, 'UF': uf, 'PRODUCT': derivado, 'MODEL':  f"{'_'.join(models_to_combine)}", 'PARAMS': str({}), 'WINDOW': window, 'HORIZON': horizon,  
                                    'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result,'MCPM': mcpm_result,  'MASE': 0.0,
                                    'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                    'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                    'P11': preds_real[10], 'P12': preds_real[11], 
                                    'error_series': [error_series],
                                    }, index=[0])
            df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)


import time
if __name__ == '__main__':
#   for directory in dirs:
#     regressors_preds(directory)
    start = time.perf_counter()
    with multiprocessing.Pool() as pool:
            tasks = [
                (directory, file) 
                for directory in dirs 
                for file in os.listdir(directory) 
            ]

            pool.map(regressor_error_series, tasks)
    end = time.perf_counter()
    finaltime = end - start
    print_log(f"EXECUTION TIME: {finaltime}")
    # with open(f"./paper_roma/SWT_MTF_fpca/execution_time.txt", "w", encoding="utf-8") as arquivo:
        # arquivo.write(str(finaltime))

    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")