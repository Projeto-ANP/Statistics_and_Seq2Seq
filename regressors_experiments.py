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
from catboost import CatBoostRegressor
import optuna
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
        preds_real = reverse_regressors(train_original, preds, format=format_v)
        
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
    # '../datasets/venda/mensal/uf/querosenedeaviacao/',
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
    chave = '_noresid'
    model_file = f'{regr}{chave}'
    results_file = f'./results/{model_file}'
    transformations = ["normal", "log", "deseasonal"]
    cols = ['train_range', 'test_range', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series',
           'Test Statistic', 'p-value', 'Lags Used', 'Observations Used', 'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)', 'Stationary'
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
        min_train_size = 36

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
                if flag:
                    train_tf = transform_regressors(train_stl, transform)
                    train_tf_val = transform_regressors(train_val, format=transform)
                    try:
                        data = rolling_window(pd.concat([train_tf, pd.Series([0,0,0,0,0,0,0,0,0,0,0,0], index=test.index)]), window)
                        data = data.dropna()
                        X_train, X_test, y_train, _ = train_test_split(data, horizon)

                        #necessita fazer isso para nao implicar na sazonalidade do val com train
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
                        else:
                            raise ValueError('nao existe esse regressor')
                        rg.fit(X_train, y_train)

                        predictions = recursive_multistep_forecasting(X_test, rg, horizon)
                        preds = pd.Series(predictions, index=test.index)
                        preds_real = reverse_regressors(train_stl, preds, format=transform)

                        # preds_real = znorm_reverse(preds_norm, mean, std)
                        error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
                        y_baseline = series[-horizon*2:-horizon].values
                        rmse_result = rmse(test, preds_real)
                        mape_result = mape(test, preds_real)
                        pocid_result = pocid(test, preds_real)
                        pbe_result = pbe(test, preds_real)
                        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                        mase_result = mase(test, preds_real, y_baseline)
                        # print_log('[RESULTADO EM TRAIN]')
                        # print_log(f'PARAMS: {str(results_rg)}')
                        # print_log(f'MCPM: {mcpm_result}')
                        # print_log(f'RMSE: {rmse_result}')
                        # print_log(f'MAPE: {mape_result}')
                        # print_log(f'POCID: {pocid_result}')
                        # print_log(f'PBE: {pbe_result}')
                        adfuller_test = analyze_stationarity(train_tf[1:])

                        path_derivado = f'{results_file}/{derivado}/{transform}'
                        os.makedirs(path_derivado, exist_ok=True)
                        csv_path = f'{path_derivado}/transform_{uf}.csv'

                        if not os.path.exists(csv_path):
                            pd.DataFrame(columns=cols).to_csv(csv_path, sep=';', index=False)

                        df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_range , 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ARIMA', 'PARAMS': str(results_rg), 'WINDOW': window, 'HORIZON': horizon,  
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
                        print_log(f"Error: Not possible to train for {derivado}-{transform} in {uf}\n {e}")
                        traceback.print_exc()


def random_wavelet_convolution_v2(serie, convs):
    features = np.array([])

    wavetypes, wavelets, levels, scales = convs
    
    for i in range(len(wavetypes)):
        coeffs = generate_random_wavelet(serie, wavetypes[i], wavelets[i], levels[i], scales[i])
        coeffs_mean = np.mean(coeffs, axis=1)
        features = np.concatenate([features, coeffs_mean ])
      
    return features

def features_target_v2(series, convs, window):
    data = []
    series = series.tolist()
    series_norm = znorm(series)
    for i in range(len(series) - window):
        example = np.array(series_norm[i:i+window])
        target_value = series[i+window]
        features = random_wavelet_convolution_v2(example, convs)
        feats_target = np.concatenate((features, [target_value]))
        norm_features = feats_target
        data.append(norm_features)

        df = pd.DataFrame(data)
    return df

def recursive_rocket1_v2(X_test, model, train, convs, horizon):
    example = X_test.iloc[0].values.reshape(1,-1)
    preds = []
    for i in range(horizon):
        pred = model.predict(example)[0]
        preds.append(pred)
        train = train[1:]
        train.append(pred)

        example_transform = random_wavelet_convolution_v2(train, convs)
        example = example_transform.reshape(1,-1)

    return preds

def features_target_v2(series, convs, window):
    data = []
    series = series.tolist()
    # series_norm = znorm(series)
    for i in range(len(series) - window):
        example = np.array(series[i:i+window])
        target_value = series[i+window]
        features = random_wavelet_convolution_v2(example, convs)
        feats_target = np.concatenate((features, [target_value]))
        norm_features = feats_target
        data.append(norm_features)

        df = pd.DataFrame(data)
    return df
def generate_convolutions_2(len_serie, num_wavelets = 10):
    wavelets = np.empty(num_wavelets, dtype=object)
    wavetypes = np.random.choice(["cwt", "dwt", "swt"], size=num_wavelets)
    # wavetypes = ['dwt', 'cwt', "cwt", 'dwt']
    # num_wavelets = 4
    # wavetypes = ['cwt', 'cwt', 'cwt', 'cwt','dwt','swt','swt','swt','swt', 'swt']
    # wavelet_types = ["cwt", "swt", "dwt"]
    # probabilities = [0.0, 0.5, 0.5]

    # Gerando a lista de wavetypes em uma linha
    # wavetypes = np.concatenate((["cwt"] * 5, np.random.choice(wavelet_types, size=num_wavelets - 5, p=probabilities)))
    # num_wavelets = 2
    # wavetypes = ['cwt', 'cwt']
   
    # wavetypes = np.random.choice(["cwt"], size=num_wavelets)
    # wavetypes = np.array(["cwt", "cwt", "cwt", "cwt", "cwt", "cwt", "cwt", "cwt", "cwt", "cwt"])
    # wavelets = np.array(["morl", 0, "mexh", 0, "morl", 0, "mexh", 0, "morl", 0, "morl", 0, "mexh", 0, "morl", 0, "mexh", 0, "morl", 0])

    levels = np.zeros(num_wavelets, dtype=int)
    scales = [np.zeros(0) for _ in range(num_wavelets)]

    for i, waveType in enumerate(wavetypes):
        if waveType == "cwt":
            wavelet = np.random.choice(["morl"])
            typeScale = np.random.randint(low = 1, high = 3)
            if typeScale == 1:
                max_scale_limit = np.random.randint(low = len_serie/2, high = len_serie + 1)
                max_scale = np.random.randint(low=len_serie/2, high=max_scale_limit + 1)
                
                scale = np.random.uniform(len_serie/3, max_scale, size=10)
            else:
                # scale = np.random.uniform(0.1, 0.9, size=1000)
                scale = np.random.choice(np.arange(0.1, 1.1, 0.1), size=10)

            # scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
            # scale = [1]
        
            wavelets[i] = wavelet
            scales[i] = scale
        elif waveType == "dwt":
            # wavelet = np.random.choice(["bior2.2", "bior3.3", "bior2.4", "db4"])
            wavelet = np.random.choice(pywt.wavelist(kind="discrete"))
            # wavelet = "db2"
            # wavelet = "db20"
            # db1_wavelets = [wavelet for wavelet in pywt.wavelist(kind="discrete") if wavelet.startswith('db1')]
            # wavelet = np.random.choice(db1_wavelets)
            wlt = pywt.Wavelet(wavelet)
            max_level = pywt.dwt_max_level(len_serie, wlt.dec_len)
            level = 1
            # if max_level > 1:
            #     level = np.random.randint(low = 1, high = max_level + 1)
            
            wavelets[i] = wavelet
            levels[i] = level
        elif waveType == "swt":
            wavelet = np.random.choice(pywt.wavelist(kind="discrete"))
            # wavelet = "bior2.2"
            # wavelet = "sym2"
            
            # bior_wavelets = [wavelet for wavelet in pywt.wavelist(kind="discrete") if wavelet.startswith('bior')]
            wlt = pywt.Wavelet(wavelet)
            max_level = pywt.swt_max_level(len_serie)
            level = np.random.randint(low = 1, high = max_level + 1)
            wavelets[i] = wavelet
            levels[i] = max_level

    return wavetypes, wavelets, levels, scales



def rocket2_error_series(args):
    directory, file = args
    global regr 
    regr = 'ridge'
    chave = ''
    model_file = f'wavelets_ridge'
    results_file = f'./results/{model_file}'
    # transformations = ["normal"]
    transform = "normal"
    cols = ['train_range', 'test_range', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE','MCPM', 'MASE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'error_series',
           ]
    if file.endswith('.csv'):
        
        uf = file.split("_")[1].upper()
        derivado = file.split("_")[2].split(".")[0]
        window = 12
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
        
            # for transform in transformations:
            path_derivado = f'{results_file}/{derivado}/{transform}'
            flag = checkFolder(path_derivado, f"transform_{uf}.csv", test_range)
            if flag:
                train_tf = train_stl
                # train_tf = transform_train(train_stl, transform)
                # train_tf_val = transform_regressors(train_val, format=transform)
                try:
                    convs = generate_convolutions_2(window, 20)
                    # train_tf = train
                    data = features_target_v2(pd.concat([train, pd.Series([0] * horizon, index=test.index)]), convs, window)
                    X_train, X_test, y_train, _ = train_test_split(data, horizon)

                    rg = RidgeCV(alphas=np.logspace(-3, 3, 10))
                    rg.fit(X_train, y_train)

                    preds_real = recursive_rocket1_v2(X_test, rg, train[-window:].tolist(), convs, horizon)
                    # preds_real = recursive_rocket2(X_test, rg, train_tf[-window:].tolist(), horizon)
                    # predictions = recursive_multistep_forecasting(X_test, rg, horizon)
                    # preds = pd.Series(predictions, index=test.index)
                    # preds_real = reverse_regressors(train_stl, preds, format=transform)

                    # preds_real = znorm_reverse(preds_norm, mean, std)
                    error_series = [a - b for a, b in zip(test.tolist(), preds_real)]
                    y_baseline = series[-horizon*2:-horizon].values
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

                    df_temp = pd.DataFrame({'train_range': train_range, 'test_range': test_range , 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'RIDGE_wavelets', 'PARAMS': '', 'WINDOW': window, 'HORIZON': horizon,  
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



if __name__ == '__main__':
#   for directory in dirs:
#     regressors_preds(directory)
    with multiprocessing.Pool() as pool:
            tasks = [
                (directory, file) 
                for directory in dirs 
                for file in os.listdir(directory) 
            ]

            pool.map(rocket2_error_series, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")