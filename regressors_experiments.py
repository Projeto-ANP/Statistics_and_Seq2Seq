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
from mango import scheduler, Tuner
import multiprocessing
import traceback
import xgboost

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
horizon = 12
window = 12
transformacao = "normal"

def objective_Xgboost(args_list):
    global X_train_v, y_train_v, X_test_v, train_original

    train_v, test = train_test_stats(train_original, horizon)
    
    # train_v = np.log(train_v) #log
    results = []
    for hyper_par in args_list:

        # rg = xgboost.XGBRegressor(**hyper_par)
        # rg = RandomForestRegressor(**hyper_par)
        rg = KNeighborsRegressor(**hyper_par)
        rg.fit(X_train_v, y_train_v)

        predictions = recursive_multistep_forecasting(X_test_v, rg, horizon)
        # mean_norm, std_norm = get_stats_norm(series_original, horizon, window)
        mean_norm = np.mean(train_v[-horizon:])
        std_norm = np.std(train_v[-horizon:])

        # preds_log = znorm_reverse(np.array(predictions), mean_norm, std_norm) #log
        preds_real = znorm_reverse(np.array(predictions), mean_norm, std_norm)
        # preds_real = np.exp(preds_log) #log
        mape_result = mape(test, preds_real)
        results.append(mape_result)
        
    return results

def find_best_parameter_xgb(train_x, test_x, train_y, train_o):
    global X_train_v
    global y_train_v
    global X_test_v
    global train_original
    
    param_xgb = {
        # 'learning_rate': [0.2, 0.3, 0.4], #xgb
        # 'subsample': [0.7, 0.8, 0.9], #xgb
        # 'n_estimators': [50, 100, 150, 200] #rf/xgb
        'n_neighbors': [1,3,5,7,9]
    }

    conf_Dict = dict()
    conf_Dict['num_iteration'] = 35
    X_train_v = train_x
    y_train_v = train_y
    X_test_v = test_x
    train_original = train_o
    tuner = Tuner(param_xgb, objective_Xgboost, conf_Dict)
    results_arima = tuner.minimize()

    return results_arima



def find_best_parameter(train, test, train_val_real):
    global train_tf_v
    global train_v_real
    global test_v_real
    param_space_arima = dict(
                    p = range(2, 25),
                    d = range(1, 4),
                    q = range(2, 25))
    conf_Dict = dict()
    conf_Dict['num_iteration'] = 35
    train_tf_v = train
    train_v_real = train_val_real
    test_v_real = test

    tuner = Tuner(param_space_arima, arima_objective_function, conf_Dict)
    results_arima = tuner.minimize()

    return results_arima

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

def regressors_preds(directory):
    results_file = './results/hybrid_arima/rolling'
    derivado = directory.split('/')[-2]
    results_derivado = f'{results_file}/{derivado}'
    os.makedirs(results_derivado, exist_ok=True)
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            full_path = os.path.join(directory, file)
            uf = file.split("_")[1].upper()
            series = read_series(full_path)
            csv_path = f'{results_derivado}/transform_{uf}.csv'
            
            train, test = train_test_stats(series, horizon)
            transformer = STLTransformer(sp=12)  
            result = transformer.fit(train)
            train = result.trend_ + result.seasonal_
            train_resid = result.resid_

            file_path = f"./results/arima/rolling/{derivado}/transform_{uf}.csv"
            # file_path_mean = f"./results/arima_mean/rolling/{derivado}/transform_{uf}.csv"
            # preds_arima_mean = get_arima_mean(file_path_mean, "combination", test.index)
            preds_arima, params = get_predictions_csv(file_path, "normal", test.index)
            order = (params['p'], params['d'], params['q'])
            seasonal_order = (params['p'], params['d'], params['q'], 12)
            train_normal = transform_train(train, format="normal", horizon=horizon)
            _, preds_noresid, final_order = fit_arima_train(train_normal, train, order, horizon, format="normal")
            # _, preds_noresid, final_order = fit_sarima_train(train_normal, train, order, seasonal_order, horizon, format="deseasonal")



            if not os.path.exists(csv_path):
                pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
            #teste
            # previsao do ruido
            transformer = STLTransformer(sp=12)  
            result = transformer.fit(series)
            series_resid = result.resid_
            data = rolling_window(pd.concat([train_resid, series_resid[-12:]]), window)
            data = data.dropna()
            X_train, X_test, y_train, y_test = train_test_split(data, horizon)

            rf = RandomForestRegressor()
            rf.fit(X_train, y_train)
            predictions = recursive_multistep_forecasting(X_test, rf, horizon)
            mean_norm, std_norm = get_stats_norm(series_resid, horizon, window)

           
            preds_resid = znorm_reverse(np.array(predictions), mean_norm, std_norm)
            preds_resid = pd.Series(preds_resid, index=test.index)
            preds_hybrid = preds_noresid + preds_resid
            # preds_arima_hybrid = pd.concat([preds_arima.iloc[:6], preds_hybrid.iloc[6:]])
            # preds_arimamean_hybrid = pd.concat([preds_arima_mean.iloc[:6], preds_hybrid.iloc[6:]])


            all_preds = [("arima-trend-seasonal", preds_hybrid)]

            for tipo, preds_real in all_preds:
                print_log(f'[PREDS] de {tipo}')
                y_test_rescaled = series[-horizon:].values
                mape_result = mape(y_test_rescaled, preds_real)

                pbe_result = pbe(y_test_rescaled, preds_real)
                pocid_result = pocid(y_test_rescaled, preds_real)
                rmse_result = rmse(y_test_rescaled, preds_real)
                mcpm_result = mcpm(rmse_result, mape_result, pocid_result)

                print_log(f'MAPE: {mape_result}')
                print_log(f'PBE: {pbe_result}')
                print_log(f'POCID: {pocid_result}')
                df_temp = pd.DataFrame({'DATA': tipo, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'HYBRID','PARAMS': str(final_order), 'WINDOW': window, 'HORIZON': horizon,  
                                                'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 
                                                'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                                'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                                'P11': preds_real[10], 'P12': preds_real[11]
                                                }, index=[0])
                df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)

def arima_components_series(args):
    results_file = './results/arima_stl/rolling'
    directory, file = args
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

            tf_series = STLTransformer(sp=12)  
            rs = tf_series.fit(series)

            _, test_seasonal = train_test_stats(rs.seasonal_, horizon)
            _, test_trend = train_test_stats(rs.trend_, horizon)
            _, test_resid = train_test_stats(rs.resid_, horizon)
            _, test_seasonal_trend = train_test_stats(rs.seasonal_ + rs.trend_, horizon)
            
            train_stl, _ = train_test_stats(series, horizon)
          
            #teste
            transformer = STLTransformer(sp=12)  
            result = transformer.fit(train_stl)
            train_trend = result.trend_ 
            train_seasonal = result.seasonal_
            train_resid = result.resid_
            train_seasonal_trend = train_seasonal + train_trend

            train_normal_trend = transform_train(train_trend, format=transformacao, horizon=horizon)
            train_normal_seasonal = transform_train(train_seasonal, format=transformacao, horizon=horizon)
            train_normal_resid = transform_train(train_resid, format=transformacao, horizon=horizon)
            train_normal_seasonal_trend = transform_train(train_seasonal_trend, format=transformacao, horizon=horizon)

            #validacao
            train_val_trend, test_val_trend = train_test_stats(train_trend, horizon)
            train_val_seasonal, test_val_seasonal = train_test_stats(train_seasonal, horizon)
            train_val_resid, test_val_resid = train_test_stats(train_resid, horizon)
            train_val_seasonal_trend, test_val_seasonal_trend = train_test_stats(train_seasonal_trend, horizon)

            train_val_normal_trend = transform_train(train_val_trend, format=transformacao, horizon=horizon)
            train_val_normal_seasonal = transform_train(train_val_seasonal, format=transformacao, horizon=horizon)
            train_val_normal_resid = transform_train(train_val_resid, format=transformacao, horizon=horizon)
            train_val_normal_seasonal_trend = transform_train(train_val_seasonal_trend, format=transformacao, horizon=horizon)

            #acoplamento

            components = [
                            
                            {'name': 'seasonal', 'train': train_normal_seasonal,'test': test_seasonal, 'train_val': train_val_normal_seasonal, 'test_val': test_val_seasonal, 'train_real': train_seasonal, 'train_val_real': train_val_seasonal},
                            {'name': 'trend', 'train': train_normal_trend,'test': test_trend, 'train_val': train_val_normal_trend, 'test_val': test_val_trend, 'train_real': train_trend, 'train_val_real': train_val_trend},
                            {'name': 'resid', 'train': train_normal_resid, 'test': test_resid, 'train_val': train_val_normal_resid, 'test_val': test_val_resid, 'train_real': train_resid, 'train_val_real': train_val_resid},
                            {'name': 'seasonal_trend', 'train': train_normal_seasonal_trend, 'test': test_seasonal_trend, 'train_val': train_val_normal_seasonal_trend, 'test_val': test_val_seasonal_trend, 'train_real': train_seasonal_trend, 'train_val_real': train_val_seasonal_trend}
                         ]

            path_derivado = f'{results_file}/{derivado}'
            os.makedirs(path_derivado, exist_ok=True)
            csv_path = f'{path_derivado}/transform_{uf}.csv'
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
            for component in components:
                derivado_result = results_file+"/"+derivado
                flag = checkFolder(derivado_result, f"transform_{uf}.csv", component['name'])
                if flag:
                    results_arima = find_best_parameter(component['train_val'], component['test_val'], component['train_val_real'])
                    print_log(f"----------------------[VALIDACAO] ENCONTRADO PARAMETROS PARA {derivado} | {component['name']} em {uf} ------------------------------")
                    initial_order = (results_arima['best_params']['p'], results_arima['best_params']['d'], results_arima['best_params']['q'])
                    _, preds_real, final_order = fit_arima_train(component['train'], component['train_real'], initial_order, horizon, format=transformacao)
                    
                    test = component['test']
                    rmse_result = rmse(test, preds_real)
                    mape_result = mape(test, preds_real)
                    pocid_result = pocid(test, preds_real)
                    pbe_result = pbe(test, preds_real)
                    mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                    print_log('[RESULTADO EM TRAIN]')
                    print_log(f'PARAMS: {str(final_order)}')
                    print_log(f'MCPM: {mcpm_result}')
                    print_log(f'RMSE: {rmse_result}')
                    print_log(f'MAPE: {mape_result}')
                    print_log(f'POCID: {pocid_result}')
                    print_log(f'PBE: {pbe_result}')

                    print_log(f"---------------------- [FINALIZADO] {derivado} | {component['name']} em {uf} ------------------------------")
                    df_temp = pd.DataFrame({'DATA': component['name'], 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ARIMA', 'PARAMS': str(final_order), 'WINDOW': window, 'HORIZON': horizon,  
                                            'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 
                                            'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                            'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                            'P11': preds_real[10], 'P12': preds_real[11]
                                            }, index=[0])
                    df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)
                    
        except Exception as e:
            print_log(f"Exception: {derivado} em {uf}: {e}\n")
            traceback.print_exc()

def NN_seasonal_trend(args):
    s=12 # quantidade de pontos para comparar da serie
    k=1 # quantidade de series similar
    dist_metrics = ["euclidean", "mahalanobis", "dtw"]
    results_file = './results/nn_seasonal_trend/rolling'
    directory, file = args
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

            tf_series = STLTransformer(sp=6)  
            rs = tf_series.fit(series)

            _, test_seasonal = train_test_stats(rs.seasonal_, horizon)
            _, test_trend = train_test_stats(rs.trend_, horizon)
            _, test_seasonal_trend = train_test_stats(rs.seasonal_ + rs.trend_, horizon)
            
            train_stl, _ = train_test_stats(series, horizon)
          
            #teste
            transformer = STLTransformer(sp=6)  
            result = transformer.fit(train_stl)
            train_trend = result.trend_ 
            train_seasonal = result.seasonal_
            train_seasonal_trend = train_seasonal + train_trend

            train_normal_trend = transform_train(train_trend, format=transformacao, horizon=horizon)
            train_normal_seasonal = transform_train(train_seasonal, format=transformacao, horizon=horizon)
            train_normal_seasonal_trend = transform_train(train_seasonal_trend, format=transformacao, horizon=horizon)

            #validacao
            train_val_trend, test_val_trend = train_test_stats(train_trend, horizon)
            train_val_seasonal, test_val_seasonal = train_test_stats(train_seasonal, horizon)
            train_val_seasonal_trend, test_val_seasonal_trend = train_test_stats(train_seasonal_trend, horizon)

            train_val_normal_trend = transform_train(train_val_trend, format=transformacao, horizon=horizon)
            train_val_normal_seasonal = transform_train(train_val_seasonal, format=transformacao, horizon=horizon)
            train_val_normal_seasonal_trend = transform_train(train_val_seasonal_trend, format=transformacao, horizon=horizon)

            #acoplamento

            components = [
                            
                            {'name': 'seasonal', 'train': train_normal_seasonal,'test': test_seasonal, 'train_val': train_val_normal_seasonal, 'test_val': test_val_seasonal, 'train_real': train_seasonal, 'train_val_real': train_val_seasonal},
                            {'name': 'trend', 'train': train_normal_trend,'test': test_trend, 'train_val': train_val_normal_trend, 'test_val': test_val_trend, 'train_real': train_trend, 'train_val_real': train_val_trend},
                            {'name': 'seasonal_trend', 'train': train_normal_seasonal_trend, 'test': test_seasonal_trend, 'train_val': train_val_normal_seasonal_trend, 'test_val': test_val_seasonal_trend, 'train_real': train_seasonal_trend, 'train_val_real': train_val_seasonal_trend}
                         ]

            path_derivado = f'{results_file}/{derivado}'
            os.makedirs(path_derivado, exist_ok=True)
            csv_path = f'{path_derivado}/transform_{uf}.csv'
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
            for component in components:
                derivado_result = results_file+"/"+derivado
                for metric in dist_metrics:
                    type_data = f"{component['name']}_{metric}"
                    flag = checkFolder(derivado_result, f"transform_{uf}.csv", type_data)
                    if flag:
                        # results_arima = find_best_parameter(component['train_val'], component['test_val'], component['train_val_real'])
                        # initial_order = (results_arima['best_params']['p'], results_arima['best_params']['d'], results_arima['best_params']['q'])
                        # _, preds_real, final_order = fit_arima_train(component['train'], component['train_real'], initial_order, horizon, format=transformacao)
                        test = component['test']
                        results = knn_similar_series(component['train'][:-s], component['train'][-s:], k, metric)
                        _, mean, std = rolling_window_series(component['train_real'], 12)


                        last_date = results[0]['similar_sequence'].index[-1]
                        
                        start_idx = component['train'].index.get_loc(last_date) + 1
                        end_idx = min(start_idx + horizon, len(component['train']))
                        preds_similar = component['train'].iloc[start_idx:end_idx]

                        preds_transformed = znorm_reverse(preds_similar, mean, std)
                        # try:
                        preds_real = pd.Series(preds_transformed.values, test.index)
                        # except:
                        #     print_log(f"----------------------ERRO PARA {derivado} | {type_data} em {uf} ------------------------------")
                        #     print_log(f"TRAIN BUSCA: {component['train'][:-s]}")
                        #     print_log(f"serie BUSCA: {component['train'][-s:]}")
                        #     print_log(f"preds: {preds_transformed}")
                    
                        rmse_result = rmse(test, preds_real)
                        mape_result = mape(test, preds_real)
                        pocid_result = pocid(test, preds_real)
                        pbe_result = pbe(test, preds_real)
                        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                        print_log('[RESULTADO EM TRAIN]')
                        print_log(f'MCPM: {mcpm_result}')
                        print_log(f'RMSE: {rmse_result}')
                        print_log(f'MAPE: {mape_result}')
                        print_log(f'POCID: {pocid_result}')
                        print_log(f'PBE: {pbe_result}')

                        print_log(f"---------------------- [FINALIZADO] {derivado} | {type_data} em {uf} ------------------------------")
                        df_temp = pd.DataFrame({'DATA': type_data, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ARIMA', 'PARAMS': metric, 'WINDOW': window, 'HORIZON': horizon,  
                                                'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 
                                                'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                                'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                                'P11': preds_real[10], 'P12': preds_real[11]
                                                }, index=[0])
                        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)
                    
        except Exception as e:
            print_log(f"Exception: {derivado} em {uf}: {e}\n")
            traceback.print_exc()

def sum_components(args):
    directory, file = args
    results_file = './results/nn_seasonal_trend_rf'
    results_nn = './results/nn/rolling'
    results_stl = './results/arima_stl/rolling'
    results_rf = './results/rf/rolling'
    derivado = directory.split('/')[-2]
    results_derivado = f'{results_file}/{derivado}'
    os.makedirs(results_derivado, exist_ok=True)
    if file.endswith('.csv'):
        uf = file.split("_")[1].upper()
        derivado = file.split("_")[2].split(".")[0]
        full_path = os.path.join(directory, file)
         
        series = read_series(full_path)
        train, test = train_test_stats(series, horizon)

        path_nn = f"{results_nn}/{derivado}/transform_{uf}.csv"
        path_rf = f"{results_rf}/{derivado}/transform_{uf}.csv"
        path_stl = f"{results_stl}/{derivado}/transform_{uf}.csv"

        preds_seasonal, _ = get_predictions_csv(path_stl, "seasonal", test.index)
        preds_trend, _ = get_predictions_csv(path_nn, "trend_mahalanobis", test.index)
        # preds_resid, _ = get_predictions_csv(path_rf, "resid", test.index)

        #prevendo ruido
        transformer = STLTransformer(sp=6)  
        result = transformer.fit(train)
        train_resid = result.resid_

        transformer = STLTransformer(sp=12)  
        result = transformer.fit(series)
        series_resid = result.resid_
        data = rolling_window(pd.concat([train_resid, series_resid[-12:]]), window)
        data = data.dropna()
        X_train, X_test, y_train, y_test = train_test_split(data, horizon)

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        predictions = recursive_multistep_forecasting(X_test, rf, horizon)
        mean_norm, std_norm = get_stats_norm(series_resid, horizon, window)

        
        preds_resid = znorm_reverse(np.array(predictions), mean_norm, std_norm)
        preds_resid = pd.Series(preds_resid, index=test.index)

        preds_real = preds_seasonal + preds_trend + preds_resid

        y_baseline = series[-horizon*2:-horizon].values
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
        print_log(f"---------------------- [FINALIZADO] {derivado} | em {uf} ------------------------------")
        csv_path = f'{results_derivado}/transform_{uf}.csv'
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
        df_temp = pd.DataFrame({'DATA': 'arima_nn_rf', 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ARIMA', 'PARAMS': 'arima_nn_rf', 'WINDOW': window, 'HORIZON': horizon,  
                                'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 'MASE': mase_result,
                                'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                'P11': preds_real[10], 'P12': preds_real[11]
                                }, index=[0])
        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)

def for_regressor(args):
    directory, file = args
    results_file = './paper2/knn'
    derivado = file.split("_")[2].split(".")[0]
    results_derivado = f'{results_file}/{derivado}'
    os.makedirs(results_derivado, exist_ok=True)
   
    if file.endswith('.csv'):
        uf = file.split("_")[1].upper()
        full_path = os.path.join(directory, file)
         
        series = read_series(full_path)
        train, _ = train_test_stats(series, horizon) 
        # train_tf = np.log(train) #log
        data = rolling_window(series, window)
        # data = rolling_window_transform(series, window) #log
        data = data.dropna()
        X_train, X_test, y_train, _ = train_test_split(data, horizon)
        X_train_v, X_test_v, y_train_v, _ = train_test_split(X_train, horizon)
        results_xgb = find_best_parameter_xgb(X_train_v, X_test_v, y_train_v, train)
        # xgb = xgboost.XGBRegressor(**results_xgb['best_params']) 
        # xgb = RandomForestRegressor(**results_xgb['best_params'])
        xgb = KNeighborsRegressor(**results_xgb['best_params'])
        xgb.fit(X_train, y_train)

        predictions = recursive_multistep_forecasting(X_test, xgb, horizon)
        mean_norm, std_norm = get_stats_norm(series, horizon, window)
        # mean_norm = np.mean(train_tf[-horizon:]) #log
        # std_norm = np.std(train_tf[-horizon:]) #log
        # preds_log = znorm_reverse(np.array(predictions), mean_norm, std_norm) #log
        preds_real = znorm_reverse(np.array(predictions), mean_norm, std_norm)

        # preds_real = np.exp(preds_log) #log

        y_test_rescaled = series[-horizon:].values
        y_baseline = series[-horizon*2:-horizon].values

        mape_result = mape(y_test_rescaled, preds_real)
        pbe_result = pbe(y_test_rescaled, preds_real)
        pocid_result = pocid(y_test_rescaled, preds_real)
        rmse_result = rmse(y_test_rescaled, preds_real)
        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
        mase_result = mase(y_test_rescaled, preds_real, y_baseline)

        print_log(f'MAPE: {mape_result}')
        print_log(f'PBE: {pbe_result}')
        print_log(f'POCID: {pocid_result}')
        csv_path = f'{results_derivado}/transform_{uf}.csv'
        print_log(f"---------------------- [FINALIZADO] {derivado} | em {uf} ------------------------------")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
        df_temp = pd.DataFrame({'DATA': 'normal', 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'knn','PARAMS': str(results_xgb['best_params']), 'WINDOW': window, 'HORIZON': horizon,  
                                        'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 'MASE': mase_result,
                                        'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                        'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                        'P11': preds_real[10], 'P12': preds_real[11]
                                        }, index=[0])
        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)

def regressor_error_calculator(args):
    regr = 'xgb'
    transformation = 'log'
    directory, file = args
    results_file = f'./paper2/norm_{regr}_error'
    derivado = file.split("_")[2].split(".")[0]
    results_derivado = f'{results_file}/{derivado}'
    os.makedirs(results_derivado, exist_ok=True)
   
    if file.endswith('.csv'):
        uf = file.split("_")[1].upper()
        full_path = os.path.join(directory, file)
         
        series = read_series(full_path)
        train, _ = train_test_stats(series, horizon) 
        train_tf = np.log(train) #log
        if transformation == 'log':
            data = rolling_window_transform(series, window) #log
        else:
            data = rolling_window(series, window)

        data = data.dropna()
        X_train, X_test, y_train, _ = train_test_split(data, horizon)
        # X_train_v, X_test_v, y_train_v, _ = train_test_split(X_train, horizon)
        # results_rg = find_best_parameter_xgb(X_train_v, X_test_v, y_train_v, train)['best_params']
        
        #validacao de errors
        rolling_val_stats = rolling_validation_stats(train)

        results_rg = get_params_model(f'./paper2/{regr}/{derivado}/transform_{uf}.csv', transformation)

        if regr == 'xgb':
            rg_val = xgboost.XGBRegressor(**results_rg) 
        elif regr == 'rf':
            rg_val = RandomForestRegressor(**results_rg)
        elif regr == 'knn':
            rg_val = KNeighborsRegressor(**results_rg)
        else:
            raise ValueError('nao existe esse regressor')

        tam_horizon = range(1, horizon+1)
        errors_h = {h: [] for h in tam_horizon}

        for i in range(0,13):
            X_rolling_train, X_rolling_test, y_rolling_train, _ = rolling_validation_regressors(X_train, y_train)[i]

            rg_val.fit(X_rolling_train, y_rolling_train)
            if transformation == 'log':
                test_tf = np.log(rolling_val_stats[i][1]) #log
            else:
                test_tf = rolling_val_stats[i][1]
                
            predictions = recursive_multistep_forecasting(X_rolling_test, rg_val, horizon)

            #aplicando media e desvio sempre dos 12 ultimos de treino
            # mean_norm_v = np.mean(rolling_val_stats[i][0].values[-horizon:])
            # std_norm_v = np.std(rolling_val_stats[i][1].values[-horizon:])
          
            #media e desvio do pedaço real da propria validacao
            mean_norm_v = np.mean(test_tf)
            std_norm_v = np.std(test_tf)
            preds_real_v = znorm_reverse(np.array(predictions), mean_norm_v, std_norm_v)

            if transformation == 'log':
                preds_real_v = np.exp(preds_real_v)

            erro =  (rolling_val_stats[i][1].values - preds_real_v).tolist()

            for h, err in zip(tam_horizon, erro):
                errors_h[h].append(err)

        max_horizon = {h: np.max(erros) for h, erros in errors_h.items()}
        
        last_preds_erro = rolling_val_stats[12][1] + list(max_horizon.values())    

        #teste
        if regr == 'xgb':
            rg = xgboost.XGBRegressor(**results_rg) 
        elif regr == 'rf':
            rg = RandomForestRegressor(**results_rg)
        elif regr == 'knn':
            rg = KNeighborsRegressor(**results_rg)
        else:
            raise ValueError('nao existe esse regressor')
        
        rg.fit(X_train, y_train)

        predictions = recursive_multistep_forecasting(X_test, rg, horizon)
        if transformation == 'log':
            mean_norm = np.mean(np.log(last_preds_erro)) #log
            std_norm = np.std(np.log(last_preds_erro)) #log
        else:
            # mean_norm, std_norm = get_stats_norm(series, horizon, window)
            mean_norm = np.mean(last_preds_erro) #log
            std_norm = np.std(last_preds_erro) #log
        # preds_log = znorm_reverse(np.array(predictions), mean_norm, std_norm) #log
        preds_real = znorm_reverse(np.array(predictions), mean_norm, std_norm)
        if transformation == 'log':
            preds_real = np.exp(preds_real)

        # preds_somadas  = [a + b for a, b in zip(mean_horizon.values(), preds_real)]

        # preds_somadas  = [a + b for a, b in zip(erros_previstos, preds_real)] #prever erro ao inves de max/mean


        y_test_rescaled = series[-horizon:].values
        y_baseline = series[-horizon*2:-horizon].values

        mape_result = mape(y_test_rescaled, preds_real)
        pbe_result = pbe(y_test_rescaled, preds_real)
        pocid_result = pocid(y_test_rescaled, preds_real)
        rmse_result = rmse(y_test_rescaled, preds_real)
        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
        mase_result = mase(y_test_rescaled, preds_real, y_baseline)

        print_log(f'MAPE: {mape_result}')
        print_log(f'PBE: {pbe_result}')
        print_log(f'POCID: {pocid_result}')
        csv_path = f'{results_derivado}/transform_{uf}.csv'
        print_log(f"---------------------- [FINALIZADO] {derivado} | em {uf} ------------------------------")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
        df_temp = pd.DataFrame({'DATA': transformation, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': regr,'PARAMS': str(results_rg), 'WINDOW': window, 'HORIZON': horizon,  
                                        'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 'MASE': mase_result,
                                        'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                        'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                        'P11': preds_real[10], 'P12': preds_real[11]
                                        }, index=[0])
        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)

def regressor_adjusted(args):
    regr = 'knn'
    transformation = 'normal'
    directory, file = args
    results_file = f'./paper2/adjusted_{regr}'
    derivado = file.split("_")[2].split(".")[0]
    results_derivado = f'{results_file}/{derivado}'
    os.makedirs(results_derivado, exist_ok=True)
   
    if file.endswith('.csv'):
        uf = file.split("_")[1].upper()
        full_path = os.path.join(directory, file)
         
        series = read_series(full_path)
        train, _ = train_test_stats(series, horizon) 
        train_tf = np.log(train) #log
        if transformation == 'log':
            data = rolling_window_transform(series, window) #log
        else:
            data = rolling_window(series, window)

        data = data.dropna()
        X_train, X_test, y_train, _ = train_test_split(data, horizon)
        # X_train_v, X_test_v, y_train_v, _ = train_test_split(X_train, horizon)
        # results_rg = find_best_parameter_xgb(X_train_v, X_test_v, y_train_v, train)['best_params']
        
        #validacao de errors
        rolling_val_stats = rolling_validation_stats(train)

        results_rg = get_params_model(f'./paper2/{regr}/{derivado}/transform_{uf}.csv', transformation)

        if regr == 'xgb':
            rg_val = xgboost.XGBRegressor(**results_rg) 
        elif regr == 'rf':
            rg_val = RandomForestRegressor(**results_rg)
        elif regr == 'knn':
            rg_val = KNeighborsRegressor(**results_rg)
        else:
            raise ValueError('nao existe esse regressor')

        tam_horizon = range(1, horizon+1)
        errors_h = {h: [] for h in tam_horizon}

        for i in range(0,13):
            X_rolling_train, X_rolling_test, y_rolling_train, _ = rolling_validation_regressors(X_train, y_train)[i]

            rg_val.fit(X_rolling_train, y_rolling_train)
            if transformation == 'log':
                test_tf = np.log(rolling_val_stats[i][1]) #log
            else:
                test_tf = rolling_val_stats[i][1]
                
            predictions = recursive_multistep_forecasting(X_rolling_test, rg_val, horizon)

            #aplicando media e desvio sempre dos 12 ultimos de treino
            # mean_norm_v = np.mean(rolling_val_stats[i][0].values[-horizon:])
            # std_norm_v = np.std(rolling_val_stats[i][1].values[-horizon:])
          
            #media e desvio do pedaço real da propria validacao
            mean_norm_v = np.mean(test_tf)
            std_norm_v = np.std(test_tf)
            preds_real_v = znorm_reverse(np.array(predictions), mean_norm_v, std_norm_v)

            if transformation == 'log':
                preds_real_v = np.exp(preds_real_v)

            erro =  (rolling_val_stats[i][1].values - preds_real_v).tolist()

            for h, err in zip(tam_horizon, erro):
                errors_h[h].append(err)

        max_horizon = {h: np.max(erros) for h, erros in errors_h.items()}
        
        last_preds_erro = rolling_val_stats[12][1] + list(max_horizon.values())    

        #teste
        if regr == 'xgb':
            rg = xgboost.XGBRegressor(**results_rg) 
        elif regr == 'rf':
            rg = RandomForestRegressor(**results_rg)
        elif regr == 'knn':
            rg = KNeighborsRegressor(**results_rg)
        else:
            raise ValueError('nao existe esse regressor')
        
        rg.fit(X_train, y_train)

        predictions = recursive_multistep_forecasting(X_test, rg, horizon)
        if transformation == 'log':
            mean_norm = np.mean(np.log(last_preds_erro)) #log
            std_norm = np.std(np.log(last_preds_erro)) #log
        else:
            # mean_norm, std_norm = get_stats_norm(series, horizon, window)
            mean_norm = np.mean(last_preds_erro) #log
            std_norm = np.std(last_preds_erro) #log
        # preds_log = znorm_reverse(np.array(predictions), mean_norm, std_norm) #log
        preds_real = znorm_reverse(np.array(predictions), mean_norm, std_norm)
        if transformation == 'log':
            preds_real = np.exp(preds_real)

        # preds_somadas  = [a + b for a, b in zip(mean_horizon.values(), preds_real)]

        # preds_somadas  = [a + b for a, b in zip(erros_previstos, preds_real)] #prever erro ao inves de max/mean


        y_test_rescaled = series[-horizon:].values
        y_baseline = series[-horizon*2:-horizon].values

        mape_result = mape(y_test_rescaled, preds_real)
        pbe_result = pbe(y_test_rescaled, preds_real)
        pocid_result = pocid(y_test_rescaled, preds_real)
        rmse_result = rmse(y_test_rescaled, preds_real)
        mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
        mase_result = mase(y_test_rescaled, preds_real, y_baseline)

        print_log(f'MAPE: {mape_result}')
        print_log(f'PBE: {pbe_result}')
        print_log(f'POCID: {pocid_result}')
        csv_path = f'{results_derivado}/transform_{uf}.csv'
        print_log(f"---------------------- [FINALIZADO] {derivado} | em {uf} ------------------------------")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
        df_temp = pd.DataFrame({'DATA': transformation, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': regr,'PARAMS': str(results_rg), 'WINDOW': window, 'HORIZON': horizon,  
                                        'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 'MASE': mase_result,
                                        'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                        'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                        'P11': preds_real[10], 'P12': preds_real[11]
                                        }, index=[0])
        df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)


if __name__ == '__main__':
#   for directory in dirs:
#     regressors_preds(directory)
    with multiprocessing.Pool() as pool:
            tasks = [
                (directory, file) 
                for directory in dirs 
                for file in os.listdir(directory) 
            ]

            pool.map(regressor_error_calculator, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")