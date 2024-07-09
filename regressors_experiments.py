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

        rg = xgboost.XGBRegressor(**hyper_par)
        # rg = RandomForestRegressor(**hyper_par)
        # rg = KNeighborsRegressor(**hyper_par)
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
        'learning_rate': [0.2, 0.3, 0.4], #xgb
        'subsample': [0.7, 0.8, 0.9], #xgb
        'n_estimators': [50, 100, 150, 200] #rf/xgb
        # 'n_neighbors': [1,3,5,7,9]
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

def transform_regressors(train, format='normal'):
    if format == 'deseasonal':
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train)
        train_deseasonal = transform.transform(train)

        return train_deseasonal
    elif format == 'log':
        train_log = np.log(train)
        return train_log
    elif format == 'normal':
        return train

def reverse_regressors(train_real, preds, format='normal'):
    if format == 'deseasonal':
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train_real)
        series_before_norm = transform.transform(train_real)

        _, mean, std = rolling_window_series(series_before_norm, 12)
        preds_transformed = znorm_reverse(preds, mean, std)

        series_real = transform.inverse_transform(preds_transformed)
        return series_real
    elif format == 'log':
        series_before_norm = np.log(train_real)
        
        _, mean, std = rolling_window_series(series_before_norm, 12)
        preds_transformed = znorm_reverse(preds, mean, std)

        return np.exp(preds_transformed)
    elif format == 'normal':
        _, mean, std = rolling_window_series(train_real, 12)
        preds_real = znorm_reverse(preds, mean, std)
        return preds_real
    
    raise ValueError('nao existe essa transformacao')


def for_regressor(args):
    directory, file = args
    regr = 'xgb'
    results_file = f'./paper2/{regr}'
    derivado = file.split("_")[2].split(".")[0]
    results_derivado = f'{results_file}/{derivado}'
    os.makedirs(results_derivado, exist_ok=True)
    transformations = ['normal', 'log', 'deseasonal']
    if file.endswith('.csv'):
        uf = file.split("_")[1].upper()
        full_path = os.path.join(directory, file)
         
        series = read_series(full_path)
        train, test_unused = train_test_stats(series, horizon) 

        for tf in transformations:
            train_tf = transform_regressors(train, tf)

            
            # train_tf = np.log(train) #log
            data = rolling_window(pd.concat([train_tf, pd.Series([1,2,3,4,5,6,7,8,9,10,11,12], index=test_unused.index)]), window)
            # data = rolling_window_transform(series, window) #log
            data = data.dropna()
            X_train, X_test, y_train, _ = train_test_split(data, horizon)
            X_train_v, X_test_v, y_train_v, _ = train_test_split(X_train, horizon)
            results_rg = find_best_parameter_xgb(X_train_v, X_test_v, y_train_v, train_tf)['best_params']
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
            preds = pd.Series(predictions, index=test_unused.index)
            preds_real = reverse_regressors(train, preds, format=tf)

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
            df_temp = pd.DataFrame({'DATA': tf, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': f'{regr}','PARAMS': str(results_rg), 'WINDOW': window, 'HORIZON': horizon,  
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

            pool.map(for_regressor, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")