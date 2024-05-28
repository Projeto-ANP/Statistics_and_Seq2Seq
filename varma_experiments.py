import multiprocessing
from all_functions import *
import os
from mango import Tuner
from concurrent.futures import ThreadPoolExecutor
from statsmodels.tsa.statespace.varmax import VARMAX
import statistics
from sklearn.metrics import mean_absolute_percentage_error as mape
from multiprocessing import Process
from collections import Counter
import ast
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

horizon = 12
window = 12
transformations = ["normal", "deseasonal", "deseasonal-log", "deseasonal-diff", "diff", "log", "log-diff"]

colunas = ['DATA', 'MCPM', 'UF', 'PRODUCT', 'MODEL', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12'
           ]
df_result = pd.DataFrame(columns=colunas)
results_file = './results/varma/rolling'


def find_best_parameter(derivado, transformation):
    p_counter = Counter()
    q_counter = Counter()
    results_arima = f"./results/arima/rolling/{derivado}/"
    files = [f for f in os.listdir(results_arima) if f.endswith('.csv')]
  
    for file in files:
        file_path = os.path.join(results_arima, file)
        df = pd.read_csv(file_path, sep=";")
        filtered_df = df[df['DATA'] == transformation]
        for params in filtered_df['PARAMS']:
            params_dict = ast.literal_eval(params)
            p_counter[params_dict['p']] += 1
            q_counter[params_dict['q']] += 1

    p = p_counter.most_common(1)[0][0] if p_counter else 2
    q = q_counter.most_common(1)[0][0] if q_counter else 2

    return p, q

def process_directory(directory):
    #validação
    # train_val_tf_dict = {}
    # test_val_dict = {}
    # train_val_dict = {}
    
    #teste
    train_dict = {}
    test_dict = {}
    train_tf_dict = {}

    derivado = directory.split('/')[-2]
    path_derivado = f'{results_file}/{derivado}'
    os.makedirs(path_derivado, exist_ok=True)
    
    for transform in transformations:
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                full_path = os.path.join(directory, file)
                uf = file.split("_")[1].upper()
                series = read_series(full_path)
                #teste
                train, test = train_test_stats(series, horizon)
                train_tf = transform_train(train, format=transform)
                train_tf_dict[uf] = train_tf
                test_dict[uf] = test
                train_dict[uf] = train
                #validação
                # train_val, test_val = train_test_stats(train, horizon)
                # train_tf_val = transform_train(train_val, format=transform)
                # train_val_tf_dict[uf] = train_tf_val
                # test_val_dict[uf] = test_val
                # train_val_dict[uf] = train_val
        #validação
        # df_train_val_tf = pd.DataFrame(train_val_tf_dict)
        # df_test_val = pd.DataFrame(test_val_dict)
        # df_train_val = pd.DataFrame(train_val_dict)

        #teste
        df_train_tf = pd.DataFrame(train_tf_dict)
        df_test = pd.DataFrame(test_dict)
        df_train = pd.DataFrame(train_dict)
        
        #do tuning-parameter
        print_log(f"[ENCONTRANDO] PARAMETROS PARA {transform} em {derivado}")
        p,q = find_best_parameter(derivado, transform)
        print_log(f"[FINALIZADO] BUSCA DE PARAMETROS VALIDAÇÃO PARA {transform} em {derivado}")
        order = (p,q)
        print(str(order))
        model = VARMAX(df_train_tf, freq = 'M', order=order)
        fitted_model = model.fit(disp=False)

        #do predict train and save
        print_log(f"[CALCULANDO] PREDICTIONS EM TRAIN PARA {transform} em {derivado}")
        predictions = fitted_model.forecast(steps=horizon)
        for column in predictions.columns:
            csv_path = f'{path_derivado}/transform_{column}.csv'
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
            preds_real = reverse_transform_norm_preds(predictions[column], df_train[column], format=transform)
            rmse_result = rmse(df_test[column], preds_real)
            mape_result = mape(df_test[column], preds_real)
            pocid_result = pocid(df_test[column], preds_real)
            pbe_result = pbe(df_test[column], preds_real)
            mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
            print_log('[RESULTADO EM TRAIN]')
            print_log(f'PARAMS: {str(order)}')
            print_log(f'MCPM: {mcpm_result}')
            print_log(f'RMSE: {rmse_result}')
            print_log(f'MAPE: {mape_result}')
            print_log(f'POCID: {pocid_result}')
            print_log(f'PBE: {pbe_result}')

            df_temp = pd.DataFrame({'DATA': transform, 'MCPM': mcpm_result, 'UF': column, 'PRODUCT': derivado, 'MODEL': 'VARMA', 'PARAMS': str(order), 'WINDOW': window, 'HORIZON': horizon,  
                                            'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 
                                            'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                            'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                            'P11': preds_real[10], 'P12': preds_real[11]
                                            }, index=[0])
            df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)


def process_file():
    #validação
    train_val_tf_dict = {}
    test_val_dict = {}
    train_val_dict = {}
    
    #teste
    train_dict = {}
    test_dict = {}
    train_tf_dict = {}
    
    for directory in dirs:
        derivado = directory.split('/')[-2]
        path_derivado = f'{results_file}/{derivado}'
        os.makedirs(path_derivado, exist_ok=True)
        for transform in transformations:
            for file in os.listdir(directory): 
                if file.endswith('.csv'):
                    full_path = os.path.join(directory, file)
                    uf = file.split("_")[1].upper()
                    series = read_series(full_path)
                    #teste
                    train, test = train_test_stats(series, horizon)
                    train_tf = transform_train(train, format=transform)
                    train_tf_dict[uf] = train_tf
                    test_dict[uf] = test
                    train_dict[uf] = train
                    #validação
                    train_val, test_val = train_test_stats(train, horizon)
                    train_tf_val = transform_train(train_val, format=transform)
                    train_val_tf_dict[uf] = train_tf_val
                    test_val_dict[uf] = test_val
                    train_val_dict[uf] = train_val


            #validação
            df_train_val_tf = pd.DataFrame(train_val_tf_dict)
            df_test_val = pd.DataFrame(test_val_dict)
            df_train_val = pd.DataFrame(train_val_dict)

            #teste
            df_train_tf = pd.DataFrame(train_tf_dict)
            df_test = pd.DataFrame(test_dict)
            df_train = pd.DataFrame(train_dict)
            
            #do tuning-parameter
            results_arima = find_best_parameter(df_train_val_tf, df_test_val, df_train_val, transform)
            order = (results_arima['best_params']['p'], results_arima['best_params']['q'])
            model = VARMAX(df_train_tf, freq = 'M', order=order)
            fitted_model = model.fit(disp=False)

            #do predict train and save
            predictions = fitted_model.forecast(steps=horizon)
            for column in predictions.columns:
                csv_path = f'{path_derivado}/transform_{column}.csv'
                if not os.path.exists(csv_path):
                    pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
                preds_real = reverse_transform_norm_preds(predictions[column], df_train[column], format=transform)
                rmse_result = rmse(df_test[column], preds_real)
                mape_result = mape(df_test[column], preds_real)
                pocid_result = pocid(df_test[column], preds_real)
                pbe_result = pbe(df_test[column], preds_real)
                mcpm_result = mcpm(rmse_result, mape_result, pocid_result)
                print_log('[RESULTADO EM TRAIN]')
                print_log(f'PARAMS: {str(order)}')
                print_log(f'MCPM: {mcpm_result}')
                print_log(f'RMSE: {rmse_result}')
                print_log(f'MAPE: {mape_result}')
                print_log(f'POCID: {pocid_result}')
                print_log(f'PBE: {pbe_result}')

                df_temp = pd.DataFrame({'DATA': transform, 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ARIMA', 'PARAMS': str(order), 'WINDOW': window, 'HORIZON': horizon,  
                                                'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 
                                                'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                                'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                                'P11': preds_real[10], 'P12': preds_real[11]
                                                }, index=[0])
                df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)



if __name__ == '__main__':
  for directory in dirs:
    process_directory(directory)