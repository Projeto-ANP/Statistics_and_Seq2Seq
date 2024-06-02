from all_functions import *
import os
import pickle
from VotingCombination import VotingCombination
from sklearn.metrics import mean_absolute_percentage_error as mape
import ast
def get_params_csv(path, format):
    df = pd.read_csv(path, sep=";")
    filtered_df = df[df['DATA'] == format]
    return  ast.literal_eval(filtered_df['PARAMS'].iloc[0])

dirs = [
    '../datasets/venda/mensal/uf/gasolinac',
    '../datasets/venda/mensal/uf/etanolhidratado/',
    '../datasets/venda/mensal/uf/gasolinadeaviacao/',
    '../datasets/venda/mensal/uf/glp/',
    '../datasets/venda/mensal/uf/oleocombustivel/',
    '../datasets/venda/mensal/uf/oleodiesel/',
    '../datasets/venda/mensal/uf/querosenedeaviacao/',
    '../datasets/venda/mensal/uf/queroseneiluminante/',
]

results_arima = "./results/arima/rolling"
transformations = ["normal", "deseasonal", "deseasonal-log", "deseasonal-diff", "diff", "log", "log-diff"]
horizon = 12
window = 12
results_moe = "./results/arima_moe/rolling"

colunas = ['DATA', 'MCPM', 'UF', 'PRODUCT', 'MODEL', 'VALIDATION-MOE', 'PARAMS', 'WINDOW', 'HORIZON', 'RMSE', 'MAPE', 'POCID', 'PBE',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12'
           ]
df_result = pd.DataFrame(columns=colunas)
for directory in dirs:
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            uf = file.split("_")[1].upper()
            derivado = file.split("_")[2].split(".")[0]
            results_derivado = f'{results_moe}/{derivado}'
            os.makedirs(results_derivado, exist_ok=True)
            series = read_series(f"../datasets/venda/mensal/uf/{derivado}/mensal_{uf.lower()}_{derivado}.csv")
            train, test = train_test_stats(series, horizon)
            train_val, test_val = train_test_stats(train, horizon)
            estimators_val = {}
            estimators = {}
            for format in transformations:
                train_tf_val = transform_train(train_val, format=format)
                train_tf = transform_train(train, format=format)

                params = get_params_csv(f"{results_arima}/{derivado}/transform_{uf}.csv", format)
                print(f'{derivado} {uf} {format} - {params}')
                initial_order = (params['p'], params['d'], params['q'])

                #preds validation
                _, preds_val_real, final_val_order = fit_arima_train(train_tf_val, train_val, initial_order, horizon, format=format)
                estimators_val[format] = preds_val_real
                
                #preds test
                _, preds_real, final_order = fit_arima_train(train_tf, train, initial_order, horizon, format=format)
                estimators[format] = preds_real
            print_log(f"[CALCULANDO] PARA {derivado} em {uf}")
            voting = VotingCombination(estimators, combination='moe')
            voting.fit_moe(estimators_val, test_val)
            preds_moe = voting.predict()
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
            csv_path = f'{results_derivado}/transform_{uf}.csv'
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=colunas).to_csv(csv_path, sep=';', index=False)
            df_temp = pd.DataFrame({'DATA': 'moe', 'MCPM': mcpm_result, 'UF': uf, 'PRODUCT': derivado, 'MODEL': 'ARIMA', 'VALIDATION-MOE': voting.get_moe(),'PARAMS': str(final_order), 'WINDOW': window, 'HORIZON': horizon,  
                                            'RMSE': rmse_result, 'MAPE': mape_result, 'POCID': pocid_result, 'PBE': pbe_result, 
                                            'P1': preds_real[0], 'P2': preds_real[1], 'P3': preds_real[2], 'P4': preds_real[3], 'P5': preds_real[4],
                                            'P6': preds_real[5], 'P7': preds_real[6], 'P8': preds_real[7], 'P9': preds_real[8], 'P10': preds_real[9],
                                            'P11': preds_real[10], 'P12': preds_real[11]
                                            }, index=[0])
            df_temp.to_csv(csv_path, sep=';', mode='a', header=False, index=False)


print_log("--------------- FINALIZADO ARIMA_MOE ----------------------")