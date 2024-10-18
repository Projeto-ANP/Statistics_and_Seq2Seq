import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from all_functions import *
import os
import time
import multiprocessing
from sklearn.ensemble import GradientBoostingRegressor

def ensure_even(n):
    return n + (n % 2)

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

            scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
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

# datasets = [
#             "monthly", 
#             "weekly", 
#             "yearly", 
#             "quarterly", 
#             "daily", 
#             "hourly"
#             ]

datasets = [
            # "kdd_cup", 
            # "saugeenday", 
            # "solar4", 
            # "sunspot", 
            # "us_births", 
            # "vehicle_trips",
            # "wind4",
            "cif2016"
            ]
exp_name = "gradient_wavelets"
cols_serie = ["dataset_index", "regressor", "smape", "rmse", "msmape", "mae", "test", "predictions", "features_time", "training_time", "prediction_time", "convolutions"]
cols_dataset = ["dataset", "regressor", "smape_mean", "rmse_mean", "msmape_mean", "mae_mean"]

def run_m4(args):
    dataset = args
    # metadata, series_data = read_tsf(f'../m4/timeseries/{dataset}.tsf')
    # for meta in metadata:
    #     if '@horizon' in meta:
    #         horizon = int(meta[9:])

    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            f'../m4/timeseries/{dataset}.tsf',
            replace_missing_vals_with="NaN",
            value_column_name="series_value",
        )
    # horizon = 14
    # window = ensure_even(horizon)
    path_experiments = f'../timeseries/results_rf2/{dataset}/'
    path_csv = f"{path_experiments}/{dataset}.csv"

    rmses = []
    smapes = []
    msmapes = []
    maes = []

    for index in range(len(df)):
        times = []
        # series = series_data.loc[index]['series']

        
        horizon = int(df.iloc[index]['horizon'])
        window = horizon
        series = df.iloc[index]['series_value'].tolist()
        train, test = train_test_stats(pd.Series(series), horizon)
        # convs = generate_convolutions_2(window, 10)
        convs = ''

        time_features_start = time.perf_counter()
        
        # data = features_target_v2(pd.concat([train, pd.Series([0] * horizon, index=test.index)]), convs, window)
        data = rolling_semnorm(pd.concat([train, pd.Series([0] * horizon, index=test.index)]), window)

        time_features_end = time.perf_counter()
        times.append(time_features_end - time_features_start)

        X_train, X_test, y_train, _ = train_test_split(data, horizon)
        # rg = RidgeCV(alphas=np.logspace(-3, 3, 10))
        rg = RandomForestRegressor()
        # rg = GradientBoostingRegressor(
        #     n_estimators=1200,        # Número de árvores
        #     learning_rate=0.05,      # Taxa de aprendizado
        #     max_depth=5,             # Profundidade máxima das árvores
        #     min_samples_split=3,     # Amostras mínimas para dividir um nó
        #     min_samples_leaf=2,      # Amostras mínimas em cada folha
        #     subsample=0.5,           # Proporção de dados usados para cada árvore
        #     max_features=None,       
        #     loss='squared_error'     
        # )
        time_training_start = time.perf_counter()
        rg.fit(X_train, y_train)
        time_training_end = time.perf_counter()
        times.append(time_training_end - time_training_start)

        time_preds_start = time.perf_counter()
        # preds_real = recursive_rocket1_v2(X_test, rg, train[-window:].tolist(), convs, horizon)
        preds_real = recursive_multistep_forecasting(X_test, rg, horizon)
        time_preds_end = time.perf_counter()
        times.append(time_preds_end - time_preds_start)
        
        preds_real_array = np.array(preds_real)
        preds_real_reshaped = preds_real_array.reshape(1, -1)
        test_reshaped = test.values.reshape(1, -1)
        smape_result = calculate_smape(preds_real_reshaped, test_reshaped)
        rmse_result = calculate_rmse(preds_real_reshaped, test_reshaped)
        msmape_result = calculate_msmape(preds_real_reshaped, test_reshaped)
        # mase_result = calculate_mase(preds_real_reshaped, test_reshaped, training_set, seasonality)
        mae_result = calculate_mae(preds_real_reshaped, test_reshaped)

        data_serie = {
            'dataset_index': f'{index}',
            'regressor': exp_name,
            'smape': smape_result,
            'rmse': rmse_result,
            'msmape': msmape_result,
            'mae': mae_result,
            'test': [test.tolist()],
            'predictions': [preds_real],
            'features_time': times[0],
            'training_time': times[1],
            'prediction_time': times[2],
            'convolutions': [convs]

        }

        os.makedirs(path_experiments, exist_ok=True)
        if not os.path.exists(path_csv):
            pd.DataFrame(columns=cols_serie).to_csv(path_csv, sep=';', index=False)

        df_new = pd.DataFrame(data_serie)
        df_new.to_csv(path_csv, sep=';', mode='a', header=False, index=False)

        maes.append(mae_result)
        rmses.append(rmse_result)
        msmapes.append(msmape_result)
        smapes.append(smape_result)


    data_dataset = {
        "dataset": dataset, 
        "regressor": exp_name, 
        "smape_mean": np.nanmean(smapes), 
        "rmse_mean": np.nanmean(rmses), 
        "msmape_mean": np.nanmean(msmapes), 
        "mae_mean": np.nanmean(maes)
    }
    path_dataset = path_experiments + "results.csv"
    if not os.path.exists(path_dataset):
        pd.DataFrame(columns=cols_dataset).to_csv(path_dataset, sep=';', index=False)

    df_final = pd.DataFrame([data_dataset])
    df_final.to_csv(path_dataset, sep=';', mode='a', header=False, index=False)
        


if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
            tasks = [
                (dataset) 
                for dataset in datasets
            ]

            pool.map(run_m4, tasks)
    print_log("--------------------- [FIM DE TODOS EXPERIMENTOS] ------------------------")
    # run_m4()