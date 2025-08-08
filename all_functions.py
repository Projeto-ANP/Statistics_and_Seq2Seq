import warnings
import pandas as pd
import numpy as np
from aeon.forecasting.arima import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error as mse
from aeon.transformations.detrend import ConditionalDeseasonalizer
from aeon.transformations.detrend import Deseasonalizer
from aeon.transformations.detrend import STLTransformer
import pywt
from scipy import signal
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
from datetime import datetime
from distutils.util import strtobool
import ast

warnings.filterwarnings("ignore")

def znorm(x):
  x_znorm = (x - np.mean(x)) / np.std(x)
  return x_znorm

def znorm_2(x):
    if isinstance(x, pd.Series):
        index_x = x.index
    else:
        index_x = None

    std = np.std(x)
    
    if std == 0:
        return pd.Series(np.zeros_like(x), index=index_x)
    else:
        x_znorm = (x - np.mean(x)) / std
        return pd.Series(x_znorm, index=index_x) if index_x is not None else x_znorm
    
def znorm_by(x, serie_ref):
  mean = np.mean(serie_ref)
  std = np.std(serie_ref)
  x_znorm = (x - mean) / std
  return x_znorm

def znorm_mean_std(x, mean, std):
  x_znorm = (x - mean) / std
  return x_znorm

def znorm_reverse(x, mean_x, std_x):
  x_denormalized = (x * std_x) + mean_x
  return x_denormalized

def get_stats_norm(series, horizon, window):
  last_subsequence = series[-(horizon+window):-horizon].values
  last_mean = np.mean(last_subsequence)
  last_std = np.std(last_subsequence)
  return last_mean, last_std

def mase(y_true, y_pred, y_baseline):
    # Calcula o MAE do modelo
    mae_pred = np.mean(np.abs(y_true - y_pred))
    # Calcula o MAE do modelo baseline Persistent Window (i.e., últimas h observações antes do teste)
    mae_naive = np.mean(np.abs(y_true - y_baseline))
    result = mae_pred/mae_naive
    return result

def rmse(y_true, y_pred):
  return np.sqrt(mse(y_true, y_pred))

def r2(y_true, y_pred):
    mean_y_true = sum(y_true) / len(y_true)
    ss_total = sum((y_true - mean_y_true) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2

def pocid(y_true, y_pred):
    n = len(y_true)
    D = [1 if (y_pred[i] - y_pred[i-1]) * (y_true[i] - y_true[i-1]) > 0 else 0 for i in range(1, n)]
    POCID = 100 * np.sum(D) / (n-1)
    return POCID

def mcpm(rmse_result, mape_result, pocid_result):
  er_result = 100 - pocid_result

  A1 = (rmse_result * mape_result * np.sin((2*np.pi)/3))/2
  A2 = (mape_result * er_result * np.sin((2*np.pi)/3))/2
  A3 = (er_result * rmse_result * np.sin((2*np.pi)/3))/2
  total = A1 + A2 + A3
  return total

def recursive_forecasting_stats(data, model, horizon):
  predictions = []
  pred = pd.Series()
  data_predict = data.copy()
  for _ in range(horizon):    
    next_prediction = model.predict(fh=1)
    data_predict = pd.concat([data_predict, next_prediction.rename('m3')])
    data_predict.index.name = 'timestamp'
    predictions.append(next_prediction.iloc[0])


    pred = pd.concat([pred, next_prediction.rename('m3')])
    model.fit(data_predict)       

  pred.index.name = 'timestamp'
  return pred

def custom_parser(date):
  return pd.to_datetime(date, format='%Y%m')


def read_series(file_path):
    df = pd.read_csv(file_path, header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
    df = df.set_index('timestamp', inplace=False)
    df.index = df.index.to_period('M')
    series = df['m3']
    return series

def rolling_window(series, window):
    data = []
    for i in range(len(series) - window):
        window_values = np.array(series[i:i+window])

        target_value = series[i+window]

        normalized_window = znorm(window_values)

        mean = np.mean(window_values)
        std = np.std(window_values)
        normalized_target = (target_value - mean) / std if std > 0 else target_value - mean

        example = np.append(normalized_window, normalized_target)

        data.append(example)

    df = pd.DataFrame(data)
    return df

def rolling_semnorm(series, window):
  data = []
  for i in range(len(series)-window):
    example = np.array(series[i:i+window+1])
    data.append(example)
  df = pd.DataFrame(data)
  return df


def rolling_window_transform(series, window, format="log"):
  data = []
  for i in range(len(series)-window):
    series_transform = np.log(np.array(series[i:i+window+1]))
    example = znorm(series_transform)
    data.append(example)
  df = pd.DataFrame(data)
  return df


def rolling_window_resid(series, window):
  data = []
  for i in range(len(series)-window):
    transformer = STLTransformer(sp=12)
    serie_window = np.array(series[i:i+window+1])
    result = transformer.fit(serie_window)
    # example = znorm(np.array(result.resid_))
    example = serie_window
    data.append(example)
  df = pd.DataFrame(data)
  mean = np.mean(np.array(result.resid_))
  std = np.std(np.array(result.resid_))
  return df, mean, std


# def rolling_window_series(series, window):
#     result_series = pd.Series(index=series.index)
#     mean = 0
#     std = 0
#     for i in range(len(series)-window):
#         window_values = series.iloc[i:i+window+1].values
#         normalized_values = znorm(window_values)
#         result_series.iloc[i:i+window+1] = normalized_values
#         mean = np.mean(series.iloc[i:i+window+1])
#         std = np.std(series.iloc[i:i+window+1])
#     return result_series, mean, std

def rolling_window_series(series, window):
    result_series = pd.Series(index=series.index, dtype=float)
    mean = 0
    std = 0
    
    # Verificar se a série tem dados suficientes
    if len(series) <= window:
        print(f"Série muito curta: {len(series)} pontos para janela de {window}")
        return result_series, 0, 0
    
    # Verificar se há valores faltantes na série original
    if series.isnull().any():
        print(f"Série contém {series.isnull().sum()} valores NaN")
        series = series.fillna(method='ffill').fillna(method='bfill')
    
    # Processar apenas as janelas válidas sem sobreposição
    for i in range(len(series) - window):
        window_values = series.iloc[i:i+window+1].values
        
        # Verificar se a janela contém NaN
        if np.isnan(window_values).any():
            continue
            
        # Verificar se todos os valores são iguais (std = 0)
        window_std = np.std(window_values)
        if window_std == 0:
            # Se std = 0, todos os valores são iguais, normalizar para 0
            normalized_values = np.zeros_like(window_values)
        else:
            normalized_values = znorm(window_values)
        
        # Preencher apenas a posição final da janela (sem sobreposição)
        result_series.iloc[i + window] = normalized_values[-1]
        
        # Atualizar mean e std com a última janela processada
        mean = np.mean(window_values)
        std = window_std
    
    # Preencher valores NaN restantes com interpolação
    if result_series.isnull().any():
        result_series = result_series.interpolate(method='linear')
        result_series = result_series.fillna(method='ffill').fillna(method='bfill')
    
    return result_series, mean, std

def train_test_split(data, horizon):
  X = data.iloc[:,:-1] # features
  y = data.iloc[:,-1] # target

  X_train = X[:-horizon] # features train
  X_test =  X[-horizon:] # features test

  y_train = y[:-horizon] # target train
  y_test = y[-horizon:] # target test
  return X_train, X_test, y_train, y_test

def train_test_stats(data, horizon):
  # data = pd.Series(np.diff(data).flatten())
  # print(pd.Series(data))
  train, test = data[:-horizon], data[-horizon:]
  return train, test

def rolling_validation_stats(treino, horizon=12):
    rolling_sets = []
    size_train = len(treino)
    
    for i in range(size_train - horizon*2, size_train - horizon + 1):
        treino_rolling = treino[:i]
        validacao_rolling = treino[i:i + horizon]
        rolling_sets.append((treino_rolling, validacao_rolling))
    
    return rolling_sets

def rolling_validation_regressors(X_train, y_train, horizonte_validacao=12):
    rolling_sets = []
    tamanho_treino = len(X_train)
    
    for i in range(tamanho_treino - 24, tamanho_treino - horizonte_validacao + 1):
        X_rolling_train = X_train.iloc[:i]
        X_rolling_val = X_train.iloc[i:i + horizonte_validacao]
        y_rolling_train = y_train.iloc[:i]
        y_rolling_val = y_train.iloc[i:i + horizonte_validacao]
        rolling_sets.append((X_rolling_train, X_rolling_val, y_rolling_train, y_rolling_val))
    
    return rolling_sets

def recursive_multistep_forecasting(X_test, model, horizon):
  # example é composto pelas últimas observações vistas
  # na prática, é o primeiro exemplo do conjunto de teste
  example = X_test.iloc[0].values.reshape(1,-1)

  preds = []
  for i in range(horizon):
    pred = model.predict(example)[0]
    preds.append(pred)

    # Descartar o valor da primeira posição do vetor de características
    example = example[:,1:]

    # Adicionar o valor predito na última posição do vetor de características
    example = np.append(example, pred)
    example = example.reshape(1,-1)
  return preds

def recursive_multistep_rocket(X_test, rocket, model, horizon):
  # example é composto pelas últimas observações vistas
  # na prática, é o primeiro exemplo do conjunto de teste
  example = X_test.iloc[0].values.reshape(1,-1)

  preds = []
  for i in range(horizon):
    t = rocket.transform(example).reshape(1, -1)
    pred = model.predict(t)[0]
    # pred = model.predict(example)[0]

    preds.append(pred)

    # Descartar o valor da primeira posição do vetor de características
    example = example[:,1:]

    # Adicionar o valor predito na última posição do vetor de características
    example = np.append(example, pred)
    example = example.reshape(1,-1)
  return preds

def smape_m4(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mase_m4(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep

def split_into_train_test(data, in_num, fh):
    """
    Splits the series into train and test sets. Each step takes multiple points as inputs

    :param data: an individual TS
    :param fh: number of out of sample points
    :param in_num: number of input points for the forecast
    :return:
    """
    train, test = data[:-fh], data[-(fh + in_num):]
    x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
    x_test, y_test = train[-in_num:], np.roll(test, -in_num)[:-in_num]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    x_train = np.reshape(x_train, (-1, 1))
    x_test = np.reshape(x_test, (-1, 1))
    temp_test = np.roll(x_test, -1)
    temp_train = np.roll(x_train, -1)
    for x in range(1, in_num):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
        temp_test = np.roll(temp_test, -1)[:-1]
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train, x_test, y_test

def plot_decompose(series, print=False):
    df_datetime = series.copy()
    df_datetime.index = df_datetime.index.to_timestamp()
    decomposition = seasonal_decompose(df_datetime, period=12, model='addictive')
    if print:
      fig = decomposition.plot()
    return decomposition

def remove_seasonal(series):
  transform = ConditionalDeseasonalizer(sp=12)
  return transform.is_seasonal_, transform.fit_transform(series)

def remove_trend(series):
  return series.diff()

def add_trend(series_diff, normal_series):
  series_diff.iloc[0] = normal_series.iloc[0]
  return series_diff.cumsum()


def pbe(y_true, y_pred):
  return 100*(np.sum(y_true - y_pred)/np.sum(y_true))


def transform_train(series_transform, format="deseasonal", horizon=12):
    if format == "deseasonal":
        # transform = ConditionalDeseasonalizer(sp=12)
        transform = Deseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ts = transform.transform(series_transform)
        series_ts_norm, _, _ = rolling_window_series(series_ts, horizon)
        return series_ts_norm
    elif format == "diff":
        series_diff = series_transform.diff()
        series_diff_norm, _, _ = rolling_window_series(series_diff, horizon)
        return series_diff_norm
    elif format == "log":
        constante = 10
        series_ts = np.log(series_transform + constante)
        series_ts_norm, _, _ = rolling_window_series(series_ts, horizon)
        return series_ts_norm
    elif format == "log-diff":
        constante = 10
        series_log = np.log(series_transform + constante)
        series_ts = series_log.diff()
        series_ts_norm, _, _ = rolling_window_series(series_ts, horizon)
        return series_ts_norm
    elif format == "deseasonal-diff":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ds = transform.transform(series_transform)
        series_ts = series_ds.diff()

        series_ts_norm, _, _ = rolling_window_series(series_ts, horizon)
        return series_ts_norm
    elif format == "deseasonal-log":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ds = transform.transform(series_transform)
        constante = 10
        series_ts = np.log(series_ds + constante)
        series_ts_norm, _, _ = rolling_window_series(series_ts, horizon)
        return series_ts_norm
    #normal
    series_transform_norm, _, _ = rolling_window_series(series_transform, horizon)
    return series_transform_norm


def transform_deep_train(series_transform, format="deseasonal", horizon=12):
    if format == "deseasonal":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ts = transform.transform(series_transform)
        series_ts_norm, mean, std = rolling_window_series(series_ts, horizon)
        return series_ts_norm, mean, std
    elif format == "diff":
        series_diff = series_transform.diff()
        series_diff_norm, mean, std = rolling_window_series(series_diff, horizon)
        return series_diff_norm, mean, std
    elif format == "log":
        constante = 10
        series_ts = np.log(series_transform + constante)
        series_ts_norm, mean, std = rolling_window_series(series_ts, horizon)
        return series_ts_norm, mean, std
    elif format == "log-diff":
        constante = 10
        series_log = np.log(series_transform + constante)
        series_ts = series_log.diff()
        series_ts_norm, mean, std= rolling_window_series(series_ts, horizon)
        return series_ts_norm, mean, std
    elif format == "deseasonal-diff":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ds = transform.transform(series_transform)
        series_ts = series_ds.diff()

        series_ts_norm, mean, std= rolling_window_series(series_ts, horizon)
        return series_ts_norm, mean, std
    elif format == "deseasonal-log":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ds = transform.transform(series_transform)
        constante = 10
        series_ts = np.log(series_ds + constante)
        series_ts_norm, mean, std = rolling_window_series(series_ts, horizon)
        return series_ts_norm, mean, std
    #normal
    series_transform_norm, mean, std = rolling_window_series(series_transform, horizon)
    return series_transform_norm, mean, std

def transform_reverse_preds(series_preds, train_norm, format="deseasonal"):
    if format == "deseasonal":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train_norm)
        series_norm = transform.inverse_transform(series_preds)
        return series_norm
    elif format == "diff":
        series_preds = pd.concat([train_norm.iloc[[-1]], series_preds])
        series_norm = series_preds.cumsum()[1:]
        return series_norm
    elif format == "log":
        constante = 10
        return np.exp(series_preds) - constante
    elif format == "log-diff":
        constante = 10
        train_log = np.log(train_norm + constante)
        series_preds = pd.concat([train_log.iloc[[-1]], series_preds])
        series_log = series_preds.cumsum()
        series_inverse = np.exp(series_log) - constante
        return series_inverse[1:]
    elif format == "deseasonal-diff":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train_norm)
        train_des = transform.transform(train_norm)
        completa = pd.concat([train_des.iloc[[-1]], series_preds])
        series_cs = completa.cumsum()
        series_inverse = transform.inverse_transform(series_cs)
        return series_inverse[1:]
    elif format == "deseasonal-log":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train_norm)
        series_ds = np.exp(series_preds)
        constante = 10
        series_ds_abs = series_ds - constante
        series_inverse = transform.inverse_transform(series_ds_abs)
        return series_inverse

    #normal
    return series_preds


def fit_arima_train(train_ds, train, initial_order, horizon, format):
    p, d, q = initial_order
    p_original, d_original, q_original = initial_order
    max_attempts = 10
    d_incremented = d+1
    try:
        forecast = ARIMA(order=(p_original, d, q_original), suppress_warnings=True)
        forecast.fit(train_ds)

        # preds = recursive_forecasting_stats(train_ds, forecast, horizon)
        preds = forecast.predict(fh=[i for i in range(1, horizon+1)] )
        preds_inverse = reverse_transform_norm_preds(preds, train, format=format)

        final_order_dict = {
            'p': p_original,
            'd': d,
            'q': q_original,
        }

        return forecast, preds_inverse, final_order_dict

    except Exception as e:
        print_log(f"Exception: Not valid original ({p},{d},{q}) for train. Error: {e}")
        try:
          forecast = ARIMA(order=(p_original, d_incremented, q_original), suppress_warnings=True)
          forecast.fit(train_ds)

          preds = forecast.predict(fh=[i for i in range(1, horizon+1)] )
          preds_inverse = reverse_transform_norm_preds(preds, train, format=format)

          final_order_dict = {
              'p': p_original,
              'd': d_incremented,
              'q': q_original,
          }
          return forecast, preds_inverse, final_order_dict
        except:
          print_log(f"Exception: Not valid (d) incrementation in ({p},{d_incremented},{q}) for train. Error: {e}")
        # raise ValueError("Problem with all possible (p,d,q) in this time series") from e
    for attempt in range(max_attempts):
        try:
            forecast = ARIMA(order=(p, d, q), suppress_warnings=True)
            forecast.fit(train_ds)

            preds = forecast.predict(fh=[i for i in range(1, horizon+1)] )
            preds_inverse = reverse_transform_norm_preds(preds, train, format=format)

            final_order_dict = {
                'p': p,
                'd': d,
                'q': q,
            }

            return forecast, preds_inverse, final_order_dict

        except Exception as e:
            print_log(f"Exception: Not valid decrementing ({p},{d},{q}) for train. Error: {e}")

            if attempt % 2 == 0: 
                if p > 0:
                  p -= 1
            else:
                if q > 0:
                  q -= 1

            if p == 0 and q == 0:
                p = p_original
                q = q_original
                d += 1
    start_train = train.index.tolist()[0]
    final_train = train.index.tolist()[-1]

    train_range = f"{start_train}_{final_train}"
    raise ValueError(f"Problem with all possible {str(initial_order)} for {format} in {train_range}")


def fit_sarima_train(train_ds, train, initial_order, seasonal_order, horizon, format):
    p, d, q = initial_order
    p_original, d_original, q_original = initial_order
    pp, dd, qq, ss = seasonal_order
    max_attempts = 10
    d_incremented = d+1
    try:
        forecast = ARIMA(order=(p_original, d, q_original), 
                           seasonal_order=seasonal_order, 
                           suppress_warnings=True
                           )
        forecast.fit(train_ds)

        preds = forecast.predict(fh=[i for i in range(1, horizon+1)] )
        preds_real = reverse_transform_norm_preds(preds, train, format=format)

        final_order_dict = {
            'p': p_original,
            'd': d,
            'q': q_original,
            'P': pp,
            'D': dd,
            'Q': qq,
            's': ss
        }

        return forecast, preds_real, final_order_dict

    except Exception as e:
        print_log(f"Exception: Not valid original ({p},{d},{q}) for train. Error: {e}")
        try:
          forecast = ARIMA(order=(p_original, d_incremented, q_original),
                                seasonal_order=seasonal_order,
                                suppress_warnings=True
                                )
          forecast.fit(train_ds)

          preds = forecast.predict(fh=[i for i in range(1, horizon+1)] )
          preds_real = reverse_transform_norm_preds(preds, train, format=format)

          final_order_dict = {
            'p': p_original,
            'd': d,
            'q': q_original,
            'P': pp,
            'D': dd,
            'Q': qq,
            's': ss
        }
          return forecast, preds_real, final_order_dict
        except:
          print_log(f"Exception: Not valid (d) incrementation in ({p},{d_incremented},{q}) for train. Error: {e}")
        # raise ValueError("Problem with all possible (p,d,q) in this time series") from e
    for attempt in range(max_attempts):
        try:
            forecast = ARIMA(order=(p, d, q),
                                seasonal_order=seasonal_order,
                                suppress_warnings=True
                                )
            forecast.fit(train_ds)

            preds = forecast.predict(fh=[i for i in range(1, horizon+1)] )
            preds_real = reverse_transform_norm_preds(preds, train, format=format)

            final_order_dict = {
            'p': p_original,
            'd': d,
            'q': q_original,
            'P': pp,
            'D': dd,
            'Q': qq,
            's': ss
        }

            return forecast, preds_real, final_order_dict

        except Exception as e:
            print_log(f"Exception: Not valid decrementing ({p},{d},{q}) for train. Error: {e}")

            if attempt % 2 == 0: 
                if p > 0:
                  p -= 1
            else:
                if q > 0:
                  q -= 1

            if p == 0 and q == 0:
                p = p_original
                q = q_original
                d += 1
    raise ValueError("Problem with all possible (p,d,q) in this time series") from e


def print_log(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] {message}")

from statsmodels.tsa.stattools import adfuller

def analyze_stationarity(series):
    adf_values = {
        'Test Statistic': "-",
        'p-value': "-",
        'Lags Used': "-",
        'Observations Used': "-",
        'Critical Value (1%)': "-",
        'Critical Value (5%)': "-",
        'Critical Value (10%)': "-",
        'Stationary': "-"
    }

    try:
      adf_result = adfuller(series, autolag='AIC')
      p_value = adf_result[1]
      is_stationary = p_value < 0.05

      adf_values = {
          'Test Statistic': adf_result[0],
          'p-value': p_value,
          'Lags Used': adf_result[2],
          'Observations Used': adf_result[3],
          'Critical Value (1%)': adf_result[4]['1%'],
          'Critical Value (5%)': adf_result[4]['5%'],
          'Critical Value (10%)': adf_result[4]['10%'],
          'Stationary': str(is_stationary)
      }
    except Exception as e:
       print_log(f"Exception: {e}")
    
    return adf_values

def get_arima_param(path_results, derivado, uf, transform):
    df = pd.read_csv(f"{path_results}/{derivado}/transform_{uf}.csv", sep=";")
    params_str = df[df['DATA'] == transform]['PARAMS'].values[0]
    params = ast.literal_eval(params_str)
    return (params['p'], params['d'], params['q'])

def reverse_transform_norm_preds(series_preds, train, format="deseasonal", w=12):
    if format == "deseasonal":
        transform = Deseasonalizer(sp=12)
        transform.fit(train)
        series_before_norm = transform.transform(train)
        
        _, mean, std = rolling_window_series(series_before_norm, w)
        preds_transformed = znorm_reverse(series_preds, mean, std)
        
        series_real = transform.inverse_transform(preds_transformed)
        return series_real
    elif format == "diff":
        series_before_norm = train.diff()
        
        _, mean, std = rolling_window_series(series_before_norm, w)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        series_preds = pd.concat([train.iloc[[-1]], preds_transformed])
        series_norm = series_preds.cumsum()[1:]
        return series_norm
    elif format == "log":
        constante = 10
        series_before_norm = np.log(train + constante)
        
        _, mean, std = rolling_window_series(series_before_norm, w)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        return np.exp(preds_transformed) - constante
    elif format == "log-diff":
        constante = 10
        series_train_log = np.log(train + constante)
        series_before_norm = series_train_log.diff()
        
        _, mean, std = rolling_window_series(series_before_norm, w)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        series_log = pd.concat([series_train_log.iloc[[-1]], preds_transformed])
        series_log = series_log.cumsum()
        series_inverse = np.exp(series_log) - constante
        return series_inverse[1:]
    elif format == "deseasonal-diff":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train)
        train_des = transform.transform(train)
        train_diff = train_des.diff()

        _, mean, std = rolling_window_series(train_diff, w)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        completa = pd.concat([train_des.iloc[[-1]], preds_transformed])
        series_cs = completa.cumsum()
        series_inverse = transform.inverse_transform(series_cs)
        return series_inverse[1:]
    elif format == "deseasonal-log":
        constante = 10
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train)
        train_des = transform.transform(train)
        train_log = np.log(train_des + constante)

        _, mean, std = rolling_window_series(train_log, w)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        series_ds = np.exp(preds_transformed)
        series_ds_abs = series_ds - constante
        series_inverse = transform.inverse_transform(series_ds_abs)
        return series_inverse
    
    #normal
    # _, mean, std = rolling_window_series(train, w)
    mean = np.mean(train)
    std = np.std(train)
    preds_real = znorm_reverse(series_preds, mean, std)
    return preds_real

from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
   
def get_pacf(series):
    df_datetime = series.copy()
    df_datetime.index = df_datetime.index.to_timestamp()
    return pacf(df_datetime.dropna(), alpha=0.05, method='ywm')

def get_acf(series):
    df_datetime = series.copy()
    df_datetime.index = df_datetime.index.to_timestamp()
    return acf(df_datetime.dropna(), alpha=0.05)

def get_AR_terms(series):
    PACF, PACF_ci = get_pacf(series)

    PACF_ci_ll = PACF_ci[:,0] - PACF
    PACF_ci_ul = PACF_ci[:,1] - PACF
    
    l_values = np.where(PACF < PACF_ci_ll)[0]
    u_values = np.where(PACF > PACF_ci_ul)[0]
    
    AR_values =  np.concatenate((l_values, u_values))
    AR_values = np.sort(AR_values)

    AR_values = np.unique(AR_values[(AR_values > 1) & (AR_values < 12)][:5])

    if len(AR_values) == 0:
        return [1]  
    
    return AR_values.tolist() 

def get_MA_terms(series):
    ACF, ACF_ci = get_acf(series)
    
    ACF_ci_ll = ACF_ci[:,0] - ACF
    ACF_ci_ul = ACF_ci[:,1] - ACF
    
    l_values = np.where(ACF < ACF_ci_ll)[0]
    u_values = np.where(ACF > ACF_ci_ul)[0]

    MA_values =  np.concatenate((l_values, u_values))
    MA_values = np.sort(MA_values)

    MA_values = np.unique(MA_values[(MA_values > 1) & (MA_values < 12)][:5])

    if len(MA_values) == 0:
        return [1]  
    
    return MA_values.tolist() 

from scipy.spatial.distance import cdist
from dtaidistance import dtw
from aeon.distances import dtw_distance

def knn_similar_series(train_series, query_series, k, metric='euclidean', horizon = 12):
    train_data = train_series.values
    query_data = query_series.values
    query_len = len(query_data)
    
    n = len(train_data)
    train_subsequences = np.array([train_data[i:i+query_len] for i in range(n - query_len + 1)])
    
    if metric == 'euclidean':
        distances = cdist([query_data], train_subsequences, metric='euclidean').flatten()
    elif metric == 'mahalanobis':
        V = np.cov(train_subsequences.T)
        VI = np.linalg.inv(V)
        distances = cdist([query_data], train_subsequences, metric='mahalanobis', VI=VI).flatten()
    elif metric == 'dtw':
        distances = np.array([dtw_distance(query_data, subsequence, window=0.2) for subsequence in train_subsequences])
    else:
        raise ValueError("Métrica desconhecida. Use 'euclidean', 'mahalanobis' ou 'dtw'.")
    
    nearest_indices = np.argsort(distances)[:k]
    
    results = []
    for idx in nearest_indices:
        similar_sequence = train_series[idx:idx + query_len]
        results.append({
            'similar_sequence': similar_sequence,
            'distance': distances[idx],
        })
    
    return results
def transform_regressors(train, format='normal'):
    if format == 'deseasonal':
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train)
        train_deseasonal = transform.transform(train)
        # train_deseasonal = fit_transform_STL(train, train)
        return train_deseasonal
    elif format == 'log':
        constante = 10
        train_log = np.log(train + constante)
        return train_log
    elif format == 'normal':
        return train

def reverse_regressors(train_real, preds, window, format='normal'):
    if format == 'deseasonal':
        transform = ConditionalDeseasonalizer(sp=12)
        transform = Deseasonalizer(sp=12)
        transform.fit(train_real)
        series_before_norm = transform.transform(train_real)

        # series_before_norm = fit_transform_STL(train_real, train_real)

        # _, mean, std = rolling_window_series(series_before_norm, window)
        mean = np.mean(series_before_norm.iloc[-window:])
        std = np.std(series_before_norm.iloc[-window:])
        preds_transformed = znorm_reverse(preds, mean, std)

        series_real = transform.inverse_transform(preds_transformed)
        # series_real = fit_inverse_transform_STL(train_real, preds_transformed)
        return series_real
    elif format == 'log':
        constante = 10
        series_before_norm = np.log(train_real)
        # _, mean, std = rolling_window_series(series_before_norm, window)
        mean = np.mean(series_before_norm.iloc[-window:])
        std = np.std(series_before_norm.iloc[-window:])

        preds_transformed = znorm_reverse(preds, mean, std)

        return np.exp(preds_transformed) - constante
    elif format == 'normal':
        # _, mean, std = rolling_window_series(train_real, window)
        mean = np.mean(train_real.iloc[-window:])
        std = np.std(train_real.iloc[-window:])

        preds_real = znorm_reverse(preds, mean, std)
        return preds_real
    
    raise ValueError('nao existe essa transformacao')

def get_error_series(path, test_date):
    df = pd.read_csv(path, sep=";")
    error_series_str = df.loc[df['test_range'] == test_date, 'error_series'].values[0]
    error_list_rf = ast.literal_eval(error_series_str)
    return error_list_rf


def get_preds_model(path, filter, coluna):
    df = pd.read_csv(path, sep=";")
    results = {}
    filtered_df = df[df[coluna] == filter]
    columns_p1_to_p12 = filtered_df.loc[:, 'P1':'P12']
    values_list = columns_p1_to_p12.values.flatten().tolist()     
    return values_list


def rolling_window_image(series, window, representation, wavelet, level, shuffle_order=None):
  data = []
  for i in range(len(series)-window):
    example = np.array(series[i:i+window+1])
    target = example[-1]

    features = np.delete(example, -1)
    features_norm = znorm(features)
    
    target_norm = znorm_by(target, features)

    rep_features = transform_series(features_norm, representation, wavelet, level, shuffle_order)
    feat_target = np.concatenate((rep_features, [target_norm]))
    data.append(feat_target)
  df = pd.DataFrame(data)
  return df

def transform_series(series, representation, wavelet, level, shuffle_order=None):
  # series = np.array(znorm(series))
  if representation == "CWT":
    coeffs, freqs = pywt.cwt(series, scales=np.arange(1, len(series) + 1), wavelet="morl") # morl
    im_final = coeffs.flatten()
    # im_final = np.concatenate((series, coeffs.flatten()))
  elif representation == "DWT":
    coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)
    coeffs_list = np.concatenate(coeffs, axis=0) 
    im_final = coeffs_list
    # im_final = np.concatenate((series, coeffs_list))

  elif representation == "FT":
    fft_values = np.fft.fft(series)
   
    fft_magnitudes = np.abs(fft_values)
    half_n = len(fft_magnitudes) // 2
    fft_features = fft_magnitudes[1:half_n].tolist()
    # im_final = np.array(fft_features)
    im_final = np.concatenate((series, fft_features))

  elif representation == "SWT":
    coeffs_swt = pywt.swt(series, wavelet, level=level)
    coeffs_list = np.concatenate([coeff[0] for coeff in coeffs_swt] + [coeff[1] for coeff in coeffs_swt], axis=0)
    im_final = np.concatenate((series, coeffs_list))

  elif representation == "SWT_GASF":
    coeffs_swt = pywt.swt(series, wavelet, level=level)
    coeffs_swt = np.concatenate([coeff[0] for coeff in coeffs_swt] + [coeff[1] for coeff in coeffs_swt], axis=0)

    series_reshape = series.reshape(1, len(series))
    rp = RecurrencePlot(threshold='distance')
    X_rp = rp.fit_transform(series_reshape)
    rp = X_rp[0].flatten()

    coeffs_list = np.concatenate((coeffs_swt.flatten(), rp))
    im_final = np.concatenate((series, coeffs_list))

  elif representation == "SWT_MTF":
    coeffs_swt = pywt.swt(series, wavelet, level=level)
    coeffs_swt = np.concatenate([coeff[0] for coeff in coeffs_swt] + [coeff[1] for coeff in coeffs_swt], axis=0)
    series_reshape = series.reshape(1, len(series))
    mtf = MarkovTransitionField(n_bins=4, strategy='uniform') #n_bins=4, strategy='uniform'
    X_mtf = mtf.fit_transform(series_reshape)
    mtf = X_mtf[0].flatten()

    coeffs_list = np.concatenate((coeffs_swt.flatten(), mtf))
    im_final = np.concatenate((series, coeffs_list))
  elif representation == "WPT":
    wp = pywt.WaveletPacket(data=series, wavelet='db1', mode='symmetric')

    nodes = wp.get_level(level, order='freq')
    coeffs_wpt = np.array([n.data for n in nodes])

    coeffs_list = np.concatenate(coeffs_wpt, axis=0)
    im_final = np.concatenate((series, coeffs_list))

  elif representation == "STFT":
    from scipy.signal import stft
    f, t, Zxx = stft(series, nperseg=64)

    coeffs_stft = np.abs(Zxx)

    coeffs_list = coeffs_stft.flatten()
    im_final = np.concatenate((series, coeffs_list))
  elif representation == "MTF":
    series_reshape = series.reshape(1, len(series))
    mtf = MarkovTransitionField(n_bins=4, strategy='uniform') #n_bins=4, strategy='uniform'
    X_mtf = mtf.fit_transform(series_reshape)
    coeffs_list = X_mtf[0].flatten()
    im_final = np.concatenate((series, coeffs_list))
  elif representation == "GADF":
    series_reshape = series.reshape(1, len(series))
    gaf = GramianAngularField(method='difference')
    X_gaf = gaf.fit_transform(series_reshape)
    coeffs_list = X_gaf[0].flatten()
    im_final = np.concatenate((series, coeffs_list))
  elif representation == "GASF":
    series_reshape = series.reshape(1, len(series))
    gaf = GramianAngularField(method='summation')
    X_gaf = gaf.fit_transform(series_reshape)
    coeffs_list = X_gaf[0].flatten()
    im_final = np.concatenate((series, coeffs_list))
  elif representation == "RP":
    series_reshape = series.reshape(1, len(series))
    rp = RecurrencePlot(threshold='distance')
    X_rp = rp.fit_transform(series_reshape)
    coeffs_list = X_rp[0].flatten()
    im_final = np.concatenate((series, coeffs_list))
  elif representation == "FIRTS":
    series_reshape = series.reshape(1, len(series))
    mtf = MarkovTransitionField(n_bins=4, strategy='uniform')
    X_mtf = mtf.fit_transform(series_reshape)
    gaf = GramianAngularField(method='difference')
    X_gaf = gaf.fit_transform(series_reshape)
    rp = RecurrencePlot(threshold='distance')
    X_rp = rp.fit_transform(series_reshape)
    coeffs_list = (X_mtf[0] + X_gaf[0] + X_rp[0]).flatten() # FIRTS é fusão entre MTF, GADF e RP (vejam o artigo que passei para vocês)
    im_final = np.concatenate((series, coeffs_list))
  return im_final

def get_test_real(series, start_date, end_date):
    start_period = pd.to_datetime(start_date).to_period('M')
    end_period = (pd.to_datetime(end_date))
    
    filtered_series = series.loc[start_period:end_period]

    return filtered_series


from aeon.utils.datetime import _get_duration, _get_freq
from statsmodels.tsa.seasonal import seasonal_decompose, STL
def fit_transform_STL(X, serie_target, model="additive", sp=12):
  # sazonalidade = seasonal_decompose(
  #           X,
  #           model=model,
  #           period=12,
  #           filt=None,
  #           two_sided=True,
  #           extrapolate_trend=0,
  #       ).seasonal.iloc[:12]

  stl = STL(X, period=sp)
  result = stl.fit()

  sazonalidade = result.seasonal.iloc[:sp]
  
  
  shift = (
            -_get_duration(
                serie_target.index[0],
                X.index[0],
                coerce_to_int=True,
                unit=_get_freq(X.index),
            )
            % sp
        )
  seasonal = np.resize(np.roll(sazonalidade, shift=shift), serie_target.shape[0])

  if model == "additive":
    return serie_target - seasonal
  else:
    return serie_target / seasonal

def fit_inverse_transform_STL(X, serie_target, model="additive", sp=12):
  # sazonalidade = seasonal_decompose(
  #           X,
  #           model=model,
  #           period=12,
  #           filt=None,
  #           two_sided=True,
  #           extrapolate_trend=0,
  #       ).seasonal.iloc[:12]

  stl = STL(X, period=sp)
  result = stl.fit()

  sazonalidade = result.seasonal.iloc[:sp]
  
  
  shift = (
            -_get_duration(
                serie_target.index[0],
                X.index[0],
                coerce_to_int=True,
                unit=_get_freq(X.index),
            )
            % sp
        )
  seasonal = np.resize(np.roll(sazonalidade, shift=shift), serie_target.shape[0])

  if model == "additive":
    return serie_target + seasonal
  else:
    return serie_target * seasonal

def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )



def morlet_wavelet(f, t, w=5):
    """Gera a wavelet de Morlet."""
    if f == 0:
        return np.zeros_like(t)
    return np.exp(1j * 2 * np.pi * f * t) * np.exp(-t ** 2 / (2 * (w / (2 * np.pi * f)) ** 2))

def apply_kernel(X, num_kernels, lengths, dilations, paddings, biases):
    X = np.array(X)
    input_length = len(X)

    # Inicializando a matriz de características
    features = np.zeros((input_length, num_kernels))

    for k in range(num_kernels):
        length = lengths[k]
        padding = paddings[k]
        bias = biases[k]
        dilation = dilations[k]

        end = (input_length + padding) - ((length - 1) * dilation)

        for i in range(-padding, end):
            _sum = bias
            index = i

            # Convolução com wavelet
            for j in range(length):
                wavelet_index = index + j * dilation
                if 0 <= wavelet_index < input_length:
                    # Usar uma frequência fixa ou variar levemente
                    frequency = 1 / (length / 2)  # Frequência baseada no comprimento
                    wavelet_value = morlet_wavelet(frequency, j / length)
                    _sum += wavelet_value.real * X[wavelet_index]

            if i + padding >= 0:
                features[i + padding, k] = _sum

    return features, np.sum(features > 0)

def recursive_rocket1(X_test, model, feats, convs, horizon):
    example = X_test.iloc[0].values.reshape(1,-1)
    preds = []
    for i in range(horizon):
        pred = model.predict(example)[0]
        preds.append(pred)
        feats = feats[1:]
        feats.append(pred)
      
        # example = example[:,1:]
        example_transform = random_wavelet_convolution(feats, convs)
        # example = np.append(example, pred)
        # example = example.reshape(1,-1)
        example = example_transform.reshape(1,-1)

    return preds

def generate_random_wavelet(serie, waveType, wavelet, level, scales):
    if waveType == "cwt":
        coeffs, _ = pywt.cwt(serie, scales, wavelet)
    elif waveType == "dwt":
        coeffs = pywt.wavedec(serie, wavelet=wavelet, level=level)
        # return np.concatenate(coeffs, axis=0) 
        return [np.concatenate(coeffs, axis=0)]
    elif waveType == "swt":
        coeffs_swt = pywt.swt(serie, wavelet, level=level)
        # return [np.concatenate([coeff[0] for coeff in coeffs_swt] + [coeff[1] for coeff in coeffs_swt], axis=0)]
        # return coeffs_swt
        coeffs_lists = [c1 for c1, c2 in coeffs_swt] + [c2 for c1, c2 in coeffs_swt]
        coeffs_final = []
        for c in coeffs_lists:
            coeffs_final.append(c)
        return coeffs_lists
    return coeffs
    

def random_wavelet_convolution(serie, convs):
    features = np.array([])

    wavetypes, wavelets, levels, scales = convs
    
    for i in range(len(wavetypes)):
        coeffs = generate_random_wavelet(serie, wavetypes[i], wavelets[i], levels[i], scales[i])
        # print(coeffs)
        # print(coeffs_mean)
        coeffs_mean = np.mean(coeffs, axis=1)
        # coeffs_median = np.median(coeffs, axis=1)
        # coeffs_std = np.std(coeffs, axis=1)
        coeffs_max = np.max(coeffs, axis=1)
        # coeffs_min = np.min(coeffs, axis=1)

        # features = np.concatenate([features, coeffs_mean])
        features = np.concatenate([features, coeffs_mean])
        # features.append(np.std(coeffs))
        # features.append(np.max(coeffs))
        # features.append(np.min(coeffs))

    return features

def generate_convolutions(len_serie, num_wavelets = 10):
    wavelets = np.empty(num_wavelets, dtype=object)
    # wavetypes = np.random.choice(["cwt", "dwt", "swt"], size=num_wavelets)
    # wavetypes = np.random.choice(["cwt","dwt", "swt"], size=num_wavelets)
    wavetypes = ['cwt', 'cwt', 'cwt', 'cwt','dwt','swt','swt','swt','swt', 'swt']

    levels = np.zeros(num_wavelets, dtype=int)
    scales = [np.zeros(0) for _ in range(num_wavelets)]

    for i, waveType in enumerate(wavetypes):
        if waveType == "cwt":
            wavelet = np.random.choice(["morl"])

            # max_scale_limit = np.random.randint(low = len_serie/2, high = len_serie + 1)
            # max_scale = np.random.randint(low=len_serie/2, high=max_scale_limit + 1)
            
            # scale = np.random.uniform(len_serie/3, max_scale, size=max_scale_limit)

            scale = np.random.choice(np.arange(0.1, 1.1, 0.1), size=30)
        
            wavelets[i] = wavelet
            scales[i] = scale
        elif waveType == "dwt":
            # wavelet = "db2"
            wavelet = np.random.choice(pywt.wavelist(kind="discrete"))
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
            wlt = pywt.Wavelet(wavelet)
            max_level = pywt.swt_max_level(len_serie)
            level = np.random.randint(low = 1, high = max_level + 1)
            wavelets[i] = wavelet
            levels[i] = max_level

    return wavetypes, wavelets, levels, scales


def features_target(series, convs, window):
    data = []
    series = series.tolist()
    for i in range(len(series) - window):
        example = np.array(series[i:i+window])
        target_value = series[i+window]
        # features = random_wavelet_convolution(example, convs)
        features = random_wavelet_convolution(example, convs)
        feats_target = np.concatenate((features, [target_value]))
        norm_features = feats_target
        data.append(norm_features)

        df = pd.DataFrame(data)
    return df



def read_tsf(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()

    metadata = []
    series = []

    for line in lines:
        if line.startswith('@') or line.startswith('#'):
            metadata.append(line.strip())
        else:
            series.append(line.strip())

    formatted_data = []

    for entry in series:
        dataset_name, start_date, series_values = entry.split(':', 2)
        start_date = start_date.strip()
        series_list = [float(value) for value in series_values.split(',')]
        
        formatted_data.append({
            'dataset': dataset_name,
            'start_date': start_date,
            'series': series_list
        })
        
    series_data = pd.DataFrame(formatted_data)


    return metadata, series_data

# Função para calcular os valores de SMAPE
def calculate_smape(forecasts, test_set):
    smape = 2 * np.abs(forecasts - test_set) / (np.abs(forecasts) + np.abs(test_set))
    smape_per_series = np.nanmean(smape, axis=1)  # Média por série
    return smape_per_series

# Função para calcular os valores de mSMAPE
def calculate_msmape(forecasts, test_set):
    epsilon = 0.1
    comparator = np.full(test_set.shape, 0.5 + epsilon)
    sum_values = np.maximum(comparator, (np.abs(forecasts) + np.abs(test_set) + epsilon))
    smape = 2 * np.abs(forecasts - test_set) / sum_values
    msmape_per_series = np.nanmean(smape, axis=1)
    return msmape_per_series

# Função para calcular os valores de MASE
def calculate_mase(forecasts, test_set, training_set, seasonality):
    mase_per_series = []
    
    for k in range(forecasts.shape[0]):
        te = test_set[k, ~np.isnan(test_set[k, :])]
        tr = training_set[k][~np.isnan(training_set[k])]
        f = forecasts[k, ~np.isnan(forecasts[k, :])]
        
        # Cálculo de MASE
        mase = MASE(te, f, np.mean(np.abs(np.diff(tr, n=1, axis=0, prepend=tr[0]))))
        
        if np.isnan(mase):
            mase = MASE(te, f, np.mean(np.abs(np.diff(tr, n=1, axis=0, prepend=tr[0]))))
        
        mase_per_series.append(mase)

    return np.array(mase_per_series)

# Função para calcular os valores de MAE
def calculate_mae(forecasts, test_set):
    mae = np.abs(forecasts - test_set)
    mae_per_series = np.nanmean(mae, axis=1)
    return mae_per_series

# Função para calcular os valores de RMSE
def calculate_rmse(forecasts, test_set):
    squared_errors = (forecasts - test_set) ** 2
    rmse_per_series = np.sqrt(np.nanmean(squared_errors, axis=1))
    return rmse_per_series

# Função para fornecer um resumo das métricas de erro
def calculate_errors(forecasts, test_set, training_set, seasonality, output_file_name):
    smape_per_series = calculate_smape(forecasts, test_set)
    msmape_per_series = calculate_msmape(forecasts, test_set)
    mase_per_series = calculate_mase(forecasts, test_set, training_set, seasonality)
    mae_per_series = calculate_mae(forecasts, test_set)
    rmse_per_series = calculate_rmse(forecasts, test_set)

    metrics = {
        "Mean SMAPE": np.nanmean(smape_per_series),
        "Median SMAPE": np.nanmedian(smape_per_series),
        "Mean mSMAPE": np.nanmean(msmape_per_series),
        "Median mSMAPE": np.nanmedian(msmape_per_series),
        "Mean MASE": np.nanmean(mase_per_series),
        "Median MASE": np.nanmedian(mase_per_series),
        "Mean MAE": np.nanmean(mae_per_series),
        "Median MAE": np.nanmedian(mae_per_series),
        "Mean RMSE": np.nanmean(rmse_per_series),
        "Median RMSE": np.nanmedian(rmse_per_series),
    }

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Escrevendo as métricas em arquivos
    np.savetxt(f"{output_file_name}_smape.txt", smape_per_series, delimiter=",")
    np.savetxt(f"{output_file_name}_msmape.txt", msmape_per_series, delimiter=",")
    np.savetxt(f"{output_file_name}_mase.txt", mase_per_series, delimiter=",")
    np.savetxt(f"{output_file_name}_mae.txt", mae_per_series, delimiter=",")
    np.savetxt(f"{output_file_name}_rmse.txt", rmse_per_series, delimiter=",")
    
    with open(f"{output_file_name}.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

# Função auxiliar para calcular MASE
def MASE(actual, forecast, training_mean):
    return np.mean(np.abs(actual - forecast)) / training_mean if training_mean != 0 else np.nan

def inverse_tranformation(train, serie_target, window, format="normal"):
    if format == 'deseasonal':
        transform = ConditionalDeseasonalizer(sp=12)
        transform = Deseasonalizer(sp=12)
        transform.fit(train)
        series_before_norm = transform.transform(train)

        mean = np.mean(series_before_norm.iloc[-window:])
        std = np.std(series_before_norm.iloc[-window:])

        preds_transformed = znorm_reverse(serie_target, mean, std)

        series_real = transform.inverse_transform(preds_transformed)
        return series_real
    elif format == 'log':
        constante = 10
        series_before_norm = np.log(train)
        mean = np.mean(series_before_norm.iloc[-window:])
        std = np.std(series_before_norm.iloc[-window:])
        preds_transformed = znorm_reverse(serie_target, mean, std)

        return np.exp(preds_transformed) - constante
    elif format == 'normal':
        mean = np.mean(train.iloc[-window:])
        std = np.std(train.iloc[-window:])
        preds_real = znorm_reverse(serie_target, mean, std)
        return preds_real
    
    raise ValueError('nao existe essa transformacao')


def recursive_step(X_test, train_completo, model, horizon, window, transform, representation, wavelet, level):
  example = X_test.iloc[0].values.reshape(1,-1)
  last_window_train = train_completo[-window:].tolist()
  last_window_train_pd = train_completo[-window:]
  preds = []
  preds_real = []
  for i in range(horizon):
    pred = model.predict(example)[0]
    preds.append(pred)
    
    #normaliza a ultima janela de train original
    last_window_train_pd = znorm_2(last_window_train_pd)

    #pega o valor do proximo index de train para adicionar em predicao
    # index_pred = last_window_train_pd.index[-1] + 1
    index_pred = last_window_train_pd.index[-1] + pd.DateOffset(months=1)
    # print("LAST WINDOW")
    # print(last_window_train_pd.index[-1])
    # print("INDEX PRED")
    # print(index_pred)

    pred_as_pd = pd.Series([pred], index=[index_pred])
    
    #concatena a predicao normalizada no ultimo pedaco do train
    last_window_train_pd = pd.concat([last_window_train_pd, pred_as_pd])

    #remove o primeiro elemento de train normalizado
    last_window_train_pd = last_window_train_pd[1:]

    #passa a serie de train+preds para desnormalizar pegando apenas o ultimo valor concatenado (pred)
    pred_real = inverse_tranformation(train_completo, last_window_train_pd, window, format=transform).iloc[-1]
    
    #adiciona preds real ao pedaco do train real para proximas reversoes
    index_pred_real = train_completo.index[-1] + pd.DateOffset(months=1)
    pred_real_as_pd = pd.Series([pred_real], index=[index_pred_real])
    train_completo = pd.concat([train_completo, pred_real_as_pd])

    #adiciona a predicao em escala real na lista de preds real
    preds_real.append(pred_real)

    #pega ultima janela do train original e tira o primeiro valor para concatenar a predicao posteriormente
    last_window_train = last_window_train[1:]
    last_window_train.append(pred_real)
    

    #normalizo o ultimo pedaco do train(window) + preds
    last_window_train_norm = znorm_2(last_window_train)

    #transformo novamente para gerar novo X_test
    rep_features = transform_series(last_window_train_norm, representation, wavelet, level)
    example = rep_features.reshape(1,-1)
    # print(example)
  return preds_real
