import warnings
import pandas as pd
import numpy as np
from aeon.forecasting.arima import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error as mse
from aeon.transformations.detrend import ConditionalDeseasonalizer
from aeon.transformations.boxcox import BoxCoxTransformer
from datetime import datetime
import ast

warnings.filterwarnings("ignore")

def znorm(x):
  x_znorm = (x - np.mean(x)) / np.std(x)
  return x_znorm

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

def rmse(y_true, y_pred):
  return np.sqrt(mse(y_true, y_pred))

def pocid(y_true, y_pred):
    n = len(y_true)
    D = [1 if (y_pred[i] - y_pred[i-1]) * (y_true[i] - y_true[i-1]) > 0 else 0 for i in range(1, n)]
    POCID = 100 * np.sum(D) / n
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

def rolling_window(series, window):
  data = []
  for i in range(len(series)-window):
    example = znorm(np.array(series[i:i+window+1]))
    data.append(example)
  df = pd.DataFrame(data)
  return df

def rolling_window_series(series, window):
    result_series = pd.Series(index=series.index)
    mean = 0
    std = 0
    for i in range(len(series)-window):
        window_values = series.iloc[i:i+window+1].values
        normalized_values = znorm(window_values)
        result_series.iloc[i:i+window+1] = normalized_values
        mean = np.mean(series.iloc[i:i+window+1])
        std = np.std(series.iloc[i:i+window+1])
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

def transform_train(series_transform, format="deseasonal"):
    if format == "deseasonal":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ts = transform.transform(series_transform)
        return series_ts
    elif format == "diff":
        series_diff = series_transform.diff()
        return series_diff
    elif format == "log":
        constante = 10
        return np.log(series_transform + constante)
    elif format == "log-diff":
        constante = 10
        series_log = np.log(series_transform + constante)
        series_diff = series_log.diff()
        return series_diff
    elif format == "deseasonal-diff":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ds = transform.transform(series_transform)
        series_diff = series_ds.diff()
        return series_diff
    elif format == "deseasonal-log":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(series_transform)
        series_ds = transform.transform(series_transform)
        constante = 10
        series_log = np.log(series_ds + constante)
        return series_log

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

def fit_arima_train(train_ds, train_norm, initial_order, horizon, format):
    p, d, q = initial_order
    p_original, d_original, q_original = initial_order
    max_attempts = 10
    d_incremented = d+1
    try:
        forecast = ARIMA(order=(p_original, d, q_original), suppress_warnings=True)
        forecast.fit(train_ds)

        preds = recursive_forecasting_stats(train_ds, forecast, horizon)
        preds_inverse = transform_reverse_preds(preds, train_norm, format=format)

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

          preds = recursive_forecasting_stats(train_ds, forecast, horizon)
          preds_inverse = transform_reverse_preds(preds, train_norm, format=format)

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

            preds = recursive_forecasting_stats(train_ds, forecast, horizon)
            preds_inverse = transform_reverse_preds(preds, train_norm, format=format)

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
    raise ValueError("Problem with all possible (p,d,q) in this time series") from e


def fit_sarima_train(train_ds, train_norm, initial_order, seasonal_order, horizon, format):
    p, d, q = initial_order
    p_original, d_original, q_original = initial_order
    max_attempts = 10
    d_incremented = d+1
    try:
        forecast = ARIMA(order=(p_original, d, q_original), seasonal_order=seasonal_order, suppress_warnings=True)
        forecast.fit(train_ds)

        preds = recursive_forecasting_stats(train_ds, forecast, horizon)
        preds_inverse = transform_reverse_preds(preds, train_norm, format=format)

        final_order_dict = {
            'p': p_original,
            'd': d,
            'q': q_original,
        }

        return forecast, preds_inverse, final_order_dict

    except Exception as e:
        print_log(f"Exception: Not valid original ({p},{d},{q}) for train. Error: {e}")
        try:
          forecast = ARIMA(order=(p_original, d_incremented, q_original), seasonal_order=seasonal_order, suppress_warnings=True)
          forecast.fit(train_ds)

          preds = recursive_forecasting_stats(train_ds, forecast, horizon)
          preds_inverse = transform_reverse_preds(preds, train_norm, format=format)

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
            forecast = ARIMA(order=(p, d, q), seasonal_order=seasonal_order, suppress_warnings=True)
            forecast.fit(train_ds)

            preds = recursive_forecasting_stats(train_ds, forecast, horizon)
            preds_inverse = transform_reverse_preds(preds, train_norm, format=format)

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

def reverse_transform_norm_preds(series_preds, train, format="deseasonal"):
    if format == "deseasonal":
        transform = ConditionalDeseasonalizer(sp=12)
        transform.fit(train)
        series_before_norm = transform.transform(train)
        
        _, mean, std = rolling_window_series(series_before_norm, 12)
        preds_transformed = znorm_reverse(series_preds, mean, std)
        
        series_real = transform.inverse_transform(preds_transformed)
        return series_real
    elif format == "diff":
        series_before_norm = train.diff()
        
        _, mean, std = rolling_window_series(series_before_norm, 12)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        series_preds = pd.concat([train.iloc[[-1]], preds_transformed])
        series_norm = series_preds.cumsum()[1:]
        return series_norm
    elif format == "log":
        constante = 10
        series_before_norm = np.log(train + constante)
        
        _, mean, std = rolling_window_series(series_before_norm, 12)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        return np.exp(preds_transformed) - constante
    elif format == "log-diff":
        constante = 10
        series_train_log = np.log(train + constante)
        series_before_norm = series_train_log.diff()
        
        _, mean, std = rolling_window_series(series_before_norm, 12)
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

        _, mean, std = rolling_window_series(train_diff, 12)
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

        _, mean, std = rolling_window_series(train_log, 12)
        preds_transformed = znorm_reverse(series_preds, mean, std)

        series_ds = np.exp(preds_transformed)
        series_ds_abs = series_ds - constante
        series_inverse = transform.inverse_transform(series_ds_abs)
        return series_inverse

    #normal
    return series_preds

