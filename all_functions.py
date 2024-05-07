import warnings
import pandas as pd
import numpy as np
from aeon.forecasting.arima import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error as mse
from aeon.transformations.detrend import ConditionalDeseasonalizer
from aeon.transformations.boxcox import BoxCoxTransformer
from datetime import datetime

warnings.filterwarnings("ignore")

def znorm(x):
  x_znorm = (x - np.mean(x)) / np.std(x)
  return x_znorm

def znorm_by(x, serie_ref):
  mean = np.mean(serie_ref)
  std = np.std(serie_ref)
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

def rolling_window_stats(series, window):
    data = []
    for i in range(len(series)-window):
      example = znorm(np.array(series.iloc[i:i+window+1]))
      data.append(pd.Series(example)) 
    result_series = pd.concat(data, ignore_index=True)
    return result_series

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

def rescaled_split_series(series_diff, normal_series, horizon=12):
  media = np.mean(normal_series.diff())
  desvio = np.std(normal_series.diff())

  completa= pd.concat(series_diff)
  completa_z_reverse = znorm_reverse(completa, media, desvio)

  serie_normalizada = add_trend(completa_z_reverse, normal_series)
  train_rescaled, test_rescaled = train_test_stats(serie_normalizada, horizon)

  return train_rescaled, test_rescaled

def pbe(y_true, y_pred):
  return 100*(np.sum(y_true - y_pred)/np.sum(y_true))

def revert_series(series_transformed, series_normal, format="box-diff", horizon=12):
    
    if format == "box-diff":
        boxcox_trans = BoxCoxTransformer()
        series_bc = boxcox_trans.fit_transform(series_normal)
        series_bc_diff = remove_trend(series_bc)
        media = np.mean(series_bc_diff)
        desvio = np.std(series_bc_diff)

        completa = pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)
        
        #adicionar tendencia
        series_trends = add_trend(completa_z_reverse, series_bc)
        
        #adicionar variancia
        series_completa_inversa = boxcox_trans.inverse_transform(series_trends)
        

        return train_test_stats(series_completa_inversa, horizon)
    elif format == "deseasonal":
        _, series_ds = remove_seasonal(series_normal)

        media = np.mean(series_ds)
        desvio = np.std(series_ds)
        completa = pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)

        serie_normalizada = deseasonal_trans.inverse_transform(completa_z_reverse)   

        return serie_normalizada    
    elif format == "log-diff":
        series_log = np.log(series_normal)
        series_diff_log = remove_trend(series_log)
        media = np.mean(series_diff_log)
        desvio = np.std(series_diff_log)

        completa = pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)

        series_trends = add_trend(completa_z_reverse, series_log)
        series_completa_inversa = np.exp(series_trends)
        
        return train_test_stats(series_completa_inversa, horizon)
    
    elif format == "diff":
        media = np.mean(series_normal.diff())
        desvio = np.std(series_normal.diff())

        completa= pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)

        serie_normalizada = add_trend(completa_z_reverse, series_normal)
        return train_test_stats(serie_normalizada, horizon)
    
    elif format == "deseasonal-log-diff":
        deseasonal_trans = ConditionalDeseasonalizer(sp=12)
        series_ds = deseasonal_trans.fit_transform(series_normal)
        
        series_log = np.log(series_ds)
        series_diff_log = remove_trend(series_log)
        media = np.mean(series_diff_log)
        desvio = np.std(series_diff_log)

        completa = pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)

        series_trends = add_trend(completa_z_reverse, series_log)
        series_completa_inversa = np.exp(series_trends)
        
        series_completa_inversa = deseasonal_trans.inverse_transform(series_completa_inversa)

        return train_test_stats(series_completa_inversa, horizon)

    elif format == "deseasonal-diff":
        deseasonal_trans = ConditionalDeseasonalizer(sp=12)
        series_ds = deseasonal_trans.fit_transform(series_normal)

        media = np.mean(series_ds.diff())
        desvio = np.std(series_ds.diff())

        completa= pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)

        serie_normalizada = add_trend(completa_z_reverse, series_ds)
        serie_normalizada = deseasonal_trans.inverse_transform(serie_normalizada)

        return train_test_stats(serie_normalizada, horizon)

    elif format == "deseasonal-box-diff":
        deseasonal_trans = ConditionalDeseasonalizer(sp=12)
        series_ds = deseasonal_trans.fit_transform(series_normal)

        boxcox_trans = BoxCoxTransformer()
        series_bc = boxcox_trans.fit_transform(series_ds)
        series_bc_diff = remove_trend(series_bc)
        media = np.mean(series_bc_diff)
        desvio = np.std(series_bc_diff)

        completa = pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)
        
        #adicionar tendencia
        series_trends = add_trend(completa_z_reverse, series_bc)
        
        #adicionar variancia
        series_completa_inversa = boxcox_trans.inverse_transform(series_trends)
        
        #adicionar sazonalidade
        series_completa_inversa = deseasonal_trans.inverse_transform(series_completa_inversa)

        return train_test_stats(series_completa_inversa, horizon)
    
    elif format == "deseasonal-log":
        deseasonal_trans = ConditionalDeseasonalizer(sp=12)
        series_ds = deseasonal_trans.fit_transform(series_normal)
        
        series_log = np.log(series_ds)
        # series_diff_log = remove_trend(series_log)
        media = np.mean(series_log)
        desvio = np.std(series_log)

        completa = pd.concat(series_transformed)
        completa_z_reverse = znorm_reverse(completa, media, desvio)

        # series_trends = add_trend(completa_z_reverse, series_log)
        series_completa_inversa = np.exp(completa_z_reverse)
        
        series_completa_inversa = deseasonal_trans.inverse_transform(series_completa_inversa)

        return train_test_stats(series_completa_inversa, horizon)

    return pd.Series(), pd.Series()


def transform_series(series, format="box-diff", horizon=12):
    _, series_noseasonal = remove_seasonal(series)
    series_log = np.log(series)
    
    if format == "deseasonal":
      return series_noseasonal
    elif format == "log":
      return series_log
    elif format == "diff":
      return remove_trend(series)
    elif format == "deseasonal-diff":
      _, series_noseasonal = remove_seasonal(series)
      series_ds_diff = remove_trend(series_noseasonal)
      return series_ds_diff
    elif format == "box-diff":
      boxcox_trans = BoxCoxTransformer()
      series_bc = boxcox_trans.fit_transform(series)
      series_bc_diff = remove_trend(series_bc)
      return series_bc_diff
    elif format == "deseasonal-box":
      series_ds_bc = boxcox_trans.fit_transform(series_noseasonal)
      return series_ds_bc
    elif format == "log-diff":
      series_log_diff = remove_trend(series_log)
      return series_log_diff
    elif format == "deseasonal-log":
      series_ds_log = np.log(series_noseasonal)
      return series_ds_log
    
    return series

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
        # series_preds.iloc[[0]] = train_norm.iloc[[-1]]
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
        # print(series_log)
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
        # train_des = transform.transform(train_norm)
        series_ds = np.exp(series_preds)
        constante = 10
        series_ds_abs = series_ds - constante
        # print(series_ds_abs)
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

def print_log(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] {message}")