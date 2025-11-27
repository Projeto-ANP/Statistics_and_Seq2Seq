import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

def detect_seasonality(time_series, period=None):
    """
    Detects seasonality in the time series using seasonal decomposition.
    Returns True if seasonality is significant, False otherwise.
    """
    if period is None:
        period = 7
    try:
        decomposition = seasonal_decompose(time_series, model='additive', period=period)
        seasonal_var = np.var(decomposition.seasonal)
        residual_var = np.var(decomposition.resid.dropna())
        if seasonal_var > residual_var * 2:  # arbitrary threshold
            return True
    except:
        pass
    return False

def selective_combination(predictions, statistical_models):
    """
    Combines predictions from statistical models only.
    predictions: dict of model_name: prediction_array
    statistical_models: list of model names considered statistical
    Returns combined prediction as numpy array
    """
    selected_preds = [predictions[model] for model in statistical_models if model in predictions]
    if not selected_preds:
        raise ValueError("No statistical models found in predictions")
    return np.mean(selected_preds, axis=0)

def full_combination(predictions):
    """
    Combines all predictions using simple average.
    predictions: dict of model_name: prediction_array
    Returns combined prediction as numpy array
    """
    if not predictions:
        raise ValueError("No predictions provided")
    all_preds = list(predictions.values())
    return np.mean(all_preds, axis=0)

# Placeholder for future combination functions
def weighted_combination(predictions, weights):
    """
    Combines predictions with given weights.
    predictions: dict of model_name: prediction_array
    weights: dict of model_name: weight
    """
    combined = np.zeros_like(list(predictions.values())[0])
    total_weight = 0
    for model, pred in predictions.items():
        weight = weights.get(model, 1.0)
        combined += weight * pred
        total_weight += weight
    return combined / total_weight if total_weight > 0 else combined