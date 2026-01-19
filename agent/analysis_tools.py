from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
import numpy as np
import json
import sys
import os
from sklearn.metrics import mean_absolute_percentage_error as mape

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import all_functions

from typing import List, Dict
from agent.context import get_context, set_context, init_context

@tool
def calculate_metrics_tool() -> str:
    """
    Calculates performance metrics (MAPE, RMSE, SMAPE, POCID) for each model in each window.

    Returns:
        JSON string with metrics for each model in each window.
    """
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        import all_functions
    except ImportError:
        print("Warning: all_functions module not found. Using mock functions.")

        class MockFunctions:
            @staticmethod
            def calculate_rmse(p, t):
                return [np.sqrt(np.mean((p - t) ** 2))]

            @staticmethod
            def calculate_smape(p, t):
                return [np.mean(2 * np.abs(p - t) / (np.abs(p) + np.abs(t)))]

            @staticmethod
            def pocid(t, p):
                return 0.0

        all_functions = MockFunctions()

    print(f"\n[TOOL CALL] calculate_metrics_tool called")
    
    # Verificar se o contexto foi inicializado
    all_validations = get_context("all_validations")
    if not all_validations or "test" not in all_validations:
        error_msg = "CONTEXT_MEMORY not initialized. Please call init_context() and generate_all_validations_context() first."
        print(f"[ERROR] {error_msg}")
        return json.dumps({"error": error_msg})
    
    tests = all_validations["test"]
    
    all_metrics = []
    for (window_idx, test_window) in enumerate(tests):

        validation_test = test_window
        validation_predictions = all_validations["predictions"][window_idx]

        if validation_test is None or validation_predictions is None:
            return json.dumps(
                {"error": "Shared context not initialized. Call set_shared_context first."}
            )

        results = {}
        y_true = np.array(validation_test)

        for model_name, y_pred_list in validation_predictions.items():
            if not y_pred_list:
                continue

            y_pred = np.array(y_pred_list)

            min_len = min(len(y_true), len(y_pred))
            if min_len == 0:
                continue

            curr_y_true = y_true[:min_len]
            curr_y_pred = y_pred[:min_len]

            mape_value = mape(curr_y_true, curr_y_pred)
            rmse = all_functions.calculate_rmse(
                curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1)
            )[0]
            smape = all_functions.calculate_smape(
                curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1)
            )[0]
            pocid_value = all_functions.pocid(curr_y_true, curr_y_pred)

            results[model_name] = {
                "MAPE": round(float(mape_value), 4),
                "RMSE": round(float(rmse), 2),
                "SMAPE": round(float(smape), 4),
                "POCID": round(float(pocid_value), 2),
            }
        all_metrics.append({"window_index": window_idx, "metrics": results})

    set_context("calculated_metrics", all_metrics)

    print(f"[TOOL RESULT] Calculated metrics for {len(results)} models")
    
    tools_called = get_context("tools_called", [])
    tools_called.append("calculate_metrics_tool")
    set_context("tools_called", tools_called)

    return json.dumps(all_metrics, indent=2)

import numpy as np
import json
from typing import Dict, List, Tuple, Union


@tool
def generate_ade_point_models_tool(top_k: int = 3) -> Dict[int, List[str]]:
    """
    Generates point-by-point model selection based on RMSE performance across validation windows.
    Analyzes which models perform best at each forecast horizon point.
    
    Uses validation data from context["all_validations"] with structure:
        {
            "predictions": [
                {window_0: {"ARIMA": [h1, h2, ...], "CatBoost": [h1, h2, ...], ...}},
                {window_1: {"ARIMA": [h1, h2, ...], "CatBoost": [h1, h2, ...], ...}},
                ...
            ],
            "test": [
                {window_0: [actual_h1, actual_h2, ...]},
                {window_1: [actual_h1, actual_h2, ...]},
                ...
            ]
        }
    
    Args:
        top_k (int): Number of top models to select for each point.
                     Default is 1 (only best model per point).
                     Use higher values for ensemble at each point.
    
    Returns:
        Dict[int, List[str]]: Dictionary with prediction points as keys and
                              list of best model names as values.
                              e.g., {0: ["CatBoost"], 1: ["ARIMA", "XGBoost"], 2: ["CatBoost"]}
    """
    print(f"\nTOOL [generate_ade_point_models_tool] | called with top_k: {top_k}")
    
    all_validations = get_context("all_validations")
    
    if all_validations is None:
        print("ERROR: all_validations not found in CONTEXT_MEMORY")
        return {}
    
    predictions_list = all_validations.get("predictions", [])
    test_list = all_validations.get("test", [])
    
    if not predictions_list or not test_list:
        print("ERROR: predictions or test data is empty")
        return {}
    
    # Get number of windows and models
    n_windows = len(predictions_list)
    first_window_preds = predictions_list[0]
    model_names = list(first_window_preds.keys())
    
    # Determine forecast horizon (number of points)
    first_model = model_names[0]
    horizon = len(first_window_preds[first_model])
    
    print(f"TOOL [generate_ade_point_models_tool] | Found {n_windows} windows, {len(model_names)} models, horizon={horizon}")
    
    # Calculate RMSE for each model at each point across all windows
    # Structure: {point: {model: [errors_across_windows]}}
    point_model_errors: Dict[int, Dict[str, List[float]]] = {
        point: {model: [] for model in model_names}
        for point in range(horizon)
    }
    
    for window_idx in range(n_windows):
        window_preds = predictions_list[window_idx]
        window_test = test_list[window_idx]
        
        for point in range(horizon):
            if point >= len(window_test):
                continue
                
            actual_value = window_test[point]
            
            for model_name in model_names:
                if model_name not in window_preds:
                    continue
                    
                model_preds = window_preds[model_name]
                if point >= len(model_preds):
                    continue
                
                pred_value = model_preds[point]
                
                # Squared error for this point
                squared_error = (pred_value - actual_value) ** 2
                point_model_errors[point][model_name].append(squared_error)
    
    # Calculate RMSE for each model at each point and select top_k
    point_models: Dict[int, List[str]] = {}
    point_model_rmse: Dict[int, Dict[str, float]] = {}
    
    for point in range(horizon):
        model_rmse = {}
        
        for model_name in model_names:
            errors = point_model_errors[point][model_name]
            if errors:
                rmse = np.sqrt(np.mean(errors))
                model_rmse[model_name] = rmse
        
        # Sort models by RMSE (ascending - lower is better)
        sorted_models = sorted(model_rmse.items(), key=lambda x: x[1])
        
        # Select top_k models
        top_models = [model for model, _ in sorted_models[:top_k]]
        point_models[point] = top_models
        point_model_rmse[point] = model_rmse
        
        print(f"  Point {point}: Best={top_models} | RMSE scores: {model_rmse}")
    
    # Store results in context
    set_context("point_models", point_models)
    set_context("point_model_rmse", point_model_rmse)
    
    print(f"TOOL [generate_ade_point_models_tool] | FINISHED TOOL CALL.")
    tools_called = get_context("tools_called", [])
    tools_called.append("generate_ade_point_models_tool")
    set_context("tools_called", tools_called)
    
    return point_models


@tool
def generate_ade_weighted_point_models_tool(top_k: int = 3) -> Dict[int, Dict[str, float]]:
    """
    Generates point-by-point model weights based on RMSE performance across validation windows.
    Uses inverse RMSE normalization to calculate weights (similar to ADE approach).
    
    Uses validation data from context["all_validations"] with structure:
        {
            "predictions": [
                {window_0: {"ARIMA": [h1, h2, ...], "CatBoost": [h1, h2, ...], ...}},
                {window_1: {"ARIMA": [h1, h2, ...], "CatBoost": [h1, h2, ...], ...}},
                ...
            ],
            "test": [
                {window_0: [actual_h1, actual_h2, ...]},
                {window_1: [actual_h1, actual_h2, ...]},
                ...
            ]
        }
    
    Args:
        top_k (int): Number of top models to include in weights for each point.
                     Models outside top_k will have weight = 0.
                     Default is 3.
    
    Returns:
        Dict[int, Dict[str, float]]: Dictionary with prediction points as keys and
                                     dictionaries of {model_name: weight} as values.
                                     e.g., {0: {"CatBoost": 0.6, "ARIMA": 0.4}, 1: {...}}
    """
    print(f"\nTOOL [generate_ade_weighted_point_models_tool] | called with top_k: {top_k}")
    
    all_validations = get_context("all_validations")
    
    if all_validations is None:
        print("ERROR: all_validations not found in CONTEXT_MEMORY")
        return {}
    
    predictions_list = all_validations.get("predictions", [])
    test_list = all_validations.get("test", [])
    
    if not predictions_list or not test_list:
        print("ERROR: predictions or test data is empty")
        return {}
    
    # Get number of windows and models
    n_windows = len(predictions_list)
    first_window_preds = predictions_list[0]
    model_names = list(first_window_preds.keys())
    
    # Determine forecast horizon (number of points)
    first_model = model_names[0]
    horizon = len(first_window_preds[first_model])
    
    print(f"TOOL [generate_ade_weighted_point_models_tool] | Found {n_windows} windows, {len(model_names)} models, horizon={horizon}")
    
    # Calculate RMSE for each model at each point across all windows
    point_model_errors: Dict[int, Dict[str, List[float]]] = {
        point: {model: [] for model in model_names}
        for point in range(horizon)
    }
    
    for window_idx in range(n_windows):
        window_preds = predictions_list[window_idx]
        window_test = test_list[window_idx]
        
        for point in range(horizon):
            if point >= len(window_test):
                continue
                
            actual_value = window_test[point]
            
            for model_name in model_names:
                if model_name not in window_preds:
                    continue
                    
                model_preds = window_preds[model_name]
                if point >= len(model_preds):
                    continue
                
                pred_value = model_preds[point]
                squared_error = (pred_value - actual_value) ** 2
                point_model_errors[point][model_name].append(squared_error)
    
    # Calculate weights using inverse RMSE with min-max normalization
    point_model_weights: Dict[int, Dict[str, float]] = {}
    
    for point in range(horizon):
        model_rmse = {}
        
        for model_name in model_names:
            errors = point_model_errors[point][model_name]
            if errors:
                rmse = np.sqrt(np.mean(errors))
                model_rmse[model_name] = rmse
        
        if not model_rmse:
            point_model_weights[point] = {}
            continue
        
        # Sort and select top_k models
        sorted_models = sorted(model_rmse.items(), key=lambda x: x[1])
        top_models_rmse = dict(sorted_models[:top_k])
        
        # Apply ADE-style normalization:
        # 1. Invert RMSE (lower error = higher weight)
        # 2. Min-Max normalize
        # 3. Convert to proportions
        
        rmse_values = np.array(list(top_models_rmse.values()))
        
        # Invert: use negative so lower RMSE becomes higher value
        inverted = -rmse_values
        
        # Min-Max normalization to [0, 1]
        min_val = inverted.min()
        max_val = inverted.max()
        
        if max_val - min_val > 0:
            normalized = (inverted - min_val) / (max_val - min_val)
        else:
            # All models have same RMSE, equal weights
            normalized = np.ones_like(inverted)
        
        # Convert to proportions (sum = 1)
        if normalized.sum() > 0:
            proportions = normalized / normalized.sum()
        else:
            proportions = np.ones_like(normalized) / len(normalized)
        
        # Build weight dictionary
        weights = {}
        for idx, model_name in enumerate(top_models_rmse.keys()):
            weights[model_name] = round(float(proportions[idx]), 4)
        
        point_model_weights[point] = weights
        
        print(f"  Point {point}: Weights={weights}")
    
    set_context("point_model_weights", point_model_weights)
    
    print(f"TOOL [generate_ade_weighted_point_models_tool] | FINISHED TOOL CALL.")
    tools_called = get_context("tools_called", [])
    tools_called.append("generate_ade_weighted_point_models_tool")
    set_context("tools_called", tools_called)
    
    return point_model_weights