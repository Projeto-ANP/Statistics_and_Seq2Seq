
from agno.agent import Agent
from agno.tools import tool
import numpy as np
import pandas as pd
import json
import sys
import os
from sklearn.metrics import mean_absolute_percentage_error as mape
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import all_functions

from typing import List
from pydantic import BaseModel, Field
from context import get_context, set_context

@tool
def mean_combination_tool(
    model_names: List[str] = Field(
        ..., description="List of model names to combine predictions from."
    )
) -> List[float]:
    """
    Combines predictions from specified models by calculating the mean.

    Args:
        model_names (List[str]): List of model names to combine predictions from.

    Returns:
        List[float]: Combined predictions as a list of floats.
    """
    print(f"\nTOOL [mean_combination_tool] | called with models: {model_names}")

    predictions = get_context("predictions")

    if predictions is None:
        print(
            "TOOL [mean_combination_tool] | ERROR Shared context not initialized for predictions"
        )
        return []

    preds_to_combine = []
    for model_name in model_names:
        if model_name in predictions:
            preds_to_combine.append(np.array(predictions[model_name]))
        else:
            print(
                f"TOOL [mean_combination_tool] | Warning: Model {model_name} not found in predictions."
            )
            return []

    if not preds_to_combine:
        return []

    combined = np.mean(preds_to_combine, axis=0)
    combined_list = [round(x, 2) for x in combined.tolist()]

    print(f"TOOL [mean_combination_tool] | FINISHED TOOL CALL.")
    tools_called = get_context("tools_called", [])
    tools_called.append("mean_combination_tool")
    set_context("tools_called", tools_called)

    return combined_list


@tool
def weight_combination_tool(
    model_weights: dict = Field(
        ..., description="Dictionary of model names and their corresponding weights."
    )
) -> List[float]:
    """
    Combines predictions from specified models using weighted average.

    Args:
        model_weights (dict): Dictionary with model names as keys and their weights as values.

    Returns:
        List[float]: Combined predictions as a list of floats.
    """
    print(
        f"\nTOOL [weight_combination_tool] | called with model weights: {model_weights}"
    )

    predictions = get_context("predictions")

    if predictions is None:
        print(
            "TOOL [weight_combination_tool] | ERROR Shared context not initialized for predictions"
        )
        return []

    preds_to_combine = []
    weights = []
    for model_name, weight in model_weights.items():
        if model_name in predictions:
            preds_to_combine.append(np.array(predictions[model_name]))
            weights.append(weight)
        else:
            print(
                f"TOOL [weight_combination_tool] | Warning: Model {model_name} not found in predictions."
            )
            return []

    if not preds_to_combine:
        return []
    combined = np.average(preds_to_combine, axis=0, weights=weights)
    combined_list = [round(x, 2) for x in combined.tolist()]
    print(f"TOOL [weight_combination_tool] | FINISHED TOOL CALL.")
    tools_called = get_context("tools_called", [])
    tools_called.append("weight_combination_tool")
    set_context("tools_called", tools_called)

    return combined_list


@tool
def point_combination_tool(model_points: dict) -> List[float]:
    """
    Combines predictions from specified models using specified points.
    This idea is that each model is good predicting a specific point in the future.
    Args:
        model_points (dict): Dictionary with model names as keys and their corresponding prediction point as values.
        e.g., {"modelA": 0, "modelB": 1, "modelD": 2}
    Returns:
        List[float]: Combined predictions as a list of floats.
    """
    print(f"\nTOOL [point_combination_tool] | called with model points: {model_points}")

    combined = []
    predictions = get_context("predictions")
    for model_name, point in model_points.items():
        if model_name in predictions:
            model_preds = predictions[model_name]
            combined.append(model_preds[point])

    print(f"TOOL [point_combination_tool] | FINISHED TOOL CALL.")
    tools_called = get_context("tools_called", [])
    tools_called.append("point_combination_tool")
    set_context("tools_called", tools_called)

    return combined

@tool
def ade_point_selection_tool(point_models: Dict[int, List[str]]) -> List[float]:
    """
    Combines predictions using ADE-style point-by-point model selection.
    For each prediction point, only the specified models are used (simple average).
    
    Args:
        point_models (dict): Dictionary with prediction points as keys and list of model names as values.
            e.g., {0: ["modelA", "modelB"], 1: ["modelB", "modelD"], 2: ["modelA", "modelC", "modelD"]}
    
    Returns:
        List[float]: Combined predictions as a list of floats, one per point.
    
    Example:
        >>> point_models = {
        ...     0: ["rf", "catboost"],      # Point 0: use rf and catboost
        ...     1: ["catboost", "ARIMA"],             # Point 1: use catboost and ARIMA
        ...     2: ["rf", "ARIMA", "THETA"]  # Point 2: use rf, ARIMA, THETA
        ... }
        >>> result = ade_point_selection_tool(point_models)
    """
    print(f"\nTOOL [ade_point_selection_tool] | called with point_models: {point_models}")
    
    combined = []
    predictions = get_context("predictions")
    
    sorted_points = sorted(point_models.keys())
    
    for point in sorted_points:
        models_for_point = point_models[point]
        point_values = []
        
        for model_name in models_for_point:
            if model_name in predictions:
                model_preds = predictions[model_name]
                if point < len(model_preds):
                    point_values.append(model_preds[point])
                else:
                    print(f"WARNING: Point {point} out of range for model {model_name}")
            else:
                print(f"WARNING: Model {model_name} not found in predictions")
        
        if point_values:
            combined.append(sum(point_values) / len(point_values))
        else:
            combined.append(None)
    
    print(f"TOOL [ade_point_selection_tool] | FINISHED TOOL CALL.")
    tools_called = get_context("tools_called", [])
    tools_called.append("ade_point_selection_tool")
    set_context("tools_called", tools_called)
    
    return combined


@tool
def ade_weighted_point_tool(point_model_weights: Dict[int, Dict[str, float]]) -> List[float]:
    """
    Combines predictions using ADE-style point-by-point weighted model combination.
    For each prediction point, models are combined using specified weights.
    
    Args:
        point_model_weights (dict): Dictionary with prediction points as keys and 
            dictionaries of {model_name: weight} as values. 
            e.g., {
                0: {"modelA": 0.6, "modelB":  0.4},
                1: {"modelB": 0.3, "modelD":  0.7},
                2: {"modelA": 0.5, "modelC":  0.3, "modelD": 0.2}
            }
            Note:  Weights for each point should sum to 1.0 for proper normalization,
                  but will be auto-normalized if they don't.
    
    Returns: 
        List[float]:  Combined predictions as a list of floats, one per point.
    
    Example: 
        >>> point_model_weights = {
        ...     0: {"rf": 0.7, "catboost": 0.3},
        ...      1: {"catboost": 0.4, "ARIMA": 0.6},
        ...     2: {"rf": 0.5, "ARIMA": 0.3, "THETA": 0.2}
        ... }
        >>> result = ade_weighted_point_tool(point_model_weights)
    """
    print(f"\nTOOL [ade_weighted_point_tool] | called with point_model_weights: {point_model_weights}")
    
    combined = []
    predictions = get_context("predictions")
    
    sorted_points = sorted(point_model_weights.keys())
    
    for point in sorted_points:
        models_weights = point_model_weights[point]
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, weight in models_weights.items():
            if model_name in predictions:
                model_preds = predictions[model_name]
                if point < len(model_preds):
                    weighted_sum += model_preds[point] * weight
                    total_weight += weight
                else:
                    print(f"WARNING: Point {point} out of range for model {model_name}")
            else:
                print(f"WARNING: Model {model_name} not found in predictions")
        
        if total_weight > 0:
            combined.append(weighted_sum / total_weight)
        else:
            combined.append(None)
    
    print(f"TOOL [ade_weighted_point_tool] | FINISHED TOOL CALL.")
    tools_called = get_context("tools_called", [])
    tools_called.append("ade_weighted_point_tool")
    set_context("tools_called", tools_called)
    
    return combined

