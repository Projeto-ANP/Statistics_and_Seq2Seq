
from agno.agent import Agent
from agno.tools import tool
import numpy as np
import pandas as pd
import json
import sys
import os
from sklearn.metrics import mean_absolute_percentage_error as mape

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import all_functions

from typing import List
from pydantic import BaseModel, Field
from agent import CONTEXT_MEMORY

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

    predictions = CONTEXT_MEMORY["predictions"]

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
    CONTEXT_MEMORY["tools_called"].append("mean_combination_tool")

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

    predictions = CONTEXT_MEMORY["predictions"]

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
    CONTEXT_MEMORY["tools_called"].append("weight_combination_tool")

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
    predictions = CONTEXT_MEMORY["predictions"]
    for model_name, point in model_points.items():
        if model_name in predictions:
            model_preds = predictions[model_name]
            combined.append(model_preds[point])
            

    print(f"TOOL [point_combination_tool] | FINISHED TOOL CALL.")
    CONTEXT_MEMORY["tools_called"].append("point_combination_tool")

    return combined