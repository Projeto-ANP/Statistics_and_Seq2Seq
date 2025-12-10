from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
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
def calculate_metrics_tool() -> str:
    """
    Calculates performance metrics (MAPE, RMSE, SMAPE, POCID) for each model.
    Uses validation data from shared context - no parameters needed.
    Call this first to analyze model performance.

    Returns:
        JSON string with metrics for each model.
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

    validation_test = CONTEXT_MEMORY["validation_test"]
    validation_predictions = CONTEXT_MEMORY["validation_predictions"]

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

    CONTEXT_MEMORY["calculated_metrics"] = results

    print(f"[TOOL RESULT] Calculated metrics for {len(results)} models")
    CONTEXT_MEMORY["tools_called"].append("calculate_metrics_tool")

    return json.dumps(results, indent=2)
