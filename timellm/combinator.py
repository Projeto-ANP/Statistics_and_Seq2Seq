from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
import os
import sys
import json
import re
import numpy as np
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_percentage_error as mape

SHARED_CONTEXT = {
    "validation_test": None,
    "validation_predictions": None,
    "final_predictions": None,
    "calculated_metrics": None
}

CONTEXT_MEMORY = {}

def set_shared_context(validation_test, validation_predictions, final_predictions):
    """Define os dados compartilhados que as tools irão usar."""
    SHARED_CONTEXT["validation_test"] = validation_test
    SHARED_CONTEXT["validation_predictions"] = validation_predictions
    SHARED_CONTEXT["final_predictions"] = final_predictions
    SHARED_CONTEXT["calculated_metrics"] = None

def set_context(key: str, value: Any):
    CONTEXT_MEMORY[key] = value

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

def clean_json_string(text: str) -> dict:
    text = text.strip()
    
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
        except:
            pass
        print(f"Erro ao parsear JSON final: {e}")
        print(f"Texto recebido: {text[:200]}...")
        raise

class CombinationResult(BaseModel):
    description: str = Field(..., description="Descrição do método de combinação utilizado")
    result: List[float] = Field(..., description="Lista com as predições combinadas resultantes")
    selected_models: str = Field(..., description="Modelos selecionados para combinação")

# --- Tools ---

@tool
def calculate_metrics_tool() -> str:
    """
    Calculates performance metrics (MAPE, RMSE, SMAPE, POCID) for each model.
    Uses validation data from shared context - no parameters needed.
    Call this first to analyze model performance.
    
    Returns:
        JSON string with metrics for each model.
    """
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        import all_functions
    except ImportError:
        print("Warning: all_functions module not found. Using mock functions.")
        class MockFunctions:
            @staticmethod
            def calculate_rmse(p, t): return [np.sqrt(np.mean((p-t)**2))]
            @staticmethod
            def calculate_smape(p, t): return [np.mean(2 * np.abs(p-t) / (np.abs(p) + np.abs(t)))]
            @staticmethod
            def pocid(t, p): return 0.0
        all_functions = MockFunctions()
    
    print(f"\n[TOOL CALL] calculate_metrics_tool called")
    
    # Usar dados da memória compartilhada
    validation_test = SHARED_CONTEXT["validation_test"]
    validation_predictions = SHARED_CONTEXT["validation_predictions"]
    
    if validation_test is None or validation_predictions is None:
        return json.dumps({"error": "Shared context not initialized. Call set_shared_context first."})

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
        rmse = all_functions.calculate_rmse(curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1))[0]
        smape = all_functions.calculate_smape(curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1))[0]
        pocid_value = all_functions.pocid(curr_y_true, curr_y_pred)

        results[model_name] = {
            "MAPE": round(float(mape_value), 4),
            "RMSE": round(float(rmse), 2),
            "SMAPE": round(float(smape), 4),
            "POCID": round(float(pocid_value), 2)
        }
    
    SHARED_CONTEXT["calculated_metrics"] = results
        
    print(f"[TOOL RESULT] Calculated metrics for {len(results)} models")
    return json.dumps(results, indent=2)


@tool
def selective_combine_tool(models_to_combine: List[str]) -> str:
    """
    Combines predictions from specified models using mean averaging.
    Uses final predictions from shared context.
    
    Args:
        models_to_combine: List of model names to combine. Example: ["ARIMA", "ETS", "THETA"]
    
    Returns:
        List with combined predictions.
    """
    print(f"\n[TOOL CALL] selective_combine_tool called")
    print(f"[TOOL INFO] Models to combine: {models_to_combine}")
    
    predictions = SHARED_CONTEXT["final_predictions"]
    
    if predictions is None:
        return json.dumps({"error": "Shared context not initialized.", "result": []})

    valid_models = [m for m in models_to_combine if m in predictions]
    
    if not valid_models:
        print(f"[TOOL WARNING] No valid models found. Using all available models.")
        valid_models = list(predictions.keys())
    
    print(f"[TOOL INFO] Combining {len(valid_models)} models: {valid_models}")
    
    pred_arrays = {k: np.array(v) for k, v in predictions.items() if k in valid_models}
    
    if not pred_arrays:
        return json.dumps({"result": [], "error": "No valid prediction arrays found."})

    combined = full_combination(pred_arrays)
    combined_list = [round(x, 2) for x in combined.tolist()]
    
    print(f"[TOOL RESULT] Combined predictions generated.")
    
    return combined_list


def agent_combinator(model_id: str, temperature: float):
    instructions = """You are a Time Series Analyst Agent with access to tools.

YOUR TASK: Analyze model performance and combine the best predictions.

AVAILABLE TOOLS:
1. calculate_metrics_tool() - No parameters needed. Calculates MAPE, RMSE, SMAPE, POCID for all models.
2. selective_combine_tool(models_to_combine) - Takes a list of model names to combine.

EXECUTION STEPS:
1. Call calculate_metrics_tool() to get metrics for all models
2. Analyze the results: Lower MAPE/RMSE/SMAPE is better, Higher POCID is better
3. Select the top 3 best performing models
4. Call selective_combine_tool with the list of selected model names
5. Return final JSON with description and result

IMPORTANT: 
- calculate_metrics_tool takes NO parameters
- selective_combine_tool takes ONLY a list of model names like ["ARIMA", "ETS", "THETA"]
"""
    
    return Agent(
        model=Ollama(
            id=model_id, 
            options={
                "temperature": temperature, 
                "num_ctx": 8192,
                "keep_alive": "5m"
            }
        ),
        tools=[calculate_metrics_tool, selective_combine_tool],
        instructions=instructions,
        markdown=True,
    )
    

    
def simple_agent(validation_test, validation_predictions, final_test_predictions):
    set_shared_context(validation_test, validation_predictions, final_test_predictions)
    
    agent = agent_combinator(
        model_id="qwen3:14b", 
        temperature=0.0
    )
    
    print("=" * 80)
    print("AGENT EXECUTION")
    print("=" * 80)
    print(f"Model: {agent.model.id}")
    print(f"Models Available: {list(validation_predictions.keys())}")
    print("=" * 80 + "\n")
    
    prompt = f"""Analyze the following models and combine the best predictions.

Available models: {list(validation_predictions.keys())}

STEPS:
1. Call calculate_metrics_tool() to get performance metrics (no parameters needed)
2. Identify the top 3 models with best performance (low MAPE/RMSE/SMAPE, high POCID)
3. Call selective_combine_tool with a list of the selected model names
4. Return a JSON with "description" and "result" fields
"""
    
    print("Sending prompt to agent...")
    print("-" * 80)
    try:
        response = agent.run(prompt)
        print("\n[AGENT RAW RESPONSE]:\n")
        print(response.content)
        
        output = clean_json_string(response.content)
        return output.get("description", ""), output.get("result", [])
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "Error", []


def simple_selective_agent(validation_test, validation_predictions, final_test_predictions):
    # set_context("validation_test", validation_test)
    # set_context("validation_predictions", validation_predictions)
    # set_context("final_predictions", final_test_predictions)
    
    set_shared_context(validation_test, validation_predictions, final_test_predictions)
    
    
    temperature = 0.0
    model_id = "qwen3:14b"
    
    instructions = """You are a Time Series Analyst Agent with access to tools.

YOUR TASK: Analyze model performance and combine the best predictions by each category.

AVAILABLE TOOLS that you will use by order:
1. calculate_metrics_tool() - No parameters needed. Calculates MAPE, RMSE, SMAPE, POCID for all models and return a list of the models_to_combine.
2. selective_combine_tool(models_to_combine) - Takes a list of model names to combine.

EXECUTION STEPS:
1. Call calculate_metrics_tool() to get metrics for all models
2. Analyze the results: Lower MAPE/RMSE/SMAPE is better, Higher POCID is better
3. Select the top best performed models by RMSE from each category: Statistical, Naive and between each ML model for each representation (e.g., for RF select the best between RF with CWT, DWT, RAW, FT)
4. After the result from calculate_metrics_tool, call selective_combine_tool with the list of selected model names
5. Return final JSON with description and result

IMPORTANT:
ALWAYS CALL THE TOOLS 
- calculate_metrics_tool takes NO parameters
- selective_combine_tool takes ONLY a list of model names like ["ARIMA", "NaiveSeasonal", "CWT_rf", "FT_svr"...] and will return the combined predictions from each model selected.
"""
    
    agent = Agent(
        model=Ollama(
            id=model_id, 
            options={
                "temperature": temperature, 
                "num_ctx": 8192,
                "keep_alive": "5m"
            }
        ),
        tools=[calculate_metrics_tool, selective_combine_tool],
        instructions=instructions,
        markdown=True,
    )
    
    print("=" * 80)
    print("AGENT EXECUTION")
    print("=" * 80)
    print(f"Model: {agent.model.id}")
    print(f"Models Available: {list(validation_predictions.keys())}")
    print("=" * 80 + "\n")
    
    prompt = f"""Analyze the following models and combine the best predictions.

Available models: {list(validation_predictions.keys())}

Always use the tools provided to you in order to get the best results.

STEPS:
1. Call calculate_metrics_tool() to get performance metrics (no parameters needed)
2. Identify the best top model by RMSE for each category: Statistical, Naive, and ML models by representation (e.g., for RF select the best between RF with CWT, DWT, RAW, FT which could be "CWT_rf", "DWT_rf", "ONLY_FT_rf", "ONLY_CWT_rf" or just "rf" if no representation is used)
3. After the result from calculate_metrics_tool, call the tool selective_combine_tool with a list of the selected model names
4. Return a JSON with "description" and "result" fields, where the result is the combined predictions from the selected models received from selective_combine_tool
"""
    
    print("Sending prompt to agent...")
    print("-" * 80)
    try:
        response = agent.run(prompt)
        print("\n[AGENT RAW RESPONSE]:\n")
        print(response.content)
        
        output = clean_json_string(response.content)
        return output.get("description", ""), output.get("result", [])
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "Error", []

    
if __name__ == "__main__":
    validation_test = [17656.623, 19507.078, 15680.762, 19775.546, 13736.136, 17221.028, 18352.012, 20327.377, 21175.424, 18516.637, 19864.665, 18523.176]
    
    validation_predictions = {
        "ARIMA": [x * 1.05 for x in validation_test], 
        "ETS": [x * 0.95 for x in validation_test], 
        "THETA": [x * 1.02 for x in validation_test]
    }
    
    final_test_series = [22026.58] * 12
    final_predictions = {
        "ARIMA": [22000.0] * 12, 
        "ETS": [21500.0] * 12, 
        "THETA": [21800.0] * 12
    }

    desc, res = simple_agent(validation_test, validation_predictions, final_predictions)
    print("\n--- FINAL OUTPUT ---")
    print(f"Description: {desc}")
    print(f"Result: {res}")