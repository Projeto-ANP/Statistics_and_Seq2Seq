from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
import os
import sys
import json
import re
import numpy as np
from typing import List
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_percentage_error as mape

# --- Funções Auxiliares ---

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
        print(f"Erro ao parsear JSON: {e}")
        print(f"Texto recebido: {text[:200]}...")
        raise

class CombinationResult(BaseModel):
    description: str = Field(..., description="Descrição do método de combinação utilizado")
    result: List[float] = Field(..., description="Lista com as predições combinadas resultantes")
    selected_models: str = Field(..., description="Modelos selecionados para combinação")

# --- Tools ---

@tool
def calculate_metrics_tool(validation_test: list, validation_predictions: dict) -> dict:
    """
    Calculates performance metrics (MAPE, RMSE, SMAPE, POCID) for each model.
    Use this tool FIRST to analyze which models are performing best on validation data.
    """
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        import all_functions
    except ImportError:
        print("Warning: all_functions module not found. Using mock functions for demonstration.")
        class MockFunctions:
            @staticmethod
            def calculate_rmse(p, t): return [np.sqrt(np.mean((p-t)**2))]
            @staticmethod
            def calculate_smape(p, t): return [np.mean(2 * np.abs(p-t) / (np.abs(p) + np.abs(t)))]
            @staticmethod
            def pocid(t, p): return 0.0
        all_functions = MockFunctions()
    
    print(f"\n[TOOL CALL] calculate_metrics_tool called")
    
    results = {}
    y_true = np.array(validation_test)
    
    for model_name, y_pred_list in validation_predictions.items():
        y_pred = np.array(y_pred_list)
        
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0: continue
            
        curr_y_true = y_true[:min_len]
        curr_y_pred = y_pred[:min_len]
        
        mape_value = mape(curr_y_true, curr_y_pred)
        rmse = all_functions.calculate_rmse(curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1))[0]
        smape = all_functions.calculate_smape(curr_y_pred.reshape(1, -1), curr_y_true.reshape(1, -1))[0]
        pocid = all_functions.pocid(curr_y_true, curr_y_pred)

        results[model_name] = {
            "MAPE": float(mape_value),
            "RMSE": float(rmse),
            "SMAPE": float(smape),
            "POCID": float(pocid)
        }
        
    print(f"[TOOL RESULT] Calculated metrics for {len(results)} models")
    print(json.dumps(results, indent=2))
    return json.dumps(results, indent=2)


@tool
def selective_combine_tool(predictions: dict, models_to_combine: list) -> dict:
    """
    Combines predictions from specified models.
    Use this tool SECOND, after choosing the best models based on the metrics.
    """
    print(f"\n[TOOL CALL] selective_combine_tool called")
    print(f"[TOOL INFO] Models to combine: {models_to_combine}")
    
    valid_models = [m for m in models_to_combine if m in predictions]
    
    if not valid_models:
        print(f"[TOOL WARNING] No valid models found. Using all available models.")
        valid_models = list(predictions.keys())
    
    print(f"[TOOL INFO] Combining {len(valid_models)} models: {valid_models}")
    
    pred_arrays = {k: np.array(v) for k, v in predictions.items() if k in valid_models}
    combined = full_combination(pred_arrays)
    print(f"[TOOL RESULT] Combined predictions generated.")
    
    return {
        "result": combined.tolist(),
        "models_used": valid_models,
        "method": f"Mean combination of {len(valid_models)} selected models"
    }

# --- Agente ---

def agent_combinator(model_id: str, temperature: float):
    instructions = """You are an expert time series analyst. Your goal is to combine model predictions to achieve the best forecast.

    PROTOCOL - FOLLOW STRICTLY:
    1. CALL `calculate_metrics_tool` with the validation data.
    2. ANALYZE the metrics returned by the tool. Look for low MAPE/SMAPE/RMSE or higher POCID.
    3. SELECT the top performing models (e.g., top 3, top 4 or top n models).
    4. CALL `selective_combine_tool` using ONLY these selected models and the 'final_test_predictions' provided in the prompt.
    5. AFTER the tool returns the combination, output the final JSON.

    OUTPUT FORMAT:
    You must output a VALID JSON object at the very end:
    {
        "description": "Brief explanation of which models were selected and why (based on which metric).",
        "result": [list of float values returned by selective_combine_tool]
    }
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
    agent = agent_combinator(
        model_id="qwen2.5:7b", 
        temperature=0.0
    )
    
    print("=" * 80)
    print("AGENT EXECUTION")
    print("=" * 80)
    print(f"Model: {agent.model.id}")
    print("=" * 80 + "\n")
    
    prompt = f"""
    Here is the data for your analysis:

    1. Validation Actuals: {validation_test}
    
    2. Validation Predictions (use this with calculate_metrics_tool):
    {json.dumps(validation_predictions)}

    3. Final Test Predictions (use this with selective_combine_tool):
    {json.dumps(final_test_predictions)}

    Start by calculating the metrics. Do not generate the final JSON until you have called both tools.
    """
    
    print("Sending prompt to agent...")
    print("-" * 80)
    try:
        response = agent.run(prompt)
        print(f"\n[RAW RESPONSE]:\n{response.content}\n")
        
        output = clean_json_string(response.content)
        return output.get("description", ""), output.get("result", [])
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "Error in processing", []

if __name__ == "__main__":
    validation_test = [17656.623, 19507.078, 15680.762, 19775.546, 13736.136, 17221.028, 18352.012, 20327.377, 21175.424, 18516.637, 19864.665, 18523.176]
    validation_predictions = {"ARIMA": [20189.34]*12, "ETS": [18062.62]*12, "THETA": [17866.08]*12}
    test_series = [22026.58]*12
    predictions = {"ARIMA": [19197.09]*12, "ETS": [18251.15]*12, "THETA": [18345.69]*12}

    description, result = simple_agent(validation_test, validation_predictions, predictions)
    print(result)