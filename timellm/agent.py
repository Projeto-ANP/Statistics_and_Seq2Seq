from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama
import numpy as np
import pandas as pd
import json
import sys
import os

# Adiciona o diretório pai ao path para importar all_functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import all_functions

from typing import List
from pydantic import BaseModel, Field
from combination_functions import detect_seasonality, selective_combination, full_combination

LOCAL_MODEL_ID = "qwen3:8b"

# ============================================
# RESPONSE MODEL - Estrutura da resposta
# ============================================

class CombinationResult(BaseModel):
    """Modelo estruturado para a resposta do agente"""
    description: str = Field(
        ..., 
        description="Análise detalhada: observações da série, resultados das análises, raciocínio sobre a decisão tomada"
    )
    result: List[float] = Field(
        ..., 
        description="Lista com as predições combinadas resultantes"
    )
    method_used: str = Field(
        ..., 
        description="Método de combinação utilizado (selective_combination ou full_combination)"
    )
    has_seasonality: bool = Field(
        ..., 
        description="Se a série apresenta sazonalidade"
    )

# ============================================
# INTERNAL FUNCTIONS (without @tool decorator for direct calling)
# ============================================

def _calculate_metrics_internal(test: list, all_predictions: dict) -> str:
    """Internal version of calculate_metrics_tool for direct execution"""
    print(f"\n[TOOL CALL] calculate_metrics_tool called")
    
    results = {}
    y_true = np.array(test)
    
    for model_name, y_pred_list in all_predictions.items():
        y_pred = np.array(y_pred_list)
        
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            continue
            
        curr_y_true = y_true[:min_len]
        curr_y_pred = y_pred[:min_len]
        
        mask = curr_y_true != 0
        mape = np.mean(np.abs((curr_y_true[mask] - curr_y_pred[mask]) / curr_y_true[mask])) * 100 if np.any(mask) else 0.0
        
        rmse = all_functions.calculate_rmse(np.array(curr_y_pred).reshape(1, -1), np.array(curr_y_true).reshape(1, -1))
        print(rmse)
        smape = all_functions.calculate_smape(np.array(curr_y_pred).reshape(1, -1), np.array(curr_y_true).reshape(1, -1))
        print(smape)
        pocid = all_functions.pocid(curr_y_true, curr_y_pred)

        results[model_name] = {
            "MAPE": float(mape),
            "RMSE": float(rmse),
            "SMAPE": float(smape),
            "POCID": float(pocid)
        }
        
    print(f"[TOOL RESULT] Calculated metrics for {len(results)} models")
    print(results)
    return json.dumps(results, indent=2)

def _selective_combine_internal(predictions: dict, metrics: dict, metric_to_optimize: str = 'MAPE', top_k: int = 3) -> dict:
    """Internal version of selective_combine_tool for direct execution"""
    print(f"\n[TOOL CALL] selective_combine_tool called. Optimizing {metric_to_optimize} with top_k={top_k}")
    
    if top_k < 2 and len(predictions) >= 2:
        print(f"[TOOL INFO] top_k={top_k} is too low for combination. Increasing to 2.")
        top_k = 2
    
    reverse = True if metric_to_optimize == 'POCID' else False
    valid_metrics = {k: v for k, v in metrics.items() if k in predictions}
    
    sorted_models = sorted(
        valid_metrics.items(),
        key=lambda x: x[1].get(metric_to_optimize, float('inf') if not reverse else float('-inf')),
        reverse=reverse
    )
    
    selected_models = [m[0] for m in sorted_models[:top_k]]
    
    if not selected_models:
        selected_models = list(predictions.keys())
        print(f"[TOOL WARNING] No models selected based on metrics. Using all models.")
    
    print(f"[TOOL INFO] Selected top {len(selected_models)} models based on {metric_to_optimize}: {selected_models}")
    
    pred_arrays = {k: np.array(v) for k, v in predictions.items() if k in selected_models}
    combined = full_combination(pred_arrays)
    print(f"[TOOL RESULT] Combined {len(combined)} predictions (Mean of selected models)")
    
    return {
        "combined_predictions": combined.tolist(),
        "models_used": selected_models,
        "method": f"selective_combination_by_{metric_to_optimize} (Mean of top {len(selected_models)})"
    }

@tool
def calculate_metrics_tool(test: list, all_predictions: dict) -> dict:
    """
    Calculates metrics (MAPE, RMSE, POCID, SMAPE) for each model's predictions against the test series.
    
    Args:
        test: list of actual values (ground truth)
        all_predictions: dict with model names as keys and list of predictions as values
    
    Returns:
        dict with model names as keys and their calculated metrics as values
    """
    print(f"\n[TOOL CALL] calculate_metrics_tool called")
    
    results = {}
    y_true = np.array(test)
    
    for model_name, y_pred_list in all_predictions.items():
        y_pred = np.array(y_pred_list)
        
        # Ensure lengths match
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            continue
            
        curr_y_true = y_true[:min_len]
        curr_y_pred = y_pred[:min_len]
        
        # MAPE
        mask = curr_y_true != 0
        mape = np.mean(np.abs((curr_y_true[mask] - curr_y_pred[mask]) / curr_y_true[mask])) * 100 if np.any(mask) else 0.0
        
        # RMSE
        rmse = all_functions.calculate_rmse(curr_y_pred, curr_y_true)
        
        # SMAPE
        smape = all_functions.calculate_smape(curr_y_pred, curr_y_true)

        pocid = all_functions.pocid(curr_y_true,curr_y_pred)

        results[model_name] = {
            "MAPE": float(mape),
            "RMSE": float(rmse),
            "SMAPE": float(smape),
            "POCID": float(pocid)
        }
        
    print(f"[TOOL RESULT] Calculated metrics for {len(results)} models")
    print(results)
    return json.dumps(results, indent=2)

@tool
def selective_combine_tool(predictions: dict, metrics: dict, metric_to_optimize: str = 'MAPE', top_k: int = 3) -> dict:
    """
    Combines predictions using the best models based on a specific metric.
    
    Args:
        predictions: dict with model names as keys and list of predictions as values
        metrics: dict with model names as keys and their metrics as values (output of calculate_metrics_tool)
        metric_to_optimize: metric to use for selection ('MAPE', 'RMSE', 'SMAPE', 'POCID')
        top_k: number of top models to combine (default: 3). Should be >= 2 for a real combination.
    
    Returns:
        dict with 'combined_predictions' (list) and 'models_used' (list)
    """
    print(f"\n[TOOL CALL] selective_combine_tool called. Optimizing {metric_to_optimize} with top_k={top_k}")
    
    # Ensure we combine at least 2 models if possible
    if top_k < 2 and len(predictions) >= 2:
        print(f"[TOOL INFO] top_k={top_k} is too low for combination. Increasing to 2.")
        top_k = 2
    
    # Sort models by metric
    # For POCID, higher is better. For others (MAPE, RMSE, SMAPE), lower is better.
    reverse = True if metric_to_optimize == 'POCID' else False
    
    # Filter metrics to only include models present in predictions
    valid_metrics = {k: v for k, v in metrics.items() if k in predictions}
    
    sorted_models = sorted(
        valid_metrics.items(),
        key=lambda x: x[1].get(metric_to_optimize, float('inf') if not reverse else float('-inf')),
        reverse=reverse
    )
    
    # Select top k models
    selected_models = [m[0] for m in sorted_models[:top_k]]
    
    if not selected_models:
        selected_models = list(predictions.keys())
        print(f"[TOOL WARNING] No models selected based on metrics. Using all models.")
    
    print(f"[TOOL INFO] Selected top {len(selected_models)} models based on {metric_to_optimize}: {selected_models}")
    
    # Prepare predictions for selected models
    pred_arrays = {k: np.array(v) for k, v in predictions.items() if k in selected_models}
    
    # Use full_combination on the subset (calculates mean)
    combined = full_combination(pred_arrays)
    print(f"[TOOL RESULT] Combined {len(combined)} predictions (Mean of selected models)")
    
    return {
        "combined_predictions": combined.tolist(),
        "models_used": selected_models,
        "method": f"selective_combination_by_{metric_to_optimize} (Mean of top {len(selected_models)})"
    }

# ============================================
# AGENT CONFIGURATION
# ============================================

def create_combination_agent(model_id=None, temperature=0.3, reasoning_mode=True):
    """
    Creates an Agno agent for prediction combination using LOCAL LLM.
    
    Args:
        model_id: Ollama model ID (e.g., "llama3.1:7b", "qwen3:8b")
        temperature: 0.0-1.0, higher = more creative thinking
        reasoning_mode: If True, allows LLM to think and reason about solutions
    
    Returns:
        Agent configured with local LLM
    """
    if model_id is None:
        model_id = LOCAL_MODEL_ID
    
    # Instruções para modo de raciocínio
    if reasoning_mode:
        instructions = """You are an expert time series analyst with tool-calling capabilities.

Your workflow:
1. First, call calculate_metrics_tool to get performance metrics for all models
2. Analyze which models perform best and which metric is most reliable
3. Call selective_combine_tool to combine the best models
4. Return the combined predictions

You have these tools available:
- calculate_metrics_tool(test, all_predictions): Returns metrics for each model
- selective_combine_tool(predictions, metrics, metric_to_optimize, top_k): Combines best models

Always call the tools in order. Choose the metric wisely based on the data.
"""
    else:
        # Modo mais direto (menos tokens)
        instructions = """You are a time series prediction combination expert.

EXECUTE THESE STEPS:
1. CALL calculate_metrics(test, all_predictions)
2. CALL selective_combine_tool based on result
3. Return: {"description": "brief analysis", "result": [predictions], "method_used": "selective_combination"}

IMPORTANT: Actually CALL the tools, don't just describe them.
"""
    
    return Agent(
        model=Ollama(id=model_id, options={"temperature": 0.2}),
        tools=[calculate_metrics_tool, selective_combine_tool],
        instructions=instructions,
        markdown=False,
        show_tool_calls=True,
    )

# ============================================
# MAIN FUNCTION
# ============================================

def run_combination_agent(
    predictions, 
    time_series, 
    period=7, 
    model_id=None,
    temperature=0.3,
    reasoning_mode=True,
    verbose=True
):
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running Combination Agent with LOCAL LLM")
        print(f"{'='*80}")
        print(f"Model: {model_id or LOCAL_MODEL_ID}")
        print(f"Time series length: {len(time_series)}")
        print(f"Available models: {list(predictions.keys())}")
        print(f"{'='*80}\n")
    
    agent = create_combination_agent(
        model_id=model_id,
        temperature=temperature,
        reasoning_mode=reasoning_mode
    )
    
    if reasoning_mode:
        message_str = f"""Task: Combine time series predictions using the best-performing models.

Test data (ground truth): {time_series}

Model predictions: {json.dumps(predictions, indent=2)}

Instructions:
1. Calculate metrics for all models
2. Analyze which metric is most appropriate (MAPE, RMSE, SMAPE, or POCID)
3. Select and combine the top 3 models based on that metric
4. Return the combined predictions"""
    else:
        message_str = f"""Analyze and combine these predictions:

Test Series (Ground Truth): {time_series}
Predictions: {json.dumps(predictions)}

Execute analysis and return combined predictions."""
    
    # Run agent
    try:
        if verbose:
            print("Running agent...\n")
        
        response = agent.run(message_str)
        
        # Se response_model foi usado, content já é estruturado
        if hasattr(response, 'content'):
            if isinstance(response.content, CombinationResult):
                # Resposta estruturada do Pydantic
                result = {
                    "description": response.content.description,
                    "result": response.content.result,
                    "method_used": response.content.method_used,
                    "has_seasonality": response.content.has_seasonality
                }
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"� RESULTADO FINAL:")
                    print(f"{'='*80}")
                    print(f"\n�💭 ANÁLISE:")
                    print(f"{result['description']}\n")
                    print(f"📈 PREDIÇÕES COMBINADAS: {result['result']}")
                    print(f"🔧 MÉTODO USADO: {result['method_used']}")
                    print(f"📊 SAZONALIDADE: {'Sim' if result['has_seasonality'] else 'Não'}")
                    print(f"{'='*80}\n")
                
                return result
            else:
                # Resposta em texto - tentar parsear
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"💭 AGENT RESPONSE:")
                    print(f"{'='*80}")
                    print(response.content)
                    print(f"{'='*80}\n")
                
                # Try to extract structured data from response
                try:
                    import re
                    # Procura por JSON na resposta
                    json_pattern = r'\{[^{}]*"(description|result)"[^{}]*\}'
                    json_match = re.search(json_pattern, response.content, re.DOTALL)
                    
                    if json_match:
                        result = json.loads(json_match.group())
                        return result
                    else:
                        return {
                            "response": response.content,
                            "raw": True
                        }
                except Exception as parse_error:
                    if verbose:
                        print(f"⚠️  Could not parse structured response: {parse_error}")
                    return {
                        "response": response.content,
                        "raw": True
                    }
        else:
            return {
                "response": str(response),
                "raw": True
            }
            
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        raise