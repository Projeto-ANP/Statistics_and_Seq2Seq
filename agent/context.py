import pandas as pd
from typing import List, Any, Optional
import re

CONTEXT_MEMORY = {}

def init_context():
    """Initialize the context memory with default structure."""
    global CONTEXT_MEMORY
    CONTEXT_MEMORY = {
        "predictions": None,
        "tools_called": [],
        "point_model_selection": {},
        "point_model_weights": {},
        "all_validations": {
            "predictions": [],
            "test": []
        },
        "models_available": []
    }

def get_context(key: str, default: Any = None) -> Any:
    """
    Safely get a value from CONTEXT_MEMORY.
    
    Args:
        key: The key to retrieve
        default: Default value if key doesn't exist
        
    Returns:
        The value associated with the key, or default if not found
    """
    return CONTEXT_MEMORY.get(key, default)

def set_context(key: str, value: Any) -> None:
    """
    Set a value in CONTEXT_MEMORY.
    
    Args:
        key: The key to set
        value: The value to store
    """
    CONTEXT_MEMORY[key] = value

def update_context(key: str, value: Any) -> None:
    """
    Update a nested value in CONTEXT_MEMORY.
    Useful for updating dictionaries without replacing them entirely.
    
    Args:
        key: The key to update
        value: The value to merge/update
    """
    if key in CONTEXT_MEMORY and isinstance(CONTEXT_MEMORY[key], dict) and isinstance(value, dict):
        CONTEXT_MEMORY[key].update(value)
    else:
        CONTEXT_MEMORY[key] = value

def get_all_context() -> dict:
    """Get the entire CONTEXT_MEMORY."""
    return CONTEXT_MEMORY.copy()

def clear_context() -> None:
    """Clear all context memory."""
    global CONTEXT_MEMORY
    CONTEXT_MEMORY = {}

def read_model_preds(model_name, dataset_index):
    df = pd.read_csv(
        f"../timeseries/mestrado/resultados/{model_name}/normal/ANP_MONTHLY.csv",
        sep=";",
    )
    df = df[df["dataset_index"] == dataset_index]

    df["start_test"] = pd.to_datetime(df["start_test"], format="%Y-%m-%d")
    df["final_test"] = pd.to_datetime(df["final_test"], format="%Y-%m-%d")
    df = df.sort_values(by="start_test")

    return df

def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []

def generate_all_validations_context(models: List[str], dataset_index) -> None:
    """Generate validation context from model predictions."""
    # Garantir que o contexto está inicializado
    if not get_context("all_validations"):
        init_context()
    
    # Resetar all_validations
    set_context("all_validations", {
        "predictions": [],
        "test": []
    })
    
    sample_model = models[0]
    df_sample = read_model_preds(sample_model, dataset_index)
    df_filtred_sample = df_sample.iloc[-4:-1]
    n_windows = len(df_filtred_sample)

    all_validations = get_context("all_validations")
    for _ in range(n_windows):
        all_validations["predictions"].append({})

    test_extracted = False  
    final_test_predictions = {}
    
    for model in models:  
        df_model = read_model_preds(model, 0)
        df_filtred = df_model.iloc[-4:-1]
        df_final_test = df_model.iloc[-1]
        predictions_final = extract_values(df_final_test["predictions"])
        final_test_predictions[model] = predictions_final
        
        for window_idx, (_, row) in enumerate(df_filtred.iterrows()):
            preds = extract_values(row["predictions"])
            all_validations["predictions"][window_idx][model] = preds
            
            if not test_extracted: 
                test = extract_values(row["test"])
                all_validations["test"].append(test)
        
        if not test_extracted: 
            test_extracted = True
    
    # Atualizar o contexto
    set_context("all_validations", all_validations)
    set_context("predictions", final_test_predictions)