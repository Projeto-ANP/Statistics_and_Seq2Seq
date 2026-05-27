from typing import List, Any, Optional
import os
import re
import pandas as pd
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
        "series_history": [],
        "models_available": [],
        "point_parameter": None
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

def read_model_preds(model_name, dataset_index, dataset="ANP_MONTHLY"):
    df = pd.read_csv(
        f"./timeseries/mestrado/resultados/{model_name}/normal/{dataset}.csv",
        sep=";",
    )
    df = df[df["dataset_index"] == dataset_index]

    df["start_test"] = pd.to_datetime(df["start_test"], errors="coerce")
    df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
    df = df.sort_values(by="start_test")

    return df

def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []

def resolve_tsf_path(tsf_path: Optional[str], dataset: Optional[str], base_dir: str = "../forecasting_datasets") -> Optional[str]:
    """Resolve the original .tsf file path.

    Priority: (1) an explicit `tsf_path` (preferred — pass it at execution time to avoid any
    name-mapping issues); if the literal path is missing, retry under base_dir by basename.
    (2) Derive from the dataset name as `{dataset.lower()}.tsf`, resolved case-insensitively
    so that e.g. dataset "ETTH1" matches the on-disk "ETTH1.tsf" (the .tsf source files do not
    follow the uppercased convention used by the model-result folders). Returns None if nothing
    matches.
    """
    if tsf_path:
        if os.path.exists(tsf_path):
            return tsf_path
        cand = os.path.join(base_dir, os.path.basename(tsf_path))
        return cand if os.path.exists(cand) else None
    if not dataset:
        return None
    target = f"{dataset.lower()}.tsf"
    direct = os.path.join(base_dir, target)
    if os.path.exists(direct):
        return direct
    try:
        for fname in os.listdir(base_dir):
            if fname.lower() == target:
                return os.path.join(base_dir, fname)
    except Exception:
        pass
    return None


def load_original_series_history(dataset_index, horizon: int, tsf_path: Optional[str] = None, dataset: Optional[str] = None, base_dir: str = "../forecasting_datasets") -> Optional[List[float]]:
    """Leakage-safe historical series for one series, read from its original .tsf.

    Returns series_value[:-horizon] — everything strictly before the final test window,
    i.e. exactly the training span the base models used to forecast the final test, so it
    never leaks the test target. Values are the raw real scale (same scale as the
    `test`/`predictions` CSV columns), so per-model transforms (normalization, STL, …) are
    irrelevant. dataset_index aligns row-for-row with df.iloc[i] in the .tsf. Pass `tsf_path`
    explicitly to be unambiguous; otherwise it is resolved from `dataset`. Returns None if the
    file/series can't be loaded (caller falls back to the validation-window proxy).
    """
    try:
        from streamfuels.datasets import DatasetLoader  # lazy: optional at import time
    except Exception:
        return None
    path = resolve_tsf_path(tsf_path, dataset, base_dir)
    if not path:
        return None
    try:
        loader = DatasetLoader()
        df_tsf, _ = loader.read_tsf(path_tsf=path)
        idx = int(dataset_index)
        if idx < 0 or idx >= len(df_tsf):
            return None
        series_value = list(df_tsf.iloc[idx]["series_value"])
        if not horizon or len(series_value) <= horizon:
            return None
        return [float(x) for x in series_value[:-horizon]]
    except Exception:
        return None

def generate_all_validations_context(models: List[str], dataset_index, train_window: int, dataset="ANP_MONTHLY", tsf_path: Optional[str] = None) -> None:
    """Generate validation context from model predictions."""
    # Track dataset identifier for downstream logging
    set_context("dataset_index", dataset_index)
    set_context("dataset_name", dataset)
    # Garantir que o contexto está inicializado
    if not get_context("all_validations"):
        init_context()
    
    # Resetar all_validations
    set_context("all_validations", {
        "predictions": [],
        "test": []
    })
    
    # Data layout (confirmed): each dataset_index has exactly (train_window + 1) rows.
    # The last row (iloc[-1]) is the REAL test window we combine for; the `train_window`
    # rows before it are validation windows. Using iloc[-(train_window+1):-1] keeps all
    # `train_window` validation windows (the previous iloc[-train_window:-1] dropped one).
    sample_model = models[0]
    df_sample = read_model_preds(sample_model, dataset_index, dataset=dataset)
    df_filtred_sample = df_sample.iloc[-(train_window + 1):-1]
    n_windows = len(df_filtred_sample)

    all_validations = get_context("all_validations")
    for _ in range(n_windows):
        all_validations["predictions"].append({})

    test_extracted = False
    final_test_predictions = {}

    for model in models:
        df_model = read_model_preds(model, dataset_index, dataset=dataset)
        df_filtred = df_model.iloc[-(train_window + 1):-1]
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

    # Recent observed history (leakage-safe), used by the SeriesAnalyst for feature extraction.
    # Preferred: the full original series truncated at -horizon (the entire clean training span
    # before the final test window), loaded from the .tsf. Fallback: concatenate the validation
    # `test` windows (the ground truth is identical across models and the windows are contiguous
    # & non-overlapping → a contiguous recent segment of train_window * horizon points). Either
    # way the final test window is excluded.
    proxy_history: List[float] = []
    for win_test in all_validations.get("test", []):
        if isinstance(win_test, list):
            proxy_history.extend(float(x) for x in win_test)

    horizon = len(all_validations["test"][0]) if all_validations.get("test") else 0
    full_history = load_original_series_history(dataset_index, horizon, tsf_path=tsf_path, dataset=dataset)
    if full_history:
        set_context("series_history", full_history)
        set_context("series_history_source", "tsf_original")
    else:
        set_context("series_history", proxy_history)
        set_context("series_history_source", "validation_proxy")