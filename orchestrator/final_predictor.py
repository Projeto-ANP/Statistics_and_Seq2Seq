from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from agent.context import get_context, set_context

from orchestrator.data_contract import load_validation_from_context
from orchestrator.schemas import CandidateStrategy
from orchestrator.strategies import RollingConfig


def _trimmed_mean_models(values: np.ndarray, trim_ratio: float) -> float:
    m = values.shape[0]
    if m == 0:
        return float("nan")

    k = int(np.floor(m * float(trim_ratio)))
    if k <= 0 or 2 * k >= m:
        return float(np.nanmean(values))

    vs = np.sort(values)
    return float(np.nanmean(vs[k : m - k]))


def _project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if np.all(v <= 0):
        return np.ones_like(v) / len(v)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones_like(v) / len(v)
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(v) / len(v)


def _weights_inverse_rmse(y_true_train: np.ndarray, y_preds_train: np.ndarray, eps: float, shrinkage: float) -> np.ndarray:
    errs = y_preds_train - y_true_train.reshape(-1, 1)
    rmse = np.sqrt(np.nanmean(errs**2, axis=0))
    inv = 1.0 / (rmse + eps)
    w = inv / np.nansum(inv)
    if shrinkage > 0:
        w = (1.0 - shrinkage) * w + shrinkage * (np.ones_like(w) / len(w))
    return _project_simplex(w)


def _ridge_weights(y_true_train: np.ndarray, y_preds_train: np.ndarray, l2: float) -> np.ndarray:
    X = np.asarray(y_preds_train, dtype=float)
    y = np.asarray(y_true_train, dtype=float)
    n_models = X.shape[1]
    if X.shape[0] == 0:
        return np.ones(n_models, dtype=float) / n_models

    A = X.T @ X + float(l2) * np.eye(n_models)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    return _project_simplex(w)


def predict_final_from_context(
    candidate: CandidateStrategy,
    rolling_cfg: RollingConfig | None = None,
) -> Dict[str, Any]:
    """Apply the chosen strategy to final (unseen) predictions.

    Reads:
      - validation windows from `context['all_validations']`
      - final predictions from `context['predictions']` (dict model -> list)

    Returns:
      {"result": [...], "debug": {...}}
    """

    rolling_cfg = rolling_cfg or RollingConfig(mode="expanding", train_window=5)

    data = load_validation_from_context()
    final_preds = get_context("predictions")
    if not isinstance(final_preds, dict) or not final_preds:
        raise RuntimeError("context['predictions'] not found. generate_all_validations_context() should set it.")

    model_names = data.model_names
    horizon = data.horizon

    # Build final matrix aligned to model_names
    final_matrix = np.full((len(model_names), horizon), np.nan, dtype=float)
    for j, m in enumerate(model_names):
        if m not in final_preds:
            continue
        arr = np.asarray(final_preds[m], dtype=float)
        final_matrix[j, : min(horizon, len(arr))] = arr[:horizon]

    method = str(candidate.params.get("method", "mean")).strip().lower()

    debug: Dict[str, Any] = {"method": method}

    if method == "mean":
        out = np.nanmean(final_matrix, axis=0)
        # Store weights for all models (uniform across horizons)
        uniform_weight = 1.0 / len(model_names)
        weights_by_h = {str(h): {m: uniform_weight for m in model_names} for h in range(horizon)}
        debug["weights_by_horizon"] = weights_by_h
        return {"result": out.tolist(), "debug": debug}

    if method == "median":
        out = np.nanmedian(final_matrix, axis=0)
        # Store weights for all models (uniform across horizons)
        uniform_weight = 1.0 / len(model_names)
        weights_by_h = {str(h): {m: uniform_weight for m in model_names} for h in range(horizon)}
        debug["weights_by_horizon"] = weights_by_h
        return {"result": out.tolist(), "debug": debug}

    if method == "trimmed_mean":
        trim_ratio = float(candidate.params.get("trim_ratio", 0.2))
        out = np.array([_trimmed_mean_models(final_matrix[:, h], trim_ratio) for h in range(horizon)], dtype=float)
        debug["trim_ratio"] = trim_ratio
        # Store weights for all models (uniform across horizons for trimmed_mean)
        uniform_weight = 1.0 / len(model_names)
        weights_by_h = {str(h): {m: uniform_weight for m in model_names} for h in range(horizon)}
        debug["weights_by_horizon"] = weights_by_h
        return {"result": out.tolist(), "debug": debug}

    # Learners trained on ALL validation windows (no leakage; final test has no y_true)
    y_true = data.y_true
    y_preds = data.y_preds
    n_windows, n_models, _ = y_preds.shape

    if method == "best_single":
        rmse_per_model = np.sqrt(np.nanmean((y_preds - y_true[:, None, :]) ** 2, axis=(0, 2)))
        best_idx = int(np.nanargmin(rmse_per_model))
        best_model = model_names[best_idx]
        debug["chosen_model"] = best_model
        # Store weights: 100% to chosen model, 0% to others (uniform across horizons)
        weights_by_h = {str(h): {m: (1.0 if m == best_model else 0.0) for m in model_names} for h in range(horizon)}
        debug["weights_by_horizon"] = weights_by_h
        return {"result": final_matrix[best_idx, :].tolist(), "debug": debug}

    if method == "best_per_horizon":
        chosen = []
        out = np.full(horizon, np.nan, dtype=float)
        weights_by_h: Dict[str, Dict[str, float]] = {}
        for h in range(horizon):
            rmse_per_model = np.sqrt(np.nanmean((y_preds[:, :, h] - y_true[:, h][:, None]) ** 2, axis=0))
            best_idx = int(np.nanargmin(rmse_per_model))
            best_model = model_names[best_idx]
            out[h] = final_matrix[best_idx, h]
            chosen.append(best_model)
            # Store weights: 100% to chosen model per horizon
            weights_by_h[str(h)] = {m: (1.0 if m == best_model else 0.0) for m in model_names}
        debug["chosen_model_by_horizon"] = chosen
        debug["weights_by_horizon"] = weights_by_h
        return {"result": out.tolist(), "debug": debug}

    if method == "topk_mean_per_horizon":
        top_k = int(candidate.params.get("top_k", 3))
        top_k = max(1, min(top_k, n_models))
        out = np.full(horizon, np.nan, dtype=float)
        chosen = []
        weights_by_h: Dict[str, Dict[str, float]] = {}
        for h in range(horizon):
            rmse_per_model = np.sqrt(np.nanmean((y_preds[:, :, h] - y_true[:, h][:, None]) ** 2, axis=0))
            order = np.argsort(rmse_per_model)
            idxs = order[:top_k]
            out[h] = float(np.nanmean(final_matrix[idxs, h]))
            chosen_names = [model_names[i] for i in idxs.tolist()]
            chosen.append(chosen_names)
            # Store weights: uniform across chosen models
            uniform_weight = 1.0 / len(idxs)
            weights_by_h[str(h)] = {m: (uniform_weight if m in chosen_names else 0.0) for m in model_names}
        debug.update({"top_k": top_k, "chosen_models_by_horizon": chosen, "weights_by_horizon": weights_by_h})
        return {"result": out.tolist(), "debug": debug}

    if method == "inverse_rmse_weights_per_horizon":
        top_k = int(candidate.params.get("top_k", n_models))
        top_k = max(1, min(top_k, n_models))
        shrinkage = float(candidate.params.get("shrinkage", 0.0))
        eps = float(candidate.params.get("eps", 1e-8))

        out = np.full(horizon, np.nan, dtype=float)
        weights_by_h: Dict[str, Dict[str, float]] = {}

        for h in range(horizon):
            rmse = np.sqrt(np.nanmean((y_preds[:, :, h] - y_true[:, h][:, None]) ** 2, axis=0))
            order = np.argsort(rmse)
            keep = order[:top_k]
            w_small = _weights_inverse_rmse(y_true[:, h], y_preds[:, keep, h], eps=eps, shrinkage=shrinkage)
            w = np.zeros(n_models, dtype=float)
            w[keep] = w_small
            out[h] = float(np.nansum(w * final_matrix[:, h]))
            weights_by_h[str(h)] = {model_names[j]: float(w[j]) for j in range(n_models)}

        debug.update({"top_k": top_k, "shrinkage": shrinkage, "weights_by_horizon": weights_by_h})
        return {"result": out.tolist(), "debug": debug}

    if method == "ridge_stacking_per_horizon":
        l2 = float(candidate.params.get("l2", 10.0))
        top_k = int(candidate.params.get("top_k", n_models))
        top_k = max(1, min(top_k, n_models))

        out = np.full(horizon, np.nan, dtype=float)
        weights_by_h: Dict[str, Dict[str, float]] = {}

        for h in range(horizon):
            rmse = np.sqrt(np.nanmean((y_preds[:, :, h] - y_true[:, h][:, None]) ** 2, axis=0))
            order = np.argsort(rmse)
            keep = order[:top_k]
            w_small = _ridge_weights(y_true[:, h], y_preds[:, keep, h], l2=l2)
            w = np.zeros(n_models, dtype=float)
            w[keep] = w_small
            out[h] = float(np.nansum(w * final_matrix[:, h]))
            weights_by_h[str(h)] = {model_names[j]: float(w[j]) for j in range(n_models)}

        debug.update({"l2": l2, "top_k": top_k, "weights_by_horizon": weights_by_h})
        return {"result": out.tolist(), "debug": debug}

    # fallback
    out = np.nanmean(final_matrix, axis=0)
    debug["fallback"] = True
    return {"result": out.tolist(), "debug": debug}
