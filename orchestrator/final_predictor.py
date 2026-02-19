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


def _keep_best_by_ratio(errors: np.ndarray, trim_ratio: float) -> np.ndarray:
    errs = np.asarray(errors, dtype=float)
    n = len(errs)
    if n == 0:
        return np.array([], dtype=int)
    ratio = float(trim_ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        return np.array([], dtype=int)
    if ratio >= 1.0:
        return np.arange(n, dtype=int)
    k = max(1, int(np.ceil(ratio * n)))
    order = np.argsort(errs)
    return order[:k]


def _weights_exp(errors: np.ndarray, eta: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    errs = np.asarray(errors, dtype=float)
    z = np.exp(-float(eta) * (errs + eps))
    s = np.nansum(z)
    if not np.isfinite(s) or s <= 0:
        return np.ones(len(errs), dtype=float) / max(1, len(errs))
    return z / s


def _weights_poly(errors: np.ndarray, power: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    errs = np.asarray(errors, dtype=float)
    z = (errs + eps) ** (-float(power))
    s = np.nansum(z)
    if not np.isfinite(s) or s <= 0:
        return np.ones(len(errs), dtype=float) / max(1, len(errs))
    return z / s


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

    if method == "exp_weighted_average_per_horizon":
        eta = float(candidate.params.get("eta", 1.0))
        trim_ratio = float(candidate.params.get("trim_ratio", 1.0))

        out = np.full(horizon, np.nan, dtype=float)
        weights_by_h: Dict[str, Dict[str, float]] = {}

        for h in range(horizon):
            rmse = np.sqrt(np.nanmean((y_preds[:, :, h] - y_true[:, h][:, None]) ** 2, axis=0))
            keep = _keep_best_by_ratio(rmse, trim_ratio)
            if keep.size == 0:
                out[h] = float(np.nanmean(final_matrix[:, h]))
                continue
            w_small = _weights_exp(rmse[keep], eta=eta)
            w = np.zeros(n_models, dtype=float)
            w[keep] = w_small
            out[h] = float(np.nansum(w * final_matrix[:, h]))
            weights_by_h[str(h)] = {model_names[j]: float(w[j]) for j in range(n_models)}

        debug.update({"eta": eta, "trim_ratio": trim_ratio, "weights_by_horizon": weights_by_h})
        return {"result": out.tolist(), "debug": debug}

    if method == "poly_weighted_average_per_horizon":
        power = float(candidate.params.get("power", 1.0))
        trim_ratio = float(candidate.params.get("trim_ratio", 1.0))

        out = np.full(horizon, np.nan, dtype=float)
        weights_by_h: Dict[str, Dict[str, float]] = {}

        for h in range(horizon):
            rmse = np.sqrt(np.nanmean((y_preds[:, :, h] - y_true[:, h][:, None]) ** 2, axis=0))
            keep = _keep_best_by_ratio(rmse, trim_ratio)
            if keep.size == 0:
                out[h] = float(np.nanmean(final_matrix[:, h]))
                continue
            w_small = _weights_poly(rmse[keep], power=power)
            w = np.zeros(n_models, dtype=float)
            w[keep] = w_small
            out[h] = float(np.nansum(w * final_matrix[:, h]))
            weights_by_h[str(h)] = {model_names[j]: float(w[j]) for j in range(n_models)}

        debug.update({"power": power, "trim_ratio": trim_ratio, "weights_by_horizon": weights_by_h})
        return {"result": out.tolist(), "debug": debug}

    if method == "ade_dynamic_error_per_horizon":
        beta = float(candidate.params.get("beta", 0.5))
        eta = float(candidate.params.get("eta", 1.0))
        trim_ratio = float(candidate.params.get("trim_ratio", 1.0))

        # EMA of absolute errors across validation windows
        ema_errors = np.full((n_models, horizon), np.nan, dtype=float)
        for i in range(n_windows):
            err = np.abs(y_preds[i, :, :] - y_true[i, :][None, :])
            if np.any(np.isnan(ema_errors)):
                ema_errors = err.copy()
            else:
                ema_errors = beta * err + (1.0 - beta) * ema_errors

        out = np.full(horizon, np.nan, dtype=float)
        weights_by_h: Dict[str, Dict[str, float]] = {}

        for h in range(horizon):
            err_h = ema_errors[:, h]
            keep = _keep_best_by_ratio(err_h, trim_ratio)
            if keep.size == 0:
                out[h] = float(np.nanmean(final_matrix[:, h]))
                continue
            w_small = _weights_exp(err_h[keep], eta=eta)
            w = np.zeros(n_models, dtype=float)
            w[keep] = w_small
            out[h] = float(np.nansum(w * final_matrix[:, h]))
            weights_by_h[str(h)] = {model_names[j]: float(w[j]) for j in range(n_models)}

        debug.update({"beta": beta, "eta": eta, "trim_ratio": trim_ratio, "weights_by_horizon": weights_by_h})
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
