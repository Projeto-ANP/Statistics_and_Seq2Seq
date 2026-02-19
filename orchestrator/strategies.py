from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from orchestrator.schemas import CandidateStrategy


@dataclass
class RollingConfig:
    mode: Literal["expanding", "rolling"] = "expanding"
    train_window: int = 5  # used only if mode == "rolling"


def _train_slice(i: int, cfg: RollingConfig) -> slice:
    if i <= 0:
        return slice(0, 0)
    if cfg.mode == "expanding":
        return slice(0, i)
    # rolling
    start = max(0, i - max(1, int(cfg.train_window)))
    return slice(start, i)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Projection onto the probability simplex {w>=0, sum(w)=1}."""

    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("simplex projection expects 1d vector")

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


def _trimmed_mean(x: np.ndarray, trim_ratio: float) -> np.ndarray:
    """Trimmed mean along axis=0 (models axis)."""

    x = np.asarray(x, dtype=float)
    m = x.shape[0]
    if m == 0:
        return np.full(x.shape[1:], np.nan)

    k = int(np.floor(m * float(trim_ratio)))
    if k <= 0 or 2 * k >= m:
        return np.nanmean(x, axis=0)

    xs = np.sort(x, axis=0)
    return np.nanmean(xs[k : m - k], axis=0)


def _uniform_weights(n_models: int) -> np.ndarray:
    return np.ones(n_models, dtype=float) / float(n_models)


def _keep_best_by_ratio(errors: np.ndarray, trim_ratio: float) -> np.ndarray:
    """Return indices of best models by error, keeping trim_ratio fraction."""

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
        return _uniform_weights(len(errs))
    return z / s


def _weights_poly(errors: np.ndarray, power: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    errs = np.asarray(errors, dtype=float)
    z = (errs + eps) ** (-float(power))
    s = np.nansum(z)
    if not np.isfinite(s) or s <= 0:
        return _uniform_weights(len(errs))
    return z / s


def _weights_inverse_rmse(
    y_true_train: np.ndarray,
    y_preds_train: np.ndarray,
    eps: float = 1e-8,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Compute weights proportional to inverse RMSE per model.

    Args:
        y_true_train: shape (n_train,)
        y_preds_train: shape (n_train, n_models)
    """

    errs = y_preds_train - y_true_train.reshape(-1, 1)
    rmse = np.sqrt(np.nanmean(errs**2, axis=0))
    inv = 1.0 / (rmse + eps)
    w = inv / np.nansum(inv)
    if shrinkage > 0:
        w = (1.0 - shrinkage) * w + shrinkage * _uniform_weights(len(w))
    return _project_simplex(w)


def _ridge_weights(
    y_true_train: np.ndarray,
    y_preds_train: np.ndarray,
    l2: float,
    nonneg_simplex: bool = True,
) -> np.ndarray:
    """Closed-form ridge weights, optionally projected to simplex."""

    X = np.asarray(y_preds_train, dtype=float)
    y = np.asarray(y_true_train, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D")

    n_models = X.shape[1]
    if X.shape[0] == 0:
        return _uniform_weights(n_models)

    XtX = X.T @ X
    A = XtX + float(l2) * np.eye(n_models)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b

    if nonneg_simplex:
        return _project_simplex(w)

    s = np.sum(w)
    if s != 0:
        w = w / s
    return w


def generate_combined_predictions(
    data_y_true: np.ndarray,
    data_y_preds: np.ndarray,
    model_names: List[str],
    candidate: CandidateStrategy,
    rolling_cfg: RollingConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Generate combined predictions for all windows (deterministic).

    Args:
        data_y_true: (n_windows, horizon)
        data_y_preds: (n_windows, n_models, horizon)

    Returns:
        (combined_preds, debug_info)
            combined_preds: (n_windows, horizon)
    """

    y_true = np.asarray(data_y_true, dtype=float)
    y_preds = np.asarray(data_y_preds, dtype=float)

    n_windows, n_models, horizon = y_preds.shape

    method = str(candidate.params.get("method", "mean")).strip().lower()
    debug: Dict[str, object] = {
        "method": method,
        "rolling": {"mode": rolling_cfg.mode, "train_window": rolling_cfg.train_window},
    }

    combined = np.full((n_windows, horizon), np.nan, dtype=float)

    if method == "mean":
        combined[:, :] = np.nanmean(y_preds, axis=1)
        return combined, debug

    if method == "median":
        combined[:, :] = np.nanmedian(y_preds, axis=1)
        return combined, debug

    if method == "trimmed_mean":
        trim_ratio = float(candidate.params.get("trim_ratio", 0.2))
        for i in range(n_windows):
            combined[i, :] = _trimmed_mean(y_preds[i, :, :], trim_ratio)
        debug["trim_ratio"] = trim_ratio
        return combined, debug

    # Rolling / expanding learners (anti-leakage)
    if method in {"best_single", "best_per_horizon", "topk_mean_per_horizon", "inverse_rmse_weights_per_horizon", "ridge_stacking_per_horizon"}:
        metric = str(candidate.params.get("selection_metric", "rmse")).strip().lower()
        if metric != "rmse":
            metric = "rmse"  # keep deterministic/simple for now
        debug["selection_metric"] = metric

    if method == "best_single":
        chosen: List[str] = []
        for i in range(n_windows):
            tr = _train_slice(i, rolling_cfg)
            if tr.stop - tr.start <= 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                chosen.append("mean_fallback")
                continue

            # pick by aggregate RMSE over horizons
            train_true = y_true[tr, :]
            train_preds = y_preds[tr, :, :]
            rmse_per_model = np.sqrt(np.nanmean((train_preds - train_true[:, None, :]) ** 2, axis=(0, 2)))
            best_idx = int(np.nanargmin(rmse_per_model))
            combined[i, :] = y_preds[i, best_idx, :]
            chosen.append(model_names[best_idx])

        debug["chosen_model_by_window"] = chosen
        return combined, debug

    if method == "best_per_horizon":
        chosen_by_h = [[] for _ in range(horizon)]
        for i in range(n_windows):
            tr = _train_slice(i, rolling_cfg)
            if tr.stop - tr.start <= 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                for h in range(horizon):
                    chosen_by_h[h].append("mean_fallback")
                continue

            train_true = y_true[tr, :]
            train_preds = y_preds[tr, :, :]
            for h in range(horizon):
                rmse_per_model = np.sqrt(np.nanmean((train_preds[:, :, h] - train_true[:, h][:, None]) ** 2, axis=0))
                best_idx = int(np.nanargmin(rmse_per_model))
                combined[i, h] = y_preds[i, best_idx, h]
                chosen_by_h[h].append(model_names[best_idx])

        debug["chosen_model_by_window_horizon"] = chosen_by_h
        return combined, debug

    if method == "topk_mean_per_horizon":
        top_k = int(candidate.params.get("top_k", 3))
        top_k = max(1, min(top_k, n_models))
        debug["top_k"] = top_k

        for i in range(n_windows):
            tr = _train_slice(i, rolling_cfg)
            if tr.stop - tr.start <= 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                continue

            train_true = y_true[tr, :]
            train_preds = y_preds[tr, :, :]
            for h in range(horizon):
                rmse_per_model = np.sqrt(np.nanmean((train_preds[:, :, h] - train_true[:, h][:, None]) ** 2, axis=0))
                order = np.argsort(rmse_per_model)
                idxs = order[:top_k]
                combined[i, h] = np.nanmean(y_preds[i, idxs, h])

        return combined, debug

    if method == "inverse_rmse_weights_per_horizon":
        top_k = int(candidate.params.get("top_k", n_models))
        top_k = max(1, min(top_k, n_models))
        shrinkage = float(candidate.params.get("shrinkage", 0.0))
        eps = float(candidate.params.get("eps", 1e-8))
        debug.update({"top_k": top_k, "shrinkage": shrinkage, "eps": eps})

        weights_by_h: List[List[float]] = []

        for h in range(horizon):
            weights_by_h.append([float("nan")] * n_models)

        for i in range(n_windows):
            tr = _train_slice(i, rolling_cfg)
            if tr.stop - tr.start <= 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                continue

            for h in range(horizon):
                y_t = y_true[tr, h]
                X = y_preds[tr, :, h]
                rmse = np.sqrt(np.nanmean((X - y_t[:, None]) ** 2, axis=0))
                order = np.argsort(rmse)
                keep = order[:top_k]

                w_small = _weights_inverse_rmse(y_t, X[:, keep], eps=eps, shrinkage=shrinkage)
                w = np.zeros(n_models, dtype=float)
                w[keep] = w_small
                combined[i, h] = float(np.nansum(w * y_preds[i, :, h]))

                if i == n_windows - 1:
                    # store last learned weights for inspection
                    for j in range(n_models):
                        weights_by_h[h][j] = float(w[j])

        debug["weights_last_window_by_horizon"] = {
            str(h): {model_names[j]: float(weights_by_h[h][j]) for j in range(n_models)}
            for h in range(horizon)
        }
        return combined, debug

    if method == "exp_weighted_average_per_horizon":
        eta = float(candidate.params.get("eta", 1.0))
        trim_ratio = float(candidate.params.get("trim_ratio", 1.0))
        debug.update({"eta": eta, "trim_ratio": trim_ratio})

        weights_by_h: List[List[float]] = []
        for h in range(horizon):
            weights_by_h.append([float("nan")] * n_models)

        for i in range(n_windows):
            tr = _train_slice(i, rolling_cfg)
            if tr.stop - tr.start <= 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                continue

            train_true = y_true[tr, :]
            train_preds = y_preds[tr, :, :]

            for h in range(horizon):
                rmse_per_model = np.sqrt(np.nanmean((train_preds[:, :, h] - train_true[:, h][:, None]) ** 2, axis=0))
                keep = _keep_best_by_ratio(rmse_per_model, trim_ratio)
                if keep.size == 0:
                    combined[i, h] = np.nanmean(y_preds[i, :, h])
                    continue
                w_small = _weights_exp(rmse_per_model[keep], eta=eta)
                w = np.zeros(n_models, dtype=float)
                w[keep] = w_small
                combined[i, h] = float(np.nansum(w * y_preds[i, :, h]))

                if i == n_windows - 1:
                    for j in range(n_models):
                        weights_by_h[h][j] = float(w[j])

        debug["weights_last_window_by_horizon"] = {
            str(h): {model_names[j]: float(weights_by_h[h][j]) for j in range(n_models)}
            for h in range(horizon)
        }
        return combined, debug

    if method == "poly_weighted_average_per_horizon":
        power = float(candidate.params.get("power", 1.0))
        trim_ratio = float(candidate.params.get("trim_ratio", 1.0))
        debug.update({"power": power, "trim_ratio": trim_ratio})

        weights_by_h: List[List[float]] = []
        for h in range(horizon):
            weights_by_h.append([float("nan")] * n_models)

        for i in range(n_windows):
            tr = _train_slice(i, rolling_cfg)
            if tr.stop - tr.start <= 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                continue

            train_true = y_true[tr, :]
            train_preds = y_preds[tr, :, :]

            for h in range(horizon):
                rmse_per_model = np.sqrt(np.nanmean((train_preds[:, :, h] - train_true[:, h][:, None]) ** 2, axis=0))
                keep = _keep_best_by_ratio(rmse_per_model, trim_ratio)
                if keep.size == 0:
                    combined[i, h] = np.nanmean(y_preds[i, :, h])
                    continue
                w_small = _weights_poly(rmse_per_model[keep], power=power)
                w = np.zeros(n_models, dtype=float)
                w[keep] = w_small
                combined[i, h] = float(np.nansum(w * y_preds[i, :, h]))

                if i == n_windows - 1:
                    for j in range(n_models):
                        weights_by_h[h][j] = float(w[j])

        debug["weights_last_window_by_horizon"] = {
            str(h): {model_names[j]: float(weights_by_h[h][j]) for j in range(n_models)}
            for h in range(horizon)
        }
        return combined, debug

    if method == "ade_dynamic_error_per_horizon":
        beta = float(candidate.params.get("beta", 0.5))
        eta = float(candidate.params.get("eta", 1.0))
        trim_ratio = float(candidate.params.get("trim_ratio", 1.0))
        debug.update({"beta": beta, "eta": eta, "trim_ratio": trim_ratio})

        # Exponential moving average of absolute errors per model/horizon
        ema_errors = np.full((n_models, horizon), np.nan, dtype=float)
        weights_by_h: List[List[float]] = []
        for h in range(horizon):
            weights_by_h.append([float("nan")] * n_models)

        for i in range(n_windows):
            if i == 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                continue

            prev_err = np.abs(y_preds[i - 1, :, :] - y_true[i - 1, :][None, :])
            if np.any(np.isnan(ema_errors)):
                ema_errors = prev_err.copy()
            else:
                ema_errors = beta * prev_err + (1.0 - beta) * ema_errors

            for h in range(horizon):
                err_h = ema_errors[:, h]
                keep = _keep_best_by_ratio(err_h, trim_ratio)
                if keep.size == 0:
                    combined[i, h] = np.nanmean(y_preds[i, :, h])
                    continue
                w_small = _weights_exp(err_h[keep], eta=eta)
                w = np.zeros(n_models, dtype=float)
                w[keep] = w_small
                combined[i, h] = float(np.nansum(w * y_preds[i, :, h]))

                if i == n_windows - 1:
                    for j in range(n_models):
                        weights_by_h[h][j] = float(w[j])

        debug["weights_last_window_by_horizon"] = {
            str(h): {model_names[j]: float(weights_by_h[h][j]) for j in range(n_models)}
            for h in range(horizon)
        }
        return combined, debug

    if method == "ridge_stacking_per_horizon":
        l2 = float(candidate.params.get("l2", 10.0))
        top_k = int(candidate.params.get("top_k", n_models))
        top_k = max(1, min(top_k, n_models))
        debug.update({"l2": l2, "top_k": top_k})

        last_weights: Dict[str, Dict[str, float]] = {}

        for i in range(n_windows):
            tr = _train_slice(i, rolling_cfg)
            if tr.stop - tr.start <= 0:
                combined[i, :] = np.nanmean(y_preds[i, :, :], axis=0)
                continue

            for h in range(horizon):
                y_t = y_true[tr, h]
                X = y_preds[tr, :, h]

                # optional top_k filter by RMSE first
                rmse = np.sqrt(np.nanmean((X - y_t[:, None]) ** 2, axis=0))
                order = np.argsort(rmse)
                keep = order[:top_k]

                w_small = _ridge_weights(y_t, X[:, keep], l2=l2, nonneg_simplex=True)
                w = np.zeros(n_models, dtype=float)
                w[keep] = w_small
                combined[i, h] = float(np.nansum(w * y_preds[i, :, h]))

                if i == n_windows - 1:
                    last_weights[str(h)] = {model_names[j]: float(w[j]) for j in range(n_models)}

        debug["weights_last_window_by_horizon"] = last_weights
        return combined, debug

    # Unknown method: fallback
    combined[:, :] = np.nanmean(y_preds, axis=1)
    debug["fallback"] = True
    return combined, debug
