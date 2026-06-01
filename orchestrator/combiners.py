"""V5 — closed menu of six robust forecast combiners.

Each function takes:
  - `final_matrix`: shape (n_models, horizon) — the final-test predictions per model.
  - `y_true_val`, `y_preds_val`: validation data (only used by inverse-rmse / single-best).

Returns:
  - `result`: shape (horizon,) — the combined point forecast.
  - `debug`: dict with weights / chosen sub-models / parameters used.

Design (see ARCHITECTURE_V5_PROPOSAL.md):
  - Zero learned-weight estimation. All six methods are deterministic with provable variance.
  - This eliminates the "combination puzzle" hit V3/V4 took on 3 validation windows.
  - The LLM Selector picks ONE of these six per series.

References:
  - Atiya (2020) "Why does forecast combination work so well?" IJF.
  - Spiliotis (2024) "Forecast combinations in the M competitions" — trimmed_mean(20%) top-3.
  - Hyndman & Athanasopoulos (FPP3) — median/trimmed-mean as the robust default.
  - James-Stein (1961) — shrinkage estimator dominates ML in dimension ≥3.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


# Canonical menu names — exposed so the LLM Selector & router speak the same vocabulary.
MENU = [
    "simple_median",
    "trimmed_mean_20",
    "winsorized_mean_10",
    "geometric_mean_positive",
    "inverse_rmse_shrunk",
    "single_best_val",
]


# ── Method 1: simple median ──────────────────────────────────────────────────
def simple_median(final_matrix: np.ndarray, **_kw) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-horizon median. Robust to up to 50% outliers in the model pool."""
    result = np.nanmedian(final_matrix, axis=0)
    n_models = final_matrix.shape[0]
    weights = np.full(n_models, 1.0 / n_models, dtype=float)
    return result, {"method": "simple_median", "uniform_weights": weights.tolist()}


# ── Method 2: trimmed mean (20% from each tail) ──────────────────────────────
def trimmed_mean_20(final_matrix: np.ndarray, trim_ratio: float = 0.20, **_kw) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-horizon trimmed mean. Top-3 method in M competitions (Spiliotis 2024).

    Drops the top/bottom `trim_ratio` fraction of model predictions per horizon, then averages
    the middle. With ratio=0.20 and n_models=23, drops 4+4 → averages 15 middle predictions.
    """
    n_models, horizon = final_matrix.shape
    k_drop = int(np.floor(n_models * float(trim_ratio)))
    if 2 * k_drop >= n_models:
        # Fall through to median if trimming would empty the pool
        return simple_median(final_matrix)
    sorted_by_h = np.sort(final_matrix, axis=0)
    middle = sorted_by_h[k_drop : n_models - k_drop, :]
    result = np.nanmean(middle, axis=0)
    return result, {
        "method": "trimmed_mean_20",
        "trim_ratio": float(trim_ratio),
        "n_kept_per_horizon": int(middle.shape[0]),
    }


# ── Method 3: winsorized mean (10% from each tail) ──────────────────────────
def winsorized_mean_10(final_matrix: np.ndarray, wins_ratio: float = 0.10, **_kw) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-horizon Winsorized mean. Preserves more information than trimmed_mean by clipping
    instead of dropping the tails. Robust to extreme outliers without discarding observations.
    """
    n_models, horizon = final_matrix.shape
    k = max(1, int(np.floor(n_models * float(wins_ratio))))
    if 2 * k >= n_models:
        return simple_median(final_matrix)
    sorted_by_h = np.sort(final_matrix, axis=0)
    low = sorted_by_h[k : k + 1, :]  # shape (1, h)
    high = sorted_by_h[n_models - k - 1 : n_models - k, :]
    winsorized = np.clip(final_matrix, low, high)
    result = np.nanmean(winsorized, axis=0)
    return result, {"method": "winsorized_mean_10", "wins_ratio": float(wins_ratio), "k_clipped": int(k)}


# ── Method 4: geometric mean (positive series only) ─────────────────────────
def geometric_mean_positive(final_matrix: np.ndarray, **_kw) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-horizon geometric mean. Dominant for log-normal positive series (sales, demand,
    counts). Requires ALL predictions strictly positive — otherwise raises (caller falls back
    to trimmed_mean_20 via the Applier safeguard).
    """
    if np.any(~np.isfinite(final_matrix)) or np.any(final_matrix <= 0):
        raise ValueError(
            "geometric_mean_positive requires strictly positive, finite forecasts; "
            "fall back to trimmed_mean_20 if not satisfied."
        )
    result = np.exp(np.nanmean(np.log(final_matrix), axis=0))
    n_models = final_matrix.shape[0]
    weights = np.full(n_models, 1.0 / n_models, dtype=float)
    return result, {"method": "geometric_mean_positive", "uniform_log_weights": weights.tolist()}


# ── Method 5: inverse-RMSE weighted with James-Stein shrinkage ──────────────
def inverse_rmse_shrunk(
    final_matrix: np.ndarray,
    y_true_val: np.ndarray,
    y_preds_val: np.ndarray,
    eps: float = 1e-8,
    **_kw,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inverse-RMSE per-model weights, shrunk toward uniform via James-Stein-style factor.

    Plain inverse-RMSE: w_i ∝ 1/(rmse_i + eps). Has high variance with few validation windows.
    James-Stein shrinkage toward the uniform vector reduces MSE of the weight estimator and
    almost surely dominates the unshrunk estimator when dim ≥ 3 (Stein 1961; Efron-Morris 1973).

    Shrinkage factor: λ = min(1, (k-2) * var_uniform / SS), where SS is the sum-of-squares of
    (w_raw - w_uniform). Higher λ when there's less signal (close to uniform); lower when
    one or two models dominate (Bayesian shrinkage of point estimates).
    """
    y_true_val = np.asarray(y_true_val, dtype=float)
    y_preds_val = np.asarray(y_preds_val, dtype=float)
    n_models = y_preds_val.shape[1]

    # Per-model RMSE across windows × horizon
    errors = y_preds_val - y_true_val[:, None, :]  # (n_windows, n_models, horizon)
    rmse = np.sqrt(np.nanmean(errors ** 2, axis=(0, 2)))  # (n_models,)
    rmse = np.where(np.isfinite(rmse) & (rmse > eps), rmse, np.nanmax(rmse[np.isfinite(rmse)]) if np.any(np.isfinite(rmse)) else 1.0)

    w_raw = 1.0 / (rmse + eps)
    w_raw = w_raw / np.sum(w_raw)
    w_uniform = np.full(n_models, 1.0 / n_models, dtype=float)

    ss = float(np.sum((w_raw - w_uniform) ** 2))
    var_uniform = float(np.var(w_uniform))  # actually zero for uniform → use 1/n^2 floor
    if ss <= 1e-12 or n_models < 3:
        shrink = 1.0
    else:
        # Empirical James-Stein-style factor; bounded to [0, 1]
        shrink = min(1.0, max(0.0, (n_models - 2) / (n_models * ss + 1e-9)))
    w_final = shrink * w_uniform + (1.0 - shrink) * w_raw
    # Renormalize for safety
    w_final = w_final / np.sum(w_final)

    result = np.nansum(w_final[:, None] * final_matrix, axis=0)
    return result, {
        "method": "inverse_rmse_shrunk",
        "weights": w_final.tolist(),
        "shrinkage_lambda": float(shrink),
        "rmse_per_model": rmse.tolist(),
    }


# ── Method 6: single best by validation RMSE ────────────────────────────────
def single_best_val(
    final_matrix: np.ndarray,
    y_true_val: np.ndarray,
    y_preds_val: np.ndarray,
    gap_threshold: float = 0.05,
    **_kw,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Use the single best model by validation RMSE — but ONLY if the gap to the 2nd-best
    is ≥ gap_threshold (default 5%). Otherwise fall back to trimmed_mean_20 to avoid
    overfitting to validation noise. Pattern from TimeSeriesScientist (NeurIPS 2025).
    """
    y_true_val = np.asarray(y_true_val, dtype=float)
    y_preds_val = np.asarray(y_preds_val, dtype=float)
    n_models = y_preds_val.shape[1]

    errors = y_preds_val - y_true_val[:, None, :]
    rmse = np.sqrt(np.nanmean(errors ** 2, axis=(0, 2)))
    order = np.argsort(rmse)
    best_idx, second_idx = int(order[0]), int(order[1]) if n_models >= 2 else (int(order[0]), int(order[0]))
    best_rmse = float(rmse[best_idx])
    second_rmse = float(rmse[second_idx]) if n_models >= 2 else float("inf")

    rel_gap = (second_rmse - best_rmse) / (abs(best_rmse) + 1e-9) if n_models >= 2 else float("inf")
    if rel_gap < gap_threshold:
        # Safeguard fallback
        result, dbg = trimmed_mean_20(final_matrix)
        dbg.update({"safeguard_triggered": "single_best gap < 5%", "best_rmse": best_rmse, "second_rmse": second_rmse, "fallback_to": "trimmed_mean_20"})
        return result, dbg

    return final_matrix[best_idx, :], {
        "method": "single_best_val",
        "chosen_idx": best_idx,
        "best_rmse": best_rmse,
        "second_rmse": second_rmse,
        "rel_gap": float(rel_gap),
    }


# ── Dispatcher ──────────────────────────────────────────────────────────────
_METHODS = {
    "simple_median": simple_median,
    "trimmed_mean_20": trimmed_mean_20,
    "winsorized_mean_10": winsorized_mean_10,
    "geometric_mean_positive": geometric_mean_positive,
    "inverse_rmse_shrunk": inverse_rmse_shrunk,
    "single_best_val": single_best_val,
}


def apply_combiner(
    method: str,
    final_matrix: np.ndarray,
    y_true_val: np.ndarray = None,
    y_preds_val: np.ndarray = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the chosen menu method to the final predictions. Safe-fallback on errors.

    The Applier layer of V5: takes the LLM's choice and produces the final point forecast,
    with automatic safeguards (geometric_mean falls back to trimmed_mean if any prediction is
    non-positive; single_best falls back if gap < 5%; unknown method → trimmed_mean).
    """
    fn = _METHODS.get(method)
    if fn is None:
        result, dbg = trimmed_mean_20(final_matrix)
        dbg.update({"safeguard_triggered": f"unknown method {method!r}", "fallback_to": "trimmed_mean_20"})
        return result, dbg
    try:
        return fn(final_matrix, y_true_val=y_true_val, y_preds_val=y_preds_val)
    except Exception as e:
        result, dbg = trimmed_mean_20(final_matrix)
        dbg.update({"safeguard_triggered": f"{method} raised: {e!s}", "fallback_to": "trimmed_mean_20"})
        return result, dbg


def evaluate_method_on_validation(
    method: str,
    y_true_val: np.ndarray,
    y_preds_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a menu method on the validation windows. Used by the Selector to know how
    each method scored locally on this series BEFORE making the final pick.

    Returns dict with smape, rmse, mae per window-aggregated metrics + composite score.
    """
    y_true_val = np.asarray(y_true_val, dtype=float)
    y_preds_val = np.asarray(y_preds_val, dtype=float)
    n_windows = y_true_val.shape[0]

    all_preds = []
    all_true = []
    for w in range(n_windows):
        y_pred_window, _ = apply_combiner(
            method,
            y_preds_val[w],  # (n_models, horizon)
            y_true_val=y_true_val[max(0, w - 0):w] if w > 0 else y_true_val[0:1],
            y_preds_val=y_preds_val[max(0, w - 0):w] if w > 0 else y_preds_val[0:1],
        )
        all_preds.append(y_pred_window)
        all_true.append(y_true_val[w])

    yp = np.concatenate(all_preds)
    yt = np.concatenate(all_true)
    err = yp - yt
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    mae = float(np.nanmean(np.abs(err)))
    smape = float(np.nanmean(2.0 * np.abs(err) / (np.abs(yp) + np.abs(yt) + 1e-9)))
    return {"rmse": rmse, "mae": mae, "smape": smape, "composite": (rmse + smape) / 2.0}
