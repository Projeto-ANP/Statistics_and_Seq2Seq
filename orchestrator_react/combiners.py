"""Combination functions — single source of truth.

These are the exact same functions used in the validation-window backtest (Phase 3)
and in the final application to the test forecast (Phase 4). That removes the
divergent duplication that exists today between `orchestrator/strategies.py` and
`orchestrator/final_predictor.py` (see EXPLORACAO.md, section 6.1).

Every function here takes an `(n_models, horizon)` matrix for a single window and
returns a `(horizon,)` vector. None of them looks at `y_true`: weight learning lives
in `weighting.py`, under the anti-leakage control of `state.py`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _check(preds: np.ndarray) -> np.ndarray:
    p = np.asarray(preds, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"expected (n_models, horizon), got shape={p.shape}")
    if p.shape[0] == 0:
        raise ValueError("empty pool: nothing to combine")
    return p


def combine_mean(preds: np.ndarray) -> np.ndarray:
    return np.nanmean(_check(preds), axis=0)


def combine_median(preds: np.ndarray) -> np.ndarray:
    return np.nanmedian(_check(preds), axis=0)


def combine_trimmed_mean(preds: np.ndarray, trim_pct: float = 0.2) -> np.ndarray:
    """Per-horizon trimmed mean: drops `trim_pct` from each tail across models."""
    p = _check(preds)
    m = p.shape[0]
    k = int(np.floor(m * float(trim_pct)))
    if k <= 0 or 2 * k >= m:
        return np.nanmean(p, axis=0)
    ordered = np.sort(p, axis=0)
    return np.nanmean(ordered[k : m - k], axis=0)


def combine_weighted(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean. `weights` is `(n_models,)` or `(n_models, horizon)`.

    Weight belonging to models whose forecast is NaN at a given horizon is
    redistributed across the rest, so the effective weights still sum to 1 and a
    single NaN does not poison the whole horizon.
    """
    p = _check(preds)
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = np.repeat(w[:, None], p.shape[1], axis=1)
    if w.shape != p.shape:
        raise ValueError(f"weights {w.shape} incompatible with forecasts {p.shape}")

    valid = np.isfinite(p)
    w_eff = np.where(valid, w, 0.0)
    denom = w_eff.sum(axis=0)
    out = np.full(p.shape[1], np.nan, dtype=float)
    good = denom > 0
    contrib = np.where(valid, p, 0.0) * w_eff
    out[good] = contrib.sum(axis=0)[good] / denom[good]
    # Horizons with no valid weight fall back to the mean of available models.
    if np.any(~good):
        out[~good] = np.nanmean(p[:, ~good], axis=0)
    return out


def combine_best_single(preds: np.ndarray, model_pos: int) -> np.ndarray:
    """Uses a single model. `model_pos` is the position inside the given pool."""
    p = _check(preds)
    if not (0 <= int(model_pos) < p.shape[0]):
        raise ValueError(f"model_pos={model_pos} outside pool of size {p.shape[0]}")
    return p[int(model_pos), :].copy()


def combine_dba(preds: np.ndarray, max_iter: int = 30, random_state: int = 7) -> np.ndarray:
    """DTW Barycenter Averaging. Falls back to the plain mean if tslearn is
    missing or fails.

    Reuses the same call used by `combinations/dba.py` and by the two legacy
    combiners in the project.

    `random_state` is not optional in practice: `dtw_barycenter_averaging` inits
    its centroid from `sklearn.utils.check_random_state(None)` when no seed is
    given, which reads the ambient global numpy RNG rather than anything tied to
    the input. Two identical NN5 series (T1==T47, T11==T50, T79==T111) that both
    picked `dba` over the same full pool produced different forecasts under the
    old default (max abs diff 0.79, sMAPE 0.1199 vs 0.1217) — not because the
    inputs differed, but because a different number of unrelated `np.random` calls
    had happened elsewhere in the process by the time each series reached this
    line. A fixed seed here closes that gap; it does not, by itself, make the rest
    of the pipeline deterministic (the LLM sampling seed is a separate control).
    """
    p = _check(preds)
    col_means = np.nanmean(p, axis=0)
    clean = np.where(np.isfinite(p), p, col_means[None, :])
    if not np.all(np.isfinite(clean)):
        clean = np.nan_to_num(clean, nan=0.0)

    try:
        from tslearn.barycenters import dtw_barycenter_averaging
    except Exception:
        return np.nanmean(p, axis=0)

    try:
        centroid = dtw_barycenter_averaging(
            clean.reshape(clean.shape[0], clean.shape[1], 1),
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
        out = np.asarray(centroid, dtype=float).ravel()[: p.shape[1]]
        if out.size != p.shape[1] or not np.all(np.isfinite(out)):
            return np.nanmean(p, axis=0)
        return out
    except Exception:
        return np.nanmean(p, axis=0)


#: Methods accepted in a strategy spec. Closed action space.
COMBINE_METHODS = ("mean", "median", "trimmed_mean", "weighted", "dba", "best_single")


def apply_combination(
    preds: np.ndarray,
    method: str,
    weights: Optional[np.ndarray] = None,
    trim_pct: float = 0.2,
    model_pos: int = 0,
    dba_max_iter: int = 30,
    dba_random_state: int = 7,
) -> np.ndarray:
    """Single dispatch point used by both the backtest and the final application."""
    method = str(method).strip().lower()
    if method == "mean":
        return combine_mean(preds)
    if method == "median":
        return combine_median(preds)
    if method == "trimmed_mean":
        return combine_trimmed_mean(preds, trim_pct=trim_pct)
    if method == "weighted":
        if weights is None:
            raise ValueError("combine 'weighted' requires a weights handle")
        return combine_weighted(preds, weights)
    if method == "dba":
        return combine_dba(preds, max_iter=dba_max_iter, random_state=dba_random_state)
    if method == "best_single":
        return combine_best_single(preds, model_pos=model_pos)
    raise ValueError(f"unknown combination method: {method!r} (valid: {COMBINE_METHODS})")
