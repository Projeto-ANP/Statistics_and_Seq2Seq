"""Closed tool catalog (Section 3.4).

Contract for every tool here:
    * deterministic — same input, same output;
    * takes `state` as the first argument and never writes outside it;
    * returns a **compact** dict (order of 100-300 tokens), never whole arrays —
      the agent sees summaries, not raw data;
    * raises `ValueError`/`KeyError` with an actionable message on a bad argument,
      so the registry can record the failure under `tool_missing`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from orchestrator_react import features as F
from orchestrator_react.config import FINAL_SCORE_PRESETS, SCORE_PRESETS
from orchestrator_react import meta_model as MM
from orchestrator_react.selection import (
    PoolRecipe,
    forward_indices,
    rank_table,
    stable_indices,
    top_k_indices,
)
from orchestrator_react.state import FULL_POOL, ReactState, _spec_label
from orchestrator_react.weighting import ERROR_METRICS, WeightsRecipe, per_model_error


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.1 — Diagnostics (read-only)
# ══════════════════════════════════════════════════════════════════════════════


def series_profile(state: ReactState) -> Dict[str, Any]:
    """Series card: length, seasonality, stationarity, outliers, catch22.

    Includes the trend/seasonality champions computed deterministically
    (`features.component_champions`).
    """
    series = state.train_series
    if series is None or np.asarray(series).size == 0:
        # Without a historical series, the profile falls back to the window actuals.
        series = state.y_true.reshape(-1)
        source = "validation_windows"
    else:
        series = np.asarray(series, dtype=float).ravel()
        source = "train_series"

    # The seasonal period comes from the dataset's declared `@frequency`, never
    # guessed from the data (monthly => 12, hourly => 24, half-hourly => 48, ...).
    period_info = F.resolve_seasonal_period(
        state.freq, int(series.size), state.horizon, explicit=state.seasonal_period
    )
    period = F.infer_seasonal_period(state.freq, int(series.size), state.horizon)
    if state.seasonal_period:
        period = int(state.seasonal_period)
    trend, seasonal, resid = F.stl_decompose(series, period)
    trend_info = F.trend_direction(trend, series)

    profile: Dict[str, Any] = {
        "source": source,
        "decomposition": F.LAST_DECOMPOSITION["method"],
        "n_points": int(np.isfinite(series).sum()),
        "frequency": period_info["frequency"],
        "seasonal_period": int(period),
        "seasonal_period_declared": period_info["declared"],
        "seasonal_period_source": period_info["source"],
        "horizon": state.horizon,
        "n_validation_windows": state.n_windows,
        "n_models": state.n_models,
        **F.stl_strengths(trend, seasonal, resid),
        "trend_direction": trend_info["direction"],
        "trend_relative_change": trend_info["relative_change"],
        "stationarity": F.stationarity(series),
        "outliers": F.outlier_flags(series),
        "features": F.fast_features(series, period),
    }

    c22 = F.catch22_features(series)
    profile["catch22"] = c22 if c22 is not None else "pycatch22 unavailable"

    # Champions are computed on the concatenated validation windows: that is the
    # only stretch where both `y_true` and every model's forecast exist together.
    profile.update(
        F.component_champions(
            state.y_true,
            state.y_preds,
            state.model_names,
            freq=state.freq,
            horizon=state.horizon,
            explicit_period=state.seasonal_period,
            contiguous_windows=state.windows_are_contiguous(),
        )
    )
    return profile


def stl_summary(state: ReactState) -> Dict[str, Any]:
    """Share of variance explained by trend, seasonality and residual."""
    series = state.train_series
    if series is None or np.asarray(series).size == 0:
        series = state.y_true.reshape(-1)
    series = np.asarray(series, dtype=float).ravel()
    period_info = F.resolve_seasonal_period(
        state.freq, int(series.size), state.horizon, explicit=state.seasonal_period
    )
    period = int(state.seasonal_period) if state.seasonal_period else F.infer_seasonal_period(
        state.freq, int(series.size), state.horizon
    )
    trend, seasonal, resid = F.stl_decompose(series, period)
    out = {
        "period": int(period),
        "period_source": period_info["source"],
        "frequency": period_info["frequency"],
        "decomposition": F.LAST_DECOMPOSITION["method"],
        **F.variance_shares(trend, seasonal, resid),
    }
    out.update(F.stl_strengths(trend, seasonal, resid))
    out["dominant_component"] = max(
        (
            ("trend", out["trend_pct"]),
            ("seasonality", out["seasonal_pct"]),
            ("residual", out["residual_pct"]),
        ),
        key=lambda kv: kv[1],
    )[0]
    return out


def error_summary(
    state: ReactState,
    window: Optional[int] = None,
    top_n: int = 8,
    metric: str = "rmse",
) -> Dict[str, Any]:
    """Per-model error table, ranked. Returns only the top-N plus a rest aggregate."""
    metric = str(metric).lower()
    if metric not in ERROR_METRICS:
        raise ValueError(f"metric={metric!r} is invalid. Valid: {list(ERROR_METRICS)}")
    top_n = int(np.clip(int(top_n), 1, state.n_models))

    windows = list(range(state.n_windows)) if window is None else [int(window)]
    bad = [w for w in windows if not (0 <= w < state.n_windows)]
    if bad:
        raise ValueError(f"window {bad} outside range [0, {state.n_windows - 1}]")

    err = per_model_error(state.y_true[windows], state.y_preds[windows], metric=metric)
    order = np.argsort(err)

    rows = [
        {
            "model": state.model_names[int(j)],
            "error": round(float(err[j]), 4) if np.isfinite(err[j]) else None,
            "rank": position + 1,
        }
        for position, j in enumerate(order[:top_n])
    ]

    rest_idx = order[top_n:]
    rest = None
    if rest_idx.size:
        finite = err[rest_idx][np.isfinite(err[rest_idx])]
        rest = {
            "n_models": int(rest_idx.size),
            "median_error": round(float(np.median(finite)), 4) if finite.size else None,
            "worst_error": round(float(np.max(finite)), 4) if finite.size else None,
        }

    finite_all = err[np.isfinite(err)]
    spread = None
    if finite_all.size >= 2 and float(finite_all.min()) > 0:
        spread = round(float((finite_all.max() - finite_all.min()) / finite_all.min()), 3)

    return {
        "metric": metric,
        "window": "all" if window is None else int(window),
        "top": rows,
        "rest": rest,
        "relative_spread": spread,
    }


def ranking_stability(state: ReactState, metric: str = "rmse") -> Dict[str, Any]:
    """Agreement between the per-window rankings, and who moves the most."""
    if state.n_windows < 2:
        return {"mean_kendall_tau": None, "reason": "fewer than 2 windows"}

    ranks = np.zeros((state.n_windows, state.n_models), dtype=float)
    for i in range(state.n_windows):
        err = per_model_error(state.y_true[i : i + 1], state.y_preds[i : i + 1], metric=metric)
        ranks[i] = np.argsort(np.argsort(err)) + 1

    taus: List[float] = []
    try:
        from scipy.stats import kendalltau

        for i in range(state.n_windows):
            for j in range(i + 1, state.n_windows):
                tau, _ = kendalltau(ranks[i], ranks[j])
                if np.isfinite(tau):
                    taus.append(float(tau))
    except Exception:
        taus = []

    spread = ranks.max(axis=0) - ranks.min(axis=0)
    movers = np.argsort(spread)[::-1][:5]
    tau_mean = float(np.mean(taus)) if taus else None

    if tau_mean is None:
        verdict = "unavailable"
    elif tau_mean >= 0.7:
        verdict = "stable"
    elif tau_mean >= 0.3:
        verdict = "moderate"
    else:
        verdict = "unstable"

    return {
        "mean_kendall_tau": round(tau_mean, 3) if tau_mean is not None else None,
        "verdict": verdict,
        "biggest_movers": [
            {
                "model": state.model_names[int(j)],
                "rank_spread": int(spread[j]),
                "ranks": [int(v) for v in ranks[:, j]],
            }
            for j in movers
            if spread[j] > 0
        ],
        "always_top3": [
            state.model_names[j]
            for j in range(state.n_models)
            if bool(np.all(ranks[:, j] <= 3))
        ],
    }


def error_correlation(
    state: ReactState,
    model_ids: Optional[Sequence[str]] = None,
    threshold: float = 0.9,
) -> Dict[str, Any]:
    """Groups of models whose errors are strongly correlated (redundancy)."""
    threshold = float(np.clip(threshold, 0.0, 1.0))
    idx = (
        list(range(state.n_models))
        if not model_ids
        else [state.model_index(m) for m in model_ids]
    )
    if len(idx) < 2:
        raise ValueError("error_correlation needs at least 2 models")

    # (n_models, n_windows * horizon) — residuals concatenated per model.
    resid = np.transpose(
        state.y_preds[:, idx, :] - state.y_true[:, None, :], (1, 0, 2)
    ).reshape(len(idx), -1)

    n = len(idx)
    corr = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            xa, xb = resid[a], resid[b]
            mask = np.isfinite(xa) & np.isfinite(xb)
            if int(mask.sum()) < 3 or np.std(xa[mask]) < 1e-12 or np.std(xb[mask]) < 1e-12:
                c = 0.0
            else:
                c = float(np.corrcoef(xa[mask], xb[mask])[0, 1])
            corr[a, b] = corr[b, a] = 0.0 if not np.isfinite(c) else c

    # Greedy grouping by correlation threshold.
    unassigned = set(range(n))
    groups: List[List[int]] = []
    while unassigned:
        seed = min(unassigned)
        group = [seed]
        unassigned.discard(seed)
        for other in sorted(unassigned):
            if all(corr[other, g] >= threshold for g in group):
                group.append(other)
                unassigned.discard(other)
        groups.append(group)

    off = corr[np.triu_indices(n, k=1)]
    redundant = [
        {
            "models": [state.model_names[idx[g]] for g in grp],
            "representative": state.model_names[idx[grp[0]]],
        }
        for grp in groups
        if len(grp) > 1
    ]
    return {
        "threshold": threshold,
        "n_models": n,
        "mean_corr": round(float(np.mean(np.abs(off))), 3) if off.size else None,
        "n_groups": len(groups),
        "redundant_groups": redundant[:6],
        "n_independent": int(sum(1 for g in groups if len(g) == 1)),
    }


def dm_test(
    state: ReactState, model_a: str, model_b: str, loss: str = "squared"
) -> Dict[str, Any]:
    """Diebold-Mariano between two models, with the HLN small-sample correction."""
    ia, ib = state.model_index(model_a), state.model_index(model_b)
    ea = (state.y_preds[:, ia, :] - state.y_true).reshape(-1)
    eb = (state.y_preds[:, ib, :] - state.y_true).reshape(-1)

    # Pure-numpy utility reused from the existing codebase.
    from orchestrator.diagnostics import diebold_mariano

    res = diebold_mariano(ea, eb, loss=str(loss), h=1)
    p = res.get("p_value")
    stat = res.get("dm_stat")
    if p is None or not np.isfinite(p):
        verdict = "undetermined"
    elif p > 0.10:
        verdict = "statistical_tie"
    else:
        verdict = f"{model_a} worse" if (stat or 0) > 0 else f"{model_a} better"
    return {
        "model_a": model_a,
        "model_b": model_b,
        "dm_stat": round(float(stat), 3) if stat is not None and np.isfinite(stat) else None,
        "p_value": round(float(p), 4) if p is not None and np.isfinite(p) else None,
        "n_obs": int(res.get("n", 0)),
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.2 — Pool selection
# ══════════════════════════════════════════════════════════════════════════════


def select_top_k(
    state: ReactState,
    k: int,
    metric: str = "rmse",
    windows: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Use when one or two models are clearly ahead: pool with the k lowest-error models."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, state.n_models)
    win = list(range(state.n_windows)) if windows is None else [int(w) for w in windows]
    bad = [w for w in win if not (0 <= w < state.n_windows)]
    if bad:
        raise ValueError(f"windows {bad} outside range [0, {state.n_windows - 1}]")

    chosen = top_k_indices(state.y_true[win], state.y_preds[win], k, metric)
    existed = set(state.pools)
    handle = state.register_pool(
        chosen, origin="top_k_error", metric=metric, windows=win, k=k,
        # Re-fittable: under nested selection each fold re-picks its own k lowest,
        # so the window being scored never voted on who is in the pool.
        recipe=PoolRecipe(method="top_k", params={"k": k, "metric": str(metric)}),
    )
    return {
        "pool": handle,
        "k": int(k),
        "criterion": f"lowest {metric}",
        "models": [state.model_names[int(j)] for j in chosen],
        **_handle_note(handle, existed),
    }


def select_stable(state: ReactState, k: int, metric: str = "rmse") -> Dict[str, Any]:
    """Use when the ranking is unstable across windows: pool with the k most consistent models.

    Criterion: lowest mean rank penalised by rank variability (`mean + std`). A
    model that is 1st in one window and 20th in another loses to one that is 4th in
    all of them.
    """
    k = int(np.clip(int(k), 1, state.n_models))
    if state.n_windows < 2:
        return select_top_k(state, k=k, metric=metric)

    ranks = rank_table(state.y_true, state.y_preds, metric=metric)
    chosen = stable_indices(state.y_true, state.y_preds, k, metric)
    existed = set(state.pools)
    handle = state.register_pool(
        chosen, origin="top_k_stable", metric=metric, k=k,
        recipe=PoolRecipe(method="stable", params={"k": k, "metric": str(metric)}),
    )
    return {
        "pool": handle,
        **_handle_note(handle, existed),
        "k": int(k),
        "criterion": "mean rank + std across windows",
        "models": [
            {
                "model": state.model_names[int(j)],
                "mean_rank": round(float(ranks[:, j].mean()), 1),
                "rank_std": round(float(ranks[:, j].std()), 1),
            }
            for j in chosen
        ],
    }


def prune_redundant(
    state: ReactState,
    pool: str = FULL_POOL,
    corr_threshold: float = 0.95,
    metric: str = "rmse",
) -> Dict[str, Any]:
    """Use when several models carry correlated error: drop the redundant ones, keeping the best of each group."""
    idx = state.get_pool(pool)
    if len(idx) < 2:
        raise ValueError(f"pool {pool!r} has fewer than 2 models")

    names = [state.model_names[i] for i in idx]
    groups = error_correlation(state, model_ids=names, threshold=float(corr_threshold))
    err = per_model_error(state.y_true, state.y_preds, metric=metric)

    removed: List[str] = []
    kept = set(names)
    for grp in groups.get("redundant_groups", []):
        members = list(grp["models"])
        best = min(members, key=lambda m: err[state.model_index(m)])
        for m in members:
            if m != best and m in kept:
                kept.discard(m)
                removed.append(m)

    keep_idx = [state.model_index(m) for m in names if m in kept]
    if not keep_idx:
        raise ValueError("pruning would remove every model; raise corr_threshold")

    existed = set(state.pools)
    handle = state.register_pool(
        keep_idx, origin="pruned", base=pool, corr_threshold=float(corr_threshold),
        recipe=PoolRecipe(
            method="prune_redundant",
            params={"corr_threshold": float(corr_threshold), "metric": str(metric)},
            base=tuple(int(i) for i in idx),
        ),
    )
    return {
        "pool": handle,
        **_handle_note(handle, existed),
        "base": pool,
        "corr_threshold": float(corr_threshold),
        "n_before": len(idx),
        "n_after": len(keep_idx),
        "removed": removed[:10],
    }


def select_forward(
    state: ReactState,
    max_steps: int = 20,
    metric: str = "rmse",
) -> Dict[str, Any]:
    """Use when no single model dominates and errors look complementary: Caruana greedy forward selection.

    Starts from the empty ensemble and adds, at each step, the model that most
    reduces the error of the running mean; the subset is the models picked at
    least once. Re-fit per backtest fold like every other pool recipe, so the
    scored window never takes part in choosing the models scored on it.
    """
    chosen = forward_indices(state.y_true, state.y_preds, max_steps=int(max_steps), metric=str(metric))
    existed = set(state.pools)
    handle = state.register_pool(
        chosen, origin="forward", metric=str(metric), max_steps=int(max_steps),
        # Re-fittable: under nested selection each fold re-runs the greedy choice
        # on the windows the protocol admits, so the scored window never voted.
        recipe=PoolRecipe(method="forward", params={"max_steps": int(max_steps), "metric": str(metric)}),
    )
    return {
        "pool": handle,
        **_handle_note(handle, existed),
        "k": int(len(chosen)),
        "criterion": f"Caruana forward selection ({metric})",
        "models": [state.model_names[int(j)] for j in chosen],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.3 — Weight computation (always returns a handle, never numbers)
# ══════════════════════════════════════════════════════════════════════════════


def _handle_note(handle: str, existed_before: set) -> Dict[str, Any]:
    """Tells the agent when it got back a handle it already had.

    An identical selection reuses its handle rather than minting a duplicate, so
    the name that comes back is not always a new one. Saying so removes the guess.
    """
    if handle not in existed_before:
        return {"reused": False}
    return {
        "reused": True,
        "note": f"this selection is identical to an existing pool; use {handle!r}",
    }


def _register(
    state: ReactState, method: str, pool: str, windows, per_horizon: bool, **params: Any
) -> Dict[str, Any]:
    recipe = WeightsRecipe(
        method=method,
        pool_handle=str(pool),
        fit_windows=None if windows is None else tuple(int(w) for w in windows),
        per_horizon=bool(per_horizon),
        params=params,
    )
    existed = set(state.weights)
    handle = state.register_weights(recipe)
    return {
        "weights": handle,
        "reused": handle in existed,
        "method": method,
        "pool": str(pool),
        "summary": state.weights_summary(handle),
        "effective_mode": recipe.meta.get("mode", method),
        "note": "raw values stay in the state; pass this handle to combine_weighted",
    }


def weights_inverse_error(
    state: ReactState,
    pool: str = FULL_POOL,
    windows: Optional[Sequence[int]] = None,
    metric: str = "rmse",
    shrinkage: float = 0.0,
    per_horizon: bool = False,
) -> Dict[str, Any]:
    """Use when models differ clearly in error: weights proportional to 1/error, optionally shrunk to uniform."""
    return _register(
        state, "inverse_error", pool, windows, per_horizon,
        metric=str(metric), shrinkage=float(np.clip(shrinkage, 0.0, 0.9)),
    )


def weights_softmax_neg_error(
    state: ReactState,
    pool: str = FULL_POOL,
    windows: Optional[Sequence[int]] = None,
    metric: str = "rmse",
    eta: float = 1.0,
    per_horizon: bool = False,
) -> Dict[str, Any]:
    """w proportional to exp(-eta * error) — the ADE form. Error median-normalised."""
    return _register(
        state, "softmax_neg_error", pool, windows, per_horizon,
        metric=str(metric), eta=float(np.clip(eta, 0.01, 20.0)),
    )


def weights_error_trend(
    state: ReactState,
    pool: str = FULL_POOL,
    windows: Optional[Sequence[int]] = None,
    metric: str = "mae",
    eta: float = 1.0,
    damping: Optional[float] = None,
) -> Dict[str, Any]:
    """Use when some models improve while others degrade: weights from where the
    error is heading, read from the pointwise error grid, not from the average.
    `damping=None` trusts the trend only as far as it is consistent."""
    return _register(
        state, "error_trend", pool, windows, False,
        metric=str(metric),
        eta=float(np.clip(eta, 0.01, 20.0)),
        damping=None if damping is None else float(np.clip(damping, 0.0, 1.0)),
    )


def weights_ols(
    state: ReactState,
    pool: str = FULL_POOL,
    windows: Optional[Sequence[int]] = None,
    l2: float = 0.0,
    nonneg: bool = True,
    per_horizon: bool = False,
) -> Dict[str, Any]:
    """Use with at least five windows: least-squares weights, projected onto the simplex by default."""
    return _register(
        state, "ols", pool, windows, per_horizon,
        l2=float(np.clip(l2, 0.0, 1000.0)), nonneg=bool(nonneg),
    )


def weights_feature_based(
    state: ReactState,
    pool: str = FULL_POOL,
    windows: Optional[Sequence[int]] = None,
    metric: str = "smape",
    eta: float = 1.0,
) -> Dict[str, Any]:
    """Meta-model in the spirit of FFORMA. Falls back to softmax(-error) when the
    sample is too small — the mode actually used is returned in `effective_mode`."""
    return _register(
        state, "feature_based", pool, windows, False, metric=str(metric), eta=float(eta)
    )


def weights_pooled_meta_model(state: ReactState, pool: str = FULL_POOL, eta: float = 1.0) -> Dict[str, Any]:
    """w from a gradient-boosted model trained across every OTHER series in this
    dataset run, predicting each pool model's error from this series' own
    historical shape (trend/seasonal strength, entropy, autocorrelation) — the
    same feature family FFORMA/ADE use, but fit on however many series the
    dataset has instead of on 3 windows of just this one, which is what actually
    lets the classical meta-learner have enough samples to work.

    Works with any pool, including one re-selected per backtest fold: what gets
    stored is a predicted error PER MODEL NAME, and each fold composes its own
    softmax over whichever members it actually has.
    """
    meta = getattr(state, "pooled_meta_model", None)
    if meta is None:
        raise ValueError(
            "no pooled meta-model is available for this run: fewer than the "
            "minimum number of series in the dataset, or xgboost unavailable. "
            "Use weights_inverse_error or weights_softmax_neg_error instead."
        )

    # Predict for EVERY model in the run, not just the requested pool: a fold may
    # re-select different members, and the recipe has to be able to answer for any
    # of them without re-querying the meta-model mid-backtest. `score_kind` tells
    # the resolver how to turn scores into weights ("error" -> softmax(-eta*s),
    # "margin" -> softmax(+s), the FFORMA objective's own output).
    profile = series_profile(state)
    features = MM.extract_meta_features(profile)
    scores, kind = meta.predict_scores(features)

    out = _register(
        state, "pooled_meta_model", pool, None, False,
        model_scores={k: (None if v is None else float(v)) for k, v in scores.items()},
        score_kind=kind,
        eta=float(eta),
    )
    out["objective"] = getattr(meta, "objective", "per_model")
    out["n_train_series"] = meta.n_train_series
    out["n_models_with_a_fit"] = sum(1 for v in scores.values() if v is not None)
    return out


def weights_meta_model(
    state: ReactState,
    pool: str = FULL_POOL,
    scope: str = "dataset",
    metric: str = "smape",
    eta: float = 1.0,
) -> Dict[str, Any]:
    """Use for cross-series knowledge (scope='dataset') or a per-series fit (scope='series').

    `scope="dataset"` uses the meta-model trained across the other series of the
    run, in the spirit of FFORMA. `scope="series"` fits a meta-model on this
    series' own windows, falling back to softmax(-error) when there are not
    enough samples. Both return a weights handle over the given pool.
    """
    scope = str(scope).strip().lower()
    if scope in ("dataset", "pooled", "pooled_meta_model"):
        out = weights_pooled_meta_model(state, pool=pool, eta=float(eta))
        out["scope"] = "dataset"
        return out
    if scope in ("series", "feature", "feature_based"):
        out = weights_feature_based(state, pool=pool, metric=str(metric), eta=float(eta))
        out["scope"] = "series"
        return out
    raise ValueError(
        f"scope={scope!r} is invalid. Valid: 'dataset' (pooled across series) "
        "or 'series' (fitted on this series' own windows)"
    )


def weights_best_per_horizon(
    state: ReactState,
    pool: str = FULL_POOL,
    metric: str = "rmse",
) -> Dict[str, Any]:
    """Use when the best model changes along the horizon: one-hot weights per step.

    For each step of the horizon, the weight goes to the pool model with the
    lowest validation error at that step. Re-fit per backtest fold like every
    other recipe, so the scored window never takes part in choosing the weights
    applied to it.
    """
    return _register(
        state, "best_per_horizon", pool, None, True, metric=str(metric),
    )


def weights_topk_per_horizon(
    state: ReactState,
    pool: str = FULL_POOL,
    k: int = 3,
    metric: str = "rmse",
) -> Dict[str, Any]:
    """Use when a few models are good but not the same ones at every step: top-k mean per step."""
    return _register(
        state, "topk_mean_per_horizon", pool, None, True,
        k=int(k), metric=str(metric),
    )


def weights_forward(
    state: ReactState,
    pool: str = FULL_POOL,
    max_steps: int = 20,
    metric: str = "rmse",
) -> Dict[str, Any]:
    """Use when the pool has no dominant model: Caruana selection counts as weights."""
    return _register(
        state, "forward_selection", pool, None, False,
        max_steps=int(max_steps), metric=str(metric),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.4 — Combination (assemble the strategy spec)
# ══════════════════════════════════════════════════════════════════════════════


def _strategy(state: ReactState, spec: Dict[str, Any]) -> Dict[str, Any]:
    norm = state.normalize_spec(spec)
    n = 1 if norm["combine"] == "best_single" else len(state.get_pool(norm["pool"]))
    return {
        "strategy": norm,
        "n_models": n,
        "next_step": "call evaluate_strategy with exactly this Action Input",
        "next_action_input": {"strategy": norm, "rationale": "<why this should work>"},
    }


def combine_mean(state: ReactState, pool: str = FULL_POOL) -> Dict[str, Any]:
    """Plain mean across the pool models."""
    return _strategy(state, {"combine": "mean", "pool": pool})


def combine_median(state: ReactState, pool: str = FULL_POOL) -> Dict[str, Any]:
    """Median across the pool models — robust to a single outlying model."""
    return _strategy(state, {"combine": "median", "pool": pool})


def combine_trimmed_mean(
    state: ReactState, pool: str = FULL_POOL, trim_pct: float = 0.2
) -> Dict[str, Any]:
    """Trimmed mean: drops trim_pct from each tail before averaging."""
    return _strategy(state, {"combine": "trimmed_mean", "pool": pool, "trim_pct": trim_pct})


def combine_weighted(state: ReactState, pool: str, weights: str) -> Dict[str, Any]:
    """Weighted mean using an already computed weights handle."""
    return _strategy(state, {"combine": "weighted", "pool": pool, "weights": weights})


def combine_dba(state: ReactState, pool: str = FULL_POOL) -> Dict[str, Any]:
    """DTW Barycenter Averaging — aligns forecasts in time before averaging."""
    return _strategy(state, {"combine": "dba", "pool": pool})


def combine_best_single(state: ReactState, model_id: str) -> Dict[str, Any]:
    """Uses a single model, no combination."""
    return _strategy(state, {"combine": "best_single", "model": model_id})


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.5 — Validation and guardrails
# ══════════════════════════════════════════════════════════════════════════════


#: Weights method names the agent may pass straight into `evaluate_strategy`,
#: with the aliases it reaches for in practice (`inverse_error`, `forward`, ...).
_WEIGHTS_METHOD_TOOLS: Dict[str, Any] = {
    "inverse": weights_inverse_error,
    "inverse_error": weights_inverse_error,
    "weights_inverse_error": weights_inverse_error,
    "softmax": weights_softmax_neg_error,
    "softmax_neg_error": weights_softmax_neg_error,
    "weights_softmax_neg_error": weights_softmax_neg_error,
    "trend": weights_error_trend,
    "error_trend": weights_error_trend,
    "weights_error_trend": weights_error_trend,
    "ols": weights_ols,
    "weights_ols": weights_ols,
    "feature": weights_feature_based,
    "feature_based": weights_feature_based,
    "weights_feature_based": weights_feature_based,
    "pooled": weights_pooled_meta_model,
    "pooled_meta_model": weights_pooled_meta_model,
    "weights_pooled_meta_model": weights_pooled_meta_model,
    "meta": weights_meta_model,
    "meta_model": weights_meta_model,
    "weights_meta_model": weights_meta_model,
    "best_per_horizon": weights_best_per_horizon,
    "weights_best_per_horizon": weights_best_per_horizon,
    "topk_mean_per_horizon": weights_topk_per_horizon,
    "topk_per_horizon": weights_topk_per_horizon,
    "weights_topk_per_horizon": weights_topk_per_horizon,
    "forward": weights_forward,
    "forward_selection": weights_forward,
    "weights_forward": weights_forward,
}
_WEIGHT_METHOD_NAMES = tuple(sorted(_WEIGHTS_METHOD_TOOLS))


def _resolve_weights_shorthand(state: ReactState, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Translates the weight shorthands the agent writes into a valid spec.

    Measured on the v2 run: 270 of the failed `evaluate_strategy` calls were one
    of three near-misses, each costing an iteration and teaching nothing. This
    tool is already the permissive one, so it accepts what the model meant:

    * `weights` names a METHOD instead of a handle (`"inverse_error"`), computed
      on the pool the strategy asks for;
    * `weights` reuses a handle computed on ANOTHER pool: the same recipe is
      re-registered for the requested pool and re-fit under the protocol;
    * `combine` is the weights tool itself (`"weights_inverse_error"`,
      `"weights"`), read as `combine="weighted"` plus that method.

    Unknown names still raise, now listing both the handles and the methods.
    """
    combine = str(spec.get("combine", "")).strip()
    low = combine.lower()
    if low == "weights":
        spec["combine"] = "weighted"
    elif low.startswith("weights_") and low not in _WEIGHTS_METHOD_TOOLS:
        stripped = low[len("weights_"):]
        if stripped in _WEIGHTS_METHOD_TOOLS:
            spec["combine"] = "weighted"
            spec.setdefault("weights", stripped)
    elif low in _WEIGHTS_METHOD_TOOLS:
        spec["combine"] = "weighted"
        spec.setdefault("weights", low)

    if str(spec.get("combine", "")).lower() != "weighted":
        return spec

    pool = str(spec.get("pool") or FULL_POOL)
    raw = spec.get("weights")
    if raw is None:
        raise ValueError(
            "combine='weighted' needs 'weights' as a handle or a method name; "
            f"valid methods: {list(_WEIGHT_METHOD_NAMES)}"
        )
    handle = str(raw).strip()
    if handle in state.weights:
        recipe = state.get_weights_recipe(handle)
        if recipe.pool_handle != pool:
            spec["weights"] = state.register_weights(replace(recipe, pool_handle=pool))
        return spec

    tool = _WEIGHTS_METHOD_TOOLS.get(handle.lower())
    if tool is None:
        raise ValueError(
            f"weights {handle!r} is neither a registered handle {sorted(state.weights)} "
            f"nor a known method; valid methods: {list(_WEIGHT_METHOD_NAMES)}"
        )
    spec["weights"] = tool(state, pool=pool)["weights"]
    return spec


def evaluate_strategy(
    state: ReactState,
    strategy: Any = None,
    combine: Optional[str] = None,
    pool: Optional[str] = None,
    weights: Optional[str] = None,
    trim_pct: Optional[float] = None,
    model: Optional[str] = None,
    models: Optional[Any] = None,
    score_preset: Optional[str] = None,
    rationale: str = "",
    iteration: Optional[int] = None,
) -> Dict[str, Any]:
    """The only action that scores a strategy: builds it, backtests it on validation, adds it to the history.

    Accepts every shape an agent naturally reaches for, because a rejected call
    costs an iteration and teaches nothing:

        {"combine": "weighted", "pool": "pool1", "weights": "w1"}   flat
        {"strategy": {"combine": "weighted", "pool": "pool1", ...}} nested
        {"strategy": "weighted", "pool": "pool1", "weights": "w1"}  method + siblings
        {"strategy": <the whole dict combine_weighted returned>}    passthrough

    `weights` accepts the handle from a weights tool, a method name
    (`"inverse_error"`, `"forward"`, ...) which is computed on the strategy's own
    pool, or a handle from another pool, which is re-fit for this one; `combine`
    may also be the weights tool itself (`"weights_inverse_error"`).

    So `combine_*` is optional sugar: this one call both assembles and scores.
    The strategy enters the history with its rationale; re-submitting one already
    tested creates no new entry and comes back with `already_tested=True`.
    """
    spec: Dict[str, Any] = {}

    if isinstance(strategy, str):
        text = strategy.strip()
        parsed = None
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, dict):
            spec = dict(parsed)
        else:
            # a bare method name, or the human-readable label from combine_*
            spec = {"combine": text.split()[0].lower() if text else ""}
    elif isinstance(strategy, dict):
        spec = dict(strategy.get("strategy") if isinstance(strategy.get("strategy"), dict) else strategy)

    # sibling arguments fill in whatever the nested form did not carry
    for key, value in (
        ("combine", combine), ("pool", pool), ("weights", weights),
        ("trim_pct", trim_pct), ("model", model), ("models", models),
    ):
        if value is not None and not spec.get(key):
            spec[key] = value

    if not spec.get("combine"):
        raise ValueError(
            "no combination method given. Pass either "
            '{"combine": "mean", "pool": "pool_full"} or the object a combine_* tool returned'
        )

    is_best_single = str(spec.get("combine", "")).strip().lower() == "best_single"
    if is_best_single and not getattr(state.config, "allow_best_single", True):
        raise ValueError(
            "combine='best_single' is disabled: this run is combination-only; "
            "use at least two models"
        )
    ignored_args: List[str] = []
    raw_models = spec.pop("models", None)
    if raw_models is not None:
        if is_best_single:
            ignored_args.append("models")
        else:
            if spec.get("pool") and spec.get("pool") != FULL_POOL:
                raise ValueError("use either 'models' or a 'pool' handle, not both")
            names = [raw_models] if isinstance(raw_models, str) else list(raw_models)
            names = [str(name).strip() for name in names]
            unknown = [name for name in names if name not in state.model_names]
            if unknown:
                raise ValueError(f"unknown models {unknown}; available: {state.model_names}")
            if not names:
                raise ValueError("'models' cannot be empty")
            indices = [state.model_names.index(name) for name in names]
            spec["pool"] = state.register_pool(indices, origin="agent_explicit", models=names)
    if is_best_single and spec.get("pool") and spec["pool"] != FULL_POOL:
        ignored_args.append("pool")
    if not getattr(state.config, "allow_best_single", True):
        if raw_models is not None:
            names = [raw_models] if isinstance(raw_models, str) else list(raw_models)
            if len(names) < 2:
                raise ValueError("combination-only run: use at least two models")
        if spec.get("pool"):
            try:
                if len(state.get_pool(str(spec["pool"]))) < 2:
                    raise ValueError("combination-only run: use at least two models")
            except KeyError:
                pass
        elif raw_models is None:
            # full pool is always >= 2 models
            pass

    preset = score_preset if score_preset is not None else spec.pop("score_preset", None)
    if preset is not None:
        if not getattr(state.config, "allow_agent_score_preset", True):
            raise ValueError(
                "the scoring criterion is fixed for this run; do not pass score_preset"
            )
        if getattr(state.config, "multi_preset_final", False):
            raise ValueError(
                "the scoring preset is chosen only at the final decision in this run; "
                "evaluate_strategy returns all three scores and does not change them"
            )
        preset = str(preset).strip()
        if preset not in SCORE_PRESETS:
            raise ValueError(
                f"unknown score_preset {preset!r}; available: {list(SCORE_PRESETS)}"
            )
        state.apply_score_weights(SCORE_PRESETS[preset])

    spec = _resolve_weights_shorthand(state, spec)
    strategy = spec

    attempt, is_new = state.evaluate(
        strategy, rationale=str(rationale), origin="agent", iteration=iteration
    )
    ranked = state.ranked_attempts()
    best = ranked[0]
    position = ranked.index(attempt) + 1

    gap = None
    if best is not attempt and np.isfinite(best.score) and np.isfinite(attempt.score):
        base = abs(best.score) or 1.0
        gap = round(float((attempt.score - best.score) / base), 4)

    twin = state.numerical_twin(attempt)
    out_twin: Dict[str, Any] = {}
    if twin is not None:
        out_twin = {
            "numerically_identical_to": twin.attempt_id,
            "note": (
                f"this produces the SAME forecasts as {twin.attempt_id} "
                f"({_spec_label(twin.spec)}) — it adds nothing new. To explore a "
                "different point, change the POOL or the KIND of combination, not "
                "just the weighting method."
            ),
        }

    out: Dict[str, Any] = {
        "id": attempt.attempt_id,
        "strategy": attempt.spec,
        "already_tested": not is_new,
        **out_twin,
        "rank": position,
        "total_attempts": len(ranked),
        "metrics": {
            "rmse": round(float(attempt.aggregate["RMSE"]), 4),
            "smape": round(float(attempt.aggregate["SMAPE"]), 4),
            "mape": round(float(attempt.aggregate["MAPE"]), 4),
            "pocid": round(float(attempt.aggregate["POCID"]), 1),
        },
        "rmse_per_window": [round(float(w["RMSE"]), 4) for w in attempt.per_window],
        "score": round(float(attempt.score), 4),
        "current_best": {
            "id": best.attempt_id,
            "score": round(float(best.score), 4),
            "strategy": best.brief(include_rationale=False)["strategy"],
            "origin": best.origin,
        },
        "worse_than_best_by": gap,
        "is_best": bool(attempt is best),
    }
    if ignored_args:
        out["ignored_arguments"] = sorted(set(ignored_args))
        out["note"] = (
            "best_single uses ONLY 'model'. The "
            + " and ".join(sorted(set(ignored_args)))
            + " argument(s) were ignored; changing the pool/models does not create "
            "a new strategy. Change the 'model' to test another single model."
        )
    if getattr(state.config, "multi_preset_final", False):
        scores_by = state.preset_scores(attempt, FINAL_SCORE_PRESETS)
        board = state.preset_board(FINAL_SCORE_PRESETS, top_n=1)
        out["scores_by_preset"] = {
            name: (round(float(value), 4) if np.isfinite(value) else None)
            for name, value in scores_by.items()
        }
        out["rank_by_preset"] = {
            name: state.ranked_attempts_for_preset(name).index(attempt) + 1
            for name in FINAL_SCORE_PRESETS
        }
        out["best_by_preset"] = {
            name: board[name]["best"] for name in FINAL_SCORE_PRESETS
        }
        out["seed_best_by_preset"] = {
            name: board[name]["best_seed"] for name in FINAL_SCORE_PRESETS
        }
    return out


def sanity_check(state: ReactState, reference: Any) -> Dict[str, Any]:
    """Compares the strategy's test forecast against historical bounds.

    Blocks nothing — only flags. `reference` is an attempt id ("a3") or a strategy
    spec.
    """
    spec = reference
    if isinstance(reference, str) and reference.startswith("a"):
        match = [a for a in state.attempts if a.attempt_id == reference]
        if not match:
            raise KeyError(f"unknown attempt: {reference!r}")
        spec = match[0].spec
    elif isinstance(reference, dict) and "strategy" in reference:
        spec = reference["strategy"]

    forecast, _ = state.apply_to_test(spec)

    hist = state.train_series
    if hist is None or np.asarray(hist).size == 0:
        hist = state.y_true.reshape(-1)
    hist = np.asarray(hist, dtype=float)
    hist = hist[np.isfinite(hist)]

    lo, hi = float(np.min(hist)), float(np.max(hist))
    med, sd = float(np.median(hist)), float(np.std(hist))
    tol = float(state.config.sanity_check_tolerance)
    band = (med - tol * sd, med + tol * sd)

    outside_hist = int(np.sum((forecast < lo) | (forecast > hi)))
    outside_band = int(np.sum((forecast < band[0]) | (forecast > band[1])))
    non_finite = int(np.sum(~np.isfinite(forecast)))

    # Extrapolating the historical range is normal for a trending series, so that
    # is information, not a warning. The warning is the robust band around the median.
    warnings: List[str] = []
    if non_finite:
        warnings.append(f"{non_finite} non-finite points")
    if outside_band:
        warnings.append(f"{outside_band} points beyond {tol} std of the historical median")

    return {
        "n_points": int(forecast.size),
        "forecast_range": [
            round(float(np.nanmin(forecast)), 3),
            round(float(np.nanmax(forecast)), 3),
        ],
        "historical_range": [round(lo, 3), round(hi, 3)],
        "extrapolates_history": bool(outside_hist > 0),
        "points_outside_history": outside_hist,
        "points_outside_band": outside_band,
        "warnings": warnings,
        "ok": not warnings,
    }


def list_attempts(state: ReactState, top_n: int = 10) -> Dict[str, Any]:
    """Ranked attempt history (best to worst)."""
    ranked = state.ranked_attempts()
    show = int(np.clip(int(top_n), 1, 50))
    include = bool(state.config.show_attempt_rationales)
    return {
        "total": len(ranked),
        "ranking": [a.brief(include_rationale=include) for a in ranked[:show]],
        "best": ranked[0].attempt_id if ranked else None,
    }
