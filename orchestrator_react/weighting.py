"""Weight recipes — always computed in code, never typed by the LLM.

The agent chooses *which* recipe to apply and over which pool; the numbers come
from here and are referenced by a handle (`w1`, `w2`, ...). The agent never sees
raw values — only the handle and a qualitative summary (concentration, number of
active models).

Every recipe receives:
    y_true : (n_fit, horizon)
    y_pool : (n_fit, n_pool, horizon)
and returns `(n_pool,)` or, when `per_horizon=True`, `(n_pool, horizon)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12
WEIGHT_METHODS = (
    "inverse_error",
    "softmax_neg_error",
    "error_trend",
    "ols",
    "feature_based",
    "pooled_meta_model",
)
ERROR_METRICS = ("rmse", "mae", "smape")


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1}."""
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n == 0:
        return v
    if not np.all(np.isfinite(v)) or np.all(v <= 0):
        return np.ones(n) / n
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
    if rho_idx.size == 0:
        return np.ones(n) / n
    rho = rho_idx[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def _uniform(n: int) -> np.ndarray:
    return np.ones(int(n), dtype=float) / max(1, int(n))


def per_model_error(
    y_true: np.ndarray,
    y_pool: np.ndarray,
    metric: str = "rmse",
    per_horizon: bool = False,
) -> np.ndarray:
    """Error of each pool model. Shape `(n_pool,)` or `(n_pool, horizon)`."""
    y_true = np.asarray(y_true, dtype=float)
    y_pool = np.asarray(y_pool, dtype=float)
    if y_pool.ndim != 3:
        raise ValueError(f"y_pool must be (n_fit, n_pool, horizon), got {y_pool.shape}")
    diff = y_pool - y_true[:, None, :]
    metric = str(metric).lower()

    if metric == "smape":
        denom = np.abs(y_pool) + np.abs(y_true[:, None, :])
        denom = np.where(denom == 0, np.nan, denom)
        pointwise = 2.0 * np.abs(diff) / denom
    elif metric == "mae":
        pointwise = np.abs(diff)
    elif metric == "rmse":
        pointwise = diff**2
    else:
        raise ValueError(f"unknown error metric: {metric!r} (valid: {ERROR_METRICS})")

    axis: Tuple[int, ...] = (0,) if per_horizon else (0, 2)
    with np.errstate(invalid="ignore"):
        agg = np.nanmean(pointwise, axis=axis)
    if metric == "rmse":
        agg = np.sqrt(agg)
    return np.nan_to_num(np.asarray(agg, dtype=float), nan=np.inf, posinf=np.inf)


# ──────────────────────────────────────────────────────────────────────────────
# recipes
# ──────────────────────────────────────────────────────────────────────────────


def weights_inverse_error(
    y_true: np.ndarray,
    y_pool: np.ndarray,
    metric: str = "rmse",
    power: float = 1.0,
    shrinkage: float = 0.0,
    per_horizon: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """w_m proportional to 1 / error_m^power, shrunk toward uniform weights."""
    err = per_model_error(y_true, y_pool, metric=metric, per_horizon=per_horizon)
    inv = 1.0 / np.power(err + eps, float(power))
    inv = np.where(np.isfinite(inv), inv, 0.0)

    def _norm(col: np.ndarray) -> np.ndarray:
        s = col.sum()
        w = col / s if s > 0 else _uniform(col.size)
        if shrinkage > 0:
            w = (1.0 - shrinkage) * w + shrinkage * _uniform(col.size)
        return project_simplex(w)

    if inv.ndim == 1:
        return _norm(inv)
    return np.stack([_norm(inv[:, h]) for h in range(inv.shape[1])], axis=1)


def weights_softmax_neg_error(
    y_true: np.ndarray,
    y_pool: np.ndarray,
    metric: str = "rmse",
    eta: float = 1.0,
    per_horizon: bool = False,
    normalize_scale: bool = True,
) -> np.ndarray:
    """w_m proportional to exp(-eta * error_m) — the form used by ADE.

    With `normalize_scale=True` the error is divided by its median before the
    softmax. Without it `eta` would carry a scale-dependent meaning, and the same
    `eta=1.0` would give uniform weights on a series in the millions and one-hot
    weights on a series in the units.
    """
    err = per_model_error(y_true, y_pool, metric=metric, per_horizon=per_horizon)

    def _soft(col: np.ndarray) -> np.ndarray:
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            return _uniform(col.size)
        scale = float(np.median(finite)) if normalize_scale else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        z = -float(eta) * (col / scale)
        z = np.where(np.isfinite(z), z, -np.inf)
        z = z - np.max(z[np.isfinite(z)], initial=0.0)
        e = np.exp(z)
        s = e.sum()
        return e / s if s > 0 else _uniform(col.size)

    if err.ndim == 1:
        return _soft(err)
    return np.stack([_soft(err[:, h]) for h in range(err.shape[1])], axis=1)


def weights_error_trend(
    y_true: np.ndarray,
    y_pool: np.ndarray,
    metric: str = "mae",
    eta: float = 1.0,
    damping: Optional[float] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """w proportional to exp(-eta * *extrapolated* error — where each model is
    heading, not where it has been on average.

    The other recipes collapse each validation window into one number, which on a
    three-window protocol leaves three observations per model. This one keeps the
    pointwise error grid `(n_fit, horizon)` — 24 numbers on NN5 — and asks a
    different question: is this model getting better or worse over the windows?

    Two confounders have to be separated, or the answer is meaningless:

    * **The horizon profile.** Step 8 is harder than step 1 for everyone. Fitting a
      slope over the concatenated error series would read that ramp as degradation.
      So a slope is fitted *per horizon step*, across windows, and the model-level
      slope is the median of those — which is also what turns `horizon` noisy
      three-point fits into one usable estimate.
    * **Slope noise.** Three windows make any single slope unreliable. `damping`
      scales the extrapolation; left at `None` it is derived from how much the
      per-step slopes agree on a direction (all agree -> 1.0, coin flip -> 0.0),
      so a trend is only trusted to the extent the data shows one.

    Returns `(weights, meta)`; `meta["mode"]` says which regime actually ran.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pool = np.asarray(y_pool, dtype=float)
    n_fit, n_pool, horizon = y_pool.shape

    # Fewer than three windows cannot support a slope and an intercept with any
    # residual left to judge them by: fall back rather than extrapolate noise.
    if n_fit < 3:
        w = weights_softmax_neg_error(y_true, y_pool, metric=metric, eta=eta)
        return w, {
            "mode": "softmax_neg_error_fallback",
            "reason": f"needs at least 3 windows to fit a trend, got {n_fit}",
        }

    grid = _pointwise_error_grid(y_true, y_pool, metric)  # (n_fit, n_pool, horizon)

    w_index = np.arange(n_fit, dtype=float)
    centred = w_index - w_index.mean()
    denom = float(np.sum(centred**2)) or 1.0

    # slopes[m, h]: least squares slope of that model's error at horizon step h
    # across the windows. NaNs (a metric can leave them) propagate as 0 slope.
    with np.errstate(invalid="ignore"):
        slopes = np.einsum("w,wmh->mh", centred, np.nan_to_num(grid, nan=0.0)) / denom

    model_slope = np.nanmedian(slopes, axis=1)  # (n_pool,)

    # Level = the most recent window, not the average: the point of the recipe is
    # to start from where the model *is* and move along its own trend.
    with np.errstate(invalid="ignore"):
        level = np.nanmean(grid[-1], axis=1)  # (n_pool,)

    if damping is None:
        # Adaptive: how much do the per-step slopes agree on a direction? A model
        # whose slopes are half up and half down has no trend worth extrapolating.
        sign = np.sign(slopes)
        with np.errstate(invalid="ignore"):
            agree = np.nanmean(sign == np.sign(model_slope)[:, None], axis=1)
        damp = np.clip(2.0 * (agree - 0.5), 0.0, 1.0)
        damp_mode = "adaptive"
    else:
        damp = np.full(n_pool, float(np.clip(damping, 0.0, 1.0)))
        damp_mode = "fixed"

    predicted = level + damp * model_slope
    # An extrapolated error may go negative; a negative error is not a stronger
    # claim than a near-zero one, so floor it at a small share of the level.
    floor = 0.05 * np.nanmax(np.abs(level)) if np.any(np.isfinite(level)) else eps
    predicted = np.maximum(predicted, max(float(floor), eps))
    predicted = np.where(np.isfinite(predicted), predicted, np.inf)

    if not np.any(np.isfinite(predicted)):
        return _uniform(n_pool), {"mode": "uniform_no_finite_error"}

    scale = float(np.nanmedian(predicted[np.isfinite(predicted)])) or 1.0
    z = -float(eta) * (predicted / scale)
    z = np.where(np.isfinite(z), z, -np.inf)
    z = z - np.max(z[np.isfinite(z)], initial=0.0)
    e = np.exp(z)
    s = e.sum()
    w = e / s if s > 0 else _uniform(n_pool)

    finite_slope = model_slope[np.isfinite(model_slope)]
    return project_simplex(w), {
        "mode": "error_trend",
        "damping": damp_mode,
        "mean_damping": round(float(np.nanmean(damp)), 3),
        "n_worsening": int(np.sum(finite_slope > 0)),
        "n_improving": int(np.sum(finite_slope < 0)),
        "n_points_per_model": int(n_fit * horizon),
    }


def _pointwise_error_grid(
    y_true: np.ndarray, y_pool: np.ndarray, metric: str
) -> np.ndarray:
    """`(n_fit, n_pool, horizon)` of pointwise error, un-aggregated.

    `per_model_error` reduces over windows and horizon; the trend recipe needs the
    grid before that reduction, which is the whole information advantage.
    """
    diff = y_pool - y_true[:, None, :]
    metric = str(metric).lower()
    if metric == "smape":
        denom = np.abs(y_pool) + np.abs(y_true[:, None, :])
        denom = np.where(denom == 0, np.nan, denom)
        return 2.0 * np.abs(diff) / denom
    if metric in ("mae", "rmse"):
        # Absolute error in both cases: squaring before a slope fit would let one
        # bad point in one window dictate the direction.
        return np.abs(diff)
    raise ValueError(f"unknown error metric: {metric!r} (valid: {ERROR_METRICS})")


def weights_ols(
    y_true: np.ndarray,
    y_pool: np.ndarray,
    l2: float = 0.0,
    nonneg: bool = True,
    per_horizon: bool = False,
) -> np.ndarray:
    """Least-squares weights (Granger-Ramanathan), optionally on the simplex.

    `l2 > 0` turns it into ridge. With `nonneg=True` the solution is projected onto
    the simplex — recommended with few windows, where unconstrained OLS tends to
    produce huge negative weights that do not generalise (forecast combination
    puzzle).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pool = np.asarray(y_pool, dtype=float)
    n_pool = y_pool.shape[1]

    def _solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[mask], y[mask]
        if X.shape[0] < 2:
            return _uniform(n_pool)
        A = X.T @ X + float(l2) * np.eye(n_pool)
        b = X.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b
        if not np.all(np.isfinite(w)):
            return _uniform(n_pool)
        if nonneg:
            return project_simplex(w)
        s = w.sum()
        return w / s if abs(s) > EPS else _uniform(n_pool)

    if not per_horizon:
        X = np.transpose(y_pool, (0, 2, 1)).reshape(-1, n_pool)
        return _solve(X, y_true.reshape(-1))
    return np.stack(
        [_solve(y_pool[:, :, h], y_true[:, h]) for h in range(y_true.shape[1])], axis=1
    )


def weights_feature_based(
    y_true: np.ndarray,
    y_pool: np.ndarray,
    metric: str = "smape",
    eta: float = 1.0,
    n_estimators: int = 40,
    max_depth: int = 2,
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Lightweight meta-model in the spirit of FFORMA, with a documented fallback.

    Trains one gradient-boosted regressor per model, mapping window features
    (level, dispersion, slope, autocorrelation) to that model's error, then turns
    the predicted errors into weights via `softmax(-error)` — FFORMA's own final
    step.

    With 3 validation windows the sample is tiny. So if XGBoost is unavailable
    **or** there are fewer samples than the meta-model needs, it falls back to
    `softmax(-mean error)`, which is exactly the documented fallback of the
    project's own FFORMA (`combinations/fforma.py::_compute_weights_softmax`). The
    mode actually used is returned in the metadata dict, for the CSV field
    `weights_handle_resolved`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pool = np.asarray(y_pool, dtype=float)
    n_fit, n_pool, _ = y_pool.shape

    err_per_window = np.stack(
        [per_model_error(y_true[i : i + 1], y_pool[i : i + 1], metric=metric) for i in range(n_fit)]
    )  # (n_fit, n_pool)

    def _softmax_fallback(reason: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        w = weights_softmax_neg_error(y_true, y_pool, metric=metric, eta=eta)
        return w, {"mode": "softmax_fallback", "reason": reason, "n_windows": int(n_fit)}

    if n_fit < 3:
        return _softmax_fallback(f"only {n_fit} fitting windows")

    try:
        from xgboost import XGBRegressor
    except Exception:
        return _softmax_fallback("xgboost unavailable")

    feats = np.stack([_window_features(y_true[i]) for i in range(n_fit)])
    if n_fit < 2 * feats.shape[1]:
        return _softmax_fallback(f"sample too small ({n_fit} windows x {feats.shape[1]} features)")

    predicted = np.zeros(n_pool, dtype=float)
    try:
        for j in range(n_pool):
            target = err_per_window[:, j]
            if not np.all(np.isfinite(target)):
                return _softmax_fallback("non-finite target for the meta-model")
            model = XGBRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=0.1,
                random_state=int(random_state),
                verbosity=0,
            )
            model.fit(feats, target)
            predicted[j] = float(np.mean(model.predict(feats)))
    except Exception as exc:  # pragma: no cover - environment dependent
        return _softmax_fallback(f"meta-model training failed: {type(exc).__name__}")

    scale = float(np.median(predicted[np.isfinite(predicted)])) or 1.0
    z = -float(eta) * (predicted / scale)
    z -= np.max(z)
    e = np.exp(z)
    w = e / e.sum() if e.sum() > 0 else _uniform(n_pool)
    return w, {"mode": "xgboost", "n_features": int(feats.shape[1]), "n_samples": int(n_fit)}


def _window_features(y: np.ndarray) -> np.ndarray:
    """Cheap window characteristics — input to the meta-model."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return np.zeros(5, dtype=float)
    level = float(np.mean(y))
    disp = float(np.std(y))
    slope = float(np.polyfit(np.arange(y.size), y, 1)[0]) if y.size >= 2 else 0.0
    scale = abs(level) + EPS
    if y.size >= 3 and np.std(y) > EPS:
        c = np.corrcoef(y[:-1], y[1:])[0, 1]
        acf1 = float(c) if np.isfinite(c) else 0.0
    else:
        acf1 = 0.0
    return np.array([level / scale, disp / scale, slope / scale, acf1, float(y.size)], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# handle
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class WeightsRecipe:
    """Reproducible description of how weights are obtained.

    We store the *recipe*, not the numbers: during the backtest it is refit per
    window under the anti-leakage protocol, and in the final application it is fit
    on the requested windows. The resolved numbers live in `resolved`, for the CSV.
    """

    method: str
    pool_handle: str
    fit_windows: Optional[Tuple[int, ...]] = None  # None => all available
    per_horizon: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
    resolved: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def spec(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "pool": self.pool_handle,
            "fit_windows": list(self.fit_windows) if self.fit_windows is not None else "all",
            "per_horizon": bool(self.per_horizon),
            "params": dict(self.params),
        }


def resolve_recipe(
    recipe: WeightsRecipe, y_true: np.ndarray, y_pool: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs the recipe over the given windows. It does not decide which windows
    those are: `state.py` owns the anti-leakage protocol."""
    method = str(recipe.method).lower()
    p = dict(recipe.params)
    n_pool = int(y_pool.shape[1])

    if y_pool.shape[0] == 0:
        return _uniform(n_pool), {"mode": "uniform_no_fit_data"}

    if method == "pooled_meta_model":
        # Deliberately ignores `y_true`/`y_pool`: the weights were already
        # computed once, from this series' own historical shape (trend/seasonal
        # strength, entropy, autocorrelation) queried against a model trained on
        # every OTHER series in the dataset. None of those four features change
        # per backtest fold — they never depended on a specific validation
        # window — so the same vector is correct for every fold. The tool that
        # registers this recipe (`tools.weights_pooled_meta_model`) refuses a pool
        # whose membership can vary per fold under `nested_selection`, which is
        # what keeps this shortcut sound: the vector's length can never disagree
        # with `y_pool.shape[1]` at resolution time.
        w = np.asarray(p.get("precomputed_weights", []), dtype=float)
        if w.size != n_pool:
            return _uniform(n_pool), {"mode": "pooled_meta_model_pool_mismatch"}
        return w, {"mode": "pooled_meta_model"}

    if method == "inverse_error":
        w = weights_inverse_error(
            y_true, y_pool,
            metric=p.get("metric", "rmse"),
            power=float(p.get("power", 1.0)),
            shrinkage=float(p.get("shrinkage", 0.0)),
            per_horizon=recipe.per_horizon,
        )
        return w, {"mode": "inverse_error"}

    if method == "softmax_neg_error":
        w = weights_softmax_neg_error(
            y_true, y_pool,
            metric=p.get("metric", "rmse"),
            eta=float(p.get("eta", 1.0)),
            per_horizon=recipe.per_horizon,
        )
        return w, {"mode": "softmax_neg_error"}

    if method == "error_trend":
        return weights_error_trend(
            y_true, y_pool,
            metric=p.get("metric", "mae"),
            eta=float(p.get("eta", 1.0)),
            damping=p.get("damping", None),
        )

    if method == "ols":
        w = weights_ols(
            y_true, y_pool,
            l2=float(p.get("l2", 0.0)),
            nonneg=bool(p.get("nonneg", True)),
            per_horizon=recipe.per_horizon,
        )
        return w, {"mode": "ols"}

    if method == "feature_based":
        return weights_feature_based(
            y_true, y_pool,
            metric=p.get("metric", "smape"),
            eta=float(p.get("eta", 1.0)),
        )

    raise ValueError(f"unknown weight method: {method!r} (valid: {WEIGHT_METHODS})")


def summarize_weights(w: np.ndarray, model_names: Sequence[str]) -> Dict[str, Any]:
    """Compact summary for the agent — never exposes the raw numbers."""
    w = np.asarray(w, dtype=float)
    flat = w.mean(axis=1) if w.ndim == 2 else w
    order = np.argsort(flat)[::-1]
    total = float(flat.sum()) or 1.0
    top = [
        {"model": str(model_names[i]), "share_pct": round(100.0 * float(flat[i]) / total, 1)}
        for i in order[:3]
    ]
    active = int(np.sum(flat > 0.01))
    # Normalised Herfindahl index: 0 = uniform, 1 = everything on one model.
    n = flat.size
    hhi = float(np.sum((flat / total) ** 2))
    concentration = (hhi - 1.0 / n) / (1.0 - 1.0 / n) if n > 1 else 1.0
    return {
        "n_models": int(n),
        "n_active": active,
        "concentration": round(float(np.clip(concentration, 0.0, 1.0)), 3),
        "top3": top,
        "per_horizon": bool(w.ndim == 2),
    }
