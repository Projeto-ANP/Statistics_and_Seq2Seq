"""Deterministic series profiling (Section 3.4.1).

Everything here is written from scratch for this architecture. In particular the
trend/seasonality "champions" — decided by an LLM agent (PatternAnalyst) in the old
architecture — become a deterministic function: Pearson correlation between the
model's STL component and the STL component of `y_true`, averaged across windows,
then `argmax`. No LLM, no prompt, fully reproducible.

Optional dependencies degrade with a flag, never with an exception:
    statsmodels -> STL and ADF/KPSS; without it, linear detrend and null p-values
    pycatch22   -> catch22 set; without it, the fast subset defined here
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12

#: Seasonal period per dataset frequency, as declared by `@frequency` in the `.tsf`.
#: This is the authoritative source — the frequency is a property of the dataset,
#: not something to be guessed from the data.
FREQ_TO_PERIOD = {
    "yearly": 1,
    "quarterly": 4,
    "monthly": 12,
    "weekly": 52,
    "daily": 7,
    "hourly": 24,
    "half_hourly": 48,
    "30min": 48,
    "15min": 96,
    "10_minutes": 144,
    "minutely": 60,
}


def resolve_seasonal_period(
    freq: str, n_points: int, horizon: int = 0, explicit: Optional[int] = None
) -> Dict[str, Any]:
    """Resolves the seasonal period and says exactly where it came from.

    Never silently invents a cycle. When the declared period does not fit in the
    available sample (STL needs at least two full cycles), the period is reported
    as not fitting rather than being shrunk into something the data does not have:
    forcing period=12 on weekly data would manufacture a yearly cycle out of noise.

    Returns:
        period          - the period to use downstream
        declared        - what the `.tsf` frequency implies (None if unknown)
        source          - "explicit" | "frequency" | "horizon_fallback"
        fits            - whether `n_points >= 2 * period`, i.e. STL is meaningful
        n_points        - sample size the decision was made against
    """
    n_points = int(n_points or 0)
    key = str(freq or "").strip().lower()
    declared = FREQ_TO_PERIOD.get(key)

    if explicit is not None and int(explicit) >= 2:
        period, source = int(explicit), "explicit"
    elif declared is not None and declared >= 2:
        period, source = int(declared), "frequency"
    else:
        # Unknown or non-seasonal frequency: fall back to the forecast horizon,
        # which is the only other scale the problem gives us.
        period, source = max(2, min(int(horizon) or 12, 12)), "horizon_fallback"

    return {
        "period": int(period),
        "declared": declared,
        "frequency": key or "unknown",
        "source": source,
        "fits": bool(n_points >= 2 * period),
        "n_points": n_points,
    }


def infer_seasonal_period(freq: str, n_points: int, horizon: int) -> int:
    """Backwards-compatible shorthand: just the period to feed STL.

    Unlike `resolve_seasonal_period`, this caps the period so STL can always run.
    Use it for whole-series decomposition, where a capped period is harmless;
    use `resolve_seasonal_period` when the caller needs to know whether the real
    seasonal cycle is estimable at all.
    """
    info = resolve_seasonal_period(freq, n_points, horizon)
    period = info["period"]
    if n_points and 2 * period > n_points:
        period = max(2, int(n_points // 2))
    return int(period)


def _clean(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


# ──────────────────────────────────────────────────────────────────────────────
# decomposition
# ──────────────────────────────────────────────────────────────────────────────


#: Filled by `stl_decompose` with the method actually used on the last call:
#: "stl" or "linear_fallback:<reason>". Read by `stl_summary`/`series_profile` so
#: the CSV records whether the decomposition was real or degraded.
LAST_DECOMPOSITION = {"method": "not_run"}


def stl_decompose(series: Sequence[float], period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust STL. Falls back to linear detrend (seasonal = 0) on short series.

    On the fallback path `seasonal` is identically zero, so seasonal strength will
    read 0 — honest, but it must be distinguished from "the series has no
    seasonality". Check `LAST_DECOMPOSITION["method"]` to know which path was taken.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    if n < 2:
        z = np.zeros_like(x)
        LAST_DECOMPOSITION["method"] = "linear_fallback:too_few_points"
        return x.copy(), z, z

    valid = np.isfinite(x)
    n_valid = int(valid.sum())
    p = max(2, int(period))
    idx = np.arange(n, dtype=float)

    def _linear(reason: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        LAST_DECOMPOSITION["method"] = f"linear_fallback:{reason}"
        if n_valid >= 2:
            trend = np.polyval(np.polyfit(idx[valid], x[valid], 1), idx)
        else:
            trend = np.full(n, float(np.nanmean(x)) if n_valid else 0.0)
        return trend, np.zeros(n), x - trend

    if n_valid < 2 * p or n_valid < 4:
        return _linear("short_series")

    try:
        from statsmodels.tsa.seasonal import STL
    except Exception:
        return _linear("statsmodels_unavailable")

    try:
        filled = x.copy()
        if not valid.all():
            filled[~valid] = float(np.nanmean(x[valid]))
        res = STL(filled, period=p, robust=True).fit()
        LAST_DECOMPOSITION["method"] = "stl"
        return (
            np.asarray(res.trend, dtype=float),
            np.asarray(res.seasonal, dtype=float),
            np.asarray(res.resid, dtype=float),
        )
    except Exception:
        return _linear("stl_failed")


def stl_strengths(trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray) -> Dict[str, float]:
    """Trend and seasonal strength (Wang, Smith & Hyndman 2006), in [0, 1]."""

    def _strength(component: np.ndarray) -> float:
        var_r = float(np.nanvar(resid))
        var_cr = float(np.nanvar(component + resid))
        if var_cr <= EPS:
            return 0.0
        return float(np.clip(1.0 - var_r / var_cr, 0.0, 1.0))

    return {
        "trend_strength": round(_strength(trend), 4),
        "seasonal_strength": round(_strength(seasonal), 4),
    }


def variance_shares(trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray) -> Dict[str, float]:
    """Percentage of variance attributed to each component (normalised to 100)."""
    parts = {
        "trend_pct": float(np.nanvar(trend)),
        "seasonal_pct": float(np.nanvar(seasonal)),
        "residual_pct": float(np.nanvar(resid)),
    }
    total = sum(parts.values())
    if total <= EPS:
        return {k: 0.0 for k in parts}
    return {k: round(100.0 * v / total, 2) for k, v in parts.items()}


# ──────────────────────────────────────────────────────────────────────────────
# stationarity and outliers
# ──────────────────────────────────────────────────────────────────────────────


def stationarity(series: Sequence[float]) -> Dict[str, Any]:
    """ADF and KPSS. Without statsmodels, or on short series, p-values are null.

    The `reliable` flag marks when the sample is too small to take the p-values
    seriously — an ADF on 40 points decides nothing.
    """
    x = _clean(series)
    n = int(x.size)
    out: Dict[str, Any] = {
        "n": n,
        "adf_pvalue": None,
        "kpss_pvalue": None,
        "verdict": "undetermined",
        "reliable": bool(n >= 50),
    }
    if n < 12:
        return out
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except Exception:
        out["verdict"] = "statsmodels_unavailable"
        return out

    try:
        out["adf_pvalue"] = round(float(adfuller(x, autolag="AIC")[1]), 4)
    except Exception:
        pass
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["kpss_pvalue"] = round(float(kpss(x, regression="c", nlags="auto")[1]), 4)
    except Exception:
        pass

    adf, kp = out["adf_pvalue"], out["kpss_pvalue"]
    if adf is not None and kp is not None:
        adf_says_stationary = adf < 0.05  # rejects unit root
        kpss_says_stationary = kp > 0.05  # fails to reject stationarity
        if adf_says_stationary and kpss_says_stationary:
            out["verdict"] = "stationary"
        elif not adf_says_stationary and not kpss_says_stationary:
            out["verdict"] = "non_stationary"
        else:
            out["verdict"] = "ambiguous"
    return out


def outlier_flags(series: Sequence[float], k: float = 3.0) -> Dict[str, Any]:
    """Outliers via robust IQR fences."""
    x = _clean(series)
    if x.size < 4:
        return {"n_outliers": 0, "pct": 0.0, "max_z": None}
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= EPS:
        return {"n_outliers": 0, "pct": 0.0, "max_z": None}
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (x < lo) | (x > hi)
    center = float(np.median(x))
    z = np.abs(x - center) / (iqr + EPS)
    return {
        "n_outliers": int(mask.sum()),
        "pct": round(100.0 * float(mask.mean()), 2),
        "max_z": round(float(np.max(z)), 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# fast features (catch22 substitute / complement)
# ──────────────────────────────────────────────────────────────────────────────


def _acf(x: np.ndarray, lag: int) -> float:
    if x.size <= lag + 1:
        return 0.0
    a, b = x[:-lag], x[lag:]
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def spectral_entropy(series: Sequence[float]) -> float:
    """0 = concentrated spectrum (predictable); 1 = flat spectrum (white noise)."""
    x = _clean(series)
    if x.size < 8:
        return float("nan")
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x))[1:] ** 2
    total = float(power.sum())
    if total <= 0:
        return float("nan")
    p = power / total
    p = p[p > 0]
    h_max = float(np.log(p.size))
    return float(-np.sum(p * np.log(p)) / h_max) if h_max > 0 else float("nan")


def hurst_exponent(series: Sequence[float]) -> float:
    """~0.5 random walk; >0.5 persistent; <0.5 mean-reverting."""
    x = _clean(series)
    n = x.size
    if n < 32:
        return float("nan")
    lags = np.unique(np.round(np.logspace(np.log10(2), np.log10(n // 2), 10)).astype(int))
    lags = lags[lags >= 2]
    tau: List[float] = []
    used: List[int] = []
    for lag in lags:
        d = x[lag:] - x[:-lag]
        if d.size == 0:
            continue
        tau.append(float(np.sqrt(np.mean(d**2))) + EPS)
        used.append(int(lag))
    if len(tau) < 3:
        return float("nan")
    slope = np.polyfit(np.log(used), np.log(tau), 1)[0]
    return float(slope)


def fast_features(series: Sequence[float], period: int) -> Dict[str, Any]:
    """Fast subset in the spirit of catch22: autocorrelation, distribution,
    outliers and fluctuation scale."""
    x = _clean(series)
    if x.size < 4:
        return {}
    scale = float(np.std(x)) + EPS
    diffs = np.diff(x)
    return {
        "acf1": round(_acf(x, 1), 4),
        "acf_seasonal": round(_acf(x, max(1, int(period))), 4),
        "acf_diff1": round(_acf(diffs, 1), 4) if diffs.size > 2 else 0.0,
        "spectral_entropy": round(float(spectral_entropy(x)), 4),
        "hurst": round(float(hurst_exponent(x)), 4) if x.size >= 32 else None,
        "skewness": round(float(((x - x.mean()) ** 3).mean() / scale**3), 4),
        "kurtosis": round(float(((x - x.mean()) ** 4).mean() / scale**4), 4),
        "coef_variation": round(float(scale / (abs(float(x.mean())) + EPS)), 4),
        "fluctuation_scale": round(float(np.std(diffs) / scale), 4) if diffs.size else 0.0,
        "crosses_zero": bool(np.any(x < 0) and np.any(x > 0)),
    }


def catch22_features(series: Sequence[float]) -> Optional[Dict[str, float]]:
    """catch22 via `pycatch22`, when installed. `None` means it is not."""
    x = _clean(series)
    if x.size < 20:
        return None
    try:
        import pycatch22
    except Exception:
        return None
    try:
        res = pycatch22.catch22_all(x.tolist(), catch24=False)
        return {
            str(k): round(float(v), 5)
            for k, v in zip(res["names"], res["values"])
            if np.isfinite(v)
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# component champions (new, deterministic)
# ──────────────────────────────────────────────────────────────────────────────


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return 0.0
    av, bv = a[mask], b[mask]
    if np.std(av) < EPS or np.std(bv) < EPS:
        return 0.0
    c = np.corrcoef(av, bv)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _linear_parts(y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Least-squares slope and the detrended residual of a short window."""
    y = np.asarray(y, dtype=float)
    idx = np.arange(y.size, dtype=float)
    valid = np.isfinite(y)
    if int(valid.sum()) < 2:
        return 0.0, np.zeros_like(y)
    coeffs = np.polyfit(idx[valid], y[valid], 1)
    return float(coeffs[0]), y - np.polyval(coeffs, idx)


def component_champions(
    y_true: np.ndarray,
    y_preds: np.ndarray,
    model_names: Sequence[str],
    freq: str = "",
    horizon: int = 0,
    explicit_period: Optional[int] = None,
    contiguous_windows: bool = True,
) -> Dict[str, Any]:
    """Which model best tracks the trend, and which best tracks the seasonal shape.

    Replaces — with a new, LLM-free implementation — what the old
    `pattern_analyst_trend_champion` / `_seas_champion` fields carried.

    **The decomposition runs on the concatenated validation windows, not on one
    window at a time.** The windows tile the original series contiguously (verified
    at ingestion), so concatenating gives a genuine stretch of `n_windows * horizon`
    points — 36 for ANP, 72 for ETTh — instead of the `horizon` points a single
    window offers. That matters: with only `horizon` points and a monthly period of
    12, STL cannot run at all, the components degrade to a straight line and a zero
    vector, and correlating those is meaningless (two lines always correlate at
    +/-1, two zero vectors at 0), so `argmax` would return the first model on every
    series.

    The seasonal period comes from the dataset's `@frequency`, which is declared in
    the `.tsf` — monthly => 12, hourly => 24, half-hourly => 48, and so on. It is
    never guessed from the data.

    Two regimes:

    * the declared period fits in the concatenated sample (`n >= 2 * period`) —
      real STL, and the champions are the models whose trend/seasonal components
      correlate best with those of `y_true`;
    * it does not fit (weekly data with period 52 has only 24-39 points across three
      windows) — no honest seasonal estimate exists at that scale, so scale-free
      surrogates are used instead: slope agreement for the trend, and correlation of
      the **detrended residuals** for the seasonal shape.

    `component_method` records which regime ran and `seasonal_period` what was used,
    so a reader of `series_profile_json` can tell the two apart.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_preds = np.asarray(y_preds, dtype=float)
    n_windows, n_models, window_len = y_preds.shape

    if contiguous_windows and n_windows > 1:
        truth = y_true.reshape(-1)
        preds = y_preds.transpose(1, 0, 2).reshape(n_models, -1)
        layout = "concatenated_windows"
    else:
        truth = y_true[-1]
        preds = y_preds[-1]
        layout = "last_window_only"

    n_points = int(truth.size)
    period_info = resolve_seasonal_period(
        freq, n_points, horizon=horizon or window_len, explicit=explicit_period
    )
    period = int(period_info["period"])
    use_stl = bool(period_info["fits"]) and n_points >= 8

    if use_stl:
        t_true, s_true, _ = stl_decompose(truth, period)
        decomposition = LAST_DECOMPOSITION["method"]
        if decomposition != "stl":
            # STL declined even though the arithmetic said it should fit.
            use_stl = False

    if use_stl:
        trend_scores = np.array([_pearson(stl_decompose(preds[j], period)[0], t_true) for j in range(n_models)])
        shape_scores = np.array([_pearson(stl_decompose(preds[j], period)[1], s_true) for j in range(n_models)])
        method = "stl_on_concatenated_windows" if layout == "concatenated_windows" else "stl_on_last_window"
        trend_metric, shape_metric = "stl_trend_corr", "stl_seasonal_corr"
    else:
        slope_true, resid_true = _linear_parts(truth)
        scale = abs(slope_true) + float(np.nanstd(truth)) / max(1, n_points) + EPS
        trend_scores = np.zeros(n_models)
        shape_scores = np.zeros(n_models)
        for j in range(n_models):
            slope_m, resid_m = _linear_parts(preds[j])
            # 1 = identical slope, decaying toward 0 as the slopes diverge.
            trend_scores[j] = 1.0 / (1.0 + abs(slope_m - slope_true) / scale)
            shape_scores[j] = _pearson(resid_m, resid_true)
        method = "linear_detrend_surrogate"
        trend_metric, shape_metric = "slope_agreement", "detrended_residual_corr"

    def _champion(scores: np.ndarray, metric: str) -> Dict[str, Any]:
        best = int(np.argmax(scores))
        top = float(scores[best])
        tied = bool(np.sum(np.abs(scores - top) < 1e-9) > 1)
        degenerate = bool(np.allclose(scores, scores[0]))
        return {
            "model": str(model_names[best]),
            "score": round(top, 3),
            "metric": metric,
            "tied": tied,
            # All models scoring the same means the signal carries no information.
            "informative": not degenerate,
        }

    return {
        "component_method": method,
        "component_layout": layout,
        "component_n_points": n_points,
        "seasonal_period": period,
        "seasonal_period_source": period_info["source"],
        "seasonal_period_fits": bool(period_info["fits"]),
        "trend_champion": _champion(trend_scores, trend_metric),
        "seasonality_champion": _champion(shape_scores, shape_metric),
        "mean_trend_score": round(float(np.nanmean(trend_scores)), 3),
        "mean_seasonality_score": round(float(np.nanmean(shape_scores)), 3),
    }
