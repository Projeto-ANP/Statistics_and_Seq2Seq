"""Deterministic time-series features for the SeriesAnalyst agent.

These features quantify the series characteristics that drive combination-strategy
choice: trend/seasonal strength (STL-based, à la tsfeatures / Montero-Manso et al. 2020),
forecastability (spectral entropy; Goerg 2013), long-memory (Hurst), stationarity (ADF),
and structural instability (variance ratio across halves).

All functions are robust to short series and NaNs and never raise — they return NaN
sentinels when the segment is too short, so the LLM can reason with explicit caveats.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def spectral_entropy(x: np.ndarray) -> float:
    """Normalized spectral entropy in [0, 1]. High => closer to white noise (less forecastable)."""

    x = _clean(x)
    n = int(x.size)
    if n < 8:
        return float("nan")
    x = x - np.mean(x)
    psd = np.abs(np.fft.rfft(x)) ** 2
    psd = psd[1:]  # drop DC component
    s = float(np.sum(psd))
    if s <= 0:
        return float("nan")
    p = psd / s
    p = p[p > 0]
    ent = -float(np.sum(p * np.log(p)))
    return ent / float(np.log(len(p))) if len(p) > 1 else float("nan")


def hurst_exponent(x: np.ndarray) -> float:
    """Rescaled-range Hurst exponent. >0.5 trending/persistent; <0.5 mean-reverting."""

    x = _clean(x)
    n = int(x.size)
    if n < 16:
        return float("nan")
    max_k = max(2, n // 2)
    lags = range(2, max_k)
    tau = []
    valid_lags = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        sd = float(np.std(diff))
        if sd > 0:
            tau.append(sd)
            valid_lags.append(lag)
    if len(valid_lags) < 4:
        return float("nan")
    coeffs = np.polyfit(np.log(valid_lags), np.log(tau), 1)
    return float(coeffs[0])


def stl_strengths(x: np.ndarray, period: int) -> Dict[str, float]:
    """Trend and seasonal strength in [0, 1] (Wang/Hyndman tsfeatures definition).

    strength = max(0, 1 - Var(resid) / Var(resid + component)).
    Falls back to linear detrending when the segment is too short for STL.
    """

    x = _clean(x)
    n = int(x.size)
    out = {"trend_strength": float("nan"), "seasonal_strength": float("nan")}
    if n < 8:
        return out

    p = max(2, int(period))
    try:
        if n >= 2 * p:
            from statsmodels.tsa.seasonal import STL

            res = STL(x, period=p, robust=True).fit()
            trend = np.asarray(res.trend, dtype=float)
            seasonal = np.asarray(res.seasonal, dtype=float)
            resid = np.asarray(res.resid, dtype=float)
        else:
            idx = np.arange(n, dtype=float)
            coeffs = np.polyfit(idx, x, 1)
            trend = np.polyval(coeffs, idx)
            seasonal = np.zeros_like(x)
            resid = x - trend

        var_r = float(np.var(resid))
        var_rt = float(np.var(resid + trend))
        var_rs = float(np.var(resid + seasonal))
        out["trend_strength"] = max(0.0, 1.0 - var_r / var_rt) if var_rt > 0 else float("nan")
        out["seasonal_strength"] = max(0.0, 1.0 - var_r / var_rs) if var_rs > 0 else float("nan")
    except Exception:
        return out
    return out


def adf_pvalue(x: np.ndarray) -> float:
    """Augmented Dickey-Fuller p-value. Low (<0.05) => stationary."""

    x = _clean(x)
    if x.size < 12:
        return float("nan")
    try:
        from statsmodels.tsa.stattools import adfuller

        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return float("nan")


def variance_ratio_halves(x: np.ndarray) -> float:
    """Var(second half) / Var(first half). >>1 or <<1 signals non-stationary variance."""

    x = _clean(x)
    n = int(x.size)
    if n < 8:
        return float("nan")
    mid = n // 2
    v1 = float(np.var(x[:mid]))
    v2 = float(np.var(x[mid:]))
    return (v2 + 1e-12) / (v1 + 1e-12)


def trend_direction(x: np.ndarray) -> str:
    """Sign of the OLS slope over the segment: up / down / flat."""

    x = _clean(x)
    n = int(x.size)
    if n < 3:
        return "flat"
    idx = np.arange(n, dtype=float)
    slope = float(np.polyfit(idx, x, 1)[0])
    scale = float(np.mean(np.abs(x))) + 1e-12
    rel = slope * n / scale
    if rel > 0.05:
        return "up"
    if rel < -0.05:
        return "down"
    return "flat"


def compute_series_features(history: np.ndarray, period: int) -> Dict[str, Any]:
    """Bundle of deterministic features for the SeriesAnalyst.

    Args:
        history: 1-D recent observed history (concatenated validation windows).
        period: candidate seasonal period (e.g. 52 for weekly, 12 for monthly).
    """

    x = _clean(history)
    n = int(x.size)
    strengths = stl_strengths(x, period=period)
    se = spectral_entropy(x)

    return {
        "n_observations": n,
        "seasonal_period_assumed": int(period),
        "history_too_short_for_period": bool(n < 2 * max(2, int(period))),
        "mean": float(np.mean(x)) if n else float("nan"),
        "std": float(np.std(x)) if n else float("nan"),
        "cv": float(np.std(x) / (np.abs(np.mean(x)) + 1e-12)) if n else float("nan"),
        "trend_direction": trend_direction(x),
        "trend_strength": strengths["trend_strength"],
        "seasonal_strength": strengths["seasonal_strength"],
        "spectral_entropy": se,
        "forecastability": (1.0 - se) if np.isfinite(se) else float("nan"),
        "hurst": hurst_exponent(x),
        "adf_pvalue": adf_pvalue(x),
        "variance_ratio_halves": variance_ratio_halves(x),
    }
