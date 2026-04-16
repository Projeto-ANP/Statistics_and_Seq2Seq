"""Advanced statistical diagnostics for the PatternAnalyst and tie-breaking.

This module provides deterministic, publication-grade signals that go beyond
aggregate RMSE: bias-per-horizon, residual autocorrelation (Ljung-Box),
heteroscedasticity proxy, drift across folds, spectral entropy of y_true,
rank-stability (Kendall tau) of the model RMSE ordering across folds, and
model-error similarity (for redundancy detection).

The tie-breaking utilities implement the Diebold-Mariano (1995) test with the
Harvey-Leybourne-Newbold (1997) small-sample correction and a paired
(block) bootstrap of the combined score to decide whether the top-2
candidates are statistically separable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# Residual-level diagnostics (per model / per fold)
# ──────────────────────────────────────────────────────────────────────────


def bias_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean of (pred - true) per horizon step, averaged across folds.

    Positive => systematic over-forecast; negative => under-forecast.
    Shape: (horizon,).
    """

    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return np.nanmean(diff, axis=0)


def ljung_box_pvalue(residuals: np.ndarray, lags: int = 4) -> float:
    """Ljung-Box Q test p-value for serial autocorrelation in residuals.

    Low p-value (< 0.05) => residuals are auto-correlated, i.e. the model
    leaves structure on the table. Returns NaN if the series is too short
    or all-NaN.
    """

    x = np.asarray(residuals, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < max(8, lags + 2):
        return float("nan")
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(x, lags=[int(lags)], return_df=True)
        p = float(lb["lb_pvalue"].iloc[-1])
        return p if np.isfinite(p) else float("nan")
    except Exception:
        return float("nan")


def heteroscedasticity_ratio(residuals: np.ndarray) -> float:
    """Ratio between the variance in the 2nd half and the 1st half of residuals.

    > 1 => variance grows (heteroscedastic expansion); < 1 => variance shrinks.
    """

    x = np.asarray(residuals, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 4:
        return float("nan")
    mid = n // 2
    v_early = float(np.var(x[:mid]))
    v_late = float(np.var(x[mid:]))
    return (v_late + EPS) / (v_early + EPS)


def drift_signal(rmse_per_fold: List[float]) -> Dict[str, float]:
    """Detect concept drift from the fold-wise RMSE series.

    Returns slope (linear regression, normalised by mean RMSE), and a
    monotonic increase probability proxy (fraction of strictly increasing
    consecutive folds).
    """

    r = np.asarray([x for x in rmse_per_fold if x is not None and np.isfinite(x)], dtype=float)
    if r.size < 2:
        return {"slope_norm": float("nan"), "mono_increase_frac": float("nan")}
    t = np.arange(r.size, dtype=float)
    slope, _ = np.polyfit(t, r, 1)
    mean_r = float(np.mean(r))
    slope_norm = float(slope) / (abs(mean_r) + EPS)
    inc = int(np.sum(np.diff(r) > 0))
    frac = float(inc) / float(max(1, r.size - 1))
    return {"slope_norm": float(slope_norm), "mono_increase_frac": frac}


def spectral_entropy(series: np.ndarray) -> float:
    """Normalised spectral entropy of a 1-D series.

    0 => concentrated on few frequencies (highly predictable);
    1 => flat spectrum (white-noise-like, unpredictable).
    """

    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 4:
        return float("nan")
    x = x - np.mean(x)
    f = np.abs(np.fft.rfft(x))[1:]  # drop DC
    if f.size == 0:
        return float("nan")
    p = f ** 2
    s = float(np.sum(p))
    if s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    h = float(-np.sum(p * np.log(p)))
    h_max = float(np.log(p.size))
    return (h / h_max) if h_max > 0 else float("nan")


def hurst_rs(series: np.ndarray) -> float:
    """Rough Hurst exponent via rescaled-range (R/S) on a single series.

    H ~ 0.5 => random walk; H > 0.5 => persistent; H < 0.5 => mean-reverting.
    Short-series approximation; returns NaN if the series is too short.
    """

    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 16:
        return float("nan")
    lags = np.unique(np.round(np.logspace(np.log10(2), np.log10(n // 2), num=8)).astype(int))
    lags = lags[lags >= 2]
    tau = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        if diff.size == 0:
            continue
        tau.append(np.sqrt(np.mean(diff ** 2)) + EPS)
    if len(tau) < 3:
        return float("nan")
    lag_arr = np.asarray(lags[: len(tau)], dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    poly = np.polyfit(np.log(lag_arr), np.log(tau_arr), 1)
    return float(poly[0])


# ──────────────────────────────────────────────────────────────────────────
# Multi-model diagnostics across folds
# ──────────────────────────────────────────────────────────────────────────


def rank_stability_kendall(per_fold_rmse: np.ndarray) -> float:
    """Average pairwise Kendall tau of model RMSE rankings across folds.

    Input shape: (n_folds, n_models) of RMSE values (lower is better).
    Returns a value in [-1, 1]; 1 => models keep the same order across folds.
    """

    X = np.asarray(per_fold_rmse, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau
    except Exception:
        return float("nan")
    n_folds = X.shape[0]
    vals: List[float] = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            xi = X[i]
            xj = X[j]
            mask = (~np.isnan(xi)) & (~np.isnan(xj))
            if int(np.sum(mask)) < 2:
                continue
            try:
                tau, _ = kendalltau(xi[mask], xj[mask])
                if np.isfinite(tau):
                    vals.append(float(tau))
            except Exception:
                continue
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def error_similarity_matrix(errors_per_model: np.ndarray) -> Dict[str, Any]:
    """Pairwise Pearson correlation of model errors.

    Input shape: (n_models, n_points). High correlation between two models
    means they are redundant — removing one keeps almost all the signal.
    Returns the mean off-diagonal correlation and the most-redundant pair.
    """

    E = np.asarray(errors_per_model, dtype=float)
    if E.ndim != 2 or E.shape[0] < 2:
        return {"mean_abs_corr": float("nan"), "most_redundant_pair": None}
    n = E.shape[0]
    corrs = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            xi = E[i]
            xj = E[j]
            mask = (~np.isnan(xi)) & (~np.isnan(xj))
            if int(np.sum(mask)) < 3:
                continue
            a = xi[mask] - np.mean(xi[mask])
            b = xj[mask] - np.mean(xj[mask])
            denom = np.std(a) * np.std(b)
            if denom < 1e-12:
                continue
            c = float(np.dot(a, b) / (a.size * denom))
            corrs[i, j] = c
            corrs[j, i] = c
    abs_vals = np.abs(corrs)
    mean_abs = float(np.nanmean(abs_vals)) if np.any(~np.isnan(abs_vals)) else float("nan")
    if np.any(~np.isnan(abs_vals)):
        flat = np.where(np.isnan(abs_vals), -np.inf, abs_vals)
        idx = np.unravel_index(int(np.argmax(flat)), flat.shape)
        return {
            "mean_abs_corr": mean_abs,
            "most_redundant_pair": [int(idx[0]), int(idx[1])],
            "most_redundant_corr": float(corrs[idx]),
        }
    return {"mean_abs_corr": mean_abs, "most_redundant_pair": None}


# ──────────────────────────────────────────────────────────────────────────
# Tie-breaking: Diebold-Mariano and paired bootstrap
# ──────────────────────────────────────────────────────────────────────────


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> Dict[str, float]:
    """Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997) correction.

    H0: the two candidates have equal expected forecast loss.
    Positive DM => candidate A is worse than B on average.

    Args:
        errors_a: per-observation errors for candidate A (any length)
        errors_b: per-observation errors for candidate B (same length)
        loss: "squared" or "absolute"
        h: forecast horizon step (for HLN correction)

    Returns:
        {"dm_stat": float, "p_value": float, "n": int}; p-values are two-sided.
    """

    ea = np.asarray(errors_a, dtype=float).ravel()
    eb = np.asarray(errors_b, dtype=float).ravel()
    n = int(min(ea.size, eb.size))
    if n < 5:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n}

    ea = ea[:n]
    eb = eb[:n]
    mask = (~np.isnan(ea)) & (~np.isnan(eb))
    ea = ea[mask]
    eb = eb[mask]
    n = int(ea.size)
    if n < 5:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n}

    if loss == "absolute":
        d = np.abs(ea) - np.abs(eb)
    else:
        d = ea ** 2 - eb ** 2

    d_mean = float(np.mean(d))
    # Long-run variance via Newey-West with bandwidth h-1
    gamma_0 = float(np.var(d, ddof=0))
    lrv = gamma_0
    max_lag = max(0, int(h) - 1)
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        cov = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        w = 1.0 - k / (max_lag + 1)
        lrv += 2.0 * w * cov
    if lrv <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n}
    dm = d_mean / np.sqrt(lrv / n)

    # HLN small-sample correction factor
    correction = np.sqrt(max(1.0, (n + 1 - 2 * h + h * (h - 1) / n) / n))
    dm_hln = dm * correction

    # Two-sided p-value via Student-t with n-1 df
    try:
        from scipy.stats import t as tdist

        p = 2.0 * (1.0 - float(tdist.cdf(abs(dm_hln), df=max(1, n - 1))))
    except Exception:
        # Normal fallback
        p = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(dm_hln) / np.sqrt(2))))
    return {"dm_stat": float(dm_hln), "p_value": float(p), "n": n}


def paired_bootstrap_score(
    scores_a_per_window: np.ndarray,
    scores_b_per_window: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap on window-level combined scores of two candidates.

    H0: E[score_A] == E[score_B].
    Returns the bootstrap mean-difference, a 95% CI, and the two-sided p-value
    (fraction of bootstrap replicates with sign opposite to the observed mean).
    """

    a = np.asarray(scores_a_per_window, dtype=float).ravel()
    b = np.asarray(scores_b_per_window, dtype=float).ravel()
    n = int(min(a.size, b.size))
    if n < 3:
        return {
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }
    a = a[:n]
    b = b[:n]
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a = a[mask]
    b = b[mask]
    n = int(a.size)
    if n < 3:
        return {
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }

    d = a - b
    obs = float(np.mean(d))
    rng = np.random.default_rng(int(seed))
    n_boot = int(max(200, min(n_boot, 10000)))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    # Two-sided p-value: min(P[boot>=0], P[boot<=0]) * 2, centred under H0 by shifting
    shifted = boot - obs
    p_two = 2.0 * float(min(
        np.mean(shifted >= abs(obs)),
        np.mean(shifted <= -abs(obs)),
    ))
    p_two = max(0.0, min(1.0, p_two))
    return {
        "mean_diff": obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_two),
        "n": n,
    }


def tie_break_analysis(
    top_ranking: List[Dict[str, Any]],
    per_window_scores: Dict[str, np.ndarray],
    per_window_errors: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """Statistical analysis of whether the top-1 dominates the top-2.

    Args:
        top_ranking: at least two entries, each {"name": str, "score": float}
        per_window_scores: per-candidate 1-D array of window-level scores
        per_window_errors: optional per-candidate flat error array for DM
        alpha: significance level for the "tie" flag

    Returns a dict combining DM and paired-bootstrap evidence plus a
    boolean `statistically_tied` flag. When the flag is True the debate
    gate should fire.
    """

    if not isinstance(top_ranking, list) or len(top_ranking) < 2:
        return {"available": False}

    name1 = str(top_ranking[0].get("name", ""))
    name2 = str(top_ranking[1].get("name", ""))
    s1 = per_window_scores.get(name1)
    s2 = per_window_scores.get(name2)
    if s1 is None or s2 is None:
        return {"available": False}

    bs = paired_bootstrap_score(s1, s2, n_boot=2000)

    dm: Dict[str, float] = {"dm_stat": float("nan"), "p_value": float("nan"), "n": 0}
    if per_window_errors is not None:
        e1 = per_window_errors.get(name1)
        e2 = per_window_errors.get(name2)
        if e1 is not None and e2 is not None:
            dm = diebold_mariano(e1, e2, loss="squared", h=1)

    # A tie is declared when the paired bootstrap cannot reject H0 at `alpha`.
    p_bs = bs.get("p_value")
    p_dm = dm.get("p_value")
    # Use the bootstrap as the primary gate (more robust with few windows);
    # DM is supporting evidence.
    tied_bs = bool(np.isfinite(p_bs) and p_bs > float(alpha))
    tied_dm = bool(np.isfinite(p_dm) and p_dm > float(alpha))
    # If both are available, require BOTH to say "tied" before declaring a tie;
    # if only one is available, that one decides.
    if np.isfinite(p_bs) and np.isfinite(p_dm):
        statistically_tied = bool(tied_bs and tied_dm)
    elif np.isfinite(p_bs):
        statistically_tied = tied_bs
    elif np.isfinite(p_dm):
        statistically_tied = tied_dm
    else:
        statistically_tied = False

    return {
        "available": True,
        "top1": name1,
        "top2": name2,
        "alpha": float(alpha),
        "paired_bootstrap": bs,
        "diebold_mariano": dm,
        "statistically_tied": bool(statistically_tied),
    }
