"""Self-contained metrics, using the same formulas as `all_functions.py`.

Reimplemented here on purpose: `all_functions` imports aeon, pywt and pyts at module
level, which makes it impossible to test the tool catalog in isolation. The formulas
are identical — `tests/test_orchestrator_react.py` compares both whenever
`all_functions` is importable.

Project conventions preserved:
    - MAPE as a fraction (not a percentage), like `sklearn.mean_absolute_percentage_error`.
    - POCID in [0, 100], comparing consecutive steps within the vector.
    - SMAPE/MSMAPE/MAE/RMSE take 2-D `(forecasts, test_set)` and reduce per row.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def mape(y_true, y_pred, zero: str = "skip", epsilon: float = 1e-8) -> float:
    """MAPE as a fraction. `zero="skip"` drops points where |y_true| <= epsilon."""
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    denom = np.abs(y_true)
    if zero == "skip":
        mask = denom > epsilon
        if not np.any(mask):
            return float("nan")
        return float(np.nanmean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))
    return float(np.nanmean(np.abs(y_pred - y_true) / np.maximum(denom, epsilon)))


def smape(y_true, y_pred) -> float:
    """SMAPE in [0, 2] (same scale as `all_functions.calculate_smape`)."""
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    denom = np.abs(y_pred) + np.abs(y_true)
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / denom))


def msmape(y_true, y_pred, epsilon: float = 0.1) -> float:
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    comparator = np.full(y_true.shape, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(y_pred) + np.abs(y_true) + epsilon)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / denom))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    return float(np.nanmean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def pocid(y_true, y_pred) -> float:
    """Percentage Of Correct Increasing Direction, in [0, 100]."""
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    n = int(min(y_true.size, y_pred.size))
    if n < 2:
        return float("nan")
    dt = np.diff(y_true[:n])
    dp = np.diff(y_pred[:n])
    ok = (dp * dt) > 0
    return float(100.0 * np.sum(ok) / (n - 1))


def all_metrics(y_true, y_pred, zero: str = "skip", epsilon: float = 1e-8) -> Dict[str, float]:
    """The six CSV metrics, computed over a pair of aligned vectors."""
    return {
        "MAPE": mape(y_true, y_pred, zero=zero, epsilon=epsilon),
        "SMAPE": smape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "POCID": pocid(y_true, y_pred),
        "MSMAPE": msmape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }


def composite_score(
    agg: Dict[str, float],
    baseline: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Composite score normalised against an anchor (lower is better).

    Each error metric enters as a ratio against the anchor (typically the plain mean
    of all models); POCID enters normalised to [0, 1] with a negative sign, since
    higher is better. Zero-weight metrics are skipped even when NaN — which is what
    makes the `scale_free_safe` preset work on series that cross zero.
    """

    def ratio(key: str) -> float:
        num, den = float(agg.get(key, np.nan)), float(baseline.get(key, np.nan))
        if not np.isfinite(den) or abs(den) < 1e-12:
            return float("inf") if np.isfinite(num) and num != 0 else 1.0
        if not np.isfinite(num):
            return float("inf")
        return num / den

    total = 0.0
    for key, wkey in (("RMSE", "a_rmse"), ("SMAPE", "b_smape"), ("MAPE", "c_mape")):
        w = float(weights.get(wkey, 0.0))
        if w == 0.0:
            continue
        total += w * ratio(key)

    w_pocid = float(weights.get("d_pocid", 0.0))
    if w_pocid:
        p = float(agg.get("POCID", np.nan))
        total -= w_pocid * (p / 100.0 if np.isfinite(p) else 0.0)
    return float(total)
