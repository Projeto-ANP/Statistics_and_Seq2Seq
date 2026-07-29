"""Cross-series meta-model — the classical-ML piece ADE/FFORMA has and we didn't.

`weighting.weights_feature_based` already exists and never runs its real path: it
trains one XGBoost regressor **per series**, on that series' 3 validation windows,
and 3 samples can never clear the "enough data to fit a model" bar no matter how
it is tuned (`n_fit < 2 * n_features` is true for every value of `n_features >= 2`
once `n_fit == 3`). That is not a bug to patch — the unit of training is wrong.

Real FFORMA does not retrain per series. It extracts features that describe each
series (trend strength, seasonal strength, entropy, autocorrelation — the same
family `series_profile` already computes) and trains **one meta-model per dataset**,
using every series in it as a training row. The sample size that matters is not
"3 windows", it is "how many series are in this dataset" — 111 on NN5, 182 on
ANP_MONTHLY, both comfortably enough for a shallow gradient-boosted regressor.

This module is that: a pre-pass over the whole dataset builds one row per series
(features + each pool model's validation error), and `build_pooled_meta_models`
fits one regressor set per series **leaving that series out** (LOSO) — mirroring
the same leave-one-out discipline `nested_selection` already applies to pool
membership, for the same reason: a model queried on the series that trained it
would be measuring memorisation, not generalisation.

The features used (trend/seasonal strength, spectral entropy, lag-1 autocorrelation)
are deliberately restricted to properties of the series' own historical shape —
computed once from `train_series`, which is fully known before Phase 3 opens and
does not depend on which validation window a backtest fold excludes. That is what
lets the produced weights be reused unchanged across every fold: nothing about
them was ever a function of a specific window, so there is nothing for a fold to
leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


#: Below this many series, pooling has too little signal to be worth training —
#: a dataset-level meta-model needs a dataset, not a handful of smoke-test rows.
MIN_SERIES_FOR_POOLED_META_MODEL = 20

#: Below this many finite targets, one model's regressor is not fit at all (its
#: prediction stays `None`, and the tool treats that the same as "no signal" for
#: that model, exactly like the existing softmax fallbacks do for a NaN error).
MIN_FINITE_TARGETS_PER_MODEL = 5

#: Named, in this order, purely so `np.array([...])` and error messages agree.
#: All four come from `series_profile`'s `trend_strength`/`seasonal_strength`/
#: `features.spectral_entropy`/`features.acf1` — computed from `train_series`,
#: which is historical by construction and identical across every backtest fold.
FEATURE_NAMES: Tuple[str, ...] = (
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "acf1",
)


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def extract_meta_features(profile: Dict[str, Any]) -> np.ndarray:
    """`FEATURE_NAMES`, in order, from a `series_profile()` card.

    Missing or non-finite entries default to 0.0 rather than raising: a series
    profile computed under `linear_fallback` (statsmodels unavailable, or too
    short a history) still has every key, just less trustworthy values, and this
    is meant to degrade the same way the rest of the pipeline does, not crash.
    """
    feats = (profile or {}).get("features", {}) or {}
    if not isinstance(feats, dict):
        feats = {}
    return np.array(
        [
            _finite(profile.get("trend_strength")),
            _finite(profile.get("seasonal_strength")),
            _finite(feats.get("spectral_entropy")),
            _finite(feats.get("acf1")),
        ],
        dtype=float,
    )


@dataclass
class MetaRow:
    """One dataset's one series, as the pooled meta-model sees it: a feature
    vector plus every pool model's validation error."""

    dataset_index: int
    features: np.ndarray
    errors: Dict[str, float]


def build_meta_row(
    dataset_index: int,
    profile: Dict[str, Any],
    y_true: np.ndarray,
    y_preds: np.ndarray,
    model_names: Sequence[str],
    metric: str = "rmse",
) -> MetaRow:
    """One training row for one series. No test data: `y_true`/`y_preds` are the
    validation windows, the same arrays `ReactState` builds them from."""
    from orchestrator_react.weighting import per_model_error

    err = per_model_error(np.asarray(y_true, dtype=float), np.asarray(y_preds, dtype=float), metric=metric)
    errors = {str(name): float(e) for name, e in zip(model_names, err)}
    return MetaRow(
        dataset_index=int(dataset_index),
        features=extract_meta_features(profile),
        errors=errors,
    )


@dataclass
class PooledMetaModel:
    """One regressor per pool model, fit on every series except the one it will
    score — the LOSO discipline that keeps a query on series *i* from measuring
    how well the model memorised series *i*'s own row."""

    feature_names: Tuple[str, ...]
    model_names: List[str]
    regressors: Dict[str, Any] = field(default_factory=dict)
    metric: str = "rmse"
    #: How many *other* series this particular held-out fit was trained on —
    #: carried through to the tool's observation so the agent (and the CSV) can
    #: see the sample size behind the number, not just trust it blindly.
    n_train_series: int = 0

    def predict_errors(self, features: np.ndarray, names: Sequence[str]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        x = np.asarray(features, dtype=float).reshape(1, -1)
        for name in names:
            reg = self.regressors.get(str(name))
            out[str(name)] = float(reg.predict(x)[0]) if reg is not None else None
        return out


def errors_to_weights(
    predicted: Dict[str, Optional[float]], names: Sequence[str], eta: float = 1.0
) -> np.ndarray:
    """`softmax(-eta * predicted_error / median)` — FFORMA's own final step,
    already used identically by `weighting.weights_softmax_neg_error` and by the
    per-series `weights_feature_based`. A model with no regressor (`None`) is
    treated as `+inf`, the same convention `weights_softmax_neg_error` uses for a
    non-finite error: it can still receive weight if every candidate is missing,
    via the uniform fallback, but never outranks a model with a real prediction.
    """
    n = len(names)
    vals = np.array(
        [predicted.get(str(nm)) if predicted.get(str(nm)) is not None else np.inf for nm in names],
        dtype=float,
    )
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.ones(n, dtype=float) / max(1, n)
    scale = float(np.median(finite)) or 1.0
    z = -float(eta) * (vals / scale)
    z = np.where(np.isfinite(z), z, -np.inf)
    top = np.max(z[np.isfinite(z)], initial=0.0)
    e = np.exp(z - top)
    s = e.sum()
    return e / s if s > 0 else np.ones(n, dtype=float) / max(1, n)


def _fit_one(
    rows: Sequence[MetaRow],
    model_names: Sequence[str],
    exclude_dataset_index: int,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    metric: str,
) -> PooledMetaModel:
    from xgboost import XGBRegressor

    train_rows = [r for r in rows if r.dataset_index != exclude_dataset_index]
    x = np.stack([r.features for r in train_rows])
    regressors: Dict[str, Any] = {}
    for name in model_names:
        y = np.array([r.errors.get(str(name), np.nan) for r in train_rows], dtype=float)
        mask = np.isfinite(y)
        if int(mask.sum()) < MIN_FINITE_TARGETS_PER_MODEL:
            regressors[str(name)] = None
            continue
        model = XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=0.1,
            random_state=int(random_state),
            verbosity=0,
        )
        model.fit(x[mask], y[mask])
        regressors[str(name)] = model
    return PooledMetaModel(
        feature_names=FEATURE_NAMES,
        model_names=[str(m) for m in model_names],
        regressors=regressors,
        metric=metric,
        n_train_series=len(train_rows),
    )


def build_pooled_meta_models(
    rows: Sequence[MetaRow],
    model_names: Sequence[str],
    metric: str = "rmse",
    min_series: int = MIN_SERIES_FOR_POOLED_META_MODEL,
    n_estimators: int = 40,
    max_depth: int = 2,
    random_state: int = 0,
) -> Dict[int, PooledMetaModel]:
    """One leave-one-series-out `PooledMetaModel` per row's `dataset_index`.

    Returns `{}` — meaning the tool is withheld for the whole run, exactly like
    `weights_ols` under too few windows — when there are fewer than `min_series`
    rows or when `xgboost` is not installed. A run under the threshold is not
    "trained on a smaller sample": FFORMA's own advantage over the per-series
    meta-model was pooling across series in the first place, so a training set
    too small to pool is not this tool's job to serve, and offering it anyway
    would cost the agent an iteration on a fit no better than
    `weights_softmax_neg_error`, which is already in the catalog.
    """
    if len(rows) < int(min_series):
        return {}
    try:
        import xgboost  # noqa: F401 — availability probe
    except Exception:
        return {}

    return {
        row.dataset_index: _fit_one(
            rows, model_names, row.dataset_index, n_estimators, max_depth, random_state, metric
        )
        for row in rows
    }
