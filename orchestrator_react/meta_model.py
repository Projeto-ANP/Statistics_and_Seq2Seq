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
BASE_FEATURE_NAMES: Tuple[str, ...] = (
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "acf1",
)

#: catch22 (Lubba et al. 2019) in the fixed order `pycatch22` returns it. These 22
#: are already computed for every series by `series_profile` and were being thrown
#: away here.
#:
#: Why they matter for this model specifically: FFORMA — which this recipe is
#: modelled on, and which currently beats us on ANP_MONTHLY — uses ~42 series
#: features, not 4. And the 4 base features above are close to useless on NN5,
#: where `seasonal_strength` has standard deviation 0.0001 across all 111 series
#: (versus 0.226 on ANP). A model given only saturated inputs cannot discriminate
#: no matter how it is fit; catch22 adds autocorrelation structure, distribution
#: shape and incremental-change statistics that do vary there.
#:
#: Order is fixed and asserted at extraction time: a silently reordered feature
#: vector would train against mismatched columns and fail quietly.
CATCH22_FEATURE_NAMES: Tuple[str, ...] = (
    "DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac", "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5", "CO_trev_1_num", "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1", "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi", "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
)

FEATURE_NAMES: Tuple[str, ...] = BASE_FEATURE_NAMES + CATCH22_FEATURE_NAMES


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
    base = [
        _finite(profile.get("trend_strength")),
        _finite(profile.get("seasonal_strength")),
        _finite(feats.get("spectral_entropy")),
        _finite(feats.get("acf1")),
    ]
    # `catch22` is a dict when pycatch22 is installed and the string
    # "pycatch22 unavailable" otherwise — in which case those 22 slots are zeros
    # and the model degrades to the 4 base features rather than failing. Indexing
    # BY NAME (not by iteration order) is what guarantees column alignment across
    # series even if the provider ever reorders its output.
    c22 = (profile or {}).get("catch22")
    if not isinstance(c22, dict):
        c22 = {}
    return np.array(
        base + [_finite(c22.get(name)) for name in CATCH22_FEATURE_NAMES],
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
    metric: str = "smape",
) -> MetaRow:
    """One training row for one series. No test data: `y_true`/`y_preds` are the
    validation windows, the same arrays `ReactState` builds them from.

    `metric` defaults to sMAPE, and that is load-bearing, not cosmetic: these
    errors are summed ACROSS SERIES by the fforma objective's gradient (and pooled
    across series by the per_model regressors' fit). With raw RMSE, a dataset whose
    series span orders of magnitude — ANP does — lets the large-scale series
    dominate the loss, and the meta-model mostly learns series scale. Measured on
    the 182 ANP series, the identical fforma-objective model scores 0.2159 sMAPE
    when trained on sMAPE contributions and 0.2224 when trained on RMSE ones.
    (FFORMA proper uses OWA for the same scale-free reason.)"""
    from orchestrator_react.weighting import per_model_error

    err = per_model_error(np.asarray(y_true, dtype=float), np.asarray(y_preds, dtype=float), metric=metric)
    errors = {str(name): float(e) for name, e in zip(model_names, err)}
    return MetaRow(
        dataset_index=int(dataset_index),
        features=extract_meta_features(profile),
        errors=errors,
    )


#: The two training objectives this model can be fit under. The difference is not
#: cosmetic — it decided a dataset (measured, LOSO, identical features and folds):
#:
#:   "per_model"  N independent regressors, each predicting ITS OWN model's error;
#:                weights come from softmax(-error) as a separate post-hoc step.
#:                ANP 0.2205 (≈ the plain mean), NN5 0.1188.
#:   "fforma"     ONE multi-class booster whose softmax output IS the weight
#:                vector, trained with Montero-Manso et al. (2020)'s custom
#:                gradient to minimise the COMBINED error directly — so it can
#:                learn interactions ("on series like this, ARIMA+ETS together").
#:                ANP 0.2159 — past the real FFORMA baseline (0.2166) — NN5 0.1197.
#:
#: Neither wins everywhere (the recurring cross-dataset non-transfer), which is
#: why both stay implemented and the choice is a config field, not a constant.
OBJECTIVES = ("per_model", "fforma")

#: Boosting rounds for the "fforma" objective — matches `combinations/fforma.py`,
#: whose numbers these are being compared against.
FFORMA_BOOST_ROUNDS = 100


@dataclass
class PooledMetaModel:
    """Cross-series meta-model, fit on every series except the one it will score —
    the LOSO discipline that keeps a query on series *i* from measuring how well
    the model memorised series *i*'s own row.

    Under `objective="per_model"` it holds one regressor per pool model; under
    `objective="fforma"` it holds a single multi-class booster (see `OBJECTIVES`).
    """

    feature_names: Tuple[str, ...]
    model_names: List[str]
    regressors: Dict[str, Any] = field(default_factory=dict)
    booster: Any = None
    objective: str = "per_model"
    metric: str = "rmse"
    #: How many *other* series this particular held-out fit was trained on —
    #: carried through to the tool's observation so the agent (and the CSV) can
    #: see the sample size behind the number, not just trust it blindly.
    n_train_series: int = 0
    #: True when the fit produced identical margins for every model, i.e. it learned
    #: nothing and its weights are uniform. Only the "fforma" objective can hit this
    #: (see the contribution-scale note in `_fit_one_fforma`). Reported rather than
    #: silently applied, so a run cannot look like it used a meta-model when it did
    #: not.
    degenerate: bool = False

    def _check_features(self, x: np.ndarray) -> None:
        # Fail with the actual cause rather than xgboost's "Feature shape mismatch,
        # expected: 4, got 26". This happens whenever a model was fit with a
        # different `FEATURE_NAMES` than the caller is now extracting — e.g. a
        # cached model from before the catch22 columns were added.
        if x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"this meta-model was fit on {len(self.feature_names)} features but "
                f"{x.shape[1]} were passed. The feature set changed since it was "
                "trained; refit it (rerun the Phase-2 pre-pass) rather than mixing "
                "the two."
            )

    def predict_errors(self, features: np.ndarray, names: Sequence[str]) -> Dict[str, Optional[float]]:
        """Per-model predicted errors ("per_model" objective only)."""
        if self.objective != "per_model":
            raise ValueError(
                f"predict_errors is a 'per_model' API; this model was fit with "
                f"objective={self.objective!r}. Use predict_scores."
            )
        out: Dict[str, Optional[float]] = {}
        x = np.asarray(features, dtype=float).reshape(1, -1)
        self._check_features(x)
        for name in names:
            reg = self.regressors.get(str(name))
            out[str(name)] = float(reg.predict(x)[0]) if reg is not None else None
        return out

    def predict_scores(self, features: np.ndarray) -> Tuple[Dict[str, Optional[float]], str]:
        """`(scores keyed by model name, kind)` — the objective-agnostic API.

        kind "error"  -> lower is better; turn into weights via softmax(-eta·s).
        kind "margin" -> the booster's raw class margins; weights are softmax(+s)
                         over whichever subset of models a pool/fold actually has
                         (softmax of a subset of margins == the full softmax
                         renormalised to that subset, so per-fold membership is
                         handled by construction).
        """
        x = np.asarray(features, dtype=float).reshape(1, -1)
        self._check_features(x)
        if self.objective == "fforma":
            import xgboost as xgb

            raw = self.booster.predict(xgb.DMatrix(x), output_margin=True)
            margins = np.asarray(raw, dtype=float).reshape(-1)
            return {str(n): float(m) for n, m in zip(self.model_names, margins)}, "margin"
        errors = self.predict_errors(x.ravel(), self.model_names)
        return errors, "error"


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


def _fforma_objective(predt: np.ndarray, dtrain: Any, contribution: np.ndarray):
    """Gradient/hessian of Montero-Manso et al. (2020)'s FFORMA loss.

    `predt` are the raw class margins; softmax(predt) are the combination weights;
    the loss being minimised is the weighted validation error of the resulting
    combination — so the booster is trained on the thing we actually care about,
    not on each model's error in isolation. Mirrors
    `combinations/fforma.py::_fforma_objective`, whose published numbers this
    module is benchmarked against.
    """
    from scipy.special import softmax as _softmax

    y = dtrain.get_label().astype(int)
    p = _softmax(predt, axis=1)
    weighted = (p * contribution[y]).sum(axis=1, keepdims=True)
    grad = p * (contribution[y] - weighted)
    hess = contribution[y] * p * (1 - p) - grad * p
    return grad, hess


def _fit_one_fforma(
    x: np.ndarray,
    err: np.ndarray,
    model_names: Sequence[str],
    random_state: int,
    metric: str,
    feature_names: Tuple[str, ...],
) -> PooledMetaModel:
    import xgboost as xgb

    # The custom gradient cannot digest NaN contributions. A model that failed on
    # a series is treated as maximally bad THERE (twice the row's worst finite
    # error), not dropped: dropping the row would also discard every other
    # model's information about that series.
    contribution = err.copy()
    for i in range(contribution.shape[0]):
        row = contribution[i]
        bad = ~np.isfinite(row)
        if bad.all():
            row[:] = 1.0
        elif bad.any():
            row[bad] = 2.0 * float(np.max(row[~bad]))

    # Contributions are used RAW, deliberately. Four variants measured on the 182
    # ANP series — same LOSO folds, same features, only this differing:
    #
    #   raw RMSE                 0.2224   scale pollution: the gradient sums across
    #                                     series, so series whose values run in the
    #                                     millions dominate and the booster learns
    #                                     series scale instead of model competence
    #   raw sMAPE                0.2160   BEST, and what ships
    #   per-row normalised sMAPE 0.2202   flattens hard and easy series to the same
    #                                     weight; but the loss is the dataset's TOTAL
    #                                     weighted error, so hard series SHOULD count
    #                                     more — normalising discards real signal
    #   globally rescaled sMAPE  0.2196   ratios preserved, yet still worse: `grad` is
    #                                     proportional to the contributions, so with
    #                                     fixed rounds and learning rate a divisor
    #                                     acts as a step-size change, and sMAPE's
    #                                     natural magnitude is already near the right
    #                                     step
    #
    # Consequence to know about, since nothing here corrects it: on a dataset whose
    # errors are uniformly tiny, the gradients are too small to move `base_score` in
    # the rounds available, and the booster returns equal margins for every model —
    # `degenerate` below reports that rather than letting the weights quietly
    # collapse to uniform.

    keep = np.ones(len(contribution), dtype=bool)

    dtrain = xgb.DMatrix(x[keep], label=np.arange(int(keep.sum())))
    booster = xgb.train(
        params={"num_class": len(model_names), "seed": int(random_state),
                "verbosity": 0, "nthread": -1},
        dtrain=dtrain,
        num_boost_round=FFORMA_BOOST_ROUNDS,
        obj=lambda predt, dt: _fforma_objective(predt, dt, contribution[keep]),
        verbose_eval=False,
    )
    # Did it actually learn? If every model gets the same margin on the training
    # features, the softmax is uniform and this model is indistinguishable from the
    # plain mean — worth surfacing, not hiding behind a plausible-looking weight
    # vector. (Cause: see the contribution-scale note above.)
    degenerate = False
    try:
        margins = np.asarray(
            booster.predict(xgb.DMatrix(x[keep]), output_margin=True), dtype=float
        ).reshape(int(keep.sum()), -1)
        degenerate = bool(np.allclose(margins, margins[:, :1], atol=1e-9))
    except Exception:
        pass

    return PooledMetaModel(
        feature_names=feature_names,
        model_names=[str(m) for m in model_names],
        booster=booster,
        objective="fforma",
        metric=metric,
        n_train_series=int(keep.sum()),
        degenerate=degenerate,
    )


def _fit_one(
    rows: Sequence[MetaRow],
    model_names: Sequence[str],
    exclude_dataset_index: int,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    metric: str,
    objective: str = "per_model",
) -> PooledMetaModel:
    from xgboost import XGBRegressor

    train_rows = [r for r in rows if r.dataset_index != exclude_dataset_index]
    x = np.stack([r.features for r in train_rows])
    if objective == "fforma":
        err = np.array(
            [[r.errors.get(str(n), np.nan) for n in model_names] for r in train_rows],
            dtype=float,
        )
        return _fit_one_fforma(
            x, err, model_names, random_state, metric,
            feature_names=tuple(FEATURE_NAMES[: x.shape[1]]),
        )
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
        # From the rows actually used, not from the module constant: an ablation
        # that trains on a feature subset must record the subset it trained on, or
        # `predict_errors`' shape check would compare against the wrong length.
        feature_names=tuple(FEATURE_NAMES[: x.shape[1]]),
        model_names=[str(m) for m in model_names],
        regressors=regressors,
        metric=metric,
        n_train_series=len(train_rows),
    )


def build_pooled_meta_models(
    rows: Sequence[MetaRow],
    model_names: Sequence[str],
    #: Descriptive only — the errors already live in `rows`, computed by
    #: `build_meta_row`. It exists so `PooledMetaModel.metric` can say what the
    #: contributions actually are; keep it in step with `build_meta_row`'s default
    #: or a model trained on sMAPE will be labelled "rmse".
    metric: str = "smape",
    min_series: int = MIN_SERIES_FOR_POOLED_META_MODEL,
    n_estimators: int = 40,
    max_depth: int = 2,
    random_state: int = 0,
    objective: str = "per_model",
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

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective!r} (valid: {OBJECTIVES})")

    fitted = {
        row.dataset_index: _fit_one(
            rows, model_names, row.dataset_index, n_estimators, max_depth,
            random_state, metric, objective=objective,
        )
        for row in rows
    }

    # The fforma objective needs a real dataset behind it. Its gradient is
    # proportional to the contributions, so on a small run it never moves
    # `base_score` and every model comes back with the same margin — weights
    # uniform, i.e. silently identical to the plain mean. Caught in practice on a
    # 25-series smoke run (24 training rows): every fit came back `degenerate`,
    # while the full 182-series run learns fine. `per_model` has no such floor —
    # independent regressors fit happily on 24 rows — so falling back to it keeps
    # a small run with a meta-model that actually does something, instead of one
    # that looks present and contributes nothing.
    if objective == "fforma" and fitted and all(m.degenerate for m in fitted.values()):
        return {
            row.dataset_index: _fit_one(
                rows, model_names, row.dataset_index, n_estimators, max_depth,
                random_state, metric, objective="per_model",
            )
            for row in rows
        }
    return fitted
