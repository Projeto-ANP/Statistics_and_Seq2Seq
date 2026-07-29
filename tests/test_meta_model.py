"""Cross-series pooled meta-model (Step: classical-ML integration).

`weighting.weights_feature_based` never fires its real path on any real run: it
fits one XGBoost regressor per series on that series' 3 validation windows, and
`n_fit < 2 * n_features` is true for every reasonable feature count once
`n_fit == 3` — confirmed by grep across every NN5 and ANP_MONTHLY run so far,
where the tool is never even called. Real FFORMA does not retrain per series: it
pools every series in the dataset into one training set. This module is that,
with leave-one-series-out (LOSO) fitting so a query on series *i* is never
answered by a model that memorised series *i*'s own row.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator_react import meta_model as MM

xgboost = pytest.importorskip("xgboost")


HORIZON = 6
MODEL_NAMES = ["good_on_trend", "good_on_flat"]


def _profile(trend_strength: float) -> dict:
    """A minimal `series_profile()`-shaped card — only the four keys this module
    reads, so the test does not depend on `features.py`'s internals."""
    return {
        "trend_strength": trend_strength,
        "seasonal_strength": 0.1,
        "features": {"spectral_entropy": 0.5, "acf1": 0.2},
    }


def _row(idx: int, trend_strength: float, seed: int = 0) -> MM.MetaRow:
    """A series whose winning model is DETERMINED by `trend_strength`: above 0.5
    `good_on_trend` wins, at or below it `good_on_flat` wins. This is what makes
    LOSO's prediction checkable — the fitted regressor must have learned the
    threshold from every OTHER row, not memorised this one."""
    rng = np.random.default_rng(seed + idx)
    y_true = rng.normal(100.0, 5.0, size=(3, HORIZON))
    # (noise for good_on_trend, noise for good_on_flat)
    noise = (1.0, 10.0) if trend_strength > 0.5 else (10.0, 1.0)
    y_preds = np.stack(
        [
            y_true + rng.normal(0, noise[0], size=(3, HORIZON)),
            y_true + rng.normal(0, noise[1], size=(3, HORIZON)),
        ],
        axis=1,
    )
    return MM.build_meta_row(idx, _profile(trend_strength), y_true, y_preds, MODEL_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# feature extraction
# ──────────────────────────────────────────────────────────────────────────────


def test_extract_meta_features_reads_the_four_named_fields():
    profile = {
        "trend_strength": 0.9, "seasonal_strength": 0.3,
        "features": {"spectral_entropy": 0.7, "acf1": -0.1},
    }
    feats = MM.extract_meta_features(profile)
    assert feats.tolist() == pytest.approx([0.9, 0.3, 0.7, -0.1])


def test_missing_or_non_finite_fields_default_to_zero_not_nan():
    feats = MM.extract_meta_features({"trend_strength": float("nan"), "features": {}})
    assert np.all(np.isfinite(feats))
    assert feats.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_an_empty_profile_does_not_raise():
    feats = MM.extract_meta_features({})
    assert feats.shape == (4,)


# ──────────────────────────────────────────────────────────────────────────────
# build_meta_row — no test data involved, ever
# ──────────────────────────────────────────────────────────────────────────────


def test_build_meta_row_uses_only_validation_windows():
    row = _row(0, trend_strength=0.9)
    assert set(row.errors) == set(MODEL_NAMES)
    assert all(np.isfinite(v) for v in row.errors.values())
    assert row.dataset_index == 0


def test_the_winning_model_in_the_row_matches_the_construction():
    """Sanity check on the fixture itself before trusting anything built on it."""
    high = _row(1, trend_strength=0.9)
    low = _row(2, trend_strength=0.1)
    assert min(high.errors, key=high.errors.get) == "good_on_trend"
    assert min(low.errors, key=low.errors.get) == "good_on_flat"


# ──────────────────────────────────────────────────────────────────────────────
# build_pooled_meta_models — the gate and the LOSO discipline
# ──────────────────────────────────────────────────────────────────────────────


def _dataset(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [_row(i, trend_strength=float(rng.uniform(0, 1)), seed=seed) for i in range(n)]


def test_too_few_series_withholds_the_whole_run():
    rows = _dataset(MM.MIN_SERIES_FOR_POOLED_META_MODEL - 1)
    assert MM.build_pooled_meta_models(rows, MODEL_NAMES) == {}


def test_enough_series_trains_one_model_per_row():
    rows = _dataset(MM.MIN_SERIES_FOR_POOLED_META_MODEL + 5)
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0)
    assert set(models) == {r.dataset_index for r in rows}
    assert all(isinstance(m, MM.PooledMetaModel) for m in models.values())


def test_loso_excludes_the_series_own_row():
    """The defining property: series i's model must be trained on n-1 rows, and
    must not have seen series i's own error at all."""
    rows = _dataset(30, seed=1)
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0)
    for row in rows:
        assert models[row.dataset_index].n_train_series == len(rows) - 1


def test_the_pooled_model_generalises_the_threshold_learned_from_other_series():
    """Series far from the boundary should be predicted correctly by a model that
    never saw that series — this is the entire point of pooling: the signal comes
    from every OTHER series, not from this one's own 3 windows."""
    rows = [_row(i, trend_strength=0.95, seed=10) for i in range(15)]
    rows += [_row(i + 100, trend_strength=0.05, seed=20) for i in range(15)]
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0)

    held_out_high = rows[0]
    model_for_it = models[held_out_high.dataset_index]
    predicted = model_for_it.predict_errors(held_out_high.features, MODEL_NAMES)
    assert predicted["good_on_trend"] < predicted["good_on_flat"]

    held_out_low = rows[15]
    model_for_it = models[held_out_low.dataset_index]
    predicted = model_for_it.predict_errors(held_out_low.features, MODEL_NAMES)
    assert predicted["good_on_flat"] < predicted["good_on_trend"]


def test_xgboost_unavailable_withholds_gracefully(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "xgboost":
            raise ImportError("simulated: xgboost not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rows = _dataset(MM.MIN_SERIES_FOR_POOLED_META_MODEL + 5)
    assert MM.build_pooled_meta_models(rows, MODEL_NAMES) == {}


def test_a_model_with_too_few_finite_targets_is_left_unfit():
    """A model that failed to produce a usable error on almost every series gets
    `None`, not a regressor trained on 2 points."""
    rows = _dataset(30, seed=2)
    for r in rows[3:]:
        r.errors["good_on_flat"] = float("nan")
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0)
    some_model = next(iter(models.values()))
    assert some_model.regressors["good_on_flat"] is None
    assert some_model.regressors["good_on_trend"] is not None


# ──────────────────────────────────────────────────────────────────────────────
# errors_to_weights — FFORMA's own final step
# ──────────────────────────────────────────────────────────────────────────────


def test_lower_predicted_error_gets_more_weight():
    w = MM.errors_to_weights({"a": 1.0, "b": 10.0}, ["a", "b"], eta=1.0)
    assert w[0] > w[1]
    assert w.sum() == pytest.approx(1.0)


def test_equal_predicted_errors_give_equal_weight():
    w = MM.errors_to_weights({"a": 5.0, "b": 5.0}, ["a", "b"], eta=1.0)
    assert w[0] == pytest.approx(w[1])


def test_a_missing_prediction_never_outranks_a_real_one():
    w = MM.errors_to_weights({"a": 1.0, "b": None}, ["a", "b"], eta=1.0)
    assert w[0] > w[1]


def test_all_missing_falls_back_to_uniform():
    w = MM.errors_to_weights({"a": None, "b": None}, ["a", "b"])
    assert w.tolist() == pytest.approx([0.5, 0.5])


def test_weights_are_always_a_valid_simplex_point():
    for eta in (0.1, 1.0, 5.0, 20.0):
        w = MM.errors_to_weights({"a": 3.0, "b": 7.0, "c": 1.0}, ["a", "b", "c"], eta=eta)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0.0)
