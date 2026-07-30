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


def test_extract_meta_features_reads_the_base_fields_in_order():
    profile = {
        "trend_strength": 0.9, "seasonal_strength": 0.3,
        "features": {"spectral_entropy": 0.7, "acf1": -0.1},
    }
    feats = MM.extract_meta_features(profile)
    assert feats[:4].tolist() == pytest.approx([0.9, 0.3, 0.7, -0.1])
    assert feats.shape == (len(MM.FEATURE_NAMES),)


def test_catch22_is_read_by_name_not_by_iteration_order():
    """Column alignment across series must not depend on dict ordering: a profile
    whose catch22 dict is shuffled has to produce the same vector."""
    values = {name: float(i) for i, name in enumerate(MM.CATCH22_FEATURE_NAMES)}
    straight = MM.extract_meta_features({"catch22": values})
    shuffled = MM.extract_meta_features({"catch22": dict(reversed(list(values.items())))})
    assert straight.tolist() == pytest.approx(shuffled.tolist())
    # and they land in the declared slots, after the 4 base features
    assert straight[4:].tolist() == pytest.approx([float(i) for i in range(22)])


def test_catch22_unavailable_degrades_to_the_base_features():
    """`series_profile` sets catch22 to the string 'pycatch22 unavailable' when the
    package is missing. Those 22 slots become zeros; nothing raises."""
    feats = MM.extract_meta_features({
        "trend_strength": 0.5, "features": {"acf1": 0.2},
        "catch22": "pycatch22 unavailable",
    })
    assert np.all(np.isfinite(feats))
    assert feats.shape == (len(MM.FEATURE_NAMES),)
    assert feats[4:].tolist() == pytest.approx([0.0] * 22)


def test_missing_or_non_finite_fields_default_to_zero_not_nan():
    feats = MM.extract_meta_features({"trend_strength": float("nan"), "features": {}})
    assert np.all(np.isfinite(feats))
    assert feats.tolist() == pytest.approx([0.0] * len(MM.FEATURE_NAMES))


def test_an_empty_profile_does_not_raise():
    feats = MM.extract_meta_features({})
    assert feats.shape == (len(MM.FEATURE_NAMES),)


def test_the_feature_vector_length_matches_the_declared_names():
    assert len(MM.FEATURE_NAMES) == len(MM.BASE_FEATURE_NAMES) + len(MM.CATCH22_FEATURE_NAMES)
    assert len(MM.CATCH22_FEATURE_NAMES) == 22
    assert len(set(MM.FEATURE_NAMES)) == len(MM.FEATURE_NAMES), "no duplicate feature names"


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


def test_predicting_with_a_different_feature_count_fails_readably():
    """xgboost's own message is 'Feature shape mismatch, expected: 4, got 26',
    which says nothing about the cause. A model fit before the catch22 columns were
    added, queried after, must say so."""
    rows = _dataset(25, seed=5)
    for r in rows:
        r.features = r.features[:4]           # train on the base features only
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, min_series=1, random_state=0)
    m = next(iter(models.values()))
    assert len(m.feature_names) == 4, "the model must record the subset it trained on"

    full = np.zeros(len(MM.FEATURE_NAMES))
    with pytest.raises(ValueError, match="was fit on 4 features"):
        m.predict_errors(full, MODEL_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# the "fforma" objective — one multi-class booster whose softmax IS the weights
# ──────────────────────────────────────────────────────────────────────────────


def test_fforma_objective_trains_one_booster_not_regressors():
    """`_fit_one` is the unit; the degenerate->per_model fallback is a policy of
    `build_pooled_meta_models`, so the structural assertion goes through the unit."""
    rows = _dataset(30, seed=3)
    m = MM._fit_one(rows, MODEL_NAMES, exclude_dataset_index=-1, n_estimators=10,
                    max_depth=2, random_state=0, metric="smape", objective="fforma")
    assert m.objective == "fforma"
    assert m.booster is not None
    assert not m.regressors


def _scale_errors(rows, factor):
    """Multiply every contribution by a constant. Ratios within a series — all this
    objective is entitled to see — are untouched; only the magnitude moves."""
    for r in rows:
        r.errors = {k: v * factor for k, v in r.errors.items()}
    return rows


def test_fforma_scores_are_margins_and_favour_the_right_model():
    """Same construction as the per_model generalisation test: above the 0.5
    threshold `good_on_trend` wins. The booster, trained on every OTHER series,
    must give the winner the larger margin (=> larger weight).

    The fixture's sMAPE values sit around 0.04, which is below the magnitude this
    objective can learn from in 100 rounds (see `_fit_one_fforma`'s note — `grad`
    is proportional to the contributions). Scaling them up is what a real dataset
    supplies naturally: ANP's are ~0.22, and there the same code reaches 0.2160."""
    rows = [_row(i, trend_strength=0.95, seed=10) for i in range(15)]
    rows += [_row(i + 100, trend_strength=0.05, seed=20) for i in range(15)]
    _scale_errors(rows, 20.0)
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0, objective="fforma")

    hi = rows[0]
    model_hi = models[hi.dataset_index]
    assert not model_hi.degenerate, "contributions large enough: it must have learned"
    scores, kind = model_hi.predict_scores(hi.features)
    assert kind == "margin"
    assert scores["good_on_trend"] > scores["good_on_flat"]

    lo = rows[15]
    scores, kind = models[lo.dataset_index].predict_scores(lo.features)
    assert scores["good_on_flat"] > scores["good_on_trend"]


def test_a_fit_that_learned_nothing_is_flagged_degenerate():
    """Uniformly tiny contributions produce gradients too small to move
    `base_score`: every model gets the same margin and the weights are uniform.
    That must be reported, not applied while looking like a real meta-model."""
    rows = [_row(i, trend_strength=0.95, seed=10) for i in range(15)]
    rows += [_row(i + 100, trend_strength=0.05, seed=20) for i in range(15)]
    _scale_errors(rows, 0.001)
    # through the unit: `build_pooled_meta_models` would fall back to per_model
    # here, which is the subject of its own test
    m = MM._fit_one(rows, MODEL_NAMES, exclude_dataset_index=-1, n_estimators=10,
                    max_depth=2, random_state=0, metric="smape", objective="fforma")
    assert m.degenerate is True
    scores, _ = m.predict_scores(rows[0].features)
    assert scores["good_on_trend"] == pytest.approx(scores["good_on_flat"])


def test_the_per_model_objective_is_never_flagged_degenerate():
    """The flag describes an fforma-only failure mode; independent regressors do
    not share it."""
    rows = _dataset(25, seed=11)
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0, objective="per_model")
    assert all(not m.degenerate for m in models.values())


def test_predict_scores_on_a_per_model_fit_returns_errors():
    rows = _dataset(30, seed=4)
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0, objective="per_model")
    m = next(iter(models.values()))
    scores, kind = m.predict_scores(rows[0].features)
    assert kind == "error"
    assert set(scores) == set(MODEL_NAMES)


def test_predict_errors_refuses_an_fforma_fit_with_a_readable_message():
    rows = _dataset(25, seed=6)
    m = MM._fit_one(rows, MODEL_NAMES, exclude_dataset_index=-1, n_estimators=10,
                    max_depth=2, random_state=0, metric="smape", objective="fforma")
    with pytest.raises(ValueError, match="per_model"):
        m.predict_errors(rows[0].features, MODEL_NAMES)


def test_fforma_handles_nan_contributions_by_penalising_not_dropping():
    rows = _dataset(30, seed=7)
    for r in rows[5:12]:
        r.errors["good_on_flat"] = float("nan")
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0, objective="fforma")
    m = next(iter(models.values()))
    assert m.n_train_series == len(rows) - 1, "NaN rows are kept (penalised), not dropped"
    scores, _ = m.predict_scores(rows[0].features)
    assert all(np.isfinite(v) for v in scores.values())


def test_an_unknown_objective_is_rejected():
    rows = _dataset(25, seed=8)
    with pytest.raises(ValueError, match="unknown objective"):
        MM.build_pooled_meta_models(rows, MODEL_NAMES, objective="banana")


def test_loso_still_holds_under_the_fforma_objective():
    rows = _dataset(30, seed=9)
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, random_state=0, objective="fforma")
    for row in rows:
        assert models[row.dataset_index].n_train_series == len(rows) - 1


def test_an_all_degenerate_fforma_run_falls_back_to_per_model():
    """Caught on a 25-series smoke run: every fforma fit came back degenerate (the
    gradient is proportional to the contributions, so a small run never moves
    `base_score`), leaving uniform weights that are silently just the mean. The
    full 182-series run learns fine, so this is a small-sample floor, not a bug in
    the objective — and `per_model` has no such floor."""
    rows = [_row(i, trend_strength=0.95, seed=10) for i in range(12)]
    rows += [_row(i + 100, trend_strength=0.05, seed=20) for i in range(12)]
    _scale_errors(rows, 0.001)  # forces every fforma fit to learn nothing
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, min_series=1,
                                         random_state=0, objective="fforma")
    assert all(m.objective == "per_model" for m in models.values())
    assert all(not m.degenerate for m in models.values())
    scores, kind = next(iter(models.values())).predict_scores(rows[0].features)
    assert kind == "error"


def test_a_healthy_fforma_run_is_not_downgraded():
    rows = [_row(i, trend_strength=0.95, seed=10) for i in range(15)]
    rows += [_row(i + 100, trend_strength=0.05, seed=20) for i in range(15)]
    _scale_errors(rows, 20.0)
    models = MM.build_pooled_meta_models(rows, MODEL_NAMES, min_series=1,
                                         random_state=0, objective="fforma")
    assert all(m.objective == "fforma" for m in models.values())
