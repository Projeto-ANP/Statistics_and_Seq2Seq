"""`weights_pooled_meta_model` — the tool/registry/state integration layer.

`tests/test_meta_model.py` covers the pure functions (LOSO fitting, feature
extraction, softmax). This file covers the parts that only make sense wired into
a real `ReactState`: the withholding gate, the fold-invariant-pool guard, and that
`resolve_recipe`'s `pooled_meta_model` branch really does reuse one vector across
every fold rather than silently drifting.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator_react import meta_model as MM
from orchestrator_react import registry as R
from orchestrator_react import tools as T
from orchestrator_react.config import ReactConfig
from orchestrator_react.state import FULL_POOL

from test_orchestrator_react import make_state  # noqa: E402

xgboost = pytest.importorskip("xgboost")


def _attach_fitted_model(state, n_train_series: int = 40) -> None:
    """A `PooledMetaModel` with a real regressor per model in `state`'s pool —
    fit on synthetic rows, since this file tests the wiring, not the fit itself."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_train_series):
        profile = {
            "trend_strength": float(rng.uniform(0, 1)),
            "seasonal_strength": float(rng.uniform(0, 1)),
            "features": {"spectral_entropy": float(rng.uniform(0, 1)), "acf1": float(rng.uniform(-1, 1))},
        }
        errors = {name: float(rng.uniform(1, 10)) for name in state.model_names}
        rows.append(MM.MetaRow(dataset_index=i, features=MM.extract_meta_features(profile), errors=errors))
    state.pooled_meta_model = MM._fit_one(
        rows, state.model_names, exclude_dataset_index=-1,
        n_estimators=10, max_depth=2, random_state=0, metric="rmse",
    )


# ──────────────────────────────────────────────────────────────────────────────
# the withholding gate
# ──────────────────────────────────────────────────────────────────────────────


def test_withheld_when_no_model_is_attached():
    s = make_state()
    assert s.pooled_meta_model is None
    withheld = R.withheld_tools(s.config, s.n_windows, state=s)
    assert "weights_pooled_meta_model" in withheld


def test_offered_once_a_model_is_attached():
    s = make_state()
    _attach_fitted_model(s)
    withheld = R.withheld_tools(s.config, s.n_windows, state=s)
    assert "weights_pooled_meta_model" not in withheld


def test_withheld_tools_without_a_state_does_not_crash_or_withhold_it():
    """Back-compat: existing callers that never pass `state` keep working."""
    cfg = ReactConfig()
    withheld = R.withheld_tools(cfg, cfg.n_validation_windows)
    assert "weights_pooled_meta_model" not in withheld


def test_a_withheld_call_is_refused_not_executed():
    s = make_state()
    withheld = R.withheld_tools(s.config, s.n_windows, state=s)
    ok, obs = R.call_tool(s, "weights_pooled_meta_model", {}, withheld=withheld)
    assert ok is False
    assert obs["error"] == "unknown_tool"
    assert not s.weights


def test_calling_it_directly_without_a_model_raises_a_readable_error():
    s = make_state()
    with pytest.raises(ValueError, match="no pooled meta-model"):
        T.weights_pooled_meta_model(s)


# ──────────────────────────────────────────────────────────────────────────────
# the fold-invariant-pool guard
# ──────────────────────────────────────────────────────────────────────────────


def test_refuses_a_refittable_pool_under_nested_selection():
    s = make_state()
    _attach_fitted_model(s)
    s.config.nested_selection = True
    top = T.select_top_k(s, k=3)["pool"]
    with pytest.raises(ValueError, match="re-selected per backtest fold"):
        T.weights_pooled_meta_model(s, pool=top)


def test_full_pool_is_always_allowed():
    s = make_state()
    _attach_fitted_model(s)
    out = T.weights_pooled_meta_model(s, pool=FULL_POOL)
    assert out["weights"].startswith("w")


def test_a_refittable_pool_is_allowed_once_nested_selection_is_off():
    s = make_state()
    _attach_fitted_model(s)
    s.config.nested_selection = False
    top = T.select_top_k(s, k=3)["pool"]
    out = T.weights_pooled_meta_model(s, pool=top)
    assert out["weights"].startswith("w")


def test_a_manually_registered_pool_is_allowed():
    """No `PoolRecipe` at all — a hand-picked list is constant by definition."""
    s = make_state()
    _attach_fitted_model(s)
    handle = s.register_pool([0, 1], origin="manual")
    out = T.weights_pooled_meta_model(s, pool=handle)
    assert out["weights"].startswith("w")


# ──────────────────────────────────────────────────────────────────────────────
# the tool's output
# ──────────────────────────────────────────────────────────────────────────────


def test_the_tool_registers_a_handle_and_reports_provenance():
    s = make_state()
    _attach_fitted_model(s, n_train_series=57)
    out = T.weights_pooled_meta_model(s, pool=FULL_POOL)
    assert out["method"] == "pooled_meta_model"
    assert out["n_train_series"] == 57
    assert 0 <= out["n_models_with_a_fit"] <= s.n_models
    recipe = s.get_weights_recipe(out["weights"])
    assert recipe.method == "pooled_meta_model"


def test_it_is_usable_end_to_end_as_a_strategy():
    s = make_state()
    _attach_fitted_model(s)
    handle = T.weights_pooled_meta_model(s, pool=FULL_POOL)["weights"]
    attempt, _ = s.evaluate({"combine": "weighted", "pool": FULL_POOL, "weights": handle})
    assert np.isfinite(attempt.score)
    forecast, _ = s.apply_to_test(attempt.spec)
    assert forecast.shape == (s.horizon,)
    assert np.all(np.isfinite(forecast))


# ──────────────────────────────────────────────────────────────────────────────
# fold invariance — the property the pool guard exists to protect
# ──────────────────────────────────────────────────────────────────────────────


def test_the_same_weight_vector_is_used_on_every_backtest_fold():
    """Unlike every other weight recipe, this one must NOT change per fold: the
    features it was computed from (trend/seasonal strength, entropy, acf1) come
    from the series' historical shape, which no fold excludes."""
    from orchestrator_react.weighting import resolve_recipe

    s = make_state()
    _attach_fitted_model(s)
    handle = T.weights_pooled_meta_model(s, pool=FULL_POOL)["weights"]
    recipe = s.get_weights_recipe(handle)
    idx = s.get_pool(FULL_POOL)

    w_all, meta_all = resolve_recipe(recipe, s.y_true, s.y_preds[:, idx])
    w_one, meta_one = resolve_recipe(recipe, s.y_true[:1], s.y_preds[:1][:, idx])
    assert meta_all["mode"] == meta_one["mode"] == "pooled_meta_model"
    assert np.allclose(w_all, w_one), "the vector must not depend on which windows were passed in"


def test_a_pool_size_mismatch_falls_back_to_uniform_instead_of_crashing():
    """Defence in depth: if the guard were ever bypassed and the pool at
    resolution time has a different size than at registration, this must degrade
    safely rather than raise mid-backtest."""
    from orchestrator_react.weighting import WeightsRecipe, resolve_recipe

    s = make_state()
    recipe = WeightsRecipe(
        method="pooled_meta_model", pool_handle=FULL_POOL,
        params={"precomputed_weights": [1.0, 0.0]},  # wrong length on purpose
    )
    w, meta = resolve_recipe(recipe, s.y_true, s.y_preds)
    assert meta["mode"] == "pooled_meta_model_pool_mismatch"
    assert w.sum() == pytest.approx(1.0)
    assert w.size == s.n_models
