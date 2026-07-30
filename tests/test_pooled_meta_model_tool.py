"""`weights_pooled_meta_model` — the tool/registry/state integration layer.

`tests/test_meta_model.py` covers the pure functions (LOSO fitting, feature
extraction, softmax). This file covers the parts that only make sense wired into
a real `ReactState`: the withholding gate, and that `resolve_recipe`'s
`pooled_meta_model` branch composes a correct per-fold vector from predicted errors
keyed by model name — which is what let the fold-invariant-pool guard be removed.
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
# pool compatibility (the old guard, now removed)
# ──────────────────────────────────────────────────────────────────────────────


def test_accepts_a_refittable_pool_under_nested_selection():
    """The guard this used to assert is GONE, on purpose. It made the tool unusable:
    on the 182-series ANP run the agent called it once, on a pruned pool (its
    dominant habit), and was rejected. Keying predicted errors by model NAME lets
    each fold compose its own vector, so a per-fold pool is fine."""
    s = make_state()
    _attach_fitted_model(s)
    s.config.nested_selection = True
    top = T.select_top_k(s, k=3)["pool"]
    out = T.weights_pooled_meta_model(s, pool=top)
    assert out["weights"].startswith("w")
    attempt, _ = s.evaluate({"combine": "weighted", "pool": top, "weights": out["weights"]})
    assert np.isfinite(attempt.score)


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
# per-fold composition — the property that replaces the guard
# ──────────────────────────────────────────────────────────────────────────────


def test_the_weights_do_not_depend_on_which_windows_are_passed():
    """The features behind it (trend/seasonal strength, entropy, acf1, catch22) are
    properties of `train_series`, which no fold excludes — so for a FIXED set of
    members the vector must be identical regardless of the windows given."""
    from orchestrator_react.weighting import resolve_recipe

    s = make_state()
    _attach_fitted_model(s)
    handle = T.weights_pooled_meta_model(s, pool=FULL_POOL)["weights"]
    recipe = s.get_weights_recipe(handle)
    idx = s.get_pool(FULL_POOL)
    names = [s.model_names[i] for i in idx]

    w_all, meta_all = resolve_recipe(recipe, s.y_true, s.y_preds[:, idx], names=names)
    w_one, meta_one = resolve_recipe(recipe, s.y_true[:1], s.y_preds[:1][:, idx], names=names)
    assert meta_all["mode"] == meta_one["mode"] == "pooled_meta_model"
    assert np.allclose(w_all, w_one)


def test_a_fold_with_fewer_members_gets_a_correctly_sized_simplex_vector():
    """The property that replaces the old guard: a fold holding a SUBSET of the
    models must get a vector of exactly its own length, still summing to 1, and
    still ordered by the same predicted errors."""
    from orchestrator_react.weighting import resolve_recipe

    s = make_state()
    _attach_fitted_model(s)
    handle = T.weights_pooled_meta_model(s, pool=FULL_POOL)["weights"]
    recipe = s.get_weights_recipe(handle)

    subset = [0, 2, 4]
    names = [s.model_names[i] for i in subset]
    w, meta = resolve_recipe(recipe, s.y_true, s.y_preds[:, subset], names=names)
    assert w.size == len(subset)
    assert w.sum() == pytest.approx(1.0)
    assert meta["n_fold_models"] == len(subset)

    # the ranking within the subset must match the full-pool ranking of the same
    # three models — the softmax is recomputed, not re-ordered
    full_names = list(s.model_names)
    w_full, _ = resolve_recipe(recipe, s.y_true, s.y_preds[:, list(range(s.n_models))],
                               names=full_names)
    assert np.argsort(w)[::-1].tolist() == np.argsort(w_full[subset])[::-1].tolist()


def test_missing_names_degrade_to_uniform_instead_of_crashing():
    """Defence in depth: a caller that resolves this recipe without passing member
    names cannot compose a keyed vector, and must degrade rather than raise
    mid-backtest."""
    from orchestrator_react.weighting import WeightsRecipe, resolve_recipe

    s = make_state()
    recipe = WeightsRecipe(
        method="pooled_meta_model", pool_handle=FULL_POOL,
        params={"predicted_errors": {n: 1.0 for n in s.model_names}},
    )
    w, meta = resolve_recipe(recipe, s.y_true, s.y_preds)  # no names=
    assert meta["mode"] == "pooled_meta_model_no_predictions"
    assert w.sum() == pytest.approx(1.0)
    assert w.size == s.n_models


def test_a_model_with_no_prediction_gets_no_weight():
    """`None` means the meta-model never fit a regressor for that model. It must
    not silently receive uniform weight alongside models that have real
    predictions."""
    from orchestrator_react.weighting import WeightsRecipe, resolve_recipe

    s = make_state()
    errs = {n: 5.0 for n in s.model_names}
    errs[s.model_names[0]] = None
    recipe = WeightsRecipe(
        method="pooled_meta_model", pool_handle=FULL_POOL,
        params={"predicted_errors": errs},
    )
    w, meta = resolve_recipe(recipe, s.y_true, s.y_preds, names=list(s.model_names))
    assert w[0] == pytest.approx(0.0)
    assert w.sum() == pytest.approx(1.0)
    assert meta["n_without_a_prediction"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 seeding — the structural answer to "the agent never calls it"
# ──────────────────────────────────────────────────────────────────────────────


def test_phase2_seeds_the_pooled_strategy_when_a_model_is_attached():
    """On the 182-series ANP v4 run the agent reached for this tool once, and that
    call failed. Seeding it makes it a floor entry evaluated on every series
    instead of an option that depends on the agent's habits."""
    from orchestrator_react import pool as POOL

    s = make_state()
    _attach_fitted_model(s)
    seeded = POOL.seed_baselines(s, stable_pools=(), seed_pooled_meta_model=True)
    pooled = [a for a in seeded if a.spec.get("weights")]
    assert len(pooled) == 1
    recipe = s.get_weights_recipe(pooled[0].spec["weights"])
    assert recipe.method == "pooled_meta_model"
    assert pooled[0].origin == "baseline"


def test_seeding_is_skipped_when_no_model_is_attached():
    """No meta-model (too few series, or xgboost missing) must not add a broken
    seed, and must not raise."""
    from orchestrator_react import pool as POOL

    s = make_state()
    assert s.pooled_meta_model is None
    seeded = POOL.seed_baselines(s, stable_pools=(), seed_pooled_meta_model=True)
    assert all(not a.spec.get("weights") for a in seeded)


def test_a_failing_pooled_seed_never_kills_the_series():
    """It is a floor entry: a missing floor entry is strictly better than a lost
    series, so failure degrades to 'no seed'."""
    from orchestrator_react import pool as POOL

    class Exploding:
        n_train_series = 10
        def predict_errors(self, features, names):
            raise RuntimeError("simulated meta-model failure")

    s = make_state()
    s.pooled_meta_model = Exploding()
    seeded = POOL.seed_baselines(s, stable_pools=(), seed_pooled_meta_model=True)
    assert len(seeded) == len(POOL.SEED_BASELINES)


def test_run_phase2_honours_the_seed_flag():
    from orchestrator_react import pool as POOL
    from orchestrator_react.config import ReactConfig as RC

    for flag, expect_pooled in ((True, 1), (False, 0)):
        s = make_state(config=RC(seed_pooled_meta_model=flag, seed_stable_pools=False))
        _attach_fitted_model(s)
        POOL.run_phase2(s, s.config)
        n = sum(1 for a in s.attempts if a.spec.get("weights"))
        assert n == expect_pooled


# ──────────────────────────────────────────────────────────────────────────────
# the "margin" score kind end-to-end (fforma objective through the tool)
# ──────────────────────────────────────────────────────────────────────────────


def _attach_fforma_model(state, n_train_series: int = 40) -> None:
    rng = np.random.default_rng(1)
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
        objective="fforma",
    )


def test_fforma_fit_flows_through_the_tool_and_backtest():
    s = make_state()
    _attach_fforma_model(s)
    out = T.weights_pooled_meta_model(s, pool=FULL_POOL)
    assert out["objective"] == "fforma"
    recipe = s.get_weights_recipe(out["weights"])
    assert recipe.params["score_kind"] == "margin"
    attempt, _ = s.evaluate({"combine": "weighted", "pool": FULL_POOL, "weights": out["weights"]})
    assert np.isfinite(attempt.score)
    fc, _ = s.apply_to_test(attempt.spec)
    assert np.all(np.isfinite(fc))


def test_margin_weights_on_a_subset_renormalise_the_full_softmax():
    """softmax over a subset of margins == the full softmax renormalised — the
    property that makes per-fold membership need no special handling."""
    from orchestrator_react.weighting import WeightsRecipe, resolve_recipe

    s = make_state()
    margins = {n: float(i) for i, n in enumerate(s.model_names)}
    recipe = WeightsRecipe(method="pooled_meta_model", pool_handle=FULL_POOL,
                           params={"model_scores": margins, "score_kind": "margin"})
    all_names = list(s.model_names)
    w_full, _ = resolve_recipe(recipe, s.y_true, s.y_preds, names=all_names)
    sub = [1, 3, 4]
    names = [all_names[i] for i in sub]
    w_sub, meta = resolve_recipe(recipe, s.y_true, s.y_preds[:, sub], names=names)
    assert meta["score_kind"] == "margin"
    expected = w_full[sub] / w_full[sub].sum()
    assert w_sub == pytest.approx(expected)


def test_a_member_without_a_margin_gets_zero_weight():
    from orchestrator_react.weighting import WeightsRecipe, resolve_recipe

    s = make_state()
    margins = {n: 1.0 for n in s.model_names}
    margins[s.model_names[2]] = None
    recipe = WeightsRecipe(method="pooled_meta_model", pool_handle=FULL_POOL,
                           params={"model_scores": margins, "score_kind": "margin"})
    w, _ = resolve_recipe(recipe, s.y_true, s.y_preds, names=list(s.model_names))
    assert w[2] == pytest.approx(0.0)
    assert w.sum() == pytest.approx(1.0)


def test_legacy_predicted_errors_params_still_resolve():
    """Recipes registered before the rename (predicted_errors, no score_kind) must
    keep resolving as the error kind."""
    from orchestrator_react.weighting import WeightsRecipe, resolve_recipe

    s = make_state()
    recipe = WeightsRecipe(method="pooled_meta_model", pool_handle=FULL_POOL,
                           params={"predicted_errors": {n: 1.0 for n in s.model_names}})
    w, meta = resolve_recipe(recipe, s.y_true, s.y_preds, names=list(s.model_names))
    assert meta["score_kind"] == "error"
    assert w.sum() == pytest.approx(1.0)


def test_phase2_seed_works_with_an_fforma_fit():
    from orchestrator_react import pool as POOL

    s = make_state()
    _attach_fforma_model(s)
    seeded = POOL.seed_baselines(s, stable_pools=(), seed_pooled_meta_model=True)
    pooled = [a for a in seeded if a.spec.get("weights")]
    assert len(pooled) == 1
    assert s.get_weights_recipe(pooled[0].spec["weights"]).params["score_kind"] == "margin"
