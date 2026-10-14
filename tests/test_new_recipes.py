"""New combination recipes tested offline, before entering the tool catalog.

Covers the per-horizon recipes (`best_per_horizon`, `topk_mean_per_horizon`),
Caruana forward selection (weights and pool recipes), the integration of the
new recipes with `resolve_recipe`/`resolve_pool_recipe`, and the active scoring
weights used by `ReactState.evaluate` after `apply_score_weights`.

Run:  python -m pytest tests/test_new_recipes.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import pool as POOL
from orchestrator_react.config import SCORE_PRESETS
from orchestrator_react.metrics import composite_score
from orchestrator_react.selection import PoolRecipe, forward_indices, resolve_pool_recipe
from orchestrator_react.state import FULL_POOL
from orchestrator_react.weighting import (
    WeightsRecipe,
    forward_selection_counts,
    resolve_recipe,
    weights_best_per_horizon,
    weights_forward_selection,
    weights_topk_mean_per_horizon,
)

from test_orchestrator_react import make_state  # noqa: E402


def _synthetic_complementary(seed: int = 1):
    """Two models whose errors are complementary over the horizon."""
    rng = np.random.default_rng(seed)
    n, h, m = 4, 8, 2
    y_true = rng.normal(size=(n, h))
    y_preds = np.zeros((n, m, h))
    for w in range(n):
        y_preds[w, 0, :] = y_true[w, :] + (0.01 if True else 0.0)
        y_preds[w, 1, :] = y_true[w, :]
    # model 0 good on even steps, model 1 good on odd steps
    y_preds[:, 0, :] = y_true + np.where(np.arange(h) % 2 == 0, 0.01, 5.0)[None, :]
    y_preds[:, 1, :] = y_true + np.where(np.arange(h) % 2 == 1, 0.01, 5.0)[None, :]
    return y_true, y_preds


def _rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - truth) ** 2)))


def test_best_per_horizon_one_hot_per_step():
    y_true, y_preds = _synthetic_complementary()
    weights = weights_best_per_horizon(y_true, y_preds)
    n_models, horizon = y_preds.shape[1], y_preds.shape[2]
    assert weights.shape == (n_models, horizon)
    for h in range(horizon):
        assert np.isclose(weights[:, h].sum(), 1.0)
        expected = 0 if h % 2 == 0 else 1
        assert int(np.argmax(weights[:, h])) == expected


def test_topk_mean_per_horizon_keeps_k_models_per_step():
    y_true, y_preds = _synthetic_complementary()
    weights = weights_topk_mean_per_horizon(y_true, y_preds, k=1)
    for h in range(y_preds.shape[2]):
        assert np.isclose(weights[:, h].sum(), 1.0)
        assert int((weights[:, h] > 0).sum()) == 1


def test_forward_selection_prefers_the_complementary_pair():
    y_true, y_preds = _synthetic_complementary()
    counts = forward_selection_counts(y_true, y_preds, max_steps=10)
    assert np.all(counts > 0)

    weights = weights_forward_selection(y_true, y_preds, max_steps=10)
    assert np.isclose(weights.sum(), 1.0)
    ensemble = np.tensordot(weights, y_preds.transpose(1, 0, 2), axes=1)
    single_best = min(_rmse(y_true, y_preds[:, j, :]) for j in range(y_preds.shape[1]))
    assert _rmse(y_true, ensemble) < single_best


def test_forward_pool_recipe_resolves_to_a_subset():
    y_true, y_preds = _synthetic_complementary()
    recipe = PoolRecipe(method="forward", params={"max_steps": 10, "metric": "rmse"})
    indices = resolve_pool_recipe(recipe, y_true, y_preds)
    assert indices.size >= 1
    assert set(indices.tolist()).issubset(set(range(y_preds.shape[1])))
    assert recipe.refittable
    assert forward_indices(y_true, y_preds).size >= 1


def test_new_methods_resolve_through_weights_recipe():
    y_true, y_preds = _synthetic_complementary()
    for method, params in (
        ("best_per_horizon", {}),
        ("topk_mean_per_horizon", {"k": 1}),
        ("forward_selection", {"max_steps": 8}),
    ):
        recipe = WeightsRecipe(
            method=method, pool_handle=FULL_POOL, per_horizon=True, params=params
        )
        weights, meta = resolve_recipe(recipe, y_true, y_preds)
        assert meta["mode"] == method
        if weights.ndim == 2:
            assert np.allclose(weights.sum(axis=0), 1.0)
        else:
            assert np.isclose(weights.sum(), 1.0)


def test_evaluate_scores_under_the_active_weights():
    state = make_state()
    POOL.seed_baselines(state)
    weights = dict(SCORE_PRESETS["rmse_focus"])
    state.apply_score_weights(weights)
    anchor = state.baseline_aggregate()
    attempt, _ = state.evaluate({"combine": "median", "pool": FULL_POOL})
    expected = composite_score(attempt.aggregate, anchor, weights)
    assert abs(attempt.score - expected) < 1e-12
    assert state.active_score_weights() == weights
