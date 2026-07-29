"""Nested pool selection — the fix for the leak that made validation anti-predictive.

Before this protocol a pool was chosen once, on all three validation windows, and
then scored on those same three windows. The score that ranked a strategy had
already seen the selection it was ranking. Measured on 111 NN5 series, ranking
sixteen fixed rules by in-sample validation against blind test gave Spearman
**-0.718**: looking better in validation predicted looking *worse* on the test.

These tests pin the mechanism, not the dataset number: membership must be
re-chosen inside each fold, the excluded window must not influence it, and
nothing may reach past the validation block.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator_react import selection as S
from orchestrator_react import tools as T
from orchestrator_react.config import ReactConfig
from orchestrator_react.state import FULL_POOL, ReactState


HORIZON = 6


def make_flipping_state(nested: bool = True, n_windows: int = 3) -> ReactState:
    """A pool where the best model changes from window to window.

    Model `m` is near-perfect on window `m` and poor elsewhere, so which models
    `select_top_k` picks depends entirely on which windows it looks at. That is
    what makes per-fold refitting observable rather than a no-op.
    """
    rng = np.random.default_rng(0)
    n_models = n_windows + 2
    y_true = rng.normal(100.0, 5.0, size=(n_windows, HORIZON))
    y_preds = np.empty((n_windows, n_models, HORIZON))
    for w in range(n_windows):
        for m in range(n_models):
            err = 0.1 if m == w else 10.0 + m
            y_preds[w, m] = y_true[w] + err
    test_preds = np.stack([y_true.mean(axis=0) + m for m in range(n_models)])
    cfg = ReactConfig(nested_selection=nested)
    return ReactState(
        y_true=y_true, y_preds=y_preds, test_preds=test_preds,
        model_names=[f"m{i}" for i in range(n_models)], config=cfg,
    )


# ──────────────────────────────────────────────────────────────────────────────
# the selectors themselves
# ──────────────────────────────────────────────────────────────────────────────


def test_top_k_on_different_windows_selects_different_models():
    s = make_flipping_state()
    a = S.top_k_indices(s.y_true[[0]], s.y_preds[[0]], k=1)
    b = S.top_k_indices(s.y_true[[1]], s.y_preds[[1]], k=1)
    assert a.tolist() == [0]
    assert b.tolist() == [1]


def test_stable_selection_degenerates_to_top_k_on_a_single_window():
    s = make_flipping_state()
    one = S.stable_indices(s.y_true[[2]], s.y_preds[[2]], k=2)
    assert one.tolist() == S.top_k_indices(s.y_true[[2]], s.y_preds[[2]], k=2).tolist()


def test_k_is_clipped_to_the_pool_size():
    s = make_flipping_state()
    assert len(S.top_k_indices(s.y_true, s.y_preds, k=999)) == s.n_models
    assert len(S.stable_indices(s.y_true, s.y_preds, k=0)) == 1


def test_pruning_never_returns_an_empty_pool():
    s = make_flipping_state()
    kept = S.prune_redundant_indices(
        s.y_true, s.y_preds, base=list(range(s.n_models)), corr_threshold=-1.0
    )
    assert len(kept) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# per-fold membership
# ──────────────────────────────────────────────────────────────────────────────


def test_the_scored_window_does_not_vote_on_its_own_pool():
    """The whole point: model `w` is the best on window `w`, so an in-sample pool
    always contains it. A nested fold must not."""
    s = make_flipping_state(nested=True)
    pool = T.select_top_k(s, k=1)["pool"]
    for w in range(s.n_windows):
        assert w not in s.pool_for_window(pool, exclude_window=w), (
            f"window {w} selected the model that only wins on window {w}"
        )


def test_with_nesting_off_the_scored_window_does_vote(monkeypatch):
    s = make_flipping_state(nested=False)
    pool = T.select_top_k(s, k=1)["pool"]
    for w in range(s.n_windows):
        assert s.pool_for_window(pool, exclude_window=w) == s.get_pool(pool)


def test_membership_actually_changes_between_folds():
    s = make_flipping_state(nested=True)
    pool = T.select_top_k(s, k=2)["pool"]
    seen = {tuple(s.pool_for_window(pool, w)) for w in range(s.n_windows)}
    assert len(seen) > 1, "refitting produced the same pool on every fold"


def test_the_final_application_uses_the_all_window_membership():
    """`apply_to_test` has no target window to exclude — the test is blind — so it
    applies the pool the agent was actually shown."""
    s = make_flipping_state(nested=True)
    pool = T.select_top_k(s, k=2)["pool"]
    assert s.pool_for_window(pool, exclude_window=None) == s.get_pool(pool)


def test_a_pool_the_agent_named_by_hand_is_constant_across_folds():
    """An explicit list has no recipe: there is nothing to re-choose, and pretending
    otherwise would silently change what the agent asked for."""
    s = make_flipping_state(nested=True)
    handle = s.register_pool([0, 1], origin="manual")
    for w in range(s.n_windows):
        assert s.pool_for_window(handle, w) == [0, 1]


def test_the_full_pool_is_constant_across_folds():
    s = make_flipping_state(nested=True)
    for w in range(s.n_windows):
        assert s.pool_for_window(FULL_POOL, w) == list(range(s.n_models))


def test_selection_folds_are_leave_one_out_not_expanding():
    """Weights follow `backtest_mode`; selection does not. Under `expanding`,
    window 0 has no prior window, so the fold would fall back to the all-window
    pool — reinstating the leak on a third of the folds."""
    s = make_flipping_state(nested=True)
    s.config.backtest_mode = "expanding"
    assert s._selection_windows(0) == [1, 2]
    assert s._selection_windows(1) == [0, 2]
    assert s._selection_windows(None) == [0, 1, 2]


def test_nesting_changes_the_backtest_score():
    s_off = make_flipping_state(nested=False)
    s_on = make_flipping_state(nested=True)
    spec = {"combine": "mean", "pool": T.select_top_k(s_off, k=1)["pool"]}
    off, _ = s_off.evaluate(spec)
    on, _ = s_on.evaluate({"combine": "mean", "pool": T.select_top_k(s_on, k=1)["pool"]})
    assert off.score != on.score
    assert off.score < on.score, "the leaky protocol must look better on validation"


# ──────────────────────────────────────────────────────────────────────────────
# handle identity
# ──────────────────────────────────────────────────────────────────────────────


def test_a_recipe_selecting_everything_still_collapses_onto_the_full_pool():
    s = make_flipping_state(nested=True)
    assert T.select_top_k(s, k=s.n_models)["pool"] == FULL_POOL


def test_two_recipes_agreeing_on_all_windows_but_not_per_fold_stay_separate():
    """Reusing a handle here would silently swap one recipe for the other."""
    s = make_flipping_state(nested=True)
    static = s.register_pool(S.top_k_indices(s.y_true, s.y_preds, k=2), origin="manual")
    dynamic = T.select_top_k(s, k=2)["pool"]
    assert s.get_pool(static) == s.get_pool(dynamic)
    assert static != dynamic


def test_the_same_recipe_twice_reuses_its_handle():
    s = make_flipping_state(nested=True)
    assert T.select_top_k(s, k=2)["pool"] == T.select_top_k(s, k=2)["pool"]
    assert T.select_stable(s, k=2)["pool"] == T.select_stable(s, k=2)["pool"]


# ──────────────────────────────────────────────────────────────────────────────
# leakage
# ──────────────────────────────────────────────────────────────────────────────


def test_selection_never_reads_the_test_window():
    """Poison the blind window: nothing about the backtest may move."""
    s = make_flipping_state(nested=True)
    pool = T.select_stable(s, k=2)["pool"]
    before = s.evaluate({"combine": "mean", "pool": pool})[0].score
    folds_before = [s.pool_for_window(pool, w) for w in range(s.n_windows)]

    s.test_preds[:] = 1e9
    s._nested_pool_cache.clear()
    s2 = make_flipping_state(nested=True)
    pool2 = T.select_stable(s2, k=2)["pool"]
    after = s2.evaluate({"combine": "mean", "pool": pool2})[0].score
    folds_after = [s.pool_for_window(pool, w) for w in range(s.n_windows)]

    assert before == pytest.approx(after)
    assert folds_before == folds_after


def test_weights_are_fit_over_the_folds_own_membership():
    """Once membership varies per fold, a weight vector fit on the all-window pool
    would be the wrong length — or worse, the right length for the wrong models."""
    s = make_flipping_state(nested=True)
    pool = T.select_top_k(s, k=2)["pool"]
    handle = T.weights_inverse_error(s, pool=pool)["weights"]
    attempt, _ = s.evaluate({"combine": "weighted", "pool": pool, "weights": handle})
    assert np.isfinite(attempt.score)


def test_the_protocol_is_recorded_in_the_run_fingerprint():
    a, b = ReactConfig(), ReactConfig(nested_selection=False)
    assert a.fingerprint() != b.fingerprint()
    assert a.to_dict()["nested_selection"] is True
