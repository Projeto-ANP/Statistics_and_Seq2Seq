"""Per-series choice of the scoring weights (`score_preset="auto_loo"`).

The harness must pick the weights by leave-one-window-out on the three
validation windows and re-rank the history with them. The test window (its
actual values) must never be reachable from the state: it lives in the pipeline
outcome, not in `ReactState`, and everything here only reads
`Attempt.per_window`, `aggregate` and the mean anchor, all computed on
validation.

Run:  python -m pytest tests/test_auto_score_weights.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import pool as POOL
from orchestrator_react.config import SCORE_PRESETS, ReactConfig
from orchestrator_react.react_loop import run_react_loop
from orchestrator_react.state import ReactState

from test_orchestrator_react import make_state  # noqa: E402


def test_state_does_not_carry_test_actuals():
    """The agent-side state must not contain the real test values."""
    state = make_state()
    assert not hasattr(state, "test_values")
    assert not hasattr(state, "y_test")
    # test_preds exists (the agent applies strategies to them), but it is not
    # the actual series: it is shaped (n_models, horizon), not (horizon,).
    assert state.test_preds.shape[0] == state.n_models
    assert state.test_preds.ndim == 2


def test_auto_choice_returns_a_preset_and_reranks():
    state = make_state()
    POOL.seed_baselines(state)
    before = [a.score for a in state.attempts]

    choice = state.auto_score_weights()
    assert choice["name"] in SCORE_PRESETS
    assert set(choice["weights"]) == set(SCORE_PRESETS["balanced"])
    assert choice["held_out"] is not None and np.isfinite(choice["held_out"])

    state.apply_score_weights(choice["weights"])
    after = [a.score for a in state.attempts]
    assert len(before) == len(after)
    assert all(len(a.per_window_scores) == state.n_windows for a in state.attempts)

    best = state.best_attempt()
    assert best is not None and np.isfinite(best.score)
    # The best attempt under the applied weights is the argmin of the scores.
    assert best.score == min(a.score for a in state.attempts)


def test_auto_choice_respects_the_candidate_list():
    state = make_state()
    POOL.seed_baselines(state)
    choice = state.auto_score_weights(candidate_presets=["balanced", "scale_free_safe"])
    assert choice["name"] in {"balanced", "scale_free_safe"}
    assert choice["candidates"] == ["balanced", "scale_free_safe"]


def test_auto_choice_skips_cleanly_with_tiny_history():
    state = make_state()
    choice = state.auto_score_weights()
    assert choice["skipped"]
    assert choice["name"] == "balanced"


def test_deterministic_arm_applies_auto_weights():
    cfg = ReactConfig()
    cfg.score_preset = "auto_loo"
    state = make_state(config=cfg)
    POOL.seed_baselines(state)
    result = run_react_loop(
        state=state,
        client=None,
        series_card={},
        pool_card={},
        config=cfg,
    )
    assert result.final_attempt is not None
    assert result.score_weights_name in SCORE_PRESETS
    assert result.stop_reason == "no_llm_client"
    assert result.score_weights  # recorded for the artifacts
