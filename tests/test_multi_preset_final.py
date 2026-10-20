"""One-level final choice among the three fixed SMAPE/POCID criteria."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import csv_writer as W
from orchestrator_react import prompts as P
from orchestrator_react import tools as T
from orchestrator_react.config import FINAL_SCORE_PRESETS, SCORE_PRESETS, SEED_TEST_PRESETS, ReactConfig
from orchestrator_react.llm import ScriptedLLM
from orchestrator_react.react_loop import run_react_loop

from test_react_loop import prepared, step


def multi_config(**kwargs):
    cfg = ReactConfig(
        max_iterations=2,
        scoring_budget=True,
        max_total_turns=4,
        multi_preset_final=True,
        score_window_mode="weighted",
        **kwargs,
    )
    return cfg


def test_every_evaluation_reports_the_three_scores_and_ranks():
    cfg = multi_config()
    s, _, _ = prepared(cfg)
    out = T.evaluate_strategy(s, {"combine": "mean", "models": ["good_a"]})
    assert set(out["scores_by_preset"]) == set(FINAL_SCORE_PRESETS)
    assert set(out["rank_by_preset"]) == set(FINAL_SCORE_PRESETS)
    line = P.summarize_observation("evaluate_strategy", True, out)
    assert "leaders 50/50=" in line
    assert "60/40=" in line and "40/60=" in line
    assert set(out["best_by_preset"]) == set(FINAL_SCORE_PRESETS)
    assert set(out["seed_best_by_preset"]) == set(FINAL_SCORE_PRESETS)
    for name in FINAL_SCORE_PRESETS:
        assert np.isfinite(out["scores_by_preset"][name])
        assert out["rank_by_preset"][name] >= 1


def test_evaluate_rejects_a_preset_during_the_loop():
    cfg = multi_config()
    s, _, _ = prepared(cfg)
    with pytest.raises(ValueError, match="final decision"):
        T.evaluate_strategy(
            s, {"combine": "mean", "models": ["good_a"]},
            score_preset="smape_pocid_50_50",
        )


def test_turn_prompt_carries_the_three_seed_comparisons():
    cfg = multi_config()
    s, series, pool = prepared(cfg)
    text = P.build_turn_prompt(
        s, series, pool, [], 1, 4,
        score_presets=FINAL_SCORE_PRESETS,
        score_window_mode="weighted",
    )
    assert "SCORE RANKINGS" in text
    for name in FINAL_SCORE_PRESETS:
        assert name in text
    assert "best seed" in text
    assert "best overall" in text


def test_system_prompt_explains_the_final_choice():
    text = P.build_system_prompt_v2(
        prompt_format="text", no_early_stop=True, direct_models=True,
        score_window_mode="weighted", multi_preset_final=True,
    )
    assert "MULTI-PRESET FINAL" in text
    assert "Do NOT pass score_preset" in text
    assert all(name in text for name in FINAL_SCORE_PRESETS)


def test_agent_chooses_a_preset_and_the_best_attempt_under_it_is_applied():
    cfg = multi_config()
    s, series, pool = prepared(cfg)
    llm = ScriptedLLM([
        step("evaluate_strategy", {"combine": "mean", "models": ["good_a"]}),
        step("evaluate_strategy", {"combine": "mean", "models": ["good_a", "good_b"]}),
        '{"score_preset": "smape60_pocid40"}',
    ])
    r = run_react_loop(s, llm, series, pool, cfg)
    assert r.final_score_preset == "smape60_pocid40"
    assert r.score_weights_name == "smape60_pocid40"
    assert s.active_score_weights() == SCORE_PRESETS["smape60_pocid40"]
    assert r.final_attempt is s.best_attempt_for_preset("smape60_pocid40")
    assert r.preset_board.keys() == set(FINAL_SCORE_PRESETS)
    assert "FINAL SCORE CHOICE" in llm.calls[-1]["user"]
    assert "smape60_pocid40" in llm.calls[-1]["user"]
    assert any(t.get("action") == "choose_score_preset" for t in r.trajectory)
    assert r.step_details[0]["observation"]["scores_by_preset"]


def test_no_llm_uses_the_50_50_baseline():
    cfg = multi_config()
    s, series, pool = prepared(cfg)
    r = run_react_loop(s, None, series, pool, cfg)
    assert r.final_score_preset == "smape_pocid_50_50"
    assert r.final_attempt is s.best_attempt_for_preset("smape_pocid_50_50")
    assert r.preset_board.keys() == set(FINAL_SCORE_PRESETS)


def test_invalid_final_choice_falls_back_to_50_50():
    cfg = multi_config()
    s, series, pool = prepared(cfg)
    llm = ScriptedLLM([
        step("evaluate_strategy", {"combine": "mean", "models": ["good_a"]}),
        step("evaluate_strategy", {"combine": "mean", "models": ["good_a", "good_b"]}),
        "I cannot choose.",
    ])
    r = run_react_loop(s, llm, series, pool, cfg)
    assert r.final_score_preset == "smape_pocid_50_50"
    assert any("falling back" in e for e in r.errors)
    assert r.final_attempt is s.best_attempt_for_preset("smape_pocid_50_50")


def test_final_prompt_lists_the_three_rankings():
    cfg = multi_config()
    s, _, _ = prepared(cfg)
    text = P.build_final_preset_prompt(s)
    assert "FINAL SCORE CHOICE" in text
    for name in FINAL_SCORE_PRESETS:
        assert name in text
    assert '{"score_preset"' in text


def test_artifacts_store_the_three_seed_test_baselines():
    cfg = multi_config()
    s, _, _ = prepared(cfg)
    actual = list(np.asarray(s.y_true[-1], dtype=float))
    react = SimpleNamespace(
        final_attempt=s.best_attempt(),
        final_score_preset="smape60_pocid40",
    )
    outcome = SimpleNamespace(
        state=s,
        react=react,
        test_values=actual,
        forecast=actual,
    )
    block = W._seed_preset_tests(outcome)
    assert set(block["presets"]) == set(SEED_TEST_PRESETS)
    for name in SEED_TEST_PRESETS:
        item = block["presets"][name]
        assert item["attempt_id"]
        assert "test_metrics" in item
        assert len(item["forecast"]) == len(actual)
    assert block["agent"]["final_score_preset"] == "smape60_pocid40"
    assert "test_metrics" in block["agent"]


def test_no_seed_attempts_starts_empty_and_the_agent_can_finish():
    cfg = ReactConfig(
        max_iterations=2,
        scoring_budget=True,
        max_total_turns=4,
        multi_preset_final=True,
        no_seed_attempts=True,
        prompt_v2=True,
        catalog_v2=True,
    )
    s, series, pool = prepared(cfg)
    assert not s.attempts
    assert len(s.pools) > 1, "stable pools must be registered as preparation"
    llm = ScriptedLLM([
        step("evaluate_strategy", {"combine": "mean", "models": ["good_a"]}),
        step("evaluate_strategy", {"combine": "mean", "models": ["good_a", "good_b"]}),
        '{"score_preset": "smape_pocid_50_50"}',
    ])
    r = run_react_loop(s, llm, series, pool, cfg)
    assert r.final_attempt is not None
    assert r.final_attempt.origin == "agent"
    assert r.final_score_preset == "smape_pocid_50_50"
    assert "NO pre-evaluated seeded baselines" in llm.calls[0]["system"]
    assert "no seeded baseline" in llm.calls[0]["user"]


def test_no_seed_prompt_removes_the_seed_floor_language():
    text = P.build_system_prompt_v2(
        prompt_format="text", no_early_stop=True, no_seed_attempts=True
    )
    assert "There are NO pre-evaluated seeded baselines" in text
    assert "The seeded baselines are already scored" not in text


def test_agent_cannot_choose_the_score_preset_when_locked():
    cfg = ReactConfig(allow_agent_score_preset=False, score_preset="smape_only")
    s, _, _ = prepared(cfg)
    with pytest.raises(ValueError, match="scoring criterion is fixed"):
        T.evaluate_strategy(
            s, {"combine": "mean", "models": ["good_a"]},
            score_preset="smape_pocid_50_50",
        )


def test_locked_smape_only_prompt_states_the_fixed_criterion():
    text = P.build_system_prompt_v2(
        prompt_format="text", no_early_stop=True, direct_models=True,
        allow_agent_score_preset=False, score_preset="smape_only",
    )
    assert "FIXED to 'smape_only'" in text
    assert "POCID is" in text
    assert "You may also choose a scoring preset" not in text


def test_ranking_trace_saves_full_validation_and_test_orders():
    cfg = ReactConfig(score_preset="smape_only", allow_agent_score_preset=False)
    s, _, _ = prepared(cfg)
    actual = list(np.asarray(s.y_true[-1], dtype=float))
    react = SimpleNamespace(final_attempt=s.best_attempt(), final_score_preset="")
    outcome = SimpleNamespace(state=s, react=react, test_values=actual, forecast=actual)
    trace = W._ranking_trace(outcome)
    assert trace["criterion"] == "smape_only"
    assert len(trace["attempts"]) == len(s.attempts)
    assert len(trace["top3_validation"]) == 3
    assert len(trace["top3_test"]) == 3
    assert trace["final"]["id"] == s.best_attempt().attempt_id
    assert trace["oracle"]["id"] is not None
    assert all("test_rank" in r for r in trace["attempts"] if "test_metrics" in r)
    assert all("test_forecast" in r for r in trace["attempts"] if "test_metrics" in r)


def test_locked_prompt_does_not_advertise_the_score_preset_argument():
    cfg = ReactConfig(allow_agent_score_preset=False, score_preset="smape_only")
    withheld = {name: "not available" for name in ("series_profile",)}
    text = P.build_system_prompt_v2(
        prompt_format="text", no_early_stop=True, direct_models=True,
        withheld_tools=withheld, allow_agent_score_preset=False,
        score_preset="smape_only",
    )
    assert "score_preset=None" not in text
    assert "Do not" in text and "score_preset." in text


def test_turn_prompt_shows_scores_and_total_turns():
    cfg = ReactConfig(score_preset="smape_only", allow_agent_score_preset=False)
    s, series, pool = prepared(cfg)
    text = P.build_turn_prompt(
        s, series, pool, [], 2, 24,
        no_accept=True, score_target=12, scores_used=3,
    )
    assert "SCORES: 3/12" in text
    assert "TURNS: 1/24" in text


def test_no_seed_prompt_lists_optional_classic_starting_points():
    text = P.build_system_prompt_v2(
        prompt_format="text", no_early_stop=True, direct_models=True,
        no_seed_attempts=True, allow_agent_score_preset=False,
        score_preset="smape_only",
    )
    assert "CLASSIC STARTING POINTS (optional" in text
    assert "mean(pool_full)" in text
    assert "weighted(pool_full" in text
    assert "There are NO pre-evaluated seeded baselines" in text


def test_no_seed_untried_menu_offers_classic_weighted_method():
    cfg = ReactConfig(no_seed_attempts=True, score_preset="smape_only")
    s, _, pool = prepared(cfg)
    cands = P.untried_candidates(s, pool, limit=50)
    assert any(
        c.get("combine") == "weighted" and c.get("weights") == "inverse_error"
        for c in cands
    )


def test_turn_prompt_lists_available_weight_methods():
    cfg = ReactConfig(score_preset="smape_only", no_seed_attempts=True)
    s, series, pool = prepared(cfg)
    text = P.build_turn_prompt(
        s, series, pool, [], 1, 24, no_accept=True,
        score_target=12, scores_used=0,
    )
    assert "WEIGHT METHODS available for evaluate_strategy['weights']" in text
    assert "inverse_error" in text
    assert "softmax_neg_error" in text
    assert "ols" not in text.split("WEIGHT METHODS", 1)[1].split("</weight_methods>", 1)[0]
    assert "pooled_meta_model" not in text.split("WEIGHT METHODS", 1)[1].split("</weight_methods>", 1)[0]


def test_best_single_disabled_rejects_single_model_combinations():
    cfg = ReactConfig(allow_best_single=False, score_preset="smape_only",
                      allow_agent_score_preset=False)
    s, _, _ = prepared(cfg)
    with pytest.raises(ValueError, match="combination-only"):
        T.evaluate_strategy(
            s, {"combine": "best_single", "model": "good_a"},
        )
    with pytest.raises(ValueError, match="at least two models"):
        T.evaluate_strategy(
            s, {"combine": "mean", "models": ["good_a"]},
        )
    out = T.evaluate_strategy(
        s, {"combine": "mean", "models": ["good_a", "good_b"]},
    )
    assert out["id"]


def test_no_best_single_prompt_and_menu_when_disabled():
    cfg = ReactConfig(allow_best_single=False, score_preset="smape_only",
                      no_seed_attempts=True)
    s, series, pool = prepared(cfg)
    text = P.build_system_prompt_v2(
        prompt_format="text", no_early_stop=True, direct_models=True,
        no_seed_attempts=True, allow_agent_score_preset=False,
        score_preset="smape_only", allow_best_single=False,
    )
    assert "COMBINATION-ONLY" in text
    assert '"combine":"best_single"' not in text
    candidates = P.untried_candidates(s, pool, limit=50)
    assert not any(c.get("combine") == "best_single" for c in candidates)


def test_turn_prompt_has_decision_card_at_the_end():
    cfg = ReactConfig(score_preset="smape_only", no_seed_attempts=True,
                      allow_best_single=False)
    s, series, pool = prepared(cfg)
    text = P.build_turn_prompt(
        s, series, pool, [], 1, 24, no_accept=True,
        untried=P.untried_candidates(s, pool),
        score_target=12, scores_used=0,
    )
    assert "SERIES DECISION CARD" in text
    assert text.index("SERIES DECISION CARD") > text.index("LAST OBSERVATION") if "LAST OBSERVATION" in text else True
    assert "order arbitrary, NOT a ranking" in text
    assert "name one feature in your Thought" in text


def test_turn_prompt_has_forecast_shape_without_test_actual():
    cfg = ReactConfig(score_preset="smape_only", no_seed_attempts=True,
                      allow_best_single=False)
    s, series, pool = prepared(cfg)
    text = P.build_turn_prompt(
        s, series, pool, [], 1, 24, no_accept=True,
        untried=P.untried_candidates(s, pool),
        score_target=12, scores_used=0,
    )
    assert "FORECAST SHAPE" in text
    assert "test actual is NOT shown" in text
    assert "last_validation_actual_z" in text
    assert '"models"' in text
    assert '"clusters"' in text
    assert '"detail_models"' in text
    assert "test_detail_z" in text
    assert "test_pairwise_corr" in text
    assert "all models are summarized" in text
