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
from orchestrator_react.config import FINAL_SCORE_PRESETS, SCORE_PRESETS, ReactConfig
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
    assert set(block["presets"]) == set(FINAL_SCORE_PRESETS)
    for name in FINAL_SCORE_PRESETS:
        item = block["presets"][name]
        assert item["attempt_id"]
        assert "test_metrics" in item
        assert len(item["forecast"]) == len(actual)
    assert block["agent"]["final_score_preset"] == "smape60_pocid40"
    assert "test_metrics" in block["agent"]
