"""v2 catalog and prompt: consolidation, merged tools and the leader block.

The v2 catalog hides the entries the v5 runs never called and replaces the
duplicated ones with merged tools, keeping the advertised action space at ten
(accept included). The v2 prompt states a procedure, groups the catalog, shows
two worked examples and feeds the deterministic LEADER DIAGNOSTICS block.

Run:  python -m pytest tests/test_catalog_v2.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import pool as POOL
from orchestrator_react import prompts as P
from orchestrator_react import registry as R
from orchestrator_react import tools as T
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import ScriptedLLM
from orchestrator_react.react_loop import run_react_loop

from test_orchestrator_react import make_state  # noqa: E402


def test_v2_catalog_keeps_the_core_actions_and_accept():
    cfg = ReactConfig()
    cfg.catalog_v2 = True
    cfg.extra_actions = False  # the ten-action catalog is the ablation arm
    # Three validation windows: weights_ols is withheld by the window gate.
    expected = {
        "select_top_k", "select_stable", "prune_redundant", "select_forward",
        "weights_inverse_error", "weights_error_trend", "weights_meta_model",
        "evaluate_strategy",
    }
    assert set(R.tool_names(R.withheld_tools(cfg, 3))) == expected
    # With enough windows, ols comes back.
    assert "weights_ols" in set(R.tool_names(R.withheld_tools(cfg, 5)))
    # The removed entries are exactly the ones the v5 runs never exercised.
    withheld = R.withheld_tools(cfg, 3)
    for name in R.CATALOG_V2_WITHHELD:
        assert name in withheld


def test_extra_actions_are_on_by_default_and_can_be_disabled():
    cfg = ReactConfig()
    cfg.catalog_v2 = True
    assert cfg.extra_actions  # on by default by decision: agent options, not floor entries
    base = set(R.tool_names(R.withheld_tools(cfg, 3)))
    assert set(R.EXTRA_ACTIONS) <= base
    cfg.extra_actions = False
    without = set(R.tool_names(R.withheld_tools(cfg, 3)))
    assert without == base - set(R.EXTRA_ACTIONS)


def test_weights_meta_model_merges_the_two_scopes():
    state = make_state()
    POOL.seed_baselines(state)
    out = T.weights_meta_model(state, scope="series")
    assert out["scope"] == "series"
    assert out["weights"] in state.weights
    with pytest.raises(ValueError):
        # No pooled meta-model was attached to this state.
        T.weights_meta_model(state, scope="dataset")
    with pytest.raises(ValueError):
        T.weights_meta_model(state, scope="banana")


def test_select_forward_registers_a_refittable_pool():
    state = make_state()
    POOL.seed_baselines(state)
    out = T.select_forward(state, max_steps=5)
    handle = out["pool"]
    assert handle in state.pools
    assert state.pool_recipes[handle].refittable
    attempt, _ = state.evaluate({"combine": "mean", "pool": handle}, origin="agent")
    assert np.isfinite(attempt.score)


def test_new_weight_actions_register_the_expected_recipes():
    state = make_state()
    POOL.seed_baselines(state)
    cases = (
        (T.weights_best_per_horizon, {}, True),
        (T.weights_topk_per_horizon, {"k": 2}, True),
        (T.weights_forward, {"max_steps": 5}, False),
    )
    for fn, kwargs, per_horizon in cases:
        out = fn(state, **kwargs)
        recipe = state.get_weights_recipe(out["weights"])
        assert recipe.per_horizon is per_horizon
        attempt, _ = state.evaluate(
            {"combine": "weighted", "pool": recipe.pool_handle, "weights": out["weights"]},
            origin="agent",
        )
        assert np.isfinite(attempt.score)


def test_prompt_v2_has_the_structure_and_the_examples():
    text = P.build_system_prompt_v2(prompt_format="text")
    for marker in (
        "PROCEDURE:", "WORKED EXAMPLES", "WHEN TO STOP", "Example 1", "Example 2",
        # The catalog line itself carries the trigger, by design.
        "Use when the ranking is unstable",
    ):
        assert marker in text
    xml = P.build_system_prompt_v2(prompt_format="xml")
    assert "<role>" in xml and "<examples>" in xml and "<stopping>" in xml


def test_catalog_descriptions_say_when_to_use():
    """Every advertised tool's description starts with its trigger, and the
    catalog the prompt renders matches what `withheld_tools` allows."""
    cfg = ReactConfig()
    cfg.catalog_v2 = True
    text = P.build_system_prompt_v2(
        withheld_tools=R.withheld_tools(cfg, 3), prompt_format="text"
    )
    assert "Use when the ranking is unstable" in text
    assert "select_stable" in text

    # Disabling the experimental recipes removes them from the catalog entirely.
    cfg.extra_actions = False
    text_without = P.build_system_prompt_v2(
        withheld_tools=R.withheld_tools(cfg, 3), prompt_format="text"
    )
    assert "weights_best_per_horizon" not in text_without
    assert "Use when the best model changes along the horizon" not in text_without

    # Default (on): they are advertised with their trigger line.
    cfg.extra_actions = True
    text_extra = P.build_system_prompt_v2(
        withheld_tools=R.withheld_tools(cfg, 3), prompt_format="text"
    )
    assert "Use when the best model changes along the horizon" in text_extra


def test_turn_prompt_carries_the_leader_block_only_when_given():
    state = make_state()
    POOL.seed_baselines(state)
    common = dict(
        state=state, series_card={}, pool_card={}, scratchpad=[],
        iteration=1, max_iterations=12,
    )
    without = P.build_turn_prompt(**common)
    assert "LEADER DIAGNOSTICS" not in without
    with_leader = P.build_turn_prompt(
        **common,
        leader={"leader_id": "a1", "verdict": "indistinguishable", "runner_up_margin": 0.02},
    )
    assert "LEADER DIAGNOSTICS" in with_leader
    assert "indistinguishable" in with_leader


def test_v2_flags_end_to_end_with_a_scripted_llm():
    cfg = ReactConfig()
    cfg.catalog_v2 = True
    cfg.prompt_v2 = True
    state = make_state(config=cfg)
    POOL.seed_baselines(state)
    client = ScriptedLLM(responses=[
        'Thought: test the plain mean of the full pool.\n'
        'Action: evaluate_strategy\n'
        'Action Input: {"combine": "mean", "pool": "pool_full"}\n',
        'Thought: the leader already looks solid.\n'
        'Action: accept\n'
        'Action Input: {"confidence": 0.6}\n',
    ])
    result = run_react_loop(state=state, client=client, series_card={}, pool_card={}, config=cfg)
    assert result.stop_reason == "agent_accepted"
    system, user = client.calls[0]["system"], client.calls[0]["user"]
    assert "WORKED EXAMPLES" in system
    assert "LEADER DIAGNOSTICS" in user
    # The withheld diagnostics must not be advertised.
    assert "series_profile(" not in system


def test_prompt_v2_no_early_stop_removes_accept_and_stop_rule():
    text = P.build_system_prompt_v2(prompt_format="text", no_early_stop=True)
    assert "NO ACCEPT IN THIS RUN" in text
    assert "accept is disabled" in text
    assert "BUDGET RULES (no accept)" in text
    assert "One tested approach = one evaluate_strategy call" in text
    assert "select_*, prune_*" in text
    assert "accept(attempt_id" not in text
    assert "WHEN TO STOP" not in text
    default = P.build_system_prompt_v2(prompt_format="text")
    assert "WHEN TO STOP" in default
    assert "accept(attempt_id" in default


def test_direct_prompt_lists_score_presets_and_window_mode():
    text = P.build_system_prompt_v2(
        prompt_format="text", no_early_stop=True, direct_models=True,
        score_window_mode="weighted",
    )
    assert "Score presets:" in text
    assert "smape_pocid_50_50" in text
    assert "score_preset" in text
    assert "'weighted'" in text


def test_v10_direct_prompt_matches_the_scoring_budget():
    cfg = ReactConfig(
        no_early_stop=True, scoring_budget=True, prompt_v2=True, catalog_v2=True,
        score_preset="smape_only", score_window_mode="weighted",
        allow_agent_score_preset=False, no_seed_attempts=True,
        allow_best_single=False,
    )
    withheld = R.withheld_tools(cfg, 3)
    text = P.build_system_prompt_v2(
        withheld_tools=withheld, prompt_format="text", no_early_stop=True,
        direct_models=True, score_window_mode="weighted",
        no_seed_attempts=True, allow_agent_score_preset=False,
        score_preset="smape_only", allow_best_single=False,
    )
    assert "select_*" not in text
    assert "weights_*" not in text
    assert "create it in this run" not in text
    assert "preparation actions" not in text
    assert "n_eff" not in text
    assert "diverse_gate" not in text
    assert "0.2/0.3/0.5" in text
    assert "evaluate_strategy" in text
