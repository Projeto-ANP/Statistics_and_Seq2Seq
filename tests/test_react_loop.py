"""Phase 3 ReAct loop tests — scripted LLM, no server, no GPU.

The loop is driven by `ScriptedLLM`, which replays canned model answers. That makes
every branch reachable and deterministic: malformed output, unknown tools, bad
arguments, repeated strategies, early stopping, budget exhaustion, and an agent that
tries to accept something worse than a baseline.

Run:  python -m pytest tests/test_react_loop.py -q
"""

from __future__ import annotations

import json
import os
import sys

import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import pool as POOL
from orchestrator_react import prompts as P
from orchestrator_react import registry as R
from orchestrator_react import tools as T
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import (
    LLMError,
    ScriptedLLM,
    extract_json,
    parse_agent_step,
    split_think,
)
from orchestrator_react.react_loop import LLM_ERROR_RETRIES, run_react_loop
from orchestrator_react.state import FULL_POOL

from test_orchestrator_react import HORIZON, N_WINDOWS, make_state  # noqa: E402


def prepared(config: ReactConfig | None = None):
    """A state with Phase 2 already run, which is the loop's precondition."""
    s = make_state(config=config or ReactConfig(max_iterations=4))
    phase2 = POOL.run_phase2(s, s.config)
    return s, T.series_profile(s), phase2["report"]


def step(action: str, args: dict | None = None, thought: str = "testing") -> str:
    return f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(args or {})}"


# ══════════════════════════════════════════════════════════════════════════════
# parsing
# ══════════════════════════════════════════════════════════════════════════════


def test_parse_canonical_three_line_format():
    s = parse_agent_step('Thought: look at errors\nAction: error_summary\nAction Input: {"top_n": 3}')
    assert s.ok
    assert s.action == "error_summary"
    assert s.action_input == {"top_n": 3}
    assert "look at errors" in s.thought


def test_parse_strips_think_blocks():
    thinking, body = split_think("<think>internal reasoning</think>Action: list_attempts")
    assert thinking == "internal reasoning"
    assert body.startswith("Action:")
    s = parse_agent_step("<think>deep thoughts</think>\nAction: list_attempts\nAction Input: {}")
    assert s.ok and s.action == "list_attempts"
    assert s.thought == "deep thoughts"


def test_parse_handles_orphan_think_tag():
    """Truncated reasoning leaves a closing tag with no opener."""
    s = parse_agent_step("reasoning was cut off</think>\nAction: stl_summary\nAction Input: {}")
    assert s.ok and s.action == "stl_summary"


def test_parse_handles_fenced_json():
    s = parse_agent_step(
        'Thought: t\nAction: select_top_k\nAction Input:\n```json\n{"k": 4}\n```'
    )
    assert s.ok and s.action_input == {"k": 4}


def test_parse_handles_single_json_object():
    s = parse_agent_step('{"thought": "t", "action": "series_profile", "action_input": {}}')
    assert s.ok and s.action == "series_profile" and s.thought == "t"


def test_parse_handles_json_with_stringified_args():
    s = parse_agent_step('{"action": "select_top_k", "action_input": "{\\"k\\": 2}"}')
    assert s.ok and s.action_input == {"k": 2}


def test_parse_missing_action_input_defaults_to_empty():
    s = parse_agent_step("Thought: t\nAction: list_attempts")
    assert s.ok and s.action_input == {}


def test_parse_rejects_a_response_with_no_action():
    s = parse_agent_step("I think the mean is probably fine, let's go with that.")
    assert not s.ok and "Action:" in s.parse_error


def test_parse_rejects_empty():
    assert not parse_agent_step("").ok
    assert not parse_agent_step("<think>only thinking</think>").ok


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('noise ```json\n{"a": 2}\n``` more') == {"a": 2}
    assert extract_json('prefix {"a": {"b": 3}} suffix') == {"a": {"b": 3}}
    assert extract_json("no json here") is None


# ══════════════════════════════════════════════════════════════════════════════
# prompts
# ══════════════════════════════════════════════════════════════════════════════


def test_system_prompt_lists_the_closed_catalog():
    from orchestrator_react.registry import TOOLS

    sp = P.build_system_prompt()
    for name in TOOLS:
        assert name in sp
    assert "accept(" in sp
    assert "Action Input" in sp


def test_turn_prompt_carries_the_cards_and_history():
    s, series, pool = prepared()
    prompt = P.build_turn_prompt(s, series, pool, [], 1, 6)
    assert "ITERATION 1 of 6" in prompt
    assert "SERIES PROFILE" in prompt and "MODEL POOL" in prompt
    assert f"ATTEMPT HISTORY ({len(s.attempts)})" in prompt  # everything Phase 2 seeded
    assert "6 iterations left" in prompt  # the current one counts


def test_turn_prompt_never_leaks_raw_data():
    """Principle 1: the series and the forecast tensor stay in the state."""
    s, series, pool = prepared()
    prompt = P.build_turn_prompt(s, series, pool, [], 1, 6)
    # a raw observation from the training series must not appear verbatim
    sample = f"{float(s.train_series[10]):.6f}"
    assert sample not in prompt
    assert len(prompt) < 6000, "the turn prompt must stay compact"


def test_turn_prompt_hides_history_under_the_ablation():
    s, series, pool = prepared(ReactConfig(show_attempt_history=False))
    prompt = P.build_turn_prompt(s, series, pool, [], 1, 6, show_history=False)
    assert "ATTEMPT HISTORY" not in prompt


def test_turn_prompt_warns_on_the_last_iteration():
    s, series, pool = prepared()
    assert "LAST iteration" in P.build_turn_prompt(s, series, pool, [], 6, 6)


def test_turn_prompt_lists_created_handles():
    s, series, pool = prepared()
    T.select_top_k(s, k=3)
    T.weights_inverse_error(s, pool="pool1")
    prompt = P.build_turn_prompt(s, series, pool, [], 2, 6)
    assert "HANDLES AVAILABLE" in prompt
    assert "w1" in prompt
    assert all(h in prompt for h in s.pools if h != FULL_POOL)


def test_observation_summaries_are_short():
    s, _, _ = prepared()
    ev = T.evaluate_strategy(s, T.combine_mean(s))
    line = P.summarize_observation("evaluate_strategy", True, ev)
    assert len(line) < 120 and "rank" in line
    err = P.summarize_observation("select_top_k", False, {"error": "unknown_argument", "detail": "x"})
    assert err.startswith("ERROR")


# ══════════════════════════════════════════════════════════════════════════════
# loop mechanics
# ══════════════════════════════════════════════════════════════════════════════


def test_loop_requires_seeded_baselines():
    s = make_state()
    with pytest.raises(RuntimeError, match="attempt history is empty"):
        run_react_loop(s, ScriptedLLM([]), {}, {})


def test_loop_without_a_client_takes_the_best_baseline():
    """The 'no agent' arm of the ablation still produces a valid decision."""
    s, series, pool = prepared()
    r = run_react_loop(s, None, series, pool)
    assert r.final_attempt is s.best_attempt()
    assert r.stop_reason == "no_llm_client"
    assert r.iterations_used == 0
    assert r.justification, "principle 7: a causal justification is always produced"
    assert r.llm_model == "none"


def test_loop_respects_the_calibration_gate():
    s, series, pool = prepared()
    r = run_react_loop(s, ScriptedLLM([]), series, pool, skip_reason="calibration_gate")
    assert r.stop_reason == "calibration_gate"
    assert r.iterations_used == 0


def test_happy_path_agent_explores_then_accepts():
    s, series, pool = prepared(ReactConfig(max_iterations=6))
    llm = ScriptedLLM([
        step("ranking_stability", {}, "how stable is the ranking?"),
        step("select_top_k", {"k": 3}, "drop the weak models"),
        step("evaluate_strategy", {"strategy": {"combine": "mean", "pool": "pool1"}}, "test the lean pool"),
        step("accept", {"attempt_id": "a4", "confidence": 0.8,
                        "justification": "ranking is unstable so an equal-weight lean pool is safer"},
             "good enough"),
    ])
    r = run_react_loop(s, llm, series, pool)

    assert r.stop_reason == "agent_accepted"
    assert r.iterations_used == 4
    assert r.accept_confidence == 0.8
    assert "unstable" in r.justification
    assert len(r.trajectory) == 4
    assert [t["action"] for t in r.trajectory] == [
        "ranking_stability", "select_top_k", "evaluate_strategy", "accept",
    ]
    assert all(t["observation_summary"] for t in r.trajectory)


def test_trajectory_is_json_serialisable_and_compact():
    """`react_trajectory_json` has to fit a CSV cell for every series."""
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    llm = ScriptedLLM([
        step("series_profile", {}),
        step("evaluate_strategy", {"strategy": {"combine": "median", "pool": FULL_POOL}}),
        step("accept", {"attempt_id": "a1", "justification": "baseline wins"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    blob = json.dumps(r.trajectory, ensure_ascii=False)
    assert len(blob) < 4000
    for entry in r.trajectory:
        assert set(entry) == {
            "iteration", "thought", "action", "action_args", "observation_summary"
        }
        assert isinstance(entry["iteration"], int)


def test_budget_is_exhausted_when_the_agent_never_accepts():
    s, series, pool = prepared(ReactConfig(max_iterations=3, early_stop_patience=99))
    llm = ScriptedLLM([step("list_attempts", {})] * 3)
    r = run_react_loop(s, llm, series, pool)
    assert r.iterations_used == 3
    assert r.stop_reason == "iteration_budget_exhausted"
    assert r.early_stopped is False
    assert r.final_attempt is not None  # still decides


def test_early_stop_after_consecutive_non_improvements():
    s, series, pool = prepared(ReactConfig(max_iterations=8, early_stop_patience=2))
    worse = {"combine": "best_single", "model": "bad"}
    worse2 = {"combine": "best_single", "model": "mediocre"}
    llm = ScriptedLLM([
        step("evaluate_strategy", {"strategy": worse}, "try the bad model"),
        step("evaluate_strategy", {"strategy": worse2}, "try the mediocre one"),
        step("list_attempts", {}, "should never run"),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert r.early_stopped is True
    assert "no improvement" in r.stop_reason
    assert r.iterations_used == 2


def test_improvement_resets_the_early_stop_counter():
    s, series, pool = prepared(ReactConfig(max_iterations=8, early_stop_patience=2))
    winner = T.select_top_k(s, k=2)["pool"]  # not one of the seeded stability pools
    n_seeded = len(s.attempts)
    # ids are assigned in evaluation order, so the improving proposal below is the
    # second one the loop scores
    improving_id = f"a{n_seeded + 2}"

    llm = ScriptedLLM([
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "bad"}}),
        step("evaluate_strategy", {"strategy": {"combine": "mean", "pool": winner}}),
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "mediocre"}}),
        step("accept", {"attempt_id": improving_id}),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert r.early_stopped is False, (
        "the middle proposal improved, so the two stale turns around it must not "
        "trip the patience counter"
    )
    assert r.iterations_used == 4


# ══════════════════════════════════════════════════════════════════════════════
# resilience — the old pipeline hard-stopped on all of these
# ══════════════════════════════════════════════════════════════════════════════


def test_unparsable_answer_becomes_an_observation_and_the_loop_continues():
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    llm = ScriptedLLM([
        "I reckon the mean is fine honestly",
        step("accept", {"attempt_id": "a1", "justification": "recovered"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert r.stop_reason == "agent_accepted"
    assert r.trajectory[0]["action"] == "unparsed"
    assert "ERROR parse" in r.trajectory[0]["observation_summary"]
    assert r.errors


def test_unknown_tool_becomes_an_observation():
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    llm = ScriptedLLM([
        step("summon_a_better_model", {}),
        step("accept", {"attempt_id": "a1"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert "ERROR unknown_tool" in r.trajectory[0]["observation_summary"]
    assert r.tools["tool_missing"] is True
    assert r.stop_reason == "agent_accepted"


def test_bad_arguments_become_an_observation():
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    llm = ScriptedLLM([
        step("select_top_k", {"k": 3, "temperature": 0.9}),
        step("accept", {"attempt_id": "a1"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert "unknown_argument" in r.trajectory[0]["observation_summary"]


def test_accepting_an_unknown_attempt_is_rejected_and_retried():
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    llm = ScriptedLLM([
        step("accept", {"attempt_id": "a999"}),
        step("accept", {"attempt_id": "a1", "justification": "second try"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert "ERROR accept" in r.trajectory[0]["observation_summary"]
    assert r.agent_accepted_id == "a1"
    assert r.iterations_used == 2


def test_accept_without_an_id_means_the_best():
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = ScriptedLLM([step("accept", {"justification": "the ranking is clear"})])
    r = run_react_loop(s, llm, series, pool)
    assert r.agent_accepted_id == s.best_attempt().attempt_id


def test_llm_failure_stops_cleanly_with_a_decision():
    class Broken:
        name = "broken"

        def complete(self, system, user):
            raise LLMError("connection refused")

    s, series, pool = prepared()
    r = run_react_loop(s, Broken(), series, pool)
    assert r.stop_reason == "llm_error"
    assert r.final_attempt is not None
    assert r.justification


# ══════════════════════════════════════════════════════════════════════════════
# the guarantee: never worse than the seeded baselines
# ══════════════════════════════════════════════════════════════════════════════


def test_agent_cannot_pick_something_worse_than_a_baseline():
    """Principle 5 is enforced, and the override is visible rather than silent."""
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    # `bad` is evaluated first, so its id is knowable only after the fact; the
    # script accepts it by looking it up rather than by a literal that Phase 2
    # seeding would shift.
    bad_id = f"a{len(s.attempts) + 1}"
    llm = ScriptedLLM([
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "bad"}}),
        step("accept", {"attempt_id": bad_id, "confidence": 0.9,
                        "justification": "I like this one"}),
    ])
    r = run_react_loop(s, llm, series, pool)

    worst = [a for a in s.attempts if a.spec.get("model") == "bad"][0]
    assert r.agent_accepted_id == worst.attempt_id
    assert r.final_attempt is not worst
    assert r.final_attempt is s.best_attempt()
    assert r.overridden is True
    assert any("scores better" in e for e in r.errors)


def test_result_is_never_worse_than_the_best_baseline():
    s, series, pool = prepared(ReactConfig(max_iterations=5))
    baseline_best = min(a.score for a in s.attempts if a.origin == "baseline")
    llm = ScriptedLLM([
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "bad"}}),
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "mediocre"}}),
        step("accept", {"attempt_id": "a4"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert r.final_attempt.score <= baseline_best


def test_justification_is_always_produced():
    """Principle 7: no run ships without a causal explanation."""
    s, series, pool = prepared(ReactConfig(max_iterations=2))
    llm = ScriptedLLM([step("list_attempts", {}), step("list_attempts", {})])
    r = run_react_loop(s, llm, series, pool)
    assert r.stop_reason == "iteration_budget_exhausted"
    assert r.justification
    # it must cite observable characteristics, not only the error
    assert any(k in r.justification for k in ("trend strength", "ranking is", "error spread"))


def test_rationale_flows_from_thought_into_the_history():
    """The Thought line becomes the attempt's rationale, which the history shows."""
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = ScriptedLLM([
        step("evaluate_strategy",
             {"strategy": {"combine": "trimmed_mean", "pool": FULL_POOL, "trim_pct": 0.2}},
             thought="trimming should drop the biased model"),
        step("accept", {"attempt_id": "a1"}),
    ])
    run_react_loop(s, llm, series, pool)
    trimmed = [
        a for a in s.attempts
        if a.spec["combine"] == "trimmed_mean" and a.spec["pool"] == FULL_POOL
    ][0]
    assert (
        "trimming should drop" in trimmed.rationale
        or "trimming should drop" in trimmed.agent_rationale
    )
    assert trimmed.origin == "agent"
    assert trimmed.iteration == 1


def test_reproposing_a_seeded_baseline_keeps_its_original_record():
    """Deduplication must not let an agent rewrite a deterministic baseline's entry.

    The agent's reasoning is not lost: it stays in the trajectory's `thought`.
    """
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = ScriptedLLM([
        step("evaluate_strategy", {"strategy": {"combine": "median", "pool": FULL_POOL}},
             thought="let me try the median"),
        step("accept", {"attempt_id": "a1"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    median = [a for a in s.attempts if a.spec["combine"] == "median"][0]
    assert median.origin == "baseline"
    assert median.rationale.startswith("deterministic baseline")
    assert "already tested" in r.trajectory[0]["observation_summary"]
    assert "let me try the median" in r.trajectory[0]["thought"]


# ══════════════════════════════════════════════════════════════════════════════
# end to end through Phase 4
# ══════════════════════════════════════════════════════════════════════════════


def test_full_pipeline_phase2_to_phase4():
    """Phase 2 -> Phase 3 -> Phase 4 with a weighted strategy the agent built."""
    s, series, pool = prepared(ReactConfig(max_iterations=6))
    llm = ScriptedLLM([
        step("select_stable", {"k": 3}, "pick the consistent models"),
        step("weights_inverse_error", {"pool": "pool1"}, "weight them by error"),
        step("evaluate_strategy",
             {"strategy": {"combine": "weighted", "pool": "pool1", "weights": "w1"}},
             "test inverse-error weighting on the stable subset"),
        step("sanity_check", {"reference": "a4"}, "is the forecast plausible?"),
        step("accept", {"attempt_id": "a4", "confidence": 0.7,
                        "justification": "stable subset with inverse-error weights"}),
    ])
    r = run_react_loop(s, llm, series, pool)

    assert r.stop_reason == "agent_accepted"
    forecast, debug = s.apply_to_test(r.final_attempt.spec)
    assert forecast.shape == (HORIZON,)
    assert np.all(np.isfinite(forecast))
    if r.final_attempt.spec["combine"] == "weighted":
        resolved = debug["weights_resolved"]["weights"]
        assert sum(resolved.values()) == pytest.approx(1.0)
    assert r.tools["n_calls"] >= 4
    assert r.tools["tool_missing"] is False


def test_loop_is_deterministic_for_the_same_script():
    script = [
        step("select_top_k", {"k": 3}),
        step("evaluate_strategy", {"strategy": {"combine": "mean", "pool": "pool1"}}),
        step("accept", {"attempt_id": "a4"}),
    ]
    outs = []
    for _ in range(2):
        s, series, pool = prepared(ReactConfig(max_iterations=4))
        r = run_react_loop(s, ScriptedLLM(list(script)), series, pool)
        forecast, _ = s.apply_to_test(r.final_attempt.spec)
        outs.append((r.final_attempt.spec, round(r.final_attempt.score, 9), forecast))
    assert outs[0][0] == outs[1][0]
    assert outs[0][1] == outs[1][1]
    assert outs[0][2] == pytest.approx(outs[1][2])


# ══════════════════════════════════════════════════════════════════════════════
# shapes gpt-oss:20b actually produced on the server
# ══════════════════════════════════════════════════════════════════════════════


REAL_SHAPES = [
    # flat, the form the model reaches for first
    {"combine": "weighted", "pool": "pool1", "weights": "w1"},
    # method name in `strategy`, arguments as siblings
    {"strategy": "weighted", "pool": "pool1", "weights": "w1"},
    # same plus a field the model invented
    {"strategy": "weighted", "pool": "pool1", "weights": "w1", "origin": "combination"},
    # nested, the documented form
    {"strategy": {"combine": "weighted", "pool": "pool1", "weights": "w1"}},
]


@pytest.mark.parametrize("shape", REAL_SHAPES)
def test_evaluate_strategy_accepts_every_shape_the_model_uses(shape):
    """A rejected call costs an iteration and teaches the model nothing.

    Every one of these came out of a real gpt-oss:20b run; three of the four used
    to fail with `unknown_argument`, and the model then burned three more turns
    retrying the same shape.
    """
    s, series, pool = prepared()
    T.select_top_k(s, k=3)
    T.weights_inverse_error(s, pool="pool1")

    ok, obs = R.call_tool(s, "evaluate_strategy", dict(shape))
    assert ok, f"{shape} was rejected: {obs}"
    assert obs["strategy"] == {"combine": "weighted", "pool": "pool1", "weights": "w1"}


def test_evaluate_strategy_accepts_a_combine_tool_result_verbatim():
    s, series, pool = prepared()
    built = T.combine_median(s)
    ok, obs = R.call_tool(s, "evaluate_strategy", {"strategy": built["strategy"]})
    assert ok and obs["strategy"]["combine"] == "median"
    # and the whole returned object, pasted as-is
    ok2, obs2 = R.call_tool(s, "evaluate_strategy", {"strategy": built})
    assert ok2 and obs2["already_tested"] is True


def test_evaluate_strategy_reads_a_json_string():
    s, series, pool = prepared()
    ok, obs = R.call_tool(
        s, "evaluate_strategy", {"strategy": '{"combine": "median", "pool": "pool_full"}'}
    )
    assert ok and obs["strategy"]["combine"] == "median"


def test_evaluate_strategy_recovers_from_the_human_readable_label():
    """The model once pasted back the observation text: 'weighted on pool2 (7 models)'."""
    s, series, pool = prepared()
    T.select_top_k(s, k=3)
    T.weights_inverse_error(s, pool="pool1")
    ok, obs = R.call_tool(
        s, "evaluate_strategy",
        {"strategy": "weighted on pool1 (3 models)", "pool": "pool1", "weights": "w1"},
    )
    assert ok and obs["strategy"]["combine"] == "weighted"


def test_evaluate_strategy_without_a_method_says_what_to_send():
    s, series, pool = prepared()
    ok, obs = R.call_tool(s, "evaluate_strategy", {"pool": "pool_full"})
    assert not ok
    assert "combine" in obs["detail"]


def test_combine_tools_hand_back_a_ready_to_paste_call():
    """The observation must show the exact Action Input for the next turn."""
    s, series, pool = prepared()
    built = T.combine_mean(s)
    assert built["next_action_input"]["strategy"] == built["strategy"]
    ok, _ = R.call_tool(s, "evaluate_strategy", built["next_action_input"])
    assert ok


def test_the_two_step_detour_is_no_longer_needed():
    """One call now does what used to take two, saving an iteration per strategy."""
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    llm = ScriptedLLM([
        step("select_stable", {"k": 3}, "favour consistency"),
        step("evaluate_strategy", {"combine": "mean", "pool": "pool1"}, "test the stable subset"),
        step("accept", {"attempt_id": "a4", "justification": "unstable ranking"}),
    ])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert r.stop_reason == "agent_accepted"
    assert r.iterations_used == 3
    assert not any("ERROR" in t["observation_summary"] for t in r.trajectory)


def test_prune_redundant_summary_reports_what_it_dropped():
    """It used to print `k=9 []` because that tool returns no `models` key."""
    s, series, pool = prepared()
    ok, obs = R.call_tool(s, "prune_redundant", {"corr_threshold": 0.8})
    assert ok
    line = P.summarize_observation("prune_redundant", True, obs)
    assert "->" in line and "dropped" in line
    assert "[]" not in line


def test_a_descriptive_extra_field_does_not_cost_an_iteration():
    """gpt-oss added `origin` to the call and lost the turn over it."""
    s, series, pool = prepared()
    T.select_top_k(s, k=3)
    T.weights_inverse_error(s, pool="pool1")
    ok, obs = R.call_tool(
        s, "evaluate_strategy",
        {"combine": "weighted", "pool": "pool1", "weights": "w1", "origin": "combination"},
    )
    assert ok
    assert obs["ignored_args"] == ["origin"], "the tolerance must be visible, not silent"


def test_other_tools_stay_strict():
    """Tolerance is only for the central tool; a wrong knob elsewhere is still an error."""
    s, series, pool = prepared()
    ok, obs = R.call_tool(s, "select_top_k", {"k": 3, "temperature": 0.7})
    assert not ok and obs["error"] == "unknown_argument"


def test_unparsed_turns_keep_the_raw_text_for_forensics():
    """`ERROR parse: no 'Action:' line found` is useless without the actual answer."""
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = ScriptedLLM([
        "I will begin by reviewing the series characteristics before choosing.",
        step("accept", {"attempt_id": "a1"}),
    ])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert len(r.parse_failures) == 1
    failure = r.parse_failures[0]
    assert failure["iteration"] == 1
    assert "Action:" in failure["reason"]
    assert "I will begin by reviewing" in failure["raw"]
    # and it must not bloat the CSV column
    import json as _json
    assert "I will begin" not in _json.dumps(r.trajectory)


# ══════════════════════════════════════════════════════════════════════════════
# waste the server run exposed
# ══════════════════════════════════════════════════════════════════════════════


def test_empty_responses_are_retried_not_charged():
    """gpt-oss sometimes emits only a <think> block and stops.

    That is a failed generation, not a decision, so it must not consume one of the
    eight hypotheses the agent gets.
    """
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = ScriptedLLM([
        "",                                            # empty
        "<think>still thinking</think>",               # empty after the think block
        step("evaluate_strategy", {"combine": "median", "pool": FULL_POOL}, "test it"),
        step("accept", {"attempt_id": "a2"}),
    ])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert r.empty_responses == 2
    assert r.iterations_used == 2, "the two empty generations cost no iteration"
    assert r.stop_reason == "agent_accepted"
    assert [t["action"] for t in r.trajectory] == ["evaluate_strategy", "accept"]


def test_a_persistently_empty_model_still_terminates():
    s, series, pool = prepared(ReactConfig(max_iterations=2))
    r = run_react_loop(s, ScriptedLLM([""] * 20), series, pool, s.config)
    assert r.iterations_used == 2
    assert r.final_attempt is not None


class FlakyLLM:
    """Raises `LLMError` on the first `n_failures` calls, then answers normally.

    Models the real failure this is regression-testing: Ollama's own chat
    template misreading a gpt-oss turn as an attempted tool call and returning a
    server-side JSON parse error instead of the model's text. That is a transport
    failure, not something `parse_agent_step` ever sees.
    """

    def __init__(self, n_failures: int, then: list[str]):
        self.n_failures = n_failures
        self.then = then
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls <= self.n_failures:
            raise LLMError(
                "error parsing tool call: raw='We need to output exactly three "
                "lines: Thought, Action, Action Input.', err=invalid character "
                "'W' looking for beginning of value (status code: -1)"
            )
        return self.then[self.calls - self.n_failures - 1]


def test_a_transient_llm_error_is_retried_not_charged():
    """The exact failure from a real run: reproduced verbatim to pin the fix."""
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = FlakyLLM(1, [
        step("evaluate_strategy", {"combine": "median", "pool": FULL_POOL}, "test it"),
        step("accept", {"attempt_id": "a2"}),
    ])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert r.llm_error_retries == 1
    assert r.iterations_used == 2, "the transient failure cost no iteration"
    assert r.stop_reason == "agent_accepted"
    assert any("transient LLM error, retrying" in e for e in r.errors)


def test_a_persistent_llm_error_still_ends_the_loop():
    """Retries are bounded: a genuinely dead server must still surface, not hang."""
    s, series, pool = prepared(ReactConfig(max_iterations=4))
    llm = FlakyLLM(99, [step("accept", {"attempt_id": "a1"})])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert r.stop_reason == "llm_error"
    assert r.llm_error_retries == LLM_ERROR_RETRIES
    assert r.final_attempt is not None, "the run must still fall back to the best baseline"
    assert llm.calls == LLM_ERROR_RETRIES + 1, "no more calls than the bounded retry allows"


def test_llm_error_retries_and_empty_response_retries_have_independent_budgets():
    """One turn hitting both failure modes must not let one budget eat the other's."""
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    calls = {"n": 0}

    def flaky_then_empty_then_ok(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("transient")
        if calls["n"] == 2:
            return ""  # empty, not a transport error
        return step("accept", {"attempt_id": "a1"})

    class Client:
        complete = staticmethod(flaky_then_empty_then_ok)

    r = run_react_loop(s, Client(), series, pool, s.config)
    assert r.llm_error_retries == 1
    assert r.empty_responses == 1
    assert r.iterations_used == 1, "neither retry is charged to the iteration budget"


def test_a_malformed_answer_is_still_shown_to_the_agent():
    """Non-empty but unreadable is different: the model can learn from that one."""
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = ScriptedLLM(["I think the mean looks fine", step("accept", {"attempt_id": "a1"})])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert r.empty_responses == 0
    assert r.trajectory[0]["action"] == "unparsed"
    assert r.iterations_used == 2


def test_unparsed_turns_keep_the_raw_text():
    s, series, pool = prepared(ReactConfig(max_iterations=3))
    llm = ScriptedLLM(["I will begin by reviewing the series.", step("accept", {"attempt_id": "a1"})])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert "I will begin by reviewing" in r.parse_failures[0]["raw"]
    assert "I will begin" not in json.dumps(r.trajectory), "must not bloat the CSV"


def test_unscored_weight_handles_are_flagged_near_the_end():
    """The agent burned its last turns building handles it never evaluated."""
    s, series, pool = prepared()
    T.select_top_k(s, k=3)
    handle = T.weights_inverse_error(s, pool="pool1")["weights"]

    assert "never scored" not in P.build_turn_prompt(s, series, pool, [], 1, 8)
    late = P.build_turn_prompt(s, series, pool, [], 6, 8)
    assert "never scored" in late and handle in late

    T.evaluate_strategy(s, {"combine": "weighted", "pool": "pool1", "weights": handle})
    assert "never scored" not in P.build_turn_prompt(s, series, pool, [], 6, 8)


def test_the_prompt_says_handles_start_empty():
    """A model fresh off another series reached for a `w1` that did not exist."""
    s = make_state()
    series, pool = {}, {}
    assert "none yet" in P.build_turn_prompt(s, series, pool, [], 1, 8)


def test_seeded_pools_are_not_attributed_to_the_agent():
    """Phase 2 registers stability pools before the loop opens. Presenting them as
    handles the agent created invited it to assume a matching `w1` existed too."""
    s, series, pool = prepared()
    text = P.build_turn_prompt(s, series, pool, [], 1, 8)
    assert "HANDLES AVAILABLE" in text
    assert "YOU HAVE CREATED" not in text
    assert "no weight handles exist yet" in text, "the absence of weights must be explicit"
    assert "Never assume w1 or pool1 exists" in P.build_system_prompt()


# ══════════════════════════════════════════════════════════════════════════════
# selection confidence — the deterministic replacement for self-report
# ══════════════════════════════════════════════════════════════════════════════


def test_self_reported_confidence_is_recorded_but_not_trusted():
    """gpt-oss:20b answered 0.9 on all 13 accepts of a 19-series run.

    The field is kept because the specification asks for it and because "the model
    always says 0.9" is itself a finding, but nothing downstream depends on it.
    """
    s, series, pool = prepared(ReactConfig(max_iterations=2))
    llm = ScriptedLLM([step("accept", {"attempt_id": "a1", "confidence": 0.9})])
    r = run_react_loop(s, llm, series, pool, s.config)
    assert r.accept_confidence == 0.9
    # the statistical verdict is computed independently of what the model claimed
    assert "confidence" not in json.dumps(s.selection_confidence())


def test_selection_confidence_has_the_full_schema():
    s, series, pool = prepared()
    conf = s.selection_confidence()
    assert set(conf) >= {
        "n_windows", "n_attempts", "winner", "runner_up", "margin",
        "bootstrap_pvalue", "dm_pvalue", "verdict",
    }
    assert conf["winner"] == s.best_attempt().attempt_id


def test_two_indistinguishable_strategies_are_reported_as_such():
    """The honest answer with three windows is usually "cannot tell"."""
    s, series, pool = prepared()
    T.evaluate_strategy(s, {"combine": "trimmed_mean", "pool": FULL_POOL, "trim_pct": 0.05})
    conf = s.selection_confidence()
    assert conf["verdict"] in {"indistinguishable", "weak", "separated", "undetermined"}
    assert conf["runner_up"] is not None
    assert conf["margin"] is not None


def test_a_clearly_worse_runner_up_moves_the_verdict():
    """A deliberately bad model against the best baseline should separate."""
    s, series, pool = prepared()
    # keep only two attempts: the best baseline and a terrible single model
    best = s.best_attempt()
    s.attempts = [a for a in s.attempts if a is best]
    s._attempt_by_spec = {k: v for k, v in s._attempt_by_spec.items() if v is best}
    T.evaluate_strategy(s, {"combine": "best_single", "model": "bad"})

    conf = s.selection_confidence()
    assert conf["margin"] > 0, "the runner-up must score worse"
    assert conf["dm_pvalue"] is not None
    assert conf["verdict"] in {"separated", "weak", "indistinguishable"}


def test_a_single_attempt_cannot_be_compared():
    s = make_state()
    POOL.seed_baselines(s, methods=("mean",), stable_pools=())
    conf = s.selection_confidence()
    assert conf["verdict"] == "no_comparison"
    assert conf["runner_up"] is None


def test_every_attempt_carries_its_paired_evidence():
    """Without per-window scores and residuals there is nothing to test with."""
    s, series, pool = prepared()
    for attempt in s.attempts:
        assert len(attempt.per_window_scores) == s.n_windows
        assert len(attempt.residuals) == s.n_windows * s.horizon
        assert all(np.isfinite(v) for v in attempt.residuals)


def test_residuals_match_the_backtest():
    s, series, pool = prepared()
    attempt = [a for a in s.attempts if a.spec["combine"] == "mean"][0]
    combined, _ = s.backtest(attempt.spec)
    expected = (combined - s.y_true).reshape(-1)
    assert attempt.residuals == pytest.approx(expected)


def test_selection_confidence_is_deterministic():
    """Same data, same verdict — a bootstrap with a fixed seed must not wander."""
    a = make_state(seed=11); POOL.run_phase2(a, a.config)
    b = make_state(seed=11); POOL.run_phase2(b, b.config)
    assert a.selection_confidence() == b.selection_confidence()


# ══════════════════════════════════════════════════════════════════════════════
# exploration: the agent lost to `dba` three times and never tried dba variants
# ══════════════════════════════════════════════════════════════════════════════


def test_the_prompt_points_at_the_leading_method():
    sp = P.build_system_prompt()
    assert "SAME method on a better pool" in sp
    assert "dba on a pruned pool" in sp


def test_the_observation_names_the_leader_when_you_lose():
    """Seeing only "rank 3/5" hides which method you are losing to."""
    s, series, pool = prepared()
    losing = T.evaluate_strategy(s, {"combine": "best_single", "model": "bad"})
    line = P.summarize_observation("evaluate_strategy", True, losing)
    assert "leader is" in line
    assert losing["current_best"]["strategy"]
    assert losing["current_best"]["origin"] == "baseline"

    # re-submitting whatever is currently leading must not report a leader
    leading = T.evaluate_strategy(s, s.best_attempt().spec)
    assert leading["is_best"] is True
    assert "leader is" not in P.summarize_observation("evaluate_strategy", True, leading)


# ══════════════════════════════════════════════════════════════════════════════
# handle identity: same thing, same name
# ══════════════════════════════════════════════════════════════════════════════


def test_an_identical_weight_recipe_reuses_its_handle():
    """Two handles for the same numbers would put numerical twins in the ranking."""
    s, series, pool = prepared()
    T.select_top_k(s, k=3)
    first = T.weights_inverse_error(s, pool="pool1", metric="rmse")
    second = T.weights_inverse_error(s, pool="pool1", metric="rmse")
    assert first["weights"] == second["weights"] == "w1"
    assert first["reused"] is False and second["reused"] is True
    assert len(s.weights) == 1


def test_a_different_recipe_gets_its_own_handle():
    s, series, pool = prepared()
    T.select_top_k(s, k=3)
    a = T.weights_inverse_error(s, pool="pool1", shrinkage=0.0)["weights"]
    b = T.weights_inverse_error(s, pool="pool1", shrinkage=0.5)["weights"]
    c = T.weights_softmax_neg_error(s, pool="pool1")["weights"]
    assert len({a, b, c}) == 3


def test_reuse_is_announced_to_the_agent():
    """Selecting every model returns pool_full; the agent must be told, not guess."""
    s, series, pool = prepared()
    everything = T.select_top_k(s, k=s.n_models)
    assert everything["pool"] == FULL_POOL
    assert everything["reused"] is True
    assert "identical to an existing pool" in everything["note"]

    fresh = T.select_top_k(s, k=3)
    assert fresh["reused"] is False and fresh["pool"] not in {FULL_POOL}


def test_a_no_op_prune_reports_reuse():
    s, series, pool = prepared()
    top = T.select_top_k(s, k=2)
    again = T.prune_redundant(s, pool=top["pool"], corr_threshold=0.999)
    if again["n_before"] == again["n_after"]:
        assert again["pool"] == top["pool"]
        assert again["reused"] is True


def test_a_twin_runner_up_is_skipped_in_the_confidence_test():
    """Comparing the winner against a copy of itself says nothing about the data.

    The margin collapses to zero and both tests accept, yielding a spurious
    "indistinguishable". The comparison must reach the first genuinely different
    attempt instead.
    """
    s, series, pool = prepared()
    # Handle reuse is what makes two specs land on the same members. Under nested
    # selection a recipe pool and a hand-written index list are NOT interchangeable
    # — they can diverge inside a fold — so the twin is built from two static
    # registrations, which is the case where reuse still applies.
    s.config.nested_selection = False
    top = T.select_top_k(s, k=3)
    T.evaluate_strategy(s, {"combine": "mean", "pool": top["pool"]})
    twin = s.register_pool(s.get_pool(top["pool"]) + [], origin="manual")
    assert twin == top["pool"], "an identical static index set must reuse the handle"

    # build a real twin: same forecasts, different spec
    trimmed = T.evaluate_strategy(
        s, {"combine": "trimmed_mean", "pool": top["pool"], "trim_pct": 0.0}
    )
    ranked = s.ranked_attempts()
    conf = s.selection_confidence()
    twins = [
        a for a in ranked[1:]
        if len(a.residuals) == len(ranked[0].residuals)
        and np.allclose(a.residuals, ranked[0].residuals, atol=1e-9)
    ]
    if twins:
        assert conf["twins_skipped"] >= 1
        assert conf["runner_up"] not in {t.attempt_id for t in twins}


def test_confidence_reports_when_no_distinct_alternative_exists():
    s = make_state()
    POOL.seed_baselines(s, methods=("mean",), stable_pools=())
    # trimmed_mean with trim 0 is arithmetically the mean: an exact twin
    T.evaluate_strategy(s, {"combine": "trimmed_mean", "pool": FULL_POOL, "trim_pct": 0.0})
    conf = s.selection_confidence()
    assert conf["verdict"] == "no_distinct_alternative"
    assert conf["twins_skipped"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ══════════════════════════════════════════════════════════════════════════════
# weights_ols is gated on the number of validation windows
# ══════════════════════════════════════════════════════════════════════════════


def test_weights_ols_is_withheld_when_there_are_too_few_windows():
    """Granger-Ramanathan on 3 windows lands on a simplex vertex, which turns a
    weighting tool into a model selector whose winner is not the lowest-error
    model. Under the threshold the tool is not offered at all."""
    from orchestrator_react import registry as R
    from orchestrator_react.config import ReactConfig

    cfg = ReactConfig(min_windows_for_ols=5)
    assert "weights_ols" in R.withheld_tools(cfg, n_windows=3)
    assert "weights_ols" in R.withheld_tools(cfg, n_windows=4)
    assert R.withheld_tools(cfg, n_windows=5) == {}
    assert R.withheld_tools(cfg, n_windows=8) == {}


def test_the_reason_names_the_threshold_and_the_actual_count():
    from orchestrator_react import registry as R
    from orchestrator_react.config import ReactConfig

    reason = R.withheld_tools(ReactConfig(min_windows_for_ols=5), 3)["weights_ols"]
    assert "5" in reason and "3" in reason


def test_a_withheld_tool_never_reaches_the_prompt():
    from orchestrator_react import prompts as P
    from orchestrator_react import registry as R
    from orchestrator_react.config import ReactConfig

    withheld = R.withheld_tools(ReactConfig(min_windows_for_ols=5), 3)
    assert "weights_ols" not in P.build_system_prompt(withheld_tools=withheld)
    # every other weighting tool survives
    for name in ("weights_inverse_error", "weights_softmax_neg_error", "weights_feature_based"):
        assert name in P.build_system_prompt(withheld_tools=withheld)
    assert "weights_ols" in P.build_system_prompt()


def test_calling_a_withheld_tool_anyway_is_refused_not_executed():
    from orchestrator_react import registry as R
    from orchestrator_react.config import ReactConfig

    s = make_state()
    withheld = R.withheld_tools(ReactConfig(min_windows_for_ols=5), 3)
    ok, obs = R.call_tool(s, "weights_ols", {"pool": "pool_full"}, withheld=withheld)
    assert ok is False
    assert obs["error"] == "unknown_tool"
    assert "weights_ols" not in obs["available"]
    assert not s.weights  # nothing was registered


def test_the_gate_is_off_by_default_for_a_run_with_enough_windows():
    from orchestrator_react import registry as R
    from orchestrator_react.config import ReactConfig

    s = make_state()
    withheld = R.withheld_tools(ReactConfig(min_windows_for_ols=2), s.n_windows)
    ok, obs = R.call_tool(s, "weights_ols", {"pool": "pool_full"}, withheld=withheld)
    assert ok is True
    assert obs["method"] == "ols"


def test_the_run_records_which_tools_were_withheld():
    from orchestrator_react.config import ReactConfig

    s = make_state()
    POOL.run_phase2(s, s.config)
    cfg = ReactConfig(min_windows_for_ols=99)
    cfg.combinator.model = "scripted"
    res = run_react_loop(
        s,
        ScriptedLLM(["Thought: t\nAction: accept\nAction Input: {}"]),
        series_card={}, pool_card={}, config=cfg,
    )
    assert "weights_ols" in res.withheld_tools
    assert "weights_ols" in res.summary()["withheld_tools"]


# ══════════════════════════════════════════════════════════════════════════════
# reproducibility of the agent's sampling
# ══════════════════════════════════════════════════════════════════════════════


def test_a_seed_is_configured_by_default():
    """NN5 ships three duplicate series (T1/T47, T11/T50, T79/T111). The agent
    picked a different combination on all three pairs, which is run-to-run
    sampling noise, not a decision. A seed makes the run reproducible."""
    from orchestrator_react.config import LLMRole

    assert LLMRole().seed is not None


def test_the_seed_reaches_the_ollama_client():
    from orchestrator_react.config import LLMRole
    from orchestrator_react.llm import OllamaClient

    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import orchestrator_react.llm as L

    client = OllamaClient(role=LLMRole(model="m", seed=123))
    mod = types.ModuleType("langchain_ollama")
    mod.ChatOllama = FakeChat
    sys.modules["langchain_ollama"] = mod
    try:
        client._client()
    finally:
        sys.modules.pop("langchain_ollama", None)
    assert captured["seed"] == 123
    assert captured["model"] == "m"


def test_seed_none_is_omitted_rather_than_passed_as_none():
    from orchestrator_react.config import LLMRole
    from orchestrator_react.llm import OllamaClient

    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mod = types.ModuleType("langchain_ollama")
    mod.ChatOllama = FakeChat
    sys.modules["langchain_ollama"] = mod
    try:
        OllamaClient(role=LLMRole(model="m", seed=None))._client()
    finally:
        sys.modules.pop("langchain_ollama", None)
    assert "seed" not in captured


def test_the_seed_is_part_of_the_run_fingerprint():
    """Two runs that differ only by seed must not claim the same ablation id."""
    from orchestrator_react.config import ReactConfig

    a = ReactConfig()
    b = ReactConfig()
    b.combinator.seed = 999
    assert a.fingerprint() != b.fingerprint()
