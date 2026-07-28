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
from orchestrator_react.react_loop import run_react_loop
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
    assert "ATTEMPT HISTORY (3)" in prompt  # the three seeded baselines
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
    assert "HANDLES YOU HAVE CREATED" in prompt
    assert "pool1" in prompt and "w1" in prompt


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
    T.select_top_k(s, k=3)  # creates pool1, which beats the full-pool mean
    llm = ScriptedLLM([
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "bad"}}),
        step("evaluate_strategy", {"strategy": {"combine": "mean", "pool": "pool1"}}),
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "mediocre"}}),
        step("accept", {"attempt_id": "a5"}),
    ])
    r = run_react_loop(s, llm, series, pool)
    assert r.early_stopped is False
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
    llm = ScriptedLLM([
        step("evaluate_strategy", {"strategy": {"combine": "best_single", "model": "bad"}}),
        step("accept", {"attempt_id": "a4", "confidence": 0.9,
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
    trimmed = [a for a in s.attempts if a.spec["combine"] == "trimmed_mean"][0]
    assert "trimming should drop" in trimmed.rationale
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
