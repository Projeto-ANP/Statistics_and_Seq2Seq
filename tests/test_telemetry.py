"""End-to-end telemetry: what the agent did must be reconstructible from the artifacts.

Runs the ReAct loop with a scripted LLM (no server), then checks that the
per-series `agent_trace` carries the turns (full observation, leader before and
after, timestamps), the attempt registry (spec, rationale, validation metrics,
per-window metrics), the decision and stop blocks, and the derived timings; and
that the compact `agent_*` CSV columns are filled. Also pins the rule that
matters for the paper: the trace is validation-only, so the test actuals must
not appear anywhere in it.

Run:  python -m pytest tests/test_telemetry.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import csv_writer as W
from orchestrator_react import pipeline as PL
from orchestrator_react import pool as POOL
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import ScriptedLLM
from orchestrator_react.react_loop import run_react_loop

from test_orchestrator_react import HORIZON, make_state  # noqa: E402

# Distinctive test values: if any of these leaked into the payload, the assertion
# below would catch it without depending on formatting.
TEST_ACTUALS = [round(1234.5 + i * 7.25, 4) for i in range(HORIZON)]

EVALUATE_RESPONSE = (
    "Thought: test the trimmed mean on the full pool.\n"
    "Action: evaluate_strategy\n"
    'Action Input: {"combine": "trimmed_mean", "pool": "pool_full", '
    '"trim_pct": 0.2, "rationale": "robust to the bad model"}\n'
)
# Accepting without naming an attempt means "the best in the history", which is
# also what the loop applies; that keeps the two ids in the decision block equal.
ACCEPT_RESPONSE = (
    "Thought: the best attempt is solid for this series.\n"
    "Action: accept\n"
    'Action Input: {"confidence": 0.7, '
    '"justification": "the best validation score already looks solid"}\n'
)


def _run_scripted():
    cfg = ReactConfig()
    state = make_state(config=cfg)
    POOL.seed_baselines(state)
    client = ScriptedLLM(responses=[EVALUATE_RESPONSE, ACCEPT_RESPONSE])
    react = run_react_loop(
        state=state, client=client, series_card={}, pool_card={}, config=cfg
    )
    outcome = PL.SeriesOutcome(
        dataset="TOY",
        dataset_index=0,
        horizon=state.horizon,
        state=state,
        react=react,
        config=cfg,
        test_values=list(TEST_ACTUALS),
        forecast=[0.0] * state.horizon,
    )
    return state, react, outcome


def test_trace_has_the_full_record():
    _state, react, outcome = _run_scripted()
    payload = W.artifacts_payload(outcome)
    trace = payload["agent_trace"]
    for key in ("meta", "turns", "attempts", "decision", "stop", "derived"):
        assert key in trace

    # Turns: the evaluate call and the accept, with full observation and leader.
    assert len(trace["turns"]) == 2
    first = trace["turns"][0]
    assert first["action"] == "evaluate_strategy"
    assert "observation" in first and isinstance(first["observation"], dict)
    assert first["turn_started_at"] and first["turn_finished_at"]
    assert first["best_before"]["id"] and first["best_after"]["id"]
    assert "tool_exec_s" in first
    second = trace["turns"][1]
    assert second["action"] == "accept"
    assert second["accepted_attempt_id"]

    # Attempt registry: validation metrics and per-window evidence for each.
    assert trace["attempts"]
    for attempt in trace["attempts"]:
        assert attempt["id"] and attempt["spec"] and attempt["origin"]
        assert len(attempt["per_window"]) == _state.n_windows
        assert "RMSE" in attempt["aggregate"]

    # Decision and stop.
    assert trace["decision"]["final_attempt_id"]
    assert trace["decision"]["agent_accepted_id"] == trace["decision"]["final_attempt_id"]
    assert trace["stop"]["stop_reason"] == "agent_accepted"
    assert trace["derived"]["turns"] == 2
    assert trace["derived"]["llm_s"] >= 0.0
    assert trace["derived"]["tool_s"] >= 0.0
    assert "selection" in trace["derived"]


def test_trace_is_validation_only():
    _state, _react, outcome = _run_scripted()
    payload = W.artifacts_payload(outcome)
    text = json.dumps(payload, default=str)
    assert "test_values" not in text
    for value in TEST_ACTUALS:
        assert str(value) not in text


def test_agent_csv_columns_are_filled():
    _state, react, outcome = _run_scripted()
    row = W.build_row(outcome, regressor="react_toy")
    assert set(row) == set(W.COLS_SERIE)
    assert row["agent_turns"] == 2
    assert row["agent_started_at"] and row["agent_finished_at"]
    assert row["agent_llm_s"] is not None
    assert row["agent_scoring_calls"] == 1
    assert row["agent_best_changes"] is not None


def test_dataset_summary_aggregates_the_traces():
    _state, _react, outcome = _run_scripted()
    summary = W.dataset_summary([outcome])
    assert summary["n_series"] == 1 and summary["n_ok"] == 1
    assert summary["stop_reasons"] == {"agent_accepted": 1}
    assert summary["final_origin"] == {outcome.react.final_attempt.origin: 1}
    assert "wall_clock_s" in summary["mean_derived"]
    assert "evaluate_strategy" in summary["tool_calls_ok"]
