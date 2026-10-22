"""Tests for the v11 blind final selection.

Run:  python -m pytest tests/test_final_choice.py -q
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import final_choice as FC
from orchestrator_react import tools as T
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import LLMError, ScriptedLLM
from orchestrator_react.state import FULL_POOL

from test_orchestrator_react import make_state  # noqa: E402
from test_ingest_and_pool import MODELS, fake_repo  # noqa: E402,F401


class FakeAttempt:
    def __init__(self, attempt_id: str, spec: dict, score: float, forecast):
        self.attempt_id = attempt_id
        self.spec = spec
        self.score = float(score)
        self.forecast = np.asarray(forecast, dtype=float)


def _state(config: ReactConfig | None = None):
    state = make_state(config=config or ReactConfig(max_iterations=2))
    for spec in (
        {"combine": "mean", "pool": FULL_POOL},
        {"combine": "median", "pool": FULL_POOL},
        {"combine": "trimmed_mean", "pool": FULL_POOL, "trim_pct": 0.2},
        {"combine": "best_single", "model": "good_a"},
    ):
        state.evaluate(spec)
    return state


def _forecast_for(state):
    return lambda attempt: state.apply_to_test(attempt.spec)[0]


def _guard_state(active: bool = True):
    """Two known candidates: leader far from the seasonal naive, challenger equal to it."""
    state = make_state(config=ReactConfig(max_iterations=2))
    period = 12
    horizon = state.horizon
    last = float(state.y_true[-1, -1])
    seasonal = np.resize(np.asarray(state.train_series[-period:], dtype=float), horizon)
    leader = FakeAttempt(
        "ID_LEADER_000", {"combine": "mean", "pool": FULL_POOL}, 0.10,
        np.full(horizon, last + 10.0),
    )
    challenger = FakeAttempt(
        "ID_SEASONAL_111", {"combine": "median", "pool": FULL_POOL}, 0.20, seasonal
    )
    state.attempts = [leader, challenger]
    card = dict(T.series_profile(state))
    card["seasonal_period"] = period
    card["seasonal_strength"] = 0.9 if active else 0.1
    card["n_points"] = 1000
    return state, card, leader, challenger


# ══════════════════════════════════════════════════════════════════════════════
# shortlist
# ══════════════════════════════════════════════════════════════════════════════


def test_shortlist_picks_best_per_family():
    attempts = [
        FakeAttempt("a1", {"combine": "mean"}, 0.50, [1, 2, 3]),
        FakeAttempt("a2", {"combine": "median"}, 0.40, [1, 2, 4]),
        FakeAttempt("a3", {"combine": "median"}, 0.30, [1, 2, 5]),
        FakeAttempt("a4", {"combine": "weighted"}, 0.20, [1, 2, 6]),
        FakeAttempt("a5", {"combine": "dba"}, 0.10, [1, 2, 7]),
    ]
    shortlist = FC.build_family_shortlist(
        attempts, lambda a: np.asarray(a.forecast, dtype=float), max_candidates=5
    )
    assert [e.attempt.attempt_id for e in shortlist] == ["a5", "a4", "a3", "a1"]
    assert len(shortlist) == 4


def test_shortlist_dedups_nearly_identical_forecasts():
    attempts = [
        FakeAttempt("a1", {"combine": "mean"}, 0.10, [1, 2, 3, 4]),
        FakeAttempt("a2", {"combine": "median"}, 0.20, [1, 2, 3, 4]),
        FakeAttempt("a3", {"combine": "dba"}, 0.30, [4, 3, 2, 1]),
    ]
    shortlist = FC.build_family_shortlist(
        attempts, lambda a: np.asarray(a.forecast, dtype=float), max_candidates=5
    )
    assert [e.attempt.attempt_id for e in shortlist] == ["a1", "a3"]


def test_shortlist_caps_candidates():
    attempts = [
        FakeAttempt(f"a{i}", {"combine": f"f{i}"}, i / 100.0, [i, i * i + 1, (i + 1) ** 2])
        for i in range(8)
    ]
    shortlist = FC.build_family_shortlist(
        attempts, lambda a: np.asarray(a.forecast, dtype=float), max_candidates=3
    )
    assert len(shortlist) == 3


# ══════════════════════════════════════════════════════════════════════════════
# blind view and prompt leakage
# ══════════════════════════════════════════════════════════════════════════════


def test_blind_candidates_have_no_attempt_identity():
    state = _state()
    card = T.series_profile(state)
    shortlist = FC.build_family_shortlist(state.attempts, _forecast_for(state))
    ctx = FC.build_series_context(state, card)
    entries, by_label = FC.blind_shortlist(shortlist, ctx, "FAKE", 0, seed=0)

    assert len(entries) == len(shortlist)
    assert set(by_label) == set(entry.label for entry in entries)
    assert len(set(by_label)) == len(by_label)
    for entry in entries:
        blind = entry.blind
        assert blind is not None
        assert not hasattr(blind, "attempt_id")
        assert not hasattr(blind, "spec")
        assert not hasattr(blind, "score")
        assert blind.label == entry.label
        assert 0 < len(blind.forecast_scaled) <= FC.MAX_PROMPT_POINTS


def test_prompt_has_no_model_names_ids_scores_or_test_actuals():
    state = _state()
    card = T.series_profile(state)
    shortlist = FC.build_family_shortlist(state.attempts, _forecast_for(state))

    # Sentinel values make the leakage checks specific: none of these strings
    # exists in legitimate prompt content (forecasts/features are plain numbers).
    state.attempts[0].attempt_id = "SENTINEL_ATTEMPT_ID"
    state.attempts[0].score = 98765.4321

    ctx = FC.build_series_context(state, card)
    entries, _ = FC.blind_shortlist(shortlist, ctx, "FAKE", 0, seed=0)
    system, user = FC.build_final_prompt(card, {}, ctx, entries, allow_keep=True)

    for name in state.model_names:
        assert name not in system
        assert name not in user
    assert "SENTINEL_ATTEMPT_ID" not in system
    assert "SENTINEL_ATTEMPT_ID" not in user
    assert "98765" not in user
    assert "validation_score" not in user.lower()
    assert "validation_rank" not in user.lower()
    assert "attempt_id" not in user.lower()
    assert "origin" not in user.lower()
    assert "trend_champion" not in user
    assert "seasonality_champion" not in user
    assert not any("test" in key.lower() for key in ctx)
    assert "test actual is NOT shown" in user


def test_prompt_builder_does_not_receive_test_values():
    """The test actuals are not a parameter anywhere in the final-choice path."""
    assert "test_values" not in inspect.signature(FC.build_final_prompt).parameters
    assert "test_values" not in inspect.signature(FC.build_series_context).parameters
    assert "test_values" not in inspect.signature(FC.choose_final).parameters


def test_prompt_does_not_change_when_hidden_attempt_fields_change():
    """Only ``entry.blind`` is read: mutating the real attempt changes nothing."""
    state = _state()
    card = T.series_profile(state)
    shortlist = FC.build_family_shortlist(state.attempts, _forecast_for(state))
    ctx = FC.build_series_context(state, card)
    entries, _ = FC.blind_shortlist(shortlist, ctx, "FAKE", 0, seed=0)
    _, before = FC.build_final_prompt(card, {}, ctx, entries, allow_keep=True)

    for entry in entries:
        entry.attempt.attempt_id = "MUTATED_" + str(entry.label)
        entry.attempt.spec = {"combine": "MUTATED", "pool": "MUTATED"}
        entry.attempt.score = 123456.789
    _, after = FC.build_final_prompt(card, {}, ctx, entries, allow_keep=True)

    assert before == after


# ══════════════════════════════════════════════════════════════════════════════
# parsing and decision
# ══════════════════════════════════════════════════════════════════════════════


def test_parse_choice_variants():
    parsed = FC.parse_choice("CHOICE: A\nCONFIDENCE: high\nREASON: because", "ABCDE")
    assert parsed.choice == "A" and parsed.confidence == "high"
    assert parsed.reason == "because" and not parsed.parse_error

    parsed = FC.parse_choice("choice: keep\nconfidence: low", "ABCDE")
    assert parsed.choice == "KEEP" and parsed.confidence == "low"

    parsed = FC.parse_choice("CHOICE: Candidate B\nCONFIDENCE: medium", "ABCDE")
    assert parsed.choice == "B"

    parsed = FC.parse_choice("<think>deliberation</think>\nCHOICE: C\nCONFIDENCE: low", "ABCDE")
    assert parsed.choice == "C" and parsed.confidence == "low"

    parsed = FC.parse_choice("CHOICE: Z", "ABCDE")
    assert parsed.parse_error

    parsed = FC.parse_choice("no structured answer", "ABCDE")
    assert parsed.parse_error


def _entry(attempt, label):
    return FC.ShortlistEntry(
        attempt=attempt,
        forecast=np.asarray(attempt.forecast, dtype=float),
        label=label,
        blind=FC.BlindCandidate(label=label, forecast_scaled=[], features={}),
    )


def test_decide_keep_override_low_confidence_and_degenerate():
    leader = FakeAttempt("a1", {"combine": "mean"}, 0.1, [10, 10, 10])
    other = FakeAttempt("a2", {"combine": "median"}, 0.2, [10, 11, 12])
    entries = [_entry(leader, "A"), _entry(other, "B")]
    ctx = {"last_actual": 10.0, "scale": 1.0}
    config = ReactConfig()
    forecast_for = lambda a: np.asarray(a.forecast, dtype=float)

    chosen, decision, fallback = FC.decide(
        FC.ParsedChoice("KEEP", "high", "", "raw"),
        entries, leader, ctx, config, forecast_for,
    )
    assert chosen is leader and decision == "keep" and fallback is None

    chosen, decision, fallback = FC.decide(
        FC.ParsedChoice("B", "high", "", "raw"),
        entries, leader, ctx, config, forecast_for,
    )
    assert chosen is other and decision == "override" and fallback is None

    chosen, decision, fallback = FC.decide(
        FC.ParsedChoice("B", "low", "", "raw"),
        entries, leader, ctx, config, forecast_for,
    )
    assert chosen is leader and decision == "fallback" and fallback == "low_confidence"

    huge = FakeAttempt("a3", {"combine": "dba"}, 0.3, [1e9, 1e9, 1e9])
    entries_huge = [_entry(leader, "A"), _entry(huge, "B")]
    chosen, decision, fallback = FC.decide(
        FC.ParsedChoice("B", "high", "", "raw"),
        entries_huge, leader, ctx, config, forecast_for,
    )
    assert chosen is leader and decision == "fallback" and fallback == "degenerate_forecast"

    chosen, decision, fallback = FC.decide(
        FC.ParsedChoice("Z", "high", "", "raw"),
        entries, leader, ctx, config, forecast_for,
    )
    assert chosen is leader and decision == "fallback" and fallback == "unknown_label"


# ══════════════════════════════════════════════════════════════════════════════
# choose_final
# ══════════════════════════════════════════════════════════════════════════════


def test_choose_final_falls_back_without_client():
    state = _state()
    card = T.series_profile(state)
    leader = state.best_attempt()
    chosen, debug = FC.choose_final(
        state, card, {}, None, ReactConfig(), "FAKE", 0
    )
    assert chosen is leader
    assert debug["fallback"] == "no_client"


def test_choose_final_keep_then_scripted_override():
    state = _state()
    card = T.series_profile(state)
    config = ReactConfig(final_strategy="agent_family", final_selector_shuffle_seed=0)

    keep_client = ScriptedLLM(
        ["CHOICE: KEEP\nCONFIDENCE: high\nREASON: no clear winner"]
    )
    chosen, debug = FC.choose_final(
        state, card, {}, keep_client, config, "FAKE", 0
    )
    assert chosen is state.best_attempt()
    assert debug["decision"] == "keep"
    assert debug["prompt"]["user"]
    assert all(name not in debug["prompt"]["user"] for name in state.model_names)

    # Pick a non-leader label from the deterministic mapping, then let the
    # selector choose it. The mapping is the same on the second call.
    non_leader = next(
        row for row in debug["shortlist"] if row["attempt_id"] != debug["leader_id"]
    )
    override_client = ScriptedLLM(
        [f"CHOICE: {non_leader['label']}\nCONFIDENCE: high\nREASON: plausible shape"]
    )
    chosen, debug = FC.choose_final(
        state, card, {}, override_client, config, "FAKE", 0
    )
    assert debug["decision"] == "override"
    assert chosen.attempt_id == non_leader["attempt_id"]
    assert debug["chosen_id"] == non_leader["attempt_id"]


def test_choose_final_low_confidence_keeps_leader():
    state = _state()
    card = T.series_profile(state)
    config = ReactConfig(final_strategy="agent_family", final_selector_shuffle_seed=0)

    probe = ScriptedLLM(["CHOICE: KEEP\nCONFIDENCE: high\nREASON: probe"])
    _, debug = FC.choose_final(state, card, {}, probe, config, "FAKE", 0)
    non_leader = next(
        row for row in debug["shortlist"] if row["attempt_id"] != debug["leader_id"]
    )
    client = ScriptedLLM(
        [f"CHOICE: {non_leader['label']}\nCONFIDENCE: low\nREASON: unsure"]
    )
    chosen, debug = FC.choose_final(state, card, {}, client, config, "FAKE", 0)
    assert chosen is state.best_attempt()
    assert debug["decision"] == "fallback"
    assert debug["fallback"] == "low_confidence"


# ══════════════════════════════════════════════════════════════════════════════
# pipeline wiring
# ══════════════════════════════════════════════════════════════════════════════


def test_pipeline_agent_family_with_fake_repo(fake_repo):
    from orchestrator_react import pipeline as PL
    from test_pipeline import step

    config = ReactConfig(
        max_iterations=2,
        final_strategy="agent_family",
        final_selector_shuffle_seed=0,
    )
    combinator = ScriptedLLM(
        [
            step("evaluate_strategy", {"combine": "mean", "pool": FULL_POOL}),
            step("evaluate_strategy", {"combine": "median", "pool": FULL_POOL}),
        ]
    )
    selector = ScriptedLLM(
        ["CHOICE: KEEP\nCONFIDENCE: high\nREASON: fake repo test"]
    )
    outcome = PL.run_series(
        MODELS,
        "FAKE",
        0,
        config=config,
        source_file="fake.tsf",
        source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        client=combinator,
        selector_client=selector,
    )
    assert outcome.success
    assert outcome.final_choice
    assert outcome.final_choice["mode"] == "agent_family"
    assert outcome.final_choice["decision"] in {"keep", "override"}
    assert outcome.react.final_attempt is not None
    assert outcome.final_choice["chosen_id"] == outcome.react.final_attempt.attempt_id
    prompt = outcome.final_choice["prompt"]["user"]
    assert prompt
    assert all(name not in prompt for name in ["good", "mediocre", "bad"])
    assert "test actual is NOT shown" in prompt
    # End-to-end leakage check: the blind-window actuals exist on the outcome but
    # never reach the prompt.
    for value in outcome.test_values:
        assert f"{float(value):.4f}" not in prompt
    for attempt in outcome.state.attempts:
        assert attempt.attempt_id not in prompt


# ══════════════════════════════════════════════════════════════════════════════
# CLI/config wiring
# ══════════════════════════════════════════════════════════════════════════════


def test_cli_flags_parse_for_agent_family():
    from run_tsf_orchestrator import build_parser

    args = build_parser().parse_args(
        [
            "--dataset", "FAKE",
            "--final-strategy", "agent_family",
            "--final-selector-model", "gpt-oss:20b",
            "--final-selector-temperature", "0.25",
            "--final-selector-reasoning", "low",
            "--final-selector-seed", "13",
            "--final-selector-shuffle-seed", "7",
            "--final-selector-max-candidates", "4",
            "--no-final-selector-allow-keep",
            "--no-final-selector-low-confidence-keep",
            "--final-selector-degenerate-factor", "50",
        ]
    )
    assert args.final_strategy == "agent_family"
    assert args.final_selector_model == "gpt-oss:20b"
    assert args.final_selector_temperature == 0.25
    assert args.final_selector_reasoning == "low"
    assert args.final_selector_seed == 13
    assert args.final_selector_shuffle_seed == 7
    assert args.final_selector_max_candidates == 4
    assert args.final_selector_allow_keep is False
    assert args.final_selector_low_confidence_keep is False
    assert args.final_selector_degenerate_factor == 50.0


def test_orchestrator_config_wires_the_final_selector(monkeypatch):
    import run_tsf_orchestrator as R

    captured = {}

    def fake_run_dataset(**kwargs):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(R._pipeline, "run_dataset", fake_run_dataset)
    monkeypatch.setattr(R, "check_client", lambda client: (True, "mock"))

    R.exec_dataset_orchestrator(
        models=["good"],
        dataset="FAKE",
        use_llm=True,
        dry_run=True,
        indices=[],
        llm_logs=False,
        final_strategy="agent_family",
        final_selector_model="qwen3:8b",
        final_selector_temperature=0.25,
        final_selector_reasoning="low",
        final_selector_seed=13,
        final_selector_shuffle_seed=7,
        final_selector_max_candidates=4,
        final_selector_allow_keep=False,
        final_selector_low_confidence_keep=False,
        final_selector_degenerate_factor=50.0,
    )
    cfg = captured["config"]
    assert cfg.final_strategy == "agent_family"
    assert cfg.final_selector.model == "qwen3:8b"
    assert cfg.final_selector.temperature == 0.25
    assert cfg.final_selector.reasoning == "low"
    assert cfg.final_selector.seed == 13
    assert cfg.final_selector_shuffle_seed == 7
    assert cfg.final_selector_max_candidates == 4
    assert cfg.final_selector_allow_keep is False
    assert cfg.final_selector_low_confidence_keep is False
    assert cfg.final_selector_degenerate_factor == 50.0


def test_artifact_payload_includes_final_choice_only_when_present():
    from orchestrator_react import csv_writer as CW
    from orchestrator_react.pipeline import SeriesOutcome

    with_choice = SeriesOutcome(
        dataset="FAKE",
        dataset_index=0,
        horizon=2,
        success=False,
        error="failed",
        final_choice={"mode": "agent_family", "decision": "keep"},
    )
    payload = CW.artifacts_payload(with_choice)
    assert payload["final_choice"]["mode"] == "agent_family"

    without_choice = SeriesOutcome(
        dataset="FAKE", dataset_index=1, horizon=2, success=False, error="failed"
    )
    assert "final_choice" not in CW.artifacts_payload(without_choice)


# ══════════════════════════════════════════════════════════════════════════════
# seasonal guard and binary veto
# ══════════════════════════════════════════════════════════════════════════════


def test_seasonal_gate_requires_strength_and_cycles():
    card = {"seasonal_strength": 0.9, "n_points": 1000, "seasonal_period": 12}
    assert FC.seasonal_gate(card)["active"]
    assert not FC.seasonal_gate({**card, "n_points": 100})["active"]
    assert not FC.seasonal_gate({**card, "seasonal_strength": 0.1})["active"]
    assert not FC.seasonal_gate({**card, "seasonal_period": 1})["active"]
    assert not FC.seasonal_gate({"seasonal_strength": None})["active"]


def test_seasonal_guard_switches_only_when_gate_active():
    state, card, leader, challenger = _guard_state(active=True)
    chosen, debug = FC.choose_seasonal_guard(
        state, card, ReactConfig(), "FAKE", 0,
        forecast_for=lambda attempt: np.asarray(attempt.forecast, dtype=float),
    )
    assert debug["gate"]["active"]
    assert debug["decision"] == "switch"
    assert chosen is challenger
    assert debug["seasonal_id"] == "ID_SEASONAL_111"
    assert debug["leader_id"] == "ID_LEADER_000"
    assert len(debug["shortlist"]) == 2
    assert debug["shortlist"][0]["seasonal_dist"] is not None

    state2, card2, leader2, challenger2 = _guard_state(active=False)
    chosen2, debug2 = FC.choose_seasonal_guard(
        state2, card2, ReactConfig(), "FAKE", 0,
        forecast_for=lambda attempt: np.asarray(attempt.forecast, dtype=float),
    )
    assert not debug2["gate"]["active"]
    assert debug2["decision"] == "keep"
    assert chosen2 is leader2


def test_seasonal_veto_accepts_and_vetoes():
    state, card, leader, challenger = _guard_state(active=True)
    config = ReactConfig(final_strategy="seasonal_veto", final_selector_shuffle_seed=0)
    forecast_for = lambda attempt: np.asarray(attempt.forecast, dtype=float)

    _, dry = FC.choose_seasonal_veto(
        state, card, {}, None, config, "FAKE", 0,
        forecast_for=forecast_for, dry_run=True,
    )
    assert dry["decision"] == "switch"
    assert dry["prompt"]["user"]
    assert dry["leader_label"] in {"A", "B"}
    assert dry["seasonal_label"] in {"A", "B"}

    accept = ScriptedLLM(
        [f"CHOICE: {dry['seasonal_label']}\nCONFIDENCE: high\nREASON: seasonal"]
    )
    chosen, debug = FC.choose_seasonal_veto(
        state, card, {}, accept, config, "FAKE", 0, forecast_for=forecast_for
    )
    assert chosen is challenger
    assert debug["decision"] == "switch" and not debug["veto"]

    veto = ScriptedLLM(
        [f"CHOICE: {dry['leader_label']}\nCONFIDENCE: high\nREASON: leader"]
    )
    chosen, debug = FC.choose_seasonal_veto(
        state, card, {}, veto, config, "FAKE", 0, forecast_for=forecast_for
    )
    assert chosen is leader
    assert debug["decision"] == "keep" and debug["veto"]


def test_seasonal_veto_low_confidence_keeps_leader():
    state, card, leader, challenger = _guard_state(active=True)
    config = ReactConfig(final_strategy="seasonal_veto", final_selector_shuffle_seed=0)
    forecast_for = lambda attempt: np.asarray(attempt.forecast, dtype=float)
    _, dry = FC.choose_seasonal_veto(
        state, card, {}, None, config, "FAKE", 0,
        forecast_for=forecast_for, dry_run=True,
    )
    client = ScriptedLLM(
        [f"CHOICE: {dry['seasonal_label']}\nCONFIDENCE: low\nREASON: unsure"]
    )
    chosen, debug = FC.choose_seasonal_veto(
        state, card, {}, client, config, "FAKE", 0, forecast_for=forecast_for
    )
    assert chosen is leader
    assert debug["decision"] == "keep" and debug["veto"]
    assert debug["veto_reason"] == "low_confidence"


class _FailingClient:
    name = "failing"

    def complete(self, system: str, user: str) -> str:
        raise LLMError("server down")


def test_seasonal_veto_llm_error_falls_back_to_guard():
    state, card, leader, challenger = _guard_state(active=True)
    config = ReactConfig(final_strategy="seasonal_veto", final_selector_shuffle_seed=0)
    forecast_for = lambda attempt: np.asarray(attempt.forecast, dtype=float)
    chosen, debug = FC.choose_seasonal_veto(
        state, card, {}, _FailingClient(), config, "FAKE", 0, forecast_for=forecast_for
    )
    assert chosen is challenger
    assert debug["fallback"] == "llm_error"
    assert debug["decision"] == "switch"


def test_binary_prompt_has_no_hidden_labels_and_shows_seasonal_naive():
    state, card, leader, challenger = _guard_state(active=True)
    config = ReactConfig(final_strategy="seasonal_veto", final_selector_shuffle_seed=0)
    forecast_for = lambda attempt: np.asarray(attempt.forecast, dtype=float)
    _, debug = FC.choose_seasonal_veto(
        state, card, {}, None, config, "FAKE", 0,
        forecast_for=forecast_for, dry_run=True,
    )
    system, user = debug["prompt"]["system"], debug["prompt"]["user"]
    assert "SEASONAL NAIVE" in user
    assert "validation_score" not in user.lower()
    assert "attempt_id" not in user.lower()
    assert "origin" not in user.lower()
    for name in state.model_names:
        assert name not in system
        assert name not in user
    for attempt in (leader, challenger):
        assert attempt.attempt_id not in user
    assert "CHOICE: <A|B>" in system


def test_pipeline_seasonal_guard_wires_the_choice(fake_repo):
    from orchestrator_react import pipeline as PL
    from test_ingest_and_pool import MODELS
    from test_pipeline import step

    config = ReactConfig(max_iterations=2, final_strategy="seasonal_guard")
    combinator = ScriptedLLM(
        [
            step("evaluate_strategy", {"combine": "mean", "pool": FULL_POOL}),
            step("evaluate_strategy", {"combine": "median", "pool": FULL_POOL}),
        ]
    )
    outcome = PL.run_series(
        MODELS,
        "FAKE",
        0,
        config=config,
        source_file="fake.tsf",
        source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        client=combinator,
    )
    assert outcome.success
    assert outcome.final_choice["mode"] == "seasonal_guard"
    assert outcome.final_choice["gate"]["active"] is False
    assert outcome.final_choice["decision"] == "keep"
    assert outcome.react.final_attempt is not None
    assert outcome.final_choice["chosen_id"] == outcome.react.final_attempt.attempt_id


def test_pipeline_seasonal_veto_skips_llm_when_gate_inactive(fake_repo):
    from orchestrator_react import pipeline as PL
    from test_ingest_and_pool import MODELS
    from test_pipeline import step

    config = ReactConfig(max_iterations=2, final_strategy="seasonal_veto")
    combinator = ScriptedLLM(
        [
            step("evaluate_strategy", {"combine": "mean", "pool": FULL_POOL}),
            step("evaluate_strategy", {"combine": "median", "pool": FULL_POOL}),
        ]
    )
    selector = ScriptedLLM([])  # must not be called: the fake series fails the gate
    outcome = PL.run_series(
        MODELS,
        "FAKE",
        0,
        config=config,
        source_file="fake.tsf",
        source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        client=combinator,
        selector_client=selector,
    )
    assert outcome.success
    assert outcome.final_choice["mode"] == "seasonal_veto"
    assert outcome.final_choice["llm_called"] is False
    assert outcome.final_choice["decision"] == "keep"
    assert not selector.calls


def test_cli_flags_parse_for_seasonal_policies():
    from run_tsf_orchestrator import build_parser

    args = build_parser().parse_args(
        [
            "--dataset", "FAKE",
            "--final-strategy", "seasonal_veto",
            "--seasonal-guard-min-strength", "0.6",
            "--seasonal-guard-min-cycles", "8",
        ]
    )
    assert args.final_strategy == "seasonal_veto"
    assert args.seasonal_guard_min_strength == 0.6
    assert args.seasonal_guard_min_cycles == 8.0


def test_orchestrator_config_wires_the_seasonal_guard(monkeypatch):
    import run_tsf_orchestrator as R

    captured = {}

    def fake_run_dataset(**kwargs):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(R._pipeline, "run_dataset", fake_run_dataset)
    R.exec_dataset_orchestrator(
        models=["good"],
        dataset="FAKE",
        use_llm=False,
        dry_run=True,
        indices=[],
        llm_logs=False,
        final_strategy="seasonal_guard",
        seasonal_guard_min_strength=0.6,
        seasonal_guard_min_cycles=8.0,
    )
    cfg = captured["config"]
    assert cfg.final_strategy == "seasonal_guard"
    assert cfg.seasonal_guard_min_strength == 0.6
    assert cfg.seasonal_guard_min_cycles == 8.0


def test_shortlist_splits_weighted_attempts_by_weight_method():
    class Recipe:
        def __init__(self, method):
            self.method = method

    class FakeState:
        weights = {
            "w1": Recipe("inverse_error"),
            "w2": Recipe("topk_mean_per_horizon"),
        }

    attempts = [
        FakeAttempt("a1", {"combine": "weighted", "weights": "w1"}, 0.10, [1, 2, 3]),
        FakeAttempt("a2", {"combine": "weighted", "weights": "w2"}, 0.20, [3, 2, 1]),
    ]
    shortlist = FC.build_family_shortlist(
        attempts,
        lambda attempt: np.asarray(attempt.forecast, dtype=float),
        family_key=lambda attempt: FC._family_key(attempt, FakeState()),
    )
    assert len(shortlist) == 2
    assert {entry.attempt.attempt_id for entry in shortlist} == {"a1", "a2"}
