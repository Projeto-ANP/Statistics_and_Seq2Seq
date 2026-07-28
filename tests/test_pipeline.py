"""Per-series pipeline tests (Step 6.5) — Phases 0 to 4, no server, no GPU.

Covers the wiring, the effective-weight mapping that feeds `weights_by_horizon`,
and the CSV payload contract of Step 4 of the specification.

Run:  python -m pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import pipeline as PL
from orchestrator_react import pool as POOL
from orchestrator_react import tools as T
from orchestrator_react.config import LLMRole, ReactConfig
from orchestrator_react.data_source import SeriesAlignmentError, load_series_source
from orchestrator_react.llm import ScriptedLLM
from orchestrator_react.state import FULL_POOL
from test_orchestrator_react import make_state as _ms

from test_ingest_and_pool import (  # noqa: E402
    HORIZON,
    MODELS,
    N_SERIES,
    N_WINDOWS,
    fake_repo,
)
from test_orchestrator_react import make_state  # noqa: E402


def step(action: str, args: dict | None = None, thought: str = "t") -> str:
    return f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(args or {})}"


def run(fake_repo, index=0, config=None, client=None, **kw):
    return PL.run_series(
        MODELS, "FAKE", index,
        config=config or ReactConfig(max_iterations=4),
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"], client=client, **kw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# effective weights -> weights_by_horizon
# ══════════════════════════════════════════════════════════════════════════════


def test_mean_weights_are_uniform():
    s = make_state()
    w = PL.effective_weights(s, {"combine": "mean", "pool": FULL_POOL})
    assert w["nominal"] is False
    for h in range(s.horizon):
        col = w["weights"][str(h)]
        assert sum(col.values()) == pytest.approx(1.0)
        assert set(col) == set(s.model_names)
        assert all(v == pytest.approx(1 / s.n_models) for v in col.values())


def test_best_single_weights_are_one_hot():
    s = make_state()
    w = PL.effective_weights(s, {"combine": "best_single", "model": "good_a"})
    col = w["weights"]["0"]
    assert col["good_a"] == 1.0
    assert sum(col.values()) == pytest.approx(1.0)


def test_median_weights_point_at_the_selected_element():
    """The median is a selection, so its implied weights must reproduce it."""
    s = make_state()
    w = PL.effective_weights(s, {"combine": "median", "pool": FULL_POOL})
    for h in range(s.horizon):
        col = w["weights"][str(h)]
        assert sum(col.values()) == pytest.approx(1.0)
        rebuilt = sum(col[n] * s.test_preds[j, h] for j, n in enumerate(s.model_names))
        assert rebuilt == pytest.approx(float(np.median(s.test_preds[:, h])))


def test_trimmed_mean_weights_reproduce_the_forecast():
    s = make_state()
    spec = {"combine": "trimmed_mean", "pool": FULL_POOL, "trim_pct": 0.2}
    w = PL.effective_weights(s, spec)
    forecast, _ = s.apply_to_test(spec)
    for h in range(s.horizon):
        col = w["weights"][str(h)]
        rebuilt = sum(col[n] * s.test_preds[j, h] for j, n in enumerate(s.model_names))
        assert rebuilt == pytest.approx(forecast[h])


def test_weighted_weights_reproduce_the_forecast():
    s = make_state()
    handle = T.weights_inverse_error(s, pool=FULL_POOL)["weights"]
    spec = {"combine": "weighted", "pool": FULL_POOL, "weights": handle}
    w = PL.effective_weights(s, spec)
    forecast, _ = s.apply_to_test(spec)
    for h in range(s.horizon):
        col = w["weights"][str(h)]
        rebuilt = sum(col[n] * s.test_preds[j, h] for j, n in enumerate(s.model_names))
        assert rebuilt == pytest.approx(forecast[h])


def test_dba_weights_are_flagged_nominal():
    """A DTW barycentre is not a weighted average; the CSV must not pretend it is."""
    s = make_state()
    w = PL.effective_weights(s, {"combine": "dba", "pool": FULL_POOL})
    assert w["nominal"] is True
    assert "not a weighted average" in w["note"]


def test_weights_only_cover_the_selected_pool():
    s = make_state()
    top = T.select_top_k(s, k=2)
    w = PL.effective_weights(s, {"combine": "mean", "pool": top["pool"]})
    col = w["weights"]["0"]
    outside = set(s.model_names) - set(top["models"])
    assert all(col[n] == 0.0 for n in outside)
    assert sum(col.values()) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# run_series
# ══════════════════════════════════════════════════════════════════════════════


def test_run_series_deterministic_arm(fake_repo):
    """No LLM configured: the best seeded baseline is applied."""
    cfg = ReactConfig(combinator=LLMRole(model=None))
    out = run(fake_repo, config=cfg)
    assert out.success and not out.error
    assert len(out.forecast) == HORIZON
    assert len(out.test_values) == HORIZON
    assert out.react.stop_reason == "no_llm_client"
    assert out.react.final_attempt.origin == "baseline"
    assert out.series_card and out.pool_card
    assert out.phase2["attempts_seeded"] == 3


def test_run_series_with_a_scripted_agent(fake_repo):
    llm = ScriptedLLM([
        step("select_top_k", {"k": 2}, "drop the biased model"),
        step("evaluate_strategy", {"strategy": {"combine": "mean", "pool": "pool1"}}, "test it"),
        step("accept", {"attempt_id": "a4", "confidence": 0.6,
                        "justification": "the pool disagrees a lot, a lean mean is safer"}),
    ])
    out = run(fake_repo, client=llm)
    assert out.success
    assert out.react.stop_reason == "agent_accepted"
    assert out.react.final_attempt.spec["pool"] == "pool1"
    assert out.selected_models() == ["good", "mediocre"]
    assert "disagrees" in out.csv_fields()["justificativa_final"]


def test_phase4_matches_the_selected_strategy(fake_repo):
    """Phase 4 must apply exactly the strategy Phase 3 chose, via the same code."""
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    direct, _ = out.state.apply_to_test(out.react.final_attempt.spec)
    assert out.forecast == pytest.approx(direct)


def test_sanity_check_runs_on_the_final_forecast(fake_repo):
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    assert "ok" in out.sanity
    assert out.sanity["n_points"] == HORIZON


def test_external_baselines_are_attached(fake_repo):
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    assert out.external_baselines["mean"]["available"] is True
    assert out.external_baselines["FFORMA"]["available"] is False


def test_calibration_gate_skips_the_loop(fake_repo):
    cfg = ReactConfig(calibration_gate=True, calibration_gate_kendall=0.0)
    out = run(fake_repo, config=cfg, client=ScriptedLLM([]))
    assert out.react.stop_reason == "calibration_gate"
    assert out.csv_fields()["calibration_gate_triggered"] is True


def test_pool_ablation_flows_into_the_outcome(fake_repo):
    cfg = ReactConfig(pool_mode="top_k_stable", pool_k=2, combinator=LLMRole(model=None))
    out = run(fake_repo, config=cfg)
    assert out.csv_fields()["pool_composition_mode"] == "top_k_stable"


# ══════════════════════════════════════════════════════════════════════════════
# CSV payload — the Step 4 contract
# ══════════════════════════════════════════════════════════════════════════════


EXPECTED_FIELDS = {
    # kept, re-pointed (Section 4.3)
    "description", "decision_report", "score_preset", "tool_missing", "tools_called",
    "best_strategy_name", "best_strategy_method", "best_strategy_params",
    "predict_debug", "selected_base_models", "weights_by_horizon",
    "n_tool_calls", "n_evaluate_calls", "provenance_ok",
    "weights_concentration", "equivalent_to_pool_mean", "pool_mean_relative_diff",
    "n_pool_models", "effective_models", "n_effective_models",
    "final_candidate_names", "final_candidate_count",
    # new (Section 4.4)
    "series_profile_json", "ranking_stability_score", "error_correlation_groups",
    "pool_composition_mode", "react_iterations_used", "react_early_stopped",
    "react_trajectory_json", "baseline_results_json", "weights_handle_resolved",
    "agent_model_combinator", "agent_model_diagnostico", "agent_model_relato",
    "accept_confidence", "calibration_gate_triggered", "ablation_config",
    "justificativa_final",
    # beyond 4.4: the deterministic replacement for self-reported confidence
    "selection_margin", "selection_bootstrap_pvalue", "selection_dm_pvalue",
    "selection_verdict",
}

REMOVED_FIELDS = {
    "debate_ran", "debate_trigger", "approach_pre_debate", "approach_post_debate",
    "debate_explanation", "proposer_selected_names", "proposer_think",
    "skeptic_think", "statistician_think", "pattern_analyst_think",
    "pattern_analyst_trend_champion", "when_good", "selection_explanation",
}


def test_csv_fields_match_the_specification(fake_repo):
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    fields = out.csv_fields()
    assert set(fields) == EXPECTED_FIELDS
    assert not (set(fields) & REMOVED_FIELDS), "old debate fields must be gone"


def test_csv_fields_are_all_serialisable(fake_repo):
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    for key, value in out.csv_fields().items():
        assert isinstance(value, (str, int, float, bool, type(None))), f"{key} is {type(value)}"


def test_json_columns_round_trip(fake_repo):
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    fields = out.csv_fields()
    for key in (
        "description", "series_profile_json", "react_trajectory_json",
        "baseline_results_json", "weights_by_horizon", "tools_called",
        "best_strategy_params", "error_correlation_groups",
    ):
        json.loads(fields[key])  # must parse


def test_decision_json_is_reproducible(fake_repo):
    """The decision block is the `decision.json` of Section 3.1."""
    llm = ScriptedLLM([
        step("select_top_k", {"k": 2}),
        step("weights_inverse_error", {"pool": "pool1", "shrinkage": 0.2}),
        step("evaluate_strategy",
             {"strategy": {"combine": "weighted", "pool": "pool1", "weights": "w1"}}),
        step("accept", {"attempt_id": "a4", "justification": "why"}),
    ])
    out = run(fake_repo, client=llm)
    d = out.decision()
    assert d["strategy"]["combine"] == "weighted"
    assert d["weights"]["computed_by"] == "inverse_error"
    assert d["weights"]["params"]["shrinkage"] == 0.2
    assert d["weights"]["fit_windows"] == [0, 1, 2]
    assert d["validation"]["n_windows"] == N_WINDOWS
    assert d["validation"]["backtest_mode"] == "expanding"
    assert d["config"]["ablation"] == out.config.fingerprint()


def test_weights_handle_resolved_is_populated_only_when_weighted(fake_repo):
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    assert json.loads(out.csv_fields()["weights_handle_resolved"]) == {}

    llm = ScriptedLLM([
        step("weights_ols", {"pool": FULL_POOL}),
        step("evaluate_strategy",
             {"strategy": {"combine": "weighted", "pool": FULL_POOL, "weights": "w1"}}),
        step("accept", {"attempt_id": "a4"}),
    ])
    out2 = run(fake_repo, client=llm)
    if out2.react.final_attempt.spec["combine"] == "weighted":
        resolved = json.loads(out2.csv_fields()["weights_handle_resolved"])
        assert sum(resolved["weights"].values()) == pytest.approx(1.0)


def test_baseline_results_json_has_both_sources(fake_repo):
    out = run(fake_repo, config=ReactConfig(combinator=LLMRole(model=None)))
    baselines = json.loads(out.csv_fields()["baseline_results_json"])
    assert set(baselines["seeded"]) == set(POOL.SEED_BASELINES)
    assert baselines["external"]["mean"]["available"] is True


def test_model_names_per_role_are_recorded(fake_repo):
    cfg = ReactConfig(
        combinator=LLMRole(model="gpt-oss:20b"),
        diagnostician=LLMRole(model="qwen3:8b"),
        reporter=LLMRole(model=None),
    )
    out = run(fake_repo, config=cfg, client=ScriptedLLM([step("accept", {})]))
    fields = out.csv_fields()
    assert fields["agent_model_combinator"] == "gpt-oss:20b"
    assert fields["agent_model_diagnostico"] == "qwen3:8b"
    assert fields["agent_model_relato"] == "none"


def test_ablation_fingerprint_distinguishes_runs(fake_repo):
    a = run(fake_repo, config=ReactConfig(name="A", combinator=LLMRole(model=None)))
    b = run(fake_repo, config=ReactConfig(name="B", pool_mode="top_k_error",
                                          combinator=LLMRole(model=None)))
    assert a.csv_fields()["ablation_config"] != b.csv_fields()["ablation_config"]


def test_trajectory_column_stays_small(fake_repo):
    llm = ScriptedLLM([step("series_profile", {}), step("accept", {"attempt_id": "a1"})])
    out = run(fake_repo, client=llm)
    assert len(out.csv_fields()["react_trajectory_json"]) < 3000


def test_effective_models_expose_a_collapsed_weighting(fake_repo):
    """A "weighted combination of N models" can be one model wearing a label.

    Real case, NN5 series 11: OLS on three windows put weight 1.0 on
    NaiveMovingAverage and 0.0 on the other eight, while the row still reported
    nine selected models. Anyone analysing pool size would have read nine.
    """
    s = make_state()
    POOL.run_phase2(s, s.config)
    top = T.select_top_k(s, k=4)
    # force a one-hot weighting, exactly what OLS produces when it collapses
    handle = T.weights_softmax_neg_error(s, pool=top["pool"], eta=20.0)["weights"]
    recipe = s.get_weights_recipe(handle)
    recipe.resolved = np.array([1.0, 0.0, 0.0, 0.0])

    out = PL.SeriesOutcome(dataset="FAKE", dataset_index=0, horizon=s.horizon, state=s,
                           config=s.config)
    attempt, _ = s.evaluate({"combine": "weighted", "pool": top["pool"], "weights": handle})
    from orchestrator_react.react_loop import ReactResult

    out.react = ReactResult(final_attempt=attempt)

    assert len(out.selected_models()) == 4, "the pool is still four models"
    assert len(out.effective_models()) == 1, "only one of them carries weight"
    fields = out.csv_fields()
    assert fields["n_pool_models"] == 4
    assert fields["n_effective_models"] == 1
    assert "effective=1" in out.decision_report()


def test_effective_models_match_the_pool_when_weights_are_spread(fake_repo):
    s = make_state()
    POOL.run_phase2(s, s.config)
    top = T.select_top_k(s, k=3)
    handle = T.weights_inverse_error(s, pool=top["pool"], shrinkage=0.5)["weights"]
    attempt, _ = s.evaluate({"combine": "weighted", "pool": top["pool"], "weights": handle})
    from orchestrator_react.react_loop import ReactResult

    out = PL.SeriesOutcome(dataset="FAKE", dataset_index=0, horizon=s.horizon, state=s,
                           config=s.config, react=ReactResult(final_attempt=attempt))
    assert out.effective_models() == sorted(out.selected_models())


def test_bootstrap_is_marked_unreliable_with_three_windows():
    """Resampling three values gives a p-value in roughly {0, 0.5, 1}.

    It over-rejects, so with few windows the verdict follows Diebold-Mariano and
    the bootstrap is reported as context rather than as evidence.
    """
    s = make_state()
    POOL.run_phase2(s, s.config)
    T.evaluate_strategy(s, {"combine": "best_single", "model": "bad"})
    conf = s.selection_confidence()
    assert s.n_windows == 3
    assert conf["bootstrap_reliable"] is False
    assert conf["bootstrap_pvalue"] is not None, "still reported, just not trusted"
    # the verdict must be consistent with DM alone
    dm = conf["dm_pvalue"]
    if dm is not None:
        assert conf["verdict"] == ("separated" if dm < 0.10 else "indistinguishable")


def test_near_uniform_weights_are_flagged_as_equivalent_to_the_mean():
    """Real NN5 case: inverse-error weights over five models landed on
    0.2056 / 0.2080 / 0.1964 / 0.1945 / 0.1955, within 4% of the 0.200 equal
    weight. Calling that a weighted combination overstates the weighting."""
    s = make_state()
    POOL.run_phase2(s, s.config)
    top = T.select_top_k(s, k=4)
    handle = T.weights_inverse_error(s, pool=top["pool"], shrinkage=0.95)["weights"]
    attempt, _ = s.evaluate({"combine": "weighted", "pool": top["pool"], "weights": handle})
    from orchestrator_react.react_loop import ReactResult

    out = PL.SeriesOutcome(dataset="FAKE", dataset_index=0, horizon=s.horizon, state=s,
                           config=s.config, react=ReactResult(final_attempt=attempt))
    red = out.reducibility()
    assert red["equivalent_to_pool_mean"] is True
    assert red["pool_mean_relative_diff"] < 0.01
    assert red["weights_concentration"] is not None


def test_a_genuinely_concentrated_weighting_is_not_equivalent():
    s = make_state()
    POOL.run_phase2(s, s.config)
    top = T.select_top_k(s, k=4)
    handle = T.weights_softmax_neg_error(s, pool=top["pool"], eta=20.0)["weights"]
    recipe = s.get_weights_recipe(handle)
    recipe.resolved = np.array([0.97, 0.01, 0.01, 0.01])
    attempt, _ = s.evaluate({"combine": "weighted", "pool": top["pool"], "weights": handle})
    from orchestrator_react.react_loop import ReactResult

    out = PL.SeriesOutcome(dataset="FAKE", dataset_index=0, horizon=s.horizon, state=s,
                           config=s.config, react=ReactResult(final_attempt=attempt))
    red = out.reducibility()
    assert red["equivalent_to_pool_mean"] is False
    assert red["pool_mean_relative_diff"] > 0.01


def test_the_plain_mean_is_trivially_equivalent_to_itself():
    s = make_state()
    POOL.run_phase2(s, s.config)
    attempt = [a for a in s.attempts if a.spec["combine"] == "mean"][0]
    from orchestrator_react.react_loop import ReactResult

    out = PL.SeriesOutcome(dataset="FAKE", dataset_index=0, horizon=s.horizon, state=s,
                           config=s.config, react=ReactResult(final_attempt=attempt))
    red = out.reducibility()
    assert red["equivalent_to_pool_mean"] is True
    assert red["pool_mean_relative_diff"] == pytest.approx(0.0, abs=1e-9)


def test_best_single_is_never_called_equivalent_to_a_mean():
    s = make_state()
    POOL.run_phase2(s, s.config)
    attempt, _ = s.evaluate({"combine": "best_single", "model": "good_a"})
    from orchestrator_react.react_loop import ReactResult

    out = PL.SeriesOutcome(dataset="FAKE", dataset_index=0, horizon=s.horizon, state=s,
                           config=s.config, react=ReactResult(final_attempt=attempt))
    assert out.reducibility()["equivalent_to_pool_mean"] is False


# ══════════════════════════════════════════════════════════════════════════════
# run_dataset
# ══════════════════════════════════════════════════════════════════════════════


def test_run_dataset_iterates_every_series(fake_repo):
    cfg = ReactConfig(combinator=LLMRole(model=None))
    outs = list(PL.run_dataset(
        MODELS, "FAKE", source_file="fake.tsf", config=cfg,
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
    ))
    assert len(outs) == N_SERIES
    assert all(o.success for o in outs)
    assert [o.dataset_index for o in outs] == list(range(N_SERIES))


def test_run_dataset_honours_an_index_subset(fake_repo):
    cfg = ReactConfig(combinator=LLMRole(model=None))
    outs = list(PL.run_dataset(
        MODELS, "FAKE", source_file="fake.tsf", config=cfg, indices=[1, 3],
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
    ))
    assert [o.dataset_index for o in outs] == [1, 3]


def test_run_dataset_captures_a_per_series_failure(fake_repo):
    """One broken series must not kill the whole dataset run."""
    cfg = ReactConfig(n_validation_windows=99, combinator=LLMRole(model=None))
    outs = list(PL.run_dataset(
        MODELS, "FAKE", source_file="fake.tsf", config=cfg, indices=[0, 1],
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
    ))
    assert len(outs) == 2
    assert all(not o.success for o in outs)
    assert all("validation windows" in o.error for o in outs)
    assert all(o.csv_fields()["description"] for o in outs)


def test_run_dataset_reraises_a_systematic_alignment_error(fake_repo, tmp_path):
    """A wrong .tsf breaks every series, so the run must stop, not emit N failures."""
    (tmp_path / "tiny.tsf").write_text(
        "@relation T\n@attribute series_name string\n@frequency monthly\n@data\nA:1,2,3\n",
        encoding="utf-8",
    )
    cfg = ReactConfig(combinator=LLMRole(model=None))
    with pytest.raises(SeriesAlignmentError):
        list(PL.run_dataset(
            MODELS, "FAKE", source_file="tiny.tsf", config=cfg,
            source_dir=str(tmp_path), results_dir=fake_repo["results_dir"],
        ))


def test_failed_outcome_still_produces_a_csv_row(fake_repo):
    out = PL.SeriesOutcome(
        dataset="FAKE", dataset_index=0, horizon=0,
        success=False, error="boom", config=ReactConfig(),
    )
    fields = out.csv_fields()
    assert set(fields) == EXPECTED_FIELDS
    assert "boom" in fields["decision_report"]
    assert json.loads(fields["description"])["error"] == "boom"


# ══════════════════════════════════════════════════════════════════════════════
# integration with the real data
# ══════════════════════════════════════════════════════════════════════════════


REAL_RESULTS = "./timeseries/mestrado/resultados"
REAL_SOURCE = os.path.expanduser("~/Documents/mestrado/forecasting_datasets")
REAL_MODELS = ["ARIMA", "ETS", "THETA", "rf", "catboost", "NaiveSeasonal"]

real_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(REAL_SOURCE, "mes_11_venda_mensal.tsf")),
    reason="real ANP data not available on this machine",
)


@real_data
def test_real_anp_end_to_end():
    """Phases 0-4 on real series, with a scripted agent standing in for Ollama."""
    cfg = ReactConfig(name="smoke", n_validation_windows=3, max_iterations=5)
    source = load_series_source(
        "mes_11_venda_mensal.tsf", n_expected_series=182, source_dir=REAL_SOURCE
    )
    llm = ScriptedLLM([
        step("select_stable", {"k": 3}, "the ranking moves across windows"),
        step("evaluate_strategy", {"strategy": {"combine": "mean", "pool": "pool1"}}, "test it"),
        step("accept", {"attempt_id": "a4", "confidence": 0.7,
                        "justification": "unstable ranking favours an equal-weight stable subset"}),
    ])
    out = PL.run_series(
        REAL_MODELS, "ANP_MONTHLY", 0, config=cfg,
        source=source, results_dir=REAL_RESULTS, client=llm,
    )

    assert out.success
    assert out.horizon == 12 and len(out.forecast) == 12
    assert np.all(np.isfinite(out.forecast))
    assert len(out.test_values) == 12

    fields = out.csv_fields()
    assert set(fields) == EXPECTED_FIELDS
    profile = json.loads(fields["series_profile_json"])
    assert profile["n_points"] == 407
    assert profile["seasonal_period"] == 12
    weights = json.loads(fields["weights_by_horizon"])
    assert len(weights) == 12
    assert sum(weights["0"].values()) == pytest.approx(1.0)
    assert json.loads(fields["baseline_results_json"])["external"]["ADE"]["available"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
