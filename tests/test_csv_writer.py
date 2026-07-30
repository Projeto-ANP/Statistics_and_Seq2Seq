"""CSV writer tests (Step 6.7) — the Step 4 output contract.

The point of this file is that the 13 evaluation columns of Section 4.1 stay
byte-for-byte compatible with what `run_tsf_orchestrator.py` produced, so rows from
the two architectures can sit in the same analysis.

Run:  python -m pytest tests/test_csv_writer.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import csv_writer as W
from orchestrator_react import pipeline as PL
from orchestrator_react.config import LLMRole, ReactConfig
from orchestrator_react.llm import ScriptedLLM

from test_ingest_and_pool import HORIZON, MODELS, N_SERIES, fake_repo  # noqa: E402


def outcome_for(fake_repo, index=0, client=None, config=None):
    return PL.run_series(
        MODELS, "FAKE", index,
        config=config or ReactConfig(combinator=LLMRole(model=None)),
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"], client=client,
    )


# ══════════════════════════════════════════════════════════════════════════════
# schema
# ══════════════════════════════════════════════════════════════════════════════


def test_core_columns_come_first_and_unchanged():
    """Section 4.1: same names, same order as the old writer."""
    assert W.COLS_SERIE[:13] == [
        "dataset_index", "horizon", "regressor", "mape", "pocid", "smape", "rmse",
        "msmape", "mae", "test", "predictions", "start_test", "final_test",
    ]


def test_removed_columns_are_absent():
    """Section 4.2: the debate schema must not survive anywhere."""
    assert not (set(W.COLS_SERIE) & set(W.REMOVED_COLUMNS))
    for name in ("debate_ran", "proposer_think", "pattern_analyst_narrative",
                 "selection_explanation", "when_good"):
        assert name not in W.COLS_SERIE


def test_new_columns_are_all_present():
    """Section 4.4: every traceability column exists."""
    for name in (
        "series_profile_json", "ranking_stability_score", "error_correlation_groups",
        "pool_composition_mode", "react_iterations_used", "react_early_stopped",
        "react_trajectory_json", "baseline_results_json", "weights_handle_resolved",
        "agent_model_combinator", "agent_model_diagnostico", "agent_model_relato",
        "accept_confidence", "calibration_gate_triggered", "ablation_config",
        "justificativa_final",
        "selection_margin", "selection_bootstrap_pvalue", "selection_dm_pvalue",
        "selection_verdict",
    ):
        assert name in W.COLS_SERIE


def test_schema_has_no_duplicates():
    assert len(W.COLS_SERIE) == len(set(W.COLS_SERIE)) == 58


# ══════════════════════════════════════════════════════════════════════════════
# metrics — must match the old computation exactly
# ══════════════════════════════════════════════════════════════════════════════


def test_metrics_match_the_legacy_computation():
    """Reproduces `run_tsf_orchestrator.py:555-577` and compares."""
    pytest.importorskip("all_functions")
    from sklearn.metrics import mean_absolute_percentage_error as sk_mape

    from all_functions import (
        calculate_mae, calculate_msmape, calculate_rmse, calculate_smape, pocid,
    )

    rng = np.random.default_rng(3)
    test = rng.normal(500, 50, 12)
    preds = test + rng.normal(0, 20, 12)

    p2, t2 = preds.reshape(1, -1), test.reshape(1, -1)
    legacy = {
        "smape": calculate_smape(p2, t2),
        "rmse": calculate_rmse(p2, t2),
        "msmape": calculate_msmape(p2, t2),
        "mae": calculate_mae(p2, t2),
        "mape": sk_mape(test, preds),
        "pocid": pocid(test, preds),
    }
    got = W.compute_metrics(preds, test)
    for key, value in legacy.items():
        assert got[key] == pytest.approx(float(np.asarray(value).ravel()[0]))


def test_metrics_truncate_to_the_common_length():
    """The old writer cut both vectors to min_len; so does this one."""
    test = [1.0, 2.0, 3.0, 4.0]
    preds = [1.0, 2.0, 3.0]
    got = W.compute_metrics(preds, test)
    expected = W.compute_metrics(preds, test[:3])
    assert got["rmse"] == pytest.approx(expected["rmse"])


def test_metrics_of_an_empty_forecast_are_nan():
    got = W.compute_metrics([], [])
    assert all(np.isnan(v) for v in got.values())


# ══════════════════════════════════════════════════════════════════════════════
# row assembly
# ══════════════════════════════════════════════════════════════════════════════


def test_row_has_every_column(fake_repo):
    row = W.build_row(outcome_for(fake_repo), regressor="react_test")
    assert set(row) == set(W.COLS_SERIE)


def test_row_core_values(fake_repo):
    out = outcome_for(fake_repo, index=2)
    row = W.build_row(out, regressor="react_test")
    assert row["dataset_index"] == "2"
    assert row["horizon"] == HORIZON
    assert row["regressor"] == "react_test"
    assert row["start_test"] == "INICIO"
    assert row["predictions"] == [out.forecast]
    assert row["test"] == [out.test_values]
    assert np.isfinite(row["rmse"])


def test_failed_series_still_produces_a_row():
    """Dropping the row would hide the failure from the analysis."""
    out = PL.SeriesOutcome(
        dataset="FAKE", dataset_index=7, horizon=0,
        success=False, error="ingestion blew up", config=ReactConfig(),
    )
    row = W.build_row(out, regressor="react_test")
    assert set(row) == set(W.COLS_SERIE)
    assert row["dataset_index"] == "7"
    assert np.isnan(row["rmse"]) and np.isnan(row["mape"])
    assert row["predictions"] == [[]]
    assert "ingestion blew up" in row["decision_report"]


def test_artifacts_payload_is_complete(fake_repo):
    out = outcome_for(fake_repo)
    payload = W.artifacts_payload(out)
    for key in ("decision", "series_card", "pool_card", "diagnosis", "phase2", "react",
                "predict_debug", "sanity", "config"):
        assert key in payload
    json.dumps(payload, default=str)  # must serialise


def test_artifacts_payload_carries_the_tool_error_detail(fake_repo):
    """Before this, `tool_missing=True` on a CSV row said nothing about WHY: the
    `kind` and `detail` of the failing call lived only in `state.tool_errors`,
    which was never written anywhere — reading it back meant re-parsing the
    `tools_called` CSV column by hand. This pins that it is now saved.

    `select_top_k` does not accept `unexpected_field`, and only `evaluate_strategy`
    is on the permissive list that tolerates a stray argument, so this call is a
    real, reproducible `unknown_argument` failure through the actual registry path
    — not a mocked one."""
    llm = ScriptedLLM([
        'Thought: t\nAction: select_top_k\nAction Input: {"k": 3, "unexpected_field": "x"}',
        'Thought: t\nAction: accept\nAction Input: {"attempt_id": "a1"}',
    ])
    out = outcome_for(fake_repo, client=llm)
    payload = W.artifacts_payload(out)

    tools = payload["react"]["tools"]
    assert tools["tool_missing"] is True
    assert any(e["kind"] == "unknown_argument" for e in tools["errors"])
    assert any(e["tool"] == "select_top_k" for e in tools["errors"])


# ══════════════════════════════════════════════════════════════════════════════
# writer
# ══════════════════════════════════════════════════════════════════════════════


def test_writer_creates_the_file_with_the_header(tmp_path, fake_repo):
    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    assert os.path.exists(w.csv_path)
    header = pd.read_csv(w.csv_path, sep=";").columns.tolist()
    assert header == W.COLS_SERIE


def test_writer_appends_and_reads_back(tmp_path, fake_repo):
    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    for i in range(3):
        w.write(outcome_for(fake_repo, index=i))
    assert w.rows_written == 3

    df = pd.read_csv(w.csv_path, sep=";")
    assert len(df) == 3
    assert df.columns.tolist() == W.COLS_SERIE
    assert df["dataset_index"].tolist() == [0, 1, 2]
    assert df["rmse"].notna().all()


def test_json_columns_survive_the_round_trip(tmp_path, fake_repo):
    """The `;` separator and embedded JSON must not corrupt each other."""
    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    w.write(outcome_for(fake_repo))
    df = pd.read_csv(w.csv_path, sep=";")
    row = df.iloc[0]
    for column in ("description", "series_profile_json", "react_trajectory_json",
                   "baseline_results_json", "weights_by_horizon", "best_strategy_params",
                   "final_candidate_names"):
        parsed = json.loads(row[column])
        assert parsed is not None
    profile = json.loads(row["series_profile_json"])
    assert profile["n_validation_windows"] == 3


def test_forecast_survives_the_round_trip(tmp_path, fake_repo):
    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    out = outcome_for(fake_repo)
    w.write(out)
    df = pd.read_csv(w.csv_path, sep=";")
    from orchestrator_react.ingest import extract_values

    assert extract_values(df.iloc[0]["predictions"]) == pytest.approx(out.forecast)
    assert extract_values(df.iloc[0]["test"]) == pytest.approx(out.test_values)


def test_writer_saves_artifacts(tmp_path, fake_repo):
    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    path = w.write(outcome_for(fake_repo, index=1))
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["dataset_index"] == 1
    df = pd.read_csv(w.csv_path, sep=";")
    assert df.iloc[0]["llm_artifacts_path"] == path


def test_writer_can_skip_artifacts(tmp_path, fake_repo):
    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path), save_artifacts=False)
    assert w.write(outcome_for(fake_repo)) == ""


def test_writer_migrates_an_old_debate_schema(tmp_path, fake_repo):
    """Re-running over a v1 file must drop the debate columns, not resurrect them."""
    base = tmp_path / "exp1"
    base.mkdir(parents=True)
    legacy = pd.DataFrame([{
        "dataset_index": 0, "horizon": 12, "regressor": "orchestrator_llm_v1_pattern",
        "mape": 0.1, "pocid": 50.0, "smape": 0.1, "rmse": 1.0, "msmape": 0.1, "mae": 1.0,
        "test": "[1,2]", "predictions": "[1,2]", "start_test": "INICIO", "final_test": "2024-11-30",
        "debate_ran": True, "proposer_think": "blah", "pattern_analyst_narrative": "blah",
        "when_good": "sometimes", "selection_explanation": "because",
    }])
    legacy.to_csv(base / "FAKE.csv", sep=";", index=False)

    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    df = pd.read_csv(w.csv_path, sep=";")
    assert df.columns.tolist() == W.COLS_SERIE
    assert "debate_ran" not in df.columns
    assert "when_good" not in df.columns
    assert len(df) == 1  # the old row is preserved, only reshaped
    assert df.iloc[0]["rmse"] == 1.0

    w.write(outcome_for(fake_repo))
    df2 = pd.read_csv(w.csv_path, sep=";")
    assert len(df2) == 2
    assert df2.columns.tolist() == W.COLS_SERIE


def test_writer_is_idempotent_on_its_own_schema(tmp_path, fake_repo):
    w1 = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    w1.write(outcome_for(fake_repo))
    w2 = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    df = pd.read_csv(w2.csv_path, sep=";")
    assert len(df) == 1
    assert df.columns.tolist() == W.COLS_SERIE


def test_agent_run_is_fully_traceable_from_the_csv(tmp_path, fake_repo):
    """Given a row, the whole decision must be reconstructible."""
    llm = ScriptedLLM([
        f'Thought: prune the pool\nAction: select_top_k\nAction Input: {{"k": 2}}',
        'Thought: test it\nAction: evaluate_strategy\nAction Input: '
        '{"strategy": {"combine": "mean", "pool": "pool1"}}',
        'Thought: done\nAction: accept\nAction Input: '
        '{"attempt_id": "a4", "confidence": 0.8, "justification": "lean pool, unstable ranking"}',
    ])
    w = W.ResultWriter("FAKE", "exp1", results_dir=str(tmp_path))
    w.write(outcome_for(fake_repo, client=llm, config=ReactConfig(max_iterations=4)))

    row = pd.read_csv(w.csv_path, sep=";").iloc[0]
    trajectory = json.loads(row["react_trajectory_json"])
    assert [t["action"] for t in trajectory] == ["select_top_k", "evaluate_strategy", "accept"]
    assert row["react_iterations_used"] == 3
    assert row["accept_confidence"] == 0.8
    assert "lean pool" in row["justificativa_final"]
    assert row["best_strategy_method"] == "mean"
    assert json.loads(row["selected_base_models"]) == ["good", "mediocre"]
    assert row["tool_missing"] in (False, "False")
    assert row["ablation_config"]


# ══════════════════════════════════════════════════════════════════════════════
# the CLI
# ══════════════════════════════════════════════════════════════════════════════


def test_old_debate_parameters_raise_a_helpful_error():
    """An old call pasted from a notebook must say what to use instead."""
    import run_tsf_orchestrator as R

    for name in ("proposer_model", "skeptic_model", "statistician_model",
                 "pattern_analyst_model", "train_window", "rolling", "debate"):
        with pytest.raises(TypeError, match=name):
            R.exec_dataset_orchestrator(MODELS, "FAKE", **{name: "x"})


def test_model_config_duck_typing():
    """The old `ModelConfig(model=..., temperature=...)` still works."""
    import run_tsf_orchestrator as R
    from dataclasses import dataclass

    @dataclass
    class ModelConfig:
        model: str
        temperature: float

    role = R._as_role(ModelConfig(model="qwen3:14b", temperature=0.7))
    assert role.model == "qwen3:14b" and role.temperature == 0.7
    assert R._as_role("gpt-oss:20b").model == "gpt-oss:20b"
    assert R._as_role(None).model is None
    assert R._as_role("none").model is None


def test_exec_dataset_orchestrator_deterministic_arm(tmp_path, fake_repo):
    """The programmatic call, exactly as it was used before."""
    import run_tsf_orchestrator as R

    summary = R.exec_dataset_orchestrator(
        MODELS,
        dataset="FAKE",
        source_file="fake.tsf",
        use_llm=False,
        source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path),
        version="prog",
        llm_logs=False,
    )
    assert summary["n_ok"] == N_SERIES and summary["n_failed"] == 0
    assert summary["experiment"] == "orchestrator_react_prog"
    df = pd.read_csv(summary["csv_path"], sep=";")
    assert len(df) == N_SERIES
    assert df.columns.tolist() == W.COLS_SERIE


def test_exec_dataset_orchestrator_with_an_agent(tmp_path, fake_repo, monkeypatch):
    """A single combinator model drives the whole decision."""
    import orchestrator_react.pipeline as PLmod
    import run_tsf_orchestrator as R

    script = ScriptedLLM([
        'Thought: prune\nAction: select_top_k\nAction Input: {"k": 2}',
        'Thought: test\nAction: evaluate_strategy\nAction Input: '
        '{"strategy": {"combine": "mean", "pool": "pool1"}}',
        'Thought: done\nAction: accept\nAction Input: '
        '{"attempt_id": "a4", "confidence": 0.7, "justification": "lean pool"}',
    ] * N_SERIES)
    monkeypatch.setattr(PLmod, "build_client", lambda role: script if role.enabled else None)
    # the preflight lives in the entry point and imports build_client directly
    monkeypatch.setattr(R, "check_client", lambda client: (True, "OK"))

    summary = R.exec_dataset_orchestrator(
        MODELS, dataset="FAKE", source_file="fake.tsf",
        combinator_model="gpt-oss:20b",
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path), version="agent", llm_logs=False, indices=[0],
    )
    assert summary["n_ok"] == 1
    row = pd.read_csv(summary["csv_path"], sep=";").iloc[0]
    assert row["agent_model_combinator"] == "gpt-oss:20b"
    assert row["react_iterations_used"] == 3
    assert json.loads(row["react_trajectory_json"])[0]["action"] == "select_top_k"


def test_exec_dataset_orchestrator_actually_uses_the_diagnostician(tmp_path, fake_repo, monkeypatch):
    """End-to-end regression test for the ordering bug this session fixed:
    `cfg.diagnostic_llm` used to be computed before `ReactConfig.from_env` could
    still change `cfg.diagnostician.model`, so an env-var-set diagnostician looked
    configured in the log but Phase 1 never actually called it. There is no
    separate flag any more — this asserts the LLM reading really lands in the row,
    not just that the config says a model name."""
    import orchestrator_react.pipeline as PLmod
    import run_tsf_orchestrator as R
    from orchestrator_react.phases import DIAGNOSIS_SYSTEM

    diag_json = (
        '{"regime": "seasonal_dominated", "predictability": "high", '
        '"combination_hint": "robust", "risks": [], "narrative": "test"}'
    )

    def fake_build_client(role):
        if not role.enabled:
            return None
        return ScriptedLLM([diag_json] * N_SERIES) if role.model == "qwen3:8b" else None

    monkeypatch.setattr(PLmod, "build_client", fake_build_client)

    summary = R.exec_dataset_orchestrator(
        MODELS, dataset="FAKE", source_file="fake.tsf",
        combinator_model=None,  # off: this test isolates the diagnostician role
        diagnostician_model="qwen3:8b",
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path), version="diag_e2e", llm_logs=False, indices=[0],
    )
    assert summary["n_ok"] == 1
    outcome = summary["outcomes"][0]
    assert outcome.diagnosis["source"] == "llm"
    assert outcome.diagnosis["regime"] == "seasonal_dominated"
    row = pd.read_csv(summary["csv_path"], sep=";").iloc[0]
    assert row["agent_model_diagnostico"] == "qwen3:8b"


def test_cli_environment_overrides_the_flags(monkeypatch, tmp_path, fake_repo):
    """Section 3.5: swapping the model per role must not need a code change."""
    import run_tsf_orchestrator as R

    monkeypatch.setenv("REACT_MODEL_COMBINATOR", "none")
    summary = R.exec_dataset_orchestrator(
        MODELS, dataset="FAKE", source_file="fake.tsf",
        combinator_model="gpt-oss:20b",
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path), version="envtest", llm_logs=False, indices=[0],
    )
    row = pd.read_csv(summary["csv_path"], sep=";").iloc[0]
    assert row["agent_model_combinator"] == "none"


def test_cli_separates_input_from_output(tmp_path, fake_repo):
    """A smoke run must be able to write outside the results tree."""
    import run_tsf_orchestrator as R

    R.main([
        "--dataset", "FAKE", "--source", "fake.tsf",
        "--source-dir", fake_repo["source_dir"],
        "--results-dir", fake_repo["results_dir"], "--output-dir", str(tmp_path),
        "--models", *MODELS, "--version", "iso", "--no-llm", "--limit", "1",
    ])
    assert (tmp_path / "orchestrator_react_iso" / "FAKE.csv").exists()
    assert not os.path.exists(
        os.path.join(fake_repo["results_dir"], "orchestrator_react_iso")
    )


def test_cli_runs_end_to_end(tmp_path, fake_repo, capsys):
    import run_tsf_orchestrator as R

    code = R.main([
        "--dataset", "FAKE",
        "--source", "fake.tsf",
        "--source-dir", fake_repo["source_dir"],
        "--results-dir", fake_repo["results_dir"],
        "--output-dir", str(tmp_path),
        "--models", *MODELS,
        "--version", "cli_smoke",
        "--no-llm",
        "--limit", "2",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "ok: 2" in out and "failed: 0" in out

    csv_path = tmp_path / "orchestrator_react_cli_smoke" / "FAKE.csv"
    df = pd.read_csv(csv_path, sep=";")
    assert len(df) == 2
    assert df.columns.tolist() == W.COLS_SERIE


def test_cli_dry_run_writes_nothing(tmp_path, fake_repo):
    import run_tsf_orchestrator as R

    code = R.main([
        "--dataset", "FAKE", "--source", "fake.tsf",
        "--source-dir", fake_repo["source_dir"],
        "--results-dir", fake_repo["results_dir"], "--output-dir", str(tmp_path),
        "--models", *MODELS, "--version", "dry", "--no-llm", "--limit", "1", "--dry-run",
    ])
    assert code == 0
    assert not (tmp_path / "orchestrator_react_dry").exists()


def test_cli_reports_a_bad_tsf_clearly(tmp_path, fake_repo, capsys):
    import run_tsf_orchestrator as R

    (tmp_path / "tiny.tsf").write_text(
        "@relation T\n@attribute series_name string\n@frequency monthly\n@data\nA:1,2,3\n",
        encoding="utf-8",
    )
    code = R.main([
        "--dataset", "FAKE", "--source", "tiny.tsf", "--source-dir", str(tmp_path),
        "--results-dir", fake_repo["results_dir"], "--output-dir", str(tmp_path / "out"),
        "--models", *MODELS, "--version", "bad", "--no-llm",
    ])
    assert code == 2
    assert "ALIGNMENT ERROR" in capsys.readouterr().err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_nested_selection_and_ols_gate_are_exposed_and_reach_the_config(tmp_path, fake_repo):
    """These two protocol flags are the correctness fix from the nested-selection
    work: `nested_selection` must default True, and both must be settable from the
    CLI-level function rather than only by hand-building a `ReactConfig`."""
    import run_tsf_orchestrator as R

    summary = R.exec_dataset_orchestrator(
        MODELS,
        dataset="FAKE",
        source_file="fake.tsf",
        use_llm=False,
        source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path),
        version="protocol_default",
        llm_logs=False,
    )
    assert summary["n_ok"] == N_SERIES

    summary_off = R.exec_dataset_orchestrator(
        MODELS,
        dataset="FAKE",
        source_file="fake.tsf",
        use_llm=False,
        source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path),
        version="protocol_off",
        nested_selection=False,
        min_windows_for_ols=1,
        llm_logs=False,
    )
    assert summary_off["n_ok"] == N_SERIES
    off_cfg = summary_off["outcomes"][0].config
    assert off_cfg.nested_selection is False
    assert off_cfg.min_windows_for_ols == 1
    # the fingerprint must actually change with the protocol, or ablation_config
    # cannot tell two runs apart
    assert summary["ablation_config"] != summary_off["ablation_config"]

    default_cfg = summary["outcomes"][0].config
    assert default_cfg.nested_selection is True
    assert default_cfg.min_windows_for_ols == 5


# ══════════════════════════════════════════════════════════════════════════════
# zero_actual_diagnostics — visibility for an sMAPE metric-artifact, not a fix
# to the metric itself (which must stay byte-identical, Section 4.1)
# ══════════════════════════════════════════════════════════════════════════════


def test_flags_a_test_window_containing_a_literal_zero():
    """The real ANP_MONTHLY case that motivated this: series 85's test window."""
    out = W.zero_actual_diagnostics([20.0, 45.0, 15.0, 20.0, 5.0, 30.0, 0.0, 0.0, 10.0, 0.0, 15.0, 5.0])
    assert out["test_has_zero_actual"] is True
    assert out["test_min_abs_actual"] == 0.0


def test_does_not_flag_a_window_with_no_zero():
    out = W.zero_actual_diagnostics([20.0, 45.0, 15.0, 3.2])
    assert out["test_has_zero_actual"] is False
    assert out["test_min_abs_actual"] == pytest.approx(3.2)


def test_a_small_but_nonzero_value_is_reported_not_flagged():
    """The boolean is deliberately strict (exact zero, not "small"): the
    continuous `test_min_abs_actual` is what lets analysis apply its own
    threshold instead of inheriting one hardcoded into the pipeline."""
    out = W.zero_actual_diagnostics([0.003, 10.0])
    assert out["test_has_zero_actual"] is False
    assert out["test_min_abs_actual"] == pytest.approx(0.003)


def test_empty_or_all_nan_actual_reports_none_not_false():
    """`False` would silently claim 'checked, no zero'; there was nothing to check."""
    assert W.zero_actual_diagnostics([])["test_has_zero_actual"] is None
    assert W.zero_actual_diagnostics([float("nan"), float("nan")])["test_has_zero_actual"] is None


def test_nan_values_are_ignored_rather_than_propagated():
    out = W.zero_actual_diagnostics([float("nan"), 0.0, 5.0])
    assert out["test_has_zero_actual"] is True
    assert out["test_min_abs_actual"] == 0.0


def test_a_negative_zero_is_still_flagged_by_magnitude():
    out = W.zero_actual_diagnostics([-1e-12, 8.0])
    assert out["test_has_zero_actual"] is True


def test_the_flag_reaches_the_csv_row(fake_repo):
    out = outcome_for(fake_repo)
    out.test_values = [0.0, 5.0, 10.0]
    row = W.build_row(out, regressor="exp1")
    assert row["test_has_zero_actual"] is True
    assert row["test_min_abs_actual"] == 0.0


def test_it_reads_only_test_values_already_used_by_the_metrics():
    """Same input the byte-identical metrics already consume, at the same point
    in the pipeline — this cannot see anything the metrics do not."""
    import inspect

    sig = inspect.signature(W.zero_actual_diagnostics)
    assert list(sig.parameters) == ["actual"]


# ══════════════════════════════════════════════════════════════════════════════
# cross-series observability — a v5 run has to be analysable afterwards
# ══════════════════════════════════════════════════════════════════════════════


def test_artifact_records_the_dataset_prior_and_the_card_shown(fake_repo):
    """The prior and the card are what the agent was given beyond its own series.
    Without them in the artifact there is no way to tell, after a run, whether the
    cross-series context helped — which is the whole question v5 exists to answer."""
    from orchestrator_react.state import _spec_label

    out = outcome_for(fake_repo)
    out.state.strategy_prior = {_spec_label(a.spec): 0.5 + i * 0.1
                                for i, a in enumerate(out.state.attempts)}
    block = W.artifacts_payload(out)["cross_series"]
    assert block["strategy_prior"], "the prior itself must be recorded"
    assert block["prior_best"] and block["prior_worst"]
    assert "best_on_this_dataset" in block["dataset_card_shown"]
    json.dumps(block, default=str)  # must serialise


def test_artifact_records_whether_the_pooled_fit_learned_anything(fake_repo):
    """`degenerate=True` means uniform weights wearing a meta-model's name. A run
    must not be able to look like it used one when it did not."""
    from orchestrator_react import meta_model as MM

    out = outcome_for(fake_repo)
    out.state.pooled_meta_model = MM.PooledMetaModel(
        feature_names=MM.FEATURE_NAMES, model_names=list(out.state.model_names),
        objective="fforma", n_train_series=181, degenerate=True,
    )
    block = W.artifacts_payload(out)["cross_series"]["pooled_meta_model"]
    assert block == {"objective": "fforma", "n_train_series": 181,
                     "n_features": len(MM.FEATURE_NAMES), "degenerate": True}


def test_cross_series_block_is_empty_when_nothing_cross_series_ran(fake_repo):
    out = outcome_for(fake_repo)
    assert W.artifacts_payload(out)["cross_series"] == {}


def test_bookkeeping_never_breaks_the_artifact(fake_repo, monkeypatch):
    """The artifact is the audit trail; a formatting helper must not be able to
    lose it."""
    import orchestrator_react.prompts as P

    out = outcome_for(fake_repo)
    out.state.strategy_prior = {"mean": 0.5}
    monkeypatch.setattr(P, "build_dataset_card", lambda *a, **k: 1 / 0)
    block = W.artifacts_payload(out)["cross_series"]
    assert block["strategy_prior"] == {"mean": 0.5}
    assert "dataset_card_shown" not in block


# ══════════════════════════════════════════════════════════════════════════════
# max_llm_failures — one flaky series must not end a 182-series run
# ══════════════════════════════════════════════════════════════════════════════


def _flaky_client(n_failing_series):
    """A client that burns through the LLM-error retry budget for the first
    `n_failing_series` series, then answers normally.

    Counting whole series rather than calls matters: `react_loop` retries a
    transport error `LLM_ERROR_RETRIES` times before giving up, so killing one
    series takes that many consecutive failures, not one."""
    from orchestrator_react.llm import LLMError
    from orchestrator_react.react_loop import LLM_ERROR_RETRIES

    per_series = LLM_ERROR_RETRIES + 1
    state = {"left": n_failing_series * per_series}

    class Flaky:
        name = "flaky"

        def complete(self, system, user):
            if "Reply with the single word OK" in system:
                return "OK"
            if state["left"] > 0:
                state["left"] -= 1
                raise LLMError("simulated transport failure")
            return 'Thought: t\nAction: accept\nAction Input: {"attempt_id": "a1"}'

    return Flaky()


def test_one_failing_series_no_longer_ends_the_run(tmp_path, fake_repo, monkeypatch):
    """The behaviour that forced the ANP resume: aborting on the FIRST llm error.
    Resuming then re-ran the remainder in a chunk small enough to silently disable
    the pooled meta-model, so the fix belongs here, not in the resume."""
    import orchestrator_react.pipeline as PLmod
    import run_tsf_orchestrator as R

    client = _flaky_client(1)
    monkeypatch.setattr(PLmod, "build_client", lambda role: client if role.enabled else None)
    monkeypatch.setattr(R, "check_client", lambda c: (True, "OK"))

    summary = R.exec_dataset_orchestrator(
        MODELS, dataset="FAKE", source_file="fake.tsf", combinator_model="gpt-oss:20b",
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path), version="flaky1", llm_logs=False,
        max_llm_failures=5, pooled_meta_model=False, dataset_card=False,
    )
    assert summary["n_ok"] == N_SERIES, "every series still produced a row"
    assert summary["n_llm_failures"] == 1


def test_a_systematically_broken_server_still_aborts(tmp_path, fake_repo, monkeypatch):
    """The original guarantee must survive: a run that silently degrades to the
    deterministic baseline answers a different question while the log says ok."""
    import orchestrator_react.pipeline as PLmod
    import run_tsf_orchestrator as R

    client = _flaky_client(N_SERIES)
    monkeypatch.setattr(PLmod, "build_client", lambda role: client if role.enabled else None)
    monkeypatch.setattr(R, "check_client", lambda c: (True, "OK"))

    with pytest.raises(RuntimeError, match="max_llm_failures budget"):
        R.exec_dataset_orchestrator(
            MODELS, dataset="FAKE", source_file="fake.tsf", combinator_model="gpt-oss:20b",
            source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
            output_dir=str(tmp_path), version="flakyall", llm_logs=False,
            max_llm_failures=2, pooled_meta_model=False, dataset_card=False,
        )


def test_allow_baseline_fallback_still_never_aborts(tmp_path, fake_repo, monkeypatch):
    import orchestrator_react.pipeline as PLmod
    import run_tsf_orchestrator as R

    client = _flaky_client(N_SERIES)
    monkeypatch.setattr(PLmod, "build_client", lambda role: client if role.enabled else None)
    monkeypatch.setattr(R, "check_client", lambda c: (True, "OK"))

    summary = R.exec_dataset_orchestrator(
        MODELS, dataset="FAKE", source_file="fake.tsf", combinator_model="gpt-oss:20b",
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
        output_dir=str(tmp_path), version="fallback", llm_logs=False,
        allow_baseline_fallback=True, max_llm_failures=1,
        pooled_meta_model=False, dataset_card=False,
    )
    assert summary["n_llm_failures"] == N_SERIES
    assert summary["n_ok"] == N_SERIES
