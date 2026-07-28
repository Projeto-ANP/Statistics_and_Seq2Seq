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
    ):
        assert name in W.COLS_SERIE


def test_schema_has_no_duplicates():
    assert len(W.COLS_SERIE) == len(set(W.COLS_SERIE)) == 43


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


def test_cli_builds_the_config_from_flags():
    import run_tsf_react as R

    args = R.build_parser().parse_args([
        "--dataset", "FAKE", "--version", "abl", "--windows", "3",
        "--pool-mode", "top_k_stable", "--pool-k", "5", "--max-iterations", "6",
        "--combinator", "qwen3:14b", "--diagnostic-llm",
    ])
    cfg = R.build_config(args)
    assert cfg.name == "abl"
    assert cfg.pool_mode == "top_k_stable" and cfg.pool_k == 5
    assert cfg.max_iterations == 6
    assert cfg.combinator.model == "qwen3:14b"
    assert cfg.diagnostic_llm is True


def test_cli_no_llm_disables_every_role():
    import run_tsf_react as R

    args = R.build_parser().parse_args(
        ["--dataset", "FAKE", "--no-llm", "--combinator", "gpt-oss:20b"]
    )
    cfg = R.build_config(args)
    assert cfg.combinator.model is None
    assert cfg.diagnostician.model is None
    assert cfg.reporter.model is None
    assert cfg.diagnostic_llm is False


def test_cli_environment_overrides_the_flags(monkeypatch):
    """Section 3.5: swapping the model per role must not need a code change."""
    import run_tsf_react as R

    monkeypatch.setenv("REACT_MODEL_COMBINATOR", "gemma4:26b")
    args = R.build_parser().parse_args(
        ["--dataset", "FAKE", "--combinator", "gpt-oss:20b"]
    )
    assert R.build_config(args).combinator.model == "gemma4:26b"


def test_cli_separates_input_from_output(tmp_path, fake_repo):
    """A smoke run must be able to write outside the results tree."""
    import run_tsf_react as R

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
    import run_tsf_react as R

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
    import run_tsf_react as R

    code = R.main([
        "--dataset", "FAKE", "--source", "fake.tsf",
        "--source-dir", fake_repo["source_dir"],
        "--results-dir", fake_repo["results_dir"], "--output-dir", str(tmp_path),
        "--models", *MODELS, "--version", "dry", "--no-llm", "--limit", "1", "--dry-run",
    ])
    assert code == 0
    assert not (tmp_path / "orchestrator_react_dry").exists()


def test_cli_reports_a_bad_tsf_clearly(tmp_path, fake_repo, capsys):
    import run_tsf_react as R

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
