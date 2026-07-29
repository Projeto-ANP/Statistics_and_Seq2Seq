"""Phase 0 (ingestion) and Phase 2 (pool evaluation) tests — no LLM, no GPU.

Two layers:

* synthetic — writes tiny `.tsf` and result CSVs into a tmp dir, so the whole
  ingestion contract is exercised without depending on the data being present;
* integration — runs against the real `ANP_MONTHLY` results and
  `../forecasting_datasets/mes_11_venda_mensal.tsf`, and skips cleanly when those
  are not on this machine. This is the check that says "it will run on the server".

Run:  python -m pytest tests/test_ingest_and_pool.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import ingest as I
from orchestrator_react import pool as P
from orchestrator_react.config import ReactConfig
from orchestrator_react.data_source import (
    SeriesAlignmentError,
    drop_zero_windows,
    load_series_source,
    parse_tsf,
    verify_alignment,
)
from orchestrator_react.state import FULL_POOL


HORIZON = 6
N_WINDOWS = 3
N_SERIES = 4
MODELS = ["good", "mediocre", "bad"]


# ──────────────────────────────────────────────────────────────────────────────
# synthetic fixture: a miniature copy of the real layout
# ──────────────────────────────────────────────────────────────────────────────


def _series(idx: int, n: int = 90) -> np.ndarray:
    rng = np.random.default_rng(100 + idx)
    t = np.arange(n, dtype=float)
    return 50.0 + idx * 10 + 0.4 * t + 5.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, n)


@pytest.fixture
def fake_repo(tmp_path):
    """Builds `<tmp>/results/<MODEL>/normal/FAKE.csv` and `<tmp>/source/fake.tsf`."""
    results_dir = tmp_path / "results"
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    all_series = [_series(i) for i in range(N_SERIES)]

    lines = [
        "# synthetic dataset",
        "@relation FAKE",
        "@attribute series_name string",
        "@attribute start_timestamp date",
        "@frequency monthly",
        "@horizon 6",
        "@missing false",
        "@equallength true",
        "@data",
    ]
    for i, s in enumerate(all_series):
        lines.append(f"S{i}:1990-01-01 00-00-00:" + ",".join(f"{v:.6f}" for v in s))
    tsf_path = source_dir / "fake.tsf"
    tsf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bias = {"good": 0.5, "mediocre": 3.0, "bad": 12.0}
    rng = np.random.default_rng(0)
    for model in MODELS:
        rows = []
        for idx, s in enumerate(all_series):
            # Windows peeled off the end, oldest first — same as the real pipeline.
            for k in range(N_WINDOWS, -1, -1):
                end = len(s) - k * HORIZON
                actual = s[end - HORIZON : end]
                preds = actual + rng.normal(bias[model], 1.0, HORIZON)
                rows.append(
                    {
                        "dataset_index": idx,
                        "horizon": HORIZON,
                        "regressor": f"{model}_normal",
                        "mape": 0.1, "pocid": 50.0, "smape": 0.1,
                        "rmse": 1.0, "msmape": 0.1, "mae": 1.0,
                        "test": str(list(actual)),
                        "predictions": str(list(preds)),
                        "start_test": pd.Timestamp("2000-01-31") + pd.DateOffset(months=(N_WINDOWS - k) * HORIZON),
                        "final_test": pd.Timestamp("2000-06-30") + pd.DateOffset(months=(N_WINDOWS - k) * HORIZON),
                    }
                )
        out = results_dir / model / "normal"
        out.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(out / "FAKE.csv", sep=";", index=False)

    # A flat-layout external baseline, like resultados/mean/<DATASET>.csv
    flat = results_dir / "mean"
    flat.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset_index": i, "horizon": HORIZON, "regressor": "mean",
                "mape": 0.05 + i / 100, "pocid": 60.0, "smape": 0.05,
                "rmse": 2.0, "msmape": 0.05, "mae": 1.5,
                "test": "[]", "predictions": "[]",
                "start_test": "2000-01-31", "final_test": "2000-06-30",
            }
            for i in range(N_SERIES)
        ]
    ).to_csv(flat / "FAKE.csv", sep=";", index=False)

    return {
        "results_dir": str(results_dir),
        "source_dir": str(source_dir),
        "tsf": str(tsf_path),
        "series": all_series,
    }


# ══════════════════════════════════════════════════════════════════════════════
# .tsf parsing
# ══════════════════════════════════════════════════════════════════════════════


def test_parse_tsf_reads_metadata_and_rows(fake_repo):
    tsf = parse_tsf(fake_repo["tsf"])
    assert len(tsf) == N_SERIES
    assert tsf.frequency == "monthly"
    assert tsf.horizon == HORIZON
    assert tsf.attributes == ["series_name", "start_timestamp"]
    assert tsf.rows[0]["series_name"] == "S0"
    assert tsf.rows[2]["series_value"] == pytest.approx(fake_repo["series"][2], abs=1e-5)


def test_parse_tsf_preserves_row_order(fake_repo):
    """Positional dataset_index only works if order is preserved."""
    tsf = parse_tsf(fake_repo["tsf"])
    assert [r["series_name"] for r in tsf.rows] == [f"S{i}" for i in range(N_SERIES)]


def test_parse_tsf_handles_missing_markers(tmp_path):
    p = tmp_path / "m.tsf"
    p.write_text(
        "@relation X\n@attribute series_name string\n@frequency daily\n@data\nA:1,?,3\n",
        encoding="utf-8",
    )
    values = parse_tsf(str(p)).rows[0]["series_value"]
    assert np.isnan(values[1])
    assert values[0] == 1.0 and values[2] == 3.0


def test_parse_tsf_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_tsf("/nope/does_not_exist.tsf")


# ══════════════════════════════════════════════════════════════════════════════
# series filter and alignment
# ══════════════════════════════════════════════════════════════════════════════


def test_drop_zero_windows_detects_zero_blocks():
    assert drop_zero_windows(np.concatenate([np.zeros(20), np.ones(4)]), 24) is True
    assert drop_zero_windows(np.ones(48), 24) is False
    # under half zeros in every window -> keep
    assert drop_zero_windows(np.concatenate([np.zeros(10), np.ones(14)]), 24) is False


def test_load_source_without_filter(fake_repo):
    src = load_series_source("fake.tsf", n_expected_series=N_SERIES, source_dir=fake_repo["source_dir"])
    assert src.n_series == N_SERIES
    assert src.filter_applied == "none"
    assert src.frequency == "monthly"


def test_load_source_applies_the_zero_filter(tmp_path):
    """When the raw count does not match, the known filters are tried."""
    lines = [
        "@relation Z", "@attribute series_name string", "@frequency monthly", "@data",
    ]
    lines.append("keep:" + ",".join(["5"] * 48))
    lines.append("drop:" + ",".join(["0"] * 24 + ["5"] * 24))
    lines.append("keep2:" + ",".join(["7"] * 48))
    p = tmp_path / "z.tsf"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    src = load_series_source("z.tsf", n_expected_series=2, source_dir=str(tmp_path))
    assert src.n_series == 2
    assert src.filter_applied.startswith("drop_zero_windows")
    assert [m["series_name"] for m in src.metadata] == ["keep", "keep2"]


def test_load_source_raises_when_counts_cannot_be_reconciled(fake_repo):
    """The reduced single-series ETT case must fail loudly, not guess."""
    with pytest.raises(SeriesAlignmentError, match="no known filter reconciles"):
        load_series_source("fake.tsf", n_expected_series=7, source_dir=fake_repo["source_dir"])


def test_verify_alignment_accepts_the_matching_tail():
    s = np.arange(50, dtype=float)
    info = verify_alignment(s, expected_tail=s[-6:], dataset_index=0)
    assert info["verified"] is True
    assert info["horizon"] == 6


def test_verify_alignment_rejects_the_wrong_series():
    s = np.arange(50, dtype=float)
    with pytest.raises(SeriesAlignmentError, match="series mismatch"):
        verify_alignment(s, expected_tail=np.arange(6) * 100.0, dataset_index=3)


def test_verify_alignment_rejects_a_too_short_series():
    with pytest.raises(SeriesAlignmentError, match="fewer than"):
        verify_alignment(np.arange(3.0), expected_tail=np.arange(6.0), dataset_index=0)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0 — ingestion
# ══════════════════════════════════════════════════════════════════════════════


def test_extract_values_parses_numpy_repr():
    cell = "[ 1.14642965e+10  8.29365261e+09\n -2.6e+01]"
    assert I.extract_values(cell) == pytest.approx([1.14642965e10, 8.29365261e09, -26.0])
    assert I.extract_values(None) == []
    assert I.extract_values([1, 2]) == [1.0, 2.0]


def test_count_series(fake_repo):
    assert I.count_series("FAKE", "good", results_dir=fake_repo["results_dir"]) == N_SERIES


def test_load_series_builds_a_usable_state(fake_repo):
    ing = I.load_series(
        MODELS, "FAKE", dataset_index=1,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    s = ing.state
    assert s.n_windows == N_WINDOWS
    assert s.n_models == len(MODELS)
    assert s.horizon == HORIZON
    assert ing.horizon == HORIZON
    assert ing.alignment["verified"] is True
    assert s.freq == "monthly"
    assert s.model_names == MODELS


def test_ingestion_windows_are_chronological(fake_repo):
    """Window 0 must be the oldest — the expanding backtest depends on it."""
    ing = I.load_series(
        MODELS, "FAKE", dataset_index=0,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    full = fake_repo["series"][0]
    for w in range(N_WINDOWS):
        end = len(full) - (N_WINDOWS - w) * HORIZON
        assert ing.state.y_true[w] == pytest.approx(full[end - HORIZON : end], abs=1e-4)


def test_train_series_excludes_the_test_window(fake_repo):
    ing = I.load_series(
        MODELS, "FAKE", dataset_index=2,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    full = fake_repo["series"][2]
    assert ing.state.train_series.size == full.size - HORIZON
    assert ing.state.train_series == pytest.approx(full[:-HORIZON], abs=1e-4)


def test_test_forecasts_come_from_the_last_row(fake_repo):
    ing = I.load_series(
        MODELS, "FAKE", dataset_index=0,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    df = I.read_model_predictions("good", "FAKE", fake_repo["results_dir"])
    last = df[df["dataset_index"] == 0].sort_values("start_test").iloc[-1]
    assert ing.state.test_preds[0] == pytest.approx(I.extract_values(last["predictions"]), abs=1e-6)


def test_ingestion_without_a_source_still_works(fake_repo):
    ing = I.load_series(
        MODELS, "FAKE", dataset_index=0, results_dir=fake_repo["results_dir"]
    )
    assert ing.state.train_series is None
    assert any("no .tsf source" in w for w in ing.warnings)


def test_ingestion_rejects_a_missing_model(fake_repo):
    with pytest.raises(I.IngestionError, match="result file not found"):
        I.load_series(MODELS + ["ghost"], "FAKE", 0, results_dir=fake_repo["results_dir"])


def test_ingestion_rejects_an_unknown_index(fake_repo):
    with pytest.raises(I.IngestionError, match="no rows"):
        I.load_series(MODELS, "FAKE", 99, results_dir=fake_repo["results_dir"])


def test_ingestion_rejects_too_few_windows(fake_repo):
    cfg = ReactConfig(n_validation_windows=10)
    with pytest.raises(I.IngestionError, match="validation windows"):
        I.load_series(MODELS, "FAKE", 0, config=cfg, results_dir=fake_repo["results_dir"])


def test_mislabelled_timestamps_warn_but_do_not_block(fake_repo):
    """Timestamps may diverge between models; position is what defines a window.

    This is the project's own convention, stated in
    `combinations/ade.py::_check_windows_alignment`: the real dates may differ
    between models, the relative position must not, and the first model's dates are
    the canonical axis. The real ETTH1 case is six `ONLY_*` models that wrote their
    index with freq="15min" on hourly data, so the identical 24 observations carry
    2016-12-29 in those files and 2018-06-26 in the other thirteen.
    """
    path = os.path.join(fake_repo["results_dir"], "bad", "normal", "FAKE.csv")
    df = pd.read_csv(path, sep=";")
    for col in ("start_test", "final_test"):
        df[col] = pd.to_datetime(df[col]) - pd.DateOffset(years=2)
    df.to_csv(path, sep=";", index=False)

    ing = I.load_series(
        MODELS, "FAKE", 0, source_file="fake.tsf",
        source_dir=fake_repo["source_dir"], results_dir=fake_repo["results_dir"],
    )
    assert ing.state.n_models == len(MODELS), "no model was dropped over a label"
    assert any("final_test timestamp" in w for w in ing.warnings)
    # the reported date comes from the reference model, per the ADE convention
    reference = pd.read_csv(
        os.path.join(fake_repo["results_dir"], MODELS[0], "normal", "FAKE.csv"), sep=";"
    )
    reference["final_test"] = pd.to_datetime(reference["final_test"])
    expected = reference[reference["dataset_index"] == 0]["final_test"].max()
    assert ing.final_test == expected


def test_different_actuals_on_the_test_window_are_fatal(fake_repo):
    """Tolerating labels must not weaken the real guarantee.

    No other component of the project checks this: `aux.get_predictions_models`
    aligns positionally with no comparison at all, and ADE/FFORMA only compare
    window COUNTS. Different values on the blind window mean the models forecast
    different periods, and combining them would mix periods.
    """
    path = os.path.join(fake_repo["results_dir"], "bad", "normal", "FAKE.csv")
    df = pd.read_csv(path, sep=";")
    df["start_test"] = pd.to_datetime(df["start_test"])
    last = df[df["dataset_index"] == 0].sort_values("start_test").index[-1]
    df.loc[last, "test"] = str([777.0] * HORIZON)
    df.to_csv(path, sep=";", index=False)

    with pytest.raises(I.IngestionError, match="not the same window"):
        I.load_series(MODELS, "FAKE", 0, results_dir=fake_repo["results_dir"])


def test_models_with_extra_windows_align_from_the_end(fake_repo):
    """A model run with a bigger budget still lines up: windows peel off the end.

    ADE and FFORMA refuse this outright (they require identical window counts).
    Here it is fine, because `run_tsf_normal_series` always peels the newest window
    first, so the last N windows of a 30-window model are the last N of a 4-window
    one.
    """
    path = os.path.join(fake_repo["results_dir"], "good", "normal", "FAKE.csv")
    df = pd.read_csv(path, sep=";")
    df["start_test"] = pd.to_datetime(df["start_test"])
    df["final_test"] = pd.to_datetime(df["final_test"])
    # invent two older windows for series 0, as if it had been run with more origins
    oldest = df[df["dataset_index"] == 0].sort_values("start_test").iloc[0]
    extra = []
    for back in (1, 2):
        row = oldest.copy()
        row["start_test"] = oldest["start_test"] - pd.DateOffset(months=HORIZON * back)
        row["final_test"] = oldest["final_test"] - pd.DateOffset(months=HORIZON * back)
        row["test"] = str([-1.0] * HORIZON)
        row["predictions"] = str([-1.0] * HORIZON)
        extra.append(row)
    pd.concat([df, pd.DataFrame(extra)]).to_csv(path, sep=";", index=False)

    ing = I.load_series(MODELS, "FAKE", 0, results_dir=fake_repo["results_dir"])
    # the invented old windows are simply not among the last 3 used
    assert not np.any(ing.state.y_true == -1.0)
    assert ing.state.n_windows == N_WINDOWS


def test_ingestion_detects_misaligned_actuals(fake_repo, tmp_path):
    """If two models report different actuals, their windows are not the same."""
    path = os.path.join(fake_repo["results_dir"], "bad", "normal", "FAKE.csv")
    df = pd.read_csv(path, sep=";")
    mask = df["dataset_index"] == 0
    df.loc[mask, "test"] = str([999.0] * HORIZON)
    df.to_csv(path, sep=";", index=False)

    with pytest.raises(I.IngestionError, match="different actuals"):
        I.load_series(MODELS, "FAKE", 0, results_dir=fake_repo["results_dir"])


def test_ingestion_catches_the_wrong_tsf(fake_repo, tmp_path):
    """Same series count, different values — only the guardrail catches this."""
    lines = [
        "@relation FAKE", "@attribute series_name string",
        "@attribute start_timestamp date", "@frequency monthly", "@data",
    ]
    for i in range(N_SERIES):
        wrong = _series(i) + 500.0
        lines.append(f"S{i}:1990-01-01 00-00-00:" + ",".join(f"{v:.6f}" for v in wrong))
    (tmp_path / "wrong.tsf").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SeriesAlignmentError, match="do not match"):
        I.load_series(
            MODELS, "FAKE", 0,
            source_file="wrong.tsf", source_dir=str(tmp_path),
            results_dir=fake_repo["results_dir"],
        )


def test_frames_cache_gives_the_same_result(fake_repo):
    frames = I.load_dataset_frames(MODELS, "FAKE", fake_repo["results_dir"])
    a = I.load_series(MODELS, "FAKE", 1, results_dir=fake_repo["results_dir"])
    b = I.load_series(MODELS, "FAKE", 1, results_dir=fake_repo["results_dir"], frames=frames)
    assert a.state.y_preds == pytest.approx(b.state.y_preds)


def test_read_external_baselines(fake_repo):
    out = I.read_external_baselines("FAKE", 2, results_dir=fake_repo["results_dir"])
    assert out["mean"]["available"] is True
    assert out["mean"]["mape"] == pytest.approx(0.07)
    assert out["FFORMA"]["available"] is False  # not written by the fixture


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — pool evaluation and baseline seeding
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def state(fake_repo):
    return I.load_series(
        MODELS, "FAKE", dataset_index=0,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    ).state


def test_build_pool_full(state):
    r = P.build_pool(state, ReactConfig(pool_mode="full"))
    assert r["pool"] == FULL_POOL
    assert r["mode"] == "full"
    assert r["k"] == state.n_models


def test_build_pool_top_k(state):
    r = P.build_pool(state, ReactConfig(pool_mode="top_k_error", pool_k=2))
    assert len(state.get_pool(r["pool"])) == 2
    assert "bad" not in r["models"]


def test_build_pool_stable(state):
    r = P.build_pool(state, ReactConfig(pool_mode="top_k_stable", pool_k=2))
    assert len(state.get_pool(r["pool"])) == 2


def test_build_pool_rejects_unknown_mode(state):
    cfg = ReactConfig()
    cfg.pool_mode = "wishful_thinking"
    with pytest.raises(ValueError, match="unknown pool_mode"):
        P.build_pool(state, cfg)


def test_seed_baselines_fills_the_history(state):
    seeded = P.seed_baselines(state)
    assert [a.spec["combine"] for a in seeded] == list(P.SEED_BASELINES)
    assert all(a.origin == "baseline" for a in seeded)
    assert len(state.attempts) == len(P.SEED_BASELINES)
    assert all(a.rationale for a in seeded)


def test_seeded_baselines_bound_the_final_result(state):
    """Principle 5: the final pick can never be worse than the seeded baselines."""
    P.seed_baselines(state)
    baseline_best = state.best_attempt().score
    from orchestrator_react import tools as T

    T.evaluate_strategy(state, T.combine_best_single(state, "bad"), rationale="deliberately bad")
    assert state.best_attempt().score <= baseline_best


def test_pool_report_is_compact(state):
    r = P.pool_report(state, top_n=2)
    assert r["n_models"] == len(MODELS)
    assert len(r["error_table"]["top"]) == 2
    assert r["ranking_stability"]["verdict"] in {"stable", "moderate", "unstable", "unavailable"}
    assert len(r["best_model_per_window"]) == N_WINDOWS
    # no raw arrays may leak into the agent's view
    import json

    assert len(json.dumps(r)) < 4000


def test_calibration_gate_off_by_default(state):
    g = P.calibration_gate(state, ReactConfig())
    assert g["enabled"] is False and g["triggered"] is False


def test_calibration_gate_fires_on_a_stable_ranking(state):
    cfg = ReactConfig(calibration_gate=True, calibration_gate_kendall=0.0)
    assert P.calibration_gate(state, cfg)["triggered"] is True


def test_run_phase2_end_to_end(state):
    out = P.run_phase2(state, ReactConfig(pool_mode="full"))
    assert out["pool_composition_mode"] == "full"
    expected = len(P.SEED_BASELINES) + sum(
        1 for k, _ in P.SEED_STABLE_POOLS if k < state.n_models
    )
    assert out["attempts_seeded"] == expected
    assert out["best_baseline"] is not None
    assert "error_table" in out["report"]
    assert out["calibration_gate"]["enabled"] is False


def test_baseline_scores_for_the_csv(state):
    P.seed_baselines(state)
    scores = P.baseline_scores(state)
    assert set(scores) == set(P.SEED_BASELINES)
    assert all(v["rmse"] is not None for v in scores.values())


# ══════════════════════════════════════════════════════════════════════════════
# integration with the real data (skips when absent)
# ══════════════════════════════════════════════════════════════════════════════


REAL_RESULTS = "./timeseries/mestrado/resultados"
REAL_SOURCE = os.path.expanduser("~/Documents/mestrado/forecasting_datasets")
REAL_MODELS = ["ARIMA", "ETS", "THETA", "rf", "catboost", "NaiveSeasonal"]


def _real_data_available() -> bool:
    if not os.path.isdir(os.path.join(REAL_SOURCE)):
        return False
    if not os.path.exists(os.path.join(REAL_SOURCE, "mes_11_venda_mensal.tsf")):
        return False
    return all(
        os.path.exists(I.model_csv_path(m, "ANP_MONTHLY", REAL_RESULTS)) for m in REAL_MODELS
    )


real_data = pytest.mark.skipif(
    not _real_data_available(), reason="real ANP_MONTHLY data not available on this machine"
)


@real_data
def test_real_anp_ingestion_and_phase2():
    """End-to-end on real data: the check that it will run on the server."""
    source = load_series_source(
        "mes_11_venda_mensal.tsf",
        n_expected_series=I.count_series("ANP_MONTHLY", REAL_MODELS[0], REAL_RESULTS),
        source_dir=REAL_SOURCE,
    )
    assert source.n_series == 182
    assert source.filter_applied == "drop_zero_windows_24"

    for idx in (0, 7, 100, 181):
        ing = I.load_series(
            REAL_MODELS, "ANP_MONTHLY", idx,
            source=source, results_dir=REAL_RESULTS,
            config=ReactConfig(n_validation_windows=3),
        )
        assert ing.horizon == 12
        assert ing.state.n_windows == 3
        assert ing.alignment["verified"] is True
        assert ing.state.train_series.size == 419 - 12

        out = P.run_phase2(ing.state, ReactConfig(pool_mode="full"))
        # the three full-pool baselines plus one entry per stability pool whose k
        # is smaller than the pool itself
        expected = len(P.SEED_BASELINES) + sum(
            1 for k, _ in P.SEED_STABLE_POOLS if k < ing.state.n_models
        )
        assert out["attempts_seeded"] == expected
        assert out["best_baseline"] is not None

        from orchestrator_react import tools as T

        profile = T.series_profile(ing.state)
        assert profile["source"] == "train_series"
        assert profile["seasonal_period"] == 12
        assert profile["n_points"] == 407

        forecast, _ = ing.state.apply_to_test({"combine": "mean", "pool": FULL_POOL})
        assert forecast.shape == (12,)
        assert np.all(np.isfinite(forecast))


@real_data
def test_real_anp_external_baselines():
    out = I.read_external_baselines("ANP_MONTHLY", 0, results_dir=REAL_RESULTS)
    assert out["mean"]["available"] is True
    assert out["mean"]["rmse"] is not None


@real_data
def test_real_ett_source_is_the_reduced_file():
    """Documents the known ETT problem on this laptop, and proves it fails loudly."""
    n = I.count_series("ETTH1", "catboost", REAL_RESULTS)
    assert n == 7
    with pytest.raises(SeriesAlignmentError, match="no known filter reconciles"):
        load_series_source("ETTh1.tsf", n_expected_series=n, source_dir=REAL_SOURCE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
