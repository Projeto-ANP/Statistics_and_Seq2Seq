"""Deep analysis of a v11 replay: LLM vs deterministic selection rules.

Reads a replay output and the v10x artifacts, loads the test actuals, and
compares:

* the blind LLM choice;
* argmin (the v10 leader);
* simple deterministic rules over the same shortlist
  (closest to pool median, closest to last, closest to seasonal naive,
  smallest outside-band fraction, median validation score);
* per-horizon median/mean of the shortlisted candidate forecasts.

It also slices the LLM gain by candidate disagreement and by the model's
self-reported confidence. No LLM calls.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator_react import ingest as ing  # noqa: E402
from orchestrator_react import metrics as M  # noqa: E402
from orchestrator_react.config import ReactConfig  # noqa: E402
from orchestrator_react.data_source import load_series_source  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402
from run_tsf_batch import DATASET_SOURCES, DEFAULT_SOURCE_DIR  # noqa: E402

from analyze_replay import _read_inputs  # noqa: E402


def _artifact_data(
    run: str, datasets: Sequence[str], results_dir: str
) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """(dataset, series, attempt_id) -> {test_smape, test_forecast, spec}."""
    out: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for dataset in datasets:
        pattern = os.path.join(results_dir, run, "llm_artifacts", dataset, "dataset_*.json")
        for path in glob.glob(pattern):
            match = re.search(r"dataset_(\d+)\.json$", path)
            if not match:
                continue
            series = int(match.group(1))
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            for attempt in (payload.get("ranking_trace") or {}).get("attempts") or []:
                attempt_id = str(attempt.get("id") or "")
                if not attempt_id:
                    continue
                forecast = attempt.get("test_forecast")
                value = attempt.get("test_smape")
                out[(dataset, series, attempt_id)] = {
                    "test_smape": (
                        float(value)
                        if value is not None and np.isfinite(float(value))
                        else None
                    ),
                    "test_forecast": (
                        np.asarray(forecast, dtype=float)
                        if isinstance(forecast, (list, tuple)) and forecast
                        else None
                    ),
                    "spec": dict(attempt.get("spec") or {}),
                }
    return out


def _state_loader(dataset: str, results_dir: str, source_dir: str):
    models = list(DEFAULT_MODELS)
    frames = ing.load_dataset_frames(models, dataset, results_dir=results_dir)
    try:
        bad = ing.find_misaligned_models(
            models, dataset, 0, results_dir=results_dir, frames=frames, n_windows=3
        )
    except Exception:
        bad = {}
    drop = sorted(bad)
    source_file = DATASET_SOURCES.get(dataset)
    if not source_file:
        raise KeyError(f"no .tsf source for {dataset}")
    source = load_series_source(
        source_file,
        n_expected_series=ing.count_series(dataset, models[0], results_dir=results_dir),
        source_dir=source_dir,
    )
    cfg = ReactConfig()

    def load(index: int):
        return ing.load_series(
            models,
            dataset,
            int(index),
            source=source,
            config=cfg,
            results_dir=results_dir,
            frames=frames,
            drop_models=drop,
        )

    return load


def _pick_min(specs: Sequence[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    usable = [
        s
        for s in specs
        if s.get("features", {}).get(key) is not None
        and np.isfinite(float(s["features"][key]))
    ]
    if not usable:
        return None
    return min(usable, key=lambda s: float(s["features"][key]))


def _test_smape(actual: np.ndarray, forecast: Optional[np.ndarray]) -> Optional[float]:
    if forecast is None or forecast.size != actual.size:
        return None
    value = float(M.smape(actual, forecast))
    return value if np.isfinite(value) else None


def _pairwise_stats(vectors: Sequence[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    vectors = [v for v in vectors if v is not None and v.size]
    if len(vectors) < 2:
        return None, None
    correlations = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            if a.size != b.size:
                continue
            if np.std(a) <= 0 or np.std(b) <= 0:
                continue
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    stacked = np.vstack(vectors)
    dispersion = float(np.nanmean(np.nanstd(stacked, axis=0)) / (
        np.nanmean(np.abs(stacked)) + 1e-9
    ))
    return (float(np.mean(correlations)) if correlations else None), dispersion


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/resultados/v11_final_choice_replay/orchestrator_react_v10x_low_gpt-oss-20b",
    )
    parser.add_argument("--run", default="orchestrator_react_v10x_low")
    parser.add_argument("--results-dir", default="./timeseries/mestrado/resultados")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    args = parser.parse_args(argv)

    results, _, mapping = _read_inputs(args.output_dir)
    if results.empty or mapping.empty:
        print("results.csv/mapping.jsonl missing or empty")
        return 1
    datasets = sorted(mapping["dataset"].unique())
    artifacts = _artifact_data(args.run, datasets, args.results_dir)
    loaders = {ds: _state_loader(ds, args.results_dir, args.source_dir) for ds in datasets}

    rows: List[Dict[str, Any]] = []
    for _, row in mapping.iterrows():
        dataset = str(row["dataset"])
        index = int(row["series"])
        try:
            specs = json.loads(row.get("shortlist_specs") or "[]")
        except json.JSONDecodeError:
            continue
        if not specs:
            continue
        try:
            ingested = loaders[dataset](index)
        except Exception as exc:
            print(f"skip {dataset} {index}: {exc}")
            continue
        actual = np.asarray(ingested.test_values, dtype=float)

        data = {}
        for spec in specs:
            key = (dataset, index, str(spec.get("id")))
            data[str(spec.get("id"))] = artifacts.get(key, {})

        def candidate_smape(spec: Optional[Dict[str, Any]]) -> Optional[float]:
            if not spec:
                return None
            entry = data.get(str(spec.get("id")), {})
            value = entry.get("test_smape")
            if value is not None and np.isfinite(float(value)):
                return float(value)
            return _test_smape(actual, entry.get("test_forecast"))

        leader = min(specs, key=lambda s: float(s.get("validation_score", np.inf)))
        leader_smape = candidate_smape(leader)
        argmin_smape = row.get("argmin_test_smape")
        try:
            argmin_smape = float(argmin_smape) if argmin_smape is not None else leader_smape
        except (TypeError, ValueError):
            argmin_smape = leader_smape
        chosen_smape = row.get("chosen_test_smape")
        try:
            chosen_smape = float(chosen_smape) if chosen_smape is not None else None
        except (TypeError, ValueError):
            chosen_smape = None

        values = [candidate_smape(s) for s in specs]
        usable_values = [v for v in values if v is not None and np.isfinite(v)]
        shortlist_oracle = min(usable_values) if usable_values else None
        vectors = [
            artifacts.get((dataset, index, str(s.get("id"))), {}).get("test_forecast")
            for s in specs
        ]
        mean_corr, dispersion = _pairwise_stats(
            [v for v in vectors if v is not None]
        )

        rules: Dict[str, Optional[float]] = {
            "argmin": argmin_smape,
            "llm": chosen_smape,
            "closest_pool_median": candidate_smape(_pick_min(specs, "pool_dist")),
            "closest_last": candidate_smape(_pick_min(specs, "naive_dist")),
            "closest_seasonal": candidate_smape(_pick_min(specs, "seasonal_dist")),
            "smallest_outside_band": candidate_smape(_pick_min(specs, "outside_band")),
            "median_validation": candidate_smape(
                sorted(specs, key=lambda s: float(s.get("validation_score", np.inf)))[
                    len(specs) // 2
                ]
            ),
        }
        forecast_vectors = [
            artifacts.get((dataset, index, str(s.get("id"))), {}).get("test_forecast")
            for s in specs
        ]
        forecast_vectors = [v for v in forecast_vectors if v is not None]
        if forecast_vectors:
            stacked = np.vstack(forecast_vectors)
            rules["shortlist_mean_forecast"] = _test_smape(
                actual, np.nanmean(stacked, axis=0)
            )
            rules["shortlist_median_forecast"] = _test_smape(
                actual, np.nanmedian(stacked, axis=0)
            )
        rows.append(
            {
                "dataset": dataset,
                "series": index,
                "decision": str(row.get("decision") or ""),
                "confidence": str(row.get("confidence") or ""),
                "shortlist_size": len(specs),
                "shortlist_oracle": shortlist_oracle,
                "mean_corr": mean_corr,
                "dispersion": dispersion,
                **rules,
            }
        )

    D = pd.DataFrame(rows)
    if D.empty:
        print("no usable rows")
        return 1

    print("=" * 78)
    print("RULES (dataset mean; lower is better)")
    print("=" * 78)
    rule_cols = [
        "argmin", "llm", "closest_pool_median", "closest_last",
        "closest_seasonal", "smallest_outside_band", "median_validation",
        "shortlist_mean_forecast", "shortlist_median_forecast", "shortlist_oracle",
    ]
    table = []
    for dataset, group in D.groupby("dataset"):
        row = {"dataset": dataset, "n": len(group)}
        for col in rule_cols:
            row[col] = pd.to_numeric(group[col], errors="coerce").mean()
        table.append(row)
    table = pd.DataFrame(table)
    overall = {"dataset": "MEAN_DATASETS", "n": int(table["n"].sum())}
    for col in rule_cols:
        overall[col] = table[col].mean()
    table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)
    print(table.round(6).to_string(index=False))

    print("\n" + "=" * 78)
    print("GAIN VS ARGMIN (positive = better)")
    print("=" * 78)
    gain_rows = []
    for dataset, group in D.groupby("dataset"):
        base = pd.to_numeric(group["argmin"], errors="coerce")
        row = {"dataset": dataset, "n": len(group)}
        for col in rule_cols:
            if col == "argmin":
                continue
            vals = pd.to_numeric(group[col], errors="coerce")
            row[col] = float((base - vals).mean())
        gain_rows.append(row)
    gains = pd.DataFrame(gain_rows)
    overall_gain = {"dataset": "MEAN_DATASETS", "n": int(gains["n"].sum())}
    for col in gains.columns:
        if col in ("dataset", "n"):
            continue
        overall_gain[col] = float(gains[col].mean())
    gains = pd.concat([gains, pd.DataFrame([overall_gain])], ignore_index=True)
    print(gains.round(6).to_string(index=False))

    print("\n" + "=" * 78)
    print("LLM BY DISAGREEMENT (test-forecast pairwise correlation)")
    print("=" * 78)
    usable = D[D["mean_corr"].notna() & D["llm"].notna() & D["argmin"].notna()].copy()
    if not usable.empty:
        usable["gain_llm"] = usable["argmin"] - usable["llm"]
        usable["q"] = pd.qcut(usable["mean_corr"], 4, labels=False, duplicates="drop")
        q = usable.groupby("q").agg(
            n=("series", "size"),
            mean_corr=("mean_corr", "mean"),
            llm_gain=("gain_llm", "mean"),
            llm_win=("gain_llm", lambda s: float((s > 1e-12).mean())),
            override=("decision", lambda s: float((s.astype(str).str.lower() == "override").mean())),
        )
        # Add deterministic-rule gains per quartile.
        for rule in ("closest_pool_median", "shortlist_median_forecast"):
            q[rule + "_gain"] = usable.groupby("q").apply(
                lambda g: float((g["argmin"] - g[rule]).mean()), include_groups=False
            )
        print(q.round(6).to_string())

    print("\n" + "=" * 78)
    print("LLM BY CONFIDENCE")
    print("=" * 78)
    conf = D[D["llm"].notna() & D["argmin"].notna()].copy()
    if not conf.empty:
        conf["gain"] = conf["argmin"] - conf["llm"]
        print(
            conf.groupby("confidence")
            .agg(
                n=("series", "size"),
                llm_gain=("gain", "mean"),
                win_rate=("gain", lambda s: float((s > 1e-12).mean())),
                override=("decision", lambda s: float((s.astype(str).str.lower() == "override").mean())),
            )
            .round(6)
            .to_string()
        )

    out_path = os.path.join(args.output_dir, "deep_analysis.csv")
    D.to_csv(out_path, sep=";", index=False)
    print(f"\nper-series deep analysis: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
