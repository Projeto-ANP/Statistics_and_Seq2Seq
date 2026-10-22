"""Deterministic seasonal baseline for the v11 shortlist (no LLM).

For each series, builds the same family shortlist as the final selector and
compares three policies on the test window:

* argmin (the v10 leader);
* closest to the seasonal naive, ungated;
* closest to the seasonal naive only when the seasonal cycle is estimable
  (seasonal_strength >= threshold and n_points >= 10 * period), else argmin.

Everything used to choose is available before the test: the forecast vectors,
the history and the series profile. The test SMAPE is used only for evaluation.

Runs on any v10x artifact folder, so the result can be replicated on both the
low and medium runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator_react import final_choice as FC  # noqa: E402

from replay_final_choice import (  # noqa: E402
    OFFICIAL_DATASETS,
    _artifact_indices,
    _attempts_from_artifact,
    _load_artifact,
    _load_dataset_state_builder,
)


def _seasonal_gate(series_card: Dict[str, Any], min_strength: float, min_cycles: float) -> bool:
    try:
        strength = float(series_card.get("seasonal_strength"))
        n_points = float(series_card.get("n_points"))
        period = float(series_card.get("seasonal_period") or 0)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(strength) or not np.isfinite(n_points) or period < 2:
        return False
    return bool(strength >= min_strength and n_points >= min_cycles * period)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="orchestrator_react_v10x_low")
    parser.add_argument("--datasets", nargs="+", default=list(OFFICIAL_DATASETS))
    parser.add_argument("--results-dir", default="./timeseries/mestrado/resultados")
    parser.add_argument("--source-dir", default="../forecasting_datasets")
    parser.add_argument("--min-strength", type=float, default=0.5)
    parser.add_argument("--min-cycles", type=float, default=10.0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--degenerate-factor", type=float, default=100.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        indices = _artifact_indices(args.run, dataset, args.results_dir)
        if not indices:
            print(f"[{dataset}] no artifacts")
            continue
        try:
            load_state = _load_dataset_state_builder(
                dataset, args.results_dir, args.source_dir
            )
        except Exception as exc:
            print(f"[{dataset}] data load failed: {exc}")
            continue
        print(f"[{dataset}] {len(indices)} series", flush=True)
        for index in indices:
            payload = _load_artifact(args.run, dataset, index, args.results_dir)
            if payload is None:
                continue
            attempts = _attempts_from_artifact(payload)
            if not attempts:
                continue
            series_card = payload.get("series_card") or {}
            try:
                ingested = load_state(index)
            except Exception as exc:
                print(f"  {dataset} {index}: load failed: {exc}")
                continue
            try:
                ctx = FC.build_series_context(ingested.state, series_card)
            except Exception as exc:
                print(f"  {dataset} {index}: context failed: {exc}")
                continue
            max_abs = float(args.degenerate_factor) * max(
                abs(float(ctx["last_actual"])), float(ctx["scale"]), 1.0
            )
            shortlist = FC.build_family_shortlist(
                attempts,
                lambda attempt: np.asarray(attempt.test_forecast, dtype=float),
                max_candidates=args.max_candidates,
                max_abs=max_abs,
            )
            if not shortlist:
                continue
            leader = min(attempts, key=lambda a: a.score)
            leader_candidate = next(
                (entry for entry in shortlist if entry.attempt.attempt_id == leader.attempt_id),
                shortlist[0],
            )
            argmin_smape = leader_candidate.attempt.test_smape
            if argmin_smape is None:
                argmin_smape = leader.test_smape
            seasonal_entries = []
            for entry in shortlist:
                features = FC._candidate_features(entry.forecast, ctx)
                value = features.get("seasonal_dist")
                if value is not None and np.isfinite(float(value)):
                    seasonal_entries.append((float(value), entry))
            if not seasonal_entries:
                continue
            seasonal_entries.sort(key=lambda item: item[0])
            best_seasonal_value, best_seasonal_entry = seasonal_entries[0]
            leader_seasonal = next(
                (
                    FC._candidate_features(entry.forecast, ctx).get("seasonal_dist")
                    for entry in shortlist
                    if entry.attempt.attempt_id == leader_candidate.attempt.attempt_id
                ),
                None,
            )
            gate = _seasonal_gate(series_card, args.min_strength, args.min_cycles)
            gated_entry = best_seasonal_entry if gate else leader_candidate
            rows.append(
                {
                    "run": args.run,
                    "dataset": dataset,
                    "series": int(index),
                    "argmin_id": leader_candidate.attempt.attempt_id,
                    "seasonal_id": best_seasonal_entry.attempt.attempt_id,
                    "argmin_smape": argmin_smape,
                    "seasonal_smape": best_seasonal_entry.attempt.test_smape,
                    "gated_smape": gated_entry.attempt.test_smape,
                    "gate": bool(gate),
                    "seasonal_strength": series_card.get("seasonal_strength"),
                    "n_points": series_card.get("n_points"),
                    "period": series_card.get("seasonal_period"),
                    "seasonal_gap": (
                        float(leader_seasonal) - best_seasonal_value
                        if leader_seasonal is not None
                        and np.isfinite(float(leader_seasonal))
                        else None
                    ),
                }
            )

    D = pd.DataFrame(rows)
    if D.empty:
        print("no usable rows")
        return 1
    for col in ("argmin_smape", "seasonal_smape", "gated_smape", "seasonal_gap"):
        D[col] = pd.to_numeric(D[col], errors="coerce")

    print("\n" + "=" * 78)
    print(f"SEASONAL BASELINE — {args.run}")
    print("=" * 78)
    for label, col in (
        ("argmin", "argmin_smape"),
        ("closest_seasonal", "seasonal_smape"),
        ("gated_seasonal", "gated_smape"),
    ):
        print(f"\n{label}")
        per = []
        for dataset, group in D.groupby("dataset"):
            values = group[col].dropna()
            if values.empty:
                continue
            per.append((dataset, len(values), float(values.mean())))
        for dataset, n, value in per:
            print(f"  {dataset:22s} n={n:3d} smape={value:.6f}")
        if per:
            print(f"  {'MEAN_DATASETS':22s} smape={np.mean([p[2] for p in per]):.6f}")
        print(f"  {'SERIES_MEAN':22s} smape={D[col].mean():.6f}")

    base = D["argmin_smape"]
    print("\n" + "=" * 78)
    print("GAIN VS ARGMIN")
    print("=" * 78)
    for label, col in (("closest_seasonal", "seasonal_smape"), ("gated_seasonal", "gated_smape")):
        gain = base - D[col]
        per_ds = gain.groupby(D["dataset"]).mean()
        print(
            f"{label:20s} dataset_mean={per_ds.mean():+.6f} series_mean={gain.mean():+.6f} "
            f"win={np.mean(gain > 1e-12):.3f} loss={np.mean(gain < -1e-12):.3f}"
        )
        print("  per dataset:", {k: round(float(v), 5) for k, v in per_ds.items()})
        print(
            "  gate true:",
            int(D["gate"].sum()),
            "gated switch rate:",
            round(float((D["gate"] & (D["seasonal_id"] != D["argmin_id"])).mean()), 3),
        )

    output = args.output or os.path.join(
        "outputs/resultados/v11_final_choice_replay",
        f"seasonal_baseline_{args.run}.csv",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    D.to_csv(output, sep=";", index=False)
    print(f"\nper-series: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
