"""Mine deterministic selection rules from the features shown to the agent.

Builds the same family shortlist used by v11, one row per candidate with the
blind features, the series profile and the test outcome. Then evaluates simple
rules per series: min/max features, conditional anchors, margins, filters and
"closest to the leader" switches.

The test SMAPE is used only to evaluate a rule, never to construct it. The low
and medium runs are reported side by side as a replication check (same series
and test windows, different agent trajectories).

Usage:
    python3 .../mine_rules.py                # builds and evaluates
    python3 .../mine_rules.py --rebuild      # force rebuilding the dataset
    python3 .../mine_rules.py --run low      # one run only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


DATASET_PATH = "outputs/resultados/v11_rule_mining/shortlist_features.pkl"


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def build_dataset(runs: Sequence[str], results_dir: str, source_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for dataset in OFFICIAL_DATASETS:
            indices = _artifact_indices(run, dataset, results_dir)
            if not indices:
                continue
            try:
                load_state = _load_dataset_state_builder(dataset, results_dir, source_dir)
            except Exception as exc:
                print(f"[{run}/{dataset}] data load failed: {exc}")
                continue
            print(f"[{run}/{dataset}] {len(indices)} series", flush=True)
            for index in indices:
                payload = _load_artifact(run, dataset, index, results_dir)
                if payload is None:
                    continue
                attempts = _attempts_from_artifact(payload)
                if not attempts:
                    continue
                card = payload.get("series_card") or {}
                try:
                    ingested = load_state(index)
                    ctx = FC.build_series_context(ingested.state, card)
                except Exception:
                    continue
                max_abs = 100.0 * max(
                    abs(float(ctx["last_actual"])), float(ctx["scale"]), 1.0
                )
                shortlist = FC.build_family_shortlist(
                    attempts,
                    lambda attempt: np.asarray(attempt.test_forecast, dtype=float),
                    max_candidates=5,
                    max_abs=max_abs,
                )
                if len(shortlist) < 2:
                    continue
                leader = min(shortlist, key=lambda e: float(e.attempt.score))
                leader_smape = leader.attempt.test_smape
                if leader_smape is None:
                    continue
                oracle = min(
                    float(e.attempt.test_smape)
                    for e in shortlist
                    if e.attempt.test_smape is not None
                )
                ordered = sorted(shortlist, key=lambda e: float(e.attempt.score))
                validation_rank = {
                    id(e.attempt): i + 1 for i, e in enumerate(ordered)
                }
                test_ordered = sorted(
                    shortlist,
                    key=lambda e: float(e.attempt.test_smape)
                    if e.attempt.test_smape is not None
                    else np.inf,
                )
                test_rank = {id(e.attempt): i + 1 for i, e in enumerate(test_ordered)}
                leader_forecast = np.asarray(leader.forecast, dtype=float)
                scale = max(float(ctx["scale"]), 1e-9)
                trend_change = _num(card.get("trend_relative_change"))
                for entry in shortlist:
                    features = FC._candidate_features(entry.forecast, ctx)
                    forecast = np.asarray(entry.forecast, dtype=float)
                    distance = float(
                        np.nanmean(np.abs(forecast - leader_forecast)) / scale
                    )
                    slope = features.get("slope")
                    matches = None
                    if slope is not None and trend_change is not None and np.isfinite(trend_change):
                        if abs(trend_change) > 1e-12:
                            matches = float(np.sign(slope) == np.sign(trend_change))
                    n_points = _num(card.get("n_points"))
                    period = _num(card.get("seasonal_period"))
                    cycles = (
                        n_points / period
                        if n_points is not None and period is not None and period > 0
                        else None
                    )
                    rows.append(
                        {
                            "run": run,
                            "dataset": dataset,
                            "series": int(index),
                            "candidate_id": entry.attempt.attempt_id,
                            "combine": entry.attempt.spec.get("combine"),
                            "validation_score": float(entry.attempt.score),
                            "validation_rank": validation_rank.get(id(entry.attempt)),
                            "test_smape": (
                                float(entry.attempt.test_smape)
                                if entry.attempt.test_smape is not None
                                else None
                            ),
                            "test_rank": test_rank.get(id(entry.attempt)),
                            "is_best": bool(
                                entry.attempt.test_smape is not None
                                and float(entry.attempt.test_smape) <= oracle + 1e-12
                            ),
                            "is_leader": bool(entry is leader),
                            "leader_id": leader.attempt.attempt_id,
                            "leader_smape": float(leader_smape),
                            "shortlist_oracle": float(oracle),
                            "distance_to_leader": distance,
                            "slope_matches_trend": matches,
                            **features,
                            "seasonal_strength": _num(card.get("seasonal_strength")),
                            "trend_strength": _num(card.get("trend_strength")),
                            "acf1": _num((card.get("features") or {}).get("acf1")),
                            "acf_seasonal": _num(
                                (card.get("features") or {}).get("acf_seasonal")
                            ),
                            "spectral_entropy": _num(
                                (card.get("features") or {}).get("spectral_entropy")
                            ),
                            "hurst": _num((card.get("features") or {}).get("hurst")),
                            "coef_variation": _num(
                                (card.get("features") or {}).get("coef_variation")
                            ),
                            "outliers_pct": _num((card.get("outliers") or {}).get("pct")),
                            "crosses_zero": (card.get("features") or {}).get("crosses_zero"),
                            "n_points": n_points,
                            "period": period,
                            "cycles": cycles,
                            "horizon": _num(card.get("horizon")),
                            "frequency": card.get("frequency"),
                        }
                    )
    return pd.DataFrame(rows)


def _leader(group: pd.DataFrame) -> pd.Series:
    return group[group["is_leader"].astype(bool)].iloc[0]


def _min_feature(group: pd.DataFrame, col: str) -> Optional[pd.Series]:
    usable = group[group[col].notna()]
    if usable.empty:
        return None
    return usable.loc[usable[col].idxmin()]


def _max_feature(group: pd.DataFrame, col: str) -> Optional[pd.Series]:
    usable = group[group[col].notna()]
    if usable.empty:
        return None
    return usable.loc[usable[col].idxmax()]


def _nonleader(group: pd.DataFrame) -> pd.DataFrame:
    return group[~group["is_leader"].astype(bool)]


def eval_rule(
    data: pd.DataFrame, rule: Callable[[pd.DataFrame], Optional[pd.Series]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (run, dataset, series), group in data.groupby(["run", "dataset", "series"]):
        leader = _leader(group)
        chosen = rule(group)
        if chosen is None:
            chosen = leader
        rows.append(
            {
                "run": run,
                "dataset": dataset,
                "series": int(series),
                "chosen_id": chosen.get("candidate_id"),
                "leader_id": leader.get("candidate_id"),
                "leader_smape": float(leader["leader_smape"]),
                "chosen_smape": (
                    float(chosen["test_smape"])
                    if chosen.get("test_smape") is not None
                    else np.nan
                ),
                "shortlist_oracle": float(leader["shortlist_oracle"]),
                "switch": bool(chosen.get("candidate_id") != leader.get("candidate_id")),
            }
        )
    out = pd.DataFrame(rows)
    out["gain"] = out["leader_smape"] - out["chosen_smape"]
    out["hit"] = out["chosen_smape"] <= out["shortlist_oracle"] + 1e-12
    return out


def summarize(evals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, ev in evals.items():
        for run, group in ev.groupby("run"):
            per_ds = []
            for dataset, dsg in group.groupby("dataset"):
                per_ds.append(
                    (
                        float(dsg["gain"].mean()),
                        float(dsg["hit"].mean()),
                        float(dsg["switch"].mean()),
                        float((dsg["gain"] > 1e-12).mean()),
                        float((dsg["gain"] < -1e-12).mean()),
                    )
                )
            if not per_ds:
                continue
            arr = np.asarray(per_ds)
            rows.append(
                {
                    "rule": name,
                    "run": run,
                    "ds_gain": float(arr[:, 0].mean()),
                    "series_gain": float(group["gain"].mean()),
                    "hit_rate": float(group["hit"].mean()),
                    "switch_rate": float(group["switch"].mean()),
                    "win_rate": float((group["gain"] > 1e-12).mean()),
                    "loss_rate": float((group["gain"] < -1e-12).mean()),
                    "tie_rate": float((group["gain"].abs() <= 1e-12).mean()),
                }
            )
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["orchestrator_react_v10x_low", "orchestrator_react_v10x_medium"])
    parser.add_argument("--results-dir", default="./timeseries/mestrado/resultados")
    parser.add_argument("--source-dir", default="../forecasting_datasets")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--data", default=DATASET_PATH)
    args = parser.parse_args(argv)

    if args.rebuild or not os.path.exists(args.data):
        data = build_dataset(args.runs, args.results_dir, args.source_dir)
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        data.to_pickle(args.data)
    else:
        data = pd.read_pickle(args.data)
    if data.empty:
        print("empty dataset")
        return 1
    print("candidates", len(data), "series", data.groupby(["run", "dataset", "series"]).ngroups)

    evals: Dict[str, pd.DataFrame] = {}
    evals["leader"] = eval_rule(data, lambda g: _leader(g))

    for col in (
        "level_1", "slope", "range", "volatility", "naive_dist",
        "seasonal_dist", "pool_dist", "outside_band", "validation_score",
    ):
        evals[f"min_{col}"] = eval_rule(data, lambda g, c=col: _min_feature(g, c))
        evals[f"max_{col}"] = eval_rule(data, lambda g, c=col: _max_feature(g, c))

    evals["nonleader_closest_to_leader"] = eval_rule(
        data,
        lambda g: (
            _nonleader(g).loc[_nonleader(g)["distance_to_leader"].idxmin()]
            if not _nonleader(g).empty
            else _leader(g)
        ),
    )
    evals["nonleader_farthest_from_leader"] = eval_rule(
        data,
        lambda g: (
            _nonleader(g).loc[_nonleader(g)["distance_to_leader"].idxmax()]
            if not _nonleader(g).empty
            else _leader(g)
        ),
    )

    def seasonal_guard(g: pd.DataFrame) -> Optional[pd.Series]:
        ss = _num(g["seasonal_strength"].iloc[0])
        cycles = _num(g["cycles"].iloc[0])
        if ss is None or cycles is None or ss < 0.5 or cycles < 10:
            return _leader(g)
        return _min_feature(g, "seasonal_dist")

    def two_anchor(g: pd.DataFrame) -> Optional[pd.Series]:
        ss = _num(g["seasonal_strength"].iloc[0])
        cycles = _num(g["cycles"].iloc[0])
        if ss is None or cycles is None:
            return _leader(g)
        if ss >= 0.5 and cycles >= 10:
            return _min_feature(g, "seasonal_dist")
        if ss < 0.5 and cycles >= 10:
            return _min_feature(g, "naive_dist")
        return _leader(g)

    def two_anchor_all_seasonal(g: pd.DataFrame) -> Optional[pd.Series]:
        ss = _num(g["seasonal_strength"].iloc[0])
        if ss is None:
            return _leader(g)
        if ss >= 0.5:
            return _min_feature(g, "seasonal_dist")
        return _min_feature(g, "naive_dist")

    evals["seasonal_guard"] = eval_rule(data, seasonal_guard)
    evals["two_anchor"] = eval_rule(data, two_anchor)
    evals["two_anchor_all_seasonal"] = eval_rule(data, two_anchor_all_seasonal)

    def _gate(g: pd.DataFrame) -> bool:
        ss = _num(g["seasonal_strength"].iloc[0])
        cycles = _num(g["cycles"].iloc[0])
        return bool(ss is not None and cycles is not None and ss >= 0.5 and cycles >= 10)

    def guard_or_outside(g: pd.DataFrame) -> Optional[pd.Series]:
        if _gate(g):
            return _min_feature(g, "seasonal_dist")
        return _max_feature(g, "outside_band")

    evals["guard_or_max_outside"] = eval_rule(data, guard_or_outside)

    for margin in (0.0, 0.05, 0.1, 0.2, 0.3):
        def guard_or_outside_margin(g: pd.DataFrame, m: float = margin) -> Optional[pd.Series]:
            if _gate(g):
                return _min_feature(g, "seasonal_dist")
            leader = _leader(g)
            best = _max_feature(g, "outside_band")
            if best is None:
                return leader
            lead_value = _num(leader.get("outside_band"))
            best_value = _num(best.get("outside_band"))
            if lead_value is None or best_value is None:
                return leader
            return best if best_value - lead_value > m else leader

        evals[f"guard_or_outside_margin_{margin}"] = eval_rule(data, guard_or_outside_margin)

    def guard_or_outside_leader_inside(g: pd.DataFrame) -> Optional[pd.Series]:
        if _gate(g):
            return _min_feature(g, "seasonal_dist")
        leader = _leader(g)
        lead_value = _num(leader.get("outside_band"))
        best = _max_feature(g, "outside_band")
        if best is None or lead_value is None:
            return leader
        best_value = _num(best.get("outside_band"))
        return best if (lead_value <= 0 and best_value is not None and best_value > 0) else leader

    evals["guard_or_outside_leader_inside"] = eval_rule(data, guard_or_outside_leader_inside)

    def guard_or_outside_low_seasonal(g: pd.DataFrame) -> Optional[pd.Series]:
        ss = _num(g["seasonal_strength"].iloc[0])
        cycles = _num(g["cycles"].iloc[0])
        if ss is None:
            return _leader(g)
        if ss >= 0.5 and cycles is not None and cycles >= 10:
            return _min_feature(g, "seasonal_dist")
        if ss < 0.5 and cycles is not None and cycles >= 10:
            return _max_feature(g, "outside_band")
        return _leader(g)

    def guard_or_outside_low_seasonal_any_cycles(g: pd.DataFrame) -> Optional[pd.Series]:
        ss = _num(g["seasonal_strength"].iloc[0])
        cycles = _num(g["cycles"].iloc[0])
        if ss is None:
            return _leader(g)
        if ss >= 0.5 and cycles is not None and cycles >= 10:
            return _min_feature(g, "seasonal_dist")
        if ss < 0.5:
            return _max_feature(g, "outside_band")
        return _leader(g)

    evals["guard_or_outside_low_seasonal"] = eval_rule(data, guard_or_outside_low_seasonal)
    evals["guard_or_outside_low_seasonal_any_cycles"] = eval_rule(
        data, guard_or_outside_low_seasonal_any_cycles
    )

    for margin in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2):
        def seasonal_margin(g: pd.DataFrame, m: float = margin) -> Optional[pd.Series]:
            leader = _leader(g)
            best = _min_feature(g, "seasonal_dist")
            if best is None:
                return leader
            lead_value = _num(leader.get("seasonal_dist"))
            best_value = _num(best.get("seasonal_dist"))
            if lead_value is None or best_value is None:
                return leader
            return best if lead_value - best_value > m else leader

        evals[f"seasonal_margin_{margin}"] = eval_rule(data, seasonal_margin)

    for margin in (0.01, 0.02, 0.05, 0.1):
        def naive_margin(g: pd.DataFrame, m: float = margin) -> Optional[pd.Series]:
            leader = _leader(g)
            best = _min_feature(g, "naive_dist")
            if best is None:
                return leader
            lead_value = _num(leader.get("naive_dist"))
            best_value = _num(best.get("naive_dist"))
            if lead_value is None or best_value is None:
                return leader
            return best if lead_value - best_value > m else leader

        evals[f"naive_margin_{margin}"] = eval_rule(data, naive_margin)

    def leader_unless_outside(g: pd.DataFrame) -> Optional[pd.Series]:
        leader = _leader(g)
        lead_out = _num(leader.get("outside_band"))
        if lead_out is None or lead_out <= 0:
            return leader
        inside = g[(g["outside_band"].fillna(1.0) <= 0)]
        if inside.empty:
            return leader
        return inside.loc[inside["validation_score"].idxmin()]

    def best_validation_inside_band(g: pd.DataFrame) -> Optional[pd.Series]:
        inside = g[g["outside_band"].fillna(1.0) <= 0]
        if inside.empty:
            return _leader(g)
        return inside.loc[inside["validation_score"].idxmin()]

    def best_validation_lower_seasonal(g: pd.DataFrame) -> Optional[pd.Series]:
        leader = _leader(g)
        lead = _num(leader.get("seasonal_dist"))
        if lead is None:
            return leader
        ok = g[g["seasonal_dist"].notna() & (g["seasonal_dist"] <= lead + 1e-12)]
        if ok.empty:
            return leader
        return ok.loc[ok["validation_score"].idxmin()]

    def best_validation_lower_naive(g: pd.DataFrame) -> Optional[pd.Series]:
        leader = _leader(g)
        lead = _num(leader.get("naive_dist"))
        if lead is None:
            return leader
        ok = g[g["naive_dist"].notna() & (g["naive_dist"] <= lead + 1e-12)]
        if ok.empty:
            return leader
        return ok.loc[ok["validation_score"].idxmin()]

    def best_validation_lower_vol(g: pd.DataFrame) -> Optional[pd.Series]:
        leader = _leader(g)
        lead = _num(leader.get("volatility"))
        if lead is None:
            return leader
        ok = g[g["volatility"].notna() & (g["volatility"] <= lead + 1e-12)]
        if ok.empty:
            return leader
        return ok.loc[ok["validation_score"].idxmin()]

    def trend_match_best(g: pd.DataFrame) -> Optional[pd.Series]:
        leader = _leader(g)
        ok = g[g["slope_matches_trend"].fillna(0.0) >= 1.0]
        if ok.empty:
            return leader
        return ok.loc[ok["validation_score"].idxmin()]

    evals["leader_unless_outside"] = eval_rule(data, leader_unless_outside)
    evals["best_validation_inside_band"] = eval_rule(data, best_validation_inside_band)
    evals["best_validation_lower_seasonal"] = eval_rule(data, best_validation_lower_seasonal)
    evals["best_validation_lower_naive"] = eval_rule(data, best_validation_lower_naive)
    evals["best_validation_lower_vol"] = eval_rule(data, best_validation_lower_vol)
    evals["trend_match_best"] = eval_rule(data, trend_match_best)

    table = summarize(evals)
    table["abs_series_gain"] = table["series_gain"].abs()
    table = table.sort_values(["run", "ds_gain"], ascending=[True, False])
    pd.set_option("display.max_rows", 200)
    for run in table["run"].unique():
        print("\n" + "=" * 100)
        print(run)
        print("=" * 100)
        print(
            table[table["run"] == run][
                ["rule", "ds_gain", "series_gain", "hit_rate", "switch_rate",
                 "win_rate", "loss_rate", "tie_rate"]
            ]
            .head(30)
            .round(6)
            .to_string(index=False)
        )

    out_path = "outputs/resultados/v11_rule_mining/rule_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table.to_csv(out_path, sep=";", index=False)
    print(f"\nrule summary: {out_path}")
    print(f"dataset: {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
