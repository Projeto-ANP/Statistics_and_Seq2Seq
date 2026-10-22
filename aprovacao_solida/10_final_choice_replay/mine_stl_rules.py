"""Test STL-derived and trend anchors for the final selection.

The series card already contains STL-based trend/seasonality champions: the base
model whose STL component best matched the actual component on validation. This
script adds, per shortlisted candidate:

* distance to the trend champion's blind forecast;
* distance to the seasonality champion's blind forecast;
* distance to their average;
* distance to a linear trend extrapolation of the recent history;
* slope difference vs the recent linear slope.

Then it evaluates deterministic rules against argmin and the shortlist oracle,
on both the low and medium runs. The candidate features come from the existing
`shortlist_features.pkl`; only the new anchors are computed here.
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
from orchestrator_react.state import FULL_POOL  # noqa: E402

import mine_rules as MR  # noqa: E402
from replay_final_choice import (  # noqa: E402
    _artifact_indices,
    _attempts_from_artifact,
    _load_artifact,
    _load_dataset_state_builder,
)


def _champ(card: Dict[str, Any], key: str, state, ctx: Dict[str, Any]):
    info = card.get(key) or {}
    model = info.get("model")
    if model is None or model not in state.model_names:
        return None, None, None
    idx = state.model_names.index(model)
    forecast = np.asarray(state.test_preds[idx], dtype=float)
    scale = float(ctx["scale"]) or 1.0
    return model, info, forecast


def build(root: str, results_dir: str, source_dir: str) -> pd.DataFrame:
    base = pd.read_pickle(root)
    extra: Dict[tuple, Dict[str, Any]] = {}
    for (run, dataset), groups in base.groupby(["run", "dataset"]):
        indices = sorted(groups["series"].unique())
        try:
            load_state = _load_dataset_state_builder(dataset, results_dir, source_dir)
        except Exception as exc:
            print(f"[{run}/{dataset}] load failed: {exc}")
            continue
        print(f"[{run}/{dataset}] {len(indices)} series", flush=True)
        for index in indices:
            payload = _load_artifact(run, dataset, int(index), results_dir)
            if payload is None:
                continue
            attempts = _attempts_from_artifact(payload)
            if not attempts:
                continue
            forecast_by_id = {
                a.attempt_id: np.asarray(a.test_forecast, dtype=float) for a in attempts
            }
            try:
                ingested = load_state(int(index))
                card = payload.get("series_card") or {}
                ctx = FC.build_series_context(ingested.state, card)
            except Exception:
                continue
            state = ingested.state
            scale = float(ctx["scale"]) or 1.0
            last = float(ctx["last_actual"])
            horizon = int(ctx["horizon"])
            hist = np.asarray(state.train_series if state.train_series is not None else state.y_true.reshape(-1), dtype=float)
            hist = hist[np.isfinite(hist)]
            period = int(card.get("seasonal_period") or 0)
            look = min(len(hist), max(2 * period, horizon, 2))
            recent = hist[-look:]
            xs = np.arange(len(recent), dtype=float)
            slope = float(np.polyfit(xs, recent, 1)[0]) if len(recent) >= 2 else 0.0
            trend_anchor = last + slope * np.arange(1, horizon + 1)
            trend_model, trend_info, trend_fc = _champ(card, "trend_champion", state, ctx)
            season_model, season_info, season_fc = _champ(card, "seasonality_champion", state, ctx)
            for _, row in groups[groups["series"] == index].iterrows():
                fc = forecast_by_id.get(str(row["candidate_id"]))
                if fc is None:
                    continue
                def dist(ref):
                    if ref is None or ref.size != fc.size:
                        return np.nan
                    return float(np.nanmean(np.abs(fc - ref)) / scale)
                trend_dist = dist(trend_fc)
                season_dist = dist(season_fc)
                avg_dist = (
                    float(np.nanmean(np.abs(fc - (trend_fc + season_fc) / 2.0)) / scale)
                    if trend_fc is not None and season_fc is not None
                    and trend_fc.size == season_fc.size == fc.size
                    else np.nan
                )
                extra[(run, dataset, int(index), str(row["candidate_id"]))] = {
                    "stl_trend_dist": trend_dist,
                    "stl_season_dist": season_dist,
                    "stl_champ_avg_dist": avg_dist,
                    "trend_anchor_dist": dist(trend_anchor),
                    "slope_anchor_diff": abs(float(row.get("slope", np.nan)) - slope / scale)
                    if row.get("slope") is not None and np.isfinite(row.get("slope", np.nan))
                    else np.nan,
                    "trend_champ_model": trend_model,
                    "trend_champ_score": (trend_info or {}).get("score"),
                    "trend_champ_informative": (trend_info or {}).get("informative"),
                    "season_champ_model": season_model,
                    "season_champ_score": (season_info or {}).get("score"),
                    "season_champ_informative": (season_info or {}).get("informative"),
                    "trend_anchor_slope": slope / scale,
                }
    if not extra:
        return base
    add = pd.DataFrame(
        [{"run": k[0], "dataset": k[1], "series": k[2], "candidate_id": k[3], **v}
         for k, v in extra.items()]
    )
    out = base.merge(add, on=["run", "dataset", "series", "candidate_id"], how="left")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="outputs/resultados/v11_rule_mining/shortlist_features.pkl")
    parser.add_argument("--out", default="outputs/resultados/v11_rule_mining/shortlist_features_stl.pkl")
    parser.add_argument("--results-dir", default="./timeseries/mestrado/resultados")
    parser.add_argument("--source-dir", default="../forecasting_datasets")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    if args.rebuild or not os.path.exists(args.out):
        data = build(args.data, args.results_dir, args.source_dir)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        data.to_pickle(args.out)
    else:
        data = pd.read_pickle(args.out)
    print("rows", len(data), "series", data.groupby(["run", "dataset", "series"]).ngroups)

    evals: Dict[str, pd.DataFrame] = {}
    evals["leader"] = MR.eval_rule(data, lambda g: MR._leader(g))
    evals["seasonal_guard"] = MR.eval_rule(data, lambda g: (
        MR._min_feature(g, "seasonal_dist")
        if (MR._num(g["seasonal_strength"].iloc[0]) or 0) >= 0.5
        and (MR._num(g["cycles"].iloc[0]) or 0) >= 10
        else MR._leader(g)
    ))
    for col in ("stl_trend_dist", "stl_season_dist", "stl_champ_avg_dist",
                "trend_anchor_dist", "slope_anchor_diff"):
        evals[f"min_{col}"] = MR.eval_rule(data, lambda g, c=col: MR._min_feature(g, c))

    def _gate(g, key):
        s = MR._num(g["seasonal_strength"].iloc[0]) or 0.0
        c = MR._num(g["cycles"].iloc[0]) or 0.0
        return s >= 0.5 and c >= 10

    def trend_if_strong(g):
        t = MR._num(g["trend_strength"].iloc[0]) or 0.0
        return MR._min_feature(g, "trend_anchor_dist") if t >= 0.6 else MR._leader(g)

    def trend_or_seasonal(g):
        t = MR._num(g["trend_strength"].iloc[0]) or 0.0
        if t >= 0.6:
            return MR._min_feature(g, "trend_anchor_dist")
        if _gate(g, None):
            return MR._min_feature(g, "seasonal_dist")
        return MR._leader(g)

    def champ_by_strength(g):
        t = MR._num(g["trend_strength"].iloc[0]) or 0.0
        s = MR._num(g["seasonal_strength"].iloc[0]) or 0.0
        ti = g["trend_champ_informative"].iloc[0]
        si = g["season_champ_informative"].iloc[0]
        if t >= s and ti:
            return MR._min_feature(g, "stl_trend_dist")
        if si:
            return MR._min_feature(g, "stl_season_dist")
        return MR._leader(g)

    def champ_by_score(g):
        ts = MR._num(g["trend_champ_score"].iloc[0])
        ss = MR._num(g["season_champ_score"].iloc[0])
        ti = g["trend_champ_informative"].iloc[0]
        si = g["season_champ_informative"].iloc[0]
        if ts is not None and (ss is None or ts >= ss) and ti:
            return MR._min_feature(g, "stl_trend_dist")
        if ss is not None and si:
            return MR._min_feature(g, "stl_season_dist")
        return MR._leader(g)

    evals["trend_if_strong"] = MR.eval_rule(data, trend_if_strong)
    evals["trend_or_seasonal"] = MR.eval_rule(data, trend_or_seasonal)
    evals["champ_by_strength"] = MR.eval_rule(data, champ_by_strength)
    evals["champ_by_score"] = MR.eval_rule(data, champ_by_score)

    table = MR.summarize(evals)
    table = table.sort_values(["run", "ds_gain"], ascending=[True, False])
    for run in table["run"].unique():
        print("\n" + "=" * 100)
        print(run)
        print("=" * 100)
        print(table[table["run"] == run][
            ["rule", "ds_gain", "series_gain", "hit_rate", "switch_rate",
             "win_rate", "loss_rate", "tie_rate"]
        ].round(6).to_string(index=False))
    out = "outputs/resultados/v11_rule_mining/stl_rule_summary.csv"
    table.to_csv(out, sep=";", index=False)
    print(f"\nsummary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
