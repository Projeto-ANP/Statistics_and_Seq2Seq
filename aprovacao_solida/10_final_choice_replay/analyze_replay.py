"""Reads a v11 offline replay and prints the diagnostics that matter.

No LLM and no data loading: it joins `results.csv`/`mapping.jsonl` with the
already-written v10x artifacts to answer:

* did the blind selector beat the v10 argmin on the test window?
* did it choose the test-best candidate inside its own shortlist, or did it
  leave easy gains on the table?
* how often did it override, keep or fall back?
* which series helped/hurt the most?

Usage:
    python3 aprovacao_solida/10_final_choice_replay/analyze_replay.py \
        --output-dir outputs/resultados/v11_final_choice_replay/orchestrator_react_v10x_low_gpt-oss-20b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _read_inputs(output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_path = os.path.join(output_dir, "results.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    mapping_path = os.path.join(output_dir, "mapping.jsonl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"missing {results_path}; run the replay first")
    results = pd.read_csv(results_path, sep=";")
    summary = (
        pd.read_csv(summary_path, sep=";")
        if os.path.exists(summary_path)
        else pd.DataFrame()
    )
    mapping_rows: List[Dict[str, Any]] = []
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        mapping_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return results, summary, pd.DataFrame(mapping_rows)


def _artifact_lookup(
    run: str, datasets: List[str], results_dir: str
) -> Dict[Tuple[str, int, str], Optional[float]]:
    """(dataset, series, attempt_id) -> test_smape, from the v10x artifacts."""
    out: Dict[Tuple[str, int, str], Optional[float]] = {}
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
                value = attempt.get("test_smape")
                if attempt_id and value is not None and np.isfinite(float(value)):
                    out[(dataset, series, attempt_id)] = float(value)
    return out


def _shortlist_metrics(
    mapping: pd.DataFrame, lookup: Dict[Tuple[str, int, str], Optional[float]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in mapping.iterrows():
        try:
            specs = json.loads(row.get("shortlist_specs") or "[]")
        except json.JSONDecodeError:
            continue
        values: List[float] = []
        for spec in specs:
            value = lookup.get((str(row["dataset"]), int(row["series"]), str(spec.get("id"))))
            if value is not None:
                values.append(float(value))
        if not values:
            continue
        chosen_value = lookup.get(
            (str(row["dataset"]), int(row["series"]), str(row.get("chosen_id")))
        )
        if chosen_value is None:
            chosen_value = row.get("chosen_test_smape")
        argmin_value = row.get("argmin_test_smape")
        try:
            chosen_value = float(chosen_value) if chosen_value is not None else np.nan
        except (TypeError, ValueError):
            chosen_value = np.nan
        try:
            argmin_value = float(argmin_value) if argmin_value is not None else np.nan
        except (TypeError, ValueError):
            argmin_value = np.nan
        rows.append(
            {
                "dataset": row["dataset"],
                "series": int(row["series"]),
                "decision": str(row.get("decision") or ""),
                "choice": str(row.get("choice") or ""),
                "chosen_id": str(row.get("chosen_id") or ""),
                "argmin_smape": argmin_value,
                "chosen_smape": chosen_value,
                "shortlist_oracle": min(values),
                "shortlist_best_id": next(
                    (
                        str(spec.get("id"))
                        for spec in specs
                        if lookup.get(
                            (str(row["dataset"]), int(row["series"]), str(spec.get("id")))
                        )
                        == min(values)
                    ),
                    "",
                ),
                "chosen_is_shortlist_best": bool(
                    np.isfinite(chosen_value) and chosen_value <= min(values) + 1e-12
                ),
                "argmin_is_shortlist_best": bool(
                    np.isfinite(argmin_value) and argmin_value <= min(values) + 1e-12
                ),
                "regret_vs_shortlist": float(chosen_value - min(values))
                if np.isfinite(chosen_value)
                else np.nan,
                "gain_vs_argmin": float(argmin_value - chosen_value)
                if np.isfinite(chosen_value) and np.isfinite(argmin_value)
                else np.nan,
                "reason": str(row.get("reason") or ""),
            }
        )
    return pd.DataFrame(rows)


def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/resultados/v11_final_choice_replay/orchestrator_react_v10x_low_gpt-oss-20b",
    )
    parser.add_argument("--run", default="orchestrator_react_v10x_low")
    parser.add_argument("--results-dir", default="./timeseries/mestrado/resultados")
    args = parser.parse_args(argv)

    results, summary, mapping = _read_inputs(args.output_dir)
    if results.empty:
        print("empty results.csv")
        return 1

    _print_section("SUMMARY BY DATASET")
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("summary.csv missing")

    usable = results[results["decision"].astype(str) != "dry_run"].copy()
    if usable.empty:
        print("\nOnly dry-run rows; nothing to analyse.")
        return 0

    _print_section("DECISIONS")
    print(usable["decision"].fillna("(missing)").value_counts().to_string())
    print("\nfallbacks:")
    print(usable["fallback"].fillna("none").value_counts().to_string())
    print("\nconfidence:")
    print(usable["confidence"].fillna("(missing)").value_counts().to_string())

    for col in ("argmin_test_smape", "chosen_test_smape", "shortlist_oracle_smape"):
        if col in usable.columns:
            usable[col] = pd.to_numeric(usable[col], errors="coerce")
    decisions = usable["decision"].astype(str).str.lower()
    chosen = usable["chosen_test_smape"].to_numpy(dtype=float)
    argmin = usable["argmin_test_smape"].to_numpy(dtype=float)
    valid = np.isfinite(chosen) & np.isfinite(argmin)

    _print_section("CHOSEN VS ARGMIN")
    print(f"series with metrics : {int(valid.sum())}/{len(usable)}")
    if valid.any():
        c, a = chosen[valid], argmin[valid]
        print(f"chosen mean         : {c.mean():.6f}")
        print(f"argmin mean         : {a.mean():.6f}")
        print(f"mean gain           : {(a - c).mean():+.6f}")
        print(f"median gain         : {np.median(a - c):+.6f}")
        print(f"win / loss / tie    : {np.mean(c < a - 1e-12):.3f} / "
              f"{np.mean(c > a + 1e-12):.3f} / {np.mean(np.abs(c - a) <= 1e-12):.3f}")

    _print_section("PER DATASET")
    rows = []
    for dataset, group in usable.groupby("dataset"):
        c = pd.to_numeric(group["chosen_test_smape"], errors="coerce").to_numpy(dtype=float)
        a = pd.to_numeric(group["argmin_test_smape"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(c) & np.isfinite(a)
        if not mask.any():
            continue
        c, a = c[mask], a[mask]
        d = group["decision"].astype(str).str.lower().to_numpy()[mask]
        rows.append(
            {
                "dataset": dataset,
                "n": int(mask.sum()),
                "argmin": float(np.mean(a)),
                "chosen": float(np.mean(c)),
                "gain": float(np.mean(a - c)),
                "win_rate": float(np.mean(c < a - 1e-12)),
                "loss_rate": float(np.mean(c > a + 1e-12)),
                "override": float(np.mean(d == "override")),
                "keep": float(np.mean(d == "keep")),
                "fallback": float(np.mean(d == "fallback")),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # Join with the v10x artifacts to know whether the chosen candidate was the
    # best of its own shortlist. This is the cleanest measure of selection quality
    # inside the ceiling the selector was given.
    if not mapping.empty:
        lookup = _artifact_lookup(args.run, sorted(usable["dataset"].unique()), args.results_dir)
        if lookup:
            sl = _shortlist_metrics(mapping, lookup)
            if not sl.empty:
                _print_section("SELECTION QUALITY INSIDE THE SHORTLIST")
                print(
                    f"chosen was shortlist-best : {sl['chosen_is_shortlist_best'].mean():.3f}"
                )
                print(
                    f"argmin was shortlist-best : {sl['argmin_is_shortlist_best'].mean():.3f}"
                )
                print(
                    f"mean regret vs shortlist  : {sl['regret_vs_shortlist'].mean():+.6f}"
                )
                print(
                    f"mean gain vs argmin       : {sl['gain_vs_argmin'].mean():+.6f}"
                )
                per_ds = []
                for dataset, group in sl.groupby("dataset"):
                    per_ds.append(
                        {
                            "dataset": dataset,
                            "n": len(group),
                            "chosen_best": float(group["chosen_is_shortlist_best"].mean()),
                            "argmin_best": float(group["argmin_is_shortlist_best"].mean()),
                            "regret": float(group["regret_vs_shortlist"].mean()),
                            "gain_vs_argmin": float(group["gain_vs_argmin"].mean()),
                        }
                    )
                print("\nper dataset:")
                print(pd.DataFrame(per_ds).to_string(index=False))

                _print_section("TOP 10 HELPED BY THE OVERRIDE")
                helped = sl[np.isfinite(sl["gain_vs_argmin"])].nlargest(
                    10, "gain_vs_argmin"
                )
                print(
                    helped[
                        ["dataset", "series", "decision", "choice", "argmin_smape",
                         "chosen_smape", "shortlist_oracle", "gain_vs_argmin", "reason"]
                    ].to_string(index=False)
                )

                _print_section("TOP 10 HURT BY THE OVERRIDE")
                hurt = sl[np.isfinite(sl["gain_vs_argmin"])].nsmallest(
                    10, "gain_vs_argmin"
                )
                print(
                    hurt[
                        ["dataset", "series", "decision", "choice", "argmin_smape",
                         "chosen_smape", "shortlist_oracle", "gain_vs_argmin", "reason"]
                    ].to_string(index=False)
                )

    _print_section("PROMPT FILES")
    print(f"prompt-only : {os.path.join(args.output_dir, 'prompts.jsonl')}")
    print(f"mapping     : {os.path.join(args.output_dir, 'mapping.jsonl')}")
    print(f"results     : {os.path.join(args.output_dir, 'results.csv')}")
    print(f"summary     : {os.path.join(args.output_dir, 'summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
