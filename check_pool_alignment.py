"""Diagnostic: is the model pool actually aligned on the same windows?

Run this before a real run whenever the orchestrator rejects a dataset. It answers,
per model:

  * how many windows exist per series (models are often run with different budgets);
  * what time range those windows carry;
  * and — the only thing that really matters — whether the last N windows contain
    the SAME observations as the reference model.

Two failures look alike in the logs but need opposite responses:

  A. same data, different labels
     The result files disagree on `start_test` / `final_test` but the `test`
     values are identical. That is a frequency-labelling bug in whoever wrote
     those files (e.g. an hourly series indexed with freq="15min"). The pool is
     usable; the orchestrator warns and carries on.

  B. genuinely different windows
     The `test` values differ, so the models forecast different periods.
     Combining them would mix periods. The orchestrator refuses, and the model
     must be dropped or re-run.

Usage
-----
    python check_pool_alignment.py --dataset ETTH1
    python check_pool_alignment.py --dataset ETTH1 --windows 3 --reference ARIMA
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator_react.ingest import (
    DEFAULT_RESULTS_DIR,
    extract_values,
    model_csv_path,
    read_model_predictions,
)


DEFAULT_MODELS: List[str] = [
    "ARIMA", "ETS", "THETA", "rf", "catboost",
    "CWT_rf", "DWT_rf", "FT_rf",
    "CWT_catboost", "DWT_catboost", "FT_catboost",
    "ONLY_CWT_catboost", "ONLY_CWT_rf",
    "ONLY_DWT_catboost", "ONLY_DWT_rf",
    "ONLY_FT_catboost", "ONLY_FT_rf",
    "NaiveSeasonal", "NaiveMovingAverage",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--windows", type=int, default=3, help="validation windows the run needs")
    p.add_argument("--reference", default=None, help="model to compare against (default: the first)")
    p.add_argument("--series", type=int, default=None, help="check one series only")
    args = p.parse_args(argv)

    models = args.models or DEFAULT_MODELS
    needed = int(args.windows) + 1  # validation windows + the blind test window

    present, missing = [], []
    for m in models:
        (present if os.path.exists(model_csv_path(m, args.dataset, args.results_dir)) else missing).append(m)
    if missing:
        print(f"MISSING result files ({len(missing)}): {missing}\n")
    if not present:
        print("no model has results for this dataset")
        return 2

    frames = {m: read_model_predictions(m, args.dataset, args.results_dir) for m in present}
    reference = args.reference or present[0]
    if reference not in frames:
        print(f"reference {reference!r} has no results")
        return 2

    # ── shape of each file ───────────────────────────────────────────────────
    print(f"dataset: {args.dataset} | reference: {reference} | windows needed: {needed}\n")
    print(f"{'model':24s} {'series':>6} {'win/series':>10} {'horizon':>7}  {'oldest start':<20} {'newest final':<20} {'step':>16}")
    print("-" * 118)

    per_series_counts = {}
    for m in present:
        df = frames[m]
        counts = df.groupby("dataset_index").size()
        per_series_counts[m] = counts
        d0 = df[df["dataset_index"] == counts.index[0]].sort_values("start_test")
        step = (d0["start_test"].iloc[1] - d0["start_test"].iloc[0]) if len(d0) > 1 else pd.NaT
        flag = "" if counts.min() >= needed else "  <-- TOO FEW WINDOWS"
        print(
            f"{m:24s} {df['dataset_index'].nunique():6d} "
            f"{str(counts.min()) + ('-' + str(counts.max()) if counts.min() != counts.max() else ''):>10} "
            f"{int(df['horizon'].iloc[0]):7d}  "
            f"{str(d0['start_test'].iloc[0]):<20} {str(d0['final_test'].iloc[-1]):<20} {str(step):>16}{flag}"
        )

    # ── the check that decides whether the pool is usable ────────────────────
    print("\n" + "=" * 118)
    print(f"Comparing the last {needed} windows against {reference}, by VALUE (labels ignored)\n")

    ref_frame = frames[reference]
    series_ids = (
        [int(args.series)] if args.series is not None
        else sorted(ref_frame["dataset_index"].unique())
    )

    same_data, label_only, different, too_few = [], [], [], []
    for m in present:
        if m == reference:
            continue
        value_ok, label_ok, enough = True, True, True
        for idx in series_ids:
            ref = ref_frame[ref_frame["dataset_index"] == idx].sort_values("start_test")
            cur = frames[m][frames[m]["dataset_index"] == idx].sort_values("start_test")
            if len(ref) < needed or len(cur) < needed:
                enough = False
                continue
            for k in range(needed):
                a = np.asarray(extract_values(ref.iloc[-1 - k]["test"]), dtype=float)
                b = np.asarray(extract_values(cur.iloc[-1 - k]["test"]), dtype=float)
                n = min(a.size, b.size)
                if n == 0 or not np.allclose(a[:n], b[:n], rtol=1e-4, atol=1e-4):
                    value_ok = False
                if ref.iloc[-1 - k]["final_test"] != cur.iloc[-1 - k]["final_test"]:
                    label_ok = False
        if not enough:
            too_few.append(m)
        elif not value_ok:
            different.append(m)
        elif not label_ok:
            label_only.append(m)
        else:
            same_data.append(m)

    def report(title: str, group: List[str], note: str) -> None:
        if group:
            print(f"{title} ({len(group)})")
            print(f"    {note}")
            for m in group:
                print(f"    - {m}")
            print()

    report("IDENTICAL", same_data, "same windows, same labels. Nothing to do.")
    report(
        "SAME DATA, WRONG LABELS", label_only,
        "the observations match but start_test/final_test disagree. A frequency-labelling\n"
        "    bug in these result files. The orchestrator warns and uses them; the reported\n"
        f"    final_test comes from the reference model ({reference}).",
    )
    report(
        "DIFFERENT WINDOWS", different,
        "these models forecast a different period. The orchestrator refuses them.\n"
        "    Drop them from `models`, or re-run them on the same origins.",
    )
    report(
        "TOO FEW WINDOWS", too_few,
        f"fewer than {needed} windows for some series. Lower --windows, or re-run them.",
    )

    if different or too_few:
        usable = [reference] + same_data + label_only
        print("Suggested pool for this dataset:\n")
        print("models = [")
        for m in usable:
            print(f'    "{m}",')
        print("]")
        return 1

    print("The pool is consistent: every model forecasts the same windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
