"""Three-way comparison for the v11 final decision.

For the same v10x trajectory, compares:

* argmin (the v10 leader);
* seasonal_guard (deterministic gate + closest to seasonal naive);
* seasonal_guard + binary LLM veto.

The LLM is called only when the guard wants to switch (about 15% of the series),
so a full low/medium run is cheap. The test actual is used only for evaluation.

Examples:
    # guard only, prompt audit, no LLM
    python3 .../compare_three.py --run orchestrator_react_v10x_low --dry-run

    # real binary veto (low)
    REACT_OLLAMA_URL=http://127.0.0.1:11435 python3 .../compare_three.py \
        --run orchestrator_react_v10x_low \
        --model gpt-oss:20b --reasoning low --temperature 0.2 --llm-seed 11
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator_react import final_choice as FC  # noqa: E402
from orchestrator_react.config import LLMRole, ReactConfig  # noqa: E402
from orchestrator_react.llm import LLMClient, build_client, check_client  # noqa: E402
from run_tsf_orchestrator import parse_reasoning  # noqa: E402

from replay_final_choice import (  # noqa: E402
    OFFICIAL_DATASETS,
    MockClient,
    _artifact_indices,
    _attempts_from_artifact,
    _load_artifact,
    _load_dataset_state_builder,
)


RESULT_COLUMNS = (
    "run", "model", "dataset", "series", "n_attempts", "shortlist_size",
    "gate_active", "gate_reason", "seasonal_strength", "n_points", "period",
    "leader_id", "argmin_smape",
    "guard_decision", "guard_id", "seasonal_id", "guard_smape",
    "seasonal_smape", "guard_gain_vs_argmin",
    "llm_called", "veto_reason", "veto", "veto_label", "veto_id", "veto_smape",
    "veto_gain_vs_argmin", "veto_gain_vs_guard",
    "oracle_two", "veto_correct", "veto_confidence", "veto_answer",
    "veto_reason_text",
)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in str(text)).strip("-") or "run"


def _test_smape(attempt: Any) -> Optional[float]:
    value = getattr(attempt, "test_smape", None)
    try:
        return float(value) if value is not None and np.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out: List[Dict[str, Any]] = []
    for dataset, group in rows.groupby("dataset"):
        def mean(col: str) -> Optional[float]:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            return float(values.mean()) if len(values) else None

        argmin = pd.to_numeric(group["argmin_smape"], errors="coerce")
        guard = pd.to_numeric(group["guard_smape"], errors="coerce")
        veto = pd.to_numeric(group["veto_smape"], errors="coerce")
        called = group["llm_called"].astype(str).str.lower().isin(["true", "1"])
        out.append(
            {
                "dataset": dataset,
                "n": int(len(group)),
                "argmin_smape": mean("argmin_smape"),
                "guard_smape": mean("guard_smape"),
                "veto_smape": mean("veto_smape"),
                "guard_gain": float((argmin - guard).mean()),
                "veto_gain": float((argmin - veto).mean()),
                "veto_gain_vs_guard": float((guard - veto).mean()),
                "guard_switch_rate": float((group["guard_decision"] == "switch").mean()),
                "llm_call_rate": float(called.mean()),
                "veto_switch_rate": float(
                    (group.loc[called, "veto_id"] == group.loc[called, "seasonal_id"]).mean()
                )
                if called.any()
                else None,
                "veto_accuracy": float(
                    pd.to_numeric(group.loc[called, "veto_correct"], errors="coerce").mean()
                )
                if called.any()
                else None,
                "gate_active_rate": float(
                    group["gate_active"].astype(str).str.lower().isin(["true", "1"]).mean()
                ),
            }
        )
    summary = pd.DataFrame(out)
    if not summary.empty:
        total = {"dataset": "MEAN_DATASETS", "n": int(summary["n"].sum())}
        for col in summary.columns:
            if col in ("dataset", "n"):
                continue
            total[col] = float(summary[col].mean())
        summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="orchestrator_react_v10x_low")
    parser.add_argument("--datasets", nargs="+", default=list(OFFICIAL_DATASETS))
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default="./timeseries/mestrado/resultados")
    parser.add_argument("--source-dir", default="../forecasting_datasets")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning", default=None, metavar="LEVEL")
    parser.add_argument("--llm-seed", type=int, default=11)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--min-strength", type=float, default=0.5)
    parser.add_argument("--min-cycles", type=float, default=10.0)
    parser.add_argument("--degenerate-factor", type=float, default=100.0)
    parser.add_argument("--no-low-confidence-keep", dest="low_confidence_keep",
                        action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="build prompts and run the guard; no LLM call")
    parser.add_argument("--mock-choice", default=None,
                        help="testing: answer this label (A/B) for every veto call")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-llm-failures", type=int, default=5)
    args = parser.parse_args(argv)

    suffix = "_dry_run" if args.dry_run else ""
    output_dir = args.output_dir or os.path.join(
        "outputs/resultados/v11_three_way",
        f"{_slug(args.run)}_{_slug(args.model)}{suffix}",
    )
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    prompts_path = os.path.join(output_dir, "prompts.jsonl")
    mapping_path = os.path.join(output_dir, "mapping.jsonl")

    config = ReactConfig()
    config.name = args.run
    config.final_strategy = "seasonal_veto"
    config.final_selector_shuffle_seed = int(args.shuffle_seed)
    config.final_selector_max_candidates = int(args.max_candidates)
    config.final_selector_low_confidence_keep = bool(args.low_confidence_keep)
    config.final_selector_degenerate_factor = float(args.degenerate_factor)
    config.seasonal_guard_min_strength = float(args.min_strength)
    config.seasonal_guard_min_cycles = float(args.min_cycles)
    # The guard's shortlist and the veto share the same cap.
    config.final_selector_max_candidates = int(args.max_candidates)

    base_url = args.base_url or os.environ.get("REACT_OLLAMA_URL", "http://127.0.0.1:11434")
    client: Optional[LLMClient] = None
    mock_response: Optional[str] = None
    if not args.dry_run:
        if args.mock_choice is not None:
            mock_response = (
                f"CHOICE: {str(args.mock_choice).strip()}\n"
                "CONFIDENCE: high\n"
                "REASON: mock comparison response"
            )
            client = MockClient(mock_response)
        else:
            role = LLMRole(
                model=args.model,
                temperature=float(args.temperature),
                base_url=base_url,
                seed=int(args.llm_seed),
                reasoning=parse_reasoning(args.reasoning),
            )
            client = build_client(role)
            ok, detail = check_client(client)
            if not ok:
                print(f"LLM preflight failed: {detail}", file=sys.stderr)
                return 2

    existing: set = set()
    if args.resume and os.path.exists(results_path):
        try:
            old = pd.read_csv(results_path, sep=";")
            existing = {
                (str(row["dataset"]), int(row["series"]))
                for _, row in old.iterrows()
                if pd.notna(row.get("dataset")) and pd.notna(row.get("series"))
            }
        except Exception:
            existing = set()

    header_written = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    started = time.time()
    llm_failures = 0
    abort = False

    for dataset in args.datasets:
        if abort:
            break
        indices = _artifact_indices(args.run, dataset, args.results_dir)
        if args.indices:
            wanted = set(int(i) for i in args.indices)
            indices = [i for i in indices if i in wanted]
        if args.limit is not None:
            indices = indices[: int(args.limit)]
        if not indices:
            print(f"[{dataset}] no artifacts/indices", flush=True)
            continue
        try:
            load_state = _load_dataset_state_builder(dataset, args.results_dir, args.source_dir)
        except Exception as exc:
            print(f"[{dataset}] data load failed: {exc}", flush=True)
            continue
        print(f"[{dataset}] {len(indices)} series", flush=True)
        for index in indices:
            key = (dataset, int(index))
            if key in existing:
                continue
            payload = _load_artifact(args.run, dataset, index, args.results_dir)
            if payload is None:
                continue
            attempts = _attempts_from_artifact(payload)
            if not attempts:
                continue
            try:
                ingested = load_state(index)
            except Exception as exc:
                print(f"  {dataset} {index}: load failed: {exc}", flush=True)
                continue
            state = ingested.state
            state.attempts = list(attempts)
            series_card = payload.get("series_card") or {}
            forecast_for = lambda attempt: np.asarray(attempt.test_forecast, dtype=float)
            leader = min(attempts, key=lambda a: a.score)

            guard_attempt, guard_debug = FC.choose_seasonal_guard(
                state, series_card, config, dataset, index, forecast_for=forecast_for
            )
            if args.dry_run:
                veto_attempt, veto_debug = FC.choose_seasonal_veto(
                    state, series_card, payload.get("diagnosis") or {}, None,
                    config, dataset, index, forecast_for=forecast_for, dry_run=True,
                )
            else:
                veto_attempt, veto_debug = FC.choose_seasonal_veto(
                    state, series_card, payload.get("diagnosis") or {}, client,
                    config, dataset, index, forecast_for=forecast_for,
                )

            seasonal_id = guard_debug.get("seasonal_id")
            seasonal_attempt = next(
                (a for a in attempts if a.attempt_id == seasonal_id), None
            )
            argmin_smape = _test_smape(leader)
            guard_smape = _test_smape(guard_attempt) if guard_attempt is not None else None
            seasonal_smape = _test_smape(seasonal_attempt) if seasonal_attempt is not None else None
            veto_smape = _test_smape(veto_attempt) if veto_attempt is not None else None
            gate = guard_debug.get("gate") or {}
            called = bool(veto_debug.get("llm_called"))
            oracle_two = (
                min(v for v in (argmin_smape, seasonal_smape) if v is not None)
                if guard_debug.get("decision") == "switch"
                else None
            )
            veto_correct = (
                veto_smape is not None
                and oracle_two is not None
                and veto_smape <= oracle_two + 1e-12
            )
            veto_id = getattr(veto_attempt, "attempt_id", None)
            row = {
                "run": args.run,
                "model": args.model,
                "dataset": dataset,
                "series": int(index),
                "n_attempts": len(attempts),
                "shortlist_size": guard_debug.get("shortlist_size"),
                "gate_active": bool(gate.get("active")),
                "gate_reason": gate.get("reason"),
                "seasonal_strength": gate.get("seasonal_strength"),
                "n_points": gate.get("n_points"),
                "period": gate.get("period"),
                "leader_id": getattr(leader, "attempt_id", None),
                "argmin_smape": argmin_smape,
                "guard_decision": guard_debug.get("decision"),
                "guard_id": getattr(guard_attempt, "attempt_id", None),
                "seasonal_id": seasonal_id,
                "guard_smape": guard_smape,
                "seasonal_smape": seasonal_smape,
                "guard_gain_vs_argmin": (
                    float(argmin_smape - guard_smape)
                    if argmin_smape is not None and guard_smape is not None
                    else None
                ),
                "llm_called": called,
                "veto_reason": veto_debug.get("veto_reason"),
                "veto": bool(veto_debug.get("veto")),
                "veto_label": veto_debug.get("parsed", {}).get("choice")
                if isinstance(veto_debug.get("parsed"), dict)
                else None,
                "veto_id": veto_id,
                "veto_smape": veto_smape,
                "veto_gain_vs_argmin": (
                    float(argmin_smape - veto_smape)
                    if argmin_smape is not None and veto_smape is not None
                    else None
                ),
                "veto_gain_vs_guard": (
                    float(guard_smape - veto_smape)
                    if guard_smape is not None and veto_smape is not None
                    else None
                ),
                "oracle_two": oracle_two,
                "veto_correct": veto_correct if called else None,
                "veto_confidence": veto_debug.get("parsed", {}).get("confidence")
                if isinstance(veto_debug.get("parsed"), dict)
                else None,
                "veto_answer": veto_debug.get("response"),
                "veto_reason_text": veto_debug.get("parsed", {}).get("reason")
                if isinstance(veto_debug.get("parsed"), dict)
                else None,
            }
            if veto_debug.get("fallback") == "llm_error":
                llm_failures += 1
            else:
                llm_failures = 0
            result_row = {col: row.get(col) for col in RESULT_COLUMNS}
            pd.DataFrame([result_row]).to_csv(
                results_path, sep=";", mode="a", header=not header_written, index=False
            )
            header_written = True
            prompt_record = {
                "dataset": dataset,
                "series": int(index),
                "system": (veto_debug.get("prompt") or {}).get("system", ""),
                "user": (veto_debug.get("prompt") or {}).get("user", ""),
                "raw_response": veto_debug.get("response", ""),
                "choice": veto_debug.get("parsed", {}).get("choice")
                if isinstance(veto_debug.get("parsed"), dict)
                else None,
                "confidence": veto_debug.get("parsed", {}).get("confidence")
                if isinstance(veto_debug.get("parsed"), dict)
                else None,
                "reason": veto_debug.get("parsed", {}).get("reason")
                if isinstance(veto_debug.get("parsed"), dict)
                else None,
            }
            with open(prompts_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(prompt_record, ensure_ascii=False, default=str) + "\n")
            with open(mapping_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(
                f"  {dataset} {index}: gate={gate.get('active')} "
                f"guard={guard_debug.get('decision')} veto={veto_debug.get('veto_reason')} "
                f"argmin={argmin_smape} guard={guard_smape} veto={veto_smape}",
                flush=True,
            )
            if args.max_llm_failures and llm_failures >= int(args.max_llm_failures):
                print("aborting after repeated LLM failures", file=sys.stderr)
                abort = True
                break

    if os.path.exists(results_path):
        rows = pd.read_csv(results_path, sep=";")
        summary = _summary(rows)
        if not summary.empty:
            summary.to_csv(summary_path, sep=";", index=False)
            print("\nSUMMARY")
            print(summary.round(6).to_string(index=False))
    print(f"\nDone in {time.time() - started:.1f}s. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
