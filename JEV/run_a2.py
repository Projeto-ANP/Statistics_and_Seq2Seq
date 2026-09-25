#!/usr/bin/env python3
"""A2 — LLM propõe (ReAct) + gate decide, em RODADAS interativas.

Por série:
  Fase 2: sementes (piso)
  Rodada r (r=1..R):
    1. gpt-oss ReAct propõe (orçamento curto por rodada, --per-round)
    2. o GATE (LAY A fine-tuned, alvo por janela LOO) pontua TODO o histórico
       com P por janela + P médio
    3. o LLM da rodada seguinte LÊ o bloco "GATE VERDICT" no prompt e adapta
       a exploração
  Final: argmax P médio do gate sobre o histórico completo.

O LLM nunca escolhe o final; o gate nunca gera. Rodada real exige
--gate-checkpoint (fine-tuned); sem ele o gate é zero-shot (só smoke).

Uso (servidor):
  python3 JEV/run_a2.py --datasets ETTM2 NN5_WEEKLY_DATASET --version a2_v1 \
      --gate-checkpoint JEV/models/laya_gate_ettm2_valw --reasoning low
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "JEV"))
os.chdir(_ROOT)

import pandas as pd  # noqa: E402

from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402
from orchestrator_react import ingest as I  # noqa: E402
from orchestrator_react import pipeline as PL  # noqa: E402
from orchestrator_react import pool as POOL  # noqa: E402
from orchestrator_react import prompts as PR  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.config import LLMRole, ReactConfig  # noqa: E402
from orchestrator_react.csv_writer import CORE_COLUMNS, compute_metrics  # noqa: E402
from orchestrator_react.data_source import load_series_source  # noqa: E402
from orchestrator_react.llm import build_client  # noqa: E402
from orchestrator_react.react_loop import run_react_loop  # noqa: E402
from laya_loop import LayaAgent, build_verdict, run_gate_pass_windows  # noqa: E402
from run_laya import (  # noqa: E402
    DATASET_SOURCES, DEFAULT_RESULTS_DIR, DEFAULT_SOURCE_DIR,
    _seed_floor_metrics, _summarise,
)

COLS: List[str] = CORE_COLUMNS + [
    "description", "origin", "react_iterations_used", "react_stop_reason",
    "seed_floor_smape", "seed_floor_rmse", "ablation_config",
]


def run_dataset(
    dataset: str,
    source_file: str,
    version: str,
    combinator_model: str = "gpt-oss:20b",
    reasoning: Optional[str] = "low",
    gate_checkpoint: Optional[str] = None,
    rounds: int = 2,
    per_round: int = 4,
    dataset_card: bool = True,
    indices: Optional[List[int]] = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    models = [m for m in DEFAULT_MODELS if os.path.exists(I.model_csv_path(m, dataset, results_dir))]
    frames = I.load_dataset_frames(models, dataset, results_dir)
    n_series = I.count_series(dataset, models[0], results_dir)
    try:
        bad = I.find_misaligned_models(
            models, dataset, 0, results_dir=results_dir, frames=frames, n_windows=3,
        )
    except Exception:
        bad = {}
    models_eff = [m for m in models if m not in bad]
    if bad:
        print(f"WARNING: {len(bad)} modelos desalinhados descartados: {sorted(bad)}", flush=True)

    cfg = ReactConfig()
    experiment = f"orchestrator_a2_{version}"
    out_dir = os.path.join(results_dir, experiment)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset}.csv")

    combinator = LLMRole(model=combinator_model, temperature=0.2, seed=7)
    combinator.reasoning = reasoning
    client = build_client(combinator)
    gate_agent = LayaAgent(checkpoint=gate_checkpoint) if gate_checkpoint else None

    todo = indices if indices is not None else list(range(n_series))
    priors: Dict[int, Dict[str, float]] = {}
    if dataset_card:
        try:
            source = load_series_source(
                source_file, n_expected_series=n_series, source_dir=source_dir,
            )
            priors = PL._build_strategy_priors(
                models_eff, dataset, todo, cfg, source, results_dir, frames, {},
            )
        except Exception as exc:
            print(f"WARNING: dataset card falhou ({exc})", flush=True)

    print(f"dataset      : {dataset}")
    print(f"llm          : {combinator_model} reasoning={reasoning} | rodadas={rounds} x {per_round} iters")
    print(f"gate         : {gate_checkpoint or 'ZERO-SHOT (smoke — não é o resultado real)'}")
    print(f"writing to   : {out_dir}")
    print("-" * 74, flush=True)

    rows: List[Dict[str, Any]] = []
    per_series: List[Dict[str, float]] = []
    floor_rows: List[Dict[str, float]] = []
    external: Dict[str, List[Dict[str, float]]] = {}
    started = time.perf_counter()
    ok = failed = 0

    for idx in todo:
        try:
            ing = I.load_series(
                models=models_eff, dataset=dataset, dataset_index=idx, config=cfg,
                results_dir=results_dir, source_file=source_file, source_dir=source_dir,
                frames=frames,
            )
            state = ing.state
            phase2 = POOL.run_phase2(state, cfg)
            series_card = T.series_profile(state)
            pool_card = phase2["report"]
            if priors.get(idx):
                state.strategy_prior = priors[idx]
            card = PR.build_dataset_card(state) if priors.get(idx) else None

            round_cfg = ReactConfig(max_iterations=per_round)
            verdict = None
            react_results = []
            for r in range(rounds):
                rr = run_react_loop(
                    state=state, client=client, series_card=series_card,
                    pool_card=pool_card, config=round_cfg, gate_verdict=verdict,
                )
                react_results.append(rr.summary())
                if gate_agent is not None:
                    scored = run_gate_pass_windows(state, gate_agent, series_card, pool_card)
                    verdict = build_verdict(scored)
                print(
                    f"[{idx:>4}] rodada {r+1}: +iters={rr.iterations_used} "
                    f"stop={rr.stop_reason} | histórico={len(state.attempts)}",
                    flush=True,
                )

            # ── final: argmax P do gate (ou argmin se sem gate) ─────────────
            final_origin = "gate"
            final_spec = None
            if gate_agent is not None:
                scored = run_gate_pass_windows(state, gate_agent, series_card, pool_card)
                best_g = max(scored, key=lambda s: s["p_mean"])
                final_spec = best_g["spec"]
                if best_g["origin"] == "baseline":
                    final_origin = "gate(baseline)"
            if final_spec is None:
                final_spec = state.best_attempt().spec
                final_origin = "argmin"
            forecast, _ = state.apply_to_test(final_spec)
            metrics = compute_metrics(forecast, ing.test_values)
            floor = _seed_floor_metrics(state, ing.test_values)

            rows.append({
                "dataset_index": str(idx), "horizon": ing.horizon, "regressor": experiment,
                **metrics,
                "test": [list(ing.test_values)], "predictions": [list(forecast)],
                "start_test": str(ing.start_test), "final_test": str(ing.final_test),
                "description": json.dumps({
                    "strategy": final_spec, "origin": final_origin,
                    "rounds": react_results,
                    "gate_scores": scored if gate_agent is not None else None,
                    "dataset_card": card,
                }, ensure_ascii=False, default=str),
                "origin": final_origin,
                "react_iterations_used": sum(rr.get("iterations_used", 0) for rr in react_results),
                "react_stop_reason": react_results[-1].get("stop_reason") if react_results else "",
                "seed_floor_smape": floor["smape"] if floor else None,
                "seed_floor_rmse": floor["rmse"] if floor else None,
                "ablation_config": f"a2_{version}_{combinator_model.replace(':','-')}_r{rounds}x{per_round}",
            })
            per_series.append(metrics)
            if floor:
                floor_rows.append(floor)
            ok += 1
            print(
                f"[{idx:>4}] final={final_spec['combine']:<14} origin={final_origin:<14} "
                f"| TEST smape={metrics['smape']:.4f} | FLOOR smape={floor['smape'] if floor else float('nan'):.4f}"
            )
            ext = I.read_external_baselines(dataset, idx, results_dir=results_dir)
            for name, stats in ext.items():
                if isinstance(stats, dict) and stats.get("available"):
                    external.setdefault(name, []).append(stats)
        except Exception as exc:
            failed += 1
            rows.append({
                "dataset_index": str(idx), "horizon": 0, "regressor": experiment,
                **{k: float("nan") for k in ("mape", "pocid", "smape", "rmse", "msmape", "mae")},
                "test": [[]], "predictions": [[]], "start_test": "", "final_test": "",
                "description": json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                "origin": "", "react_iterations_used": 0, "react_stop_reason": "error",
                "seed_floor_smape": None, "seed_floor_rmse": None,
                "ablation_config": f"a2_{version}_{combinator_model.replace(':','-')}",
            })
            print(f"[{idx:>4}] FAILED: {type(exc).__name__}: {exc}")

    frame = pd.DataFrame(rows).reindex(columns=COLS)
    frame.to_csv(csv_path, sep=";", index=False)

    elapsed = time.perf_counter() - started
    summary = _summarise(per_series)
    floor_sum = _summarise(floor_rows)
    print("-" * 74)
    print(f"done in {elapsed:.1f}s | ok: {ok} | failed: {failed}")
    if summary:
        print(f"DATASET SUMMARY over {len(per_series)} series (mean across series):")
        print(f"  this run   smape={summary['smape']:.4f}  rmse={summary['rmse']:.4f}"
              f"  pocid={summary['pocid']:.2f}  mape={summary['mape']:.4f}")
        if floor_sum:
            note = ("  <- this run is better" if summary["rmse"] < floor_sum["rmse"]
                    else "  <- this run is WORSE than the seed floor")
            print(f"  baseline   smape={floor_sum['smape']:.4f}  rmse={floor_sum['rmse']:.4f}"
                  f"  pocid={floor_sum['pocid']:.2f}  mape={floor_sum['mape']:.4f}{note}")
        for name, rows_ in external.items():
            stats = _summarise(rows_)
            print(f"  {name:<10} smape={stats['smape']:.4f}  rmse={stats['rmse']:.4f}"
                  f"  pocid={stats['pocid']:.2f}  mape={stats['mape']:.4f}")
    print(f"csv: {csv_path}")
    return {"dataset": dataset, "experiment": experiment, "n_ok": ok, "n_failed": failed,
            "csv_path": csv_path, "elapsed_s": elapsed, "summary": summary,
            "baseline": floor_sum}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="A2: ReAct propõe + gate decide, em rodadas.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--version", default="a2_v1")
    p.add_argument("--combinator", default="gpt-oss:20b")
    p.add_argument("--reasoning", default="low")
    p.add_argument("--gate-checkpoint", default=None)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--per-round", type=int, default=4)
    p.add_argument("--no-dataset-card", action="store_true")
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = p.parse_args(argv)

    failures = 0
    for dataset in args.datasets:
        source = DATASET_SOURCES.get(dataset)
        if source is None:
            print(f"dataset '{dataset}' não está no mapa de fontes")
            failures += 1
            continue
        try:
            run_dataset(
                dataset=dataset, source_file=source, version=args.version,
                combinator_model=args.combinator,
                reasoning=None if args.reasoning == "None" else args.reasoning,
                gate_checkpoint=args.gate_checkpoint,
                rounds=args.rounds, per_round=args.per_round,
                dataset_card=not args.no_dataset_card,
                indices=args.indices,
                source_dir=args.source_dir, results_dir=args.results_dir,
            )
        except Exception as exc:
            print(f"\nDATASET {dataset} ABORTOU: {type(exc).__name__}: {exc}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
