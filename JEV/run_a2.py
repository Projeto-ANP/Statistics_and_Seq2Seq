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
import numpy as np  # noqa: E402

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
    gate_mode: str = "logistic",
    gate_data: str = "JEV/data/gate_dataset.jsonl",
    rounds: int = 2,
    per_round: int = 4,
    dataset_card: bool = True,
    consensus: bool = False,
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
    gate_cache: Dict[str, Any] = {}
    if gate_mode == "logistic":
        from logistic_gate import LogisticGateW, candidate_features

        def gate_pass(state, series_card, pool_card):
            gate_w = gate_cache.setdefault(dataset, LogisticGateW(gate_data, holdout=dataset))
            seen, scored, inputs = set(), [], []
            for a in state.ranked_attempts():
                key = json.dumps(a.spec, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                feats = candidate_features(state, a, pool_card)
                pw, pm = gate_w.score(feats)
                scored.append({
                    "id": a.attempt_id, "spec": a.spec,
                    "strategy": a.brief(include_rationale=False)["strategy"],
                    "score_val": round(float(a.score), 6),
                    "p_windows": [round(x, 3) for x in pw],
                    "p_mean": round(pm, 3),
                    "origin": a.origin,
                })
                inputs.append({
                    "id": a.attempt_id,
                    "strategy": a.brief(include_rationale=False)["strategy"],
                    "features": {k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in feats.items()},
                })
            return scored, inputs
    else:
        def gate_pass(state, series_card, pool_card):
            return run_gate_pass_windows(state, gate_agent, series_card, pool_card)

    todo = indices if indices is not None else list(range(n_series))
    priors: Dict[int, Dict[str, float]] = {}
    t_meta_start = time.perf_counter()
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
    gate_label = (
        f"logístico por-janela (LOO, {gate_data})"
        if gate_mode == "logistic"
        else (gate_checkpoint or "LAY A ZERO-SHOT (smoke — não é o resultado real)")
    )
    print(f"gate         : {gate_label}")
    print(f"meta pre-pass: {time.perf_counter() - t_meta_start:.1f}s (dataset card LOO)")
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
            t_series_start = time.perf_counter()
            t_fase0 = time.perf_counter()
            ing = I.load_series(
                models=models_eff, dataset=dataset, dataset_index=idx, config=cfg,
                results_dir=results_dir, source_file=source_file, source_dir=source_dir,
                frames=frames,
            )
            t_fase0 = time.perf_counter() - t_fase0
            t_p2 = time.perf_counter()
            state = ing.state
            phase2 = POOL.run_phase2(state, cfg)
            t_p2 = time.perf_counter() - t_p2
            series_card = T.series_profile(state)
            pool_card = phase2["report"]
            # consenso das previsões de TESTE (entradas) — estático por série
            consensus_agr = T.test_forecast_agreement(state) if consensus else None

            def _cons_rel(spec):
                if consensus_agr is None:
                    return None
                fc, _ = state.apply_to_test(spec)
                dist = float(np.mean(np.abs(fc - np.asarray(consensus_agr["consensus"]))))
                return round(dist / (consensus_agr["dispersion"] + 1e-9), 3)
            if priors.get(idx):
                state.strategy_prior = priors[idx]
            card = PR.build_dataset_card(state) if priors.get(idx) else None

            round_cfg = ReactConfig(max_iterations=per_round)
            verdict = None
            react_results = []
            rounds_info: List[Dict[str, Any]] = []
            timing_acc = {"llm_s": 0.0, "gate_s": 0.0}

            # ── início da série: o que a etapa determinística entregou ─────
            seed_leader = state.best_attempt()
            print(
                f"[{idx:>4}] INÍCIO: melhor semente = {seed_leader.brief(include_rationale=False)['strategy']} "
                f"(score {seed_leader.score:.4f}) | histórico inicial = {len(state.attempts)}",
                flush=True,
            )

            def _on_step(_i, entry):
                args = json.dumps(entry.get("action_args") or {}, ensure_ascii=False, default=str)[:90]
                thought = " ".join(str(entry.get("thought") or "").split())[:120]
                print(
                    f"[{idx:>4}]   r{r+1}.{entry['iteration']} | {entry.get('action')} {args}",
                    flush=True,
                )
                if thought:
                    print(f"[{idx:>4}]     think: {thought}", flush=True)

            for r in range(rounds):
                rr = run_react_loop(
                    state=state, client=client, series_card=series_card,
                    pool_card=pool_card, config=round_cfg, gate_verdict=verdict,
                    on_step=_on_step,
                )
                react_results.append(rr.summary())
                t_llm_r = sum(float(e.get("llm_call_s", 0.0)) for e in rr.trajectory)
                t_tool_r = sum(float(e.get("tool_exec_s", 0.0)) for e in rr.trajectory)
                t_gate_r0 = time.perf_counter()
                scored, gate_inputs = gate_pass(state, series_card, pool_card)
                t_gate_r = time.perf_counter() - t_gate_r0
                verdict = build_verdict(scored)
                rounds_info.append({
                    "round": r + 1,
                    "react": {
                        "summary": rr.summary(),
                        "trajectory": rr.trajectory,
                        "step_details": rr.step_details,
                        "prompts": rr.prompts,
                        "errors": rr.errors,
                    },
                    "gate": {"scores": scored, "inputs": gate_inputs,
                             "verdict": verdict},
                    "timing": {"llm_s": round(t_llm_r, 3),
                               "llm_tool_exec_s": round(t_tool_r, 3),
                               "gate_s": round(t_gate_r, 3)},
                })
                timing_acc["llm_s"] += t_llm_r
                timing_acc["gate_s"] += t_gate_r
                print(
                    f"[{idx:>4}] rodada {r+1}: +iters={rr.iterations_used} "
                    f"stop={rr.stop_reason} | histórico={len(state.attempts)} "
                    f"| LLM={t_llm_r:.1f}s gate={t_gate_r:.2f}s",
                    flush=True,
                )
                top = sorted(scored, key=lambda s: -s["p_mean"])[:4]
                for s in top:
                    cons_s = f" cons={_cons_rel(s['spec'])}" if consensus else ""
                    print(
                        f"[{idx:>4}]     GATE {s['id']:<4} {s['strategy'][:38]:<38} "
                        f"P={s['p_windows']} mean={s['p_mean']}{cons_s} ({s['origin']})",
                        flush=True,
                    )

            # ── final: argmax P do gate (ou argmin se sem gate) ─────────────
            final_origin = "gate"
            final_spec = None
            scored, final_gate_inputs = gate_pass(state, series_card, pool_card)
            best_g = max(scored, key=lambda s: s["p_mean"])
            if consensus and consensus_agr is not None:
                # top-3 por P → desempate pelo MAIS PRÓXIMO do consenso das
                # previsões de teste (hipótese: quando o gate não separa o
                # topo, o consenso transfere melhor)
                top3 = sorted(scored, key=lambda s: -s["p_mean"])[:3]
                chosen = min(top3, key=lambda s: _cons_rel(s["spec"]))
                if chosen["id"] != best_g["id"]:
                    best_g = chosen
                    final_origin = "gate+consensus"
            final_spec = best_g["spec"]
            if best_g["origin"] == "baseline" and final_origin == "gate":
                final_origin = "gate(baseline)"
            if final_spec is None:
                final_spec = state.best_attempt().spec
                final_origin = "argmin"
            argmin_g = min(scored, key=lambda s: s["score_val"])
            forecast, _ = state.apply_to_test(final_spec)
            metrics = compute_metrics(forecast, ing.test_values)
            floor = _seed_floor_metrics(state, ing.test_values)

            print(
                f"[{idx:>4}] ESCOLHA DO GATE: {best_g['id']} {best_g['strategy'][:38]} "
                f"(P={best_g['p_mean']}, windows {best_g['p_windows']}, origin={best_g['origin']})",
                flush=True,
            )
            print(
                f"[{idx:>4}]   argmin seria : {argmin_g['id']} {argmin_g['strategy'][:38]} "
                f"(score {argmin_g['score_val']:.4f})"
                f"{'  <- gate discorda do argmin' if argmin_g['id'] != best_g['id'] else ''}",
                flush=True,
            )

            # ── artifact completo por série (prompts, turnos, gate, decisão) ─
            art_dir = os.path.join(out_dir, "llm_artifacts", dataset)
            os.makedirs(art_dir, exist_ok=True)
            try:
                with open(os.path.join(art_dir, f"dataset_{idx}.json"), "w", encoding="utf-8") as fh:
                    json.dump({
                        "dataset": dataset, "series": int(idx),
                        "config": {
                            "combinator": combinator_model, "reasoning": reasoning,
                            "rounds": rounds, "per_round": per_round,
                            "gate_mode": gate_mode, "gate_data": gate_data,
                        },
                        "seed_leader": seed_leader.brief(include_rationale=False)
                        if seed_leader else None,
                        "floor": floor,
                        # histórico COMPLETO de tentativas: cada estratégia com
                        # métricas agregadas e por janela, pesos resolvidos,
                        # racional e origem — a trilha de cálculo integral
                        "attempt_history": [
                            {
                                "id": a.attempt_id,
                                "spec": a.spec,
                                "origin": a.origin,
                                "iteration": a.iteration,
                                "rationale": a.rationale,
                                "score": round(float(a.score), 6),
                                "aggregate": {k: (round(v, 6) if isinstance(v, float) else v)
                                              for k, v in (a.aggregate or {}).items()},
                                "per_window": a.per_window,
                                "per_window_scores": [round(float(v), 6) for v in a.per_window_scores],
                                "n_models": a.n_models,
                                "weights_resolved": (
                                    state.resolved_weights_map(a.spec["weights"])
                                    if a.spec.get("combine") == "weighted"
                                    and a.spec.get("weights") in state.weights
                                    else None
                                ),
                            }
                            for a in state.attempts
                        ],
                        "final": {
                            "spec": final_spec, "origin": final_origin,
                            "gate_choice": {
                                "id": best_g["id"], "strategy": best_g["strategy"],
                                "p_mean": best_g["p_mean"],
                                "p_windows": best_g["p_windows"],
                            },
                            "argmin_choice": {
                                "id": argmin_g["id"], "strategy": argmin_g["strategy"],
                                "score_val": argmin_g["score_val"],
                            },
                        },
                        "metrics": {k: metrics[k] for k in ("smape", "rmse", "pocid", "mape")},
                        "timing": {
                            "fase0_ingest_s": round(t_fase0, 3),
                            "fase2_sementes_s": round(t_p2, 3),
                            "llm_total_s": round(timing_acc["llm_s"], 3),
                            "gate_total_s": round(timing_acc["gate_s"], 3),
                            "serie_total_s": round(time.perf_counter() - t_series_start, 3),
                            "per_round": [r_["timing"] for r_ in rounds_info],
                        },
                        "rounds": rounds_info,
                        "final_gate": {"scores": scored, "inputs": final_gate_inputs},
                    }, fh, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass

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
                "ablation_config": f"a2_{version}_{combinator_model.replace(':','-')}_r{rounds}x{per_round}{'_consensus' if consensus else ''}",
            })
            per_series.append(metrics)
            if floor:
                floor_rows.append(floor)
            ok += 1
            print(
                f"[{idx:>4}] RESULTADO    {final_spec['combine']:<14} origin={final_origin:<14} "
                f"| TEST smape={metrics['smape']:.4f} | FLOOR smape={floor['smape'] if floor else float('nan'):.4f}"
            )
            print(
                f"[{idx:>4}] TEMPOS: ingest={t_fase0:.1f}s sementes={t_p2:.1f}s "
                f"| LLM={timing_acc['llm_s']:.1f}s gate={timing_acc['gate_s']:.2f}s "
                f"| série={time.perf_counter() - t_series_start:.1f}s",
                flush=True,
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
    p.add_argument("--gate-mode", choices=["logistic", "laya"], default="logistic",
                   help="logistic = gate logístico por-janela (default, sem GPU); "
                        "laya = checkpoint fine-tuned")
    p.add_argument("--gate-data", default="JEV/data/gate_dataset.jsonl")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--per-round", type=int, default=4)
    p.add_argument("--consensus", action="store_true",
                   help="final = entre os top-3 por P do gate, o mais próximo "
                        "do CONSENSO das previsões de teste (entradas, sem atual)")
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
                gate_mode=args.gate_mode,
                gate_data=args.gate_data,
                rounds=args.rounds, per_round=args.per_round,
                dataset_card=not args.no_dataset_card,
                consensus=args.consensus,
                indices=args.indices,
                source_dir=args.source_dir, results_dir=args.results_dir,
            )
        except Exception as exc:
            print(f"\nDATASET {dataset} ABORTOU: {type(exc).__name__}: {exc}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
