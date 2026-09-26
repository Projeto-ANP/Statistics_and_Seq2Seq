#!/usr/bin/env python3
"""D2 — VERIFICADOR POR ETAPA (VEGAS): o gpt-oss continua o ReAct; o LAYA
verifica cada ação ANTES da execução ("Think Twice, Act Once").

Estrutura por turno:

  1. O gpt-oss propõe a ação (Thought/Action/Action Input), como no ReAct atual.
  2. VERIFICADOR (LAY A zero-shot, 1 pergunta noul): "esta ação vai plausível-
     mente ajudar a achar uma estratégia melhor que a atual?" — a ação executa
     só se o score ≥ 0.5; senão, a REJEIÇÃO vira a observation do turno
     (o agente lê por que foi barrado e repropõe no turno seguinte).
  3. O gate (LAY A zero-shot por janela) reavalia o histórico e devolve o
     veredito do turno seguinte (como no A2).
  4. FINAL: argmin (melhor score de validação). Gate fica como dado.

O verificador NÃO vê accept (a decisão terminal é da seleção final) e se
desliga após 3 rejeições totais (não pode queimar o orçamento do loop).

Sem prompt-crutch: nenhum prompt menciona piso/baseline.

Referências: VEGAS (arXiv 2605.12620), ToolVerifier (Meta, EMNLP 2024),
Agent-as-a-Router (arXiv 2606.22902).
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
from laya_loop import (  # noqa: E402
    LayaAgent,
    build_state_text,
    build_verdict,
    run_gate_pass_windows,
)
from run_laya import (  # noqa: E402
    DATASET_SOURCES,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SOURCE_DIR,
    _seed_floor_metrics,
    _summarise,
)

COLS: List[str] = CORE_COLUMNS + [
    "description", "origin", "react_iterations_used", "react_stop_reason",
    "seed_floor_smape", "seed_floor_rmse", "ablation_config",
]

VERIFY_THRESHOLD = 0.5
MAX_REJECTIONS = 3  # depois disso o verificador se desliga (não queima o loop)


def run_dataset(
    dataset: str,
    source_file: str,
    version: str,
    combinator_model: str = "gpt-oss:20b",
    reasoning: Optional[str] = "low",
    max_iterations: int = 12,
    early_stop_patience: int = 4,
    dataset_card: bool = True,
    indices: Optional[List[int]] = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    models = [m for m in DEFAULT_MODELS if os.path.exists(I.model_csv_path(m, dataset, results_dir))]
    frames = I.load_dataset_frames(models, dataset, results_dir)
    n_series = I.count_series(dataset, models[0], results_dir)
    try:
        bad = I.find_misaligned_models(models, dataset, 0, results_dir=results_dir,
                                       frames=frames, n_windows=3)
    except Exception:
        bad = {}
    models_eff = [m for m in models if m not in bad]
    if bad:
        print(f"WARNING: {len(bad)} modelos desalinhados descartados: {sorted(bad)}", flush=True)

    cfg = ReactConfig()
    experiment = f"orchestrator_d2_{version}"
    out_dir = os.path.join(results_dir, experiment)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    combinator = LLMRole(model=combinator_model, temperature=0.2, seed=7)
    combinator.reasoning = reasoning
    client = build_client(combinator)
    gate_agent = LayaAgent(checkpoint="multilingual", max_len=8192)

    def gate_pass(state, series_card, pool_card):
        return run_gate_pass_windows(state, gate_agent, series_card, pool_card)

    todo = indices if indices is not None else list(range(n_series))
    priors: Dict[int, Dict[str, float]] = {}
    if dataset_card:
        try:
            source = load_series_source(source_file, n_expected_series=n_series,
                                        source_dir=source_dir)
            priors = PL._build_strategy_priors(models_eff, dataset, todo, cfg,
                                               source, results_dir, frames, {})
        except Exception as exc:
            print(f"WARNING: dataset card falhou ({exc})", flush=True)

    print(f"dataset    : {dataset}")
    print(f"proposta   : D2 verificador por etapa (VEGAS) — gpt-oss ReAct + LAYA verifica cada ação")
    print(f"llm        : {combinator_model} reasoning={reasoning} | budget={max_iterations} iters, patience {early_stop_patience}")
    print(f"gate       : LAYA zero-shot por janela (veredito por turno + seleção final)")
    print("verifier   : LAYA zero-shot, limiar "
          f"{VERIFY_THRESHOLD}, desliga após {MAX_REJECTIONS} rejeições")
    print(f"writing to : {out_dir}")
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
            ing = I.load_series(models=models_eff, dataset=dataset, dataset_index=idx,
                                config=cfg, results_dir=results_dir,
                                source_file=source_file, source_dir=source_dir,
                                frames=frames)
            state = ing.state
            POOL.run_phase2(state, cfg)
            series_card = T.series_profile(state)
            pool_card = POOL.pool_report(state)
            if priors.get(idx):
                state.strategy_prior = priors[idx]
            card = PR.build_dataset_card(state) if priors.get(idx) else None

            round_cfg = ReactConfig(max_iterations=max_iterations,
                                    early_stop_patience=early_stop_patience)
            react_results = []
            gate_turns: List[Dict[str, Any]] = []
            last_verdict: Dict[str, Any] = {}
            n_rejections = {"value": 0}
            verify_log: List[Dict[str, Any]] = []

            seed_leader = state.best_attempt()
            print(
                f"[{idx:>4}] INÍCIO: melhor semente = "
                f"{seed_leader.brief(include_rationale=False)['strategy']} "
                f"(score {float(seed_leader.score):.4f})", flush=True,
            )

            def _on_step(_i, entry):
                args = json.dumps(entry.get("action_args") or {}, ensure_ascii=False,
                                  default=str)[:90]
                thought = " ".join(str(entry.get("thought") or "").split())[:120]
                print(f"[{idx:>4}] iter {entry['iteration']} | {entry.get('action')} {args}",
                      flush=True)
                if thought:
                    print(f"[{idx:>4}]     think: {thought}", flush=True)
                if "VERIFIER" in str(entry.get("observation_summary") or ""):
                    print(f"[{idx:>4}]     {entry['observation_summary']}", flush=True)

            def _gate_verdict_fn():
                nonlocal last_verdict
                scored, gate_inputs = gate_pass(state, series_card, pool_card)
                last_verdict = build_verdict(scored)
                gate_turns.append({"scores": scored, "inputs": gate_inputs,
                                   "verdict": last_verdict})
                top = sorted(scored, key=lambda s: -s["p_mean"])[:4]
                for s in top:
                    print(
                        f"[{idx:>4}]     GATE {s['id']:<4} {s['strategy'][:38]:<38} "
                        f"P={s['p_windows']} mean={s['p_mean']} ({s['origin']})",
                        flush=True,
                    )
                return last_verdict

            def _verify(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                # VEGAS: verifica a ação ANTES da execução. Não verifica accept.
                if n_rejections["value"] >= MAX_REJECTIONS:
                    return None
                action, args = entry["action"], entry["action_args"]
                stext = build_state_text(series_card, pool_card, state, budget=6000)
                stext += (
                    f"\nPROPOSED ACTION: {action} "
                    f"{json.dumps(args, sort_keys=True, default=str)}"
                )
                question = {
                    "verify": {
                        "type": "noul",
                        "instructions": (
                            "Will executing this proposed action plausibly help "
                            "find a strategy better than the current best? "
                            "Answer no if the action is incoherent with the "
                            "state, clearly unpromising, or repeats an idea "
                            "already tried."
                        ),
                    },
                }
                t_v = time.perf_counter()
                try:
                    out = gate_agent.predict(stext, question)
                    score = float(out["answers"]["verify"].get("noul", 0.5))
                except Exception as exc:
                    print(f"[{idx:>4}]     verifier erro ({type(exc).__name__}) → permite",
                          flush=True)
                    verify_log.append({
                        "iteration": entry["iteration"], "action": action,
                        "args": args, "state_text": stext, "question": question,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    return None
                verify_log.append({
                    "iteration": entry["iteration"], "action": action,
                    "args": args, "state_text": stext, "question": question,
                    "answer": dict(out.get("answers") or {}),
                    "score": round(score, 4),
                    "verdict": "allow" if score >= VERIFY_THRESHOLD else "reject",
                    "verify_s": round(time.perf_counter() - t_v, 3),
                })
                if score >= VERIFY_THRESHOLD:
                    return None
                n_rejections["value"] += 1
                return {"rejected": f"verifier score {score:.2f} < {VERIFY_THRESHOLD}"}

            rr = run_react_loop(
                state=state, client=client, series_card=series_card,
                pool_card=pool_card, config=round_cfg, gate_verdict=_gate_verdict_fn,
                on_step=_on_step, pre_action_check=_verify,
            )
            react_results.append(rr.summary())
            rr_full = {
                "summary": rr.summary(),
                "prompts": rr.prompts,          # system+user+veredito de CADA turno
                "trajectory": rr.trajectory,    # pensamento/ação/obs por turno
                "step_details": rr.step_details,  # resposta crua + meta Ollama
                "errors": rr.errors,
                "rejections": rr.rejections,
                "rejection_details": rr.rejection_details,
            }
            print(
                f"[{idx:>4}] loop: iters={rr.iterations_used} stop={rr.stop_reason} "
                f"| rejeições do verificador={rr.rejections} | histórico={len(state.attempts)}",
                flush=True,
            )

            # ── final: argmin (seleção de referência); gate vira dado ────────
            # O gate zero-shot como juiz final foi medido e perde feio (v2/v3 e
            # D1 v1: escolheu dba→2.0). O final agora é o MELHOR score de
            # validação do histórico; os scores do gate ficam gravados.
            final_origin = "argmin"
            scored, final_gate_inputs = gate_pass(state, series_card, pool_card)
            best_g = max(scored, key=lambda s: s["p_mean"])
            argmin_g = min(scored, key=lambda s: s["score_val"])
            final_spec = argmin_g["spec"]
            forecast, _ = state.apply_to_test(final_spec)
            metrics = compute_metrics(forecast, ing.test_values)
            floor = _seed_floor_metrics(state, ing.test_values)

            print(
                f"[{idx:>4}] FINAL (argmin): {argmin_g['id']} {argmin_g['strategy'][:38]} "
                f"(score {argmin_g['score_val']:.4f})", flush=True,
            )
            print(
                f"[{idx:>4}]   gate diria  : {best_g['id']} {best_g['strategy'][:38]} "
                f"(P={best_g['p_mean']}, windows {best_g['p_windows']}, origin={best_g['origin']})"
                f"{'  <- gate discorda do argmin' if argmin_g['id'] != best_g['id'] else ''}",
                flush=True,
            )

            art_dir = os.path.join(out_dir, "llm_artifacts", dataset)
            os.makedirs(art_dir, exist_ok=True)
            try:
                with open(os.path.join(art_dir, f"dataset_{idx}.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump({
                        "proposal": "d2_verifier",
                        "dataset": dataset, "series": int(idx),
                        "config": {"combinator": combinator_model, "reasoning": reasoning,
                                   "max_iterations": max_iterations,
                                   "early_stop_patience": early_stop_patience,
                                   "verifier_threshold": VERIFY_THRESHOLD,
                                   "verifier_max_rejections": MAX_REJECTIONS,
                                   "checkpoint": "multilingual"},
                        "seed_leader": seed_leader.brief(include_rationale=False),
                        "floor": floor,
                        "loop": rr_full,
                        "attempt_history": [
                            {
                                "id": a.attempt_id, "spec": a.spec, "origin": a.origin,
                                "iteration": a.iteration, "rationale": a.rationale,
                                "score": round(float(a.score), 6),
                                "aggregate": {k: (round(v, 6) if isinstance(v, float) else v)
                                              for k, v in (a.aggregate or {}).items()},
                                "per_window": a.per_window,
                                "per_window_scores": [round(float(v), 6)
                                                     for v in a.per_window_scores],
                                "n_models": a.n_models,
                            }
                            for a in state.attempts
                        ],
                        "verifier": {"rejections": rr.rejections,
                                     "details": rr.rejection_details,
                                     "checks": verify_log},
                        "final": {"spec": final_spec, "origin": final_origin,
                                  "gate_choice": {"id": best_g["id"],
                                                  "p_mean": best_g["p_mean"],
                                                  "p_windows": best_g["p_windows"]},
                                  "argmin_choice": {"id": argmin_g["id"],
                                                    "score_val": argmin_g["score_val"]}},
                        "gate_turns": gate_turns,
                        "final_gate": {"scores": scored, "inputs": final_gate_inputs},
                        "metrics": {k: metrics[k] for k in ("smape", "rmse", "pocid", "mape")},
                        "timing": {"serie_total_s": round(time.perf_counter() - t_series_start, 3)},
                    }, fh, ensure_ascii=False, indent=2, default=str)
            except Exception as exc:
                print(f"[{idx:>4}] WARNING: artifact falhou: {type(exc).__name__}: {exc}",
                      flush=True)

            rows.append({
                "dataset_index": str(idx), "horizon": ing.horizon,
                "regressor": experiment, **metrics,
                "test": [list(ing.test_values)], "predictions": [list(forecast)],
                "start_test": str(ing.start_test), "final_test": str(ing.final_test),
                "description": json.dumps({
                    "proposal": "d2_verifier", "strategy": final_spec,
                    "origin": final_origin, "loop": react_results,
                    "gate_scores": scored, "dataset_card": card,
                    "verifier_rejections": rr.rejections,
                }, ensure_ascii=False, default=str),
                "origin": final_origin,
                "react_iterations_used": rr.iterations_used,
                "react_stop_reason": rr.stop_reason,
                "seed_floor_smape": floor["smape"] if floor else None,
                "seed_floor_rmse": floor["rmse"] if floor else None,
                "ablation_config": f"d2_{version}_{combinator_model.replace(':','-')}_i{max_iterations}p{early_stop_patience}",
            })
            per_series.append(metrics)
            if floor:
                floor_rows.append(floor)
            ok += 1
            pd.DataFrame([rows[-1]]).reindex(columns=COLS).to_csv(
                csv_path, sep=";", mode="a",
                header=not os.path.exists(csv_path), index=False,
            )
            print(
                f"[{idx:>4}] RESULTADO    {final_spec['combine']:<14} "
                f"origin={final_origin:<14} | TEST smape={metrics['smape']:.4f} | "
                f"FLOOR smape={floor['smape'] if floor else float('nan'):.4f}", flush=True,
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
                "ablation_config": f"d2_{version}",
            })
            print(f"[{idx:>4}] FAILED: {type(exc).__name__}: {exc}")
            pd.DataFrame([rows[-1]]).reindex(columns=COLS).to_csv(
                csv_path, sep=";", mode="a",
                header=not os.path.exists(csv_path), index=False,
            )

    elapsed = time.perf_counter() - started
    summary = _summarise(per_series)
    floor_sum = _summarise(floor_rows)
    print("-" * 74)
    print(f"done in {elapsed:.1f}s | ok: {ok} | failed: {failed}")
    if summary:
        print(f"DATASET SUMMARY over {len(per_series)} series (mean across series):")
        print(f"  D2 run     smape={summary['smape']:.4f}  rmse={summary['rmse']:.4f}"
              f"  pocid={summary['pocid']:.2f}  mape={summary['mape']:.4f}")
        if floor_sum:
            note = ("  <- D2 better" if summary["rmse"] < floor_sum["rmse"]
                    else "  <- D2 WORSE than the seed floor")
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
    p = argparse.ArgumentParser(description="D2: verificador por etapa (VEGAS) — gpt-oss ReAct + LAYA verifica.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--combinator", default="gpt-oss:20b")
    p.add_argument("--reasoning", default="low")
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
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
                max_iterations=args.max_iterations, early_stop_patience=args.patience,
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
