#!/usr/bin/env python3
"""D3 — ROTEAMENTO POR TURNO (Switchcraft): o LAYA decide se o turno é ROTINA
(escolhe a ação da shortlist) ou EXPLORAÇÃO (o gpt-oss propõe).

Estrutura por turno:

  1. ROTEADOR (LAY A zero-shot, 1 pergunta noul): "a melhor próxima ação já é
     evidente pelo estado (rotina) ou precisa de exploração nova?"
  2a. ROTINA: o LAYA escolhe a chamada concreta da SHORTLIST determinística
      (mesmo retriever do D1 — tool retrieval) e ela executa com a cola
      (peso→evaluate; seleção/poda→mediana do pool novo→evaluate).
  2b. EXPLORAÇÃO: o gpt-oss propõe a ação (ReAct padrão, com o veredito do
      gate no prompt) e ela executa.
  3. O gate (LAY A zero-shot por janela) reavalia o histórico por turno.
  4. FINAL: argmin (melhor score de validação). Gate fica como dado.

Custo: rotina = 1 passada do LAYA (sem LLM); exploração = 1 chamada do gpt-oss.
Sem prompt-crutch: nenhum prompt menciona piso/baseline.

Referências: Switchcraft (Microsoft 2026), "How Many Tools Should an LLM
Agent See?" (arXiv 2605.24660), Agent-as-a-Router (arXiv 2606.22902).
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
from orchestrator_react.react_loop import (  # noqa: E402
    TERMINAL_ACTION,
    _read_accept,
    call_tool,
    parse_agent_step,
)
from orchestrator_react.registry import withheld_tools  # noqa: E402
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

from d1_plan_verify import (  # noqa: E402
    STAGNATION_STOP,
    build_shortlist,
    execute_with_glue,
    _pick_entry,
)

COLS: List[str] = CORE_COLUMNS + [
    "description", "origin", "react_iterations_used", "react_stop_reason",
    "seed_floor_smape", "seed_floor_rmse", "ablation_config",
]

ROUTINE_THRESHOLD = 0.5
FULL_POOL = "pool1"


# ─────────────────────────── UM TURNO DO GPT-OSS ─────────────────────────────


def _gptoss_turn(
    state: Any,
    client: Any,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    system: str,
    scratchpad: List[Dict[str, Any]],
    iteration: int,
    max_iterations: int,
    verdict: Optional[Dict[str, Any]],
    config: ReactConfig,
    withheld: Dict[str, str],
) -> Dict[str, Any]:
    """Um turno de exploração do gpt-oss: prompt → parse → executa. Sem loop
    próprio — o ROTEADOR do D3 é o loop."""
    user = PR.build_turn_prompt(
        state=state, series_card=series_card, pool_card=pool_card,
        scratchpad=scratchpad, iteration=iteration, max_iterations=max_iterations,
        last_observation=None, show_history=config.show_attempt_history,
        show_rationales=config.show_attempt_rationales, diagnosis=None,
        prompt_format=config.prompt_format, gate_verdict=verdict,
    )
    t0 = time.perf_counter()
    raw = client.complete(system, user)
    llm_s = time.perf_counter() - t0
    step = parse_agent_step(raw)
    rec: Dict[str, Any] = {
        "branch": "exploration", "iteration": iteration,
        "system_prompt": system, "user_prompt": user,
        "raw": raw, "llm_call_s": round(llm_s, 4),
        "thought": (step.thought or "")[:400],
    }
    if not step.ok:
        rec["outcome"] = f"parse error: {step.parse_error}"
        rec["action"] = step.action or "unparsed"
        return rec

    rec["action"] = step.action
    rec["action_args"] = step.action_input

    if step.action.strip().lower() == TERMINAL_ACTION:
        accepted, _conf, justification, problem = _read_accept(state, step)
        if problem:
            rec["outcome"] = f"invalid accept: {problem}"
            return rec
        rec["outcome"] = "accepted"
        rec["accepted_id"] = accepted.attempt_id
        rec["justification"] = justification
        return rec

    args = dict(step.action_input)
    if step.action == "evaluate_strategy":
        args.setdefault("rationale", step.thought or "")
        args["iteration"] = iteration
    ok, obs = call_tool(state, step.action, args, withheld=withheld)
    rec["outcome"] = "ok" if ok else f"tool error: {str(obs.get('detail',''))[:120]}"
    rec["observation"] = obs if isinstance(obs, dict) else {"value": str(obs)}
    if ok and "already_tested" in obs and obs.get("already_tested"):
        rec["outcome"] = "already tested"
    return rec


# ─────────────────────────── LOOP DO D3 ───────────────────────────────────────


def run_d3_loop(
    state: Any,
    gate_agent: LayaAgent,
    client: Any,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    dataset_card: Optional[Dict[str, Any]],
    max_iterations: int,
    config: ReactConfig,
    withheld: Dict[str, str],
    gate_pass: Any,
) -> Dict[str, Any]:
    """O roteador decide o turno: rotina (LAY A escolhe da shortlist) ou
    exploração (gpt-oss propõe). Paradas estruturais (sem prompt-crutch):
    shortlist exaurida → aceita; 3 turnos de proposta sem novo líder → aceita."""
    scratchpad: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    executed: set = set()
    stop_reason = "iteration_budget_exhausted"
    best_prev = state.best_attempt()
    best_seen = float(best_prev.score) if best_prev is not None else float("inf")
    best_id = best_prev.attempt_id if best_prev is not None else None
    stale = 0
    system = PR.build_system_prompt(
        include_history_rules=config.show_attempt_history,
        withheld_tools=withheld, prompt_format=config.prompt_format,
        reorder_weight_tools=config.reorder_weight_tools,
    )

    def _after_execution(action: str, kind: Optional[str]) -> None:
        nonlocal best_seen, best_id, stale
        is_proposal = kind in ("weight", "combine", "select", "prune") or (
            kind is None and action.startswith(("weights_", "combine_", "select_", "prune_", "evaluate_"))
        )
        if not is_proposal:
            return
        now_best = state.best_attempt()
        if (now_best is not None and now_best.attempt_id != best_id
                and float(now_best.score) < best_seen - 1e-12):
            best_seen = float(now_best.score)
            best_id = now_best.attempt_id
            stale = 0
        else:
            stale += 1

    for turn in range(1, max_iterations + 1):
        verdict = gate_pass()  # veredito fresco do gate para o turno
        stext = build_state_text(
            series_card, pool_card, state, budget=6000,
            scratchpad=scratchpad, dataset_card=dataset_card,
        )
        router_question = {
            "routine": {
                "type": "noul",
                "instructions": (
                    "Given the validation landscape and the history, is the "
                    "best next action already evident from the state (a "
                    "routine decision), or does it need fresh exploration?"
                ),
            },
        }
        try:
            out = gate_agent.predict(stext, router_question)
            routine_score = float(out["answers"]["routine"].get("noul", 0.5))
            router_answer = dict(out.get("answers") or {})
        except Exception as exc:
            print(f"      roteador erro ({type(exc).__name__}) → exploração")
            routine_score = 0.0
            router_answer = {"error": f"{type(exc).__name__}: {exc}"}
        routine = routine_score >= ROUTINE_THRESHOLD

        if routine:
            shortlist = build_shortlist(state, series_card, pool_card, withheld,
                                        executed=executed)
            if not any(e["kind"] in ("weight", "combine", "select", "prune")
                       for e in shortlist):
                turns.append({"branch": "routine", "iteration": turn,
                              "outcome": "shortlist exhausted → accept"})
                stop_reason = "shortlist_exhausted"
                break
            try:
                call_question = {
                    "call": {
                        "type": "choice",
                        "instructions": "Which concrete tool call from this shortlist should be executed now?",
                        "criteria": {e["label"]: e["desc"] for e in shortlist},
                    },
                }
                out2 = gate_agent.predict(stext, call_question)
                choice = str(out2["answers"]["call"].get("choice", ""))
                call_answer = dict(out2.get("answers") or {})
            except Exception as exc:
                print(f"      roteador erro ({type(exc).__name__}) → accept")
                choice = ""
                call_question = {"call": {"error": f"{type(exc).__name__}: {exc}"}}
                call_answer = {}
            entry = _pick_entry(shortlist, choice)
            if entry is None:
                entry = next((e for e in shortlist if e["action"] == "accept"), None)
            rec: Dict[str, Any] = {
                "branch": "routine", "iteration": turn,
                "routine_score": round(routine_score, 3),
                "chosen": entry["label"] if entry else None,
                "state_text": stext,
                "router_question": router_question,
                "router_answer": router_answer,
                "call_question": call_question,
                "call_answer": call_answer,
            }
            if entry is None or entry["action"] == "accept":
                rec["outcome"] = "accepted"
                turns.append(rec)
                stop_reason = "agent_accepted"
                break
            ok, obs = execute_with_glue(
                state, entry["action"], entry["args"], turn, withheld,
            )
            executed.add(f"{entry['action']}|{json.dumps(entry['args'], sort_keys=True, default=str)}")
            rec["outcome"] = "ok" if ok else f"tool error: {str(obs.get('detail',''))[:120]}"
            rec["observation"] = obs if isinstance(obs, dict) else {"value": str(obs)}
            if ok:
                _after_execution(entry["action"], entry["kind"])
            scratchpad.append({
                "iteration": turn, "branch": "routine",
                "action": entry["label"],
                "action_args": entry["args"],
                "observation_summary": rec["outcome"],
            })
            turns.append(rec)
            if stale >= STAGNATION_STOP:
                stop_reason = "stagnation"
                break
            continue

        rec = _gptoss_turn(
            state=state, client=client, series_card=series_card,
            pool_card=pool_card, system=system, scratchpad=scratchpad,
            iteration=turn, max_iterations=max_iterations, verdict=verdict,
            config=config, withheld=withheld,
        )
        rec["branch"] = "exploration"
        rec["state_text"] = stext
        rec["router_question"] = router_question
        rec["router_answer"] = router_answer
        rec["routine_score"] = round(routine_score, 3)
        if rec.get("outcome") == "ok":
            _after_execution(rec.get("action", ""), None)
        scratchpad.append({
            "iteration": turn, "branch": "exploration",
            "action": rec.get("action", "?"),
            "action_args": rec.get("action_args") or {},
            "observation_summary": rec.get("outcome", "?")[:140],
        })
        turns.append(rec)
        if rec.get("outcome") == "accepted":
            stop_reason = "agent_accepted"
            break
        if stale >= STAGNATION_STOP:
            stop_reason = "stagnation"
            break

    return {
        "turns": turns,
        "scratchpad": scratchpad,
        "stop_reason": stop_reason,
        "iterations_used": len(turns),
        "n_routine": sum(1 for t in turns if t.get("branch") == "routine"),
        "n_exploration": sum(1 for t in turns if t.get("branch") == "exploration"),
    }


# ─────────────────────────── PLUMBING (dataset → CSV) ────────────────────────


def run_dataset(
    dataset: str,
    source_file: str,
    version: str,
    combinator_model: str = "gpt-oss:20b",
    reasoning: Optional[str] = "low",
    max_iterations: int = 12,
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
    experiment = f"orchestrator_d3_{version}"
    out_dir = os.path.join(results_dir, experiment)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    combinator = LLMRole(model=combinator_model, temperature=0.2, seed=7)
    combinator.reasoning = reasoning
    client = build_client(combinator)
    gate_agent = LayaAgent(checkpoint="multilingual", max_len=8192)

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
    print("proposta   : D3 roteamento por turno (Switchcraft) — LAYA roteia rotina/exploração")
    print(f"llm        : {combinator_model} reasoning={reasoning} (só nos turnos de exploração)")
    print(f"roteador   : LAYA zero-shot, limiar {ROUTINE_THRESHOLD} | gate = LAYA zero-shot por janela")
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

            seed_leader = state.best_attempt()
            print(
                f"[{idx:>4}] INÍCIO: melhor semente = "
                f"{seed_leader.brief(include_rationale=False)['strategy']} "
                f"(score {float(seed_leader.score):.4f})", flush=True,
            )

            withheld = withheld_tools(cfg, state.n_windows, state=state)
            last_verdict: Dict[str, Any] = {}
            gate_turns: List[Dict[str, Any]] = []

            def gate_pass():
                nonlocal last_verdict
                scored, gate_inputs = run_gate_pass_windows(
                    state, gate_agent, series_card, pool_card,
                )
                last_verdict = build_verdict(scored)
                gate_turns.append({"scores": scored, "inputs": gate_inputs,
                                   "verdict": last_verdict})
                top = sorted(scored, key=lambda s: -s["p_mean"])[:3]
                for s in top:
                    print(
                        f"[{idx:>4}]     GATE {s['id']:<4} {s['strategy'][:38]:<38} "
                        f"P={s['p_windows']} mean={s['p_mean']} ({s['origin']})",
                        flush=True,
                    )
                return last_verdict

            loop = run_d3_loop(
                state=state, gate_agent=gate_agent, client=client,
                series_card=series_card, pool_card=pool_card, dataset_card=card,
                max_iterations=max_iterations, config=cfg, withheld=withheld,
                gate_pass=gate_pass,
            )
            for t in loop["turns"]:
                print(
                    f"[{idx:>4}] turn {t['iteration']:>2} | {t.get('branch','?'):<11} "
                    f"→ {t.get('chosen') or t.get('action','?')} [{t.get('outcome','?')[:50]}]",
                    flush=True,
                )

            # ── final: argmin (seleção de referência); gate vira dado ────────
            scored, final_gate_inputs = run_gate_pass_windows(
                state, gate_agent, series_card, pool_card,
            )
            best_g = max(scored, key=lambda s: s["p_mean"])
            argmin_g = min(scored, key=lambda s: s["score_val"])
            final_spec = argmin_g["spec"]
            final_origin = "argmin"
            forecast, _ = state.apply_to_test(final_spec)
            metrics = compute_metrics(forecast, ing.test_values)
            floor = _seed_floor_metrics(state, ing.test_values)

            print(
                f"[{idx:>4}] FINAL (argmin): {argmin_g['id']} "
                f"{argmin_g['strategy'][:38]} (score {argmin_g['score_val']:.4f})",
                flush=True,
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
                        "proposal": "d3_router",
                        "dataset": dataset, "series": int(idx),
                        "config": {"combinator": combinator_model, "reasoning": reasoning,
                                   "max_iterations": max_iterations,
                                   "routine_threshold": ROUTINE_THRESHOLD,
                                   "checkpoint": "multilingual"},
                        "seed_leader": seed_leader.brief(include_rationale=False),
                        "floor": floor,
                        "loop": loop,
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
                    "proposal": "d3_router", "strategy": final_spec,
                    "origin": final_origin, "loop": loop,
                    "gate_scores": scored, "dataset_card": card,
                }, ensure_ascii=False, default=str),
                "origin": final_origin,
                "react_iterations_used": loop["iterations_used"],
                "react_stop_reason": loop["stop_reason"],
                "seed_floor_smape": floor["smape"] if floor else None,
                "seed_floor_rmse": floor["rmse"] if floor else None,
                "ablation_config": f"d3_{version}_{combinator_model.replace(':','-')}_i{max_iterations}",
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
                "ablation_config": f"d3_{version}",
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
        print(f"  D3 run     smape={summary['smape']:.4f}  rmse={summary['rmse']:.4f}"
              f"  pocid={summary['pocid']:.2f}  mape={summary['mape']:.4f}")
        if floor_sum:
            note = ("  <- D3 better" if summary["rmse"] < floor_sum["rmse"]
                    else "  <- D3 WORSE than the seed floor")
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
    p = argparse.ArgumentParser(description="D3: roteamento por turno (Switchcraft) — LAYA roteia, gpt-oss explora.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--combinator", default="gpt-oss:20b")
    p.add_argument("--reasoning", default="low")
    p.add_argument("--max-iterations", type=int, default=12)
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
                max_iterations=args.max_iterations,
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
