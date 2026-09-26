#!/usr/bin/env python3
"""D1 — PLANO + VERIFICAÇÃO (ReWOO x VEGAS): o LAYA É o agente ReAct.

A tese continua "ReAct sobre ações" — as ações são as MESMAS ferramentas do
catálogo. O que muda é QUEM decide a ação (o classificador System One, em uma
passada de 3 perguntas por turno) e a ESTRUTURA do turno:

  1. RETRIEVER (determinístico): monta a SHORTLIST de chamadas concretas de
     ferramenta que fazem sentido AGORA (tool retrieval — "How Many Tools
     Should an LLM Agent See?", arXiv 2605.24660): diagnósticas, seleção de
     pool, poda, pesos, combinações e accept. ~9 entradas, não 25.
  2. PLANO (ReWOO): o LAYA responde EM UMA passada: (q_move) que TIPO de
     movimento, (q_call) qual chamada concreta, (q_gain) "esta ação vai
     melhorar o melhor atual?" (verificação — VEGAS, "Think Twice, Act Once").
  3. EXECUÇÃO determinística com COLA: proposta de peso → evaluate automático;
     seleção/poda de pool → combinação mediana do novo pool → evaluate.
     (Avaliar não é decisão; é mecânica. Só a escolha da proposta é decisão.)
  4. FEEDBACK (Agent-as-a-Router, C-A-F): o resultado verificado da ação
     (score aninhado = recompensa determinística, RLVR) entra no contexto do
     turno seguinte via scratchpad.
  5. FINAL: gate zero-shot por janela sobre o histórico completo.

Sem prompt-crutch: nada aqui menciona piso/baseline; as sementes são apenas
mais candidatos no histórico.

Referências: ReWOO (Xu et al. 2023), VEGAS (arXiv 2605.12620),
Hansen-Lunde-James (2011, MCS), RLVR para tool-use (arXiv 2607.01465).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
from orchestrator_react.config import ReactConfig  # noqa: E402
from orchestrator_react.csv_writer import CORE_COLUMNS, compute_metrics  # noqa: E402
from orchestrator_react.data_source import load_series_source  # noqa: E402
from orchestrator_react.react_loop import call_tool  # noqa: E402
from orchestrator_react.registry import withheld_tools  # noqa: E402
from laya_loop import (  # noqa: E402
    LayaAgent,
    build_state_text,
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

FULL_POOL = "pool1"
MAX_REJECTIONS = 2  # verificação negativa consecutiva → aceita o líder

# ─────────────────────────── RETRIEVER (tool retrieval determinístico) ───────


def _leader_pool(state: Any) -> str:
    best = state.best_attempt()
    if best is not None and best.spec.get("pool"):
        return str(best.spec["pool"])
    return FULL_POOL


def _weight_method_used(state: Any, method: str, pool: str) -> bool:
    used = {a.spec.get("weights") for a in state.attempts
            if a.spec.get("combine") == "weighted"}
    for handle, recipe in state.weights.items():
        if recipe.method == method and recipe.pool_handle == pool and handle in used:
            return True
    return False


def build_shortlist(
    state: Any,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    withheld: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Monta a shortlist de chamadas CONCRETAS de ferramenta (tool retrieval).

    Cada entrada é uma Action + Action Input prontos para executar — o espaço
    de decisão do classificador é pequeno e concreto (literatura de tool
    retrieval: mostrar poucas ferramentas relevantes, não o catálogo inteiro).
    """
    pool = _leader_pool(state)
    entries: List[Tuple[str, str, str, Dict[str, Any]]] = []  # (action, kind, desc, args)

    if "error_summary" not in withheld:
        entries.append(("error_summary", "diagnose",
                        "summarize which models err most, window by window",
                        {"metric": "smape"}))
    if "ranking_stability" not in withheld:
        entries.append(("ranking_stability", "diagnose",
                        "check if the model ranking is stable across windows",
                        {}))
    if "select_stable" not in withheld:
        entries.append(("select_stable", "select",
                        "build the pool of the 5 most ranking-stable models",
                        {"k": 5}))
    if "select_top_k" not in withheld:
        entries.append(("select_top_k", "select",
                        "build the pool of the 8 models with lowest validation error",
                        {"k": 8}))
    if "prune_redundant" not in withheld:
        entries.append(("prune_redundant", "prune",
                        f"prune near-duplicate models (corr>0.95) from {pool}",
                        {"pool": pool, "corr_threshold": 0.95}))
    for method, desc, extra in (
        ("weights_inverse_error", "weight each model by inverse validation error", {}),
        ("weights_softmax_neg_error", "weight each model by softmax of negative error", {"eta": 1.0}),
        ("weights_error_trend", "weight by whether error is falling across windows", {"eta": 1.0}),
    ):
        if method in withheld or _weight_method_used(state, method, pool):
            continue
        entries.append((method, "weight",
                        f"{desc} over {pool}", {"pool": pool, **extra}))
    if "weights_feature_based" not in withheld and not _weight_method_used(
        state, "feature_based", pool
    ):
        entries.append(("weights_feature_based", "weight",
                        f"learn weights from model/window features over {pool}",
                        {"pool": pool}))
    for combine, desc in (
        ("combine_median", "plain median (robust to one bad model)"),
        ("combine_trimmed_mean", "trimmed mean (drop 20% of each tail)"),
        ("combine_dba", "median after DBA barycentric averaging"),
    ):
        if combine in withheld:
            continue
        entries.append((combine, "combine", f"{desc} over {pool}", {"pool": pool}))
    entries.append(("accept", "accept",
                    "accept the current best strategy and stop exploring", {}))

    shortlist: List[Dict[str, Any]] = []
    for i, (action, kind, desc, args) in enumerate(entries):
        shortlist.append({
            "id": f"e{i}",
            "action": action,
            "kind": kind,
            "desc": desc,
            "args": args,
            "label": f"e{i} {action}",
        })
    return shortlist


# ─────────────────────────── EXECUÇÃO COM COLA ───────────────────────────────


def execute_with_glue(
    state: Any, action: str, args: Dict[str, Any], iteration: int,
    withheld: Dict[str, str],
) -> Tuple[bool, Dict[str, Any]]:
    """Executa a chamada e cola a avaliação mecânica quando o resultado é uma
    RECEITA e não uma estratégia pontuada (avaliar não é decisão)."""
    ok, obs = call_tool(state, action, args, withheld=withheld)
    if not ok:
        return ok, obs
    if action.startswith("weights_"):
        return call_tool(
            state, "evaluate_strategy",
            {"combine": "weighted", "pool": obs.get("pool"),
             "weights": obs.get("weights"),
             "rationale": f"auto-evaluation after {action}", "iteration": iteration},
            withheld=withheld,
        )
    if action in ("select_stable", "select_top_k", "prune_redundant"):
        ok2, obs2 = call_tool(state, "combine_median", {"pool": obs.get("pool")},
                              withheld=withheld)
        if not ok2:
            return ok2, obs2
        nxt = dict(obs2.get("next_action_input") or {})
        nxt.setdefault("rationale", f"auto-evaluation after {action}")
        return call_tool(state, "evaluate_strategy", nxt, withheld=withheld)
    if action.startswith("combine_"):
        nxt = dict(obs.get("next_action_input") or {})
        nxt.setdefault("rationale", f"auto-evaluation after {action}")
        return call_tool(state, "evaluate_strategy", nxt, withheld=withheld)
    return ok, obs


def _pick_entry(shortlist: List[Dict[str, Any]], choice: str) -> Optional[Dict[str, Any]]:
    if not choice:
        return None
    choice = choice.strip()
    for e in shortlist:
        if choice == e["label"]:
            return e
    for e in shortlist:
        if choice.split()[0] == e["id"]:
            return e
    for e in shortlist:
        if e["action"] in choice:
            return e
    return None


# ─────────────────────────── LOOP DO D1 ───────────────────────────────────────


def run_d1_loop(
    state: Any,
    gate_agent: LayaAgent,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    dataset_card: Optional[Dict[str, Any]],
    max_iterations: int,
    withheld: Dict[str, str],
) -> Dict[str, Any]:
    """Um turno = 1 passada do LAYA (plano: move+call+verificação) → execução.

    Devolve o diário completo do loop (turns, respostas, execuções, aceite).
    """
    scratchpad: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    no_gain_streak = 0
    stop_reason = "iteration_budget_exhausted"
    best_seen = float("inf")
    best_prev = state.best_attempt()
    if best_prev is not None:
        best_seen = float(best_prev.score)

    for turn in range(1, max_iterations + 1):
        shortlist = build_shortlist(state, series_card, pool_card, withheld)
        stext = build_state_text(
            series_card, pool_card, state, budget=7000,
            scratchpad=scratchpad, dataset_card=dataset_card,
        )
        kinds: Dict[str, str] = {}
        for e in shortlist:
            kinds.setdefault(e["kind"], e["kind"])
        question = {
            "move": {
                "type": "choice",
                "instructions": "Given the series and the current history, what kind of move should be made now?",
                "criteria": {
                    k: f"make a {k} move" for k in kinds
                },
            },
            "call": {
                "type": "choice",
                "instructions": "Which concrete tool call from this shortlist should be executed now?",
                "criteria": {e["label"]: e["desc"] for e in shortlist},
            },
            "gain": {
                "type": "noul",
                "instructions": (
                    "Will executing this action plausibly improve on the best "
                    "strategy found so far? Answer no if it repeats an idea "
                    "already tried without new information."
                ),
            },
        }
        try:
            out = gate_agent.predict(stext, question)
            answers = dict(out.get("answers") or {})
        except Exception as exc:
            turns.append({"turn": turn, "error": f"{type(exc).__name__}: {exc}"})
            break
        move = str((answers.get("move") or {}).get("choice") or "")
        call_choice = str((answers.get("call") or {}).get("choice") or "")
        gain = float((answers.get("gain") or {}).get("noul", 0.5))
        entry = _pick_entry(shortlist, call_choice)

        rec: Dict[str, Any] = {
            "turn": turn, "move": move, "call_choice": call_choice,
            "gain": round(gain, 3),
        }
        if entry is None:
            entry = next((e for e in shortlist if e["action"] == "accept"), None)
            rec["fallback"] = "unrecognized choice → accept"
        rec["chosen"] = entry["label"] if entry else None

        if entry is None or entry["action"] == "accept" or move.strip() == "accept":
            stop_reason = "agent_accepted"
            rec["outcome"] = "accepted"
            turns.append(rec)
            break

        # ── verificação (VEGAS): a ação só executa se o gate achar que ajuda ──
        if gain < 0.5:
            no_gain_streak += 1
            rec["outcome"] = f"verifier skipped (gain={gain:.2f})"
            scratchpad.append({
                "turn": turn, "action": entry["label"],
                "result": f"skipped by the verifier (gain={gain:.2f})",
            })
            turns.append(rec)
            if no_gain_streak >= MAX_REJECTIONS:
                stop_reason = "verifier_forced_accept"
                break
            continue
        no_gain_streak = 0

        ok, obs = execute_with_glue(
            state, entry["action"], entry["args"], turn, withheld,
        )
        rec["outcome"] = "ok" if ok else f"tool error: {obs.get('detail','')[:120]}"
        summary = ""
        if ok:
            summary = _obs_summary(state, obs)
        scratchpad.append({
            "turn": turn, "action": entry["label"],
            "result": summary or str(obs.get("detail", ""))[:160],
        })
        turns.append(rec)

        now_best = state.best_attempt()
        if now_best is not None and float(now_best.score) < best_seen - 1e-12:
            best_seen = float(now_best.score)
        if ok and entry["action"] == "evaluate_strategy":
            pass  # melhoria é verificada pelo score; sem early-stop extra

    return {
        "turns": turns,
        "scratchpad": scratchpad,
        "stop_reason": stop_reason,
        "iterations_used": len(turns),
        "best_score": round(best_seen, 6),
    }


def _obs_summary(state: Any, obs: Dict[str, Any]) -> str:
    if "weights" in obs and obs.get("weights"):
        return f"registered {obs['weights']} ({obs.get('method','?')})"
    if "strategy" in obs:
        return f"strategy {json.dumps(obs['strategy'], sort_keys=True, default=str)[:90]}"
    if "already_tested" in obs and obs.get("already_tested"):
        return "already tested — no new entry"
    if "ranked" in obs or "score" in obs:
        b = state.best_attempt()
        if b is not None:
            return f"best is now {b.brief(include_rationale=False)['strategy']} score={float(b.score):.4f}"
    if "summary" in obs:
        return str(obs["summary"])[:160]
    return str(list(obs.keys()))[:120]


# ─────────────────────────── PLUMBING (dataset → CSV) ────────────────────────


def run_dataset(
    dataset: str,
    source_file: str,
    version: str,
    max_iterations: int = 10,
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
    experiment = f"orchestrator_d1_{version}"
    out_dir = os.path.join(results_dir, experiment)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

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
    print(f"proposta   : D1 plano+verificação (ReWOO x VEGAS) — LAYA zero-shot, 1 passada/turno")
    print(f"budget     : {max_iterations} turnos | gate final = LAYA zero-shot por janela")
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
            loop = run_d1_loop(
                state=state, gate_agent=gate_agent, series_card=series_card,
                pool_card=pool_card, dataset_card=card,
                max_iterations=max_iterations, withheld=withheld,
            )
            for t in loop["turns"]:
                print(
                    f"[{idx:>4}] turn {t['turn']:>2} | move={t.get('move','?'):<9} "
                    f"gain={t.get('gain','?'):<5} → {t.get('chosen','?')} "
                    f"[{t.get('outcome','?')[:60]}]", flush=True,
                )

            # ── final: argmax P do gate zero-shot por janela ──────────────────
            scored, gate_inputs = run_gate_pass_windows(
                state, gate_agent, series_card, pool_card,
            )
            best_g = max(scored, key=lambda s: s["p_mean"])
            argmin_g = min(scored, key=lambda s: s["score_val"])
            final_spec = best_g["spec"]
            final_origin = "gate"
            if best_g["origin"] == "baseline":
                final_origin = "gate(baseline)"
            forecast, _ = state.apply_to_test(final_spec)
            metrics = compute_metrics(forecast, ing.test_values)
            floor = _seed_floor_metrics(state, ing.test_values)

            print(
                f"[{idx:>4}] ESCOLHA DO GATE: {best_g['id']} "
                f"{best_g['strategy'][:38]} (P={best_g['p_mean']}, "
                f"windows {best_g['p_windows']}, origin={best_g['origin']})", flush=True,
            )
            print(
                f"[{idx:>4}]   argmin seria : {argmin_g['id']} "
                f"{argmin_g['strategy'][:38]} (score {argmin_g['score_val']:.4f})"
                f"{'  <- gate discorda do argmin' if argmin_g['id'] != best_g['id'] else ''}",
                flush=True,
            )

            art_dir = os.path.join(out_dir, "llm_artifacts", dataset)
            os.makedirs(art_dir, exist_ok=True)
            try:
                with open(os.path.join(art_dir, f"dataset_{idx}.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump({
                        "proposal": "d1_plan_verify",
                        "dataset": dataset, "series": int(idx),
                        "config": {"max_iterations": max_iterations,
                                   "checkpoint": "multilingual"},
                        "seed_leader": seed_leader.brief(include_rationale=False),
                        "floor": floor,
                        "loop": loop,
                        "final": {"spec": final_spec, "origin": final_origin,
                                  "gate_choice": {"id": best_g["id"],
                                                  "p_mean": best_g["p_mean"],
                                                  "p_windows": best_g["p_windows"]},
                                  "argmin_choice": {"id": argmin_g["id"],
                                                    "score_val": argmin_g["score_val"]}},
                        "gate_scores": scored,
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
                    "proposal": "d1_plan_verify", "strategy": final_spec,
                    "origin": final_origin, "loop": loop,
                    "gate_scores": scored, "dataset_card": card,
                }, ensure_ascii=False, default=str),
                "origin": final_origin,
                "react_iterations_used": loop["iterations_used"],
                "react_stop_reason": loop["stop_reason"],
                "seed_floor_smape": floor["smape"] if floor else None,
                "seed_floor_rmse": floor["rmse"] if floor else None,
                "ablation_config": f"d1_{version}_i{max_iterations}",
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
                "ablation_config": f"d1_{version}",
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
        print(f"  D1 run     smape={summary['smape']:.4f}  rmse={summary['rmse']:.4f}"
              f"  pocid={summary['pocid']:.2f}  mape={summary['mape']:.4f}")
        if floor_sum:
            note = ("  <- D1 better" if summary["rmse"] < floor_sum["rmse"]
                    else "  <- D1 WORSE than the seed floor")
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
    p = argparse.ArgumentParser(description="D1: plano + verificação (ReWOO x VEGAS) com LAYA.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--max-iterations", type=int, default=10)
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
