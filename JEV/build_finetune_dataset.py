#!/usr/bin/env python3
"""Exporta o dataset de decisões rotuladas para o fine-tune do GATE.

Uma linha por estratégia avaliada (propostas do agente E sementes), com:
- `state`: texto RICO (cards + histórico + features numéricas + spec candidata)
- `label`: 1 se a estratégia bateu o piso das sementes NA JANELA DE TESTE
  (rótulo de DESFECHO, não de imitação)
- features tabulares para o baseline logístico (score_val, rank, margem_pct,
  n_attempts, tau, n_models, origin)

Fontes (todas replay determinístico, ZERO chamadas de LLM):
- gpt-oss v5: artifacts (trajectories) — 7 datasets
- laya_v0 / ml8192 / noseeds: CSVs com description.loop.trace

Anti-vazamento: o `dataset` e `series` de cada linha permitem o split
leave-one-dataset-out no treino/avaliação.

Uso:
  python JEV/build_finetune_dataset.py [--out JEV/data/gate_dataset.jsonl]
                                        [--source-dir ../forecasting_datasets]
                                        [--limit-series N] [--no-seed-examples]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "JEV"))
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from orchestrator_react import ingest as I  # noqa: E402
from orchestrator_react import metrics as M  # noqa: E402
from orchestrator_react import pool as POOL  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.config import ReactConfig  # noqa: E402
from orchestrator_react.registry import call_tool  # noqa: E402
from orchestrator_react.state import FULL_POOL  # noqa: E402
from laya_loop import build_state_text  # noqa: E402

BASE = "timeseries/mestrado/resultados"
DEFAULT_SOURCE_DIR = os.path.expanduser("~/forecasting_datasets")

TSF = {
    "ETTM1": "ETTM1.tsf", "ETTM2": "ETTM2.tsf",
    "ETTH1": "ETTH1.tsf", "ETTH2": "ETTH2.tsf",
    "ANP_MONTHLY": "mes_11_venda_mensal.tsf",
    "NN5_WEEKLY_DATASET": "nn5_weekly_dataset.tsf",
    "M4_WEEKLY_DATASET": "m4_weekly_dataset.tsf",
}
DATASETS = list(TSF)

MODELS = [
    "ARIMA", "ETS", "THETA", "rf", "catboost",
    "CWT_rf", "DWT_rf", "FT_rf",
    "CWT_catboost", "DWT_catboost", "FT_catboost",
    "ONLY_CWT_catboost", "ONLY_CWT_rf",
    "ONLY_DWT_catboost", "ONLY_DWT_rf",
    "ONLY_FT_catboost", "ONLY_FT_rf",
    "NaiveSeasonal", "NaiveMovingAverage",
]

SOURCES = [
    {"kind": "gpt_oss", "run": "orchestrator_react_v5",
     "datasets": DATASETS, "no_seeds": False},
    {"kind": "laya", "run": "orchestrator_laya_laya_v0",
     "datasets": DATASETS, "no_seeds": False},
    {"kind": "laya", "run": "orchestrator_laya_laya_v0_ml8192",
     "datasets": ["ETTM2", "NN5_WEEKLY_DATASET"], "no_seeds": False},
    {"kind": "laya", "run": "orchestrator_laya_laya_v0_noseeds",
     "datasets": DATASETS, "no_seeds": True},
]


def smape(preds, actual):
    p2 = np.asarray(preds, dtype=float).reshape(1, -1)
    t2 = np.asarray(actual, dtype=float).reshape(1, -1)
    return float(np.nanmean(2 * np.abs(p2 - t2) / (np.abs(p2) + np.abs(t2)), axis=1)[0])


def label_to_spec(label: str):
    if label.startswith("best_"):
        return {"combine": "best_single", "model": label[5:]}
    if label.startswith("trim_"):
        return {"combine": "trimmed_mean", "pool": label[5:], "trim_pct": 0.2}
    method, pool = label.split("_", 1)
    return {"combine": method, "pool": pool}


TOOL_LABELS = {
    "select_stable_k5": ("select_stable", {"k": 5}),
    "select_stable_k7": ("select_stable", {"k": 7}),
    "select_stable_k9": ("select_stable", {"k": 9}),
    "select_top_k_k5": ("select_top_k", {"k": 5}),
    "prune_redundant_full": ("prune_redundant", {"pool": FULL_POOL}),
}


class Emitter:
    """Acumula exemplos e grava JSONL, com dedupe."""

    def __init__(self, path: str, include_seeds: bool):
        self.path = path
        self.include_seeds = include_seeds
        self.rows: list[dict] = []
        self.seen: set = set()
        self.counts: dict = {}

    def emit(self, row: dict, key: tuple):
        if key in self.seen:
            return
        self.seen.add(key)
        self.rows.append(row)
        src = f"{row['source']}/{row['dataset']}"
        self.counts[src] = self.counts.get(src, 0) + 1

    def flush(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return len(self.rows)


def make_example(
    source: str, dataset: str, idx: int, st, series_card, pool_card,
    spec: dict, attempt, floor: float, ing, turn: int, origin: str,
    ref_best: float, ref_winner: str, references: dict,
) -> dict:
    """Uma linha do dataset. `st` é o estado NO MOMENTO da avaliação.

    Dois rótulos:
    - `label`     : 1 se bate a MELHOR referência (piso ou FFORMA/ADE/mean/
                    median/dba) — a meta real do trabalho.
    - `label_seed`: 1 se bate só o piso das sementes (ablação).
    """
    try:
        fc, _ = st.apply_to_test(spec)
        ts = smape(fc, ing.test_values)
    except Exception:
        return None
    ranked = st.ranked_attempts()
    rank = ranked.index(attempt) + 1 if attempt in ranked else None
    best = st.best_attempt()
    margem = None
    if best is not None and best is not attempt and np.isfinite(best.score):
        margem = round(float((attempt.score - best.score) / (abs(best.score) or 1.0)), 6)
    n_models = 1 if spec.get("combine") == "best_single" else len(st.get_pool(spec.get("pool") or FULL_POOL))
    tau = (pool_card.get("ranking_stability") or {}).get("mean_kendall_tau")
    state_text = build_state_text(series_card, pool_card, st, budget=8000)
    state_text += "\nCANDIDATE STRATEGY: " + json.dumps(spec, sort_keys=True, default=str)
    return {
        "source": source, "dataset": dataset, "series": int(idx),
        "state": state_text,
        "spec": spec,
        "score_val": round(float(attempt.score), 6) if np.isfinite(attempt.score) else None,
        "smape_test": round(ts, 6), "floor_smape_test": round(floor, 6),
        "label": int(ts < ref_best),
        "label_seed": int(ts < floor),
        "ref_best_smape": round(ref_best, 6),
        "ref_winner": ref_winner,
        "references": {k: round(v, 6) for k, v in references.items()},
        "origin": origin, "turn": int(turn),
        "rank": rank, "margem_pct": margem, "n_attempts": len(st.attempts),
        "tau": round(float(tau), 4) if tau is not None else None,
        "n_models": n_models,
    }


def series_references(dataset: str, idx: int, floor: float) -> tuple:
    """Melhor referência por série = min(piso, FFORMA, ADE, mean, median, dba)."""
    refs = {"seed_floor": floor}
    try:
        ext = I.read_external_baselines(dataset, idx, results_dir=BASE)
        for name, stats in ext.items():
            if isinstance(stats, dict) and stats.get("available") and stats.get("smape") is not None:
                refs[name] = float(stats["smape"])
    except Exception:
        pass
    winner = min(refs, key=lambda k: refs[k])
    return refs[winner], winner, refs


def _spec_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)


def _val_score(st, spec: dict) -> float:
    """Score ANINHADO de validação do spec, SEM tocar no histórico do estado.

    Igual ao `state.evaluate`, mas sem registrar a tentativa: backtest com
    seleção aninhada (LOO dentro das 3 janelas) + composite score normalizado
    contra a âncora (média do pool). É o MESMO número que o agente otimiza.
    """
    combined, _ = st.backtest(spec)
    agg, _ = st._score_metrics(combined)
    return float(M.composite_score(agg, st.baseline_aggregate(), st.config.score_weights()))


def evaluate_universe(st, ing) -> dict:
    """Universo de candidatos da série: tudo que o pipeline sabe produzir.

    = sementes (mean/median/dba/trimmed/estáveis, já no histórico) + best_single
    de CADA modelo do pool + median/mean/dba do pool completo (se faltar).

    Para cada candidato: `val_score` (score aninhado de VALIDAÇÃO — o único
    usado nos rótulos de treino) e `test_smape` (SÓ para a avaliação final).
    """
    out: dict = {}
    for a in st.attempts:
        key = _spec_key(a.spec)
        if key not in out:
            fc, _ = st.apply_to_test(a.spec)
            out[key] = {"val_score": float(a.score), "test_smape": smape(fc, ing.test_values)}
    for name in st.model_names:
        spec = {"combine": "best_single", "model": name}
        key = _spec_key(spec)
        if key not in out:
            try:
                fc, _ = st.apply_to_test(spec)
                out[key] = {"val_score": _val_score(st, spec),
                            "test_smape": smape(fc, ing.test_values)}
            except Exception:
                pass
    for method in ("mean", "median", "dba"):
        spec = {"combine": method, "pool": FULL_POOL}
        key = _spec_key(spec)
        if key not in out:
            try:
                fc, _ = st.apply_to_test(spec)
                out[key] = {"val_score": _val_score(st, spec),
                            "test_smape": smape(fc, ing.test_values)}
            except Exception:
                pass
    return out


def make_example(
    source: str, dataset: str, idx: int, st, series_card, pool_card,
    spec: dict, attempt, floor: float, floor_val: float, ing, turn: int,
    origin: str, ref_best: float, ref_winner: str, references: dict,
    universe: dict,
) -> dict:
    """Uma linha do dataset. `st` é o estado NO MOMENTO da avaliação.

    Rótulos DE TREINO (SÓ validação — o teste fica intocado até a avaliação):
    - `label_val`     : 1 se o candidato tem o MELHOR score aninhado de
                        validação do universo da série (ranqueador dinâmico).
    - `label_val_seed`: 1 se o candidato vence TODAS as sementes no score
                        aninhado de validação.

    Rótulos baseados no teste (guardados SÓ PARA ANÁLISE, nunca para treinar):
    - `label` / `label_seed` / `label_dyn`: mesmas perguntas medidas no teste.
    """
    try:
        fc, _ = st.apply_to_test(spec)
        ts = smape(fc, ing.test_values)
    except Exception:
        return None
    ranked = st.ranked_attempts()
    rank = ranked.index(attempt) + 1 if attempt in ranked else None
    best = st.best_attempt()
    margem = None
    if best is not None and best is not attempt and np.isfinite(best.score):
        margem = round(float((attempt.score - best.score) / (abs(best.score) or 1.0)), 6)
    n_models = 1 if spec.get("combine") == "best_single" else len(st.get_pool(spec.get("pool") or FULL_POOL))
    tau = (pool_card.get("ranking_stability") or {}).get("mean_kendall_tau")
    state_text = build_state_text(series_card, pool_card, st, budget=8000)
    state_text += "\nCANDIDATE STRATEGY: " + json.dumps(spec, sort_keys=True, default=str)
    val_score = float(attempt.score) if np.isfinite(attempt.score) else None
    best_universe_test = min(v["test_smape"] for v in universe.values()) if universe else float("inf")
    best_universe_val = min(v["val_score"] for v in universe.values() if v["val_score"] == v["val_score"]) if universe else float("inf")
    return {
        "source": source, "dataset": dataset, "series": int(idx),
        "state": state_text,
        "spec": spec,
        "score_val": round(val_score, 6) if val_score is not None else None,
        "smape_test": round(ts, 6), "floor_smape_test": round(floor, 6),
        # rótulos de treino (SÓ validação)
        "label_val": int(val_score is not None and val_score <= best_universe_val + 1e-9),
        "label_val_seed": int(val_score is not None and val_score < floor_val),
        "universe_best_val_score": round(best_universe_val, 6),
        "universe_size": len(universe),
        # rótulos de teste (SÓ análise)
        "label": int(ts < ref_best),
        "label_seed": int(ts < floor),
        "label_dyn": int(ts <= best_universe_test + 1e-9),
        "universe_best_smape": round(best_universe_test, 6),
        "ref_best_smape": round(ref_best, 6),
        "ref_winner": ref_winner,
        "references": {k: round(v, 6) for k, v in references.items()},
        "origin": origin, "turn": int(turn),
        "rank": rank, "margem_pct": margem, "n_attempts": len(st.attempts),
        "tau": round(float(tau), 4) if tau is not None else None,
        "n_models": n_models,
    }


def load_series_state(dataset, idx, source_file, models_eff, frames, cfg, source_dir):
    ing = I.load_series(
        models=models_eff, dataset=dataset, dataset_index=idx, config=cfg,
        results_dir=BASE, source_file=source_file, source_dir=source_dir, frames=frames,
    )
    st = ing.state
    phase2 = POOL.run_phase2(st, cfg)
    series_card = T.series_profile(st)
    pool_card = phase2["report"]
    seeds = [a for a in st.attempts if a.origin == "baseline"]
    floor = min(smape(st.apply_to_test(a.spec)[0], ing.test_values) for a in seeds)
    return ing, st, series_card, pool_card, seeds, floor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="JEV/data/gate_dataset.jsonl")
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--limit-series", type=int, default=None, help="smoke: N séries por dataset")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="filtrar datasets (default: todos os 7)")
    ap.add_argument("--no-seed-examples", action="store_true")
    ap.add_argument("--skip", nargs="+", default=None, help="fontes a pular (nomes dos runs)")
    args = ap.parse_args()

    cfg = ReactConfig()
    emitter = Emitter(args.out, include_seeds=not args.no_seed_examples)

    for src_cfg in SOURCES:
        run = src_cfg["run"]
        if args.skip and run in args.skip:
            continue
        kind, no_seeds = src_cfg["kind"], src_cfg["no_seeds"]
        for dataset in src_cfg["datasets"]:
            if args.datasets and dataset not in args.datasets:
                continue
            if args.skip and f"{run}/{dataset}" in args.skip:
                continue
            csv_path = f"{BASE}/{run}/{dataset}.csv"
            if kind == "gpt_oss":
                art_dir = f"{BASE}/{run}/llm_artifacts/{dataset}"
                if not os.path.isdir(art_dir):
                    print(f"pulando {run}/{dataset}: sem artifacts")
                    continue
            elif not os.path.exists(csv_path):
                print(f"pulando {run}/{dataset}: sem CSV")
                continue

            models = [m for m in MODELS if os.path.exists(I.model_csv_path(m, dataset, BASE))]
            frames = I.load_dataset_frames(models, dataset, BASE)
            n_series = I.count_series(dataset, models[0], BASE)
            try:
                bad = I.find_misaligned_models(
                    models, dataset, 0, results_dir=BASE, frames=frames, n_windows=3,
                )
            except Exception:
                bad = {}
            models_eff = [m for m in models if m not in bad]

            n = min(n_series, args.limit_series) if args.limit_series else n_series
            print(f"{run}/{dataset}: {n} séries...", flush=True)
            for idx in range(n):
                try:
                    ing, st, series_card, pool_card, seeds, floor = load_series_state(
                        dataset, idx, TSF[dataset], models_eff, frames, cfg, args.source_dir,
                    )
                except Exception as exc:
                    print(f"  [warn] {dataset}#{idx}: {type(exc).__name__}: {exc}")
                    continue

                ref_best, ref_winner, references = series_references(dataset, idx, floor)
                # universo estático: sementes + best_single de CADA modelo +
                # mean/median/dba do pool completo (calculado antes de qualquer
                # replay; não depende do agente)
                universe = evaluate_universe(st, ing)
                floor_val = min(a.score for a in seeds)

                if emitter.include_seeds and not no_seeds:
                    for a in seeds:
                        row = make_example(
                            run, dataset, idx, st, series_card, pool_card,
                            a.spec, a, floor, floor_val, ing, turn=0, origin="baseline",
                            ref_best=ref_best, ref_winner=ref_winner,
                            references=references, universe=universe,
                        )
                        if row:
                            emitter.emit(row, (dataset, idx, json.dumps(a.spec, sort_keys=True), "seed"))

                if kind == "gpt_oss":
                    art = json.load(open(f"{BASE}/{run}/llm_artifacts/{dataset}/dataset_{idx}.json"))
                    for entry in art["react"]["trajectory"]:
                        action = entry["action"]
                        a = dict(entry.get("action_args") or {})
                        if action != "evaluate_strategy":
                            try:
                                call_tool(st, action, a, withheld={})
                            except Exception:
                                pass
                            continue
                        try:
                            ok, _ = call_tool(st, "evaluate_strategy", a, withheld={})
                            if not ok:
                                continue
                            attempt = st.attempts[-1]
                        except Exception:
                            continue
                        row = make_example(
                            run, dataset, idx, st, series_card, pool_card,
                            attempt.spec, attempt, floor, floor_val, ing,
                            turn=int(entry.get("iteration", 0)), origin="agent",
                            ref_best=ref_best, ref_winner=ref_winner,
                            references=references, universe=universe,
                        )
                        if row:
                            emitter.emit(row, (run, dataset, idx, json.dumps(attempt.spec, sort_keys=True), entry.get("iteration")))
                else:
                    # laya: traces no CSV
                    df = pd.read_csv(csv_path, sep=";")
                    row_csv = df[df.dataset_index == idx].iloc[0]
                    desc = json.loads(row_csv.description)
                    trace = desc.get("loop", {}).get("trace", [])
                    if no_seeds:
                        st.attempts.clear()
                        for h in [h for h in list(st.pools) if h != FULL_POOL]:
                            del st.pools[h]
                    for entry in trace:
                        action = entry["action"]
                        if action == "accept":
                            break
                        if action in TOOL_LABELS:
                            tool, targs = TOOL_LABELS[action]
                            try:
                                getattr(T, tool)(st, **targs)
                            except Exception:
                                pass
                            continue
                        spec = label_to_spec(action)
                        try:
                            attempt, _ = st.evaluate(spec, rationale="replay", origin="agent", iteration=int(entry["iteration"]))
                        except Exception:
                            continue
                        row = make_example(
                            run, dataset, idx, st, series_card, pool_card,
                            attempt.spec, attempt, floor, floor_val, ing,
                            turn=int(entry.get("iteration", 0)), origin="agent",
                            ref_best=ref_best, ref_winner=ref_winner,
                            references=references, universe=universe,
                        )
                        if row:
                            emitter.emit(row, (run, dataset, idx, json.dumps(attempt.spec, sort_keys=True), entry.get("iteration")))

    n_rows = emitter.flush()
    print("\nExemplos por fonte:")
    for src, cnt in sorted(emitter.counts.items()):
        print(f"  {src:<60} {cnt}")
    print(f"\ntotal: {n_rows} exemplos -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
