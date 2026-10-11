#!/usr/bin/env python3
"""Braco de busca NAO-LLM sobre o mesmo catalogo e orcamento do agente.

Grava em `<results_dir>/orchestrator_search_<version>/<DATASET>.csv`, uma linha
por (serie, braco), com `regressor = search_<seeds|random|greedy|ucb>`.

Uso tipico (servidor):

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    nohup python3 aprovacao_solida/01_busca_nao_llm/run_search.py \\
        --dataset ANP_MONTHLY --version v1 --methods seeds,random,greedy,ucb \\
        > logs/search_v1_anp.log 2>&1 &

Ou todos os datasets de uma vez com `aprovacao_solida/01_busca_nao_llm/run_all.sh`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import (  # noqa: E402
    DATASETS,
    DEFAULT_RESULTS_DIR,
    append_rows,
    detect_drop,
    load_context,
    v5_config,
)
from search_lib import run_method  # noqa: E402

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402


def _artifacts_path(out_dir, dataset, idx, method):
    d = os.path.join(out_dir, "search_artifacts", dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"dataset_{idx}_{method}.json")


def run_one(ingested, state, cfg, method, evals, out_dir, dataset):
    n_evals = run_method(state, method, evals)
    best = state.best_attempt()
    if best is None:
        raise RuntimeError("nenhuma tentativa no historico")
    forecast, debug = state.apply_to_test(best.spec)
    metrics = compute_metrics(forecast, ingested.test_values)

    seeds = [a for a in state.attempts if a.origin == "baseline"]
    floor = min(seeds, key=lambda a: a.score) if seeds else best

    try:
        payload = {
            "dataset": dataset,
            "dataset_index": ingested.dataset_index,
            "method": method,
            "budget": int(evals),
            "n_evals": int(n_evals),
            "best": {
                "id": best.attempt_id,
                "spec": best.spec,
                "origin": best.origin,
                "score": float(best.score),
                "aggregate": {k: float(v) for k, v in best.aggregate.items()},
            },
            "floor": {
                "id": floor.attempt_id,
                "spec": floor.spec,
                "score": float(floor.score),
            },
            "attempts": [
                {
                    "id": a.attempt_id,
                    "spec": a.spec,
                    "origin": a.origin,
                    "score": float(a.score),
                }
                for a in state.ranked_attempts()
            ],
            "predict_debug": {k: v for k, v in debug.items() if k != "weights_resolved"},
        }
        with open(_artifacts_path(out_dir, dataset, ingested.dataset_index, method), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

    return {
        "dataset_index": str(ingested.dataset_index),
        "horizon": ingested.horizon,
        "regressor": f"search_{method}",
        "mape": metrics["mape"],
        "pocid": metrics["pocid"],
        "smape": metrics["smape"],
        "rmse": metrics["rmse"],
        "msmape": metrics["msmape"],
        "mae": metrics["mae"],
        "test": json.dumps(list(ingested.test_values)),
        "predictions": json.dumps(list(forecast)),
        "start_test": str(ingested.start_test),
        "final_test": str(ingested.final_test),
        "best_strategy_params": json.dumps(best.spec, ensure_ascii=False, default=str),
        "best_score": float(best.score) if np.isfinite(best.score) else float("nan"),
        "best_origin": best.origin,
        "floor_strategy_params": json.dumps(floor.spec, ensure_ascii=False, default=str),
        "floor_score": float(floor.score) if np.isfinite(floor.score) else float("nan"),
        "n_evals": int(n_evals),
    }


def build_parser():
    p = argparse.ArgumentParser(description="Busca nao-LLM sobre o catalogo do CREST.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--version", default="v1", help="sufixo de orchestrator_search_<version>")
    p.add_argument("--methods", default="seeds,random,greedy,ucb",
                   help="braços, separados por virgula: seeds,random,greedy,ucb")
    p.add_argument("--evals", type=int, default=12,
                   help="orcamento de avaliacoes novas por braço (default 12 = max_iterations do agente)")
    p.add_argument("--models", nargs="+", default=None, help="pool de modelos (default: os 19 da v5)")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", default=None, help="onde gravar (default: --results-dir)")
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="apenas as N primeiras series")
    p.add_argument("--no-pooled-meta-model", action="store_true",
                   help="pula o pre-passo entre series (tambem evita xgboost)")
    p.add_argument("--no-source", action="store_true", help="nao carrega o .tsf (perfil cai nas janelas)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    cfg = v5_config(no_pooled=args.no_pooled_meta_model, name=f"search_{args.version}")
    cfg.max_iterations = int(args.evals)

    if not args.quiet:
        print(f"dataset : {args.dataset}")
        print(f"models  : {len(models)}")
        print(f"methods : {methods} | evals/braço: {args.evals}")
        print(f"index   : {args.indices if args.indices else ('first %s' % args.limit if args.limit else 'all')}")

    frames, n_series, source = load_context(
        models, args.dataset, args.results_dir, args.source_dir, use_source=not args.no_source
    )
    if args.indices is not None:
        todo = list(args.indices)
    elif args.limit:
        todo = list(range(min(int(args.limit), n_series)))
    else:
        todo = list(range(n_series))

    drop = detect_drop(models, args.dataset, todo[0], cfg, args.results_dir, frames) if todo else []
    if drop and not args.quiet:
        print(f"WARNING: modelos desalinhados descartados: {drop}")

    meta_models = {}
    if cfg.pooled_meta_model:
        meta_models = pipeline_mod._build_pooled_meta_models(
            models, args.dataset, todo, config=cfg, source=source,
            results_dir=args.results_dir, frames=frames, drop_models=drop,
        )
        if not meta_models and not args.quiet:
            print("WARNING: pooled meta-model nao treinado; a decima semente nao existe")

    out_root = args.output_dir or args.results_dir
    out_dir = os.path.join(out_root, f"orchestrator_search_{args.version}")
    csv_path = os.path.join(out_dir, f"{args.dataset}.csv")

    rows = []
    started = time.perf_counter()
    for idx in todo:
        for method in methods:
            t0 = time.perf_counter()
            try:
                ingested = ingest_mod.load_series(
                    models=models, dataset=args.dataset, dataset_index=int(idx),
                    source=source, config=cfg, results_dir=args.results_dir,
                    frames=frames, drop_models=drop,
                )
                state = ingested.state
                state.pooled_meta_model = meta_models.get(int(idx))
                pool_mod.run_phase2(state, cfg)
                row = run_one(ingested, state, cfg, method, args.evals, out_dir, args.dataset)
                rows.append(row)
                append_rows(csv_path, [row])
                if not args.quiet:
                    best = json.loads(row["best_strategy_params"])
                    print(
                        f"[{idx:>4} {method:<7}] smape={row['smape']:.4f} "
                        f"pocid={row['pocid']:.1f} score={row['best_score']:.4f} "
                        f"evals={row['n_evals']} best={best} ({time.perf_counter()-t0:.1f}s)",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{idx:>4} {method:<7}] FALHOU: {type(exc).__name__}: {exc}", flush=True)

    if not args.quiet:
        print(f"\ndone in {time.perf_counter()-started:.1f}s | rows={len(rows)} | csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
