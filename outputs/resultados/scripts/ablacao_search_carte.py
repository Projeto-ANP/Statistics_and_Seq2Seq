"""Ablacao do CARTE: buscas sem LLM no protocolo agent-only.

Roda random, greedy e UCB partindo de um historico vazio, com o mesmo
catalogo, o mesmo orcamento de avaliacoes e a mesma regra final do CARTE
(seasonal_guard). Grava um CSV por dataset em

    timeseries/mestrado/resultados/orchestrator_search_carte/<DATASET>.csv

com uma linha por (serie, braco), `regressor = search_<random_s7|random_s8|
random_s9|greedy|ucb>`. O artefato de cada serie guarda todas as tentativas,
com score de validacao e metricas de teste, para o oraculo sobre o universo
de estrategias avaliadas.

Uso no servidor:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    mkdir -p logs
    nohup python3 outputs/resultados/scripts/ablacao_search_carte.py \
        --methods random,greedy,ucb --random-seeds 7 8 9 --evals 12 \
        > logs/ablacao_search_carte.log 2>&1 &

O script roda os sete datasets em sequencia. Para um dataset:

    python3 outputs/resultados/scripts/ablacao_search_carte.py \
        --datasets ANP_MONTHLY --methods random,greedy,ucb \
        --random-seeds 7 8 9 --evals 12

Para retomar depois de uma interrupcao, sem refazer as linhas ja gravadas:

    python3 outputs/resultados/scripts/ablacao_search_carte.py \
        --methods random,greedy,ucb --random-seeds 7 8 9 --evals 12 --resume

Para um teste rapido:

    python3 outputs/resultados/scripts/ablacao_search_carte.py \
        --datasets ANP_MONTHLY --limit 2 --methods random,greedy,ucb \
        --random-seeds 7 --evals 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_APROV = os.path.join(_REPO, "aprovacao_solida")
for _p in (_REPO, _APROV, os.path.join(_APROV, "01_busca_nao_llm"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import (  # noqa: E402
    DATASETS,
    DEFAULT_RESULTS_DIR,
    append_rows,
    detect_drop,
    load_context,
)
from search_lib_carte import run_method  # noqa: E402

from orchestrator_react import final_choice as final_choice_mod  # noqa: E402
from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.config import LLMRole, ReactConfig  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402

OUT_DIRNAME = "orchestrator_search_carte"
ARTIFACTS_DIR = os.path.join(_REPO, "outputs", "resultados", "ablacao_search_carte", "artifacts")


def carte_config(evals: int, no_pooled_meta_model: bool) -> ReactConfig:
    cfg = ReactConfig()
    cfg.name = "ablacao_search_carte"
    cfg.n_validation_windows = 3
    cfg.backtest_mode = "expanding"
    cfg.nested_selection = True
    cfg.min_windows_for_ols = 5
    cfg.max_iterations = int(evals)
    cfg.early_stop_patience = 4
    cfg.no_early_stop = True
    cfg.scoring_budget = True
    cfg.max_total_turns = 24
    cfg.direct_preparation = True
    cfg.final_strategy = "seasonal_guard"
    cfg.final_selector_max_candidates = 10
    cfg.seasonal_guard_min_gap = 0.10
    cfg.pool_mode = "full"
    cfg.score_preset = "smape_only"
    cfg.score_window_mode = "weighted"
    cfg.allow_agent_score_preset = False
    cfg.no_seed_attempts = True
    cfg.allow_best_single = False
    cfg.catalog_v2 = True
    cfg.prompt_v2 = True
    cfg.extra_actions = True
    cfg.seed_stable_pools = True
    cfg.seed_pooled_meta_model = not no_pooled_meta_model
    cfg.pooled_meta_model = not no_pooled_meta_model
    cfg.dataset_card = True
    cfg.combinator = LLMRole(model=None)
    cfg.diagnostician = LLMRole(model=None)
    cfg.reporter = LLMRole(model=None)
    cfg.final_selector = LLMRole(model=None)
    return cfg


def artifacts_path(dataset: str, idx: int, method: str) -> str:
    d = os.path.join(ARTIFACTS_DIR, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"dataset_{idx}_{method}.json")


def run_one(ingested, state, cfg, method, label, seed, evals, dataset, quiet):
    run_method(state, method, evals, seed=seed)
    n_evals = len(state.attempts)
    leader = state.best_attempt()
    if leader is None:
        raise RuntimeError("nenhuma tentativa no historico")

    series_card = T.series_profile(state)
    try:
        chosen, choice_debug = final_choice_mod.choose_seasonal_guard(
            state=state,
            series_card=series_card,
            config=cfg,
            dataset=dataset,
            dataset_index=int(ingested.dataset_index),
        )
    except Exception as exc:
        chosen, choice_debug = leader, {"decision": "fallback", "fallback": f"error:{type(exc).__name__}"}
    if chosen is None:
        chosen = leader

    forecast, debug = state.apply_to_test(chosen.spec)
    metrics = compute_metrics(forecast, ingested.test_values)

    attempt_rows = []
    for attempt in state.ranked_attempts():
        test_smape = None
        test_pocid = None
        try:
            attempt_forecast, _ = state.apply_to_test(attempt.spec)
            attempt_metrics = compute_metrics(attempt_forecast, ingested.test_values)
            test_smape = float(attempt_metrics["smape"])
            test_pocid = float(attempt_metrics["pocid"])
        except Exception:
            pass
        attempt_rows.append(
            {
                "id": attempt.attempt_id,
                "spec": attempt.spec,
                "origin": attempt.origin,
                "validation_score": float(attempt.score),
                "test_smape": test_smape,
                "test_pocid": test_pocid,
            }
        )

    try:
        payload = {
            "dataset": dataset,
            "dataset_index": int(ingested.dataset_index),
            "method": method,
            "budget": int(evals),
            "n_evals": int(n_evals),
            "chosen": {
                "id": chosen.attempt_id,
                "spec": chosen.spec,
                "origin": chosen.origin,
                "score": float(chosen.score),
            },
            "leader": {
                "id": leader.attempt_id,
                "spec": leader.spec,
                "score": float(leader.score),
            },
            "final_choice": choice_debug,
            "attempts": attempt_rows,
            "predict_debug": {k: v for k, v in debug.items() if k != "weights_resolved"},
        }
        with open(artifacts_path(dataset, int(ingested.dataset_index), label), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

    row = {
        "dataset_index": str(ingested.dataset_index),
        "horizon": ingested.horizon,
        "regressor": f"search_{label}",
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
        "best_strategy_params": json.dumps(chosen.spec, ensure_ascii=False, default=str),
        "best_score": float(chosen.score) if np.isfinite(chosen.score) else float("nan"),
        "leader_strategy_params": json.dumps(leader.spec, ensure_ascii=False, default=str),
        "leader_score": float(leader.score) if np.isfinite(leader.score) else float("nan"),
        "final_choice_reason": choice_debug.get("reason") or choice_debug.get("decision"),
        "n_evals": int(n_evals),
    }
    if not quiet:
        best = chosen.spec
        print(
            f"[{dataset} {ingested.dataset_index:>4} {label:<10}] "
            f"smape={row['smape']:.4f} pocid={row['pocid']:.1f} "
            f"score={row['best_score']:.4f} evals={row['n_evals']} "
            f"chosen={best}",
            flush=True,
        )
    return row


def build_parser():
    p = argparse.ArgumentParser(description="Buscas sem LLM no protocolo agent-only do CARTE.")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--methods", default="random,greedy,ucb")
    p.add_argument("--random-seeds", nargs="+", type=int, default=[7],
                   help="sementes do braco aleatorio (as demais buscas sao deterministicas)")
    p.add_argument("--evals", type=int, default=12)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-pooled-meta-model", action="store_true")
    p.add_argument("--no-source", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="pula pares (serie, braco) ja presentes no CSV do dataset")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in methods if m not in ("random", "greedy", "ucb")]
    if invalid:
        raise SystemExit(f"metodos invalidos: {invalid}; use random,greedy,ucb")

    cfg = carte_config(args.evals, args.no_pooled_meta_model)
    out_root = args.output_dir or args.results_dir
    out_dir = os.path.join(out_root, OUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for method in methods:
        if method == "random":
            for seed in args.random_seeds:
                runs.append(("random", f"random_s{seed}", int(seed)))
        else:
            runs.append((method, method, 7))

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===", flush=True)
        frames, n_series, source = load_context(
            models, dataset, args.results_dir, args.source_dir, use_source=not args.no_source
        )
        if args.indices is not None:
            todo = list(args.indices)
        elif args.limit:
            todo = list(range(min(int(args.limit), n_series)))
        else:
            todo = list(range(n_series))

        drop = detect_drop(models, dataset, todo[0], cfg, args.results_dir, frames) if todo else []
        if drop and not args.quiet:
            print(f"WARNING: modelos desalinhados descartados: {drop}", flush=True)

        meta_models = {}
        if cfg.pooled_meta_model and todo:
            meta_models = pipeline_mod._build_pooled_meta_models(
                models,
                dataset,
                todo,
                config=cfg,
                source=source,
                results_dir=args.results_dir,
                frames=frames,
                drop_models=drop,
            )
            if not meta_models and not args.quiet:
                print("WARNING: pooled meta-model nao treinado; weights_meta_model fica indisponivel.", flush=True)

        csv_path = os.path.join(out_dir, f"{dataset}.csv")
        done = set()
        if args.resume and os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
            old = pd.read_csv(csv_path, sep=";", usecols=["dataset_index", "regressor"])
            for row in old.itertuples():
                idx = int(row.dataset_index)
                label = str(row.regressor).replace("search_", "")
                if os.path.isfile(artifacts_path(dataset, idx, label)):
                    done.add((idx, str(row.regressor)))
            if not args.quiet:
                print(f"retomando: {len(done)} linhas ja gravadas em {csv_path}", flush=True)

        for idx in todo:
            for method, label, seed in runs:
                if (int(idx), f"search_{label}") in done:
                    continue
                t0 = time.perf_counter()
                try:
                    ingested = ingest_mod.load_series(
                        models=models,
                        dataset=dataset,
                        dataset_index=int(idx),
                        source=source,
                        config=cfg,
                        results_dir=args.results_dir,
                        frames=frames,
                        drop_models=drop,
                    )
                    state = ingested.state
                    state.pooled_meta_model = meta_models.get(int(idx))
                    pool_mod.run_phase2(state, cfg)
                    row = run_one(ingested, state, cfg, method, label, seed, args.evals, dataset, args.quiet)
                    row["elapsed_s"] = time.perf_counter() - t0
                    append_rows(csv_path, [row])
                    if not args.quiet:
                        print(f"  -> {time.perf_counter() - t0:.1f}s", flush=True)
                except Exception as exc:
                    print(
                        f"[{dataset} {idx:>4} {label:<10}] FALHOU: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
        print(f"CSV: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
