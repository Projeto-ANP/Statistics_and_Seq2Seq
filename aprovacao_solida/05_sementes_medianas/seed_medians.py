#!/usr/bin/env python3
"""A "mediana ganha em tudo" vira hipotese testavel no PISO deterministico.

O conjunto de sementes atual tem media/media aparada nas pools estaveis, mas
nao tem MEDIANA nessas pools (a mediana so aparece no pool completo). Este script
avalia, por serie, dois pisos deterministicos:

    atual    : 3 classicas + 6 estaveis (media/media aparada) + meta pooled
    com_med  : o mesmo + mediana nas pools estaveis k=5,7,9

e mede no TESTE. Tudo determinístico (sem LLM). Se "com_med" vencer, da para
ligar essa mudanca no `pool.SEED_STABLE_POOLS` com evidencia, sem tocar na
arquitetura.

Uso:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    nohup python3 aprovacao_solida/05_sementes_medianas/seed_medians.py \\
        > logs/seed_medians.log 2>&1 &
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
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import DATASETS, DEFAULT_RESULTS_DIR, detect_drop, load_context, v5_config  # noqa: E402

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from orchestrator_react.state import FULL_POOL  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402

MEDIAN_STABLE = ((5, "median"), (7, "median"), (9, "median"))


def is_median_stable(attempt):
    return attempt.spec.get("combine") == "median" and attempt.spec.get("pool") != FULL_POOL


def main(argv=None):
    p = argparse.ArgumentParser(description="Piso deterministico com e sem medianas estaveis.")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-pooled-meta-model", action="store_true")
    p.add_argument("--no-source", action="store_true")
    p.add_argument("--out", default="outputs/resultados/aprovacao_solida/sementes_medianas.csv")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    cfg = v5_config(no_pooled=args.no_pooled_meta_model, name="seed_medians")
    all_rows = []

    for dataset in args.datasets:
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
        meta_models = {}
        if cfg.pooled_meta_model:
            meta_models = pipeline_mod._build_pooled_meta_models(
                models, dataset, todo, config=cfg, source=source,
                results_dir=args.results_dir, frames=frames, drop_models=drop,
            )
            if not meta_models:
                print(f"{dataset}: pooled meta-model nao treinado; piso efetivo com 9 sementes")

        rows = []
        for idx in todo:
            try:
                ingested = ingest_mod.load_series(
                    models=models, dataset=dataset, dataset_index=int(idx), source=source,
                    config=cfg, results_dir=args.results_dir, frames=frames, drop_models=drop,
                )
                state = ingested.state
                state.pooled_meta_model = meta_models.get(int(idx))
                stable = list(pool_mod.SEED_STABLE_POOLS) + list(MEDIAN_STABLE)
                pool_mod.seed_baselines(
                    state, pool=FULL_POOL, methods=pool_mod.SEED_BASELINES,
                    stable_pools=stable,
                    seed_pooled_meta_model=cfg.seed_pooled_meta_model and not args.no_pooled_meta_model,
                )
                attempts = [a for a in state.attempts if a.origin == "baseline" and np.isfinite(a.score)]
                if not attempts:
                    continue
                current = [a for a in attempts if not is_median_stable(a)]
                floor_current = min(current or attempts, key=lambda a: a.score)
                floor_plus = min(attempts, key=lambda a: a.score)

                fc, _ = state.apply_to_test(floor_current.spec)
                fp, _ = state.apply_to_test(floor_plus.spec)
                mc = compute_metrics(fc, ingested.test_values)
                mp = compute_metrics(fp, ingested.test_values)

                row = {
                    "dataset": dataset,
                    "dataset_index": int(idx),
                    "current_strategy": json.dumps(floor_current.spec, ensure_ascii=False, default=str),
                    "current_score": float(floor_current.score),
                    "current_smape": mc["smape"],
                    "current_pocid": mc["pocid"],
                    "current_rmse": mc["rmse"],
                    "plus_strategy": json.dumps(floor_plus.spec, ensure_ascii=False, default=str),
                    "plus_score": float(floor_plus.score),
                    "plus_smape": mp["smape"],
                    "plus_pocid": mp["pocid"],
                    "plus_rmse": mp["rmse"],
                    "changed": int(floor_current.spec != floor_plus.spec),
                    "dsmape": mp["smape"] - mc["smape"],
                }
                rows.append(row)
                if not args.quiet:
                    print(
                        f"[{idx:>4}] cur={row['current_smape']:.4f} plus={row['plus_smape']:.4f} "
                        f"changed={row['changed']} cur={row['current_strategy'][:60]}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{dataset} {idx}] FALHOU: {type(exc).__name__}: {exc}", flush=True)

        if rows:
            all_rows.extend(rows)
            sub = pd.DataFrame(rows)
            print(f"\n=== {dataset}: n={len(sub)} ===")
            print(f"  smape atual={sub['current_smape'].mean():.4f} | com medianas={sub['plus_smape'].mean():.4f} "
                  f"| diff={sub['dsmape'].mean():+.5f}")
            print(f"  com medianas melhor em {(sub['dsmape'] < -1e-6).sum()}, pior em {(sub['dsmape'] > 1e-6).sum()}, "
                  f"empate em {(sub['dsmape'].abs() <= 1e-6).sum()}; piso mudou em {int(sub['changed'].sum())}")

    if all_rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.out, sep=";", index=False)
        print(f"\nsalvo em: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
