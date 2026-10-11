#!/usr/bin/env python3
"""Sensibilidade da decisao final aos pesos da pontuacao.

Para cada serie, recomputa a pontuacao das mesmas estrategias candidatas
(sementes deterministicas mais o vencedor aplicado pelo run) sob cada preset de
`orchestrator_react/config.py:SCORE_PRESETS`, escolhe o argmin de cada preset e
compara duas coisas: em quantas series a estrategia selecionada muda, e qual e o
SMAPE e o POCID de teste da estrategia selecionada em cada preset.

Serve para responder ao orientador sobre a escolha (0,3, 0,3, 0,2, 0,2): se a
selecao muda pouco e o teste varia pouco, os pesos nao dirigem os resultados.
Deterministico, sem LLM.

Uso:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    python3 aprovacao_solida/06_sensibilidade_score/run_score_sensitivity.py \\
        --run orchestrator_react_v5 --limit 20     # smoke
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

from _common import (  # noqa: E402
    DATASETS,
    DEFAULT_RESULTS_DIR,
    detect_drop,
    load_context,
    load_dataset_csv,
    v5_config,
)

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react.config import SCORE_PRESETS  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from orchestrator_react.metrics import composite_score  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402


def _label(spec):
    bits = [str(spec.get("combine"))]
    if spec.get("pool") and spec["pool"] != "pool_full":
        bits.append(str(spec["pool"]))
    if spec.get("weights"):
        bits.append(str(spec["weights"]))
    if spec.get("model"):
        bits.append(str(spec["model"]))
    return "_".join(bits)


def _candidates(state, row):
    """Sementes (recalculadas) mais o vencedor do run (agregados do CSV)."""
    cands = []
    for a in state.attempts:
        if a.origin != "baseline":
            continue
        cands.append(
            {
                "label": "seed:" + _label(a.spec),
                "aggregate": a.aggregate,
                "spec": a.spec,
                "test": None,
            }
        )
    description = json.loads(row["description"]) if isinstance(row.get("description"), str) else {}
    agg = (description.get("validation") or {}).get("aggregate")
    if isinstance(agg, dict) and "RMSE" in agg:
        try:
            spec = json.loads(row["best_strategy_params"])
        except Exception:
            spec = None
        cands.append(
            {
                "label": "winner:" + (_label(spec) if spec else "?"),
                "aggregate": agg,
                "spec": spec,
                "test": {"smape": float(row["smape"]), "pocid": float(row["pocid"])},
            }
        )
    return cands


def main(argv=None):
    p = argparse.ArgumentParser(description="Sensibilidade da selecao aos pesos do score.")
    p.add_argument("--run", default="orchestrator_react_v5", help="pasta do run no results-dir")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-pooled-meta-model", action="store_true")
    p.add_argument("--no-source", action="store_true")
    p.add_argument("--out", default="outputs/resultados/aprovacao_solida/sensibilidade_score.csv")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    cfg = v5_config(no_pooled=args.no_pooled_meta_model, name="score_sensitivity")
    presets = list(SCORE_PRESETS)
    all_rows = []

    for dataset in args.datasets:
        df = load_dataset_csv(args.run, dataset, args.results_dir)
        if df is None or df.empty:
            if not args.quiet:
                print(f"{dataset}: sem CSV em {args.run}, pulando")
            continue
        todo = df.copy()
        if args.indices is not None:
            todo = todo[todo["dataset_index"].astype(int).isin(args.indices)]
        if args.limit:
            todo = todo.head(int(args.limit))
        if todo.empty:
            continue

        frames, _n, source = load_context(
            models, dataset, args.results_dir, args.source_dir,
            use_source=not args.no_source,
        )
        first = int(todo["dataset_index"].astype(int).iloc[0])
        drop = detect_drop(models, dataset, first, cfg, args.results_dir, frames)
        meta_models = {}
        if cfg.pooled_meta_model:
            meta_models = pipeline_mod._build_pooled_meta_models(
                models, dataset, [int(i) for i in todo["dataset_index"].astype(int)],
                config=cfg, source=source, results_dir=args.results_dir,
                frames=frames, drop_models=drop,
            )

        rows = []
        for _pos, roww in todo.iterrows():
            try:
                idx = int(roww["dataset_index"])
                ingested = ingest_mod.load_series(
                    models=models, dataset=dataset, dataset_index=idx, source=source,
                    config=cfg, results_dir=args.results_dir, frames=frames, drop_models=drop,
                )
                state = ingested.state
                state.pooled_meta_model = meta_models.get(idx)
                pool_mod.run_phase2(state, cfg)
                anchor = state.baseline_aggregate()
                cands = _candidates(state, roww)
                if not cands:
                    continue
                cache = {}
                per = {}
                for preset, weights in SCORE_PRESETS.items():
                    best, best_score = None, float("inf")
                    for c in cands:
                        sc = composite_score(c["aggregate"], anchor, weights)
                        if sc < best_score:
                            best_score, best = sc, c
                    if best["test"] is None:
                        key = best["label"]
                        if key not in cache:
                            fc, _ = state.apply_to_test(best["spec"])
                            m = compute_metrics(fc, ingested.test_values)
                            cache[key] = {"smape": float(m["smape"]), "pocid": float(m["pocid"])}
                        test = cache[key]
                    else:
                        test = best["test"]
                    per[preset] = {"label": best["label"], "score": best_score, **test}

                ref = per.get("balanced")
                out = {
                    "dataset": dataset,
                    "dataset_index": idx,
                    "n_candidates": len(cands),
                    "winner_aggregate_available": int(any(c["label"].startswith("winner:") for c in cands)),
                }
                for preset in presets:
                    out[f"{preset}_label"] = per[preset]["label"]
                    out[f"{preset}_smape"] = per[preset]["smape"]
                    out[f"{preset}_pocid"] = per[preset]["pocid"]
                    out[f"{preset}_score"] = per[preset]["score"]
                out["selection_changed_vs_balanced"] = int(
                    any(per[p]["label"] != ref["label"] for p in presets)
                )
                out["smape_delta_max_preset"] = float(
                    max(abs(per[p]["smape"] - ref["smape"]) for p in presets)
                )
                rows.append(out)
                if not args.quiet:
                    changed = [p for p in presets if per[p]["label"] != ref["label"]]
                    print(
                        f"[{dataset} {idx:>4}] candidatos={len(cands)} "
                        f"mudam={changed if changed else 'nenhum'} "
                        f"balanced={ref['smape']:.4f} "
                        f"max_delta={out['smape_delta_max_preset']:.4f}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{dataset} {roww.get('dataset_index')}] FALHOU: {type(exc).__name__}: {exc}", flush=True)

        if rows:
            all_rows.extend(rows)
            sub = pd.DataFrame(rows)
            print(f"\n=== {dataset}: n={len(sub)} ===")
            for preset in presets:
                col = f"{preset}_smape"
                print(
                    f"  {preset:<16} smape={sub[col].mean():.4f} "
                    f"pocid={sub[f'{preset}_pocid'].mean():.1f}"
                )
            print(f"  selecao mudou em {int(sub['selection_changed_vs_balanced'].sum())} series "
                  f"(delta max de SMAPE por serie: media {sub['smape_delta_max_preset'].mean():.4f})")

    if all_rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.out, sep=";", index=False)
        print(f"\nsalvo em: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
