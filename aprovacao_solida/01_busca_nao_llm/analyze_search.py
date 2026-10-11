#!/usr/bin/env python3
"""Compara os braços de busca nao-LLM com CREST, sementes e baselines.

Le `orchestrator_search_<version>` (escrito por `run_search.py`) e os runs
existentes em `timeseries/mestrado/resultados`, e produz:

* uma tabela por dataset com media de SMAPE/POCID/RMSE por metodo;
* comparacoes pareadas (Wilcoxon + Holm) contra uma referencia;
* `search_best`: por serie, o melhor braço de busca escolhido pelo SCORE DE
  VALIDACAO (um teto para "busca simples" — se nem esse teto bate o agente,
  a conclusao de que busca nao agrega fica forte).

Uso:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    python3 aprovacao_solida/01_busca_nao_llm/analyze_search.py --version v1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import DATASETS, DEFAULT_RESULTS_DIR, collect_methods  # noqa: E402
from _stats import holm, pairwise_table  # noqa: E402


def print_dataset_table(dataset, methods, metrics):
    order = [m for m in ("seeds", "random", "greedy", "ucb", "search_best", "sem_agente",
                         "CREST", "CREST_qwen", "CREST_gemma", "median", "FFORMA", "ADE")
             if m in methods]
    print(f"\n=== {dataset} ===")
    header = f"{'metodo':<14}" + "".join(f"{m:>10}" for m in metrics) + f"{'n':>6}"
    print(header)
    for name in order:
        m = methods[name]
        cells = []
        for metric in metrics:
            cells.append(f"{m[metric].astype(float).mean():10.4f}" if metric in m.columns else f"{'-':>10}")
        print(f"{name:<14}" + "".join(cells) + f"{len(m):>6}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Analise dos braços de busca nao-LLM.")
    p.add_argument("--version", default="v1", help="sufixo de orchestrator_search_<version>")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--reference", default="sem_agente",
                   help="metodo de referencia das comparacoes pareadas")
    p.add_argument("--metrics", nargs="+", default=["smape", "pocid", "rmse"])
    p.add_argument("--out", default="outputs/resultados/aprovacao_solida/analise_busca.csv")
    args = p.parse_args(argv)

    all_rows = []
    headline = {}
    for dataset in args.datasets:
        methods = collect_methods(dataset, args.results_dir, search_version=args.version)
        if not methods:
            print(f"{dataset}: nenhum CSV encontrado, pulando")
            continue
        print_dataset_table(dataset, methods, args.metrics)
        if args.reference in methods:
            rows = pairwise_table(methods, args.reference, dataset, tuple(args.metrics))
            holm(rows)
            all_rows.extend(rows)
        headline[dataset] = {
            name: float(m["smape"].astype(float).mean()) for name, m in methods.items()
            if "smape" in m.columns
        }

    if all_rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.out, sep=";", index=False)
        print(f"\ncomparacoes salvas em: {args.out}")

    if headline:
        names = sorted({n for v in headline.values() for n in v})
        print("\n=== SMAPE medio por dataset (menor melhor) ===")
        print(f"{'metodo':<14}" + "".join(f"{d[:12]:>13}" for d in headline))
        for name in names:
            cells = "".join(
                f"{headline[d][name]:13.4f}" if name in headline[d] else f"{'-':>13}"
                for d in headline
            )
            print(f"{name:<14}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
