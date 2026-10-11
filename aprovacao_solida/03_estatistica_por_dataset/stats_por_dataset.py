#!/usr/bin/env python3
"""Tabela pareada por dataset com correcao de Holm — a versao defensavel do item_07.

Por que existe: a analise agregada atual junta 680 series de 7 datasets num unico
Wilcoxon. Para a banca, cada dataset e uma populacao com dinamica propria; a
comparacao correta e pareada DENTRO do dataset, e o conjunto de comparacoes de um
dataset recebe correcao de multiplicidade. Este script faz isso para todos os
metodos disponiveis (CREST, variantes, sem-agente, FFORMA, ADE, mediana/estaticas,
e os braços de busca nao-LLM, se `--search-version` for passado).

Uso:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    python3 aprovacao_solida/03_estatistica_por_dataset/stats_por_dataset.py \\
        --search-version v1
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
from _stats import LOWER_IS_BETTER, holm, pairwise_table  # noqa: E402

ORDER = (
    "seeds", "random", "greedy", "ucb", "search_best", "sem_agente",
    "CREST", "CREST_qwen", "CREST_gemma", "median", "FFORMA", "ADE",
)


def print_means(dataset, methods, metrics):
    names = [m for m in ORDER if m in methods] + [m for m in sorted(methods) if m not in ORDER]
    print(f"\n=== {dataset} ===")
    print(f"{'metodo':<14}" + "".join(f"{m:>10}" for m in metrics) + f"{'n':>6}")
    for name in names:
        m = methods[name]
        cells = []
        for metric in metrics:
            if metric in m.columns:
                cells.append(f"{m[metric].astype(float).mean():10.4f}")
            else:
                cells.append(f"{'-':>10}")
        print(f"{name:<14}" + "".join(cells) + f"{len(m):>6}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Comparacoes pareadas por dataset com Holm.")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--reference", default="sem_agente",
                   help="metodo de referencia para as comparacoes pareadas")
    p.add_argument("--metrics", nargs="+", default=["smape", "pocid", "rmse"])
    p.add_argument("--search-version", default=None,
                   help="inclui os braços de orchestrator_search_<version> nas tabelas")
    p.add_argument("--extra", nargs="*", default=None,
                   help="metodos extras no formato rotulo=pasta (ex.: CREST_v6=orchestrator_react_v6)")
    p.add_argument("--out", default="outputs/resultados/aprovacao_solida/stats_por_dataset.csv")
    args = p.parse_args(argv)

    extra_folders = {}
    for item in (args.extra or []):
        if "=" in item:
            label, folder = item.split("=", 1)
            extra_folders[label.strip()] = folder.strip()

    all_rows = []
    for dataset in args.datasets:
        methods = collect_methods(
            dataset, args.results_dir, search_version=args.search_version,
            extra_folders=extra_folders or None,
        )
        if not methods:
            print(f"{dataset}: nada encontrado")
            continue
        print_means(dataset, methods, args.metrics)
        if args.reference not in methods:
            print(f"  (referencia {args.reference!r} ausente; sem comparacoes pareadas)")
            continue
        rows = pairwise_table(methods, args.reference, dataset, tuple(args.metrics))
        holm(rows)
        all_rows.extend(rows)
        sig = [r for r in rows if r.get("significant")]
        if sig:
            print(f"  significativos (Holm<0.05) vs {args.reference}:")
            for r in sig:
                lower_better = r["metric"] in LOWER_IS_BETTER
                method_better = (r["mean_diff"] < 0) if lower_better else (r["mean_diff"] > 0)
                direction = "melhor" if method_better else "pior"
                print(
                    f"    {r['method']:<14} {r['metric']:<6} "
                    f"diff={r['mean_diff']:+.4f} p={r['p_value']:.4g} p_holm={r['p_holm']:.4g} "
                    f"W/T/L={r['wins']}/{r['ties']}/{r['losses']} ({direction})"
                )

    if all_rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.out, sep=";", index=False)
        print(f"\nsalvo em: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
