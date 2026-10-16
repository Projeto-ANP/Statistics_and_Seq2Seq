#!/usr/bin/env python3
"""Busca nao-LLM restrita ao catalogo final (catalog v2) + analise, em um comando.

Roda, na sequencia e em um unico processo:

    run_search.py --dataset <ds> --version catalogo_final --methods seeds,random,greedy,ucb
    analyze_search.py --version catalogo_final

para os sete conjuntos de dados, imprimindo o progresso série a série. Assim o
terminal nao fica preso em duas etapas (o "for" e o comando de analise).

Uso no servidor (a partir da raiz do repositorio):

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    mkdir -p logs
    nohup python3 aprovacao_solida/01_busca_nao_llm/run_catalogo_final.py \
        > logs/catalogo_final_busca.log 2>&1 &

Para acompanhar:  tail -f logs/catalogo_final_busca.log

Ao final, traga de volta (git ou copia manual):
    outputs/resultados/aprovacao_solida/analise_busca.csv
(e, por garantia, a pasta timeseries/mestrado/resultados/orchestrator_search_catalogo_final/)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.dirname(_HERE), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(_REPO)

import analyze_search  # noqa: E402
import run_search  # noqa: E402
from _common import DATASETS, DEFAULT_RESULTS_DIR  # noqa: E402
from search_lib import allowed_weight_methods  # noqa: E402

VERSION = "catalogo_final"
REQUIRED_WEIGHT_METHODS = {
    "inverse", "trend", "meta", "best_per_horizon", "topk_per_horizon", "forward",
}


def check_search_space() -> None:
    """Aborta se o search_lib.py do servidor ainda for o de espaco amplo."""
    dummy = SimpleNamespace(n_windows=3, config=SimpleNamespace(min_windows_for_ols=5))
    methods = set(allowed_weight_methods(dummy))
    if not REQUIRED_WEIGHT_METHODS.issubset(methods) or {"softmax", "feature"} & methods:
        raise SystemExit(
            "search_lib.py desatualizado: o espaco de pesos nao e o do catalogo final "
            f"(metodos vistos: {sorted(methods)}). Atualize "
            "aprovacao_solida/01_busca_nao_llm/search_lib.py antes de rodar."
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Busca restrita ao catalogo final + analise, em um unico comando."
    )
    p.add_argument("--version", default=VERSION, help=f"sufixo orchestrator_search_<version> (default: {VERSION})")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--methods", default="seeds,random,greedy,ucb")
    p.add_argument("--evals", type=int, default=12, help="orcamento de avaliacoes novas por braço (default 12)")
    args = p.parse_args(argv)

    check_search_space()

    out_dir = os.path.join(DEFAULT_RESULTS_DIR, f"orchestrator_search_{args.version}")
    if os.path.isdir(out_dir):
        print(f"[limpeza] removendo resultados anteriores em {out_dir}", flush=True)
        shutil.rmtree(out_dir)

    t0 = time.perf_counter()
    for ds in args.datasets:
        ts = time.perf_counter()
        print(f"\n[{time.strftime('%H:%M:%S')}] busca: {ds}", flush=True)
        code = run_search.main([
            "--dataset", ds,
            "--version", args.version,
            "--methods", args.methods,
            "--evals", str(args.evals),
        ])
        print(
            f"[{time.strftime('%H:%M:%S')}] {ds}: terminou em {time.perf_counter()-ts:.0f}s "
            f"(exit {code})",
            flush=True,
        )

    print(f"\n[{time.strftime('%H:%M:%S')}] analise (analyze_search.py --version {args.version}) ...", flush=True)
    code = analyze_search.main(["--version", args.version])
    print(
        f"\n[{time.strftime('%H:%M:%S')}] busca + analise concluidas em "
        f"{time.perf_counter()-t0:.0f}s (exit {code})",
        flush=True,
    )
    print(
        "\nTraga de volta para atualizar as tabelas do guia:\n"
        "  outputs/resultados/aprovacao_solida/analise_busca.csv\n",
        flush=True,
    )
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
