"""
Combinação de previsões pela mediana entre modelos base (mais robusta que a
média a um modelo que degenera — ver aviso sobre `mean` no ETTM, onde o
catboost prevê ~1e12 e a média explode).

Não precisa de env dedicado: usa só numpy/pandas, já presentes no `agno`.

    conda activate agno
    cd Statistics_and_Seq2Seq
    python -m combinations.median --dataset ANP_MONTHLY
"""

from __future__ import annotations

import argparse

import numpy as np

from .simple_combiners import run_simple_combination


def median_combination(
    dataset_name: str,
    models: list[str] | None = None,
    exp_name: str = "median",
    horizon: int | None = None,
    resume: bool = False,
    drop_misaligned: bool = True,
) -> None:
    run_simple_combination(
        dataset_name=dataset_name,
        aggregate=lambda m: np.median(m, axis=0),
        exp_name=exp_name,
        models=models,
        horizon=horizon,
        resume=resume,
        drop_misaligned=drop_misaligned,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m combinations.median",
        description="Combinação de previsões pela mediana entre modelos base.",
    )
    p.add_argument("--dataset", required=True, help="ex: ANP_MONTHLY, M4_WEEKLY_DATASET")
    p.add_argument("--models", nargs="+", default=None, help="default: os 19 modelos base")
    p.add_argument("--exp-name", default="median", help="subpasta de saída em resultados/")
    p.add_argument("--horizon", type=int, default=None, help="default: lido do CSV")
    p.add_argument("--resume", action="store_true", help="continua em vez de apagar a saída")
    p.add_argument(
        "--keep-misaligned", action="store_true",
        help="não dropar modelos cujo alvo de teste diverge do de referência (default: dropa)",
    )
    args = p.parse_args(argv)

    median_combination(
        dataset_name=args.dataset,
        models=args.models,
        exp_name=args.exp_name,
        horizon=args.horizon,
        resume=args.resume,
        drop_misaligned=not args.keep_misaligned,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
