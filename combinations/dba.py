"""
Combinação de previsões via DTW Barycenter Averaging (tslearn) entre modelos
base — a "média" na métrica de distância DTW em vez da distância euclidiana.

Não precisa de env dedicado: usa numpy/pandas/tslearn, já presentes no `agno`.

    conda activate agno
    cd Statistics_and_Seq2Seq
    python -m combinations.dba --dataset ANP_MONTHLY

Determinismo: `dtw_barycenter_averaging` (tslearn 0.6.3, instalado no `agno`)
NÃO aceita `random_state` como argumento — passar esse kwarg levanta
TypeError. Sem seed nenhuma, ela inicializa o centroide a partir do RNG global
do numpy (via `sklearn.utils.check_random_state(None)`), então rodar a mesma
série duas vezes pode dar resultados diferentes se qualquer código não
relacionado tiver consumido números aleatórios entre as duas chamadas — é
exatamente o que acontecia aqui: duas séries idênticas do NN5 (T1==T47)
escolhiam DBA e saíam com sMAPE 0.1199 vs 0.1217. A correção é semear o RNG
global do numpy imediatamente antes da chamada (`np.random.seed`), não passar
`random_state` para a função.
"""

from __future__ import annotations

import argparse

import numpy as np

from .simple_combiners import run_simple_combination

try:
    from tslearn.barycenters import dtw_barycenter_averaging
except ImportError as e:  # pragma: no cover - depende do ambiente
    raise ImportError(
        "tslearn não está instalado neste ambiente.\n"
        "    conda run -n agno pip install tslearn"
    ) from e


def _dba_aggregate(preds: np.ndarray, max_iter: int, random_state: int) -> np.ndarray:
    n_models, horizon = preds.shape
    X3 = preds.reshape(n_models, horizon, 1)

    np.random.seed(random_state)  # ver docstring do módulo — dtw_barycenter_averaging
    # não tem parâmetro random_state; a inicialização lê o RNG global do numpy.
    centroid = dtw_barycenter_averaging(X3, max_iter=max_iter)
    return np.asarray(centroid, dtype=float).ravel()


def dba_combination(
    dataset_name: str,
    models: list[str] | None = None,
    exp_name: str = "dba",
    horizon: int | None = None,
    resume: bool = False,
    drop_misaligned: bool = True,
    max_iter: int = 30,
    random_state: int = 7,
) -> None:
    run_simple_combination(
        dataset_name=dataset_name,
        aggregate=lambda m: _dba_aggregate(m, max_iter=max_iter, random_state=random_state),
        exp_name=exp_name,
        models=models,
        horizon=horizon,
        resume=resume,
        drop_misaligned=drop_misaligned,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m combinations.dba",
        description="Combinação de previsões via DTW Barycenter Averaging (tslearn).",
    )
    p.add_argument("--dataset", required=True, help="ex: ANP_MONTHLY, M4_WEEKLY_DATASET")
    p.add_argument("--models", nargs="+", default=None, help="default: os 19 modelos base")
    p.add_argument("--exp-name", default="dba", help="subpasta de saída em resultados/")
    p.add_argument("--horizon", type=int, default=None, help="default: lido do CSV")
    p.add_argument("--resume", action="store_true", help="continua em vez de apagar a saída")
    p.add_argument(
        "--keep-misaligned", action="store_true",
        help="não dropar modelos cujo alvo de teste diverge do de referência (default: dropa)",
    )
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--random-state", type=int, default=7)
    args = p.parse_args(argv)

    dba_combination(
        dataset_name=args.dataset,
        models=args.models,
        exp_name=args.exp_name,
        horizon=args.horizon,
        resume=args.resume,
        drop_misaligned=not args.keep_misaligned,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
