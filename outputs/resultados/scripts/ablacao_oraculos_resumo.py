"""Consolida os oráculos do CARTE para a seção de ablação.

Combina o CSV dos oráculos das tentativas (melhor tentativa, melhores dez,
melhor individual e CARTE médio) com o CSV do oráculo amplo sobre U (teto de
validação e oráculo de teste). Gera uma tabela por série e o resumo por
conjunto de dados, com a média dos sete e o agregado das 680 séries.

Saídas:
  outputs/resultados/ablacao_oraculos_resumo/por_serie.csv
  outputs/resultados/ablacao_oraculos_resumo/resumo.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
OUT_BASE = os.path.join(_REPO, "outputs", "resultados")
ATTEMPTS = os.path.join(OUT_BASE, "ablacao_oraculos_carte", "por_serie.csv")
UNIVERSE = os.path.join(OUT_BASE, "ablacao_oraculo_u_carte", "por_serie.csv")
OUT_DIR = os.path.join(OUT_BASE, "ablacao_oraculos_resumo")
DATASETS = [
    "ANP_MONTHLY",
    "NN5_WEEKLY_DATASET",
    "M4_WEEKLY_DATASET",
    "ETTH1",
    "ETTH2",
    "ETTM1",
    "ETTM2",
]
METRICS = [
    "carte_smape",
    "carte_pocid",
    "best_individual_smape",
    "best_individual_pocid",
    "best_attempt_smape",
    "best_attempt_pocid",
    "best_first10_smape",
    "best_first10_pocid",
    "u_val_smape",
    "u_val_pocid",
    "u_test_smape",
    "u_test_pocid",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida os oráculos do CARTE.")
    parser.add_argument("--attempts", default=ATTEMPTS)
    parser.add_argument("--universe", default=UNIVERSE)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    attempts = pd.read_csv(args.attempts)
    universe = pd.read_csv(args.universe, sep=";")
    rows = attempts.merge(
        universe[
            [
                "dataset",
                "dataset_index",
                "u_val_smape",
                "u_val_pocid",
                "u_test_smape",
                "u_test_pocid",
                "n_specs",
            ]
        ],
        on=["dataset", "dataset_index"],
        how="inner",
        validate="one_to_one",
    )

    os.makedirs(args.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(args.out_dir, "por_serie.csv"), index=False)

    out = []
    for dataset, group in rows.groupby("dataset"):
        item = {"dataset": dataset, "n_series": len(group)}
        item.update({m: group[m].mean() for m in METRICS})
        out.append(item)
    summary = pd.DataFrame(out)
    n_datasets = rows["dataset"].nunique()
    total_label = "Total 680" if len(rows) == 680 else f"Total {len(rows)}"
    media_label = "Media 7" if n_datasets == 7 else f"Media {n_datasets}"
    for label, group in ((total_label, rows), (media_label, None)):
        item = {"dataset": label, "n_series": len(rows) if group is not None else n_datasets}
        for m in METRICS:
            item[m] = group[m].mean() if group is not None else summary[m].mean()
        out.append(item)
    summary = pd.DataFrame(out)
    summary.to_csv(os.path.join(args.out_dir, "resumo.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
