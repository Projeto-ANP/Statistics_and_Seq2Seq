"""Passo 2 -- Quantificar quantas series divergem, por modelo e conjunto de dados.

Serie "divergente" em (dataset, regressor): max(abs(predictions)) > 100x
max(abs(train)) (ou, na ausencia de coluna de treino no CSV -- o caso de
todos os CSVs usados aqui --, 100x max(abs(test)) da mesma serie).

Saida: outputs/diagnostico/passo2_contagem_divergencia.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_lib as D

import pandas as pd


def main():
    rows = []
    for folder in D.ALL_REGRESSORS:
        for dataset in D.C.DATASETS:
            df = D.C.load_filtered(folder, dataset)
            flags = df.apply(D.is_divergent, axis=1)
            n_series = len(df)
            n_div = int(flags.sum())
            rows.append({
                "dataset": dataset,
                "regressor": folder,
                "n_series": n_series,
                "n_divergentes": n_div,
                "pct_divergentes": n_div / n_series * 100.0 if n_series else 0.0,
            })

    out_dir = D.DIAGNOSTICO_BASE
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "passo2_contagem_divergencia.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"salvo em {out_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
