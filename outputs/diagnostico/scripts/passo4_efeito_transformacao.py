"""Passo 4 -- Verificar se a transformacao (wavelet/Fourier) piora ou nao o problema.

Dentro de cada familia, compara a taxa media de divergencia entre as 3
representacoes: original (sem transformacao), concatenada (features
originais + transformadas) e only (so as features transformadas).

Saida: outputs/diagnostico/passo4_efeito_transformacao.csv
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
                "familia": D.FAMILY_OF[folder],
                "representacao": D.REPRESENTACAO_OF[folder],
                "pct_divergentes": n_div / n_series * 100.0 if n_series else 0.0,
            })
    base = pd.DataFrame(rows)

    result = (
        base.groupby(["dataset", "familia", "representacao"], as_index=False)["pct_divergentes"]
        .mean()
        .rename(columns={"pct_divergentes": "pct_divergentes_medio"})
    )
    result["dataset"] = pd.Categorical(result["dataset"], categories=D.C.DATASETS, ordered=True)
    result["representacao"] = pd.Categorical(
        result["representacao"], categories=["original", "concatenada", "only"], ordered=True
    )
    result = result.sort_values(["dataset", "familia", "representacao"]).reset_index(drop=True)

    out_dir = D.DIAGNOSTICO_BASE
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "passo4_efeito_transformacao.csv")
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nsalvo em {out_path}")


if __name__ == "__main__":
    main()
