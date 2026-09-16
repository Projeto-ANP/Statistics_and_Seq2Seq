"""Passo 3 -- Testar a hipotese de que a divergencia e especifica do CatBoost.

Compara, por dataset, a taxa media de divergencia (media simples entre as 7
variantes de cada familia) entre CatBoost e RF.

Saida: outputs/diagnostico/passo3_catboost_vs_rf.csv
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
                "familia": D.FAMILY_OF[folder],
                "pct_divergentes": n_div / n_series * 100.0 if n_series else 0.0,
            })
    base = pd.DataFrame(rows)

    pivot = base.groupby(["dataset", "familia"])["pct_divergentes"].mean().unstack("familia")
    result = pd.DataFrame({
        "dataset": pivot.index,
        "pct_divergentes_catboost_medio": pivot["catboost"].values,
        "pct_divergentes_rf_medio": pivot["rf"].values,
    }).reset_index(drop=True)
    result = result.set_index("dataset").loc[D.C.DATASETS].reset_index()

    out_dir = D.DIAGNOSTICO_BASE
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "passo3_catboost_vs_rf.csv")
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nsalvo em {out_path}")


if __name__ == "__main__":
    main()
