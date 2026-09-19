"""Item 06 -- Ranking consolidado entre conjuntos de dados.

Para cada dataset, ordena {melhor_individual, melhor_estatica, FFORMA, ADE,
CREST} pelo SMAPE medio (menor = melhor), calculado a partir dos itens 01-04.

Saida: outputs/resultados/item_06/ranking_consolidado.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd


def main():
    best_ind, warn_ind, df_ind = C.pick_best_per_dataset("individual")
    best_est, warn_est, df_est = C.pick_best_per_dataset("estatica")
    fforma_ade_df, warn_fa = C.build_metrics_table(C.DATASETS, [(f, f) for f in C.FFORMA_ADE])
    crest_df, warn_crest = C.build_metrics_table(C.DATASETS, [(C.ORCHESTRATOR_GPTOSS, "CREST")])

    all_warnings = warn_ind + warn_est + warn_fa + warn_crest

    rows = []
    for dataset in C.DATASETS:
        smape_map = {
            "melhor_individual": df_ind.loc[df_ind["dataset"] == dataset, "smape"].min(),
            "melhor_estatica": df_est.loc[df_est["dataset"] == dataset, "smape"].min(),
            "FFORMA": fforma_ade_df.loc[
                (fforma_ade_df["dataset"] == dataset) & (fforma_ade_df["regressor"] == "FFORMA"), "smape"
            ].iloc[0],
            "ADE": fforma_ade_df.loc[
                (fforma_ade_df["dataset"] == dataset) & (fforma_ade_df["regressor"] == "ADE"), "smape"
            ].iloc[0],
            "CREST": crest_df.loc[crest_df["dataset"] == dataset, "smape"].iloc[0],
        }
        ranked = sorted(smape_map.items(), key=lambda kv: kv[1])
        for rank, (method, _) in enumerate(ranked, start=1):
            rows.append({"dataset": dataset, "method": method, "rank": rank})

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_06")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ranking_consolidado.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(all_warnings)
    print(f"salvo em {out_path}")


if __name__ == "__main__":
    main()
