"""Item 13 -- Redutibilidade.

Usa a coluna equivalent_to_pool_mean (booleana) de orchestrator_react_v5.

Saida: outputs/resultados/item_13/redutibilidade.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd


def main():
    warnings = []
    rows = []

    for dataset in C.DATASETS:
        df = C.load_filtered(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        n_series = len(df)
        n_redutiveis = int(df["equivalent_to_pool_mean"].sum())
        rows.append({
            "dataset": dataset,
            "n_series": n_series,
            "n_redutiveis": n_redutiveis,
            "pct_redutiveis": n_redutiveis / n_series * 100.0,
        })

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_13")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "redutibilidade.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path}")


if __name__ == "__main__":
    main()
