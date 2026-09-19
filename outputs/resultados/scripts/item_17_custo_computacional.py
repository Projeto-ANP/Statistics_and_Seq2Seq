"""Item 17 -- Custo computacional.

Usa description_json.loop.elapsed_s de todas as series de orchestrator_react_v5,
agregado por dataset.

Saida: outputs/resultados/item_17/custo_computacional.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import numpy as np
import pandas as pd


def main():
    warnings = []
    rows = []

    for dataset in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        elapsed = df["description_json"].map(lambda d: C.dget(d, "loop", "elapsed_s")).astype(float)
        rows.append({
            "dataset": dataset,
            "elapsed_s_medio": elapsed.mean(),
            "elapsed_s_desvio_padrao": elapsed.std(ddof=1),
        })

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_17")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "custo_computacional.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path}")


if __name__ == "__main__":
    main()
