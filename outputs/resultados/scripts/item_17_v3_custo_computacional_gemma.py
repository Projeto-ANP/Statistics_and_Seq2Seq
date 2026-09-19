"""Item 17 v3 -- Custo computacional do CREST com Gemma (orchestrator_react_v5_gemma26).

Mesma logica do item_17_v2 (qwen): description_json.loop.elapsed_s por serie,
media e desvio-padrao (ddof=1) por dataset. Reporta tambem quantas series cada
dataset tem processadas, para evidenciar execucao parcial.

Saida: outputs/resultados/item_17/custo_computacional_gemma.csv
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
        df = C.load_orchestrator(C.ORCHESTRATOR_GEMMA, dataset)
        warning = C.check_series_count(dataset, "CREST_gemma", len(df))
        if warning:
            warnings.append(warning)
        elapsed = df["description_json"].map(lambda d: C.dget(d, "loop", "elapsed_s")).astype(float)
        rows.append({
            "dataset": dataset,
            "tempo_medio_s": elapsed.mean(),
            "desvio_padrao_s": elapsed.std(ddof=1),
        })
        print(f"{dataset}: {len(df)} series")

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_17")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "custo_computacional_gemma.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path}")


if __name__ == "__main__":
    main()
