"""Item 16 -- Distribuicao da melhoria relativa.

Para cada serie de orchestrator_react_v5: diferenca entre o score da
estrategia final (description_json.validation.score) e o menor score entre
as estrategias em baseline_results_json.seeded.

Saida: outputs/resultados/item_16/melhoria_relativa.csv
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
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        for row in df.itertuples(index=False):
            score_final = C.dget(row.description_json, "validation", "score")
            seeded = C.dget(row.baseline_results_json_parsed, "seeded", default={})
            seeded_scores = [info["score"] for info in seeded.values() if "score" in info]
            melhor_score_semeado = min(seeded_scores) if seeded_scores else None

            diferenca = (
                score_final - melhor_score_semeado
                if score_final is not None and melhor_score_semeado is not None
                else None
            )
            rows.append({
                "dataset": dataset,
                "dataset_index": int(row.dataset_index),
                "score_final": score_final,
                "melhor_score_semeado": melhor_score_semeado,
                "diferenca": diferenca,
            })

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_16")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "melhoria_relativa.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
