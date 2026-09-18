"""Item 12 -- Confianca autodeclarada vs. veredito estatistico.

Para series de orchestrator_react_v5 com agent_accepted_id != null (dentro de
description_json.loop): accept_confidence, selection_verdict, selection_margin,
selection_dm_pvalue (colunas top-level do CSV de origem) e bootstrap_reliable
(dentro de description_json.selection_confidence).

Saidas:
  outputs/resultados/item_12/confianca_vs_veredito.csv
  outputs/resultados/item_12/sem_confianca_contagem.csv (series com
    agent_accepted_id nulo, por dataset)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd


def main():
    warnings = []
    rows = []
    sem_confianca_rows = []

    for dataset in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        agent_accepted_id = df["description_json"].map(lambda d: C.dget(d, "loop", "agent_accepted_id"))
        bootstrap_reliable = df["description_json"].map(
            lambda d: C.dget(d, "selection_confidence", "bootstrap_reliable")
        )

        has_confidence = agent_accepted_id.notna()
        sub = df.loc[has_confidence, [
            "dataset_index", "accept_confidence", "selection_verdict",
            "selection_margin", "selection_dm_pvalue",
        ]].copy()
        sub.insert(0, "dataset", dataset)
        sub["bootstrap_reliable"] = bootstrap_reliable.loc[has_confidence].values
        rows.append(sub)

        sem_confianca_rows.append({
            "dataset": dataset,
            "n_series_sem_confianca": int((~has_confidence).sum()),
            "n_series_total": len(df),
        })

    result = pd.concat(rows, axis=0, ignore_index=True)

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_12")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "confianca_vs_veredito.csv")
    result.to_csv(out_path, index=False)

    sem_confianca_path = os.path.join(out_dir, "sem_confianca_contagem.csv")
    pd.DataFrame(sem_confianca_rows).to_csv(sem_confianca_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path} ({len(result)} linhas) e {sem_confianca_path}")


if __name__ == "__main__":
    main()
