"""Item 10 -- Frequencia de uso do catalogo de acoes.

Le tools_called de todas as series de orchestrator_react_v5 (todos os
datasets), explode a lista de objetos, conta por nome de acao (tool) onde
ok = true.

Saida: outputs/resultados/item_10/frequencia_acoes.csv
  colunas: action, n_chamadas, dataset (dataset = "TOTAL" para o agregado
  entre todos os datasets).
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd


def main():
    warnings = []
    per_dataset_counts = {}
    total_counts = Counter()

    for dataset in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        counts = Counter()
        for calls in df["tools_called_json"]:
            for call in calls:
                if call.get("ok") is True:
                    counts[call.get("tool")] += 1
        per_dataset_counts[dataset] = counts
        total_counts.update(counts)

    rows = []
    for dataset in C.DATASETS:
        for action, n in sorted(per_dataset_counts[dataset].items()):
            rows.append({"action": action, "n_chamadas": n, "dataset": dataset})
    for action, n in sorted(total_counts.items()):
        rows.append({"action": action, "n_chamadas": n, "dataset": "TOTAL"})

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_10")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "frequencia_acoes.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
