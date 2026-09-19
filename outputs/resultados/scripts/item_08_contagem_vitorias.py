"""Item 08 -- Contagem de vitorias.

Mesma populacao do item 07 (5 metodos, escolhidos globalmente, todos os
datasets). Para cada serie, o metodo com menor SMAPE vence.

Saida: outputs/resultados/item_08/contagem_vitorias.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd


def main():
    mapping, warnings = C.five_method_folder_map_global()
    long_df, warns2 = C.build_long_population(C.DATASETS, mapping, extra_columns=["smape"])
    all_warnings = warnings + warns2

    wide = long_df.pivot(index="serie", columns="method", values="smape")[C.FIVE_METHOD_LABELS]
    winner = wide.idxmin(axis=1)

    total = len(winner)
    counts = winner.value_counts().reindex(C.FIVE_METHOD_LABELS, fill_value=0)
    rows = [
        {"method": method, "n_vitorias": int(n), "pct_vitorias": float(n) / total * 100.0}
        for method, n in counts.items()
    ]

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_08")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "contagem_vitorias.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(all_warnings)
    print(f"salvo em {out_path} (populacao total: {total} series)")


if __name__ == "__main__":
    main()
