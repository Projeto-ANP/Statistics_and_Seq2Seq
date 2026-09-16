"""Item 03 -- Metricas de FFORMA e ADE, por conjunto de dados.

Saida: outputs/resultados/item_03/fforma_ade_por_dataset.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    pairs = [(folder, folder) for folder in C.FFORMA_ADE]
    df, warnings = C.build_metrics_table(C.DATASETS, pairs)

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_03")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fforma_ade_por_dataset.csv")
    df.to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path} ({len(df)} linhas, {len(warnings)} avisos de validacao)")


if __name__ == "__main__":
    main()
