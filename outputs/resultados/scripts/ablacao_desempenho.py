"""Ablacao (CREST sem agente) -- desempenho por dataset, formato do item_04.

Le orchestrator_baseline_v1 (so a fase deterministica) e agrega com a mesma
funcao de item_04 (aggregate_metrics), regressor = CREST_sem_agente.

Saida: outputs/resultados/ablacao/desempenho_sem_agente_por_dataset.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    pairs = [(C.ORCHESTRATOR_BASELINE, "CREST_sem_agente")]
    df, warnings = C.build_metrics_table(C.DATASETS, pairs)

    out_dir = os.path.join(C.OUTPUTS_BASE, "ablacao")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "desempenho_sem_agente_por_dataset.csv")
    df.to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(df.to_string(index=False))
    print(f"salvo em {out_path}")


if __name__ == "__main__":
    main()
