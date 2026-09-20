"""Item 11 -- Motivo de encerramento do ciclo.

Classifica cada serie de orchestrator_react_v5 em {aceite, estagnacao,
orcamento_esgotado}, usando description_json.loop (early_stopped,
stop_reason, agent_accepted_id, iterations_used -- os mesmos dados das
colunas top-level react_early_stopped/react_iterations_used, so que lidos
de dentro do JSON estruturado).

Series cujo stop_reason nao se encaixa em nenhuma das 3 categorias (ex.:
falha de LLM a meio do loop) caem em "outro", para nao forcar uma
classificacao incorreta.

Saida: outputs/resultados/item_11/motivo_encerramento.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd

# orchestrator_react/config.py: ReactConfig.max_iterations default = 12
MAX_ITERATIONS = 12


def classify(loop: dict) -> str:
    if loop.get("agent_accepted_id") is not None:
        return "aceite"
    stop_reason = loop.get("stop_reason") or ""
    if stop_reason.startswith("no improvement"):
        return "estagnacao"
    if stop_reason == "iteration_budget_exhausted":
        return "orcamento_esgotado"
    if not loop.get("early_stopped", False) and loop.get("iterations_used", 0) >= MAX_ITERATIONS:
        return "orcamento_esgotado"
    return "outro"


def main():
    warnings = []
    rows = []

    for dataset in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        motivos = df["description_json"].map(lambda d: classify(C.dget(d, "loop", default={})))
        n_series = len(df)
        for motivo, n in motivos.value_counts().items():
            rows.append({
                "dataset": dataset,
                "motivo": motivo,
                "n_series": int(n),
                "pct_series": float(n) / n_series * 100.0,
            })

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_11")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "motivo_encerramento.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
