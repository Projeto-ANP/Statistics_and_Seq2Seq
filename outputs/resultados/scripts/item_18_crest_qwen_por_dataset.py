"""Item 18 -- Efeito do modelo de linguagem (gpt-oss vs. Qwen3).

Repete a agregacao do item 04, lendo de orchestrator_react_v5_qwen.
Saida: outputs/resultados/item_18/crest_qwen_por_dataset.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    pairs = [(C.ORCHESTRATOR_QWEN, "CREST", "CREST_qwen3")]
    df, warnings = C.build_metrics_table(C.DATASETS, pairs)
    df["agent_model"] = "qwen3"

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_18")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "crest_qwen_por_dataset.csv")
    df.to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path} ({len(df)} linhas, {len(warnings)} avisos de validacao)")


if __name__ == "__main__":
    main()
