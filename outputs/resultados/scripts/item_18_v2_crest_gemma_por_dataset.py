"""Item 18 v2 -- CREST com Gemma (orchestrator_react_v5_gemma26).

Mesma agregacao do item_18 (qwen). `agent_model` vem dos metadados salvos em
cada artefato JSON (config.combinator.model e decision.loop.llm_model), nao de
um nome presumido; o script aborta se os artefatos discordarem entre si.

Saida: outputs/resultados/item_18/crest_gemma_por_dataset.csv
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def confirmed_model_name() -> str:
    pattern = os.path.join(C.RESULTADOS_BASE, C.ORCHESTRATOR_GEMMA, "llm_artifacts", "*", "*.json")
    names = set()
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        names.add(d["config"]["combinator"]["model"])
        names.add(d["decision"]["loop"]["llm_model"])
    if len(names) != 1:
        raise SystemExit(f"nome de modelo inconsistente nos artefatos: {sorted(names)}")
    return names.pop()


def main():
    model = confirmed_model_name()
    pairs = [(C.ORCHESTRATOR_GEMMA, "CREST", "CREST_gemma")]
    df, warnings = C.build_metrics_table(C.DATASETS, pairs)
    df["agent_model"] = model

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_18")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "crest_gemma_por_dataset.csv")
    df.to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"modelo confirmado: {model}")
    print(f"salvo em {out_path} ({len(df)} linhas, {len(warnings)} avisos de validacao)")


if __name__ == "__main__":
    main()
