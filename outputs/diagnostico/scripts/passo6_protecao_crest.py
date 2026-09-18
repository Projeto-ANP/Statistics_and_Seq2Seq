"""Passo 6 (passo 1 do prompt "Por que o CREST nao herda a divergencia do
CatBoost nos ETT") -- levanta o que o CREST fez, serie a serie, nas 26
series onde catboost diverge nos 4 datasets ETT (de
outputs/resultados/diagnostico/passo5_contexto_divergencia.csv).

Colunas pedidas no prompt: dataset, dataset_index, catboost_no_pool_efetivo
(booleano -- True = catboost ESTA no effective_models dessa serie),
best_strategy_method, peso_catboost (peso de catboost em weights_by_horizon
da janela 0 se a estrategia final for "weighted" e catboost estiver la,
senao vazio -- horizonte 0 pela mesma convencao que effective_models usa,
ver orchestrator_react/pipeline.py:effective_weights/SeriesOutcome.effective_models).

Colunas extras (necessarias para o passo 2 contar "a partir da tabela do
passo 1", conforme pedido, sem precisar reabrir os CSVs de origem):
origin (baseline = uma das 9 baselines semeadas em orchestrator_react/pool.py
antes do loop ReAct comecar; agent = a melhor tentativa veio da trajetoria do
agente), pool_nominal_tamanho (len(selected_base_models)), e
acao_pruning_na_trajetoria (True se tools_called daquela serie contem uma
chamada bem-sucedida a prune_redundant/select_stable/select_top_k -- ver
passo 2 para a leitura).

Saida: outputs/diagnostico/passo6_protecao_crest.csv
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_lib as D

import pandas as pd

PASSO5_PATH = os.path.join(D.REPO_ROOT, "outputs", "resultados", "diagnostico", "passo5_contexto_divergencia.csv")
ETT_DATASETS = ["ETTH1", "ETTH2", "ETTM1", "ETTM2"]
POOL_SHAPING_TOOLS = {"prune_redundant", "select_stable", "select_top_k"}


def divergent_catboost_ett_series():
    df = pd.read_csv(PASSO5_PATH)
    df = df[(df["dataset"].isin(ETT_DATASETS)) & (df["regressor"] == "catboost")]
    return list(df[["dataset", "dataset_index"]].itertuples(index=False, name=None))


def main():
    pairs = divergent_catboost_ett_series()
    orchestrator_cache = {ds: D.C.load_orchestrator(D.C.ORCHESTRATOR_GPTOSS, ds) for ds in ETT_DATASETS}

    rows = []
    for dataset, dataset_index in pairs:
        df = orchestrator_cache[dataset]
        row = df[df["dataset_index"] == dataset_index].iloc[0]

        effective_models = json.loads(row["effective_models"])
        selected_base_models = json.loads(row["selected_base_models"])
        best_strategy_method = row["best_strategy_method"]
        origin = row["description_json"].get("origin")

        peso_catboost = ""
        if best_strategy_method == "weighted" and pd.notna(row["weights_by_horizon"]):
            weights = json.loads(row["weights_by_horizon"])
            w0 = weights.get("0", {})
            if "catboost" in w0:
                peso_catboost = w0["catboost"]

        acao_pruning = any(
            call.get("tool") in POOL_SHAPING_TOOLS and call.get("ok") is True
            for call in row["tools_called_json"]
        )

        rows.append({
            "dataset": dataset,
            "dataset_index": int(dataset_index),
            "catboost_no_pool_efetivo": "catboost" in effective_models,
            "best_strategy_method": best_strategy_method,
            "peso_catboost": peso_catboost,
            "origin": origin,
            "pool_nominal_tamanho": len(selected_base_models),
            "catboost_no_pool_nominal": "catboost" in selected_base_models,
            "acao_pruning_na_trajetoria": acao_pruning,
        })

    result = pd.DataFrame(rows)
    out_dir = D.DIAGNOSTICO_BASE
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "passo6_protecao_crest.csv")
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nsalvo em {out_path} ({len(result)} linhas)")


if __name__ == "__main__":
    main()
