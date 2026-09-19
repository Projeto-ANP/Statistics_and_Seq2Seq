"""Interpretabilidade -- casos de sucesso e fracasso do CREST (orchestrator_react_v5).

Diferente do item_15 (foco na trajetoria de raciocinio), aqui o foco e nos
NUMEROS da previsao: serie real, previsao de cada modelo individual que
compos a estrategia final vencedora, previsao combinada, metodo/pesos.

Selecao dos candidatos (ver nota no script -- criterio documentado, nao
apenas o extremo absoluto):
- Sucesso: ANP_MONTHLY, dataset_index=103. SMAPE=0.0273 (3o melhor de 182
  series do dataset), estrategia "weighted" com 5 modelos efetivos (DWT_rf,
  FT_catboost, FT_rf, catboost, rf) -- nao e so "escolheu o melhor modelo".
- Fracasso: NN5_WEEKLY_DATASET, dataset_index=109. SMAPE=0.3898 (3o pior de
  111 series do dataset), estrategia "weighted" com 8 modelos efetivos --
  evita de proposito as duas series de M4_WEEKLY_DATASET (idx 208 e 75) que
  tambem estao entre as piores do CREST, porque essas duas coincidem com a
  patologia de alvo de treino corrompido ja documentada em
  outputs/diagnostico/RESUMO_passo7_rf_m4.md (um bug de normalizacao de
  janela, no upstream dos modelos individuais, nao do metodo de combinacao)
  -- usa-las aqui confundiria "a combinacao nao ajudou" com "o dado de
  entrada de alguns modelos individuais ja estava corrompido antes de
  qualquer combinacao".

Saida:
  outputs/resultados/interpretabilidade/caso_sucesso.json
  outputs/resultados/interpretabilidade/caso_fracasso.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CASES = {
    "caso_sucesso": ("ANP_MONTHLY", 103),
    "caso_fracasso": ("NN5_WEEKLY_DATASET", 109),
}


def extract_case(dataset: str, dataset_index: int) -> dict:
    df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
    row = df[df["dataset_index"] == dataset_index].iloc[0]

    test = C.parse_num_list(row["test"])
    final_pred = C.parse_num_list(row["predictions"])
    effective_models = json.loads(row["effective_models"])
    best_strategy_params = json.loads(row["best_strategy_params"])
    weights_by_horizon = json.loads(row["weights_by_horizon"]) if C.pd.notna(row["weights_by_horizon"]) else None

    modelos_individuais = {}
    for model in effective_models:
        mdf = C.load_filtered(model, dataset)
        mrow = mdf[mdf["dataset_index"] == dataset_index].iloc[0]
        modelos_individuais[model] = C.parse_num_list(mrow["predictions"])

    pesos_por_modelo = None
    if row["best_strategy_method"] == "weighted" and weights_by_horizon is not None:
        pesos_por_modelo = {
            horizon_step: {m: w.get(m) for m in effective_models}
            for horizon_step, w in weights_by_horizon.items()
        }

    return {
        "dataset": dataset,
        "dataset_index": int(dataset_index),
        "test": test,
        "modelos_individuais_predictions": modelos_individuais,
        "predicao_combinada_final": final_pred,
        "metodo_agregacao": row["best_strategy_method"],
        "estrategia_params": best_strategy_params,
        "pesos_por_modelo": pesos_por_modelo,
        "smape_final": float(row["smape"]),
        "pocid_final": float(row["pocid"]),
        "justificativa_final": row["justificativa_final"],
        "origin": "origin_" + row["description_json"].get("origin", ""),
    }


def main():
    out_dir = os.path.join(C.OUTPUTS_BASE, "interpretabilidade")
    os.makedirs(out_dir, exist_ok=True)

    for name, (dataset, dataset_index) in CASES.items():
        case = extract_case(dataset, dataset_index)
        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(case, fh, ensure_ascii=False, indent=2)
        print(f"{name}: {dataset} idx={dataset_index} smape={case['smape_final']:.4f} "
              f"metodo={case['metodo_agregacao']} n_modelos={len(case['modelos_individuais_predictions'])} "
              f"-> {out_path}")


if __name__ == "__main__":
    main()
