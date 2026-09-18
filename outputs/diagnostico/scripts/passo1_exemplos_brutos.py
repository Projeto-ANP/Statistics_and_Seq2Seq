"""Passo 1 -- Confirmar se e a previsao em si, ou o calculo da metrica.

O prompt pede "os cinco pares (dataset, regressor) com o maior RMSE medio"
mas tambem lista 5 pares especificos como piso minimo ("pelo menos"). Os dois
criterios divergem na pratica: o RMSE medio real mais alto esta 100% em
M4_WEEKLY_DATASET x variantes catboost (ate 1e133), enquanto ETTH1/ETTM1/
ETTM2 x catboost (citados no prompt) nem entram no top-10. Em vez de escolher
um dos dois em silencio, este script investiga a uniao: o top-5 real por RMSE
medio (dentro do universo catboost-em-todo-dataset + rf-so-em-M4_WEEKLY) mais
os 5 pares citados nominalmente no prompt que ainda nao estejam nesse top-5.

Saida: outputs/diagnostico/passo1_exemplos_brutos.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_lib as D

import pandas as pd

PROMPT_NAMED_PAIRS = [
    ("M4_WEEKLY_DATASET", "ONLY_DWT_catboost"),
    ("M4_WEEKLY_DATASET", "FT_catboost"),
    ("ETTH1", "catboost"),
    ("ETTM1", "catboost"),
    ("ETTM2", "catboost"),
]


def candidate_universe():
    pairs = []
    for folder in D.CATBOOST_FAMILY:
        for dataset in D.C.DATASETS:
            pairs.append((dataset, folder))
    for folder in D.RF_FAMILY:
        pairs.append(("M4_WEEKLY_DATASET", folder))
    return pairs


def main():
    rows = []
    for dataset, folder in candidate_universe():
        df = D.C.load_filtered(folder, dataset)
        rows.append({"dataset": dataset, "regressor": folder, "rmse_medio": df["rmse"].mean()})
    ranking = pd.DataFrame(rows).sort_values("rmse_medio", ascending=False)
    top5_real = list(ranking.head(5)[["dataset", "regressor"]].itertuples(index=False, name=None))

    investigated_pairs = list(top5_real)
    for pair in PROMPT_NAMED_PAIRS:
        if pair not in investigated_pairs:
            investigated_pairs.append(pair)

    results = []
    for dataset, folder in investigated_pairs:
        df = D.C.load_filtered(folder, dataset)
        worst = df.loc[df["rmse"].idxmax()]

        test_vals = D.C.parse_num_list(worst["test"])
        pred_vals = D.C.parse_num_list(worst["predictions"])
        max_test = max(abs(v) for v in test_vals) if test_vals else 0.0
        max_pred = max(abs(v) for v in pred_vals) if pred_vals else 0.0

        if max_test > 0 and max_pred > 0:
            log10_ratio = __import__("math").log10(max_pred / max_test)
        else:
            log10_ratio = None

        results.append({
            "dataset": dataset,
            "regressor": folder,
            "dataset_index": int(worst["dataset_index"]),
            "rmse": float(worst["rmse"]),
            "mape": float(worst["mape"]),
            "smape": float(worst["smape"]),
            "test": test_vals,
            "predictions": pred_vals,
            "max_abs_test": max_test,
            "max_abs_predictions": max_pred,
            "log10_razao_pred_sobre_test": log10_ratio,
            "in_top5_rmse_medio_real": (dataset, folder) in top5_real,
            "in_lista_nomeada_no_prompt": (dataset, folder) in PROMPT_NAMED_PAIRS,
        })

    out_dir = D.DIAGNOSTICO_BASE
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "passo1_exemplos_brutos.json")
    payload = {
        "nota_metodologica": (
            "Uniao entre o top-5 real por RMSE medio (universo: familia catboost "
            "em todos os 7 datasets + familia rf so em M4_WEEKLY_DATASET) e os 5 "
            "pares citados nominalmente no prompt -- os dois criterios divergem, "
            "ver 'in_top5_rmse_medio_real' e 'in_lista_nomeada_no_prompt' por item."
        ),
        "top5_rmse_medio_real": [{"dataset": d, "regressor": r} for d, r in top5_real],
        "casos": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    for r in results:
        print(
            f"{r['dataset']:20s} {r['regressor']:20s} idx={r['dataset_index']:<4d} "
            f"rmse={r['rmse']:.3e}  max|test|={r['max_abs_test']:.3e}  "
            f"max|pred|={r['max_abs_predictions']:.3e}  "
            f"log10(pred/test)={r['log10_razao_pred_sobre_test']}"
        )
    print(f"\nsalvo em {out_path} ({len(results)} casos)")


if __name__ == "__main__":
    main()
