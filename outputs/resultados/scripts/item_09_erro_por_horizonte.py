"""Item 09 -- Erro por passo do horizonte.

SMAPE ponto a ponto entre test e predictions (alinhados por posicao), para
cada um dos 5 metodos do item 07, agregado por posicao no horizonte entre
todas as series de todos os datasets.

O horizonte real difere por dataset (ANP_MONTHLY=12, NN5_WEEKLY_DATASET=8,
M4_WEEKLY_DATASET=13, os 4 ETT=24) -- cada serie so contribui um ponto por
passo ate o seu proprio horizonte, entao a media de cada horizon_step usa
so os datasets que de fato alcancam aquele passo (groupby do pandas so
enxerga as linhas que existem, nao preenche com NaN nem usa um subconjunto
silenciosamente). Por isso n_series cai de 680 (passos 1-8, todos os 7
datasets) para 569 (9-12, NN5 sai), 387 (13, ANP sai) e so 28 (14-24, so os
4 ETT restam) -- reportado explicitamente na coluna n_series abaixo.

Saida: outputs/resultados/item_09/erro_por_horizonte.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd


def pointwise_smape(test_vals, pred_vals):
    out = []
    for t, p in zip(test_vals, pred_vals):
        denom = abs(t) + abs(p)
        out.append(0.0 if denom == 0 else 2.0 * abs(p - t) / denom)
    return out


def main():
    mapping, warnings = C.five_method_folder_map_global()
    long_df, warns2 = C.build_long_population(C.DATASETS, mapping, extra_columns=["test", "predictions"])
    all_warnings = warnings + warns2

    records = []
    for row in long_df.itertuples(index=False):
        test_vals = C.parse_num_list(row.test)
        pred_vals = C.parse_num_list(row.predictions)
        n = min(len(test_vals), len(pred_vals))
        smapes = pointwise_smape(test_vals[:n], pred_vals[:n])
        for step, s in enumerate(smapes, start=1):
            records.append({"method": row.method, "horizon_step": step, "smape": s})

    points = pd.DataFrame(records)
    result = points.groupby(["method", "horizon_step"], as_index=False)["smape"].agg(
        smape_medio="mean", n_series="count"
    )
    result["method"] = pd.Categorical(result["method"], categories=C.FIVE_METHOD_LABELS, ordered=True)
    result = result.sort_values(["method", "horizon_step"]).reset_index(drop=True)

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_09")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "erro_por_horizonte.csv")
    result.to_csv(out_path, index=False)

    C.merge_validation_warnings(all_warnings)
    print(f"salvo em {out_path} ({len(result)} linhas)")


if __name__ == "__main__":
    main()
