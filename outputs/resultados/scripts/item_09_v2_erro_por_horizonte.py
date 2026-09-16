"""Item 09 v2 -- Erro por passo do horizonte, com POCID e desvio-padrao.

Mesma logica de agregacao do item_09 original (item_09_erro_por_horizonte.py):
5 metodos escolhidos globalmente (pick_best_global, via
five_method_folder_map_global), populacao de series por horizon_step
restrita a quem de fato alcanca aquele passo (nada de preencher com NaN nem
usar subconjunto silencioso -- groupby so agrega o que existe).

Adiciona:
- smape_desvio_padrao: desvio-padrao do SMAPE ponto a ponto, mesma populacao
  de smape_medio no passo.
- pocid_medio / pocid_desvio_padrao: POCID ponto a ponto (1 se
  sign(pred[t]-pred[t-1]) == sign(test[t]-test[t-1]), 0 caso contrario --
  mesma formula de all_functions.py:134-141 e orchestrator_react/metrics.py:
  62-71, so que aplicada ponto a ponto em vez de agregada na serie toda).

Passo 0 do POCID (t=1 nao tem passo anterior dentro da propria janela de
teste): usa o ultimo valor da SEGUNDA janela de validacao mais recente de
cada serie (ver common.pre_test_reference_values). Essa janela ja esta no
CSV bruto, antes do filtro de load_filtered -- nao depende de nenhum
arquivo externo. Cross-validado contra a reconstrucao via .tsf original nos
datasets onde essa reconstrucao bate (identico a 6 casas decimais); .tsf
sozinho falhou para ETTM1/ETTM2 (o .tsf atual em forecasting_datasets nao
bate com o `test` salvo nos resultados desses dois datasets -- achado
registrado no RESUMO deste item).

Saida: outputs/resultados/item_09/erro_por_horizonte_v2.csv
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


def pointwise_pocid(test_vals, pred_vals, step0):
    """1 ponto por horizon_step (nao 1 a menos, como no pocid agregado de
    all_functions.py): o primeiro ponto usa `step0` (ultimo valor antes do
    inicio do teste) como referencia anterior."""
    out = []
    prev_test = step0
    prev_pred = step0
    for t, p in zip(test_vals, pred_vals):
        d_real = t - prev_test
        d_pred = p - prev_pred
        out.append(1 if d_real * d_pred > 0 else 0)
        prev_test, prev_pred = t, p
    return out


def main():
    mapping, warnings = C.five_method_folder_map_global()
    long_df, warns2 = C.build_long_population(C.DATASETS, mapping, extra_columns=["test", "predictions"])
    all_warnings = warnings + warns2

    step0_by_dataset = {dataset: C.pre_test_reference_values(dataset) for dataset in C.DATASETS}

    records = []
    for row in long_df.itertuples(index=False):
        test_vals = C.parse_num_list(row.test)
        pred_vals = C.parse_num_list(row.predictions)
        n = min(len(test_vals), len(pred_vals))
        test_vals, pred_vals = test_vals[:n], pred_vals[:n]

        smapes = pointwise_smape(test_vals, pred_vals)

        dataset_index = int(row.serie.rsplit("_", 1)[-1])
        step0 = step0_by_dataset[row.dataset][dataset_index]
        pocids = pointwise_pocid(test_vals, pred_vals, step0)

        for step, (s, d) in enumerate(zip(smapes, pocids), start=1):
            records.append({"method": row.method, "horizon_step": step, "smape": s, "pocid": d})

    points = pd.DataFrame(records)
    result = points.groupby(["method", "horizon_step"], as_index=False).agg(
        smape_medio=("smape", "mean"),
        smape_desvio_padrao=("smape", "std"),
        pocid_medio=("pocid", "mean"),
        pocid_desvio_padrao=("pocid", "std"),
        n_series=("smape", "count"),
    )
    result["method"] = pd.Categorical(result["method"], categories=C.FIVE_METHOD_LABELS, ordered=True)
    result = result.sort_values(["method", "horizon_step"]).reset_index(drop=True)
    result = result[[
        "method", "horizon_step", "smape_medio", "smape_desvio_padrao",
        "pocid_medio", "pocid_desvio_padrao", "n_series",
    ]]

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_09")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "erro_por_horizonte_v2.csv")
    result.to_csv(out_path, index=False)

    C.merge_validation_warnings(all_warnings)
    print(f"salvo em {out_path} ({len(result)} linhas)")


if __name__ == "__main__":
    main()
