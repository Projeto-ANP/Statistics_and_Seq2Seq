"""Item 07 -- MCM agregada entre todos os conjuntos de dados.

Mesma orientacao do item 05 (usa multi_comp_matrix.MCM.compare, a mesma
convencao de chamada de combinations/multi_comparison_matrix.py, sem
reimplementar a comparacao estatistica), mas com um unico conjunto de
entrada concatenando as series dos 7 datasets. melhor_individual/
melhor_estatica sao escolhidos globalmente (nao por dataset). Gera uma
figura para SMAPE e outra para POCID (sem RMSE).

Requer o ambiente conda `metrics`:
    conda run -n metrics python outputs/resultados/scripts/item_07_mcm_agregado.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd

try:
    from multi_comp_matrix import MCM
except ImportError as exc:
    raise SystemExit(
        "multi_comp_matrix nao encontrado no python atual. Rode com o ambiente "
        "conda 'metrics': conda run -n metrics python " + os.path.abspath(__file__)
    ) from exc

METRICS = [
    ("smape", "SMAPE", "increasing"),
    ("pocid", "POCID", "decreasing"),
]


def main():
    mapping, all_warnings = C.five_method_folder_map_global()

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_07")
    os.makedirs(out_dir, exist_ok=True)
    out_dir_sep = out_dir + os.sep

    pd.DataFrame([{
        "melhor_individual": mapping["melhor_individual"],
        "melhor_estatica": mapping["melhor_estatica"],
    }]).to_csv(os.path.join(out_dir, "melhor_global.csv"), index=False)

    # Rotulos genericos (melhor_individual/melhor_estatica) trocados pelo nome
    # de exibicao do modelo que de fato venceu globalmente. FFORMA/ADE/CREST
    # ja sao o nome de exibicao final -- nao renomear (renomear por
    # display_name(mapping[label]) aqui reintroduziria o nome de pasta bruto,
    # ex.: CREST -> orchestrator_react_v5).
    rename_map = {label: C.display_name(mapping[label]) for label in ("melhor_individual", "melhor_estatica")}

    for metric, used_statistic, order_better in METRICS:
        wide, warns = C.build_wide_population(C.DATASETS, mapping, metric)
        wide = wide[C.FIVE_METHOD_LABELS]
        wide = wide.rename(columns=rename_map)
        all_warnings.extend(warns)

        savename = f"mcm_agregado_{metric}"
        input_path = os.path.join(out_dir, f"mcm_input_agregado_{metric}.csv")
        wide.to_csv(input_path, index=True)

        mcm_df = wide.reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MCM.compare(
                df_results=mcm_df,
                output_dir=out_dir_sep,
                png_savename=savename,
                pdf_savename=savename,
                csv_savename=savename,
                used_statistic=used_statistic,
                order_better=order_better,
                **C.mcm_plot_kwargs(),
            )
        print(f"MCM agregada ({metric}) salva em {out_dir}/{savename}.png")

    C.merge_validation_warnings(all_warnings)


if __name__ == "__main__":
    main()
