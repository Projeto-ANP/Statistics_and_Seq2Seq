"""Item 05 -- MCM por conjunto de dados.

Compara {melhor_individual, melhor_estatica, FFORMA, ADE, CREST} por dataset,
onde melhor_individual/melhor_estatica sao escolhidos pelo menor SMAPE medio
do item 01/02 naquele dataset. Usa multi_comp_matrix.MCM.compare (o mesmo
pacote chamado por combinations/multi_comparison_matrix.py) com a mesma
convencao de chamada desse script (ver build_results_df/_run_mcm nele) --
a comparacao estatistica nao e reimplementada aqui.

Requer o ambiente conda `metrics` (onde multi_comp_matrix esta instalado):
    conda run -n metrics python outputs/resultados/scripts/item_05_mcm_por_dataset.py
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


def main():
    mapping, base_warnings = C.five_method_folder_map_per_dataset()

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_05")
    os.makedirs(out_dir, exist_ok=True)
    out_dir_sep = out_dir + os.sep

    melhor_rows = []
    all_warnings = list(base_warnings)

    for dataset in C.DATASETS:
        per_label = mapping[dataset]
        melhor_rows.append({
            "dataset": dataset,
            "melhor_individual": per_label["melhor_individual"],
            "melhor_estatica": per_label["melhor_estatica"],
        })

        wide, warns = C.build_wide_population([dataset], {dataset: per_label}, "smape")
        wide = wide[C.FIVE_METHOD_LABELS]
        all_warnings.extend(warns)

        # Rotulos genericos (melhor_individual/melhor_estatica) trocados pelo
        # nome de exibicao do modelo que de fato venceu naquele dataset.
        # FFORMA/ADE/CREST ja sao o nome de exibicao final -- nao renomear
        # (renomear por display_name(per_label[label]) aqui reintroduziria o
        # nome de pasta bruto, ex.: CREST -> orchestrator_react_v5).
        rename_map = {label: C.display_name(per_label[label]) for label in ("melhor_individual", "melhor_estatica")}
        wide = wide.rename(columns=rename_map)

        input_path = os.path.join(out_dir, f"mcm_input_{dataset}.csv")
        wide.to_csv(input_path, index=True)

        mcm_df = wide.reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MCM.compare(
                df_results=mcm_df,
                output_dir=out_dir_sep,
                png_savename=f"mcm_{dataset}",
                pdf_savename=f"mcm_{dataset}",
                csv_savename=f"mcm_{dataset}",
                used_statistic="SMAPE",
                order_better="increasing",
                **C.mcm_plot_kwargs(),
            )
        print(f"MCM salvo para {dataset} em {out_dir}/mcm_{dataset}.png")

    pd.DataFrame(melhor_rows).to_csv(os.path.join(out_dir, "melhor_por_dataset.csv"), index=False)
    C.merge_validation_warnings(all_warnings)


if __name__ == "__main__":
    main()
