"""Item 14 -- Transferencia entre conjuntos de dados.

Extrai baseline_results_json.seeded de cada serie de orchestrator_react_v5,
monta uma matriz dataset x estrategia_semeada -> score medio, e calcula a
correlacao de Spearman par a par entre todos os pares de dataset.

Saida: outputs/resultados/item_14/transferencia_datasets.csv
"""
import itertools
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd
from scipy.stats import spearmanr


def main():
    warnings = []
    matrix = {}

    for dataset in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        strategy_scores = defaultdict(list)
        for baseline in df["baseline_results_json_parsed"]:
            for strategy, info in C.dget(baseline, "seeded", default={}).items():
                if "score" in info:
                    strategy_scores[strategy].append(info["score"])
        matrix[dataset] = {strategy: sum(vals) / len(vals) for strategy, vals in strategy_scores.items()}

    all_strategies = sorted(set().union(*[set(v.keys()) for v in matrix.values()]))
    mat_df = pd.DataFrame(matrix).T[all_strategies]

    # nem toda estrategia semeada roda em todo dataset (ex.: "weighted" e pulada
    # quando ha poucas janelas de validacao) -- omit faz a correlacao usar so as
    # estrategias presentes nos dois datasets do par, em vez de propagar NaN.
    rows = []
    for dataset_a, dataset_b in itertools.combinations(C.DATASETS, 2):
        corr, _ = spearmanr(mat_df.loc[dataset_a], mat_df.loc[dataset_b], nan_policy="omit")
        rows.append({"dataset_a": dataset_a, "dataset_b": dataset_b, "spearman_corr": corr})

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_14")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transferencia_datasets.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    C.merge_validation_warnings(warnings)
    print(f"salvo em {out_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
