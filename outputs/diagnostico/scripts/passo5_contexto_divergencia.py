"""Passo 5 -- Relacao com comprimento da serie ou valores proximos de zero.

Para cada (dataset, regressor, dataset_index) marcado como divergente no
Passo 2 (mesmo criterio, recalculado aqui em granularidade por serie em vez
de agregado): comprimento_serie (comprimento total da serie no .tsf original
-- ver diag_lib.tsf_series_lengths, os CSVs de origem nao tem uma coluna de
treino explicita) e tem_valor_pequeno (True se o `test` daquela serie contem
algum valor com modulo menor que 1).

Nota de caminho: o prompt pede esse arquivo especificamente em
outputs/resultados/diagnostico/ (nao outputs/diagnostico/, onde vivem os
passos 1-4 e o RESUMO) -- caminho mantido exatamente como pedido.

Saida: outputs/resultados/diagnostico/passo5_contexto_divergencia.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_lib as D

import pandas as pd


def main():
    rows = []
    lengths_cache = {}

    for folder in D.ALL_REGRESSORS:
        for dataset in D.C.DATASETS:
            if dataset not in lengths_cache:
                lengths_cache[dataset] = D.tsf_series_lengths(dataset)
            lengths = lengths_cache[dataset]

            df = D.C.load_filtered(folder, dataset)
            for row in df.itertuples(index=False):
                minimal = pd.Series({"predictions": row.predictions, "test": row.test})
                if not D.is_divergent(minimal):
                    continue
                test_vals = D.C.parse_num_list(row.test)
                tem_valor_pequeno = any(abs(v) < 1 for v in test_vals)
                idx = int(row.dataset_index)
                comprimento = lengths[idx] if 0 <= idx < len(lengths) else None
                rows.append({
                    "dataset": dataset,
                    "regressor": folder,
                    "dataset_index": idx,
                    "comprimento_serie": comprimento,
                    "tem_valor_pequeno": tem_valor_pequeno,
                })

    out_dir = os.path.join(D.REPO_ROOT, "outputs", "resultados", "diagnostico")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "passo5_contexto_divergencia.csv")
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nsalvo em {out_path} ({len(result)} linhas)")


if __name__ == "__main__":
    main()
