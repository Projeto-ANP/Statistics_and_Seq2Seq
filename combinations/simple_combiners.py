"""
Loop compartilhado por `mean.py`, `median.py` e `dba.py`.

Os três scripts são idênticos exceto pela função de agregação (média,
mediana, centroide DTW): mesma leitura de modelos, mesma checagem de
alinhamento, mesmo jeito de achar a janela de teste de cada modelo, mesma
gravação de CSV. Isso vivia triplicado — inclusive o bug de horizonte
hardcoded (`dba.py` gravava `horizon=12` rodando ETTM1, que tem horizonte 24).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import aux
from .dataset_specs import (
    DEFAULT_MODELS,
    check_windows_alignment,
    describe,
    output_csv_path,
    prepare_output,
    read_all_model_dfs,
    resolve_active_models,
    resolve_spec,
    validate_models_have_dataset,
)

# (n_models, horizon) -> (horizon,)
AggregateFn = Callable[[np.ndarray], np.ndarray]


def _final_windows(model_dfs: dict[str, pd.DataFrame], models: list[str]) -> dict[str, dict[int, pd.Series]]:
    """{modelo: {dataset_index: linha da janela mais recente}}."""
    out: dict[str, dict[int, pd.Series]] = {}
    for m in models:
        tail = (
            model_dfs[m]
            .sort_values(["dataset_index", "start_test"])
            .groupby("dataset_index")
            .tail(1)
        )
        out[m] = {int(r["dataset_index"]): r for _, r in tail.iterrows()}
    return out


def run_simple_combination(
    dataset_name: str,
    aggregate: AggregateFn,
    exp_name: str,
    models: list[str] | None = None,
    horizon: int | None = None,
    resume: bool = False,
    drop_misaligned: bool = True,
) -> None:
    """
    Args:
        dataset_name:    'ANP_MONTHLY', 'ETTM1', 'M4_WEEKLY_DATASET', …
        aggregate:       função (n_modelos, horizonte) -> (horizonte,)
        exp_name:        rótulo usado nos logs e subpasta de saída em resultados/
        models:          modelos base; default = os 19 de `DEFAULT_MODELS`
        horizon:         sobrescreve o horizonte lido do CSV
        resume:          continua de onde parou em vez de apagar a saída
        drop_misaligned: remove modelos cujo alvo de teste diverge do modelo
                         de referência (caso real: ETTM1/ETTM2)
    """
    models = list(models or DEFAULT_MODELS)
    validate_models_have_dataset(models, dataset_name)
    model_dfs = read_all_model_dfs(models, dataset_name)
    check_windows_alignment(model_dfs)
    ref_model = models[0]
    models, model_dfs = resolve_active_models(
        models, model_dfs, ref_model, drop_misaligned=drop_misaligned, label=exp_name
    )

    spec = resolve_spec(dataset_name, models, horizon=horizon)
    print(f"[{exp_name}] {describe(spec)} modelos={len(models)}")

    finals = _final_windows(model_dfs, models)
    done = prepare_output(exp_name, dataset_name, resume)
    dataset_indices = [i for i in sorted(finals[ref_model].keys()) if i not in done]
    print(f"[{exp_name}] {len(dataset_indices)} série(s) a gravar de {len(finals[ref_model])}")

    for ds_idx in dataset_indices:
        ref_row = finals[ref_model][ds_idx]
        test_values = np.array(aux.extract_values(ref_row["test"]))
        h = len(test_values)

        rows = []
        for m in models:
            row = finals[m].get(ds_idx)
            if row is None:
                continue
            p = aux.extract_values(row["predictions"])[:h]
            if len(p) == h:
                rows.append(p)

        if not rows:
            raise RuntimeError(
                f"Série {ds_idx} de '{dataset_name}': nenhum modelo tem {h} previsões "
                f"alinhadas ao horizonte de teste. CSVs inconsistentes."
            )

        combined = aggregate(np.vstack(rows))

        aux.save_to_csv(
            exp_name=exp_name,
            predictions=pd.Series(combined),
            test_values=test_values,
            dataset_name=dataset_name,
            dataset_index=ds_idx,
            horizon=spec.horizon,
            start_test=ref_row["start_test"],
            final_test=ref_row["final_test"],
        )

    print(f"\n[{exp_name}] Concluído: {output_csv_path(exp_name, dataset_name)}")
