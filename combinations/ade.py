"""
Combinação de previsões via ADE (Arbitrated Dynamic Ensemble) — metaforecast.

Segue a mesma estrutura de chamada de `combinations/mean.py` e `combinations/dba.py`:
configura `models`, `dataset_name`, `freq` e `horizon`, e o script gera as
previsões combinadas para cada série (dataset_index) do dataset.

Como ADE precisa de histórico para aprender os pesos por especialista, todos os
rows de validação presentes no CSV de cada modelo são usados como treino
(janelas mais antigas), e o row mais recente (test) é usado para predição.

Para rodar:
    conda activate ade-combinations
cd Statistics_and_Seq2Seq
python -m combinations.ade

"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape

from . import aux, metrics

# Import metaforecast só quando rodar (é pesado e o erro fica claro se faltar)
try:
    from metaforecast.ensembles import ADE
except ImportError as e:
    raise ImportError(
        "metaforecast não está instalado no ambiente. Rode:\n"
        "    conda run -n agno pip install metaforecast"
    ) from e


BASE_RESULTS = "./timeseries/mestrado/resultados"

# Frequências nativamente suportadas por ADE (metaforecast.ensembles.ADE.WINDOW_SIZE_BY_FREQ).
# Half-hourly ('30min') não existe na lista — mapeamos para 'H' (window_size=48, ~2 dias),
# que é a aproximação mais próxima. Um aviso é emitido em _normalize_freq.
_ADE_FREQ_ALIASES = {
    "h": "H", "H": "H",
    "30min": "H", "30T": "H", "30t": "H",  # half-hourly → H (melhor aproximação disponível)
    "d": "D", "D": "D",
    "w": "W", "W": "W",
    "m": "M", "M": "M", "ME": "ME", "MS": "MS",
    "q": "Q", "Q": "Q", "QS": "QS",
    "y": "Y", "Y": "Y",
}

# Mapeamentos que precisam de aviso por serem aproximações
_ADE_FREQ_WARNINGS = {
    "30min": "H", "30T": "H", "30t": "H",
}


def _normalize_freq(freq: str) -> str:
    if freq not in _ADE_FREQ_ALIASES:
        raise ValueError(
            f"freq='{freq}' não é suportado pelo ADE. "
            f"Use uma de: {sorted(set(_ADE_FREQ_ALIASES.values()))}"
        )
    if freq in _ADE_FREQ_WARNINGS:
        mapped = _ADE_FREQ_ALIASES[freq]
        print(
            f"[ADE] Aviso: freq='{freq}' não é suportado nativamente pelo ADE. "
            f"Usando '{mapped}' como aproximação (window_size=48)."
        )
    return _ADE_FREQ_ALIASES[freq]


# ---------------------------------------------------------------------------
# Validação e leitura
# ---------------------------------------------------------------------------

def _model_csv_path(model_name: str, dataset_name: str) -> str:
    return f"{BASE_RESULTS}/{model_name}/normal/{dataset_name}.csv"


def validate_models_have_dataset(models: Iterable[str], dataset_name: str) -> None:
    """Garante que cada modelo possui o CSV do dataset. Senão, levanta FileNotFoundError."""
    missing = [m for m in models if not os.path.exists(_model_csv_path(m, dataset_name))]
    if missing:
        paths = "\n  - ".join(_model_csv_path(m, dataset_name) for m in missing)
        raise FileNotFoundError(
            f"Os modelos abaixo não possuem resultados para o dataset '{dataset_name}':\n"
            f"  - {paths}"
        )


def _read_model_df(model_name: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(_model_csv_path(model_name, dataset_name), sep=";")
    df["start_test"] = pd.to_datetime(df["start_test"], errors="coerce")
    df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
    df = df.sort_values(by=["dataset_index", "start_test"]).reset_index(drop=True)
    return df


def _windows_count_per_series(df: pd.DataFrame) -> dict[int, int]:
    """{dataset_index: número de janelas}"""
    out: dict[int, int] = {}
    for ds_idx, group in df.groupby("dataset_index"):
        out[int(ds_idx)] = len(group)
    return out


def _check_windows_alignment(model_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Confere que todos os modelos têm:
      - o mesmo conjunto de dataset_index
      - o mesmo número de janelas por dataset_index
    As datas reais (start_test/final_test) podem divergir entre modelos
    (cada modelo pode ter sido treinado/testado num split temporal distinto),
    mas a *posição relativa* (val mais antiga → val mais recente → test) tem
    que ser consistente. As datas canônicas do primeiro modelo são usadas
    como eixo temporal comum.
    """
    ref_name = next(iter(model_dfs))
    ref_counts = _windows_count_per_series(model_dfs[ref_name])
    for name, df in model_dfs.items():
        cur_counts = _windows_count_per_series(df)
        if cur_counts.keys() != ref_counts.keys():
            raise ValueError(
                f"Modelo '{name}' tem dataset_index diferente de '{ref_name}': "
                f"{sorted(cur_counts.keys())} vs {sorted(ref_counts.keys())}"
            )
        for k in ref_counts:
            if cur_counts[k] != ref_counts[k]:
                raise ValueError(
                    f"Modelo '{name}' tem {cur_counts[k]} janelas para "
                    f"dataset_index={k}, mas '{ref_name}' tem {ref_counts[k]}. "
                    f"Os modelos precisam ter o mesmo número de janelas por série."
                )


# ---------------------------------------------------------------------------
# Construção dos DataFrames no formato ADE
# ---------------------------------------------------------------------------

def _build_long_dataframes(
    models: list[str],
    model_dfs: dict[str, pd.DataFrame],
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.Series]]:
    """
    Constrói dois DataFrames longos no formato esperado pelo metaforecast:

    - combined_df  → linhas de treino/validação: unique_id, ds, y, MODEL1, MODEL2, …
    - df_predictions → linhas do horizonte de teste: unique_id, ds, MODEL1, MODEL2, …

    Também devolve test_per_series: {dataset_index: pd.Series dos valores reais do teste}.
    """
    # 1) Pegar as datas canônicas do primeiro modelo (referência).
    #    Para cada dataset_index → lista de (start_test, final_test) ordenadas crescentes.
    ref_name = models[0]
    ref_df = model_dfs[ref_name]
    canonical_windows: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ds_idx, group in ref_df.groupby("dataset_index"):
        sorted_group = group.sort_values("start_test")
        canonical_windows[int(ds_idx)] = list(
            zip(sorted_group["start_test"], sorted_group["final_test"])
        )

    # 2) Coletar predições e y em estrutura long por modelo, usando as datas canônicas
    combined_df = pd.DataFrame()
    df_predictions = pd.DataFrame()
    test_per_series: dict[int, pd.Series] = {}

    for model_name in models:
        model_data_train: list[dict] = []
        model_data_test: list[dict] = []

        df = model_dfs[model_name]

        for ds_idx, group in df.groupby("dataset_index"):
            unique_id = str(int(ds_idx))
            group = group.sort_values("start_test").reset_index(drop=True)
            windows = canonical_windows[int(ds_idx)]  # mesma ordem cronológica

            for pos, row in group.iterrows():
                start_ds, end_ds = windows[pos]
                date_range = pd.date_range(start=start_ds, end=end_ds, freq=freq)
                preds = aux.extract_values(row["predictions"])
                tests = aux.extract_values(row["test"])

                is_test_window = pos == len(windows) - 1

                if is_test_window:
                    if model_name == ref_name:
                        test_per_series[int(unique_id)] = pd.Series(tests)
                    for i in range(len(date_range)):
                        entry = {"unique_id": unique_id, "ds": date_range[i]}
                        if i < len(preds):
                            entry[model_name] = preds[i]
                        model_data_test.append(entry)
                else:
                    for i in range(len(date_range)):
                        entry = {"unique_id": unique_id, "ds": date_range[i]}
                        if i < len(tests):
                            entry["y"] = tests[i]
                        if i < len(preds):
                            entry[model_name] = preds[i]
                        model_data_train.append(entry)

        model_df_train = pd.DataFrame(model_data_train)
        model_df_test = pd.DataFrame(model_data_test)

        # merge wide
        if df_predictions.empty:
            df_predictions = model_df_test
        else:
            df_predictions = pd.merge(
                df_predictions,
                model_df_test[["unique_id", "ds", model_name]],
                on=["unique_id", "ds"],
                how="outer",
            )

        if combined_df.empty:
            combined_df = model_df_train
        else:
            combined_df = pd.merge(
                combined_df,
                model_df_train[["unique_id", "ds", model_name]],
                on=["unique_id", "ds"],
                how="outer",
            )

    combined_df = combined_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df_predictions = df_predictions.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return combined_df, df_predictions, test_per_series


# ---------------------------------------------------------------------------
# ADE por série
# ---------------------------------------------------------------------------

def _ade_predict_one_series(
    combined_df: pd.DataFrame,
    df_predictions: pd.DataFrame,
    unique_id: str,
    horizon: int,
    freq: str,
    trim_ratio: float = 1.0,
) -> pd.Series:
    df_train = combined_df[combined_df["unique_id"] == unique_id].copy()
    df_test = df_predictions[df_predictions["unique_id"] == unique_id].copy()
    main_df = df_train[["unique_id", "ds", "y"]].dropna(subset=["y"])

    # meta_lags não pode passar do tamanho da série de treino disponível
    n_train = len(main_df)
    max_lag = max(1, min(horizon, max(1, n_train - 1)))
    meta_lags = list(range(1, max_lag + 1))

    ensemble = ADE(freq=freq, meta_lags=meta_lags, trim_ratio=trim_ratio)
    ensemble.fit(df_train)
    ade_fcst = ensemble.predict(df_test, train=main_df, h=horizon)
    return pd.Series(ade_fcst.tolist())


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def ade_combination(
    models: list[str],
    dataset_name: str,
    freq: str,
    horizon: int,
    exp_name: str = "ADE",
    trim_ratio: float = 1.0,
) -> None:
    """
    Gera previsões combinadas via ADE para todas as séries do dataset.

    Args:
        models: lista de modelos base (precisam ter CSV em resultados/<MODEL>/normal/<DATASET>.csv)
        dataset_name: ex: 'ETTH1', 'ETTH2', 'ANP_MONTHLY'
        freq: frequência pandas. 'h' para horário, 'ME' para mês final, 'D' para diário, etc.
        horizon: tamanho do horizonte de previsão (ex: 12, 24)
        exp_name: nome da pasta de saída em resultados/ (default 'ADE')
        trim_ratio: fração de modelos top a manter no ensemble (default 1.0 = todos)
    """
    # 0) freq_data = frequência real do dataset (para date_range)
    #    freq_ade  = frequência mapeada para o ADE (que não suporta '30min')
    freq_data = freq
    freq_ade = _normalize_freq(freq)

    # 1) validar disponibilidade
    validate_models_have_dataset(models, dataset_name)

    # 2) carregar e checar alinhamento das janelas
    model_dfs = {m: _read_model_df(m, dataset_name) for m in models}
    _check_windows_alignment(model_dfs)

    # 3) construir DataFrames longos usando a freq real do dataset
    combined_df, df_predictions, test_per_series = _build_long_dataframes(
        models, model_dfs, freq=freq_data
    )

    # 4) rodar ADE por série e salvar
    unique_ids = sorted(df_predictions["unique_id"].unique(), key=lambda x: int(x))
    print(f"[ADE] Dataset={dataset_name}  séries={len(unique_ids)}  modelos={len(models)}")

    for uid in unique_ids:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = _ade_predict_one_series(
                combined_df, df_predictions, uid,
                horizon=horizon, freq=freq_ade, trim_ratio=trim_ratio,
            )

        # localizar metadados da janela de teste (start/final) usando o primeiro modelo
        first_model_df = model_dfs[models[0]]
        latest_row = (
            first_model_df[first_model_df["dataset_index"] == int(uid)]
            .sort_values("start_test")
            .iloc[-1]
        )
        start_test = latest_row["start_test"]
        final_test = latest_row["final_test"]
        test_values = test_per_series[int(uid)].values

        aux.save_to_csv(
            exp_name=exp_name,
            predictions=preds,
            test_values=test_values,
            dataset_name=dataset_name,
            dataset_index=int(uid),
            horizon=horizon,
            start_test=start_test,
            final_test=final_test,
        )
        print(f"  [{uid}] salvo em resultados/{exp_name}/{dataset_name}.csv")


# ---------------------------------------------------------------------------
# Execução padrão
# ---------------------------------------------------------------------------
"""
conda activate ade-combinations
cd Statistics_and_Seq2Seq
python -m combinations.ade
"""
if __name__ == "__main__":
    models = [
        "ARIMA",
        "ETS",
        "THETA",
        "rf",
        "catboost",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
        "ONLY_CWT_catboost",
        "ONLY_CWT_rf",
        "ONLY_DWT_catboost",
        "ONLY_DWT_rf",
        "ONLY_FT_catboost",
        "ONLY_FT_rf",
        "NaiveSeasonal",
        "NaiveMovingAverage",
    ]

    dataset_name = "ANP_MONTHLY"  # 'ETTH1', 'ETTH2', 'ETTM1', 'ETTM2', 'ANP_MONTHLY', 'NN5_WEEKLY_DATASET'
    map_dataset_name_to_freq = {
        "ETTH1": {"horizon": 24, "freq": "H"},
        "ETTH2": {"horizon": 24, "freq": "H"},
        "ETTM1": {"horizon": 24, "freq": "30min"},
        "ETTM2": {"horizon": 24, "freq": "30min"},
        "ANP_MONTHLY": {"horizon": 12, "freq": "M"},
        "NN5_WEEKLY_DATASET": {"horizon": 8, "freq": "W"},
    }
    # Use as chaves aceitas pelo ADE: H, D, W, M, ME, MS, Q, QS, Y
    freq = map_dataset_name_to_freq[dataset_name]["freq"]
    horizon = map_dataset_name_to_freq[dataset_name]["horizon"]
    ade_combination(
        models=models,
        dataset_name=dataset_name,
        freq=freq,       # ETT é horário
        horizon=horizon,     # ETT usa horizonte 24
        exp_name="ADE",
    )
