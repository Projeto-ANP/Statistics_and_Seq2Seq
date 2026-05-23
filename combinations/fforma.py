"""
Combinação de previsões via FFORMA (Feature-based Forecast Model Averaging).

Referência: Montero-Manso et al. (2020) — https://robjhyndman.com/publications/fforma/

Segue a mesma estrutura de chamada de mean.py, dba.py e ade.py:
configure `models`, `dataset_name`, `seasonality` e `horizon`, e o script
gera resultados/<exp_name>/<DATASET>.csv com as previsões combinadas.

Fluxo implementado:
  1. Para cada série, calcula o erro (SMAPE) de cada modelo nas janelas de
     VALIDAÇÃO (todas exceto a última = teste).
  2. Extrai tsfeatures das janelas de validação como meta-features.
  3. Treina um meta-learner LightGBM que prevê qual modelo tem menor erro
     dado as features da série (objetivo multiclasse softmax).
  4. Usa os raw scores do LightGBM via softmax como pesos e aplica sobre
     as previsões de teste de cada modelo.

  O dado de TESTE nunca entra no treinamento. Os valores reais do
  horizonte de teste são usados apenas para calcular métricas finais.

  Nota: com poucas séries (<20), o meta-learner tem pouca capacidade de
  generalizar; o script avisa e oferece fallback para softmax direto dos erros.

Para rodar:
    cd Statistics_and_Seq2Seq
    conda run -n agno python -m combinations.fforma
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax
from tsfeatures import tsfeatures

from . import aux, metrics

BASE_RESULTS = "./timeseries/mestrado/resultados"

# Número mínimo de séries para treinar o meta-learner com sentido.
# Abaixo disso, usamos softmax direto dos erros de validação.
_MIN_SERIES_FOR_LGB = 10


# ---------------------------------------------------------------------------
# Helpers de I/O (mesmos de ade.py)
# ---------------------------------------------------------------------------

def _model_csv_path(model_name: str, dataset_name: str) -> str:
    return f"{BASE_RESULTS}/{model_name}/normal/{dataset_name}.csv"


def validate_models_have_dataset(models: Iterable[str], dataset_name: str) -> None:
    """Levanta FileNotFoundError se algum modelo não tiver o CSV do dataset."""
    missing = [m for m in models if not os.path.exists(_model_csv_path(m, dataset_name))]
    if missing:
        paths = "\n  - ".join(_model_csv_path(m, dataset_name) for m in missing)
        raise FileNotFoundError(
            f"Os modelos abaixo não possuem resultados para '{dataset_name}':\n"
            f"  - {paths}"
        )


def _read_model_df(model_name: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(_model_csv_path(model_name, dataset_name), sep=";")
    df["start_test"] = pd.to_datetime(df["start_test"], errors="coerce")
    df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
    return df.sort_values(["dataset_index", "start_test"]).reset_index(drop=True)


def _check_windows_alignment(model_dfs: dict[str, pd.DataFrame]) -> None:
    """Verifica que todos os modelos têm o mesmo número de janelas por série."""
    ref_name = next(iter(model_dfs))
    ref_counts = {int(k): len(v) for k, v in model_dfs[ref_name].groupby("dataset_index")}
    for name, df in model_dfs.items():
        cur_counts = {int(k): len(v) for k, v in df.groupby("dataset_index")}
        if cur_counts.keys() != ref_counts.keys():
            raise ValueError(
                f"Modelo '{name}' tem dataset_index diferentes de '{ref_name}'."
            )
        for k in ref_counts:
            if cur_counts[k] != ref_counts[k]:
                raise ValueError(
                    f"Modelo '{name}' tem {cur_counts[k]} janelas para "
                    f"dataset_index={k}, mas '{ref_name}' tem {ref_counts[k]}."
                )


# ---------------------------------------------------------------------------
# Separação validação / teste
# ---------------------------------------------------------------------------

def _split_val_test(
    model_dfs: dict[str, pd.DataFrame],
    ref_model: str,
) -> tuple[
    dict[int, dict[str, list[list[float]]]],  # val_preds[ds_idx][model] = [[step...], ...]
    dict[int, list[list[float]]],              # val_actual[ds_idx] = [[step...], ...]
    dict[int, dict[str, list[float]]],         # test_preds[ds_idx][model] = [step...]
    dict[int, list[float]],                    # test_actual[ds_idx] = [step...]
]:
    """
    Separa janelas de validação (todas menos a última) da janela de teste
    (a mais recente por dataset_index).

    A janela de teste nunca é usada no treinamento do meta-learner.
    """
    dataset_indices = sorted(model_dfs[ref_model]["dataset_index"].unique().astype(int))

    val_preds: dict[int, dict[str, list]] = {i: {m: [] for m in model_dfs} for i in dataset_indices}
    val_actual: dict[int, list] = {i: [] for i in dataset_indices}
    test_preds: dict[int, dict[str, list]] = {i: {} for i in dataset_indices}
    test_actual: dict[int, list] = {}

    for model_name, df in model_dfs.items():
        for ds_idx, group in df.groupby("dataset_index"):
            ds_idx = int(ds_idx)
            group = group.sort_values("start_test").reset_index(drop=True)
            n = len(group)
            for pos, row in group.iterrows():
                preds = aux.extract_values(row["predictions"])
                actual = aux.extract_values(row["test"])
                if pos == n - 1:                      # última janela = TESTE
                    test_preds[ds_idx][model_name] = preds
                    if model_name == ref_model:
                        test_actual[ds_idx] = actual
                else:                                 # demais janelas = VALIDAÇÃO
                    val_preds[ds_idx][model_name].append(preds)
                    if model_name == ref_model:
                        val_actual[ds_idx].append(actual)

    return val_preds, val_actual, test_preds, test_actual


# ---------------------------------------------------------------------------
# Erros de validação (n_series × n_models)
# ---------------------------------------------------------------------------

def _compute_errors(
    val_preds: dict[int, dict[str, list]],
    val_actual: dict[int, list],
    models: list[str],
) -> pd.DataFrame:
    """
    SMAPE médio de cada modelo em cada série nas janelas de validação.

    Retorna DataFrame (index=unique_id str, colunas=modelos).
    Remove modelos que nunca ganham (SMAPE mínimo) com aviso.
    """
    rows = []
    for ds_idx, actual_windows in val_actual.items():
        row: dict = {"unique_id": str(ds_idx)}
        for model_name in models:
            pred_windows = val_preds[ds_idx][model_name]
            smapes = []
            for preds, actuals in zip(pred_windows, actual_windows):
                a = np.array(actuals)
                p = np.array(preds[: len(a)])
                s = metrics.calculate_smape(p.reshape(1, -1), a.reshape(1, -1))[0]
                smapes.append(float(s))
            row[model_name] = float(np.mean(smapes)) if smapes else np.nan
        rows.append(row)

    errors_df = pd.DataFrame(rows).set_index("unique_id")

    # Remove modelos com erros sempre inválidos
    bad = errors_df.columns[errors_df.isna().all() | np.isinf(errors_df).all()].tolist()
    if bad:
        print(f"[FFORMA] Modelos com erros inválidos removidos: {bad}")
        errors_df = errors_df.drop(columns=bad)

    # Avisa modelos que nunca são os melhores (nunca venceriam na multiclasse)
    best_per_series = errors_df.idxmin(axis=1)
    never_best = [m for m in errors_df.columns if m not in best_per_series.values]
    if never_best:
        print(f"[FFORMA] Modelos que nunca vencem (mantidos nos pesos): {never_best}")

    return errors_df


# ---------------------------------------------------------------------------
# Features tsfeatures dos valores reais de validação
# ---------------------------------------------------------------------------

def _build_series_df(
    val_actual: dict[int, list],
    model_dfs: dict[str, pd.DataFrame],
    ref_model: str,
    freq: str,
) -> pd.DataFrame:
    """
    Monta DataFrame long (unique_id, ds, y) com os valores reais das janelas
    de validação, usando datas canônicas do modelo de referência.
    Usado exclusivamente para extração de tsfeatures — sem dados de teste.
    """
    ref_df = model_dfs[ref_model]
    rows = []
    for ds_idx, actual_windows in val_actual.items():
        group = (
            ref_df[ref_df["dataset_index"] == ds_idx]
            .sort_values("start_test")
            .reset_index(drop=True)
        )
        val_group = group.iloc[:-1]  # todas menos a última (teste)
        for (_, meta_row), actual_vals in zip(val_group.iterrows(), actual_windows):
            start_ds = pd.Timestamp(meta_row["start_test"])
            end_ds = pd.Timestamp(meta_row["final_test"])
            dates = pd.date_range(start=start_ds, end=end_ds, freq=freq)
            for dt, y in zip(dates, actual_vals):
                rows.append({"unique_id": str(int(ds_idx)), "ds": dt, "y": y})

    return (
        pd.DataFrame(rows)
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Meta-learner: LightGBM (FFORMA) ou fallback softmax de erros
# ---------------------------------------------------------------------------

def _fforma_objective(
    predt: np.ndarray, dtrain: xgb.DMatrix, contribution_to_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradiente e hessiana do objetivo FFORMA (Montero-Manso et al. 2020).

    XGBoost 2.1+ passa predt com shape (n_samples, n_classes).
    """
    y = dtrain.get_label().astype(int)
    preds_sm = softmax(predt, axis=1)
    weighted = (preds_sm * contribution_to_error[y]).sum(axis=1, keepdims=True)
    grad = preds_sm * (contribution_to_error[y] - weighted)
    hess = contribution_to_error[y] * preds_sm * (1 - preds_sm) - grad * preds_sm
    return grad, hess


def _train_lgb_fforma(
    feats: pd.DataFrame,
    errors_df: pd.DataFrame,
    params: dict | None,
    n_estimators: int = 100,
    seed: int = 42,
) -> xgb.Booster:
    """
    Treina o meta-learner XGBoost com o objetivo FFORMA.

    Usa XGBoost em vez de LightGBM porque o `fobj` customizado do LightGBM
    falha com `reset_parameter(objective=none)` nesta plataforma.

    feats:       (n_series, n_features)  — tsfeatures, index=unique_id
    errors_df:   (n_series, n_models)    — SMAPE validação, index=unique_id
    """
    X = feats.loc[errors_df.index].fillna(0).values.astype(float)
    contribution = errors_df.values.astype(float)

    fobj = lambda predt, dtrain: _fforma_objective(predt, dtrain, contribution)

    base_params = {
        "num_class": errors_df.shape[1],
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
    }
    if params:
        base_params.update(params)

    dtrain = xgb.DMatrix(X, label=np.arange(len(X)))

    booster = xgb.train(
        params=base_params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        obj=fobj,
        verbose_eval=False,
    )
    return booster, contribution


def _compute_weights_lgb(
    booster: xgb.Booster,
    feats: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aplica softmax nos raw scores do XGBoost para obter pesos por série."""
    X = feats.loc[errors_df.index].fillna(0).values.astype(float)
    dmat = xgb.DMatrix(X)
    raw_scores = booster.predict(dmat, output_margin=True)  # shape: (n_series, n_models)
    weights = softmax(raw_scores, axis=1)
    return pd.DataFrame(weights, index=errors_df.index, columns=errors_df.columns)


def _compute_weights_softmax(
    errors_df: pd.DataFrame,
    smape_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Fallback: pesos como softmax(−SMAPE) por série.
    Modelos com SMAPE médio de validação acima de `smape_threshold` recebem
    peso zero antes da normalização — evita que predições absurdas contaminem
    a combinação final mesmo com peso pequeno.
    """
    values = errors_df.values.astype(float).copy()

    # Zera (peso mínimo antes do softmax) modelos acima do threshold
    mask_bad = values >= smape_threshold  # shape (n_series, n_models)
    values[mask_bad] = np.inf            # softmax(−inf) → 0

    neg_errors = -values
    weights_arr = softmax(neg_errors, axis=1)
    return pd.DataFrame(weights_arr, index=errors_df.index, columns=errors_df.columns)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def fforma_combination(
    models: list[str],
    dataset_name: str,
    seasonality: int,
    horizon: int,
    freq: str = "h",
    exp_name: str = "FFORMA",
    lgb_params: dict | None = None,
    n_estimators: int = 100,
    force_softmax: bool = False,
    smape_threshold: float = 1.5,
) -> None:
    """
    Gera previsões combinadas via FFORMA para todas as séries do dataset.

    Args:
        models:       modelos base — precisam ter CSV em resultados/<M>/normal/<D>.csv
        dataset_name: ex. 'ETTH1', 'ETTH2', 'ANP_MONTHLY'
        seasonality:  período sazonal para tsfeatures (48 = half-hourly, 24 = horário, 12 = mensal)
        horizon:      tamanho do horizonte de previsão
        freq:         frequência pandas para reconstruir os date_range ('30min', 'H', 'D', 'W', etc.)
        exp_name:     nome da subpasta de saída em resultados/ (default 'FFORMA')
        lgb_params:   parâmetros extras para o meta-learner XGBoost (None = defaults)
        n_estimators: número de árvores do XGBoost (default 100)
        force_softmax: se True, pula o XGBoost e usa softmax direto dos erros
    """
    # 1) Validar disponibilidade dos CSVs
    validate_models_have_dataset(models, dataset_name)

    # 2) Carregar e verificar alinhamento
    model_dfs = {m: _read_model_df(m, dataset_name) for m in models}
    _check_windows_alignment(model_dfs)
    ref_model = models[0]

    # 3) Separar validação e teste (sem leakage)
    val_preds, val_actual, test_preds, test_actual = _split_val_test(model_dfs, ref_model)

    # 4) Calcular erros de validação por modelo por série
    errors_df = _compute_errors(val_preds, val_actual, models)
    active_models = errors_df.columns.tolist()
    n_series = len(errors_df)

    # 5) Extrair tsfeatures das janelas de validação (sem dados de teste)
    series_df = _build_series_df(val_actual, model_dfs, ref_model, freq=freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = tsfeatures(series_df, freq=seasonality)

    feats = feats.set_index("unique_id") if "unique_id" in feats.columns else feats
    feats = feats.reindex(errors_df.index)

    print(
        f"[FFORMA] Dataset={dataset_name}  séries={n_series}  "
        f"modelos={len(active_models)}  features={feats.shape[1]}"
    )

    # 6) Meta-learner: LightGBM (FFORMA) se tiver séries suficientes, softmax caso contrário
    use_lgb = (n_series >= _MIN_SERIES_FOR_LGB) and not force_softmax
    if use_lgb:
        print("[FFORMA] Treinando meta-learner LightGBM...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster, _ = _train_lgb_fforma(feats, errors_df, lgb_params, n_estimators)
        weights = _compute_weights_lgb(booster, feats, errors_df)
    else:
        reason = "force_softmax=True" if force_softmax else f"poucas séries ({n_series} < {_MIN_SERIES_FOR_LGB})"
        print(f"[FFORMA] Usando softmax direto dos erros ({reason}).")
        weights = _compute_weights_softmax(errors_df, smape_threshold=smape_threshold)

    # 7) Aplicar pesos sobre previsões de teste e salvar
    dataset_indices = sorted(val_actual.keys())
    for ds_idx in dataset_indices:
        uid = str(ds_idx)
        w = weights.loc[uid]

        horizon_len = len(test_actual[ds_idx])
        combined = np.zeros(horizon_len)
        for model_name in active_models:
            preds_arr = np.array(test_preds[ds_idx].get(model_name, [])[:horizon_len])
            if len(preds_arr) == horizon_len:
                combined += w[model_name] * preds_arr

        ref_group = (
            model_dfs[ref_model][model_dfs[ref_model]["dataset_index"] == ds_idx]
            .sort_values("start_test")
        )
        last_row = ref_group.iloc[-1]

        aux.save_to_csv(
            exp_name=exp_name,
            predictions=pd.Series(combined),
            test_values=np.array(test_actual[ds_idx]),
            dataset_name=dataset_name,
            dataset_index=ds_idx,
            horizon=horizon,
            start_test=last_row["start_test"],
            final_test=last_row["final_test"],
        )
        print(f"  [{ds_idx}] salvo em resultados/{exp_name}/{dataset_name}.csv")


# ---------------------------------------------------------------------------
# Execução padrão
# ---------------------------------------------------------------------------
#conda activate fforma
#cd Statistics_and_Seq2Seq
#python -m combinations.fforma

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

    dataset_name = "NN5_WEEKLY_DATASET"  # 'ETTH1', 'ETTH2', 'ETTM1', 'ETTM2', 'ANP_MONTHLY', 'NN5_WEEKLY_DATASET'
    map_dataset_name_to_freq = {
        "ETTH1": {"horizon": 24, "freq": "h"},
        "ETTH2": {"horizon": 24, "freq": "h"},
        "ETTM1": {"horizon": 24, "freq": "30min"},
        "ETTM2": {"horizon": 24, "freq": "30min"},
        "ANP_MONTHLY": {"horizon": 12, "freq": "ME"},
        "NN5_WEEKLY_DATASET": {"horizon": 8, "freq": "W"},
    }
    seasonality_map = {
        "ETTH1": 24,
        "ETTH2": 24,
        "ETTM1": 48,
        "ETTM2": 48,
        "ANP_MONTHLY": 12,
        "NN5_WEEKLY_DATASET": 7,
    }
    freq = map_dataset_name_to_freq[dataset_name]["freq"]
    horizon = map_dataset_name_to_freq[dataset_name]["horizon"]
    seasonality = seasonality_map[dataset_name]
    fforma_combination(
        models=models,
        dataset_name=dataset_name,
        seasonality=seasonality,
        horizon=horizon,
        freq=freq,
        exp_name="FFORMA",
    )
