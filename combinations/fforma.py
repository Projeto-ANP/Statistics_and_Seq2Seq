"""
Combinação de previsões via FFORMA (Feature-based FORecast Model Averaging).

Referência: Montero-Manso, Athanasopoulos, Hyndman & Talagala (2020),
"FFORMA: Feature-based forecast model averaging", IJF 36(1):86-92.

Primeira vez na máquina (o env `fforma-combinations` ainda não existe):
    conda env create -f fforma_environment.yml

Toda execução:
    conda activate fforma-combinations
    cd Statistics_and_Seq2Seq
    python -m combinations.fforma --dataset ANP_MONTHLY

Datasets, frequências, sazonalidades e horizontes vêm de
`combinations/dataset_specs.py` (o horizonte é lido do próprio CSV).
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax
from tsfeatures import tsfeatures

from . import aux, metrics
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
    window_dates,
)

# Abaixo disso o meta-learner não tem amostra para generalizar entre séries e
# caímos no softmax direto dos erros de validação.
_MIN_SERIES_FOR_META = 10


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
    Separa as janelas de validação (todas menos a última) da janela de teste (a
    mais recente por dataset_index). A janela de teste nunca entra no treino do
    meta-learner — só no cálculo das métricas finais.
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

    SMAPE (e não RMSE) porque o gradiente do objetivo FFORMA **soma as
    contribuições entre séries**: com erro em escala bruta, as séries de maior
    magnitude dominam o gradiente e o meta-modelo aprende escala em vez de
    competência. O FFORMA original usa OWA pelo mesmo motivo.
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
                if len(a) == 0 or len(p) != len(a):
                    continue
                s = metrics.calculate_smape(p.reshape(1, -1), a.reshape(1, -1))[0]
                smapes.append(float(s))
            row[model_name] = float(np.mean(smapes)) if smapes else np.nan
        rows.append(row)

    errors_df = pd.DataFrame(rows).set_index("unique_id")

    bad = errors_df.columns[errors_df.isna().all() | np.isinf(errors_df).all()].tolist()
    if bad:
        print(f"[FFORMA] Modelos com erros inválidos removidos: {bad}")
        errors_df = errors_df.drop(columns=bad)

    # NaN residual (série sem janela válida para um modelo) vira o pior erro
    # daquela série; sem isso o softmax propaga NaN e a combinação sai vazia.
    if errors_df.isna().any().any():
        n_nan = int(errors_df.isna().sum().sum())
        print(f"[FFORMA] {n_nan} par(es) (série, modelo) sem erro válido — preenchidos com o pior da série.")
        errors_df = errors_df.apply(lambda r: r.fillna(r.max()), axis=1)

    best_per_series = errors_df.idxmin(axis=1)
    never_best = [m for m in errors_df.columns if m not in best_per_series.values]
    if never_best:
        print(f"[FFORMA] Modelos que nunca vencem (mantidos nos pesos): {never_best}")

    return errors_df


# ---------------------------------------------------------------------------
# Features (tsfeatures) sobre os valores reais de validação
# ---------------------------------------------------------------------------

def _build_series_df(
    val_actual: dict[int, list],
    model_dfs: dict[str, pd.DataFrame],
    ref_model: str,
    freq: str,
) -> pd.DataFrame:
    """
    DataFrame long (unique_id, ds, y) com os valores reais das janelas de
    validação, no eixo canônico do modelo de referência. Usado só para extrair
    tsfeatures — sem nenhum dado de teste.
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
            if not len(actual_vals):
                continue
            dates = window_dates(pd.Timestamp(meta_row["start_test"]), len(actual_vals), freq)
            for dt, y in zip(dates, actual_vals):
                rows.append({"unique_id": str(int(ds_idx)), "ds": dt, "y": y})

    return pd.DataFrame(rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Meta-learner
# ---------------------------------------------------------------------------

def _fforma_objective(
    predt: np.ndarray, dtrain: xgb.DMatrix, contribution_to_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradiente e hessiana do objetivo FFORMA (Montero-Manso et al. 2020).

    Não otimiza "acertar o modelo vencedor" (cross-entropy); minimiza o erro
    combinado esperado  L = softmax(score) · erro_por_modelo. A saída softmax do
    booster **é** o vetor de pesos.

    XGBoost 2.1+ passa predt com shape (n_samples, n_classes).
    """
    y = dtrain.get_label().astype(int)
    preds_sm = softmax(predt, axis=1)
    weighted = (preds_sm * contribution_to_error[y]).sum(axis=1, keepdims=True)
    grad = preds_sm * (contribution_to_error[y] - weighted)
    hess = contribution_to_error[y] * preds_sm * (1 - preds_sm) - grad * preds_sm
    return grad, hess


def _train_fforma_booster(
    feats: pd.DataFrame,
    errors_df: pd.DataFrame,
    params: dict | None,
    n_estimators: int = 100,
    seed: int = 42,
) -> tuple[xgb.Booster, np.ndarray]:
    """
    Treina o meta-learner XGBoost com o objetivo FFORMA.

    XGBoost em vez de LightGBM porque o `fobj` customizado do LightGBM falha com
    `reset_parameter(objective=none)` nesta plataforma.

    feats:     (n_series, n_features) — tsfeatures, index=unique_id
    errors_df: (n_series, n_models)   — SMAPE de validação, index=unique_id
    """
    X = feats.loc[errors_df.index].fillna(0).values.astype(float)
    contribution = errors_df.values.astype(float)

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
        obj=lambda predt, dm: _fforma_objective(predt, dm, contribution),
        verbose_eval=False,
    )
    return booster, contribution


def _compute_weights_booster(
    booster: xgb.Booster,
    feats: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Softmax dos raw scores do XGBoost -> pesos por série."""
    X = feats.loc[errors_df.index].fillna(0).values.astype(float)
    raw_scores = booster.predict(xgb.DMatrix(X), output_margin=True)
    return pd.DataFrame(
        softmax(raw_scores, axis=1), index=errors_df.index, columns=errors_df.columns
    )


def _compute_weights_softmax(
    errors_df: pd.DataFrame,
    smape_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Fallback: pesos = softmax(−SMAPE) por série. Modelos com SMAPE de validação
    acima de `smape_threshold` recebem peso zero antes da normalização — evita
    que previsões numericamente absurdas contaminem a combinação.
    """
    values = errors_df.values.astype(float).copy()
    values[values >= smape_threshold] = np.inf  # softmax(−inf) -> 0
    return pd.DataFrame(
        softmax(-values, axis=1), index=errors_df.index, columns=errors_df.columns
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def fforma_combination(
    dataset_name: str,
    models: list[str] | None = None,
    exp_name: str = "FFORMA",
    seasonality: int | None = None,
    horizon: int | None = None,
    lgb_params: dict | None = None,
    n_estimators: int = 100,
    force_softmax: bool = False,
    smape_threshold: float = 1.5,
    resume: bool = False,
    drop_misaligned: bool = True,
) -> None:
    """
    Gera previsões combinadas via FFORMA para todas as séries do dataset.

    Args:
        dataset_name:  'ANP_MONTHLY', 'NN5_WEEKLY_DATASET', 'M4_WEEKLY_DATASET', …
        models:        modelos base; default = os 19 de `DEFAULT_MODELS`
        exp_name:      subpasta de saída em resultados/ (default 'FFORMA')
        seasonality:   sobrescreve a sazonalidade do registro (tsfeatures)
        horizon:       sobrescreve o horizonte lido do CSV
        lgb_params:    parâmetros extras do XGBoost
        n_estimators:  número de rodadas de boosting (default 100)
        force_softmax: pula o meta-learner e usa softmax direto dos erros
        smape_threshold: no fallback, modelos com SMAPE ≥ isso recebem peso zero
        resume:        continua de onde parou em vez de apagar a saída
        drop_misaligned: remove modelos cujo alvo de teste diverge do modelo
                      de referência (caso real: ETTM1/ETTM2). Ver
                      `dataset_specs.detect_misaligned_models`.
    """
    models = list(models or DEFAULT_MODELS)
    validate_models_have_dataset(models, dataset_name)
    model_dfs = read_all_model_dfs(models, dataset_name)
    check_windows_alignment(model_dfs)
    ref_model = models[0]
    models, model_dfs = resolve_active_models(
        models, model_dfs, ref_model, drop_misaligned=drop_misaligned, label="FFORMA"
    )

    spec = resolve_spec(dataset_name, models, seasonality=seasonality, horizon=horizon)
    print(f"[FFORMA] {describe(spec)} modelos={len(models)}")

    if spec.n_windows < 2:
        raise ValueError(
            f"'{dataset_name}' tem {spec.n_windows} janela por série; o FFORMA precisa "
            f"de pelo menos 2 (uma de validação + uma de teste)."
        )

    val_preds, val_actual, test_preds, test_actual = _split_val_test(model_dfs, ref_model)
    errors_df = _compute_errors(val_preds, val_actual, models)
    active_models = errors_df.columns.tolist()
    n_series = len(errors_df)

    series_df = _build_series_df(val_actual, model_dfs, ref_model, freq=spec.freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = tsfeatures(series_df, freq=spec.seasonality)

    feats = feats.set_index("unique_id") if "unique_id" in feats.columns else feats
    feats.index = feats.index.astype(str)
    feats = feats.reindex(errors_df.index)
    print(f"[FFORMA] modelos ativos={len(active_models)} features={feats.shape[1]}")

    use_meta = (n_series >= _MIN_SERIES_FOR_META) and not force_softmax
    if use_meta:
        print(f"[FFORMA] Treinando meta-learner XGBoost (objetivo FFORMA, {n_estimators} rodadas)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster, _ = _train_fforma_booster(feats, errors_df, lgb_params, n_estimators)
        weights = _compute_weights_booster(booster, feats, errors_df)
    else:
        reason = (
            "force_softmax=True" if force_softmax
            else f"poucas séries ({n_series} < {_MIN_SERIES_FOR_META})"
        )
        print(f"[FFORMA] Usando softmax direto dos erros ({reason}).")
        weights = _compute_weights_softmax(errors_df, smape_threshold=smape_threshold)

    done = prepare_output(exp_name, dataset_name, resume)
    dataset_indices = [i for i in sorted(val_actual.keys()) if i not in done]
    print(f"[FFORMA] {len(dataset_indices)} série(s) a gravar de {len(val_actual)}")

    for ds_idx in dataset_indices:
        w = weights.loc[str(ds_idx)]
        horizon_len = len(test_actual[ds_idx])
        combined = np.zeros(horizon_len)
        used = 0.0
        for model_name in active_models:
            preds_arr = np.array(test_preds[ds_idx].get(model_name, [])[:horizon_len])
            if len(preds_arr) == horizon_len:
                combined += w[model_name] * preds_arr
                used += w[model_name]
        if used <= 0:
            raise RuntimeError(
                f"Série {ds_idx}: nenhum modelo tem previsão com {horizon_len} passos. "
                f"CSVs inconsistentes para '{dataset_name}'."
            )
        # renormaliza se algum modelo ficou de fora por comprimento incompatível
        combined /= used

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
            horizon=spec.horizon,
            start_test=last_row["start_test"],
            final_test=last_row["final_test"],
        )

    print(f"\n[FFORMA] Concluído: {output_csv_path(exp_name, dataset_name)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m combinations.fforma",
        description="Combinação de previsões via FFORMA (Montero-Manso et al. 2020).",
    )
    p.add_argument("--dataset", required=True, help="ex: ANP_MONTHLY, M4_WEEKLY_DATASET")
    p.add_argument("--models", nargs="+", default=None, help="default: os 19 modelos base")
    p.add_argument("--exp-name", default="FFORMA", help="subpasta de saída em resultados/")
    p.add_argument("--seasonality", type=int, default=None, help="default: dataset_specs.py")
    p.add_argument("--horizon", type=int, default=None, help="default: lido do CSV")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--force-softmax", action="store_true", help="pula o meta-learner")
    p.add_argument("--smape-threshold", type=float, default=1.5)
    p.add_argument("--resume", action="store_true", help="continua em vez de apagar a saída")
    p.add_argument(
        "--keep-misaligned", action="store_true",
        help="não dropar modelos cujo alvo de teste diverge do de referência (default: dropa)",
    )
    args = p.parse_args(argv)

    fforma_combination(
        dataset_name=args.dataset,
        models=args.models,
        exp_name=args.exp_name,
        seasonality=args.seasonality,
        horizon=args.horizon,
        n_estimators=args.n_estimators,
        force_softmax=args.force_softmax,
        smape_threshold=args.smape_threshold,
        resume=args.resume,
        drop_misaligned=not args.keep_misaligned,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
