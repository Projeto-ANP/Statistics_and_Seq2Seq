"""
Combinação de previsões via ADE (Arbitrated Dynamic Ensemble) — metaforecast.

Todas as janelas de validação do CSV de cada modelo (as mais antigas) viram
treino do meta-modelo; a janela mais recente (teste) é a que se prevê.

Primeira vez na máquina (o env `ade-combinations` ainda não existe):
    conda env create -f ade_environment.yml

Toda execução:
    conda activate ade-combinations
    cd Statistics_and_Seq2Seq
    python -m combinations.ade --dataset ANP_MONTHLY

Datasets, frequências e horizontes vêm de `combinations/dataset_specs.py`
(o horizonte é lido do próprio CSV). Rodar um dataset novo é só passar --dataset.
"""

from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd

from . import aux
from .dataset_specs import (
    DEFAULT_MODELS,
    DatasetSpec,
    check_windows_alignment,
    describe,
    output_csv_path,
    prepare_output,
    read_all_model_dfs,
    resolve_spec,
    validate_models_have_dataset,
    window_dates,
)

try:
    from metaforecast.ensembles import ADE
except ImportError as e:  # pragma: no cover - depende do ambiente
    raise ImportError(
        "metaforecast não está instalado neste ambiente.\n"
        "    conda env create -f ade_environment.yml\n"
        "    conda activate ade-combinations"
    ) from e


# Frequências que o ADE aceita (metaforecast.ensembles.ADE.WINDOW_SIZE_BY_FREQ).
_ADE_WINDOW_BY_FREQ = {
    "H": 48, "D": 14, "W": 16, "M": 12, "ME": 12, "MS": 12, "Q": 4, "QS": 4, "Y": 6,
}


def normalize_freq_for_ade(spec: DatasetSpec) -> str:
    """
    `spec.freq_ade` já vem no vocabulário do ADE; aqui só validamos e avisamos
    quando o mapeamento é aproximação (meia-hora não existe na tabela do ADE).
    """
    freq_ade = spec.freq_ade
    if freq_ade not in _ADE_WINDOW_BY_FREQ:
        raise ValueError(
            f"freq_ade='{freq_ade}' não é suportado pelo ADE. "
            f"Use uma de: {sorted(_ADE_WINDOW_BY_FREQ)}"
        )
    if spec.freq in ("30min", "15min"):
        print(
            f"[ADE] Aviso: dados em '{spec.freq}' não têm janela nativa no ADE. "
            f"Usando '{freq_ade}' (window_size={_ADE_WINDOW_BY_FREQ[freq_ade]}) como aproximação."
        )
    return freq_ade


# ---------------------------------------------------------------------------
# Construção dos DataFrames no formato metaforecast
# ---------------------------------------------------------------------------

def _build_long_dataframes(
    models: list[str],
    model_dfs: dict[str, pd.DataFrame],
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.Series]]:
    """
    Dois DataFrames longos no formato esperado pelo metaforecast:

    - combined_df    -> janelas de validação: unique_id, ds, y, MODEL1, MODEL2, …
    - df_predictions -> janela de teste:      unique_id, ds,    MODEL1, MODEL2, …

    Também devolve {dataset_index: Series com os valores reais do teste}.

    As datas do primeiro modelo são o eixo canônico: modelos podem ter splits
    temporais distintos, mas a posição relativa das janelas é a mesma (garantido
    por `check_windows_alignment`).
    """
    ref_name = models[0]
    ref_df = model_dfs[ref_name]

    canonical_starts: dict[int, list[pd.Timestamp]] = {}
    for ds_idx, group in ref_df.groupby("dataset_index"):
        canonical_starts[int(ds_idx)] = list(group.sort_values("start_test")["start_test"])

    combined_df = pd.DataFrame()
    df_predictions = pd.DataFrame()
    test_per_series: dict[int, pd.Series] = {}

    for model_name in models:
        rows_train: list[dict] = []
        rows_test: list[dict] = []
        df = model_dfs[model_name]

        for ds_idx, group in df.groupby("dataset_index"):
            unique_id = str(int(ds_idx))
            group = group.sort_values("start_test").reset_index(drop=True)
            starts = canonical_starts[int(ds_idx)]
            n_windows = len(starts)

            for pos, row in group.iterrows():
                preds = aux.extract_values(row["predictions"])
                tests = aux.extract_values(row["test"])
                # o eixo cobre o maior dos dois; window_dates garante o
                # comprimento exato, então nada é descartado em silêncio
                n_points = max(len(preds), len(tests))
                if n_points == 0:
                    continue
                dates = window_dates(starts[pos], n_points, freq)

                if pos == n_windows - 1:  # janela de teste
                    if model_name == ref_name:
                        test_per_series[int(unique_id)] = pd.Series(tests)
                    for i in range(n_points):
                        entry = {"unique_id": unique_id, "ds": dates[i]}
                        if i < len(preds):
                            entry[model_name] = preds[i]
                        rows_test.append(entry)
                else:                      # janelas de validação
                    for i in range(n_points):
                        entry = {"unique_id": unique_id, "ds": dates[i]}
                        if i < len(tests):
                            entry["y"] = tests[i]
                        if i < len(preds):
                            entry[model_name] = preds[i]
                        rows_train.append(entry)

        model_df_train = pd.DataFrame(rows_train)
        model_df_test = pd.DataFrame(rows_test)

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
    freq_ade: str,
    trim_ratio: float = 1.0,
) -> pd.Series:
    df_train = combined_df[combined_df["unique_id"] == unique_id].copy()
    df_test = df_predictions[df_predictions["unique_id"] == unique_id].copy()
    main_df = df_train[["unique_id", "ds", "y"]].dropna(subset=["y"])

    # meta_lags não pode passar do tamanho do histórico disponível
    n_train = len(main_df)
    max_lag = max(1, min(horizon, max(1, n_train - 1)))
    meta_lags = list(range(1, max_lag + 1))

    ensemble = ADE(freq=freq_ade, meta_lags=meta_lags, trim_ratio=trim_ratio)
    ensemble.fit(df_train)
    ade_fcst = ensemble.predict(df_test, train=main_df, h=horizon)
    return pd.Series(ade_fcst.tolist())


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def ade_combination(
    dataset_name: str,
    models: list[str] | None = None,
    exp_name: str = "ADE",
    trim_ratio: float = 1.0,
    horizon: int | None = None,
    resume: bool = False,
    on_error: str = "raise",
) -> None:
    """
    Gera previsões combinadas via ADE para todas as séries do dataset.

    Args:
        dataset_name: 'ANP_MONTHLY', 'NN5_WEEKLY_DATASET', 'M4_WEEKLY_DATASET', …
        models:       modelos base; default = os 19 de `DEFAULT_MODELS`
        exp_name:     subpasta de saída em resultados/ (default 'ADE')
        trim_ratio:   fração dos melhores especialistas a manter (1.0 = todos)
        horizon:      sobrescreve o horizonte lido do CSV
        resume:       continua de onde parou em vez de apagar a saída
        on_error:     'raise' aborta na primeira série que falhar (default,
                      porque saída incompleta desalinha a comparação no MCM);
                      'skip' registra a falha e segue.
    """
    models = list(models or DEFAULT_MODELS)
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error deve ser 'raise' ou 'skip'")

    spec = resolve_spec(dataset_name, models, horizon=horizon)
    freq_ade = normalize_freq_for_ade(spec)
    print(f"[ADE] {describe(spec)} modelos={len(models)}")

    validate_models_have_dataset(models, dataset_name)
    model_dfs = read_all_model_dfs(models, dataset_name)
    check_windows_alignment(model_dfs)

    if spec.n_windows < 2:
        raise ValueError(
            f"'{dataset_name}' tem {spec.n_windows} janela por série; o ADE precisa "
            f"de pelo menos 2 (uma de validação + uma de teste)."
        )

    combined_df, df_predictions, test_per_series = _build_long_dataframes(
        models, model_dfs, freq=spec.freq
    )

    done = prepare_output(exp_name, dataset_name, resume)
    unique_ids = sorted(df_predictions["unique_id"].unique(), key=int)
    todo = [u for u in unique_ids if int(u) not in done]
    print(f"[ADE] {len(todo)} série(s) a processar de {len(unique_ids)}")

    ref_df = model_dfs[models[0]]
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    for n, uid in enumerate(todo, start=1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = _ade_predict_one_series(
                    combined_df, df_predictions, uid,
                    horizon=spec.horizon, freq_ade=freq_ade, trim_ratio=trim_ratio,
                )
        except Exception as exc:
            if on_error == "raise":
                raise RuntimeError(
                    f"ADE falhou na série dataset_index={uid} de '{dataset_name}'. "
                    f"Use --on-error skip para pular séries com problema."
                ) from exc
            print(f"  [{uid}] FALHOU ({type(exc).__name__}: {exc}) — pulada")
            failures.append((uid, f"{type(exc).__name__}: {exc}"))
            continue

        latest_row = ref_df[ref_df["dataset_index"] == int(uid)].sort_values("start_test").iloc[-1]
        aux.save_to_csv(
            exp_name=exp_name,
            predictions=preds,
            test_values=test_per_series[int(uid)].values,
            dataset_name=dataset_name,
            dataset_index=int(uid),
            horizon=spec.horizon,
            start_test=latest_row["start_test"],
            final_test=latest_row["final_test"],
        )

        elapsed = time.time() - t0
        eta = elapsed / n * (len(todo) - n)
        print(f"  [{n}/{len(todo)}] série {uid} salva  ({elapsed:.0f}s decorridos, ETA {eta:.0f}s)")

    print(f"\n[ADE] Concluído: {output_csv_path(exp_name, dataset_name)}")
    if failures:
        print(f"[ADE] {len(failures)} série(s) falharam e NÃO estão no CSV:")
        for uid, msg in failures:
            print(f"  - dataset_index={uid}: {msg}")
        print("  Atenção: a saída está incompleta e não é comparável no MCM.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m combinations.ade",
        description="Combinação de previsões via ADE (metaforecast).",
    )
    p.add_argument("--dataset", required=True, help="ex: ANP_MONTHLY, M4_WEEKLY_DATASET")
    p.add_argument("--models", nargs="+", default=None, help="default: os 19 modelos base")
    p.add_argument("--exp-name", default="ADE", help="subpasta de saída em resultados/")
    p.add_argument("--trim-ratio", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=None, help="default: lido do CSV")
    p.add_argument("--resume", action="store_true", help="continua em vez de apagar a saída")
    p.add_argument("--on-error", choices=["raise", "skip"], default="raise")
    args = p.parse_args(argv)

    ade_combination(
        dataset_name=args.dataset,
        models=args.models,
        exp_name=args.exp_name,
        trim_ratio=args.trim_ratio,
        horizon=args.horizon,
        resume=args.resume,
        on_error=args.on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
