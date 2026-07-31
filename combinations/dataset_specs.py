"""
Registro de datasets + carregadores compartilhados por `ade.py` e `fforma.py`.

Antes, cada script carregava três dicionários hardcoded (`map_dataset_name_to_freq`,
`seasonality_map`) e uma cópia da lista de modelos. Adicionar um dataset novo
exigia editar os dois arquivos em quatro lugares — e eles chegaram a divergir
(ade usava freq="M" para o ANP, fforma usava "ME"; ade usava "H" para o ETTH,
fforma usava "h").

Aqui a regra é: **o CSV é a fonte da verdade para o que dá para medir**
(horizonte, número de séries, número de janelas, passo temporal) e o registro
abaixo só guarda o que é escolha do pesquisador (sazonalidade). Um dataset não
registrado ainda roda: `resolve_spec` infere a frequência a partir do espaçamento
real dos timestamps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import aux

BASE_RESULTS = "./timeseries/mestrado/resultados"

# Os 19 modelos base usados nos experimentos. Era uma lista duplicada no
# `__main__` de ade.py e fforma.py.
DEFAULT_MODELS = [
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


# ---------------------------------------------------------------------------
# Spec de dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """
    freq        alias pandas para reconstruir o eixo temporal (`pd.date_range`).
                Use os aliases modernos ('h', 'ME'); 'H' e 'M' foram removidos
                no pandas 3.
    freq_ade    chave aceita por `ADE.WINDOW_SIZE_BY_FREQ`, que ainda usa os
                aliases antigos ('H', 'D', 'W', 'M', 'ME', 'MS', 'Q', 'QS', 'Y').
                É deliberadamente separada de `freq`: são vocabulários distintos.
    seasonality período sazonal passado ao `tsfeatures` (só o FFORMA usa).
    horizon     preenchido a partir da coluna `horizon` do CSV por `resolve_spec`.
    """

    name: str
    freq: str
    freq_ade: str
    seasonality: int
    horizon: int | None = None
    n_series: int | None = None
    n_windows: int | None = None


# Sazonalidade é a única coisa aqui que é julgamento, não medição.
#
# NOTA: NN5_WEEKLY_DATASET usa seasonality=7 — que é o valor com que os
# resultados publicados (mcm_output_v5) foram gerados. Para dado *semanal* o
# período natural seria 52 (anual) ou 1 (sem sazonalidade); 7 vem de uma
# confusão com a versão diária do NN5. Mantido como está para não invalidar as
# comparações já rodadas. Trocar exige regerar FFORMA para todos os métodos.
DATASET_SPECS: dict[str, DatasetSpec] = {
    "ETTH1": DatasetSpec("ETTH1", freq="h", freq_ade="H", seasonality=24),
    "ETTH2": DatasetSpec("ETTH2", freq="h", freq_ade="H", seasonality=24),
    # ADE não tem janela para meia-hora; 'H' (window=48, ~1 dia) é a aproximação
    # mais próxima. `ade.normalize_freq_for_ade` avisa quando isso acontece.
    "ETTM1": DatasetSpec("ETTM1", freq="30min", freq_ade="H", seasonality=48),
    "ETTM2": DatasetSpec("ETTM2", freq="30min", freq_ade="H", seasonality=48),
    "ANP_MONTHLY": DatasetSpec("ANP_MONTHLY", freq="ME", freq_ade="ME", seasonality=12),
    "NN5_WEEKLY_DATASET": DatasetSpec(
        "NN5_WEEKLY_DATASET", freq="W", freq_ade="W", seasonality=7  # ver NOTA acima
    ),
    # --- datasets novos ---
    # M4 semanal: a competição M4 trata as séries semanais como não-sazonais
    # (Makridakis et al. 2020), daí seasonality=1.
    "M4_WEEKLY_DATASET": DatasetSpec(
        "M4_WEEKLY_DATASET", freq="7D", freq_ade="W", seasonality=1
    ),
    # US births: diário com sazonalidade semanal (padrão do Monash TSF Archive).
    "US_BIRTHS_DATASET": DatasetSpec(
        "US_BIRTHS_DATASET", freq="D", freq_ade="D", seasonality=7
    ),
}


# Passo temporal medido -> (freq pandas, freq ADE, sazonalidade default).
# Usado só quando o dataset não está em DATASET_SPECS.
_STEP_TABLE: list[tuple[pd.Timedelta, pd.Timedelta, str, str, int]] = [
    # (passo_min, passo_max, freq, freq_ade, seasonality)
    (pd.Timedelta(minutes=25), pd.Timedelta(minutes=35), "30min", "H", 48),
    (pd.Timedelta(minutes=55), pd.Timedelta(minutes=65), "h", "H", 24),
    (pd.Timedelta(hours=23), pd.Timedelta(hours=25), "D", "D", 7),
    (pd.Timedelta(days=6.9), pd.Timedelta(days=7.1), "7D", "W", 1),
    (pd.Timedelta(days=28), pd.Timedelta(days=31.5), "ME", "ME", 12),
    (pd.Timedelta(days=89), pd.Timedelta(days=93), "QE", "Q", 4),
    (pd.Timedelta(days=364), pd.Timedelta(days=367), "YE", "Y", 1),
]


# ---------------------------------------------------------------------------
# I/O compartilhado
# ---------------------------------------------------------------------------

def model_csv_path(model_name: str, dataset_name: str) -> str:
    return f"{BASE_RESULTS}/{model_name}/normal/{dataset_name}.csv"


def validate_models_have_dataset(models: Iterable[str], dataset_name: str) -> None:
    """Levanta FileNotFoundError listando exatamente quais CSVs faltam."""
    missing = [m for m in models if not os.path.exists(model_csv_path(m, dataset_name))]
    if missing:
        paths = "\n  - ".join(model_csv_path(m, dataset_name) for m in missing)
        raise FileNotFoundError(
            f"Os modelos abaixo não possuem resultados para '{dataset_name}':\n"
            f"  - {paths}"
        )


def read_model_df(model_name: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(model_csv_path(model_name, dataset_name), sep=";")
    df["start_test"] = pd.to_datetime(df["start_test"], errors="coerce")
    df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
    return df.sort_values(["dataset_index", "start_test"]).reset_index(drop=True)


def read_all_model_dfs(models: Iterable[str], dataset_name: str) -> dict[str, pd.DataFrame]:
    return {m: read_model_df(m, dataset_name) for m in models}


def check_windows_alignment(model_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Confere que todos os modelos têm o mesmo conjunto de `dataset_index` e o
    mesmo número de janelas por série.

    As *datas* podem divergir entre modelos (splits temporais diferentes); só a
    posição relativa (mais antiga -> mais recente) precisa bater. As datas do
    primeiro modelo da lista viram o eixo canônico.
    """
    ref_name = next(iter(model_dfs))
    ref_counts = {int(k): len(v) for k, v in model_dfs[ref_name].groupby("dataset_index")}

    for name, df in model_dfs.items():
        cur_counts = {int(k): len(v) for k, v in df.groupby("dataset_index")}

        only_ref = sorted(set(ref_counts) - set(cur_counts))
        only_cur = sorted(set(cur_counts) - set(ref_counts))
        if only_ref or only_cur:
            raise ValueError(
                f"Modelo '{name}' tem dataset_index diferentes de '{ref_name}'.\n"
                f"  faltando em '{name}': {only_ref[:20]}{' ...' if len(only_ref) > 20 else ''}\n"
                f"  sobrando em '{name}': {only_cur[:20]}{' ...' if len(only_cur) > 20 else ''}\n"
                f"Remova '{name}' da lista de modelos ou regere o CSV dele."
            )

        bad = [(k, cur_counts[k], ref_counts[k]) for k in ref_counts if cur_counts[k] != ref_counts[k]]
        if bad:
            head = ", ".join(f"idx={k}: {c} vs {r}" for k, c, r in bad[:5])
            raise ValueError(
                f"Modelo '{name}' tem número de janelas diferente de '{ref_name}' "
                f"em {len(bad)} série(s) ({head}{' ...' if len(bad) > 5 else ''}).\n"
                f"Remova '{name}' da lista de modelos ou regere o CSV dele."
            )


def detect_misaligned_models(
    model_dfs: dict[str, pd.DataFrame],
    ref_model: str,
    mismatch_frac_threshold: float = 0.5,
) -> list[str]:
    """
    Modelos cujo alvo de teste (coluna `test` da janela mais recente) diverge
    sistematicamente do modelo de referência, mesmo que `check_windows_alignment`
    já tenha confirmado que o número de janelas por série bate.

    Caso real que motivou isso: no ETTM1/ETTM2, cinco modelos `ONLY_*` foram
    gerados com passo de 15min (em vez de 30min) e começam um dia depois. O CSV
    tem o mesmo formato e o mesmo número de janelas — só que a "janela de teste"
    deles cobre um trecho diferente da série. Combinar esses modelos com os
    demais soma previsões para alvos diferentes; RMSE explode para a casa dos
    bilhões sem nenhum erro em tempo de execução.

    Um modelo é marcado quando o alvo diverge do de referência em mais de
    `mismatch_frac_threshold` das séries — maioria sistemática, não ruído
    pontual numa série isolada.
    """
    ref_df = model_dfs[ref_model]
    ref_finals: dict[int, np.ndarray] = {}
    for ds_idx, g in ref_df.groupby("dataset_index"):
        row = g.sort_values("start_test").iloc[-1]
        ref_finals[int(ds_idx)] = np.array(aux.extract_values(row["test"]))

    misaligned = []
    for name, df in model_dfs.items():
        if name == ref_model:
            continue
        mismatches = 0
        total = 0
        for ds_idx, g in df.groupby("dataset_index"):
            ref = ref_finals.get(int(ds_idx))
            if ref is None or len(ref) == 0:
                continue
            row = g.sort_values("start_test").iloc[-1]
            t = np.array(aux.extract_values(row["test"]))
            total += 1
            if len(t) != len(ref) or not np.allclose(t, ref, rtol=1e-4, atol=1e-6):
                mismatches += 1
        if total and (mismatches / total) > mismatch_frac_threshold:
            misaligned.append(name)
    return misaligned


def resolve_active_models(
    models: list[str],
    model_dfs: dict[str, pd.DataFrame],
    ref_model: str,
    drop_misaligned: bool = True,
    label: str = "",
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """
    Roda `detect_misaligned_models` e, se `drop_misaligned`, remove os modelos
    sinalizados. Chame depois de `check_windows_alignment` — aquela função só
    garante que a CONTAGEM de janelas bate; esta pega o caso em que a contagem
    bate mas o CONTEÚDO da janela de teste não.
    """
    misaligned = detect_misaligned_models(model_dfs, ref_model)
    if not misaligned:
        return models, model_dfs

    prefix = f"[{label}] " if label else ""
    if not drop_misaligned:
        print(
            f"{prefix}Aviso: {len(misaligned)} modelo(s) com alvo de teste divergente "
            f"do de referência ('{ref_model}'), MANTIDOS por configuração: {misaligned}"
        )
        return models, model_dfs

    print(
        f"{prefix}Aviso: {len(misaligned)} modelo(s) com alvo de teste divergente "
        f"do de referência ('{ref_model}') e por isso DROPADOS: {misaligned}"
    )
    kept = [m for m in models if m not in misaligned]
    return kept, {m: model_dfs[m] for m in kept}


def window_dates(start: pd.Timestamp, n: int, freq: str) -> pd.DatetimeIndex:
    """
    Eixo temporal de uma janela com **exatamente** `n` timestamps.

    Usa `periods=n` em vez de `end=final_test` de propósito: com `end=`, qualquer
    desencontro entre o alias de frequência e o passo real dos dados devolve um
    índice mais curto (ou mais longo) que a lista de previsões, e o `zip`/`range`
    que consome esse índice descarta pontos **em silêncio**. Com `periods=n` o
    comprimento é garantido.
    """
    return pd.date_range(start=start, periods=n, freq=freq)


# ---------------------------------------------------------------------------
# Resolução do spec
# ---------------------------------------------------------------------------

def _infer_step(df: pd.DataFrame) -> pd.Timedelta | None:
    """Passo temporal mediano dentro de uma janela, medido nos timestamps."""
    row = df.sort_values(["dataset_index", "start_test"]).iloc[0]
    n = len(aux.extract_values(row["test"]))
    if n < 2:
        return None
    span = pd.Timestamp(row["final_test"]) - pd.Timestamp(row["start_test"])
    return span / (n - 1)


def _spec_from_step(name: str, step: pd.Timedelta) -> DatasetSpec:
    for lo, hi, freq, freq_ade, seas in _STEP_TABLE:
        if lo <= step <= hi:
            print(
                f"[spec] '{name}' não está em DATASET_SPECS. Passo medido={step} "
                f"-> freq='{freq}', freq_ade='{freq_ade}', seasonality={seas}. "
                f"Confirme a sazonalidade e registre em dataset_specs.py."
            )
            return DatasetSpec(name, freq=freq, freq_ade=freq_ade, seasonality=seas)
    raise ValueError(
        f"Não consegui inferir a frequência de '{name}' (passo medido={step}). "
        f"Registre o dataset em DATASET_SPECS (combinations/dataset_specs.py)."
    )


def resolve_spec(
    dataset_name: str,
    models: list[str] | None = None,
    seasonality: int | None = None,
    horizon: int | None = None,
) -> DatasetSpec:
    """
    Devolve o `DatasetSpec` completo, com `horizon`/`n_series`/`n_windows` lidos
    do CSV do modelo de referência e a frequência conferida contra os timestamps.

    `seasonality` e `horizon` sobrescrevem o registro quando passados (flags CLI).
    """
    models = models or DEFAULT_MODELS
    validate_models_have_dataset(models[:1], dataset_name)
    ref_df = read_model_df(models[0], dataset_name)

    step = _infer_step(ref_df)
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        if step is None:
            raise ValueError(
                f"Dataset '{dataset_name}' não registrado e com janela de 1 ponto: "
                f"não dá para inferir a frequência. Registre em DATASET_SPECS."
            )
        spec = _spec_from_step(dataset_name, step)
    elif step is not None:
        # Confere o registro contra a realidade: pega erro de digitação e
        # dataset regerado com outra resolução (foi assim que o ETTM apareceu).
        probe = window_dates(pd.Timestamp(ref_df.iloc[0]["start_test"]), 3, spec.freq)
        declared = probe[1] - probe[0]
        if not (0.75 * declared <= step <= 1.25 * declared):
            raise ValueError(
                f"Frequência declarada para '{dataset_name}' não bate com os dados: "
                f"DATASET_SPECS diz freq='{spec.freq}' (passo {declared}), mas os "
                f"timestamps do modelo '{models[0]}' têm passo {step}. "
                f"Corrija dataset_specs.py ou confira se o CSV foi regerado com "
                f"outra resolução."
            )

    horizons = sorted(int(h) for h in ref_df["horizon"].unique())
    if horizon is None:
        if len(horizons) > 1:
            raise ValueError(
                f"'{dataset_name}' tem horizontes diferentes no CSV: {horizons}. "
                f"Passe --horizon explicitamente."
            )
        horizon = horizons[0]

    counts = ref_df.groupby("dataset_index").size()
    return DatasetSpec(
        name=dataset_name,
        freq=spec.freq,
        freq_ade=spec.freq_ade,
        seasonality=spec.seasonality if seasonality is None else seasonality,
        horizon=horizon,
        n_series=int(ref_df["dataset_index"].nunique()),
        n_windows=int(counts.iloc[0]),
    )


def describe(spec: DatasetSpec) -> str:
    return (
        f"dataset={spec.name} séries={spec.n_series} janelas/série={spec.n_windows} "
        f"horizon={spec.horizon} freq={spec.freq} freq_ade={spec.freq_ade} "
        f"seasonality={spec.seasonality}"
    )


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------

def output_csv_path(exp_name: str, dataset_name: str) -> str:
    return f"{BASE_RESULTS}/{exp_name}/{dataset_name}.csv"


def existing_indices(exp_name: str, dataset_name: str) -> set[int]:
    """dataset_index já gravados na saída (para `--resume`)."""
    path = output_csv_path(exp_name, dataset_name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    try:
        df = pd.read_csv(path, sep=";", engine="python")
    except Exception:
        return set()
    if "dataset_index" not in df.columns:
        return set()
    return {int(x) for x in df["dataset_index"].dropna().unique()}


def prepare_output(exp_name: str, dataset_name: str, resume: bool) -> set[int]:
    """
    `aux.save_to_csv` faz *append*. Sem isso, rerodar um dataset duplica as linhas
    e o `drop_duplicates(keep='first')` do MCM passa a ler as linhas antigas —
    ou seja, o resultado "novo" nunca aparece na comparação.

    resume=False (default): apaga o CSV de saída e começa limpo.
    resume=True: mantém e devolve os índices já feitos, para pular.
    """
    path = output_csv_path(exp_name, dataset_name)
    if resume:
        done = existing_indices(exp_name, dataset_name)
        if done:
            print(f"[resume] {len(done)} série(s) já gravadas em {path} — serão puladas.")
        return done
    if os.path.exists(path):
        print(f"[saída] removendo {path} (use --resume para continuar de onde parou)")
        os.remove(path)
    return set()
