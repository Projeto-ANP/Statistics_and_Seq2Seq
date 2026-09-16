"""Utilidades compartilhadas pelos scripts item_XX_*.py do capitulo de resultados.

Cada item importa deste modulo (nao dos outros scripts de item), para que
todo item_XX_*.py permaneca executavel de forma independente.
"""
import json
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTADOS_BASE = os.path.join(REPO_ROOT, "timeseries", "mestrado", "resultados")
TSF_BASE = os.path.expanduser("~/Documents/mestrado/forecasting_datasets")
OUTPUTS_BASE = os.path.join(REPO_ROOT, "outputs", "resultados")
AVISOS_PATH = os.path.join(OUTPUTS_BASE, "avisos_validacao.txt")

# ---------------------------------------------------------------------------
# Listas de inclusao / exclusao (ver prompt do capitulo de resultados)
# ---------------------------------------------------------------------------

INDIVIDUAL_MODELS = [
    "ARIMA", "ETS", "THETA",
    "rf", "catboost",
    "CWT_rf", "DWT_rf", "FT_rf", "CWT_catboost", "DWT_catboost", "FT_catboost",
    "ONLY_CWT_catboost", "ONLY_CWT_rf", "ONLY_DWT_catboost", "ONLY_DWT_rf",
    "ONLY_FT_catboost", "ONLY_FT_rf",
    "NaiveSeasonal", "NaiveMovingAverage",
]

STATIC_COMBINATIONS = ["mean", "median", "dba"]
FFORMA_ADE = ["FFORMA", "ADE"]
ORCHESTRATOR_GPTOSS = "orchestrator_react_v5"
ORCHESTRATOR_QWEN = "orchestrator_react_v5_qwen"

INCLUDED_FOLDERS = set(INDIVIDUAL_MODELS) | set(STATIC_COMBINATIONS) | set(FFORMA_ADE) | {
    ORCHESTRATOR_GPTOSS, ORCHESTRATOR_QWEN,
}

EXCLUDED_FOLDERS = {
    "ridge", "CWT_ridge", "DWT_ridge", "ONLY_CWT_ridge", "ONLY_DWT_ridge", "ONLY_FT_ridge",
    "CWT_svr", "DWT_svr", "ONLY_CWT_svr", "ONLY_DWT_svr", "ONLY_FT_svr",
    "NaiveDrift", "NBEATS",
    "orchestrator_react_v5_x2",
}

DATASETS = [
    "ANP_MONTHLY", "NN5_WEEKLY_DATASET", "M4_WEEKLY_DATASET",
    "ETTH1", "ETTH2", "ETTM1", "ETTM2",
]

TSF_MAP = {
    "ANP_MONTHLY": "mes_11_venda_mensal.tsf",
    "NN5_WEEKLY_DATASET": "nn5_weekly_dataset.tsf",
    "M4_WEEKLY_DATASET": "m4_weekly_dataset.tsf",
    "ETTH1": "ETTH1.tsf",
    "ETTH2": "ETTH2.tsf",
    "ETTM1": "ETTM1.tsf",
    "ETTM2": "ETTM2.tsf",
}

MAPE_ZERO_THRESHOLD = 1e-3

# ---------------------------------------------------------------------------
# Localizacao dos CSVs de origem
# ---------------------------------------------------------------------------


def resolve_csv_path(folder: str, dataset: str) -> str:
    """Resolve o caminho do CSV de `folder` para `dataset`.

    So aceita pastas da lista de inclusao. Os resultados vivem em dois layouts:
    `{folder}/{dataset}.csv` (combinacoes/FFORMA/ADE/orchestrator) ou
    `{folder}/normal/{dataset}.csv` (modelos individuais).
    """
    if folder in EXCLUDED_FOLDERS:
        raise ValueError(f"pasta na lista de exclusao, leitura proibida: {folder}")
    if folder not in INCLUDED_FOLDERS:
        raise ValueError(f"pasta fora da lista de inclusao: {folder}")

    direct = os.path.join(RESULTADOS_BASE, folder, f"{dataset}.csv")
    if os.path.isfile(direct):
        return direct
    nested = os.path.join(RESULTADOS_BASE, folder, "normal", f"{dataset}.csv")
    if os.path.isfile(nested):
        return nested
    raise FileNotFoundError(f"CSV nao encontrado para folder={folder!r} dataset={dataset!r}")


# ---------------------------------------------------------------------------
# Parsing de campos com listas numericas (test/predictions)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def parse_num_list(value) -> list:
    """Converte a representacao textual de uma lista numerica em list[float].

    Robusto a duas formatacoes observadas nos CSVs de origem: lista Python
    separada por virgulas (`[1.0, 2.0]`) e repr bruto de numpy, sem virgulas
    e possivelmente quebrado em varias linhas (`[1.0 2.0\\n3.0]`).
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if value is None or (isinstance(value, float) and pd.isnull(value)):
        return []
    return [float(tok) for tok in _NUM_RE.findall(str(value))]


# ---------------------------------------------------------------------------
# Filtro obrigatorio: uma linha por (dataset_index, regressor), final_test mais recente
# ---------------------------------------------------------------------------


def load_filtered(folder: str, dataset: str) -> pd.DataFrame:
    """Le o CSV de `folder`/`dataset`, aplica o filtro de janela de teste.

    Agrupa por (dataset_index, regressor) e mantem so a linha com o valor mais
    recente de `final_test` em cada grupo (as demais linhas sao janelas de
    validacao, nao a janela de teste real).
    """
    path = resolve_csv_path(folder, dataset)
    df = pd.read_csv(path, sep=";", engine="python")
    df["final_test"] = pd.to_datetime(df["final_test"])
    df = df.sort_values("final_test", ascending=False)
    df = df.drop_duplicates(subset=["dataset_index", "regressor"], keep="first")
    df = df.sort_values("dataset_index").reset_index(drop=True)
    return df


#: Pasta-sonda usada por pre_test_reference_values: qualquer pasta com >=2
#: janelas de validacao por serie serve (o valor de `test` e o mesmo dado
#: real independente do modelo). "rf" tem >=4 janelas em toda serie de todo
#: dataset (confirmado manualmente) e nunca esta na lista de exclusao.
_PRE_TEST_PROBE_FOLDER = "rf"


def pre_test_reference_values(dataset: str) -> dict:
    """Retorna {dataset_index: valor} com o ultimo ponto observado ANTES do
    inicio da janela de teste mais recente ("passo 0"), para toda serie de
    `dataset`.

    Nao ha coluna de treino nos CSVs de origem, e cruzar contra o .tsf
    original nao e confiavel para todo dataset (ETTM1/ETTM2 divergem do
    .tsf atualmente em disco). Em vez disso, usa a JANELA DE VALIDACAO
    imediatamente anterior a mais recente, que ja esta no CSV bruto (antes
    do filtro de load_filtered) -- a divisao treino/teste e sempre
    `aux_series[:-horizon]` / `aux_series[-horizon:]`, andando para tras em
    blocos de `horizon`, entao o ultimo valor do `test` da segunda janela
    mais recente e exatamente o valor que precede o primeiro ponto da janela
    mais recente. Verificado batendo exatamente com a reconstrucao via .tsf
    nos datasets onde essa reconstrucao e confiavel (M4_WEEKLY_DATASET
    idx=208: as duas dao 4185.45).
    """
    path = resolve_csv_path(_PRE_TEST_PROBE_FOLDER, dataset)
    raw = pd.read_csv(path, sep=";", engine="python")
    raw["final_test"] = pd.to_datetime(raw["final_test"])
    raw = raw.sort_values(["dataset_index", "final_test"], ascending=[True, False])

    values = {}
    for dataset_index, group in raw.groupby("dataset_index"):
        if len(group) < 2:
            continue
        second_most_recent = group.iloc[1]
        test_vals = parse_num_list(second_most_recent["test"])
        if test_vals:
            values[dataset_index] = test_vals[-1]
    return values


# ---------------------------------------------------------------------------
# Validacao: contagem de series contra o .tsf original
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def tsf_series_count(dataset: str) -> int:
    """Conta linhas de serie apos a linha `@data` no .tsf original do dataset."""
    tsf_path = os.path.join(TSF_BASE, TSF_MAP[dataset])
    count = 0
    in_data = False
    # latin-1 never raises on decode (every byte maps to a codepoint) -- some
    # .tsf headers have non-utf8 punctuation in the comment lines above @data,
    # and only line-counting after @data matters here, not the header text.
    with open(tsf_path, encoding="latin-1") as fh:
        for line in fh:
            if in_data:
                if line.strip():
                    count += 1
            elif line.strip().lower() == "@data":
                in_data = True
    return count


def merge_validation_warnings(records) -> None:
    """Mescla `records` (iteravel de tuplas dataset, regressor, esperado, encontrado)
    no arquivo consolidado outputs/resultados/avisos_validacao.txt.

    Faz merge por chave (dataset, regressor) em vez de so anexar linhas, para que
    reexecutar qualquer item_XX de forma isolada ou repetida nao duplique avisos.
    """
    records = list(records)
    if not records:
        return
    os.makedirs(OUTPUTS_BASE, exist_ok=True)
    existing = {}
    if os.path.isfile(AVISOS_PATH):
        with open(AVISOS_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) == 4:
                    existing[(parts[0], parts[1])] = tuple(parts)
    for dataset, regressor, expected, found in records:
        existing[(dataset, regressor)] = (dataset, regressor, str(expected), str(found))
    with open(AVISOS_PATH, "w", encoding="utf-8") as fh:
        for key in sorted(existing.keys()):
            fh.write(";".join(existing[key]) + "\n")


def canonical_regressor_name(folder: str) -> str:
    """Nome de regressor usado nos avisos de validacao e nas tabelas de saida.

    Mantem o nome da pasta para modelos individuais/estaticas/FFORMA/ADE (ja
    e o nome canonico), mas usa "CREST" para as pastas orchestrator_react_v5*,
    igual ao rotulo usado nos itens 04/18.
    """
    if folder in (ORCHESTRATOR_GPTOSS, ORCHESTRATOR_QWEN):
        return "CREST"
    return folder


_STATIC_DISPLAY = {"mean": "Média", "median": "Mediana", "dba": "DBA"}
_FAMILY_DISPLAY = {"catboost": "CatBoost", "rf": "Random Forest"}
_OTHER_DISPLAY = {
    "ARIMA": "ARIMA", "ETS": "ETS", "THETA": "Theta",
    "NaiveSeasonal": "Naive Seasonal", "NaiveMovingAverage": "Naive Moving Average",
}


def display_name(folder: str) -> str:
    """Nome de exibicao em figuras (MCM etc.) para um nome de pasta/regressor.

    catboost/rf -> "X (Original)"; ONLY_{T}_catboost/rf -> "X ({T})";
    {T}_catboost/rf sem ONLY -> "X (Original + {T})", T em {CWT,DWT,FT};
    ARIMA/ETS mantidos; THETA -> "Theta"; Naive* com espaco;
    mean/median/dba -> Média/Mediana/DBA. Qualquer outro nome (FFORMA, ADE,
    CREST, ou um rotulo generico como "melhor_individual") volta inalterado.
    """
    if folder in _STATIC_DISPLAY:
        return _STATIC_DISPLAY[folder]
    if folder in _FAMILY_DISPLAY:
        return f"{_FAMILY_DISPLAY[folder]} (Original)"
    for transform in ("CWT", "DWT", "FT"):
        only_prefix = f"ONLY_{transform}_"
        if folder.startswith(only_prefix):
            base = folder[len(only_prefix):]
            if base in _FAMILY_DISPLAY:
                return f"{_FAMILY_DISPLAY[base]} ({transform})"
        concat_prefix = f"{transform}_"
        if folder.startswith(concat_prefix):
            base = folder[len(concat_prefix):]
            if base in _FAMILY_DISPLAY:
                return f"{_FAMILY_DISPLAY[base]} (Original + {transform})"
    if folder in _OTHER_DISPLAY:
        return _OTHER_DISPLAY[folder]
    return folder


def check_series_count(dataset: str, regressor_label: str, n_series: int):
    """Retorna (dataset, regressor_label, esperado, encontrado) se divergir do
    .tsf original, ou None se bater."""
    expected = tsf_series_count(dataset)
    if n_series != expected:
        return (dataset, regressor_label, expected, n_series)
    return None


# ---------------------------------------------------------------------------
# Agregacao de metricas por (dataset, regressor) -- usada nos itens 01-04, 18
# ---------------------------------------------------------------------------


def aggregate_metrics(dataset: str, folder: str, label: str = None, warn_label: str = None):
    """Agrega mape/smape/rmse/pocid entre series de (dataset, folder).

    RMSE/SMAPE/POCID: media entre todas as series filtradas.
    MAPE: media excluindo series cujo `test` contem algum valor com modulo
    menor que MAPE_ZERO_THRESHOLD.

    `warn_label` nomeia esse (dataset, folder) em avisos_validacao.txt; usar
    quando `label` for reaproveitado por outra fonte (ex.: "CREST" e usado
    tanto para orchestrator_react_v5 quanto para orchestrator_react_v5_qwen,
    e as duas precisam de linhas de aviso distintas).

    Retorna (row_dict, warning_or_None). row_dict tem colunas
    dataset, regressor, mape, smape, rmse, pocid, n_series, n_series_mape.
    """
    label = label or folder
    warn_label = warn_label or label
    df = load_filtered(folder, dataset)
    n_series = len(df)
    warning = check_series_count(dataset, warn_label, n_series)

    test_lists = df["test"].map(parse_num_list)
    eligible = test_lists.map(
        lambda vals: len(vals) > 0 and min(abs(v) for v in vals) >= MAPE_ZERO_THRESHOLD
    )

    row = {
        "dataset": dataset,
        "regressor": label,
        "mape": df.loc[eligible, "mape"].mean(),
        "smape": df["smape"].mean(),
        "rmse": df["rmse"].mean(),
        "pocid": df["pocid"].mean(),
        "n_series": n_series,
        "n_series_mape": int(eligible.sum()),
    }
    return row, warning


def build_metrics_table(dataset_list, folder_label_pairs):
    """Roda aggregate_metrics para todo (dataset, folder) e devolve
    (DataFrame, lista_de_warnings).

    folder_label_pairs: iteravel de (folder, label) ou (folder, label, warn_label).
    """
    rows = []
    warnings = []
    for dataset in dataset_list:
        for entry in folder_label_pairs:
            folder, label = entry[0], entry[1]
            warn_label = entry[2] if len(entry) > 2 else None
            row, warning = aggregate_metrics(dataset, folder, label, warn_label)
            rows.append(row)
            if warning:
                warnings.append(warning)
    return pd.DataFrame(rows), warnings


# ---------------------------------------------------------------------------
# Selecao do melhor individual / melhor estatica por menor SMAPE medio
# ---------------------------------------------------------------------------


def pick_best_per_dataset(kind: str):
    """kind in {'individual', 'estatica'}. Retorna dict dataset -> nome da pasta
    vencedora (menor SMAPE medio no dataset), e lista de warnings de validacao."""
    folders = INDIVIDUAL_MODELS if kind == "individual" else STATIC_COMBINATIONS
    pairs = [(f, f) for f in folders]
    df, warnings = build_metrics_table(DATASETS, pairs)
    best = {}
    for dataset, sub in df.groupby("dataset"):
        best[dataset] = sub.loc[sub["smape"].idxmin(), "regressor"]
    return best, warnings, df


def pick_best_global(kind: str):
    """kind in {'individual', 'estatica'}. Retorna (nome_da_pasta_vencedora,
    lista_de_warnings, df_metricas_por_dataset) escolhendo pelo menor SMAPE
    medio calculado sobre a populacao de series de TODOS os datasets juntos
    (nao a media das medias por dataset)."""
    folders = INDIVIDUAL_MODELS if kind == "individual" else STATIC_COMBINATIONS
    warnings = []
    smape_by_folder = {}
    for folder in folders:
        all_smape = []
        for dataset in DATASETS:
            df = load_filtered(folder, dataset)
            warning = check_series_count(dataset, folder, len(df))
            if warning:
                warnings.append(warning)
            all_smape.append(df["smape"])
        smape_by_folder[folder] = pd.concat(all_smape, ignore_index=True).mean()
    best_folder = min(smape_by_folder, key=smape_by_folder.get)
    return best_folder, warnings, smape_by_folder


# ---------------------------------------------------------------------------
# Populacao "5 metodos" usada nos itens 05-09 (series com prefixo dataset_idx)
# ---------------------------------------------------------------------------

FIVE_METHOD_LABELS = ["melhor_individual", "melhor_estatica", "FFORMA", "ADE", "CREST"]


def five_method_folder_map_per_dataset():
    """Para cada dataset, resolve {label: folder} dos 5 metodos do item 05.
    Retorna (dict dataset -> dict label->folder, lista de warnings)."""
    best_ind, warn_ind, _ = pick_best_per_dataset("individual")
    best_est, warn_est, _ = pick_best_per_dataset("estatica")
    warnings = warn_ind + warn_est
    mapping = {}
    for dataset in DATASETS:
        mapping[dataset] = {
            "melhor_individual": best_ind[dataset],
            "melhor_estatica": best_est[dataset],
            "FFORMA": "FFORMA",
            "ADE": "ADE",
            "CREST": ORCHESTRATOR_GPTOSS,
        }
    return mapping, warnings


def five_method_folder_map_global():
    """Mesma escolha do item 05, mas melhor_individual/melhor_estatica
    escolhidos globalmente (populacao de todos os datasets juntos)."""
    best_ind, warn_ind, _ = pick_best_global("individual")
    best_est, warn_est, _ = pick_best_global("estatica")
    warnings = warn_ind + warn_est
    mapping = {
        "melhor_individual": best_ind,
        "melhor_estatica": best_est,
        "FFORMA": "FFORMA",
        "ADE": "ADE",
        "CREST": ORCHESTRATOR_GPTOSS,
    }
    return mapping, warnings


def build_long_population(dataset_list, label_folder_map, extra_columns=("test", "predictions", "smape", "pocid")):
    """Formato longo: uma linha por (dataset, serie, method), alinhado por
    interseccao dos dataset_index presentes em todos os methods, por dataset.

    label_folder_map: dict dataset -> {label: folder} (por dataset) OU
        {label: folder} (mesmo mapping para todo dataset, ex.: escolha global).
    """
    frames = []
    warnings = []
    for dataset in dataset_list:
        mapping = label_folder_map
        if dataset in mapping and isinstance(mapping[dataset], dict):
            per_label = mapping[dataset]
        else:
            per_label = mapping

        df_by_label = {}
        for label, folder in per_label.items():
            df = load_filtered(folder, dataset)
            warning = check_series_count(dataset, canonical_regressor_name(folder), len(df))
            if warning:
                warnings.append(warning)
            df_by_label[label] = df.set_index("dataset_index")

        common_idx = None
        for df in df_by_label.values():
            common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
        common_idx = common_idx.sort_values()

        for label, df in df_by_label.items():
            sub = df.loc[common_idx, list(extra_columns)].copy()
            sub.insert(0, "method", label)
            sub.insert(0, "serie", [f"{dataset}_{i}" for i in common_idx])
            sub.insert(0, "dataset", dataset)
            frames.append(sub.reset_index(drop=False))

    long_df = pd.concat(frames, axis=0, ignore_index=True)
    return long_df, warnings


def load_per_series_column(folder: str, dataset: str, column: str) -> pd.Series:
    """Le `column` (ja filtrada por janela de teste) indexada por dataset_index."""
    df = load_filtered(folder, dataset)
    s = df.set_index("dataset_index")[column]
    s.name = column
    return s


def build_wide_population(dataset_list, label_folder_map_per_dataset, column: str):
    """Constroi DataFrame wide (linhas=serie com prefixo dataset_idx, colunas=label)
    para `column` (ex.: 'smape' ou 'pocid'), alinhando por interseccao dos
    dataset_index presentes em todos os 5 metodos, por dataset.

    label_folder_map_per_dataset: dict dataset -> {label: folder} (por dataset)
        OU {label: folder} (global, mesmo mapping para todo dataset).
    """
    frames = []
    warnings = []
    for dataset in dataset_list:
        mapping = label_folder_map_per_dataset
        if dataset in mapping and isinstance(mapping[dataset], dict):
            per_label = mapping[dataset]
        else:
            per_label = mapping

        series_by_label = {}
        for label, folder in per_label.items():
            df = load_filtered(folder, dataset)
            warning = check_series_count(dataset, canonical_regressor_name(folder), len(df))
            if warning:
                warnings.append(warning)
            s = df.set_index("dataset_index")[column]
            series_by_label[label] = s

        common_idx = None
        for s in series_by_label.values():
            common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
        common_idx = common_idx.sort_values()

        cols = {label: s.loc[common_idx].values for label, s in series_by_label.items()}
        df_ds = pd.DataFrame(cols, index=[f"{dataset}_{i}" for i in common_idx])
        frames.append(df_ds)

    wide = pd.concat(frames, axis=0)
    wide.index.name = "serie"
    return wide, warnings


# ---------------------------------------------------------------------------
# Orchestrator (items 10-17): parsing dos campos JSON
# ---------------------------------------------------------------------------


def load_orchestrator(folder: str, dataset: str) -> pd.DataFrame:
    """load_filtered especializado para orchestrator_react_v5[_qwen], com as
    colunas JSON (description, tools_called, react_trajectory_json,
    baseline_results_json) ja decodificadas.
    """
    df = load_filtered(folder, dataset)
    df["description_json"] = df["description"].map(json.loads)
    df["tools_called_json"] = df["tools_called"].map(json.loads)
    df["react_trajectory_json_parsed"] = df["react_trajectory_json"].map(json.loads)
    df["baseline_results_json_parsed"] = df["baseline_results_json"].map(json.loads)
    return df


def dget(d, *path, default=None):
    """Acesso encadeado seguro a dicts aninhados: dget(d, 'loop', 'stop_reason')."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------------
# MCM figure style (item_05 / item_07)
# ---------------------------------------------------------------------------

# multi_comp_matrix defaults: fig_size='auto' (~18x6 for 5 methods),
# font_size='auto' (10 for 5x5), colorbar vertical on the right.
# Ajustado para figuras em documento (PDF/LaTeX), onde a imagem costuma ser
# reduzida e o texto auto fica ilegivel.
MCM_FIG_SIZE = (21, 16)              # inches (width, height)
MCM_FONT_SIZE = 17                  # rotulos, texto das celulas, colorbar
MCM_COLORBAR_ORIENTATION = "horizontal"


def mcm_plot_kwargs(**overrides):
    """kwargs compartilhados para multi_comp_matrix.MCM.compare."""
    base = {
        "include_ProbaWinTieLoss": True,
        "pvalue_test": "wilcoxon",
        "save_as_json": False,
        "fig_size": MCM_FIG_SIZE,
        "font_size": MCM_FONT_SIZE,
        "colorbar_orientation": MCM_COLORBAR_ORIENTATION,
    }
    base.update(overrides)
    return base
