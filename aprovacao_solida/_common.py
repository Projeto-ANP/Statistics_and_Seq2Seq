"""Helpers compartilhados pelos experimentos de `aprovacao_solida/`.

Todos os scripts desta pasta sao executados a partir da raiz do repositorio:

    python3 aprovacao_solida/<subpasta>/<script>.py ...

`_common.py` garante que a raiz esteja no sys.path e reconstrói a configuracao
exata usada pelos runs v5 (lida dos artefatos desses runs), para que qualquer
comparacao seja apples-to-apples.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react.config import ReactConfig  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR, load_series_source  # noqa: E402

DEFAULT_RESULTS_DIR = ingest_mod.DEFAULT_RESULTS_DIR  # ./timeseries/mestrado/resultados

DATASETS = [
    "ANP_MONTHLY",
    "NN5_WEEKLY_DATASET",
    "M4_WEEKLY_DATASET",
    "ETTH1",
    "ETTH2",
    "ETTM1",
    "ETTM2",
]

#: mesmo mapeamento de run_tsf_batch.py / run_tsf_baselines.py
SOURCE_FILES = {
    "ANP_MONTHLY": "mes_11_venda_mensal.tsf",
    "NN5_WEEKLY_DATASET": "nn5_weekly_dataset.tsf",
    "M4_WEEKLY_DATASET": "m4_weekly_dataset.tsf",
    "ETTH1": "ETTH1.tsf",
    "ETTH2": "ETTH2.tsf",
    "ETTM1": "ETTM1.tsf",
    "ETTM2": "ETTM2.tsf",
}

#: configuracao lida dos artefatos dos runs v5 (ver README.md da pasta)
DEFAULT_METHOD_FOLDERS = {
    "CREST": "orchestrator_react_v5",
    "CREST_qwen": "orchestrator_react_v5_qwen",
    "CREST_gemma": "orchestrator_react_v5_gemma26",
    "sem_agente": "orchestrator_baseline_v1",
}


def v5_config(no_pooled: bool = False, name: str = "search") -> ReactConfig:
    """A configuracao dos runs v5, campo a campo (lida dos artefatos)."""
    cfg = ReactConfig()
    cfg.name = name
    cfg.n_validation_windows = 3
    cfg.backtest_mode = "expanding"
    cfg.nested_selection = True
    cfg.min_windows_for_ols = 5
    cfg.max_iterations = 12
    cfg.early_stop_patience = 4
    cfg.final_strategy = "argmin"
    cfg.seed_stable_pools = True
    cfg.pooled_meta_model = not no_pooled
    cfg.pooled_meta_model_min_series = 20
    cfg.pooled_meta_model_objective = "fforma"
    cfg.seed_pooled_meta_model = True
    cfg.dataset_card = True
    cfg.pool_mode = "full"
    cfg.score_preset = "balanced"
    return cfg


def load_context(
    models,
    dataset,
    results_dir=DEFAULT_RESULTS_DIR,
    source_dir=DEFAULT_SOURCE_DIR,
    use_source=True,
):
    """CSVs de resultado (frames), numero de series e o .tsf, como o pipeline faz."""
    frames = ingest_mod.load_dataset_frames(models, dataset, results_dir)
    n_series = ingest_mod.count_series(dataset, models[0], results_dir)
    source = None
    if use_source:
        source_file = SOURCE_FILES.get(dataset)
        if source_file:
            try:
                source = load_series_source(
                    source_file, n_expected_series=n_series, source_dir=source_dir
                )
            except Exception as exc:  # .tsf opcional: o perfil cai nas janelas
                print(
                    f"WARNING: fonte {source_file!r} nao carregada ({exc}); "
                    "o perfil vira fallback das janelas",
                    flush=True,
                )
    return frames, n_series, source


def detect_drop(models, dataset, first_index, config, results_dir, frames):
    """Modelos com janelas desalinhadas (ETTM), como pipeline.run_dataset faz."""
    drop = []
    if config.drop_misaligned_models:
        try:
            bad = ingest_mod.find_misaligned_models(
                models,
                dataset,
                first_index,
                results_dir=results_dir,
                frames=frames,
                n_windows=config.n_validation_windows,
            )
        except Exception:
            bad = {}
        drop = sorted(bad)
    return drop


def load_dataset_csv(folder, dataset, results_dir=DEFAULT_RESULTS_DIR):
    """CSV de um metodo, filtrado para a janela de teste mais recente por serie.

    Mesmo filtro do `outputs/resultados/scripts/common.py::load_filtered`:
    agrupa por (dataset_index, regressor) e mantem a linha com `final_test`
    mais recente, que e a unica janela de teste real.
    """
    path = os.path.join(results_dir, folder, f"{dataset}.csv")
    if not os.path.isfile(path):
        path = os.path.join(results_dir, folder, "normal", f"{dataset}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, sep=";", engine="python")
    if "final_test" in df.columns:
        df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
        df = df.sort_values("final_test", ascending=False)
    df = df.drop_duplicates(subset=["dataset_index", "regressor"], keep="first")
    return df.sort_values("dataset_index").reset_index(drop=True)


SEARCH_ARMS = ("seeds", "random", "greedy", "ucb")


def collect_methods(dataset, results_dir=DEFAULT_RESULTS_DIR, search_version=None, extra_folders=None):
    """Todos os metodos disponiveis no results-dir para um dataset, indexados.

    Inclui os braços de busca (se `search_version` for dado), os runs com e sem
    agente, e as combinacoes estaticas/FFORMA/ADE quando presentes. Adiciona
    `search_best`: por serie, o braço de busca de menor score de VALIDACAO.
    """
    methods = {}
    if search_version:
        sdf = load_dataset_csv(f"orchestrator_search_{search_version}", dataset, results_dir)
        if sdf is not None and not sdf.empty:
            for reg, sub in sdf.groupby("regressor"):
                name = str(reg).replace("search_", "")
                methods[name] = sub.drop_duplicates(subset=["dataset_index"]).set_index("dataset_index")

    folders = dict(DEFAULT_METHOD_FOLDERS)
    folders.update({"median": "median", "FFORMA": "FFORMA", "ADE": "ADE"})
    if extra_folders:
        folders.update(extra_folders)
    for label, folder in folders.items():
        df = load_dataset_csv(folder, dataset, results_dir)
        if df is not None and not df.empty:
            methods[label] = df.drop_duplicates(subset=["dataset_index"]).set_index("dataset_index")

    arms = [a for a in SEARCH_ARMS if a in methods and "best_score" in methods[a].columns]
    if len(arms) >= 2:
        idx = methods[arms[0]].index
        for a in arms[1:]:
            idx = idx.intersection(methods[a].index)
        if len(idx):
            scores = pd.concat(
                {a: methods[a].loc[idx, "best_score"].astype(float) for a in arms}, axis=1
            )
            pick = scores.idxmin(axis=1)
            frames = [methods[str(pick.loc[i])].loc[i] for i in idx]
            methods["search_best"] = pd.DataFrame(frames, index=idx)
    return methods


def append_rows(path, rows):
    """Escreve/atualiza um CSV com `;`, substituindo linhas da mesma chave."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame(rows)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        old = pd.read_csv(path, sep=";", engine="python")
        combined = pd.concat([old, new], ignore_index=True)
        key = [c for c in ("dataset_index", "regressor") if c in combined.columns]
        if key:
            combined = combined.drop_duplicates(subset=key, keep="last")
        combined.to_csv(path, sep=";", index=False)
    else:
        new.to_csv(path, sep=";", index=False)
