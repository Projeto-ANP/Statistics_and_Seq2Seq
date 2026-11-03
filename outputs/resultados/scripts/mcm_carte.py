"""MCM do CARTE por conjunto de dados e agregada.

Compara cinco metodos por dataset: melhor modelo individual, melhor combinacao
estatica, FFORMA, ADE e CARTE. Gera SMAPE e POCID por dataset e as duas MCMs
agregadas. As figuras finais sao gravadas em overleaf/resultados_final.

Requer o ambiente conda `metrics`:
    conda run -n metrics python outputs/resultados/scripts/mcm_carte.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import pandas as pd  # noqa: E402

from multi_comp_matrix import MCM  # noqa: E402


CARTE_FOLDER = "orchestrator_react_v12_gptoss_low"
CARTE_LABEL = "CARTE"
INDIVIDUAL_MODELS = list(C.INDIVIDUAL_MODELS)
STATIC_FOLDERS = ["mean", "median", "dba"]
METRICS = [
    ("smape", "SMAPE", "increasing"),
    ("pocid", "POCID", "decreasing"),
]
OUT_DIR = os.path.join(C.REPO_ROOT, "overleaf", "resultados_final")
AUX_DIR = os.path.join(C.OUTPUTS_BASE, "mcm_carte")


def load_latest(folder: str, dataset: str) -> pd.DataFrame:
    """Le o CSV e mantem a janela de teste mais recente por serie."""
    direct = os.path.join(C.RESULTADOS_BASE, folder, f"{dataset}.csv")
    nested = os.path.join(C.RESULTADOS_BASE, folder, "normal", f"{dataset}.csv")
    path = direct if os.path.isfile(direct) else nested
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=";", engine="python")
    df["final_test"] = pd.to_datetime(df["final_test"])
    df = df.sort_values("final_test", ascending=False)
    df = df.drop_duplicates(subset=["dataset_index"], keep="first")
    return df.sort_values("dataset_index").reset_index(drop=True)


def mean_metric(folder: str, dataset: str, metric: str) -> float:
    df = load_latest(folder, dataset)
    return float(pd.to_numeric(df[metric], errors="coerce").mean())


def pick_best(folders, dataset: str | None, metric: str = "smape") -> str:
    """Menor media de `metric`; dataset=None agrega todas as series."""
    scores = {}
    for folder in folders:
        if dataset is None:
            vals = []
            for ds in C.DATASETS:
                vals.extend(pd.to_numeric(load_latest(folder, ds)[metric], errors="coerce").tolist())
            scores[folder] = float(pd.Series(vals).mean())
        else:
            scores[folder] = mean_metric(folder, dataset, metric)
    return min(scores, key=scores.get)


def wide_for_dataset(dataset: str, best_ind: str, best_static: str, metric: str) -> pd.DataFrame:
    folders = [best_ind, best_static, "FFORMA", "ADE", CARTE_FOLDER]
    labels = [
        C.display_name(best_ind),
        C.display_name(best_static),
        "FFORMA",
        "ADE",
        CARTE_LABEL,
    ]
    series = {}
    for folder in folders:
        df = load_latest(folder, dataset).set_index("dataset_index")[metric]
        series[folder] = pd.to_numeric(df, errors="coerce")
    common_idx = None
    for s in series.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    common_idx = common_idx.sort_values()
    data = {label: series[folder].loc[common_idx].values for folder, label in zip(folders, labels)}
    return pd.DataFrame(data, index=[f"{dataset}_{i}" for i in common_idx])


def wide_aggregate(best_ind: str, best_static: str, metric: str) -> pd.DataFrame:
    frames = [wide_for_dataset(ds, best_ind, best_static, metric) for ds in C.DATASETS]
    return pd.concat(frames, axis=0)



def wide_mean_by_dataset(best_ind: str, best_static: str, metric: str) -> pd.DataFrame:
    """Uma linha por dataset, com a media da metrica entre as series."""
    folders = [best_ind, best_static, "FFORMA", "ADE", CARTE_FOLDER]
    labels = [
        C.display_name(best_ind),
        C.display_name(best_static),
        "FFORMA",
        "ADE",
        CARTE_LABEL,
    ]
    rows = {}
    for dataset in C.DATASETS:
        rows[dataset] = {}
        for folder, label in zip(folders, labels):
            df = load_latest(folder, dataset)
            rows[dataset][label] = float(pd.to_numeric(df[metric], errors="coerce").mean())
    return pd.DataFrame.from_dict(rows, orient="index")

def run_mcm(frame: pd.DataFrame, savename: str, metric_label: str, order_better: str, include_proba: bool = True):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUX_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, f"{savename}.png")
    if os.path.isfile(png_path) and os.path.getsize(png_path) > 0:
        print(f"MCM ja existe, pulando: {savename}")
        return
    frame.to_csv(os.path.join(AUX_DIR, f"mcm_input_{savename}.csv"), index=True)
    mcm_df = frame.reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # PDF/CSV ficam apenas em outputs/; em overleaf/resultados_final
        # gravamos somente o PNG usado pelo capitulo.
        MCM.compare(
            df_results=mcm_df,
            output_dir=AUX_DIR + os.sep,
            png_savename=savename,
            pdf_savename=savename,
            csv_savename=savename,
            used_statistic=metric_label,
            order_better=order_better,
            **C.mcm_plot_kwargs(include_ProbaWinTieLoss=include_proba),
        )
    shutil.move(os.path.join(AUX_DIR, f"{savename}.png"), png_path)
    print(f"MCM salva: {png_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUX_DIR, exist_ok=True)

    per_dataset = {}
    for dataset in C.DATASETS:
        best_ind = pick_best(INDIVIDUAL_MODELS, dataset)
        best_static = pick_best(STATIC_FOLDERS, dataset)
        per_dataset[dataset] = {
            "melhor_individual": best_ind,
            "melhor_estatica": best_static,
        }
        for metric, metric_label, order_better in METRICS:
            suffix = "" if metric == "smape" else "_pocid"
            frame = wide_for_dataset(dataset, best_ind, best_static, metric)
            run_mcm(frame, f"mcm_{dataset}{suffix}", metric_label, order_better)

    best_ind_global = pick_best(INDIVIDUAL_MODELS, None)
    best_static_global = pick_best(STATIC_FOLDERS, None)
    for metric, metric_label, order_better in METRICS:
        frame = wide_aggregate(best_ind_global, best_static_global, metric)
        run_mcm(frame, f"mcm_agregado_{metric}", metric_label, order_better, include_proba=False)
        mean_frame = wide_mean_by_dataset(best_ind_global, best_static_global, metric)
        run_mcm(
            mean_frame,
            f"mcm_media_datasets_{metric}",
            metric_label,
            order_better,
            include_proba=True,
        )

    manifest = {
        "carte_folder": CARTE_FOLDER,
        "per_dataset": per_dataset,
        "global": {
            "melhor_individual": best_ind_global,
            "melhor_estatica": best_static_global,
        },
    }
    with open(os.path.join(AUX_DIR, "mcm_carte_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
