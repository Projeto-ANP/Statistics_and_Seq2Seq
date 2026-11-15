"""Gera apenas as MCMs de POCID usando comparadores escolhidos por POCID.

O MCM de SMAPE continua usando os melhores por SMAPE. Este script corrige o
painel de POCID para comparar o CARTE com o melhor individual e a melhor
combinacao estatica *dessa metrica*, e nao com os escolhidos por SMAPE.
"""
import importlib.util
import os

import pandas as pd

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcm_carte.py")
spec = importlib.util.spec_from_file_location("mcm_carte_base", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ROOT = "/home/lucas/Documents/mestrado/Statistics_and_Seq2Seq"
m.CARTE_FOLDER = "orchestrator_react_consist_seed11_20_mean"
m.OUT_DIR = os.path.join(ROOT, "outputs", "resultados", "mcm_carte_pocid_metric_png")
m.AUX_DIR = os.path.join(ROOT, "outputs", "resultados", "mcm_carte")

os.makedirs(m.OUT_DIR, exist_ok=True)
os.makedirs(m.AUX_DIR, exist_ok=True)


def pick_best_max(folders, dataset, metric="pocid"):
    scores = {}
    for folder in folders:
        if dataset is None:
            vals = []
            for ds in m.C.DATASETS:
                vals.extend(pd.to_numeric(m.load_latest(folder, ds)[metric], errors="coerce").tolist())
            scores[folder] = float(pd.Series(vals).mean())
        else:
            scores[folder] = m.mean_metric(folder, dataset, metric)
    return max(scores, key=scores.get)


def main():
    per_dataset = {}
    for dataset in m.C.DATASETS:
        best_ind_p = pick_best_max(m.INDIVIDUAL_MODELS, dataset, "pocid")
        best_static_p = pick_best_max(m.STATIC_FOLDERS, dataset, "pocid")
        per_dataset[dataset] = {
            "melhor_individual_pocid": best_ind_p,
            "melhor_estatica_pocid": best_static_p,
        }
        frame = m.wide_for_dataset(dataset, best_ind_p, best_static_p, "pocid")
        m.run_mcm(frame, f"mcm_{dataset}_pocid", "POCID", "decreasing")

    best_ind_p = pick_best_max(m.INDIVIDUAL_MODELS, None, "pocid")
    best_static_p = pick_best_max(m.STATIC_FOLDERS, None, "pocid")
    frame = m.wide_aggregate(best_ind_p, best_static_p, "pocid")
    m.run_mcm(frame, "mcm_agregado_pocid", "POCID", "decreasing", include_proba=False)
    mean_frame = m.wide_mean_by_dataset(best_ind_p, best_static_p, "pocid")
    m.run_mcm(mean_frame, "mcm_media_datasets_pocid", "POCID", "decreasing", include_proba=True)

    import json
    manifest = {
        "carte_folder": m.CARTE_FOLDER,
        "criterion": "POCID: melhor individual e melhor estatica por POCID",
        "per_dataset": per_dataset,
        "global": {
            "melhor_individual_pocid": best_ind_p,
            "melhor_estatica_pocid": best_static_p,
        },
    }
    with open(os.path.join(m.AUX_DIR, "mcm_carte_pocid_metric_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
