"""Oráculos ex-post do CARTE no protocolo das dez execuções.

A coluna do CARTE usa a média das dez execuções (sementes 11 a 20). Os
oráculos são calculados execução a execução e depois agregados, para que a
comparação tenha o mesmo protocolo do CARTE:

  - melhor tentativa: menor SMAPE de teste entre as tentativas do agente,
    em cada execução; média das dez;
  - melhor das dez primeiras: o mesmo, restrito às tentativas a1 a a10;
  - teto de validação: tentativa de menor score de validação, em cada
    execução, e seu resultado no teste; média das dez;
  - oráculo de teste sobre U: menor SMAPE de teste (e maior POCID) no
    universo formado pelas tentativas do agente na execução e pelas
    tentativas das buscas sem LLM; média das dez;
  - melhor modelo individual: menor SMAPE de teste entre os dezenove
    preditores, usando apenas os modelos cujo vetor de teste coincide com o
    do CARTE.

Para POCID, cada oráculo usa o mesmo critério, trocando mínimo por máximo.
O teto de validação escolhe a tentativa pelo score de validação, que é
calculado apenas com SMAPE, e reporta o POCID dessa mesma tentativa.

Saídas:
  outputs/resultados/ablacao_oraculos_carte/por_serie.csv
  outputs/resultados/ablacao_oraculos_carte/resumo.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402

BASE = os.path.join(_REPO, "timeseries", "mestrado", "resultados")
RUNS = [f"orchestrator_react_consist_seed{s}" for s in range(11, 21)]
MEAN_RUN = "orchestrator_react_consist_seed11_20_mean"
SEARCH_RUN = "orchestrator_search_carte"
SEARCH_ARTIFACTS = os.path.join(
    _REPO, "outputs", "resultados", "ablacao_search_carte", "artifacts"
)
OUT_DIR = os.path.join(_REPO, "outputs", "resultados", "ablacao_oraculos_carte")
DATASETS = [
    "ANP_MONTHLY",
    "NN5_WEEKLY_DATASET",
    "M4_WEEKLY_DATASET",
    "ETTH1",
    "ETTH2",
    "ETTM1",
    "ETTM2",
]
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
FIRST10 = {f"a{i}" for i in range(1, 11)}


def parse_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.asarray([float(v) for v in NUMBER_RE.findall(str(value))], dtype=float)


def load_individual_models(dataset: str) -> dict[str, dict[int, tuple[np.ndarray, float, float]]]:
    out = {}
    for model in DEFAULT_MODELS:
        path = os.path.join(BASE, model, "normal", f"{dataset}.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, sep=";")
        df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
        per = {}
        for idx, group in df.groupby("dataset_index"):
            group = group.sort_values("final_test")
            row = group.iloc[-1]
            per[int(idx)] = (
                parse_array(row["test"]),
                float(row["smape"]) * 100.0,
                float(row["pocid"]),
            )
        out[model] = per
    return out


def metrics_from_attempt(attempt: dict) -> tuple[float, float, float] | None:
    """Devolve (validation_score, test_smape_pct, test_pocid_pct)."""
    test_smape = attempt.get("test_smape")
    if test_smape is None:
        return None
    pocid = None
    metrics = attempt.get("test_metrics") or {}
    if isinstance(metrics, dict):
        pocid = metrics.get("pocid")
    if pocid is None:
        pocid = attempt.get("test_pocid")
    if pocid is None:
        return None
    validation = attempt.get("validation_score")
    if validation is None:
        validation = attempt.get("score")
    validation = float(validation) if validation is not None else np.nan
    return validation, float(test_smape) * 100.0, float(pocid)


def read_attempts(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    out = []
    for attempt in data.get("ranking_trace", {}).get("attempts", []) or []:
        value = metrics_from_attempt(attempt)
        if value is None:
            continue
        out.append(
            {
                "id": attempt.get("id"),
                "validation": value[0],
                "smape": value[1],
                "pocid": value[2],
                "spec": attempt.get("spec"),
            }
        )
    return out


def read_search_attempts(dataset: str, idx: int, labels: list[str]) -> list[dict]:
    out = []
    for label in labels:
        path = os.path.join(SEARCH_ARTIFACTS, dataset, f"dataset_{idx}_{label}.json")
        if not os.path.isfile(path):
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for attempt in data.get("attempts", []) or []:
            value = metrics_from_attempt(attempt)
            if value is None:
                continue
            out.append(
                {
                    "id": attempt.get("id"),
                    "validation": value[0],
                    "smape": value[1],
                    "pocid": value[2],
                    "spec": attempt.get("spec"),
                }
            )
    return out


def search_labels(dataset: str) -> list[str]:
    path = os.path.join(BASE, SEARCH_RUN, f"{dataset}.csv")
    if not os.path.isfile(path):
        return []
    try:
        df = pd.read_csv(path, sep=";", usecols=["regressor"])
    except Exception:
        return []
    labels = sorted({str(v).replace("search_", "") for v in df["regressor"].unique()})
    return labels


def best_by_test(attempts: list[dict], key: str, maximize: bool = False) -> dict | None:
    values = [a for a in attempts if np.isfinite(a.get(key, np.nan))]
    if not values:
        return None
    return max(values, key=lambda a: a[key]) if maximize else min(values, key=lambda a: a[key])


def best_by_validation(attempts: list[dict]) -> dict | None:
    values = [a for a in attempts if np.isfinite(a.get("validation", np.nan))]
    if not values:
        return None
    return min(values, key=lambda a: a["validation"])


def mean_or_nan(values: list[float]) -> float:
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def build_rows(datasets: list[str], indices: set[int] | None = None) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        mean_path = os.path.join(BASE, MEAN_RUN, f"{dataset}.csv")
        seed11_path = os.path.join(BASE, RUNS[0], f"{dataset}.csv")
        if not os.path.isfile(mean_path):
            print(f"WARNING: {mean_path} ausente; usando {RUNS[0]}", flush=True)
            mean_path = seed11_path
        mean = pd.read_csv(mean_path, sep=";")
        seed11 = pd.read_csv(seed11_path, sep=";")
        seed11 = seed11.sort_values("final_test").drop_duplicates("dataset_index", keep="last")
        test_by_idx = {
            int(row.dataset_index): parse_array(row.test) for row in seed11.itertuples()
        }
        models = load_individual_models(dataset)
        labels = search_labels(dataset)
        if not labels:
            print(f"WARNING: buscas sem LLM ausentes para {dataset}", flush=True)

        for _, row in mean.iterrows():
            idx = int(row["dataset_index"])
            if indices is not None and idx not in indices:
                continue
            test = test_by_idx.get(idx)
            if test is None:
                continue

            run_attempts = []
            for run in RUNS:
                path = os.path.join(
                    BASE, run, "llm_artifacts", dataset, f"dataset_{idx}.json"
                )
                run_attempts.append(read_attempts(path))
            search = read_search_attempts(dataset, idx, labels)

            best_attempt_smape = []
            best_attempt_pocid = []
            best_first10_smape = []
            best_first10_pocid = []
            val_ceiling_smape = []
            val_ceiling_pocid = []
            test_oracle_smape = []
            test_oracle_pocid = []

            for attempts in run_attempts:
                if attempts:
                    best = best_by_test(attempts, "smape")
                    if best is not None:
                        best_attempt_smape.append(best["smape"])
                    best_p = best_by_test(attempts, "pocid", maximize=True)
                    if best_p is not None:
                        best_attempt_pocid.append(best_p["pocid"])

                    first10 = [a for a in attempts if str(a.get("id")) in FIRST10]
                    if first10:
                        best10 = best_by_test(first10, "smape")
                        if best10 is not None:
                            best_first10_smape.append(best10["smape"])
                        best10_p = best_by_test(first10, "pocid", maximize=True)
                        if best10_p is not None:
                            best_first10_pocid.append(best10_p["pocid"])

                    picked = best_by_validation(attempts)
                    if picked is not None:
                        val_ceiling_smape.append(picked["smape"])
                        val_ceiling_pocid.append(picked["pocid"])

                universe = attempts + search
                if universe:
                    best_u = best_by_test(universe, "smape")
                    if best_u is not None:
                        test_oracle_smape.append(best_u["smape"])
                    best_u_p = best_by_test(universe, "pocid", maximize=True)
                    if best_u_p is not None:
                        test_oracle_pocid.append(best_u_p["pocid"])

            best_ind = None
            best_ind_pocid = None
            n_aligned = 0
            for model, per in models.items():
                if idx not in per:
                    continue
                model_test, smape, pocid = per[idx]
                n = min(model_test.size, test.size)
                if not np.allclose(model_test[:n], test[:n], equal_nan=True):
                    continue
                n_aligned += 1
                if best_ind is None or smape < best_ind["smape"]:
                    best_ind = {"model": model, "smape": smape, "pocid": pocid}
                if best_ind_pocid is None or pocid > best_ind_pocid["pocid"]:
                    best_ind_pocid = {"model": model, "smape": smape, "pocid": pocid}

            n_agent = sum(len(a) for a in run_attempts)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_index": idx,
                    "carte_smape": float(row["smape"]) * 100.0,
                    "carte_pocid": float(row["pocid"]),
                    "best_attempt_smape": mean_or_nan(best_attempt_smape),
                    "best_attempt_pocid": mean_or_nan(best_attempt_pocid),
                    "best_first10_smape": mean_or_nan(best_first10_smape),
                    "best_first10_pocid": mean_or_nan(best_first10_pocid),
                    "val_ceiling_smape": mean_or_nan(val_ceiling_smape),
                    "val_ceiling_pocid": mean_or_nan(val_ceiling_pocid),
                    "test_oracle_smape": mean_or_nan(test_oracle_smape),
                    "test_oracle_pocid": mean_or_nan(test_oracle_pocid),
                    "best_individual_smape": best_ind["smape"] if best_ind else np.nan,
                    "best_individual_pocid": best_ind["pocid"] if best_ind else np.nan,
                    "best_individual_smape_model": best_ind["model"] if best_ind else None,
                    "best_individual_pocid_max": best_ind_pocid["pocid"] if best_ind_pocid else np.nan,
                    "best_individual_pocid_max_model": best_ind_pocid["model"] if best_ind_pocid else None,
                    "n_models_aligned": n_aligned,
                    "n_agent_attempts": n_agent,
                    "n_search_attempts": len(search),
                    "n_runs": sum(1 for a in run_attempts if a),
                }
            )
        print(f"{dataset}: {len([r for r in rows if r['dataset'] == dataset])} series", flush=True)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame, n_datasets: int) -> pd.DataFrame:
    metrics = [
        "carte_smape",
        "carte_pocid",
        "best_attempt_smape",
        "best_attempt_pocid",
        "best_first10_smape",
        "best_first10_pocid",
        "val_ceiling_smape",
        "val_ceiling_pocid",
        "test_oracle_smape",
        "test_oracle_pocid",
        "best_individual_smape",
        "best_individual_pocid",
    ]
    out = []
    for dataset, group in rows.groupby("dataset"):
        item = {"dataset": dataset, "n_series": len(group)}
        item.update({m: group[m].mean() for m in metrics})
        out.append(item)
    summary = pd.DataFrame(out)
    total_label = f"Total {len(rows)}" if n_datasets == 7 else f"Total {len(rows)} (parcial)"
    media_label = "Media 7" if n_datasets == 7 else f"Media {n_datasets}"
    for label, group in ((total_label, rows), (media_label, None)):
        item = {"dataset": label, "n_series": len(rows) if group is not None else n_datasets}
        for m in metrics:
            item[m] = group[m].mean() if group is not None else summary[m].mean()
        out.append(item)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Oráculos ex-post do CARTE.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    indices = None
    if args.indices is not None:
        indices = set(int(i) for i in args.indices)
    elif args.limit is not None:
        indices = set(range(int(args.limit)))

    os.makedirs(args.out_dir, exist_ok=True)
    rows = build_rows(args.datasets, indices)
    rows.to_csv(os.path.join(args.out_dir, "por_serie.csv"), index=False)
    summary = summarize(rows, len(args.datasets))
    summary.to_csv(os.path.join(args.out_dir, "resumo.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
