"""Resume as buscas sem LLM do CARTE e as compara com a média das dez execuções.

Lê os CSVs de `orchestrator_search_carte/` e a média das dez execuções do
CARTE, agrega as sementes do braço aleatório por série e monta as tabelas da
seção de ablação. O braço `aleatório` é a média das sementes; `guloso` e
`UCB` são determinísticos. O melhor braço é escolhido por série pelo menor
score de validação e reportado no teste.

Saídas:
  outputs/resultados/ablacao_search_carte_resumo/por_serie.csv
  outputs/resultados/ablacao_search_carte_resumo/resumo.csv
  outputs/resultados/ablacao_search_carte_resumo/estatisticas.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

BASE = os.path.join(_REPO, "timeseries", "mestrado", "resultados")
SEARCH_RUN = os.path.join(BASE, "orchestrator_search_carte")
CARTE_RUN = os.path.join(BASE, "orchestrator_react_consist_seed11_20_mean")
OUT_DIR = os.path.join(_REPO, "outputs", "resultados", "ablacao_search_carte_resumo")
DATASETS = [
    "ANP_MONTHLY",
    "NN5_WEEKLY_DATASET",
    "M4_WEEKLY_DATASET",
    "ETTH1",
    "ETTH2",
    "ETTM1",
    "ETTM2",
]
ARMS = ("random", "greedy", "ucb")


def load_carte(dataset: str) -> pd.DataFrame:
    path = os.path.join(CARTE_RUN, f"{dataset}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=";")
    df = df.sort_values("final_test").drop_duplicates("dataset_index", keep="last")
    out = df[["dataset_index", "smape", "pocid"]].copy()
    out["dataset_index"] = out["dataset_index"].astype(int)
    out["carte_smape"] = pd.to_numeric(out["smape"], errors="coerce") * 100.0
    out["carte_pocid"] = pd.to_numeric(out["pocid"], errors="coerce")
    return out[["dataset_index", "carte_smape", "carte_pocid"]]


def load_search(dataset: str) -> pd.DataFrame:
    path = os.path.join(SEARCH_RUN, f"{dataset}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=";")
    df["dataset_index"] = df["dataset_index"].astype(int)
    df["arm"] = df["regressor"].astype(str).str.replace("search_", "", regex=False)
    df["arm"] = df["arm"].str.replace(r"_s\d+$", "", regex=True)
    for col in ("smape", "pocid", "best_score", "n_evals", "elapsed_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["smape"] = df["smape"] * 100.0
    return df[
        ["dataset_index", "arm", "smape", "pocid", "best_score", "n_evals", "elapsed_s"]
    ]


def aggregate_arm(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    sub = df[df["arm"] == arm]
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "dataset_index",
                f"{arm}_smape",
                f"{arm}_pocid",
                f"{arm}_score",
                f"{arm}_n",
                f"{arm}_evals",
                f"{arm}_elapsed",
            ]
        )
    out = (
        sub.groupby("dataset_index")
        .agg(
            smape=("smape", "mean"),
            pocid=("pocid", "mean"),
            score=("best_score", "mean"),
            n=("smape", "size"),
            evals=("n_evals", "mean"),
            elapsed=("elapsed_s", "mean"),
        )
        .reset_index()
    )
    out.columns = [
        "dataset_index",
        f"{arm}_smape",
        f"{arm}_pocid",
        f"{arm}_score",
        f"{arm}_n",
        f"{arm}_evals",
        f"{arm}_elapsed",
    ]
    return out


def build_dataset(dataset: str) -> pd.DataFrame:
    carte = load_carte(dataset)
    expected = len(carte)
    search = load_search(dataset)
    out = carte.copy()
    for arm in ARMS:
        out = out.merge(aggregate_arm(search, arm), on="dataset_index", how="inner")
    if len(out) != expected:
        print(
            f"WARNING: {dataset}: {len(out)} de {expected} series com todos os bracos",
            flush=True,
        )
    out["dataset"] = dataset

    score_cols = [f"{arm}_score" for arm in ARMS]
    smape_cols = [f"{arm}_smape" for arm in ARMS]
    pocid_cols = [f"{arm}_pocid" for arm in ARMS]
    best = out[score_cols].idxmin(axis=1)
    out["best_arm"] = best.astype("string").str.replace("_score", "", regex=False)
    out.loc[best.isna(), "best_arm"] = pd.NA
    best_smape = []
    best_pocid = []
    for _, row in out.iterrows():
        arm = row["best_arm"]
        if isinstance(arm, str):
            best_smape.append(row[f"{arm}_smape"])
            best_pocid.append(row[f"{arm}_pocid"])
        else:
            best_smape.append(np.nan)
            best_pocid.append(np.nan)
    out["best_arm_smape"] = best_smape
    out["best_arm_pocid"] = best_pocid
    out["best_arm_score"] = out[score_cols].min(axis=1)
    return out


def summarize(rows: pd.DataFrame, n_datasets: int) -> pd.DataFrame:
    metrics = [
        "carte_smape",
        "carte_pocid",
        "random_smape",
        "random_pocid",
        "greedy_smape",
        "greedy_pocid",
        "ucb_smape",
        "ucb_pocid",
        "best_arm_smape",
        "best_arm_pocid",
        "random_n",
        "random_evals",
        "greedy_evals",
        "ucb_evals",
        "random_elapsed",
        "greedy_elapsed",
        "ucb_elapsed",
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


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, pos in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * float(pvalues[pos]))
        running = max(running, value)
        adjusted[pos] = running
    return adjusted


def wilcoxon_row(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    """p-valor bilateral de Wilcoxon para amostras pareadas."""
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 0.0, 1.0
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(np.mean(diff)), float(result.pvalue)
    except Exception:
        return float(np.mean(diff)), float("nan")


def statistics(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arm in (*ARMS, "best_arm"):
        for metric, label in (("smape", "SMAPE"), ("pocid", "POCID")):
            a = rows[f"carte_{metric}"].to_numpy()
            b = rows[f"{arm}_{metric}"].to_numpy()
            delta, pvalue = wilcoxon_row(a, b)
            better = a < b if metric == "smape" else a > b
            ties = np.isclose(a, b, atol=1e-9)
            out.append(
                {
                    "arm": arm,
                    "metric": label,
                    "n": len(rows),
                    "delta_carte_menos_braco": delta,
                    "carte_melhor": int(np.sum(better & ~ties)),
                    "empates": int(np.sum(ties)),
                    "braco_melhor": int(np.sum((~better) & ~ties)),
                    "p_wilcoxon": pvalue,
                }
            )
    stats = pd.DataFrame(out)
    for metric in ("SMAPE", "POCID"):
        mask = stats["metric"] == metric
        stats.loc[mask, "p_holm"] = holm(stats.loc[mask, "p_wilcoxon"].tolist())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumo das buscas sem LLM do CARTE.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames = []
    for dataset in args.datasets:
        frame = build_dataset(dataset)
        frames.append(frame)
        print(f"{dataset}: {len(frame)} series", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(os.path.join(args.out_dir, "por_serie.csv"), index=False)
    summary = summarize(rows, len(args.datasets))
    summary.to_csv(os.path.join(args.out_dir, "resumo.csv"), index=False)
    stats = statistics(rows)
    stats.to_csv(os.path.join(args.out_dir, "estatisticas.csv"), index=False)
    print("\n", summary.to_string(index=False), sep="")
    print("\n", stats.to_string(index=False), sep="")


if __name__ == "__main__":
    main()
