"""Erro por passo do horizonte com intervalo de confiança.

Calcula SMAPE e POCID ponto a ponto para os cinco metodos da MCM agregada:

    Random Forest (DWT)  ONLY_DWT_rf
    Mediana              median
    FFORMA               FFORMA
    ADE                  ADE
    CARTE                media das dez execucoes (sementes 11-20)

Para o CARTE, cada execucao e avaliada separadamente e os valores por passo
sao averaged across the ten seeds before the cross-series aggregation. As
figuras mostram a media por passo e uma faixa de confianca de 95% para a
media (nao um painel separado de desvio-padrao).

Saidas:
    outputs/resultados/sintese_horizonte/horizonte_resumo.csv
    overleaf/resultados_final/erro_horizonte_smape.png
    overleaf/resultados_final/erro_horizonte_pocid.png
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RESULTS_BASE = os.path.join(REPO_ROOT, "timeseries", "mestrado", "resultados")
OUTPUT_BASE = os.path.join(REPO_ROOT, "outputs", "resultados", "sintese_horizonte")
FIG_SMAPE = os.path.join(REPO_ROOT, "overleaf", "resultados_final", "erro_horizonte_smape.png")
FIG_POCID = os.path.join(REPO_ROOT, "overleaf", "resultados_final", "erro_horizonte_pocid.png")

DATASETS = [
    "ANP_MONTHLY",
    "NN5_WEEKLY_DATASET",
    "M4_WEEKLY_DATASET",
    "ETTH1",
    "ETTH2",
    "ETTM1",
    "ETTM2",
]
SEEDS = list(range(11, 21))
STEPS = list(range(1, 9))
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

METHODS = [
    ("Random Forest (DWT)", "individual", "ONLY_DWT_rf"),
    ("Mediana", "flat", "median"),
    ("FFORMA", "flat", "FFORMA"),
    ("ADE", "flat", "ADE"),
]


def parse_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.asarray([float(v) for v in NUMBER_RE.findall(str(value))], dtype=float)


def load_flat(folder: str, dataset: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    path = os.path.join(RESULTS_BASE, folder, f"{dataset}.csv")
    df = pd.read_csv(path, sep=";")
    out = {}
    for _, row in df.iterrows():
        idx = int(row["dataset_index"])
        out[idx] = (parse_array(row["test"]), parse_array(row["predictions"]))
    return out


def load_individual(folder: str, dataset: str) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    path = os.path.join(RESULTS_BASE, folder, "normal", f"{dataset}.csv")
    df = pd.read_csv(path, sep=";")
    df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
    out = {}
    for idx, group in df.groupby("dataset_index"):
        group = group.sort_values("final_test")
        last = group.iloc[-1]
        step0 = np.nan
        if len(group) >= 2:
            previous = group.iloc[-2]
            previous_test = parse_array(previous["test"])
            if previous_test.size:
                step0 = float(previous_test[-1])
        out[int(idx)] = (parse_array(last["test"]), parse_array(last["predictions"]), step0)
    return out


def load_carte(dataset: str) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    out: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for seed in SEEDS:
        path = os.path.join(
            RESULTS_BASE,
            f"orchestrator_react_consist_seed{seed}",
            f"{dataset}.csv",
        )
        df = pd.read_csv(path, sep=";")
        for _, row in df.iterrows():
            idx = int(row["dataset_index"])
            out.setdefault(idx, []).append(
                (parse_array(row["test"]), parse_array(row["predictions"]))
            )
    return out


def pointwise_smape(test: np.ndarray, pred: np.ndarray) -> np.ndarray:
    n = min(test.size, pred.size)
    test, pred = test[:n], pred[:n]
    denom = np.abs(test) + np.abs(pred)
    out = np.zeros(n, dtype=float)
    valid = denom > 0
    out[valid] = 2.0 * np.abs(pred[valid] - test[valid]) / denom[valid]
    return out


def pointwise_pocid(test: np.ndarray, pred: np.ndarray, step0: float) -> np.ndarray:
    n = min(test.size, pred.size)
    test, pred = test[:n], pred[:n]
    out = np.zeros(n, dtype=float)
    prev_test = float(step0)
    prev_pred = float(step0)
    for i in range(n):
        d_real = test[i] - prev_test
        d_pred = pred[i] - prev_pred
        out[i] = 1.0 if d_real * d_pred > 0 else 0.0
        prev_test, prev_pred = test[i], pred[i]
    return out


def step0_from_rf(dataset: str) -> dict[int, float]:
    data = load_individual("rf", dataset)
    return {idx: step0 for idx, (_, _, step0) in data.items()}


def build_rows() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        rf_step0 = step0_from_rf(dataset)

        for label, kind, folder in METHODS:
            if kind == "flat":
                data = load_flat(folder, dataset)
                for idx, (test, pred) in data.items():
                    smape = pointwise_smape(test, pred)
                    pocid = pointwise_pocid(test, pred, rf_step0.get(idx, np.nan))
                    for step in STEPS:
                        if step <= smape.size:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "dataset_index": idx,
                                    "method": label,
                                    "horizon_step": step,
                                    "smape": float(smape[step - 1]),
                                    "pocid": float(pocid[step - 1]),
                                }
                            )
            else:
                data = load_individual(folder, dataset)
                for idx, (test, pred, step0) in data.items():
                    smape = pointwise_smape(test, pred)
                    pocid = pointwise_pocid(test, pred, step0)
                    for step in STEPS:
                        if step <= smape.size:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "dataset_index": idx,
                                    "method": label,
                                    "horizon_step": step,
                                    "smape": float(smape[step - 1]),
                                    "pocid": float(pocid[step - 1]),
                                }
                            )

        carte = load_carte(dataset)
        for idx, executions in carte.items():
            smape_by_seed = []
            pocid_by_seed = []
            for test, pred in executions:
                smape_by_seed.append(pointwise_smape(test, pred))
                pocid_by_seed.append(pointwise_pocid(test, pred, rf_step0.get(idx, np.nan)))
            max_len = min([values.size for values in smape_by_seed], default=0)
            for step in STEPS:
                if step <= max_len:
                    smape_values = [values[step - 1] for values in smape_by_seed if values.size >= step]
                    pocid_values = [values[step - 1] for values in pocid_by_seed if values.size >= step]
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_index": idx,
                            "method": "CARTE",
                            "horizon_step": step,
                            "smape": float(np.nanmean(smape_values)),
                            "pocid": float(np.nanmean(pocid_values)),
                        }
                    )
        print(f"{dataset} concluido", flush=True)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, step), group in rows.groupby(["method", "horizon_step"]):
        for metric in ("smape", "pocid"):
            values = group[metric].dropna().to_numpy(dtype=float)
            n = values.size
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 0 else 0.0
            half = float(t_dist.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
            out.append(
                {
                    "method": method,
                    "horizon_step": step,
                    "metric": metric,
                    "mean": mean * 100.0,
                    "ci_lower": (mean - half) * 100.0,
                    "ci_upper": (mean + half) * 100.0,
                    "std": std * 100.0,
                    "n": n,
                }
            )
    return pd.DataFrame(out)


def plot_metric(summary: pd.DataFrame, metric: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "Random Forest (DWT)": {"color": "#4C72B0", "marker": "o"},
        "Mediana": {"color": "#55A868", "marker": "s"},
        "FFORMA": {"color": "#C44E52", "marker": "^"},
        "ADE": {"color": "#8172B3", "marker": "d"},
        "CARTE": {"color": "black", "marker": "*"},
    }
    order = ["Random Forest (DWT)", "Mediana", "FFORMA", "ADE", "CARTE"]
    ylabel = "SMAPE médio (%)" if metric == "smape" else "POCID médio (%)"

    fig, ax = plt.subplots(figsize=(10, 6.4))
    for method in order:
        sub = summary[(summary["method"] == method) & (summary["metric"] == metric)].sort_values("horizon_step")
        if sub.empty:
            continue
        steps = sub["horizon_step"].to_numpy()
        mean = sub["mean"].to_numpy()
        lower = sub["ci_lower"].to_numpy()
        upper = sub["ci_upper"].to_numpy()
        color = style[method]["color"]
        ax.plot(
            steps,
            mean,
            label=method,
            color=color,
            marker=style[method]["marker"],
            markersize=6,
            linewidth=1.9,
            zorder=5 if method == "CARTE" else 3,
        )
        ax.fill_between(
            steps,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=2 if method == "CARTE" else 1,
        )

    ax.set_xlabel("Passo do horizonte", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(STEPS)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=1.4,
        columnspacing=1.2,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out_path}")


def main() -> None:
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(os.path.dirname(FIG_SMAPE), exist_ok=True)
    rows = build_rows()
    rows.to_csv(os.path.join(OUTPUT_BASE, "horizonte_series.csv"), index=False)
    summary = summarize(rows)
    summary.to_csv(os.path.join(OUTPUT_BASE, "horizonte_resumo.csv"), index=False)
    plot_metric(summary, "smape", FIG_SMAPE)
    plot_metric(summary, "pocid", FIG_POCID)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
