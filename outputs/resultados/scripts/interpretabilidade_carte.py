"""Casos de interpretabilidade do CARTE na execucao representativa.

Gera as figuras de sucesso e fracasso usadas na subsecao
`Interpretabilidade das previsoes`:

    overleaf/resultados_final/caso_sucesso.png
    overleaf/resultados_final/caso_fracasso.png

Os casos sao os mesmos da versao anterior do texto, mas com a estrategia
final da execucao representativa (pasta
`orchestrator_react_consist_seed11`), nao com a execucao antiga.
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RESULTS_BASE = os.path.join(REPO_ROOT, "timeseries", "mestrado", "resultados")
RUN = "orchestrator_react_consist_seed11"
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "resultados", "interpretabilidade_carte")
FIG_DIR = os.path.join(REPO_ROOT, "overleaf", "resultados_final")

CASES = {
    "caso_sucesso": ("M4_WEEKLY_DATASET", 206),
    "caso_fracasso": ("ANP_MONTHLY", 13),
}
BASELINES = {
    "ADE": "ADE",
    "FFORMA": "FFORMA",
    "Mediana": "median",
    "DBA": "dba",
}
COMBINACOES_STYLE = {
    "ADE": {"color": "#8172B3", "marker": "d", "markersize": 7, "linewidth": 2.0},
    "FFORMA": {"color": "#C44E52", "marker": "^", "markersize": 7, "linewidth": 2.0},
    "Mediana": {"color": "#55A868", "marker": "s", "markersize": 6.5, "linewidth": 2.0},
    "DBA": {"color": "#DD8452", "marker": "v", "markersize": 7, "linewidth": 2.0},
}
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.asarray([float(v) for v in NUMBER_RE.findall(str(value))], dtype=float)


def load_model_predictions(model: str, dataset: str, dataset_index: int) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(RESULTS_BASE, model, "normal", f"{dataset}.csv")
    df = pd.read_csv(path, sep=";")
    df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce")
    sub = df[df["dataset_index"] == int(dataset_index)].sort_values("final_test")
    if sub.empty:
        raise ValueError(f"model {model} has no rows for {dataset} idx={dataset_index}")
    row = sub.iloc[-1]
    return parse_array(row["test"]), parse_array(row["predictions"])


def load_baseline(folder: str, dataset: str, dataset_index: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    path = os.path.join(RESULTS_BASE, folder, f"{dataset}.csv")
    df = pd.read_csv(path, sep=";")
    sub = df[df["dataset_index"] == int(dataset_index)]
    if sub.empty:
        raise ValueError(f"baseline {folder} has no row for {dataset} idx={dataset_index}")
    row = sub.iloc[0]
    return (
        parse_array(row["test"]),
        parse_array(row["predictions"]),
        float(row["smape"]) * 100.0,
        float(row["pocid"]),
    )


def extract_case(dataset: str, dataset_index: int) -> dict:
    path = os.path.join(RESULTS_BASE, RUN, f"{dataset}.csv")
    df = pd.read_csv(path, sep=";")
    row = df[df["dataset_index"] == int(dataset_index)].iloc[0]
    test = parse_array(row["test"])
    final_pred = parse_array(row["predictions"])
    selected = json.loads(row["selected_base_models"])
    params = json.loads(row["best_strategy_params"])

    individuais = {}
    for model in selected:
        model_test, pred = load_model_predictions(model, dataset, dataset_index)
        n = min(model_test.size, test.size)
        if not np.allclose(model_test[:n], test[:n], equal_nan=True):
            raise ValueError(f"model {model} has a different test window")
        individuais[model] = pred

    combinacoes = {}
    for label, folder in BASELINES.items():
        base_test, pred, smape, pocid = load_baseline(folder, dataset, dataset_index)
        n = min(base_test.size, test.size)
        if not np.allclose(base_test[:n], test[:n], equal_nan=True):
            raise ValueError(f"baseline {label} has a different test window")
        combinacoes[label] = {"predictions": pred, "smape": smape, "pocid": pocid}

    return {
        "dataset": dataset,
        "dataset_index": int(dataset_index),
        "test": test,
        "modelos_individuais_predictions": individuais,
        "outras_combinacoes": combinacoes,
        "predicao_combinada_final": final_pred,
        "metodo_agregacao": row["best_strategy_method"],
        "estrategia": row["best_strategy_name"],
        "estrategia_params": params,
        "smape_final": float(row["smape"]) * 100.0,
        "pocid_final": float(row["pocid"]),
        "justificativa_final": row["justificativa_final"],
    }


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def plot_case(case: dict, out_path: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.5, 9.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    steps = list(range(1, len(case["test"]) + 1))

    def draw(ax, individuais=False):
        if individuais:
            first = True
            for pred in case["modelos_individuais_predictions"].values():
                ax.plot(
                    steps,
                    pred,
                    color="#BBBBBB",
                    lw=0.9,
                    zorder=2,
                    label="Individuais" if first else None,
                )
                first = False
        ax.plot(
            steps,
            case["test"],
            color="#4C72B0",
            lw=2.6,
            marker="o",
            ms=6.0,
            zorder=4,
            label="Valor real",
        )
        for name, data in case["outras_combinacoes"].items():
            ax.plot(
                steps,
                data["predictions"],
                zorder=3,
                label=name,
                **COMBINACOES_STYLE[name],
            )
        ax.plot(
            steps,
            case["predicao_combinada_final"],
            color="black",
            lw=3.0,
            zorder=5,
            label="CARTE",
        )
        ax.set_ylabel("Valor")
        ax.set_xticks(steps)
        ax.grid(alpha=0.25, lw=0.6)

    draw(ax1, individuais=True)
    draw(ax2)
    ax1.set_title("Escala completa", loc="left", fontsize=18)
    ax2.set_title("Detalhe das estratégias de combinação", loc="left", fontsize=18)
    ax2.set_xlabel("Passo do horizonte")

    cinco = [case["predicao_combinada_final"]] + [
        d["predictions"] for d in case["outras_combinacoes"].values()
    ]
    lo = min(float(np.min(v)) for v in cinco)
    hi = max(float(np.max(v)) for v in cinco)
    margem = 0.06 * (hi - lo) if hi > lo else 1.0
    ax2.set_ylim(lo - margem, hi + margem)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out_path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    for name, (dataset, dataset_index) in CASES.items():
        case = extract_case(dataset, dataset_index)
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(to_serializable(case), fh, ensure_ascii=False, indent=2)
        plot_case(case, os.path.join(FIG_DIR, f"{name}.png"))
        print(
            f"{name}: {dataset} idx={dataset_index} "
            f"smape={case['smape_final']:.2f} pocid={case['pocid_final']:.2f} "
            f"estrategia={case['estrategia']} n_modelos={len(case['modelos_individuais_predictions'])}"
        )


if __name__ == "__main__":
    main()
