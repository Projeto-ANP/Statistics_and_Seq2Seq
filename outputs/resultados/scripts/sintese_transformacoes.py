"""Sintese -- impacto das transformacoes nas escolhas e nos resultados.

Duas partes:

1. Composicao dos pools finais do CARTE nas dez execucoes (sementes 11-20):
   quantos pools usam variantes transformadas, quanto dos lugares elas ocupam
   e quanto peso recebem nas estrategias ponderadas.

2. Contrafactual sem transformacoes. Mantem a estrategia final de cada serie
   (mesma regra, mesmo trim, mesmos pesos renormalizados) e remove do pool as
   doze variantes transformadas. Quando o pool fica vazio, usa o pool completo
   dos sete modelos nao transformados como fallback. Recalcula a previsao de
   teste e compara SMAPE/POCID com o CARTE original.

A reconstrucao da previsao original e conferida contra a coluna `predictions`
dos CSVs do CARTE. Rodar no ambiente `agno` para que o DBA use o tslearn:

    conda run -n agno python outputs/resultados/scripts/sintese_transformacoes.py

Saidas:
    outputs/resultados/sintese_transformacoes/composicao_pools.csv
    outputs/resultados/sintese_transformacoes/contrafactual_series.csv
    outputs/resultados/sintese_transformacoes/contrafactual_resumo_series.csv
    outputs/resultados/sintese_transformacoes/contrafactual_resumo.csv
    overleaf/resultados_final/sintese_transformacoes_delta.png
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from orchestrator_react.combiners import apply_combination  # noqa: E402
from orchestrator_react.metrics import pocid, smape  # noqa: E402

RESULTS_BASE = os.path.join(REPO_ROOT, "timeseries", "mestrado", "resultados")
OUTPUT_BASE = os.path.join(REPO_ROOT, "outputs", "resultados", "sintese_transformacoes")
FIGURE_PATH = os.path.join(REPO_ROOT, "overleaf", "resultados_final", "sintese_transformacoes_delta.png")

SEEDS = list(range(11, 21))
DATASETS = [
    "ANP_MONTHLY",
    "NN5_WEEKLY_DATASET",
    "M4_WEEKLY_DATASET",
    "ETTH1",
    "ETTH2",
    "ETTM1",
    "ETTM2",
]
DATASET_LABELS = {
    "ANP_MONTHLY": "ANP",
    "NN5_WEEKLY_DATASET": "NN5",
    "M4_WEEKLY_DATASET": "M4",
    "ETTH1": "ETTh1",
    "ETTH2": "ETTh2",
    "ETTM1": "ETTm1",
    "ETTM2": "ETTm2",
}
MODELS = [
    "ARIMA", "ETS", "THETA", "rf", "catboost",
    "CWT_rf", "DWT_rf", "FT_rf",
    "CWT_catboost", "DWT_catboost", "FT_catboost",
    "ONLY_CWT_rf", "ONLY_DWT_rf", "ONLY_FT_rf",
    "ONLY_CWT_catboost", "ONLY_DWT_catboost", "ONLY_FT_catboost",
    "NaiveSeasonal", "NaiveMovingAverage",
]
TRANSFORMED = {
    "CWT_rf", "DWT_rf", "FT_rf",
    "CWT_catboost", "DWT_catboost", "FT_catboost",
    "ONLY_CWT_rf", "ONLY_DWT_rf", "ONLY_FT_rf",
    "ONLY_CWT_catboost", "ONLY_DWT_catboost", "ONLY_FT_catboost",
}
ONLY = {m for m in MODELS if m.startswith("ONLY_")}
BASE_ML = {"rf", "catboost"}
STATISTICAL = {"ARIMA", "ETS", "THETA"}
NAIVE = {"NaiveSeasonal", "NaiveMovingAverage"}
BASE7 = ["ARIMA", "ETS", "THETA", "rf", "catboost", "NaiveSeasonal", "NaiveMovingAverage"]

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.asarray([float(v) for v in NUMBER_RE.findall(str(value))], dtype=float)


def load_predictions(dataset: str) -> dict[str, dict[int, np.ndarray]]:
    """Last window (test) of each model for each dataset_index."""
    cache: dict[str, dict[int, np.ndarray]] = {}
    for model in MODELS:
        path = os.path.join(RESULTS_BASE, model, "normal", f"{dataset}.csv")
        frame = pd.read_csv(path, sep=";")
        frame["start_test"] = pd.to_datetime(frame["start_test"], errors="coerce")
        per_series: dict[int, np.ndarray] = {}
        for index, group in frame.groupby("dataset_index"):
            group = group.sort_values("start_test")
            per_series[int(index)] = parse_array(group.iloc[-1]["predictions"])
        cache[model] = per_series
    return cache


def weights_for(row: pd.Series, pool: list[str]) -> tuple[np.ndarray, bool]:
    resolved = json.loads(row["weights_handle_resolved"])
    weights = resolved["weights"]
    if resolved.get("per_horizon"):
        horizons = sorted(weights.keys(), key=lambda value: int(value))
        matrix = np.array(
            [[weights[h].get(model, 0.0) for h in horizons] for model in pool]
        )
        return matrix, True
    vector = np.array([weights.get(model, 0.0) for model in pool])
    return vector, False


def combine(
    predictions: dict[str, dict[int, np.ndarray]],
    row: pd.Series,
    dataset_index: int,
    pool: list[str],
    method: str,
    weights: tuple[np.ndarray, bool] | None = None,
) -> np.ndarray:
    kwargs = {}
    if method == "weighted":
        if weights is None:
            weights = weights_for(row, pool)
        kwargs["weights"] = weights[0]
    elif method == "trimmed_mean":
        spec = json.loads(row["best_strategy_params"])
        kwargs["trim_pct"] = spec.get("trim_pct", 0.2)
    elif method == "best_single":
        spec = json.loads(row["best_strategy_params"])
        kwargs["model_pos"] = spec.get("model_pos", 0)
    matrix = np.vstack([predictions[model][dataset_index] for model in pool])
    return apply_combination(matrix, method, **kwargs)


def run_seed(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counterfactual rows and pool-composition rows for one seed."""
    cf_rows: list[dict] = []
    composition_rows: list[dict] = []
    for dataset in DATASETS:
        predictions = load_predictions(dataset)
        orchestrator = pd.read_csv(
            os.path.join(RESULTS_BASE, f"orchestrator_react_consist_seed{seed}", f"{dataset}.csv"),
            sep=";",
        )
        for _, row in orchestrator.iterrows():
            dataset_index = int(row["dataset_index"])
            selected = json.loads(row["selected_base_models"])
            method = row["best_strategy_method"]
            actual = parse_array(row["test"])

            original = combine(predictions, row, dataset_index, selected, method)
            max_diff = float(np.nanmax(np.abs(original - parse_array(row["predictions"]))))
            original_smape = smape(actual, original)
            original_pocid = pocid(actual, original)

            remaining = [model for model in selected if model not in TRANSFORMED]
            fallback = not remaining
            pool = remaining if remaining else BASE7
            if method == "weighted":
                if fallback:
                    weights = (np.ones(len(pool), dtype=float) / len(pool), False)
                else:
                    weights = weights_for(row, pool)
            else:
                weights = None
            counterfactual = combine(
                predictions, row, dataset_index, pool, method, weights=weights
            )

            counterfactual_smape = smape(actual, counterfactual)
            counterfactual_pocid = pocid(actual, counterfactual)
            cf_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "method": method,
                    "n_pool": len(selected),
                    "n_remaining": len(remaining),
                    "fallback": fallback,
                    "maxdiff_reconstruction": max_diff,
                    "smape_original": original_smape,
                    "pocid_original": original_pocid,
                    "smape_sem_transformacoes": counterfactual_smape,
                    "pocid_sem_transformacoes": counterfactual_pocid,
                    "smape_sem_transformacoes_sem_fallback": (
                        counterfactual_smape if remaining else np.nan
                    ),
                    "pocid_sem_transformacoes_sem_fallback": (
                        counterfactual_pocid if remaining else np.nan
                    ),
                }
            )

            n_models = len(selected)
            n_transformed = sum(model in TRANSFORMED for model in selected)
            n_only = sum(model in ONLY for model in selected)
            n_base = sum(model in BASE_ML for model in selected)
            n_stat = sum(model in STATISTICAL for model in selected)
            n_naive = sum(model in NAIVE for model in selected)
            weights_equal = np.ones(n_models, dtype=float) / n_models
            if method == "weighted":
                vector, per_horizon = weights_for(row, selected)
                if per_horizon:
                    weights_equal = vector.mean(axis=1)
                else:
                    weights_equal = vector
                total = weights_equal.sum()
                if total > 0:
                    weights_equal = weights_equal / total
            composition_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "method": method,
                    "n_pool": n_models,
                    "n_transformed": n_transformed,
                    "n_only": n_only,
                    "n_base_ml": n_base,
                    "n_statistical": n_stat,
                    "n_naive": n_naive,
                    "weight_transformed": float(
                        sum(weights_equal[i] for i, model in enumerate(selected) if model in TRANSFORMED)
                    ),
                    "weight_only": float(
                        sum(weights_equal[i] for i, model in enumerate(selected) if model in ONLY)
                    ),
                }
            )
        print(f"seed {seed} {dataset} concluido", flush=True)
    return pd.DataFrame(cf_rows), pd.DataFrame(composition_rows)


def summarize_counterfactual(series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_series = (
        series.groupby(["dataset", "dataset_index"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            smape_original=("smape_original", "mean"),
            pocid_original=("pocid_original", "mean"),
            smape_sem_transformacoes=("smape_sem_transformacoes", "mean"),
            pocid_sem_transformacoes=("pocid_sem_transformacoes", "mean"),
            smape_sem_fallback=("smape_sem_transformacoes_sem_fallback", "mean"),
            pocid_sem_fallback=("pocid_sem_transformacoes_sem_fallback", "mean"),
            n_seeds_sem_fallback=("smape_sem_transformacoes_sem_fallback", "count"),
            n_pool=("n_pool", "mean"),
            fallback_share=("fallback", "mean"),
        )
    )
    rows = []
    for dataset, group in per_series.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "n_series": len(group),
                "smape_original": group["smape_original"].mean(),
                "smape_sem_transformacoes": group["smape_sem_transformacoes"].mean(),
                "delta_smape": (
                    group["smape_sem_transformacoes"].mean() - group["smape_original"].mean()
                ),
                "pocid_original": group["pocid_original"].mean(),
                "pocid_sem_transformacoes": group["pocid_sem_transformacoes"].mean(),
                "delta_pocid": (
                    group["pocid_sem_transformacoes"].mean() - group["pocid_original"].mean()
                ),
                "smape_sem_fallback": group["smape_sem_fallback"].mean(),
                "pocid_sem_fallback": group["pocid_sem_fallback"].mean(),
                "n_series_sem_fallback": group["smape_sem_fallback"].notna().sum(),
                "fallback_share": group["fallback_share"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    overall = {
        "dataset": "Total 680",
        "n_series": len(per_series),
        "smape_original": per_series["smape_original"].mean(),
        "smape_sem_transformacoes": per_series["smape_sem_transformacoes"].mean(),
        "delta_smape": (
            per_series["smape_sem_transformacoes"].mean() - per_series["smape_original"].mean()
        ),
        "pocid_original": per_series["pocid_original"].mean(),
        "pocid_sem_transformacoes": per_series["pocid_sem_transformacoes"].mean(),
        "delta_pocid": (
            per_series["pocid_sem_transformacoes"].mean() - per_series["pocid_original"].mean()
        ),
        "smape_sem_fallback": per_series["smape_sem_fallback"].mean(),
        "pocid_sem_fallback": per_series["pocid_sem_fallback"].mean(),
        "n_series_sem_fallback": per_series["smape_sem_fallback"].notna().sum(),
        "fallback_share": per_series["fallback_share"].mean(),
    }
    mean7 = {
        "dataset": "Media 7",
        "n_series": 7,
        "smape_original": summary["smape_original"].mean(),
        "smape_sem_transformacoes": summary["smape_sem_transformacoes"].mean(),
        "delta_smape": summary["delta_smape"].mean(),
        "pocid_original": summary["pocid_original"].mean(),
        "pocid_sem_transformacoes": summary["pocid_sem_transformacoes"].mean(),
        "delta_pocid": summary["delta_pocid"].mean(),
        "smape_sem_fallback": summary["smape_sem_fallback"].mean(),
        "pocid_sem_fallback": summary["pocid_sem_fallback"].mean(),
        "n_series_sem_fallback": summary["n_series_sem_fallback"].sum(),
        "fallback_share": summary["fallback_share"].mean(),
    }
    summary = pd.concat([summary, pd.DataFrame([overall, mean7])], ignore_index=True)
    return per_series, summary


def summarize_composition(composition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in composition.groupby("dataset"):
        weighted = group[group["method"] == "weighted"]
        rows.append(
            {
                "dataset": dataset,
                "n_series_seeds": len(group),
                "pool_mean": group["n_pool"].mean(),
                "pools_with_transformed": (group["n_transformed"] > 0).mean(),
                "slots_transformed": group["n_transformed"].sum() / group["n_pool"].sum(),
                "slots_only": group["n_only"].sum() / group["n_pool"].sum(),
                "pools_without_base_ml": (group["n_base_ml"] == 0).mean(),
                "weighted_share": len(weighted) / len(group),
                "weight_transformed_weighted": weighted["weight_transformed"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    weighted = composition[composition["method"] == "weighted"]
    overall = {
        "dataset": "Total",
        "n_series_seeds": len(composition),
        "pool_mean": composition["n_pool"].mean(),
        "pools_with_transformed": (composition["n_transformed"] > 0).mean(),
        "slots_transformed": composition["n_transformed"].sum() / composition["n_pool"].sum(),
        "slots_only": composition["n_only"].sum() / composition["n_pool"].sum(),
        "pools_without_base_ml": (composition["n_base_ml"] == 0).mean(),
        "weighted_share": len(weighted) / len(composition),
        "weight_transformed_weighted": weighted["weight_transformed"].mean(),
    }
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


def plot_boxplot(per_series: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = DATASETS
    # A figura usa a comparação sem fallback, restrita às séries em que o pool
    # final ainda tem ao menos um modelo não transformado.
    smape_delta = [
        (per_series.loc[per_series["dataset"] == dataset, "smape_sem_fallback"]
         - per_series.loc[per_series["dataset"] == dataset, "smape_original"]).to_numpy() * 100
        for dataset in order
    ]
    pocid_delta = [
        (per_series.loc[per_series["dataset"] == dataset, "pocid_sem_fallback"]
         - per_series.loc[per_series["dataset"] == dataset, "pocid_original"]).to_numpy()
        for dataset in order
    ]
    smape_delta = [values[np.isfinite(values)] for values in smape_delta]
    pocid_delta = [values[np.isfinite(values)] for values in pocid_delta]
    labels = [DATASET_LABELS[dataset] for dataset in order]

    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 7.6), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    box_color = "#4C72B0"
    for ax, data, ylabel in (
        (axes[0], smape_delta, "Δ SMAPE (pontos percentuais)"),
        (axes[1], pocid_delta, "Δ POCID (pontos percentuais)"),
    ):
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "#333333", "linewidth": 1.0},
            capprops={"color": "#333333", "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "edgecolor": "#333333"},
            flierprops={
                "marker": "o",
                "markersize": 3.0,
                "markerfacecolor": "#777777",
                "markeredgecolor": "#777777",
                "alpha": 0.45,
            },
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(box_color)
            patch.set_alpha(0.75)
        means = [float(np.mean(values)) for values in data]
        ax.scatter(
            range(1, len(data) + 1),
            means,
            marker="D",
            s=18,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.axhline(0.0, color="black", linewidth=0.9, zorder=0)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=11)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylim(-15, 55)
    axes[0].set_yticks([-10, 0, 10, 20, 30, 40, 50])
    axes[1].set_ylim(-50, 35)
    axes[1].set_yticks([-40, -20, 0, 20])

    fig.align_ylabels(axes)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_bars(per_series: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = DATASETS
    labels = [DATASET_LABELS[dataset] for dataset in order]
    smape = []
    pocid = []
    for dataset in order:
        subset = per_series.loc[per_series["dataset"] == dataset]
        smape.append(
            float(np.nanmean((subset["smape_sem_fallback"] - subset["smape_original"]).to_numpy() * 100))
        )
        pocid.append(
            float(np.nanmean((subset["pocid_sem_fallback"] - subset["pocid_original"]).to_numpy()))
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"wspace": 0.32})
    positions = np.arange(len(order))
    colors_smape = "#4C72B0"
    colors_pocid = ["#C44E52" if value < 0 else "#4C72B0" for value in pocid]

    axes[0].barh(positions, smape, color=colors_smape, height=0.62)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="black", linewidth=0.9)
    axes[0].set_xlabel("Δ SMAPE médio (pontos percentuais)", fontsize=11)
    axes[0].set_xlim(0, max(smape) * 1.18)
    for y, value in zip(positions, smape):
        axes[0].text(
            value + max(smape) * 0.02,
            y,
            f"{value:+.2f}",
            va="center",
            fontsize=9,
        )

    axes[1].barh(positions, pocid, color=colors_pocid, height=0.62)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.9)
    axes[1].set_xlabel("Δ POCID médio (pontos percentuais)", fontsize=11)
    limit = max(abs(min(pocid)), abs(max(pocid))) * 1.28
    axes[1].set_xlim(-limit, limit)
    for y, value in zip(positions, pocid):
        offset = limit * 0.03
        axes[1].text(
            value + (offset if value >= 0 else -offset),
            y,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    for ax in axes:
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=10)

    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)

    cf_frames = []
    composition_frames = []
    for seed in SEEDS:
        cf, composition = run_seed(seed)
        cf_frames.append(cf)
        composition_frames.append(composition)

    series = pd.concat(cf_frames, ignore_index=True)
    composition = pd.concat(composition_frames, ignore_index=True)

    series.to_csv(os.path.join(OUTPUT_BASE, "contrafactual_series.csv"), index=False)
    composition.to_csv(os.path.join(OUTPUT_BASE, "composicao_pools.csv"), index=False)

    per_series, summary = summarize_counterfactual(series)
    per_series.to_csv(
        os.path.join(OUTPUT_BASE, "contrafactual_resumo_series.csv"), index=False
    )
    summary.to_csv(os.path.join(OUTPUT_BASE, "contrafactual_resumo.csv"), index=False)
    summarize_composition(composition).to_csv(
        os.path.join(OUTPUT_BASE, "composicao_resumo.csv"), index=False
    )

    plot_delta_bars(per_series)
    print(summary.to_string(index=False))
    print(f"figura salva em {FIGURE_PATH}")


if __name__ == "__main__":
    main()
