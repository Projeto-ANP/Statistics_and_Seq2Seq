#!/usr/bin/env python3
"""Figuras de erro por passo do horizonte e de contagem de vitorias.

Le os CSVs ja calculados com a configuracao final e gera:

    outputs/resultados/item_09/figuras/erro_horizonte_smape.png   (2 paineis)
    outputs/resultados/item_09/figuras/erro_horizonte_pocid.png   (2 paineis)
    outputs/resultados/item_09/figuras/erro_por_horizonte_8.png   (media, 8 passos)
    outputs/resultados/item_09/figuras/erro_por_horizonte.png     (horizonte completo)
    outputs/resultados/item_09/figuras/vitorias_por_metodo.png    (contagem de vitorias)

Uso:
    conda activate agno
    python3 outputs/resultados/scripts/figuras_horizonte.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HORIZON_CSV = os.path.join(C.OUTPUTS_BASE, "item_09", "erro_por_horizonte_v2.csv")
WINS_CSV = os.path.join(C.OUTPUTS_BASE, "item_08", "contagem_vitorias.csv")
OUT_DIR = os.path.join(C.OUTPUTS_BASE, "item_09", "figuras")

DISPLAY = {
    "ADE": "ADE",
    "CREST": "CREST",
    "FFORMA": "FFORMA",
    "melhor_estatica": "Mediana",
    "melhor_individual": "Random Forest (DWT)",
}
ORDER = ["ADE", "CREST", "FFORMA", "Mediana", "Random Forest (DWT)"]
STYLE = {
    "ADE": {"color": "#8172B3", "marker": "d", "markersize": 7, "linewidth": 1.8},
    "CREST": {"color": "black", "marker": "*", "markersize": 12, "linewidth": 2.2},
    "FFORMA": {"color": "#C44E52", "marker": "^", "markersize": 8, "linewidth": 1.8},
    "Mediana": {"color": "#55A868", "marker": "s", "markersize": 7, "linewidth": 1.8},
    "Random Forest (DWT)": {"color": "#4C72B0", "marker": "o", "markersize": 7, "linewidth": 1.8},
}
METRICS = {
    "smape": {"mean": "smape_medio", "std": "smape_desvio_padrao", "ylabel": "SMAPE médio (%)"},
    "pocid": {"mean": "pocid_medio", "std": "pocid_desvio_padrao", "ylabel": "POCID médio (%)"},
}


def load_horizon():
    df = pd.read_csv(HORIZON_CSV)
    df["label"] = df["method"].map(DISPLAY).fillna(df["method"])
    return df


def series_by_label(df, column, steps):
    out = {}
    for label in ORDER:
        sub = df[(df.label == label) & (df.horizon_step.isin(steps))].sort_values("horizon_step")
        out[label] = (sub.horizon_step.to_numpy(), sub[column].to_numpy() * 100.0)
    return out


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=17)


def plot_two_panel(df, metric, out_path, steps=range(1, 9)):
    info = METRICS[metric]
    steps = list(steps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for label, (x, y) in series_by_label(df, info["mean"], steps).items():
        ax1.plot(x, y, label=label, **STYLE[label])
    for label, (x, y) in series_by_label(df, info["std"], steps).items():
        ax2.plot(x, y, label=label, **STYLE[label])
    ax1.set_title("Média", loc="left", fontsize=20)
    ax2.set_title("Desvio-padrão", loc="left", fontsize=20)
    ax1.set_ylabel(info["ylabel"], fontsize=20)
    ax2.set_ylabel("Desvio-padrão (p.p.)", fontsize=20)
    ax2.set_xlabel("Passo do horizonte", fontsize=20)
    ax1.set_xticks(steps)
    style_axes(ax1)
    style_axes(ax2)
    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=5, frameon=False, fontsize=17, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", out_path)


def plot_mean_only(df, metric, out_path, steps=range(1, 9)):
    info = METRICS[metric]
    steps = list(steps)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for label, (x, y) in series_by_label(df, info["mean"], steps).items():
        ax.plot(x, y, label=label, **STYLE[label])
    ax.set_ylabel(info["ylabel"], fontsize=20)
    ax.set_xlabel("Passo do horizonte", fontsize=20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False,
              fontsize=17, handlelength=1.6)
    ax.set_xticks(steps)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", out_path)


def plot_full_horizon(df, metric, out_path):
    info = METRICS[metric]
    steps = sorted(df.horizon_step.unique())
    counts = df[df.label == "CREST"].set_index("horizon_step")["n_series"]
    fig, ax = plt.subplots(figsize=(12, 8))
    # faixas de populacao constante
    bands, start = [], steps[0]
    for i in range(1, len(steps) + 1):
        if i == len(steps) or counts[steps[i]] != counts[steps[i - 1]]:
            bands.append((start, steps[i - 1], int(counts[steps[i - 1]])))
            if i < len(steps):
                start = steps[i]
    n_datasets = {680: 7, 569: 6, 387: 5, 28: 4}
    for k, (a, b, n) in enumerate(bands):
        if k % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color="0.92", zorder=0)
        ax.axvline(b + 0.5, color="white", linewidth=2, zorder=1)
        nds = n_datasets.get(n)
        label = "n=680\n(7 datasets)" if nds == 7 else ("n=28\n(só ETT)" if n == 28 else f"n={n}\n({nds} datasets)")
        y_band = 0.97 if k % 2 == 0 else 0.86
        ax.text((a + b) / 2, y_band, label, ha="center", va="top", fontsize=13,
                color="0.35", transform=ax.get_xaxis_transform())
    for label, (x, y) in series_by_label(df, info["mean"], steps).items():
        ax.plot(x, y, label=label, zorder=3, **STYLE[label])
    ax.set_ylabel(info["ylabel"], fontsize=20)
    ax.set_xlabel("Passo do horizonte", fontsize=20)
    ax.set_xticks(list(range(1, max(steps) + 1, 2)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False,
              fontsize=17, handlelength=1.6)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", out_path)


def plot_wins(out_path):
    df = pd.read_csv(WINS_CSV).sort_values("n_vitorias", ascending=False)
    labels = [DISPLAY.get(m, m) for m in df.method]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels[::-1], df.n_vitorias[::-1],
                   color=[STYLE[l]["color"] for l in labels[::-1]])
    for bar, n, pct in zip(bars, df.n_vitorias[::-1], df.pct_vitorias[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{int(n)} ({pct:.1f}%)", va="center", fontsize=16)
    ax.set_xlabel("Séries com o menor SMAPE entre as 680", fontsize=20)
    ax.set_xlim(0, df.n_vitorias.max() * 1.18)
    style_axes(ax)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print("salvo:", out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_horizon()
    plot_two_panel(df, "smape", os.path.join(OUT_DIR, "erro_horizonte_smape.png"))
    plot_two_panel(df, "pocid", os.path.join(OUT_DIR, "erro_horizonte_pocid.png"))
    plot_mean_only(df, "smape", os.path.join(OUT_DIR, "erro_por_horizonte_8.png"))
    plot_full_horizon(df, "smape", os.path.join(OUT_DIR, "erro_por_horizonte.png"))
    plot_wins(os.path.join(OUT_DIR, "vitorias_por_metodo.png"))


if __name__ == "__main__":
    main()
