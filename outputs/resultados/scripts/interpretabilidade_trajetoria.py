#!/usr/bin/env python3
"""Figura do estudo de caso ilustrativo (Subseção 2.5.3).

Previsoes dos dois casos de trajetoria da ANP (series 5 e 8): os dezenove
preditores individuais, o valor real, a combinacao final do CREST e a decisao
da versao sem o agente (melhor semente). Na serie 5, a estrategia proposta
pelo agente difere do piso; na serie 8, as duas coincidem.

Saida: outputs/resultados/interpretabilidade/figuras/caso_trajetoria.png
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CASES = [
    ("ANP_MONTHLY", 5, "Série 5 da ANP: estratégia proposta pelo agente"),
    ("ANP_MONTHLY", 8, "Série 8 da ANP: estratégia semeada aceita"),
]


def dados(dataset: str, idx: int) -> dict:
    gpt = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
    row = gpt[gpt.dataset_index == idx].iloc[0]
    base = C.load_orchestrator(C.ORCHESTRATOR_BASELINE, dataset)
    brow = base[base.dataset_index == idx].iloc[0]

    individuais = {}
    for modelo in C.INDIVIDUAL_MODELS:
        mdf = C.load_filtered(modelo, dataset)
        mr = mdf[mdf.dataset_index == idx].iloc[0]
        individuais[modelo] = C.parse_num_list(mr["predictions"])

    return {
        "test": C.parse_num_list(row["test"]),
        "comb": C.parse_num_list(row["predictions"]),
        "base": C.parse_num_list(brow["predictions"]),
        "individuais": individuais,
        "smape": float(row["smape"]) * 100,
        "smape_base": float(brow["smape"]) * 100,
    }


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 19,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
    })

    casos = [dados(ds, idx) for ds, idx, _ in CASES]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 10), sharex=True)
    for ax, (ds, idx, titulo), d in zip(axes, CASES, casos):
        steps = list(range(1, len(d["test"]) + 1))

        primeiro = True
        for pred in d["individuais"].values():
            ax.plot(steps, pred, color="#BBBBBB", lw=1.0, zorder=2,
                    label="Modelos individuais" if primeiro else None)
            primeiro = False

        ax.plot(steps, d["test"], color="#4C72B0", lw=2.6, marker="o", ms=6.5,
                zorder=4, label="Valor real")
        ax.plot(steps, d["base"], color="#DD8452", lw=2.4, ls="--", zorder=3,
                label="Sem agente")
        ax.plot(steps, d["comb"], color="black", lw=3.0, zorder=5,
                label="CREST (combinação final)")

        ax.set_title(titulo, loc="left", fontsize=20)
        ax.set_ylabel("Valor")
        ax.set_xticks(steps)
        ax.grid(alpha=0.25, lw=0.6)

    axes[-1].set_xlabel("Passo do horizonte")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05),
               ncol=4, frameon=False, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = os.path.join(C.OUTPUTS_BASE, "interpretabilidade", "figuras")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "caso_trajetoria.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for (ds, idx, _), d in zip(CASES, casos):
        print(f"{ds} {idx}: smape={d['smape']:.2f} sem_agente={d['smape_base']:.2f}")
    print("salvo:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
