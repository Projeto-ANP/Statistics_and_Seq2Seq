"""Interpretabilidade -- casos de sucesso e fracasso do CREST (configuracao final).

Diferente do item_15 (foco na trajetoria de raciocinio), aqui o foco e nos
NUMEROS da previsao: serie real, previsao de cada modelo individual que
compos a estrategia final vencedora, previsao combinada, metodo/pesos. A
figura tem dois paineis: a escala completa (com os modelos individuais em
cinza) e a ampliacao na faixa das estrategias de combinacao (sem eles, para o
detalhe entre os metodos ficar visivel).

Selecao dos candidatos: entre as 47 series cuja estrategia final tem origem
no proprio agente na configuracao final (as series do texto antigo, ANP 103
e NN5 109, terminaram com estrategia de origem semeada e nao servem mais),
exigindo que a estrategia combine multiplos preditores -- nao um unico
modelo isolado:
- Sucesso: M4_WEEKLY_DATASET, dataset_index=206. Maior vantagem do CREST
  sobre as demais combinacoes entre as 47 series de origem no agente
  (SMAPE=0.1037, contra 0.1390 do segundo melhor), com a serie real sem
  oscilacoes extremas e estrategia "weighted" sobre 7 modelos.
- Fracasso: ANP_MONTHLY, dataset_index=13. Serie com queda e recuperacao
  claras que a combinacao nao acompanha (SMAPE=0.2155), com as cinco
  estrategias de combinacao errando na mesma faixa (0.2117 a 0.2227),
  estrategia "weighted" sobre 9 modelos; evita de proposito as series com
  oscilacoes extremas (como a antiga ANP 64, de 13,8 a 107,3) e as duas
  series de M4_WEEKLY_DATASET (idx 208 e 75) entre as piores do CREST,
  porque essas duas coincidem com a patologia de alvo de treino corrompido
  ja documentada em outputs/diagnostico/RESUMO_passo7_rf_m4.md (um bug de
  normalizacao de janela, no upstream dos modelos individuais, nao do
  metodo de combinacao) -- usa-las aqui confundiria "a combinacao nao
  ajudou" com "o dado de entrada de alguns modelos individuais ja estava
  corrompido antes de qualquer combinacao".

Saida:
  outputs/resultados/interpretabilidade/caso_sucesso.json
  outputs/resultados/interpretabilidade/caso_fracasso.json
  outputs/resultados/interpretabilidade/figuras/caso_sucesso.png
  outputs/resultados/interpretabilidade/figuras/caso_fracasso.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CASES = {
    "caso_sucesso": ("M4_WEEKLY_DATASET", 206),
    "caso_fracasso": ("ANP_MONTHLY", 13),
}

OUTRAS_COMBINACOES = {
    "ADE": "ADE",
    "FFORMA": "FFORMA",
    "median": "Mediana",
    "dba": "DBA",
}


def extract_case(dataset: str, dataset_index: int) -> dict:
    df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
    row = df[df["dataset_index"] == dataset_index].iloc[0]

    test = C.parse_num_list(row["test"])
    final_pred = C.parse_num_list(row["predictions"])
    effective_models = json.loads(row["effective_models"])
    best_strategy_params = json.loads(row["best_strategy_params"])
    weights_by_horizon = json.loads(row["weights_by_horizon"]) if C.pd.notna(row["weights_by_horizon"]) else None

    modelos_individuais = {}
    for model in effective_models:
        mdf = C.load_filtered(model, dataset)
        mrow = mdf[mdf["dataset_index"] == dataset_index].iloc[0]
        modelos_individuais[model] = C.parse_num_list(mrow["predictions"])

    outras_combinacoes = {}
    for folder, name in OUTRAS_COMBINACOES.items():
        odf = C.load_filtered(folder, dataset)
        orow = odf[odf["dataset_index"] == dataset_index].iloc[0]
        outras_combinacoes[name] = {
            "predictions": C.parse_num_list(orow["predictions"]),
            "smape": float(orow["smape"]),
            "pocid": float(orow["pocid"]),
        }

    pesos_por_modelo = None
    if row["best_strategy_method"] == "weighted" and weights_by_horizon is not None:
        pesos_por_modelo = {
            horizon_step: {m: w.get(m) for m in effective_models}
            for horizon_step, w in weights_by_horizon.items()
        }

    return {
        "dataset": dataset,
        "dataset_index": int(dataset_index),
        "test": test,
        "modelos_individuais_predictions": modelos_individuais,
        "outras_combinacoes": outras_combinacoes,
        "predicao_combinada_final": final_pred,
        "metodo_agregacao": row["best_strategy_method"],
        "estrategia_params": best_strategy_params,
        "pesos_por_modelo": pesos_por_modelo,
        "smape_final": float(row["smape"]),
        "pocid_final": float(row["pocid"]),
        "justificativa_final": row["justificativa_final"],
        "origin": "origin_" + row["description_json"].get("origin", ""),
    }


COMBINACOES_STYLE = {
    "ADE":     {"color": "#8172B3", "marker": "d", "markersize": 7, "linewidth": 2.0},
    "FFORMA":  {"color": "#C44E52", "marker": "^", "markersize": 7, "linewidth": 2.0},
    "Mediana": {"color": "#55A868", "marker": "s", "markersize": 6.5, "linewidth": 2.0},
    "DBA":     {"color": "#DD8452", "marker": "v", "markersize": 7, "linewidth": 2.0},
}


def plot_case(case: dict, out_path: str) -> None:
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

    import numpy as np

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 9.6), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    steps = list(range(1, len(case["test"]) + 1))

    def desenhar(ax, individuais=False):
        if individuais:
            primeiro = True
            for pred in case["modelos_individuais_predictions"].values():
                ax.plot(steps, pred, color="#BBBBBB", lw=1.0, zorder=2,
                        label="Individuais" if primeiro else None)
                primeiro = False
        ax.plot(steps, case["test"], color="#4C72B0", lw=2.6, marker="o", ms=6.5,
                zorder=4, label="Valor real")
        for name, data in case["outras_combinacoes"].items():
            ax.plot(steps, data["predictions"], zorder=3, label=name,
                    **COMBINACOES_STYLE[name])
        ax.plot(steps, case["predicao_combinada_final"], color="black", lw=3.0,
                zorder=5, label="CREST")
        ax.set_ylabel("Valor")
        ax.set_xticks(steps)
        ax.grid(alpha=0.25, lw=0.6)

    desenhar(ax1, individuais=True)
    desenhar(ax2)
    ax1.set_title("Escala completa", loc="left", fontsize=20)
    ax2.set_title("Detalhe das estratégias de combinação", loc="left", fontsize=20)
    ax2.set_xlabel("Passo do horizonte")

    cinco = [case["predicao_combinada_final"]] + [
        d["predictions"] for d in case["outras_combinacoes"].values()
    ]
    lo = min(float(np.min(v)) for v in cinco)
    hi = max(float(np.max(v)) for v in cinco)
    margem = 0.06 * (hi - lo) if hi > lo else 1.0
    ax2.set_ylim(lo - margem, hi + margem)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=4, frameon=False, handlelength=1.4, columnspacing=1.2,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = os.path.join(C.OUTPUTS_BASE, "interpretabilidade")
    fig_dir = os.path.join(out_dir, "figuras")
    os.makedirs(fig_dir, exist_ok=True)

    for name, (dataset, dataset_index) in CASES.items():
        case = extract_case(dataset, dataset_index)
        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(case, fh, ensure_ascii=False, indent=2)
        fig_path = os.path.join(fig_dir, f"{name}.png")
        plot_case(case, fig_path)
        print(f"{name}: {dataset} idx={dataset_index} origin={case['origin']} smape={case['smape_final']:.4f} "
              f"metodo={case['metodo_agregacao']} n_modelos={len(case['modelos_individuais_predictions'])} "
              f"-> {out_path} + {fig_path}")


if __name__ == "__main__":
    main()
