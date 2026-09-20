"""Ablacao -- Passo 4: quantas series o agente de fato altera.

Pareia as 680 series de orchestrator_react_v5 (com agente) e
orchestrator_baseline_v1 (so sementes) e compara a estrategia final
(`best_strategy_params`: combine + pool + weights/trim_pct/model, como JSON canonico)
e o SMAPE de teste. Empate = |diferenca de SMAPE| <= EPS (1e-4).

Sensibilidade: o seed a3 (dba) do v5 era numericamente a media (bug de combine_dba
ja documentado no docstring de orchestrator_react/combiners.py), entao a comparacao
tambem e reportada excluindo as series em que o vencedor do baseline_v1 e o DBA.

Saida: stdout (markdown) + outputs/resultados/ablacao/pareado_por_serie.csv
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import numpy as np
import pandas as pd

EPS = 1e-4


def canonical(params_json: str) -> str:
    return json.dumps(json.loads(params_json), sort_keys=True)


def load(folder):
    frames = []
    for dataset in C.DATASETS:
        df = C.load_orchestrator(folder, dataset)
        df = df.assign(dataset=dataset)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(sub: pd.DataFrame, label: str):
    n = len(sub)
    same = sub[sub["mesma_estrategia"]]
    diff = sub[~sub["mesma_estrategia"]]
    better = int((diff["smape_v5"] < diff["smape_base"] - EPS).sum())
    worse = int((diff["smape_v5"] > diff["smape_base"] + EPS).sum())
    tie = len(diff) - better - worse
    print(f"\n### {label} (n={n})")
    print(f"- estrategia final identica: {len(same)} ({len(same)/n*100:.1f}%)")
    print(f"- estrategia final diferente: {len(diff)} ({len(diff)/n*100:.1f}%)")
    print(f"  - agente melhor (SMAPE menor, > {EPS}): {better}")
    print(f"  - agente pior (SMAPE maior, > {EPS}): {worse}")
    print(f"  - empate (|dif| <= {EPS}): {tie}")
    if len(diff):
        print(f"  - diferenca media de SMAPE (v5 - baseline) nas diferentes: {diff['dsmape'].mean():+.4f}")
    return {"n": n, "same": len(same), "diff": len(diff), "better": better, "worse": worse, "tie": tie}


def main():
    v5 = load(C.ORCHESTRATOR_GPTOSS)
    base = load(C.ORCHESTRATOR_BASELINE)
    cols = ["dataset", "dataset_index", "best_strategy_params", "smape", "pocid", "predictions"]
    m = v5[cols + ["description_json"]].merge(
        base[cols], on=["dataset", "dataset_index"], suffixes=("_v5", "_base"), validate="one_to_one"
    )
    print(f"series pareadas: {len(m)}")

    m["origin_v5"] = m["description_json"].map(lambda d: d.get("origin"))
    m["estrategia_v5"] = m["best_strategy_params_v5"].map(canonical)
    m["estrategia_base"] = m["best_strategy_params_base"].map(canonical)
    m["mesma_estrategia"] = m["estrategia_v5"] == m["estrategia_base"]
    m["smape_v5"] = m["smape_v5"].astype(float)
    m["smape_base"] = m["smape_base"].astype(float)
    m["dsmape"] = m["smape_v5"] - m["smape_base"]
    m["base_vencedor_dba"] = m["estrategia_base"].map(lambda s: json.loads(s).get("combine") == "dba")

    def maxdiff(r):
        a = np.array(C.parse_num_list(r["predictions_v5"]))
        b = np.array(C.parse_num_list(r["predictions_base"]))
        return float(np.max(np.abs(a - b)))

    m["max_dif_previsao"] = m.apply(maxdiff, axis=1)

    same = m[m["mesma_estrategia"]]
    print(f"\nnas {len(same)} series com estrategia identica: previsoes identicas (dif max <= 1e-6) em "
          f"{int((same['max_dif_previsao'] <= 1e-6).sum())}; diferentes em {int((same['max_dif_previsao'] > 1e-6).sum())} "
          f"(dessas, vencedor DBA: {int(same[same['max_dif_previsao'] > 1e-6]['base_vencedor_dba'].sum())})")
    print(f"series em que o vencedor do baseline_v1 e DBA: {int(m['base_vencedor_dba'].sum())}")
    print(f"origin do vencedor no v5 -- agent: {int((m['origin_v5'] == 'agent').sum())}, baseline: {int((m['origin_v5'] == 'baseline').sum())}")
    print(f"estrategia diferente com origin_v5=baseline: {int(((~m['mesma_estrategia']) & (m['origin_v5'] == 'baseline')).sum())}")

    res_all = summarize(m, "Todas as series")
    summarize(m[~m["base_vencedor_dba"]], "Excluindo series em que o vencedor sem agente e DBA")

    print("\n### Por dataset (todas as series)")
    print("| dataset | n | identica | diferente | agente melhor | agente pior | empate |")
    print("|---|---|---|---|---|---|---|")
    for dataset in C.DATASETS:
        sub = m[m["dataset"] == dataset]
        d = sub[~sub["mesma_estrategia"]]
        b = int((d["smape_v5"] < d["smape_base"] - EPS).sum())
        w = int((d["smape_v5"] > d["smape_base"] + EPS).sum())
        print(f"| {dataset} | {len(sub)} | {int(sub['mesma_estrategia'].sum())} | {len(d)} | {b} | {w} | {len(d) - b - w} |")

    out_dir = os.path.join(C.OUTPUTS_BASE, "ablacao")
    os.makedirs(out_dir, exist_ok=True)
    m[["dataset", "dataset_index", "origin_v5", "estrategia_v5", "estrategia_base", "mesma_estrategia",
       "smape_v5", "smape_base", "dsmape", "base_vencedor_dba", "max_dif_previsao"]].to_csv(
        os.path.join(out_dir, "pareado_por_serie.csv"), index=False)


if __name__ == "__main__":
    main()
