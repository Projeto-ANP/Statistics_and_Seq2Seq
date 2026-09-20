"""Aceite forcado? -- Passos 3 e 4: iteracoes/chamadas/tempo por origem da estrategia final.

Fonte: CSVs de orchestrator_react_v5 (gpt-oss) e _qwen (o _gemma26 so existe na branch v1_gemma) (description.loop,
n_evaluate_calls). Saida em stdout (markdown), usada em
outputs/resultados/ablacao/RESUMO_aceite_forcado.md.

Definicoes:
- iteracoes = loop.iterations_used (turnos do ciclo, ate max_iterations=12).
- chamadas ao LLM = n_trajectory_steps + empty_responses + llm_error_retries
  (as respostas vazias/erros sao re-perguntadas sem gastar iteracao).
- avaliacoes = n_evaluate_calls (chamadas evaluate_strategy, so estas contam para estagnacao).
"""
import os
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import pandas as pd

MODELS = {
    "gpt-oss:20b": C.ORCHESTRATOR_GPTOSS,
    "qwen3:30b-a3b": C.ORCHESTRATOR_QWEN,
}


def collect(folder):
    rows = []
    for ds in C.DATASETS:
        df = C.load_orchestrator(folder, ds)
        for r in df.itertuples(index=False):
            loop = r.description_json["loop"]
            rows.append({
                "dataset": ds,
                "origin": r.description_json.get("origin"),
                "iters": loop["iterations_used"],
                "steps": loop["n_trajectory_steps"],
                "calls": loop["n_trajectory_steps"] + loop.get("empty_responses", 0) + loop.get("llm_error_retries", 0),
                "evals": int(r.n_evaluate_calls),
                "elapsed": float(loop["elapsed_s"]),
                "accepted": loop.get("agent_accepted_id") is not None,
                "stop": loop.get("stop_reason") or "",
            })
    return pd.DataFrame(rows)


def dist(s):
    return f"{s.min():.0f} / {s.median():.0f} / {s.mean():.1f} / {s.max():.0f}"


def stop_kind(row):
    if row["accepted"]:
        return "aceite explicito (accept)"
    if row["stop"].startswith("no improvement"):
        return "estagnacao"
    if row["stop"] == "iteration_budget_exhausted":
        return "orcamento esgotado"
    return "outro (" + row["stop"] + ")"


def main():
    for name, folder in MODELS.items():
        d = collect(folder)
        d["stop_kind"] = d.apply(stop_kind, axis=1)
        print(f"\n## {name} (n={len(d)})")
        print("| grupo | n | iteracoes min/mediana/media/max | chamadas ao LLM min/med/media/max | avaliacoes min/med/media/max | tempo medio (s) |")
        print("|---|---|---|---|---|---|")
        for label, sub in (("origin=baseline", d[d.origin == "baseline"]), ("origin=agent", d[d.origin == "agent"]), ("todas", d)):
            print(f"| {label} | {len(sub)} | {dist(sub['iters'])} | {dist(sub['calls'])} | {dist(sub['evals'])} | {sub['elapsed'].mean():.1f} |")

        b = d[d.origin == "baseline"]
        print("\nmotivo de encerramento (origin=baseline):", dict(Counter(b["stop_kind"])))
        print("motivo de encerramento (origin=agent):", dict(Counter(d[d.origin == 'agent']["stop_kind"])))
        acc_b = b[b.accepted]
        print(f"origin=baseline com accept: {len(acc_b)}; accept na iteracao 1: {int((acc_b['iters'] == 1).sum())}; "
              f"iteracoes ate o accept (min/med/media/max): {dist(acc_b['iters']) if len(acc_b) else 'n/a'}")
        print(f"origin=baseline com 0 avaliacoes: {int((b['evals'] == 0).sum())}; com 1: {int((b['evals'] == 1).sum())}; "
              f">=4: {int((b['evals'] >= 4).sum())}")
        print(f"origin=baseline que usaram >1 iteracao: {int((b['iters'] > 1).sum())} de {len(b)} ({(b['iters'] > 1).mean()*100:.1f}%)")

        acc = d[d.accepted]
        print(f"\naccept (todas as series): {len(acc)}; por iteracao: {dict(sorted(Counter(acc['iters']).items()))}")
        print(f"accept na iteracao 1: {int((acc['iters'] == 1).sum())}; na ultima iteracao (12): {int((acc['iters'] == 12).sum())}")

        tot = d["elapsed"].sum()
        print(f"\nTempo total da execucao: {tot/3600:.2f} h; tempo nas series origin=baseline: {b['elapsed'].sum()/3600:.2f} h "
              f"({b['elapsed'].sum()/tot*100:.1f}% do total); chamadas ao LLM nessas series: {b['calls'].sum()} de {d['calls'].sum()} "
              f"({b['calls'].sum()/d['calls'].sum()*100:.1f}%)")
        sec_call = d["elapsed"].sum() / d["calls"].sum()
        avoid = (b["elapsed"] - sec_call * 1).clip(lower=0).sum()
        print(f"segundos por chamada (media global): {sec_call:.2f}; estimativa de tempo evitavel se essas series encerrassem "
              f"apos 1 chamada: {avoid/3600:.2f} h ({avoid/tot*100:.1f}% do total); chamadas evitaveis: {int((b['calls'] - 1).clip(lower=0).sum())}")

        if name == "gpt-oss:20b":
            print("\n| dataset | series baseline | tempo medio baseline (s) | tempo medio total do dataset (s) | % do tempo do dataset em series baseline |")
            print("|---|---|---|---|---|")
            for ds in C.DATASETS:
                x = d[d.dataset == ds]
                xb = x[x.origin == "baseline"]
                print(f"| {ds} | {len(xb)}/{len(x)} | {xb['elapsed'].mean():.1f} | {x['elapsed'].mean():.1f} | {xb['elapsed'].sum()/x['elapsed'].sum()*100:.1f}% |")


if __name__ == "__main__":
    main()
