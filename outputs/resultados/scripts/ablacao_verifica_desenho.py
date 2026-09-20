"""Ablacao -- Passo 1: confere se orchestrator_baseline_v1 e orchestrator_react_v5
so diferem pelo agente.

Compara, nos artefatos JSON por serie (llm_artifacts): (a) config salva, (b) metadados
do ciclo (llm_model, iteracoes, tentativas de agente), (c) conjunto de sementes
(id, estrategia, tamanho do pool) e (d) score/rmse/smape de cada semente.

Saida: stdout (usado no RESUMO_contribuicao_agente.md).
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def load(folder):
    out = {}
    pattern = os.path.join(C.RESULTADOS_BASE, folder, "llm_artifacts", "*", "*.json")
    for f in glob.glob(pattern):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        out[(os.path.basename(os.path.dirname(f)), d["dataset_index"])] = d
    return out


def main():
    A = load(C.ORCHESTRATOR_BASELINE)
    B = load(C.ORCHESTRATOR_GPTOSS)
    keys = sorted(set(A) & set(B))
    print(f"artefatos: baseline={len(A)} v5={len(B)} em comum={len(keys)}")

    print("\n== config salva: chaves que diferem entre as execucoes ==")
    for k in sorted(set(next(iter(A.values()))["config"]) | set(next(iter(B.values()))["config"])):
        va = collections.Counter(json.dumps(d["config"].get(k), sort_keys=True) for d in A.values())
        vb = collections.Counter(json.dumps(d["config"].get(k), sort_keys=True) for d in B.values())
        if set(va) != set(vb):
            print(f"  {k}: baseline={dict(va)} | v5={dict(vb)}")

    print("\n== ciclo do agente (baseline) ==")
    meta = collections.Counter(
        (d["decision"]["loop"].get("llm_model"), d["decision"]["loop"].get("iterations_used"),
         d["decision"]["loop"].get("stop_reason"), d["decision"]["provenance"]["n_agent_attempts"])
        for d in A.values()
    )
    print("  (llm_model, iteracoes, stop_reason, n_agent_attempts):", dict(meta))

    print("\n== sementes ==")
    def seeds(d):
        return {b["id"]: b for b in d["phase2"]["baselines"]}
    n_set_diff = 0
    diff_ids = collections.Counter()
    a3_eq_a1 = {"baseline": 0, "v5": 0}
    counts = collections.Counter()
    for k in keys:
        sa, sb = seeds(A[k]), seeds(B[k])
        counts[(len(sa), len(sb))] += 1
        if set(sa) != set(sb) or any((sa[i]["strategy"], sa[i]["n_models"]) != (sb[i]["strategy"], sb[i]["n_models"]) for i in sa):
            n_set_diff += 1
            continue
        for i in sa:
            if abs(sa[i]["score"] - sb[i]["score"]) > 1e-9:
                diff_ids[i] += 1
        for tag, s in (("baseline", sa), ("v5", sb)):
            if s["a3"]["score"] == s["a1"]["score"] and s["a3"]["rmse"] == s["a1"]["rmse"]:
                a3_eq_a1[tag] += 1
    print("  n sementes (baseline, v5) -> n series:", dict(counts))
    print("  series cujo conjunto de sementes (id, estrategia, n_models) difere:", n_set_diff)
    print("  sementes com score diferente (por id):", dict(diff_ids))
    print("  a3 (dba) numericamente igual a a1 (mean):", a3_eq_a1)


if __name__ == "__main__":
    main()
