#!/usr/bin/env python3
"""Analisa os runs isolados do Passo 1 (ETTM2) e imprime SMAPE/POCID médios e por série."""
import csv, json, os, sys

BASE = "/home/lucas.castro/Statistics_and_Seq2Seq/timeseries/mestrado/resultados"

RUNS = {
    "iso_control_repeat": "orchestrator_react_iso_control_repeat",
    "iso_reasoning_low": "orchestrator_react_iso_reasoning_low",
    "iso_reorder_low": "orchestrator_react_iso_reorder_low",
    "iso_dropcomb_low": "orchestrator_react_iso_dropcomb_low",
    "iso_reducedseed_low": "orchestrator_react_iso_reducedseed_low",
    "v5 (publicado)": "orchestrator_react_v5",
    "v2_gpt_low_nivel (combinado)": "orchestrator_react_v2_gpt_low_nivel",
}

def read_csv(run_name, folder):
    path = os.path.join(BASE, folder, "ETTM2.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(r)
    return rows

def summarize(rows):
    sm = [float(r["smape"]) for r in rows]
    pc = [float(r["pocid"]) for r in rows]
    return sum(sm) / len(sm), sum(pc) / len(pc), len(sm)

print(f"{'run':<28} {'n':>2} {'SMAPE%':>8} {'POCID%':>8}")
for label, folder in RUNS.items():
    rows = read_csv(label, folder)
    if rows is None:
        print(f"{label:<28}  -- ausente --")
        continue
    s, p, n = summarize(rows)
    print(f"{label:<28} {n:>2} {s*100:>8.2f} {p:>8.2f}")

print()
print("Por serie (SMAPE %):")
labels = list(RUNS.keys())
header = f"{'idx':>3} " + " ".join(f"{l:>16}" for l in labels)
print(header)
for idx in range(7):
    cells = []
    for label, folder in RUNS.items():
        rows = read_csv(label, folder)
        if rows is None:
            cells.append("             --")
            continue
        r = [x for x in rows if int(x["dataset_index"]) == idx]
        if not r:
            cells.append("             --")
            continue
        cells.append(f"{float(r[0]['smape'])*100:>16.2f}")
    print(f"{idx:>3} " + " ".join(cells))

print()
print("Estrategia final por serie (metodo | pool | pesos):")
for label, folder in RUNS.items():
    rows = read_csv(label, folder)
    if rows is None:
        print(f"{label}: ausente")
        continue
    line = []
    for r in rows:
        m = r.get("best_strategy_method")
        p = r.get("best_strategy_params") or ""
        try:
            pp = json.loads(p)
            pool = pp.get("pool", "")
            w = pp.get("weights", "") or pp.get("model", "")
        except Exception:
            pool, w = "", ""
        line.append(f"[{r['dataset_index']}]{m}/{pool}/{w}")
    print(f"{label}: " + " ".join(line))

print()
print("Origem (agent/baseline) por serie:")
for label, folder in RUNS.items():
    rows = read_csv(label, folder)
    if rows is None:
        continue
    # origem vem dos artefatos
    arts = os.path.join(BASE, folder, "llm_artifacts", "ETTM2")
    line = []
    for r in rows:
        idx = int(r["dataset_index"])
        ap = os.path.join(arts, f"dataset_{idx}.json")
        try:
            with open(ap) as f:
                d = json.load(f)
            origin = d["decision"]["origin"]
        except Exception:
            origin = "?"
        line.append(f"[{idx}]{origin}")
    print(f"{label}: " + " ".join(line))
