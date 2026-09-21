#!/usr/bin/env python3
"""Analisa as 10 execucoes da serie 5 (N=5 x 2 condicoes) + 2 ancoras.

Extrai: SMAPE, estrategia final, origem, se final = mean pool4, se o agente
AVALIOU mean pool4 em algum momento, elapsed_s, erros de parse, erros de handle
de pesos, erros de LLM (respostas vazias/transporte).
"""
import csv, json, os, re

BASE = "/home/lucas.castro/Statistics_and_Seq2Seq/timeseries/mestrado/resultados"
LOGS = "/home/lucas.castro/Statistics_and_Seq2Seq/logs"

RUNS = [f"iso_s5_none_{i:02d}" for i in range(1, 6)] + \
       [f"iso_s5_low_{i:02d}" for i in range(1, 6)] + \
       ["iso_s5_anchor_none", "iso_s5_anchor_low"]

def csv_row(version):
    path = os.path.join(BASE, f"orchestrator_react_{version}", "ETTM2.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if not rows:
        return None
    # runs so da serie 5: 1 linha; ancoras: filtra dataset_index==5
    for r in rows:
        if int(r["dataset_index"]) == 5:
            return r
    return None

def artifact(version):
    p = os.path.join(BASE, f"orchestrator_react_{version}", "llm_artifacts", "ETTM2", "dataset_5.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def elapsed(version):
    p = os.path.join(LOGS, f"{version}_ettm2.log")
    if not os.path.exists(p):
        return None
    text = open(p).read()
    m = re.findall(r"done in ([\d.]+)s", text)
    return float(m[-1]) if m else None

def analyze(version):
    row = csv_row(version)
    art = artifact(version)
    out = {"version": version, "ok": row is not None and art is not None}
    if not out["ok"]:
        return out
    out["smape"] = float(row["smape"])
    dec = art["decision"]
    strat = dec["strategy"]
    out["combine"] = strat.get("combine")
    out["pool"] = strat.get("pool")
    out["origin"] = dec.get("origin")
    out["is_mean_pool4"] = (strat.get("combine") == "mean" and strat.get("pool") == "pool4")
    out["elapsed_s"] = elapsed(version)

    traj = art["react"].get("trajectory") or []
    # o agente avaliou mean pool4 em algum momento?
    out["evaluated_mean_pool4"] = any(
        t.get("action") == "evaluate_strategy"
        and (t.get("action_args") or {}).get("combine") == "mean"
        and (t.get("action_args") or {}).get("pool") == "pool4"
        for t in traj
    )
    # erros de parse (acoes unparsed + parse_failures registrados)
    n_unparsed = sum(1 for t in traj if t.get("action") == "unparsed")
    pf = art["react"].get("parse_failures") or []
    out["n_parse"] = n_unparsed + len(pf)
    # erros de handle de pesos / argumentos invalidos nas observacoes
    obs = [t.get("observation_summary") or "" for t in traj]
    out["n_handle_err"] = sum(1 for o in obs if ("unknown weights handle" in o or "computed over" in o or "invalid_argument" in o))
    # erros de LLM (vazio/transporte)
    errs = art["react"].get("errors") or []
    out["n_llm_err"] = len(errs)
    out["llm_err_kinds"] = sorted({(e.get("kind") or "?") if isinstance(e, dict) else str(e)[:40] for e in errs})
    out["n_iters"] = art["react"].get("summary", {}).get("iterations_used")
    out["stop_reason"] = art["react"].get("summary", {}).get("stop_reason")
    return out

rows = [analyze(v) for v in RUNS]
rows = [r for r in rows if r.get("ok")]

def cond(r):
    return "none" if "none" in r["version"] else "low"

core = [r for r in rows if "anchor" not in r["version"]]
anchors = [r for r in rows if "anchor" in r["version"]]

print("=== EXECUCOES (serie 5) ===")
print(f"{'version':<20} {'cond':<5} {'smape%':>7} {'estrategia':<28} {'origem':<9} {'meanP4':<6} {'evalMeanP4':<10} {'parse':>5} {'handle':>6} {'llm':>4} {'elapsed_s':>9} {'iters':>5} {'stop':<25}")
for r in core + anchors:
    if not r.get("ok"):
        print(f"{r['version']:<20} FALHOU/AUSENTE")
        continue
    print(f"{r['version']:<20} {cond(r):<5} {r['smape']*100:>7.2f} {r['combine']+'/'+(r['pool'] or ''):<28} {r['origin']:<9} {str(r['is_mean_pool4']):<6} {str(r['evaluated_mean_pool4']):<10} {r['n_parse']:>5} {r['n_handle_err']:>6} {r['n_llm_err']:>4} {r['elapsed_s']:>9.1f} {str(r['n_iters']):>5} {str(r['stop_reason']):<25}")

print()
print("=== ANALISE POR CONDICAO (core, N=5 cada) ===")
for c in ["none", "low"]:
    rs = [r for r in core if cond(r) == c]
    sm = [r["smape"] for r in rs]
    succ = [r for r in rs if r["is_mean_pool4"]]
    import statistics
    print(f"\n-- {c} (n={len(rs)}) --")
    print(f"  sucessos (final=mean pool4): {len(succ)}/5")
    print(f"  SMAPE%: media={statistics.mean(sm)*100:.2f} sd={statistics.pstdev(sm)*100:.3f} min={min(sm)*100:.2f} max={max(sm)*100:.2f}")
    print(f"  valores: {[round(s*100,2) for s in sm]}")
    el = [r["elapsed_s"] for r in rs if r["elapsed_s"] is not None]
    print(f"  elapsed_s: media={statistics.mean(el):.1f} sd={statistics.pstdev(el):.1f} valores={[round(e,1) for e in el]}")
    print(f"  parse_errors total: {sum(r['n_parse'] for r in rs)} | handle_errors total: {sum(r['n_handle_err'] for r in rs)} | llm_errors total: {sum(r['n_llm_err'] for r in rs)}")
    print(f"  avaliou mean pool4 em algum momento: {sum(1 for r in rs if r['evaluated_mean_pool4'])}/5")

# Fisher
from scipy.stats import fisher_exact
a = sum(1 for r in core if cond(r) == "none" and r["is_mean_pool4"])
b = sum(1 for r in core if cond(r) == "none" and not r["is_mean_pool4"])
c_ = sum(1 for r in core if cond(r) == "low" and r["is_mean_pool4"])
d_ = sum(1 for r in core if cond(r) == "low" and not r["is_mean_pool4"])
odds, p = fisher_exact([[a, b], [c_, d_]])
print()
print(f"=== FISHER (sucessos mean pool4) === table=[[{a},{b}],[{c_},{d_}]] p={p:.4f} oddsratio={odds:.3f}")

print()
print("=== ANCORAS (dataset inteiro, com DATASET CARD) ===")
for r in anchors:
    print(f"  {r['version']}: smape={r['smape']*100:.2f}% estrat={r['combine']}/{r['pool']} origem={r['origin']} meanP4={r['is_mean_pool4']} evalMeanP4={r['evaluated_mean_pool4']} elapsed={r['elapsed_s']:.1f}s parse={r['n_parse']} handle={r['n_handle_err']}")
