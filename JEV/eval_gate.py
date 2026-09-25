#!/usr/bin/env python3
"""Avaliação do gate fine-tuned (Task 4 do PLANO_FINETUNE.md).

Para o fold de avaliação (mesmo --holdout do treino):
1. P(true) do checkpoint para cada exemplo do holdout; AUC/Brier vs rótulo.
2. Calibração: P média por rótulo.
3. Simulação da Fase 4: por série, escolhe a proposta com maior P entre as que
   passam do limiar τ; senão fica no piso das sementes. Reporta sMAPE final
   média vs piso (oráculo) para vários τ — usando APENAS P na escolha (o
   desfecho só entra para medir o resultado).

Uso (servidor):
  python JEV/eval_gate.py --checkpoint JEV/models/laya_gate_ettm2 \
      --data JEV/data/gate_dataset.jsonl --holdout ETTM2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

QUESTION = {
    "transfer": {
        "type": "noul",
        "instructions": (
            "Will this candidate strategy beat the best seeded baseline on the "
            "blind test window? Answer yes only if you expect it to be strictly better."
        ),
    }
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="diretório do checkpoint fine-tuned")
    ap.add_argument("--data", default="JEV/data/gate_dataset.jsonl")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    rows = []
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["dataset"] == args.holdout:
                    rows.append(r)
    print(f"holdout {args.holdout}: {len(rows)} exemplos")

    import laya
    agent = laya.Agent(args.checkpoint, device=args.device)

    ys, ps = [], []
    for r in rows:
        try:
            out = agent.predict(r["state"], QUESTION)
            p = float(out["answers"]["transfer"].get("noul", 0.5))
        except Exception as exc:
            print(f"  [warn] {r['source']} série {r['series']}: {type(exc).__name__}: {exc}")
            p = 0.5
        ys.append(int(r["label"]))
        ps.append(p)

    ys = np.array(ys)
    ps = np.array(ps)
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    brier = brier_score_loss(ys, ps)
    print(f"AUC={auc:.3f}  Brier={brier:.4f}  base={ys.mean():.3f}  n={len(ys)}")
    print(f"P médio: label=0 -> {ps[ys == 0].mean():.3f} | label=1 -> {ps[ys == 1].mean():.3f}")

    # ── simulação da Fase 4 por série ────────────────────────────────────────
    df = pd.DataFrame(rows)
    df["p"] = ps
    df["smape_test"] = df["smape_test"].astype(float)
    df["floor_smape_test"] = df["floor_smape_test"].astype(float)
    proposals = df[df.origin == "agent"]

    floor_mean = df.drop_duplicates(subset=["series"])["floor_smape_test"].mean()
    print(f"\npiso das sementes (oráculo, este dataset): {floor_mean:.4f}")
    print(f"{'tau':>5} {'séries c/ troca':>14} {'sMAPE final':>11} {'vs piso':>9}")
    for tau in [float(t) for t in args.sweep.split(",")]:
        finals = []
        n_swap = 0
        for series, g in proposals.groupby("series"):
            floor = df[(df.series == series)].iloc[0]["floor_smape_test"]
            above = g[g.p > tau]
            if len(above) == 0:
                finals.append(floor)
            else:
                # escolhe pela MAIOR P (não pelo desfecho)
                best = above.loc[above.p.idxmax()]
                finals.append(float(best["smape_test"]))
                n_swap += 1
        mean_s = float(np.mean(finals))
        print(f"{tau:>5.2f} {n_swap:>14} {mean_s:>11.4f} {mean_s - floor_mean:>+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
