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
    "label": {
        "transfer": {
            "type": "noul",
            "instructions": (
                "Will this candidate strategy beat every reference combination (FFORMA, "
                "ADE, and the seeded baselines) on the blind test window? Answer yes "
                "only if you expect it to be strictly better than all of them."
            ),
        }
    },
    "label_dyn": {
        "transfer": {
            "type": "noul",
            "instructions": (
                "Is this candidate strategy the best available strategy for this "
                "series — better than every individual model, every combination "
                "(mean, median, dba, trimmed, weighted) and every seeded baseline? "
                "Answer yes only if you expect it to be strictly the best."
            ),
        }
    },
    "label_val": {
        "transfer": {
            "type": "noul",
            "instructions": (
                "Is this candidate strategy the best available strategy for this "
                "series on the validation windows (nested leave-one-out)? Answer "
                "yes only if you expect it to be strictly the best on validation."
            ),
        }
    },
    "label_val_seed": {
        "transfer": {
            "type": "noul",
            "instructions": (
                "Will this candidate strategy beat every seeded baseline on the "
                "validation windows (nested leave-one-out)? Answer yes only if you "
                "expect it to be strictly better than all of them on validation."
            ),
        }
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="diretório do checkpoint fine-tuned")
    ap.add_argument("--data", default="JEV/data/gate_dataset.jsonl")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", choices=["label", "label_dyn", "label_val",
                                          "label_val_seed", "label_val_w"],
                    default="label_val_w",
                    help="label_val/label_val_seed/label_val_w = alvos SÓ de "
                         "validação; label/label_dyn = teste (só análise)")
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
            if args.target == "label_val_w":
                pw = []
                for w in range(3):
                    q = {
                        "transfer": {
                            "type": "noul",
                            "instructions": (
                                f"Will this candidate strategy be the winner of "
                                f"validation window {w + 1} of 3, scored leave-one-out "
                                f"(everything else fitted on the other two windows)? "
                                f"Answer yes only if you expect it to win that window."
                            ),
                        }
                    }
                    out = agent.predict(r["state"], q)
                    pw.append(float(out["answers"]["transfer"].get("noul", 0.5)))
                p = float(np.mean(pw))
            else:
                out = agent.predict(r["state"], QUESTION[args.target])
                p = float(out["answers"]["transfer"].get("noul", 0.5))
        except Exception as exc:
            print(f"  [warn] {r['source']} série {r['series']}: {type(exc).__name__}: {exc}")
            p = 0.5
        ys.append(int(r[args.target]))
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
    df["ref_best_smape"] = df["ref_best_smape"].astype(float)
    proposals = df[df.origin == "agent"]

    per_series = df.drop_duplicates(subset=["series"])
    ref_mean = per_series["ref_best_smape"].mean()
    floor_mean = per_series["floor_smape_test"].mean()
    print(f"\npiso das sementes (oráculo, este dataset): {floor_mean:.4f}")
    print(f"melhor referência (piso ou FFORMA/ADE/etc): {ref_mean:.4f}")

    if args.target in ("label_dyn", "label_val", "label_val_w"):
        # ranqueador dinâmico: por série, escolhe o candidato de MAIOR P entre
        # TODOS os candidatos do universo (sementes + individuais + propostas)
        finals, universe_best = [], []
        for series, g in df.groupby("series"):
            best = g.loc[g.p.idxmax()]
            finals.append(float(best["smape_test"]))
            universe_best.append(float(g["universe_best_smape"].iloc[0]))
        print(f"\nRANQUEADOR DINÂMICO (argmax de P sobre todos os candidatos):")
        print(f"  sMAPE final      : {np.mean(finals):.4f}")
        print(f"  oráculo universo : {np.mean(universe_best):.4f} "
              f"(o que seria possível acertando sempre)")
        print(f"  melhor referência: {ref_mean:.4f}")
        print(f"  vs melhor ref    : {np.mean(finals) - ref_mean:+.4f}")
        print(f"  vs oráculo       : {np.mean(finals) - np.mean(universe_best):+.4f}")
        return 0

    if args.target == "label_val_seed":
        # gate treinado só em validação: troca a semente se P > tau
        print(f"{'tau':>5} {'séries c/ troca':>14} {'sMAPE final':>11} {'vs piso':>9}")
        for tau in [float(t) for t in args.sweep.split(",")]:
            finals, n_swap = [], 0
            for series, g in proposals.groupby("series"):
                floor = df[(df.series == series)].iloc[0]["floor_smape_test"]
                above = g[g.p > tau]
                if len(above) == 0:
                    finals.append(float(floor))
                else:
                    best = above.loc[above.p.idxmax()]
                    finals.append(float(best["smape_test"]))
                    n_swap += 1
            mean_s = float(np.mean(finals))
            print(f"{tau:>5.2f} {n_swap:>14} {mean_s:>11.4f} {mean_s - floor_mean:>+9.4f}")
        return 0

    print(f"{'tau':>5} {'séries c/ troca':>14} {'sMAPE final':>11} {'vs melhor ref':>14}")
    for tau in [float(t) for t in args.sweep.split(",")]:
        finals = []
        n_swap = 0
        for series, g in proposals.groupby("series"):
            fallback = df[(df.series == series)].iloc[0]["ref_best_smape"]
            above = g[g.p > tau]
            if len(above) == 0:
                finals.append(float(fallback))
            else:
                # escolhe pela MAIOR P (não pelo desfecho)
                best = above.loc[above.p.idxmax()]
                finals.append(float(best["smape_test"]))
                n_swap += 1
        mean_s = float(np.mean(finals))
        print(f"{tau:>5.2f} {n_swap:>14} {mean_s:>11.4f} {mean_s - ref_mean:>+14.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
