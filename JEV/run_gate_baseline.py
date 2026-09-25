#!/usr/bin/env python3
"""Baseline do gate: regressão logística sobre features tabulares.

Responde, ANTES de gastar GPU: existe sinal aprendível em "essa estratégia vai
bater o piso das sementes no teste?" com as features que já computamos?

Avaliação leave-one-dataset-out: treina em todos os datasets exceto um, avalia
no que ficou de fora. Reporta por fold: AUC, Brier, taxa base e n.

Critério do plano (PLANO_FINETUNE.md Task 2):
- AUC >= 0.60 em algum fold -> vale seguir para o fine-tune do LAYA
- AUC <= 0.55 em todos os folds -> revisar features antes de gastar GPU

Uso:
  python JEV/run_gate_baseline.py --data JEV/data/gate_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

FEATURES = ["score_val", "rank", "margem_pct", "n_attempts", "tau", "n_models",
            "origin_agent", "turn"]

#: Para alvos de VALIDAÇÃO o score/rank/margem carregam o rótulo na própria
#: feature (seria trapaça medir AUC com eles). O modelo real (LAYA texto) vê o
#: histórico com scores no estado — a questão honesta é se ele generaliza além
#: do argmin; o baseline tabular é medido SEM essas colunas.
VAL_FEATURES = ["n_attempts", "tau", "n_models", "origin_agent", "turn"]


def load(data_path: str) -> pd.DataFrame:
    rows = []
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["origin_agent"] = (df["origin"] == "agent").astype(float)
    for col in ["score_val", "rank", "margem_pct", "n_attempts", "tau", "n_models", "turn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="JEV/data/gate_dataset.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-agents", action="store_true",
                    help="avaliar só propostas de origem agent (sem as sementes)")
    ap.add_argument("--target", choices=["label", "label_seed", "label_dyn",
                                          "label_val", "label_val_seed",
                                          "label_val_w"],
                    default="label_val_w",
                    help="label/label_seed/label_dyn = alvos de teste (só análise); "
                         "label_val = melhor do universo na VALIDAÇÃO; "
                         "label_val_seed = vence as sementes na VALIDAÇÃO; "
                         "label_val_w = vence a MAIORIA das 3 janelas LOO (treino)")
    args = ap.parse_args()

    df = load(args.data)
    if args.only_agents:
        df = df[df.origin == "agent"]
    df["label"] = df[args.target]
    feats = VAL_FEATURES if args.target.startswith("label_val") else FEATURES
    print(f"alvo: {args.target} | features: {feats} | exemplos: {len(df)} "
          f"| label=1: {df.label.mean():.3f}")

    datasets = sorted(df.dataset.unique())
    print(f"\n{'fold (avaliado)':<22} {'n':>5} {'AUC':>7} {'Brier':>7} {'taxa base':>10}")
    aucs = []
    for holdout in datasets:
        tr = df[df.dataset != holdout]
        te = df[df.dataset == holdout]
        if len(te) < 20 or len(tr) < 50:
            print(f"{holdout:<22} {len(te):>5}  (poucos exemplos, pulado)")
            continue
        X_tr = tr[feats].fillna(0.0).values
        y_tr = tr.label.values.astype(int)
        X_te = te[feats].fillna(0.0).values
        y_te = te.label.values.astype(int)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p) if len(set(y_te)) > 1 else float("nan")
        brier = brier_score_loss(y_te, p)
        aucs.append(auc)
        print(f"{holdout:<22} {len(te):>5} {auc:>7.3f} {brier:>7.4f} {y_te.mean():>10.3f}")

    valid = [a for a in aucs if a == a]
    print(f"\nAUC médio (folds válidos): {np.mean(valid):.3f}")
    if any(a >= 0.60 for a in valid):
        print("VEREDITO: há sinal -> seguir para o fine-tune do LAYA (Task 3).")
    elif valid and max(valid) <= 0.55:
        print("VEREDITO: sem sinal com estas features -> revisar features antes da GPU.")
    else:
        print("VEREDITO: inconclusivo (AUC entre 0.55 e 0.60) — olhar por dataset.")
    # importância das features no último fold treinado
    if datasets:
        tr = df[df.dataset != datasets[-1]]
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
        m.fit(tr[feats].fillna(0.0).values, tr.label.values.astype(int))
        print("\ncoeficientes (último fold):")
        for name, coef in sorted(zip(feats, m.coef_[0]), key=lambda kv: -abs(kv[1])):
            print(f"  {name:<14} {coef:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
