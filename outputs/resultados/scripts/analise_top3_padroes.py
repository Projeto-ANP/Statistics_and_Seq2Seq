"""Padroes para escolher entre os tres melhores da validacao.

Le top3_candidatos.csv (cada linha e um candidato do top-3 de validacao de uma
serie, com diagnostico/perfil/pool, evidencia de validacao e o resultado no
teste). Analisa:

  - em que rank de validacao costuma estar o melhor do teste;
  - o efeito do combination_hint do diagnostico e da familia da estrategia;
  - o efeito do gap/veredito de validacao e de features da serie;
  - a arvore de decisao e as features mais associadas ao melhor do top-3;
  - regras simples e um seletor treinado por validacao cruzada, com o resultado
    final de cada uma no teste.

O teste NUNCA entra como feature; ele so avalia as regras.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT_DIR = os.path.join(_REPO, "outputs", "resultados", "top3_val_teste")
IN = os.path.join(OUT_DIR, "top3_candidatos.csv")

CATS = [
    "dataset", "family", "combine", "hint", "regime", "predictability",
    "stability_verdict", "selection_verdict", "agent_final", "seed_final",
]
NUMS = [
    "val", "val_rank", "val_gap_best", "val_gap_pct", "worst_window",
    "mean_window", "std_window", "pool_size", "trim_pct",
    "trend", "seasonal", "entropy", "stability_tau", "spread", "n_models",
    "selection_margin",
]


def hint_match(row):
    hint = str(row.get("hint") or "")
    fam = str(row.get("family") or "")
    combine = fam.split(":")[0]
    if hint == "robust":
        return combine in ("median", "trimmed_mean")
    if hint == "weighted":
        return combine == "weighted"
    if hint == "selective":
        return combine in ("best_single",) or (
            pd.notna(row.get("pool_size")) and float(row["pool_size"]) <= 5
        )
    if hint == "full_pool":
        return combine in ("mean", "median", "trimmed_mean") and (
            pd.isna(row.get("pool_size")) or float(row["pool_size"]) >= 15
        )
    return False


def pick(df, mask, fallback="val"):
    sub = df[mask]
    if len(sub):
        return sub.loc[sub["val"].idxmin()]
    if fallback == "val":
        return df.loc[df["val"].idxmin()]
    return df.loc[df["smape"].idxmin()]


def evaluate_series(df, picks, label):
    out = {}
    for metric, col in (("smape", "smape"), ("pocid", "pocid")):
        vals = []
        for (dataset, idx), df_s in df.groupby(["dataset", "dataset_index"]):
            cand = picks.get((dataset, idx))
            if cand is None:
                cand = df_s.loc[df_s["val"].idxmin()]
            vals.append(float(cand[col]))
        out[metric] = float(np.mean(vals))
    return label, out["smape"], out["pocid"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default=IN)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.inp, sep=";")
    df[NUMS] = df[NUMS].replace([np.inf, -np.inf], np.nan)
    lo, hi = df[NUMS].quantile(0.01), df[NUMS].quantile(0.99)
    df[NUMS] = df[NUMS].clip(lower=lo, upper=hi, axis=1)
    df["agent_final"] = df["agent_final"].astype(str)
    df["seed_final"] = df["seed_final"].astype(str)
    print(f"{len(df)} candidatos, {df.groupby(['dataset','dataset_index']).ngroups} series")
    print("candidatos por serie:", df.groupby(["dataset", "dataset_index"]).size().value_counts().to_dict())

    rows = []
    # 1) rank do melhor do teste dentro do top-3
    for metric, target in (("smape", "is_test_best_smape"), ("pocid", "is_test_best_pocid")):
        best = df[df[target]]
        dist = best["val_rank"].value_counts().sort_index()
        print(f"\n== {metric}: rank de validacao do melhor do top-3 ==")
        print(dist.to_dict())
        for k in (1, 2, 3):
            sub = df[df["val_rank"] == k]
            picks = {}
            for key, df_s in df.groupby(["dataset", "dataset_index"]):
                picks[key] = df_s.loc[df_s["val"].idxmin()] if k == 1 else None
            # regra "sempre rank k" (se existir; senao o melhor da validacao)
            picks = {}
            for key, df_s in df.groupby(["dataset", "dataset_index"]):
                cand = df_s[df_s["val_rank"] == k]
                picks[key] = cand.loc[cand["val"].idxmin()] if len(cand) else df_s.loc[df_s["val"].idxmin()]
            label, s, p = evaluate_series(df, picks, f"rank{k}")
            rows.append({"regra": label, "smape": s, "pocid": p})

    # 2) hint, familias e regras simples
    for key, df_s in df.groupby(["dataset", "dataset_index"]):
        pass
    picks_hint, picks_worst, picks_mean, picks_agent, picks_seed, picks_val = {}, {}, {}, {}, {}, {}
    picks_oracle, picks_oracle_p = {}, {}
    for key, df_s in df.groupby(["dataset", "dataset_index"]):
        picks_val[key] = df_s.loc[df_s["val"].idxmin()]
        picks_oracle[key] = df_s.loc[df_s["smape"].idxmin()]
        picks_oracle_p[key] = df_s.loc[df_s["pocid"].idxmax()]
        m = df_s.apply(hint_match, axis=1)
        picks_hint[key] = pick(df_s, m)
        picks_worst[key] = df_s.loc[df_s["worst_window"].idxmin()] if df_s["worst_window"].notna().any() else df_s.loc[df_s["val"].idxmin()]
        picks_mean[key] = df_s.loc[df_s["mean_window"].idxmin()] if df_s["mean_window"].notna().any() else df_s.loc[df_s["val"].idxmin()]
        a = df_s[df_s["agent_final"] == "True"]
        picks_agent[key] = a.loc[a["val"].idxmin()] if len(a) else df_s.loc[df_s["val"].idxmin()]
        sd = df_s[df_s["seed_final"] == "True"]
        picks_seed[key] = sd.loc[sd["val"].idxmin()] if len(sd) else df_s.loc[df_s["val"].idxmin()]
    _, _so, _ = evaluate_series(df, picks_oracle, "top3_oracle")
    _, _, _po = evaluate_series(df, picks_oracle_p, "top3_oracle")
    rows.append({"regra": "top3_oracle", "smape": _so, "pocid": _po})
    for label, picks in (
        ("validacao", picks_val),
        ("hint", picks_hint),
        ("worst_window", picks_worst),
        ("mean_window", picks_mean),
        ("agent_final", picks_agent),
        ("seed_final", picks_seed),
    ):
        rows.append(dict(zip(("regra", "smape", "pocid"), evaluate_series(df, picks, label))))

    # 3) arvore: alvo = melhor SMAPE do top-3; validacao cruzada por serie
    y = df["is_test_best_smape"].astype(int)
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("pass", "passthrough")])
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
         ("num", num_pipe, NUMS)]
    )
    tree = Pipeline([
        ("pre", pre),
        ("tree", DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=20, class_weight="balanced", random_state=7)),
    ])
    groups = df["dataset"].astype(str) + "_" + df["dataset_index"].astype(str)
    proba = cross_val_predict(tree, df[CATS + NUMS], y, groups=groups, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    df["tree_proba"] = proba
    picks_tree = {}
    for key, df_s in df.groupby(["dataset", "dataset_index"]):
        picks_tree[key] = df_s.loc[df_s["tree_proba"].idxmax()]
    _, _ts, _tp = evaluate_series(df, picks_tree, "arvore_cv")
    rows.append({"regra": "arvore_cv", "smape": _ts, "pocid": _tp})

    yp = df["is_test_best_pocid"].astype(int)
    proba_p = cross_val_predict(tree, df[CATS + NUMS], yp, groups=groups, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    df["tree_proba_p"] = proba_p
    picks_tree_p = {}
    for key, df_s in df.groupby(["dataset", "dataset_index"]):
        picks_tree_p[key] = df_s.loc[df_s["tree_proba_p"].idxmax()]
    _, _ts, _tp = evaluate_series(df, picks_tree_p, "arvore_cv_pocid")
    rows.append({"regra": "arvore_cv_pocid", "smape": _ts, "pocid": _tp})

    tree.fit(df[CATS + NUMS], y)
    names = list(tree.named_steps["pre"].get_feature_names_out())
    importances = pd.Series(tree.named_steps["tree"].feature_importances_, index=names).sort_values(ascending=False)
    print("\n== features mais importantes da arvore ==")
    print(importances.head(15).round(3).to_string())
    print("\n== arvore (profundidade", args.depth, ") ==")
    print(export_text(tree.named_steps["tree"], feature_names=names, max_depth=args.depth))

    resumo = pd.DataFrame(rows)
    print("\n== regras no teste ==")
    print(resumo.round(2).to_string(index=False))
    resumo.to_csv(os.path.join(OUT_DIR, "resumo_top3_regras.csv"), index=False, sep=";")

    # 4) contingencias
    print("\n== hint x melhor do top-3 (SMAPE) ==")
    print(pd.crosstab(df["hint"], df["is_test_best_smape"]).to_string())
    print("\n== familia x melhor do top-3 (SMAPE) ==")
    print(pd.crosstab(df["family"], df["is_test_best_smape"]).to_string())
    print("\n== veredito da selecao x melhor do top-3 (SMAPE) ==")
    print(pd.crosstab(df["selection_verdict"], df["is_test_best_smape"]).to_string())
    print("\n== regime x melhor do top-3 (SMAPE) ==")
    print(pd.crosstab(df["regime"], df["is_test_best_smape"]).to_string())

    # 5) por dataset das regras principais
    rule_cols = ["validacao", "top3_oracle", "hint", "worst_window", "agent_final", "arvore_cv"]
    per_ds = []
    for dataset, sub in df.groupby("dataset"):
        for label, picks in (
            ("validacao", picks_val), ("top3_oracle", picks_oracle), ("hint", picks_hint),
            ("worst_window", picks_worst), ("agent_final", picks_agent), ("arvore_cv", picks_tree), ("arvore_cv_pocid", picks_tree_p),
        ):
            keyed = {k: v for k, v in picks.items() if k[0] == dataset}
            _, s, p = evaluate_series(sub, keyed, label)
            per_ds.append({"dataset": dataset, "regra": label, "smape": s, "pocid": p})
    per_ds = pd.DataFrame(per_ds)
    per_ds.to_csv(os.path.join(OUT_DIR, "resumo_top3_regras_por_dataset.csv"), index=False, sep=";")
    print("\n== regras principais por dataset ==")
    print(per_ds.pivot(index="regra", columns="dataset", values="smape").round(2).to_string())


if __name__ == "__main__":
    main()
