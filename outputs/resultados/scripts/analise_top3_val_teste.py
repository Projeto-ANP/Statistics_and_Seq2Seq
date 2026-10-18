"""Top-3 na validacao x top-3 no teste, por serie, sobre o universo do oraculo.

Para cada serie, enumera as mesmas estrategias do oraculo condicional (pools do
agente + pool_full + pool da melhor semente) e registra, para cada candidata:
score de validacao, SMAPE e POCID no teste, familia da estrategia e ranks.

Depois:
  - mede a sobreposicao entre o top-3 da validacao e o top-3 do teste;
  - mede a correlacao de postos validacao x teste;
  - calcula regras de escolha e o resultado final de cada uma no teste:
      validacao       melhor score de validacao (oraculo de validacao)
      teste           melhor SMAPE/POCID no teste (oraculo de teste, hindsight)
      top3_val        melhor teste entre as tres melhores da validacao
      top3_intersec   melhor validacao entre as que estao no top-3 de ambos
      familia         melhor validacao da familia mais frequente no top-3 do teste
                      das OUTRAS series do mesmo dataset (leave-one-out)

Saidas em outputs/resultados/top3_val_teste/:
  por_serie.csv, resumo_por_dataset.csv, resumo.md
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "outputs", "resultados", "scripts"))

import oraculo_condicional as O  # noqa: E402

OUT_DIR = os.path.join(_REPO, "outputs", "resultados", "top3_val_teste")


def family_of(st, spec):
    combine = spec.get("combine", "?")
    if combine == "weighted":
        handle = spec.get("weights")
        try:
            method = st.get_weights_recipe(handle).method
        except Exception:
            method = "?"
        return f"weighted:{method}"
    return str(combine)


def enumerate_candidates(st, ing, pools):
    records = []

    def add(spec):
        try:
            attempt, _ = st.evaluate(spec, rationale="top3", origin="top3")
            forecast, _ = st.apply_to_test(spec)
            metrics = O.compute_metrics(forecast, ing.test_values)
            records.append(
                {
                    "val": float(attempt.score),
                    "smape": float(100.0 * metrics["smape"]),
                    "pocid": float(metrics["pocid"]),
                    "spec": spec,
                    "family": family_of(st, spec),
                    "attempt_id": attempt.attempt_id,
                    "per_window_scores": list(getattr(attempt, "per_window_scores", []) or []),
                    "per_window": getattr(attempt, "per_window", []) or [],
                }
            )
        except Exception:
            pass

    T = O.T
    for pool in sorted(pools):
        try:
            add(T.combine_mean(st, pool)["strategy"])
            add(T.combine_median(st, pool)["strategy"])
            add(T.combine_dba(st, pool)["strategy"])
            for tp in (0.0, 0.1, 0.2, 0.3, 0.4):
                add(T.combine_trimmed_mean(st, pool, trim_pct=tp)["strategy"])
            for sh in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                h = T.weights_inverse_error(st, pool=pool, shrinkage=sh)["weights"]
                add(T.combine_weighted(st, pool, weights=h)["strategy"])
            for fn in (T.weights_error_trend, T.weights_best_per_horizon):
                h = fn(st, pool=pool)["weights"]
                add(T.combine_weighted(st, pool, weights=h)["strategy"])
            h = T.weights_topk_per_horizon(st, pool=pool, k=3)["weights"]
            add(T.combine_weighted(st, pool, weights=h)["strategy"])
            h = T.weights_forward(st, pool=pool, max_steps=20)["weights"]
            add(T.combine_weighted(st, pool, weights=h)["strategy"])
        except Exception:
            continue
    for model in st.model_names:
        add(T.combine_best_single(st, model)["strategy"])
    return records


def series_rows(dataset, idx, agent_row, baseline_row):
    frames, _, source = series_rows.frames[dataset]
    drop = series_rows.drops[dataset]
    ing = O.ingest_mod.load_series(
        models=O.DEFAULT_MODELS,
        dataset=dataset,
        dataset_index=int(idx),
        source=source,
        config=O.v5_config(no_pooled=True, name="top3"),
        results_dir=O.DEFAULT_RESULTS_DIR,
        frames=frames,
        drop_models=drop,
    )
    st = ing.state
    st.pooled_meta_model = None
    O.pool_mod.run_phase2(st, O.v5_config(no_pooled=True, name="top3"))

    trace_path = os.path.join(O.TRACE_ROOT, dataset, f"dataset_{idx}.json")
    artifact = json.load(open(trace_path, encoding="utf-8"))
    trace = artifact["agent_trace"]
    series_card = artifact.get("series_card") or {}
    pool_card = artifact.get("pool_card") or {}
    diagnosis = artifact.get("diagnosis") or {}
    decision = trace.get("decision") or {}
    selection = decision.get("selection_confidence") or {}
    features = {
        "regime": diagnosis.get("regime"),
        "predictability": diagnosis.get("predictability"),
        "hint": diagnosis.get("combination_hint"),
        "risks": "|".join(diagnosis.get("risks") or []),
        "trend": series_card.get("trend_strength"),
        "seasonal": series_card.get("seasonal_strength"),
        "entropy": (series_card.get("features") or {}).get("spectral_entropy")
        if isinstance(series_card.get("features"), dict) else series_card.get("spectral_entropy"),
        "stability_tau": (pool_card.get("ranking_stability") or {}).get("mean_kendall_tau"),
        "stability_verdict": (pool_card.get("ranking_stability") or {}).get("verdict"),
        "spread": (pool_card.get("error_table") or {}).get("relative_spread"),
        "n_models": series_card.get("n_models"),
        "selection_verdict": selection.get("verdict"),
        "selection_margin": selection.get("margin"),
        "selection_winner": selection.get("winner"),
        "selection_runner_up": selection.get("runner_up"),
    }
    pools = O.replay(st, trace)
    for attempt in trace.get("attempts") or []:
        if attempt.get("origin") == "agent":
            pool = (attempt.get("spec") or {}).get("pool")
            if pool:
                pools.add(pool)
    final_spec = (agent_row.get("description_json") or {}).get("strategy") or {}
    if final_spec.get("pool"):
        pools.add(final_spec["pool"])
    seed_pool = O.best_seed_pool(pd.Series(baseline_row))
    if seed_pool:
        pools.add(seed_pool)
    pools.add("pool_full")

    records = enumerate_candidates(st, ing, pools)
    if not records:
        return None

    df = pd.DataFrame(records)
    df["val_rank"] = df["val"].rank(method="min", ascending=True)
    df["smape_rank"] = df["smape"].rank(method="min", ascending=True)
    df["pocid_rank"] = df["pocid"].rank(method="min", ascending=False)

    top3v = df[df["val_rank"] <= 3]
    top3s = df[df["smape_rank"] <= 3]
    top3p = df[df["pocid_rank"] <= 3]
    best_v = df.loc[df["val"].idxmin()]
    best_s = df.loc[df["smape"].idxmin()]
    best_p = df.loc[df["pocid"].idxmax()]

    inter_s = df[(df["val_rank"] <= 3) & (df["smape_rank"] <= 3)]
    inter_p = df[(df["val_rank"] <= 3) & (df["pocid_rank"] <= 3)]
    pick_inter_s = inter_s.loc[inter_s["val"].idxmin()] if len(inter_s) else best_v
    pick_inter_p = inter_p.loc[inter_p["val"].idxmin()] if len(inter_p) else best_v

    pick_top3_s = top3v.loc[top3v["smape"].idxmin()]
    pick_top3_p = top3v.loc[top3v["pocid"].idxmax()]

    sp_s = spearmanr(df["val"], df["smape"]).statistic
    sp_p = spearmanr(df["val"], df["pocid"]).statistic
    kt_s = kendalltau(df["val"], df["smape"]).statistic
    kt_p = kendalltau(df["val"], df["pocid"]).statistic

    agent_desc = agent_row.get("description_json") or {}
    seed_desc = baseline_row.get("description_json") or {}
    agent_val = float((agent_desc.get("validation") or {}).get("score", np.nan))
    seed_val = float((seed_desc.get("validation") or {}).get("score", np.nan))

    best_top3_smape = float(top3v["smape"].min())
    best_top3_pocid = float(top3v["pocid"].max())
    final_spec_key = json.dumps(final_spec, sort_keys=True, default=str)
    seed_spec = None
    try:
        seed_spec = json.loads(baseline_row.get("best_strategy_params"))
    except Exception:
        seed_spec = None
    seed_spec_key = json.dumps(seed_spec, sort_keys=True, default=str) if isinstance(seed_spec, dict) else ""
    agent_family = family_of(st, final_spec) if isinstance(final_spec, dict) and final_spec else ""
    seed_family = family_of(st, seed_spec) if isinstance(seed_spec, dict) and seed_spec else ""

    candidate_rows = []
    for _, cand in top3v.iterrows():
        spec = cand["spec"]
        pool = spec.get("pool")
        pool_size = len(st.get_pool(pool)) if pool else 1
        method = ""
        shrinkage = None
        if spec.get("combine") == "weighted":
            try:
                recipe = st.get_weights_recipe(spec.get("weights"))
                method = recipe.method
                shrinkage = (recipe.meta or {}).get("shrinkage")
            except Exception:
                pass
        scores = [float(x) for x in cand["per_window_scores"] if np.isfinite(float(x))] if cand["per_window_scores"] else []
        spec_key = json.dumps(spec, sort_keys=True, default=str)
        candidate_rows.append({
            "dataset": dataset,
            "dataset_index": int(idx),
            "candidate_id": cand["attempt_id"],
            "val": float(cand["val"]),
            "val_rank": float(cand["val_rank"]),
            "smape": float(cand["smape"]),
            "pocid": float(cand["pocid"]),
            "smape_rank": float(cand["smape_rank"]),
            "pocid_rank": float(cand["pocid_rank"]),
            "family": cand["family"],
            "combine": spec.get("combine"),
            "pool": pool,
            "pool_size": int(pool_size),
            "weights_method": method,
            "shrinkage": shrinkage,
            "trim_pct": spec.get("trim_pct"),
            "worst_window": max(scores) if scores else np.nan,
            "mean_window": float(np.mean(scores)) if scores else np.nan,
            "std_window": float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
            "val_gap_best": float(cand["val"] - best_v["val"]),
            "val_gap_pct": float((cand["val"] - best_v["val"]) / best_v["val"]) if best_v["val"] else np.nan,
            "is_test_best_smape": bool(abs(cand["smape"] - best_top3_smape) < 1e-12),
            "is_test_best_pocid": bool(abs(cand["pocid"] - best_top3_pocid) < 1e-12),
            "test_smape_regret": float(cand["smape"] - best_top3_smape),
            "test_pocid_regret": float(best_top3_pocid - cand["pocid"]),
            "agent_final": bool(spec_key == final_spec_key),
            "seed_final": bool(spec_key == seed_spec_key),
            "agent_family": agent_family,
            "seed_family": seed_family,
            "agent_smape": 100.0 * float(agent_row["smape"]),
            "agent_pocid": float(agent_row["pocid"]),
            "seed_smape": 100.0 * float(baseline_row["smape"]),
            "seed_pocid": float(baseline_row["pocid"]),
            **features,
        })

    row = {
        "dataset": dataset,
        "dataset_index": int(idx),
        "n_candidates": len(df),
        "best_val_id": best_v["attempt_id"],
        "best_val_family": best_v["family"],
        "best_val_score": best_v["val"],
        "best_val_smape": best_v["smape"],
        "best_val_pocid": best_v["pocid"],
        "best_val_smape_rank": best_v["smape_rank"],
        "best_val_pocid_rank": best_v["pocid_rank"],
        "best_test_smape": best_s["smape"],
        "best_test_smape_id": best_s["attempt_id"],
        "best_test_smape_family": best_s["family"],
        "best_test_smape_rank_val": best_s["val_rank"],
        "best_test_pocid": best_p["pocid"],
        "best_test_pocid_id": best_p["attempt_id"],
        "best_test_pocid_family": best_p["family"],
        "best_test_pocid_rank_val": best_p["val_rank"],
        "top3_val_ids": "|".join(top3v["attempt_id"].astype(str)),
        "top3_val_families": "|".join(top3v["family"].astype(str)),
        "top3_test_smape_ids": "|".join(top3s["attempt_id"].astype(str)),
        "top3_test_smape_families": "|".join(top3s["family"].astype(str)),
        "top3_test_pocid_ids": "|".join(top3p["attempt_id"].astype(str)),
        "top3_test_pocid_families": "|".join(top3p["family"].astype(str)),
        "overlap_val_smape": len(inter_s),
        "overlap_val_pocid": len(inter_p),
        "pick_inter_smape": pick_inter_s["smape"],
        "pick_inter_pocid": pick_inter_p["pocid"],
        "pick_inter_smape_family": pick_inter_s["family"],
        "pick_inter_pocid_family": pick_inter_p["family"],
        "pick_top3_smape": pick_top3_s["smape"],
        "pick_top3_pocid": pick_top3_p["pocid"],
        "pick_top3_smape_family": pick_top3_s["family"],
        "pick_top3_pocid_family": pick_top3_p["family"],
        "spearman_smape": sp_s,
        "spearman_pocid": sp_p,
        "kendall_smape": kt_s,
        "kendall_pocid": kt_p,
        "agent_smape": 100.0 * float(agent_row["smape"]),
        "agent_pocid": float(agent_row["pocid"]),
        "agent_val": agent_val,
        "seed_smape": 100.0 * float(baseline_row["smape"]),
        "seed_pocid": float(baseline_row["pocid"]),
        "seed_val": seed_val,
    }
    return row, df, candidate_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(O.C.DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "por_serie.csv"))
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    first_pass = {}
    all_candidates = []

    for dataset in args.datasets:
        agent = O.C.load_orchestrator(O.C.ORCHESTRATOR_GPTOSS, dataset).set_index("dataset_index")
        baseline = O.C.load_orchestrator(O.C.ORCHESTRATOR_BASELINE, dataset).set_index("dataset_index")
        frames, n_series, source = O.load_context(
            O.DEFAULT_MODELS, dataset, O.DEFAULT_RESULTS_DIR, O.DEFAULT_SOURCE_DIR, use_source=True
        )
        todo = list(range(n_series if args.limit is None else min(args.limit, n_series)))
        drop = O.detect_drop(O.DEFAULT_MODELS, dataset, todo[0], O.v5_config(no_pooled=True), O.DEFAULT_RESULTS_DIR, frames)
        series_rows.frames = {dataset: (frames, n_series, source)}
        series_rows.drops = {dataset: drop}

        print(f"\n== {dataset}: {len(todo)} series ==", flush=True)
        for idx in todo:
            out = series_rows(dataset, idx, agent.loc[idx], baseline.loc[idx])
            if out is None:
                continue
            row, df, candidate_rows = out
            first_pass[(dataset, int(idx))] = df
            rows.append(row)
            all_candidates.extend(candidate_rows)
            if len(rows) % 50 == 0:
                print(f"  {len(rows)} series", flush=True)

    df = pd.DataFrame(rows)

    # familia leave-one-out, por dataset, a partir do top-3 do teste das outras series
    fam_prior_s = {}
    fam_prior_p = {}
    for dataset in args.datasets:
        sub = df[df["dataset"] == dataset]
        for _, row in sub.iterrows():
            counts_s = Counter()
            counts_p = Counter()
            for _, other in sub.iterrows():
                if other["dataset_index"] == row["dataset_index"]:
                    continue
                counts_s.update(other["top3_test_smape_families"].split("|"))
                counts_p.update(other["top3_test_pocid_families"].split("|"))
            fam_prior_s[(dataset, int(row["dataset_index"]))] = counts_s.most_common(1)[0][0] if counts_s else ""
            fam_prior_p[(dataset, int(row["dataset_index"]))] = counts_p.most_common(1)[0][0] if counts_p else ""

    picks = []
    for _, row in df.iterrows():
        key = (row["dataset"], int(row["dataset_index"]))
        cand = first_pass[key]
        fam_s = fam_prior_s[key]
        fam_p = fam_prior_p[key]
        fam_rows_s = cand[cand["family"] == fam_s]
        fam_rows_p = cand[cand["family"] == fam_p]
        pick_s = fam_rows_s.loc[fam_rows_s["val"].idxmin()] if len(fam_rows_s) else cand.loc[cand["val"].idxmin()]
        pick_p = fam_rows_p.loc[fam_rows_p["val"].idxmin()] if len(fam_rows_p) else cand.loc[cand["val"].idxmin()]
        picks.append(
            {
                "family_prior_smape": fam_s,
                "family_prior_pocid": fam_p,
                "pick_family_smape": pick_s["smape"],
                "pick_family_pocid": pick_p["pocid"],
                "pick_family_smape_name": pick_s["family"],
                "pick_family_pocid_name": pick_p["family"],
            }
        )
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(picks)], axis=1)

    df.to_csv(args.out, index=False, sep=";")
    pd.DataFrame(all_candidates).to_csv(
        os.path.join(OUT_DIR, "top3_candidatos.csv"), index=False, sep=";"
    )
    print(f"\nsalvo em {args.out} ({len(df)} series)", flush=True)
    print(f"top3 candidatos: {len(all_candidates)} linhas", flush=True)

    metrics = {
        "validation": ("best_val_smape", "best_val_pocid"),
        "test": ("best_test_smape", "best_test_pocid"),
        "top3_val": ("pick_top3_smape", "pick_top3_pocid"),
        "top3_intersec": ("pick_inter_smape", "pick_inter_pocid"),
        "familia": ("pick_family_smape", "pick_family_pocid"),
        "agente": ("agent_smape", "agent_pocid"),
        "semente": ("seed_smape", "seed_pocid"),
    }
    resumo = []
    for name, (cs, cp) in metrics.items():
        resumo.append(
            {
                "regra": name,
                "smape": df[cs].mean(),
                "pocid": df[cp].mean(),
            }
        )
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(os.path.join(OUT_DIR, "resumo_regras.csv"), index=False, sep=";")
    print("\n== regras (teste) ==")
    print(resumo.round(2).to_string(index=False))

    rule_cols = [cs for cs, _ in metrics.values()] + [cp for _, cp in metrics.values()]
    per_ds = df.groupby("dataset")[rule_cols].mean()
    per_ds.to_csv(os.path.join(OUT_DIR, "resumo_por_dataset.csv"), sep=";")
    print("\n== regras por dataset ==")
    print(per_ds.round(2).to_string())

    print("\n== cobertura do melhor teste pelo rank de validacao ==")
    for metric, rankcol in (("SMAPE", "best_test_smape_rank_val"), ("POCID", "best_test_pocid_rank_val")):
        r = df[rankcol]
        print(
            metric,
            {k: round(float((r <= k).mean()), 3) for k in (1, 3, 5, 10)},
        )
    print("\ncorrelacao media (validacao x teste):")
    print(
        "spearman smape",
        round(float(df["spearman_smape"].mean()), 3),
        "| spearman pocid",
        round(float(df["spearman_pocid"].mean()), 3),
        "| kendall smape",
        round(float(df["kendall_smape"].mean()), 3),
        "| kendall pocid",
        round(float(df["kendall_pocid"].mean()), 3),
    )
    print("\nsobreposicao top-3:")
    print(
        "intersecao smape: media",
        round(float(df["overlap_val_smape"].mean()), 2),
        "| nao vazia",
        round(float((df["overlap_val_smape"] > 0).mean()), 3),
    )
    print(
        "intersecao pocid: media",
        round(float(df["overlap_val_pocid"].mean()), 2),
        "| nao vazia",
        round(float((df["overlap_val_pocid"] > 0).mean()), 3),
    )
    print("\nfamilias mais frequentes no top-3:")
    for col, label in (
        ("top3_val_families", "validacao"),
        ("top3_test_smape_families", "teste SMAPE"),
        ("top3_test_pocid_families", "teste POCID"),
    ):
        c = Counter()
        for x in df[col]:
            c.update(str(x).split("|"))
        print(label, c.most_common(8))
    print("\nprior de familia (leave-one-out):")
    print("smape", Counter(df["family_prior_smape"]).most_common(8))
    print("pocid", Counter(df["family_prior_pocid"]).most_common(8))

    with open(os.path.join(OUT_DIR, "resumo.md"), "w", encoding="utf-8") as fh:
        fh.write("# Top-3 validacao x top-3 teste\n\n")
        fh.write(resumo.round(2).to_string(index=False))
        fh.write("\n\n")
        fh.write(per_ds.round(2).to_string())
        fh.write("\n")


if __name__ == "__main__":
    main()
