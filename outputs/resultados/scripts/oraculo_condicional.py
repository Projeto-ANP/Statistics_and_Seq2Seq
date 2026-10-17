"""Oraculo condicional aos pools que o agente usou/criou.

Para cada serie:
  U = pools referenciados nas tentativas do agente + pools criados por ele
      + pool_full, com todas as combinacoes usuais (mean, median, trimmed,
      dba, best_single, weighted com as receitas e grade de shrinkage).
  - oraculo de validacao: menor score de validacao em U (mesmo score do
    agente), incluindo tambem o aceite do agente e a melhor semente;
  - oraculo de teste: menor SMAPE / maior POCID no teste em U (hindsight),
    incluindo o agente e a semente.

Saida: outputs/resultados/oraculo_condicional/por_serie.csv
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "aprovacao_solida"))
sys.path.insert(0, os.path.join(_REPO, "aprovacao_solida", "01_busca_nao_llm"))
sys.path.insert(0, os.path.join(_REPO, "outputs", "resultados", "scripts"))

from _common import (  # noqa: E402
    DATASETS,
    DEFAULT_RESULTS_DIR,
    detect_drop,
    load_context,
    v5_config,
)
import common as C  # noqa: E402
from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402

OUT_DIR = os.path.join(_REPO, "outputs", "resultados", "oraculo_condicional")
TRACE_ROOT = os.path.join(DEFAULT_RESULTS_DIR, C.ORCHESTRATOR_GPTOSS, "llm_artifacts")
POOL_ACTIONS = ("select_stable", "select_top_k", "prune_redundant", "select_forward")


def parse_action(turn):
    act = turn.get("action") or ""
    raw = turn.get("raw_response") or ""
    args = None
    for x in re.findall(r"Action Input:\s*(\{.*?\})", raw, re.S):
        try:
            args = json.loads(x)
        except Exception:
            continue
    return act, args


def replay(st, trace):
    pools = set()
    for turn in trace.get("turns") or []:
        act, args = parse_action(turn)
        if not args:
            continue
        try:
            if act in ("select_stable", "select_top_k"):
                out = getattr(T, act)(st, k=args.get("k"), metric=args.get("metric", "rmse"))
                pools.add(out["pool"])
            elif act == "prune_redundant":
                out = T.prune_redundant(
                    st,
                    pool=args.get("pool", "pool_full"),
                    corr_threshold=args.get("corr_threshold", 0.95),
                    metric=args.get("metric", "rmse"),
                )
                pools.add(out["pool"])
            elif act == "select_forward":
                out = T.select_forward(
                    st,
                    max_steps=args.get("max_steps", 20),
                    metric=args.get("metric", "rmse"),
                )
                pools.add(out["pool"])
            elif act.startswith("weights_"):
                fn = getattr(T, act)
                fn(
                    st,
                    **{
                        k: v
                        for k, v in args.items()
                        if k in (
                            "pool",
                            "shrinkage",
                            "scope",
                            "metric",
                            "eta",
                            "damping",
                            "k",
                            "max_steps",
                            "per_horizon",
                        )
                    },
                )
        except Exception:
            continue
    return pools


def evaluate(st, ing, spec, records):
    try:
        attempt, _ = st.evaluate(spec, rationale="oraculo", origin="oracle")
        forecast, _ = st.apply_to_test(spec)
        metrics = compute_metrics(forecast, ing.test_values)
        records.append(
            (
                float(attempt.score),
                float(100.0 * metrics["smape"]),
                float(metrics["pocid"]),
            )
        )
    except Exception:
        pass


def enumerate_pools(st, ing, pools):
    records = []
    for pool in sorted(pools):
        try:
            evaluate(st, ing, T.combine_mean(st, pool)["strategy"], records)
            evaluate(st, ing, T.combine_median(st, pool)["strategy"], records)
            evaluate(st, ing, T.combine_dba(st, pool)["strategy"], records)
            for tp in (0.0, 0.1, 0.2, 0.3, 0.4):
                evaluate(
                    st,
                    ing,
                    T.combine_trimmed_mean(st, pool, trim_pct=tp)["strategy"],
                    records,
                )
            for sh in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                handle = T.weights_inverse_error(st, pool=pool, shrinkage=sh)["weights"]
                evaluate(
                    st,
                    ing,
                    T.combine_weighted(st, pool, weights=handle)["strategy"],
                    records,
                )
            for fn in (T.weights_error_trend, T.weights_best_per_horizon):
                handle = fn(st, pool=pool)["weights"]
                evaluate(
                    st,
                    ing,
                    T.combine_weighted(st, pool, weights=handle)["strategy"],
                    records,
                )
            handle = T.weights_topk_per_horizon(st, pool=pool, k=3)["weights"]
            evaluate(
                st,
                ing,
                T.combine_weighted(st, pool, weights=handle)["strategy"],
                records,
            )
            handle = T.weights_forward(st, pool=pool, max_steps=20)["weights"]
            evaluate(
                st,
                ing,
                T.combine_weighted(st, pool, weights=handle)["strategy"],
                records,
            )
        except Exception:
            continue
    for model in st.model_names:
        evaluate(st, ing, T.combine_best_single(st, model)["strategy"], records)
    return records


def best_seed_pool(baseline_row):
    params = baseline_row.get("best_strategy_params")
    if not isinstance(params, str):
        return None
    try:
        spec = json.loads(params)
    except Exception:
        return None
    return spec.get("pool")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "por_serie.csv"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    models = list(DEFAULT_MODELS)
    cfg = v5_config(no_pooled=True, name="oraculo")
    rows = []
    started = time.perf_counter()

    for dataset in args.datasets:
        agent = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        baseline = C.load_orchestrator(C.ORCHESTRATOR_BASELINE, dataset)
        agent = agent.set_index("dataset_index")
        baseline = baseline.set_index("dataset_index")

        frames, n_series, source = load_context(
            models, dataset, DEFAULT_RESULTS_DIR, DEFAULT_SOURCE_DIR, use_source=True
        )
        todo = list(range(n_series if args.limit is None else min(args.limit, n_series)))
        drop = detect_drop(models, dataset, todo[0], cfg, DEFAULT_RESULTS_DIR, frames)

        for idx in todo:
            t0 = time.perf_counter()
            try:
                ing = ingest_mod.load_series(
                    models=models,
                    dataset=dataset,
                    dataset_index=int(idx),
                    source=source,
                    config=cfg,
                    results_dir=DEFAULT_RESULTS_DIR,
                    frames=frames,
                    drop_models=drop,
                )
                st = ing.state
                st.pooled_meta_model = None
                pool_mod.run_phase2(st, cfg)

                trace_path = os.path.join(TRACE_ROOT, dataset, f"dataset_{idx}.json")
                trace = json.load(open(trace_path, encoding="utf-8"))["agent_trace"]
                pools = replay(st, trace)

                for attempt in trace.get("attempts") or []:
                    if attempt.get("origin") == "agent":
                        pool = (attempt.get("spec") or {}).get("pool")
                        if pool:
                            pools.add(pool)

                final_spec = (agent.loc[idx].description_json or {}).get("strategy") or {}
                if final_spec.get("pool"):
                    pools.add(final_spec["pool"])
                seed_pool = best_seed_pool(baseline.loc[idx])
                if seed_pool:
                    pools.add(seed_pool)
                pools.add("pool_full")

                records = enumerate_pools(st, ing, pools)

                agent_desc = agent.loc[idx].description_json or {}
                seed_desc = baseline.loc[idx].description_json or {}
                agent_val = float((agent_desc.get("validation") or {}).get("score"))
                seed_val = float((seed_desc.get("validation") or {}).get("score"))
                agent_smape = 100.0 * float(agent.loc[idx].smape)
                agent_pocid = float(agent.loc[idx].pocid)
                seed_smape = 100.0 * float(baseline.loc[idx].smape)
                seed_pocid = float(baseline.loc[idx].pocid)

                candidates = records + [
                    (agent_val, agent_smape, agent_pocid),
                    (seed_val, seed_smape, seed_pocid),
                ]
                oracle_val = min(candidates, key=lambda r: r[0])
                oracle_smape = min(candidates, key=lambda r: r[1])
                oracle_pocid = max(candidates, key=lambda r: r[2])

                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_index": int(idx),
                        "agent_smape": agent_smape,
                        "agent_pocid": agent_pocid,
                        "seed_smape": seed_smape,
                        "seed_pocid": seed_pocid,
                        "oracle_val_smape": oracle_val[1],
                        "oracle_val_pocid": oracle_val[2],
                        "oracle_test_smape": oracle_smape[1],
                        "oracle_test_pocid": oracle_pocid[2],
                        "agent_val_score": agent_val,
                        "seed_val_score": seed_val,
                        "oracle_val_score": oracle_val[0],
                        "n_candidates": len(records),
                        "pools": ",".join(sorted(pools)),
                    }
                )
            except Exception as exc:
                print(f"[{dataset} {idx}] FALHOU: {type(exc).__name__}: {exc}", flush=True)

            if not args.quiet:
                print(
                    f"[{dataset} {idx}] {time.perf_counter() - t0:.1f}s "
                    f"({len(rows)} series, {time.perf_counter() - started:.0f}s)",
                    flush=True,
                )

        pd.DataFrame(rows).to_csv(args.out, index=False, sep=";")
        print(f"parcial salvo em {args.out} ({len(rows)} series)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, sep=";")
    print(f"salvo em {args.out}", flush=True)

    summary = df.groupby("dataset")[
        [
            "agent_smape",
            "agent_pocid",
            "seed_smape",
            "seed_pocid",
            "oracle_val_smape",
            "oracle_val_pocid",
            "oracle_test_smape",
            "oracle_test_pocid",
        ]
    ].mean()
    summary.loc["GERAL"] = df[
        [
            "agent_smape",
            "agent_pocid",
            "seed_smape",
            "seed_pocid",
            "oracle_val_smape",
            "oracle_val_pocid",
            "oracle_test_smape",
            "oracle_test_pocid",
        ]
    ].mean()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
