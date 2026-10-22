"""Search new combination recipes for ETTM (and optionally other datasets).

The agent's v10x search concentrated on mean/median/trimmed/dba and
weighted(inverse_error|softmax) with the default RMSE. This script evaluates a
larger catalog on the SAME validation/test protocol:

* SMAPE-based inverse/softmax weights;
* best/top-k per horizon by SMAPE;
* Caruana forward selection and forward-selected pools;
* top-k/stable/prune pools selected by SMAPE;
* winsorized mean, quantile combinations and simple blends.

For each recipe it records the validation score and the blinded test SMAPE/POCID,
checks whether the agent's history contains the same test forecast, and then
compares: agent argmin, FFORMA, validation top1, validation top3 oracle and the
all-recipe oracle.

The test actual is used only to evaluate the recipe, never to construct it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator_react import combiners as C  # noqa: E402
from orchestrator_react import ingest as ing  # noqa: E402
from orchestrator_react import metrics as M  # noqa: E402
from orchestrator_react.config import ReactConfig  # noqa: E402
from orchestrator_react.data_source import load_series_source  # noqa: E402
from orchestrator_react.selection import (  # noqa: E402
    PoolRecipe,
    forward_indices,
    prune_redundant_indices,
    stable_indices,
    top_k_indices,
)
from orchestrator_react.state import FULL_POOL, ReactState  # noqa: E402
from orchestrator_react.weighting import WeightsRecipe  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402
from run_tsf_batch import DATASET_SOURCES, DEFAULT_SOURCE_DIR  # noqa: E402

from replay_final_choice import _attempts_from_artifact, _artifact_indices, _load_artifact  # noqa: E402


def _load_state(dataset: str, index: int, results_dir: str, source_dir: str, cache: Dict[str, Any]):
    if dataset not in cache:
        models = list(DEFAULT_MODELS)
        frames = ing.load_dataset_frames(models, dataset, results_dir=results_dir)
        try:
            bad = ing.find_misaligned_models(models, dataset, 0, results_dir=results_dir, frames=frames, n_windows=3)
        except Exception:
            bad = {}
        drop = sorted(bad)
        source = load_series_source(
            DATASET_SOURCES[dataset],
            n_expected_series=ing.count_series(dataset, models[0], results_dir=results_dir),
            source_dir=source_dir,
        )
        cfg = ReactConfig(score_preset="smape_only", score_window_mode="weighted")
        cache[dataset] = (models, frames, source, drop, cfg)
    models, frames, source, drop, cfg = cache[dataset]
    return ing.load_series(
        models, dataset, int(index), source=source, config=cfg,
        results_dir=results_dir, frames=frames, drop_models=drop,
    )


def _register_pool(state: ReactState, kind: str, metric: str, k: int = 5, max_steps: int = 20,
                    corr: float = 0.95) -> str:
    if kind == "top_k":
        idx = top_k_indices(state.y_true, state.y_preds, k, metric)
        recipe = PoolRecipe(method="top_k", params={"k": int(k), "metric": metric})
    elif kind == "stable":
        idx = stable_indices(state.y_true, state.y_preds, k, metric)
        recipe = PoolRecipe(method="stable", params={"k": int(k), "metric": metric})
    elif kind == "forward":
        idx = forward_indices(state.y_true, state.y_preds, max_steps=max_steps, metric=metric)
        recipe = PoolRecipe(method="forward", params={"max_steps": int(max_steps), "metric": metric})
    elif kind == "prune":
        idx = prune_redundant_indices(
            state.y_true, state.y_preds, list(range(state.n_models)),
            corr_threshold=corr, metric=metric,
        )
        recipe = PoolRecipe(method="prune_redundant", params={"corr_threshold": float(corr), "metric": metric})
    else:
        raise ValueError(kind)
    return state.register_pool(idx, origin=f"search_{kind}_{metric}", recipe=recipe, k=len(idx))


def _register_weights(state: ReactState, method: str, pool: str, metric: str,
                       per_horizon: bool = False, **params: Any) -> str:
    recipe = WeightsRecipe(
        method=method,
        pool_handle=pool,
        per_horizon=bool(per_horizon),
        params={"metric": metric, **params},
    )
    return state.register_weights(recipe)


def _score_custom(state: ReactState, val_forecast: np.ndarray) -> float:
    agg, per_window = state._score_metrics(val_forecast)
    weights = state.active_score_weights()
    anchor = state.baseline_per_window()
    per_window_scores = [
        float(M.composite_score(per_window[i], anchor[i], weights))
        for i in range(min(len(per_window), len(anchor)))
    ]
    return float(state._window_score(per_window_scores))


def _test_metrics(forecast: np.ndarray, actual: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if forecast is None or forecast.size != actual.size:
        return None, None
    sm = float(M.smape(actual, forecast))
    po = float(M.pocid(actual, forecast))
    return (sm if np.isfinite(sm) else None, po if np.isfinite(po) else None)


def _same_forecast(a: np.ndarray, b: np.ndarray) -> bool:
    if a is None or b is None or a.size != b.size:
        return False
    return bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))


def _build_catalog(state: ReactState, series_card: Dict[str, Any]):
    specs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    pools: Dict[str, str] = {FULL_POOL: FULL_POOL}

    def add_pool(name: str, kind: str, metric: str, **kw):
        handle = _register_pool(state, kind, metric, **kw)
        pools[name] = handle
        return handle

    for metric in ("smape", "rmse"):
        for k in (5, 7, 9):
            add_pool(f"top{k}_{metric}", "top_k", metric, k=k)
        for k in (5, 7):
            add_pool(f"stable{k}_{metric}", "stable", metric, k=k)
        for steps in (10, 20):
            add_pool(f"fwd{steps}_{metric}", "forward", metric, max_steps=steps)
        for corr in (0.9, 0.95):
            add_pool(f"prune{corr}_{metric}", "prune", metric, corr=corr)

    def add_spec(label: str, spec: Dict[str, Any], meta: Optional[Dict[str, Any]] = None):
        specs.append((label, spec, meta or {}))

    for combine in ("mean", "median"):
        add_spec(f"{combine}|pool_full", {"combine": combine, "pool": FULL_POOL},
                 {"pool_name": FULL_POOL, "weights_method": None})
    for trim in (0.1, 0.2, 0.3):
        add_spec(f"trimmed_mean|pool_full|trim={trim}",
                 {"combine": "trimmed_mean", "pool": FULL_POOL, "trim_pct": trim},
                 {"pool_name": FULL_POOL, "weights_method": None, "weights_params": {"trim_pct": trim}})
    add_spec("dba|pool_full", {"combine": "dba", "pool": FULL_POOL, "dba_max_iter": 30},
             {"pool_name": FULL_POOL, "weights_method": None})

    for pname, handle in pools.items():
        if pname == FULL_POOL:
            continue
        for combine in ("mean", "median", "trimmed_mean"):
            spec = {"combine": combine, "pool": handle}
            label = f"{combine}|{pname}"
            meta = {"pool_name": pname, "weights_method": None}
            if combine == "trimmed_mean":
                spec["trim_pct"] = 0.2
                label += "|trim=0.2"
                meta["weights_params"] = {"trim_pct": 0.2}
            add_spec(label, spec, meta)
        w = _register_weights(state, "inverse_error", handle, "smape", power=1.0)
        add_spec(
            f"weighted|{pname}|inverse_error(metric=smape,power=1.0,shrinkage=0.0)",
            {"combine": "weighted", "pool": handle, "weights": w},
            {"pool_name": pname, "weights_method": "inverse_error",
             "weights_params": {"metric": "smape", "power": 1.0, "shrinkage": 0.0}},
        )

    for pname in (FULL_POOL, "top7_smape"):
        handle = pools[pname]
        for metric in ("smape", "rmse"):
            for power in (1.0, 2.0):
                for shrink in (0.0, 0.1):
                    w = _register_weights(state, "inverse_error", handle, metric,
                                          power=power, shrinkage=shrink)
                    add_spec(
                        f"weighted|{pname}|inverse_error(metric={metric},power={power},shrinkage={shrink})",
                        {"combine": "weighted", "pool": handle, "weights": w},
                        {"pool_name": pname, "weights_method": "inverse_error",
                         "weights_params": {"metric": metric, "power": power, "shrinkage": shrink}},
                    )
            for eta in (1.0, 3.0, 5.0):
                w = _register_weights(state, "softmax_neg_error", handle, metric, eta=eta)
                add_spec(
                    f"weighted|{pname}|softmax_neg_error(metric={metric},eta={eta})",
                    {"combine": "weighted", "pool": handle, "weights": w},
                    {"pool_name": pname, "weights_method": "softmax_neg_error",
                     "weights_params": {"metric": metric, "eta": eta}},
                )
            w = _register_weights(state, "best_per_horizon", handle, metric, per_horizon=True)
            add_spec(
                f"weighted|{pname}|best_per_horizon(metric={metric})",
                {"combine": "weighted", "pool": handle, "weights": w},
                {"pool_name": pname, "weights_method": "best_per_horizon",
                 "weights_params": {"metric": metric}},
            )
            for k in (3, 5):
                w = _register_weights(state, "topk_mean_per_horizon", handle, metric,
                                      per_horizon=True, k=k)
                add_spec(
                    f"weighted|{pname}|topk_mean_per_horizon(metric={metric},k={k})",
                    {"combine": "weighted", "pool": handle, "weights": w},
                    {"pool_name": pname, "weights_method": "topk_mean_per_horizon",
                     "weights_params": {"metric": metric, "k": k}},
                )
            for steps in (10, 20):
                w = _register_weights(state, "forward_selection", handle, metric, max_steps=steps)
                add_spec(
                    f"weighted|{pname}|forward_selection(metric={metric},max_steps={steps})",
                    {"combine": "weighted", "pool": handle, "weights": w},
                    {"pool_name": pname, "weights_method": "forward_selection",
                     "weights_params": {"metric": metric, "max_steps": steps}},
                )
        w = _register_weights(state, "error_trend", handle, "smape")
        add_spec(
            f"weighted|{pname}|error_trend(metric=smape)",
            {"combine": "weighted", "pool": handle, "weights": w},
            {"pool_name": pname, "weights_method": "error_trend",
             "weights_params": {"metric": "smape"}},
        )

    custom: List[Tuple[str, str, Any, Dict[str, Any]]] = []
    for pname in (FULL_POOL, "top7_smape"):
        custom.append((f"winsor10|{pname}", pname, lambda p, q=0.10: _winsor(p, q),
                       {"pool_name": pname, "weights_method": None}))
        custom.append((f"winsor05|{pname}", pname, lambda p, q=0.05: _winsor(p, q),
                       {"pool_name": pname, "weights_method": None}))
        for q in (0.40, 0.45, 0.55, 0.60):
            custom.append((f"quantile{q}|{pname}", pname,
                           lambda p, q=q: np.nanquantile(p, q, axis=0),
                           {"pool_name": pname, "weights_method": None, "weights_params": {"q": q}}))
    custom.append(("blend_mean_median|pool_full", FULL_POOL,
                   lambda p: (C.combine_mean(p) + C.combine_median(p)) / 2.0,
                   {"pool_name": FULL_POOL, "weights_method": None}))
    for key, label in (("trend_champion", "trend"), ("seasonality_champion", "season")):
        info = (series_card or {}).get(key) or {}
        model = info.get("model")
        if model in state.model_names:
            idx = state.model_names.index(model)
            def _champ_blend(p, idx=idx):
                inv = np.ones(p.shape[0]) / p.shape[0]
                return 0.5 * p[idx, :] + 0.5 * C.combine_weighted(p, inv)
            custom.append((f"champ_{label}|pool_full", FULL_POOL, _champ_blend,
                           {"pool_name": FULL_POOL, "weights_method": "champion",
                            "weights_params": {"model": model}}))
    return specs, custom, pools


def _winsor(preds: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(preds, dtype=float)
    lo = np.nanquantile(p, q, axis=0)
    hi = np.nanquantile(p, 1.0 - q, axis=0)
    return np.nanmean(np.clip(p, lo[None, :], hi[None, :]), axis=0)


def _eval_custom(state: ReactState, idx: Sequence[int], fn) -> Tuple[float, np.ndarray, np.ndarray]:
    val = np.stack([fn(state.y_preds[i][idx, :]) for i in range(state.n_windows)])
    score = _score_custom(state, val)
    test = fn(state.test_preds[idx, :])
    return score, val, test


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["ETTM1", "ETTM2"])
    parser.add_argument("--run", default="orchestrator_react_v10x_low")
    parser.add_argument("--results-dir", default="./timeseries/mestrado/resultados")
    parser.add_argument("--source-dir", default="../forecasting_datasets")
    parser.add_argument("--output-dir", default="outputs/resultados/v11_combo_search")
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    cache: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        indices = _artifact_indices(args.run, dataset, args.results_dir)
        print(f"[{dataset}] {len(indices)} series", flush=True)
        fforma_path = os.path.join(args.results_dir, "FFORMA", f"{dataset}.csv")
        fforma_map = {}
        if os.path.exists(fforma_path):
            fdf = pd.read_csv(fforma_path, sep=";")
            fforma_map = {int(r.dataset_index): float(r.smape) for _, r in fdf.drop_duplicates("dataset_index").iterrows()}
        for index in indices:
            payload = _load_artifact(args.run, dataset, index, args.results_dir)
            if payload is None:
                continue
            attempts = _attempts_from_artifact(payload)
            if not attempts:
                continue
            try:
                ingested = _load_state(dataset, index, args.results_dir, args.source_dir, cache)
            except Exception as exc:
                print(f"  {dataset} {index}: load failed: {exc}")
                continue
            state = ingested.state
            actual = np.asarray(ingested.test_values, dtype=float)
            card = payload.get("series_card") or {}
            leader = min(attempts, key=lambda a: a.score)
            agent_forecasts = [np.asarray(a.test_forecast) for a in attempts if a.test_forecast]
            specs, custom, pools = _build_catalog(state, card)

            def record(name: str, kind: str, spec: Dict[str, Any], val_score: float,
                       test_forecast: np.ndarray, meta: Optional[Dict[str, Any]] = None,
                       extra: Optional[Dict[str, Any]] = None):
                sm, po = _test_metrics(test_forecast, actual)
                row = {
                    "run": args.run, "dataset": dataset, "series": int(index),
                    "recipe": name, "kind": kind,
                    "combine": spec.get("combine"), "pool": spec.get("pool"),
                    "weights": spec.get("weights"), "val_score": float(val_score),
                    "test_smape": sm, "test_pocid": po,
                    "agent_tried": any(_same_forecast(test_forecast, af) for af in agent_forecasts),
                    "agent_argmin_smape": leader.test_smape,
                    "fforma_smape": fforma_map.get(int(index)),
                }
                if meta:
                    row.update(meta)
                if extra:
                    row.update(extra)
                rows.append(row)

            # agent argmin baseline
            record("agent_argmin", "agent", leader.spec, leader.score,
                   np.asarray(leader.test_forecast), {"pool_name": "agent"}, {"agent_tried": True})
            # existing specs
            for label, spec, meta in specs:
                try:
                    attempt, _ = state.evaluate(spec)
                    fc, _ = state.apply_to_test(spec)
                    record(label, "existing", spec, float(attempt.score), np.asarray(fc), meta)
                except Exception as exc:
                    print(f"  {dataset} {index}: spec failed {label}: {type(exc).__name__}: {exc}")
            # custom specs
            for label, pname, fn, meta in custom:
                idx = state.get_pool(pools[pname])
                try:
                    score, val, test = _eval_custom(state, idx, fn)
                    record(label, "custom", {"combine": label, "pool": pname}, score, test, meta)
                except Exception as exc:
                    print(f"  {dataset} {index}: custom failed {label}: {type(exc).__name__}: {exc}")

    D = pd.DataFrame(rows)
    if D.empty:
        print("no rows")
        return 1
    per_series_path = os.path.join(args.output_dir, f"per_series_{args.run}.csv")
    D.to_csv(per_series_path, sep=";", index=False)

    # per-recipe summary
    summary_rows = []
    for recipe, group in D.groupby("recipe"):
        val = pd.to_numeric(group["val_score"], errors="coerce")
        test = pd.to_numeric(group["test_smape"], errors="coerce")
        summary_rows.append({
            "recipe": recipe,
            "kind": group["kind"].iloc[0],
            "n": len(group),
            "val_mean": float(val.mean()),
            "test_mean": float(test.mean()),
            "test_pocid_mean": float(pd.to_numeric(group["test_pocid"], errors="coerce").mean()),
            "agent_tried_rate": float(group["agent_tried"].astype(str).str.lower().isin(["true", "1"]).mean()),
            "test_best_count": 0,
            "val_top1_count": 0,
            "val_top3_count": 0,
        })
    S = pd.DataFrame(summary_rows).set_index("recipe")

    top1_rows = []
    for (dataset, series), group in D.groupby(["dataset", "series"]):
        g = group.copy()
        g["val_score"] = pd.to_numeric(g["val_score"], errors="coerce")
        g["test_smape"] = pd.to_numeric(g["test_smape"], errors="coerce")
        g = g[g["val_score"].notna() & g["test_smape"].notna()]
        if g.empty:
            continue
        by_val = g.sort_values("val_score")
        top1 = by_val.iloc[0]
        top3 = by_val.head(3)
        best_test = g.sort_values("test_smape").iloc[0]
        S.loc[top1["recipe"], "val_top1_count"] += 1
        for rec in top3["recipe"]:
            S.loc[rec, "val_top3_count"] += 1
        S.loc[best_test["recipe"], "test_best_count"] += 1
        top1_rows.append({
            "dataset": dataset, "series": int(series),
            "val_top1_recipe": top1["recipe"], "val_top1_test": float(top1["test_smape"]),
            "val_top1_agent_tried": top1["agent_tried"],
            "top3_oracle_test": float(top3["test_smape"].min()),
            "all_oracle_test": float(best_test["test_smape"]),
            "best_test_recipe": best_test["recipe"],
            "best_test_val_rank": int((by_val["recipe"] == best_test["recipe"]).idxmax() if False else np.where(by_val["recipe"].values == best_test["recipe"])[0][0] + 1),
            "agent_argmin": float(top1["agent_argmin_smape"]),
            "fforma": float(top1["fforma_smape"]) if pd.notna(top1["fforma_smape"]) else np.nan,
        })
    T = pd.DataFrame(top1_rows)
    if not T.empty:
        T.to_csv(os.path.join(args.output_dir, f"selection_{args.run}.csv"), sep=";", index=False)
        print("\nSELECTION SUMMARY")
        print(f"  agent argmin      : {T['agent_argmin'].mean():.6f}")
        print(f"  FFORMA            : {T['fforma'].mean():.6f}")
        print(f"  validation top1   : {T['val_top1_test'].mean():.6f}")
        print(f"  top3 oracle       : {T['top3_oracle_test'].mean():.6f}")
        print(f"  all-recipe oracle : {T['all_oracle_test'].mean():.6f}")
        print("\n  recipes most often validation top1:")
        print(T["val_top1_recipe"].value_counts().head(10).to_string())
        print("\n  recipes most often test-best:")
        print(T["best_test_recipe"].value_counts().head(10).to_string())

    S = S.sort_values("test_mean")
    S.to_csv(os.path.join(args.output_dir, f"recipes_{args.run}.csv"), sep=";")
    print("\nTOP RECIPES BY TEST MEAN")
    print(S.head(25)[["kind", "val_mean", "test_mean", "test_pocid_mean",
                      "agent_tried_rate", "val_top1_count", "val_top3_count",
                      "test_best_count"]].round(6).to_string())
    print("\nTOP RECIPES BY VALIDATION MEAN")
    print(S.sort_values("val_mean").head(25)[["kind", "val_mean", "test_mean",
                                              "agent_tried_rate", "val_top1_count",
                                              "val_top3_count", "test_best_count"]].round(6).to_string())
    print(f"\nSaved: {per_series_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
