#!/usr/bin/env python3
"""Replay offline da decisao final: vencedor vs melhor semente, e regras de gate.

Pergunta que responde, sem rodar LLM de novo: o vencedor aplicado pelos runs
(rejeitando o empate com a melhor semente) realmente ganha da melhor semente na
JANELA DE TESTE? E o que aconteceria se a decisao final fosse:

    argmin            o vencedor, como esta hoje;
    seed              a melhor semente sempre;
    margin_X          a semente, a menos que o vencedor melhore o score de
                      validacao em X% (X = 0.005, 0.01, 0.02, 0.05);
    verdict_separated o vencedor so se a selecao for "separated" (bootstrap+DM)
                      no artefato da serie; senao, a semente.

Nada aqui le a janela de teste para decidir: a decisao e tomada com score de
VALIDACAO (do vencedor, do JSON `description`, e da semente, recalculada pelo
mesmo pre-passo determinístico). O teste so mede o resultado das regras.

Uso:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \\
        --run orchestrator_react_v5 --limit 20     # smoke
    # depois, sem --limit, para todos os datasets:
    nohup python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \\
        --run orchestrator_react_v5 > logs/replay_gate_v5.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import (  # noqa: E402
    DATASETS,
    DEFAULT_RESULTS_DIR,
    detect_drop,
    load_context,
    load_dataset_csv,
    v5_config,
)

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402


def _artifact_verdict(row):
    path = row.get("llm_artifacts_path")
    if not isinstance(path, str) or not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        conf = ((payload.get("decision") or {}).get("selection_confidence") or {})
        return str(conf.get("verdict") or "")
    except Exception:
        return ""


def _apply(state, spec):
    forecast, _ = state.apply_to_test(spec)
    return np.asarray(forecast, dtype=float)


def process_series(row, models, dataset, cfg, results_dir, frames, source, drop, margins, use_artifacts, meta_models=None):
    idx = int(row["dataset_index"])
    description = json.loads(row["description"]) if isinstance(row.get("description"), str) else {}
    # v5 grava o `decision()` novo: o score do vencedor esta em validation.score;
    # runs antigos podiam trazer um "score" no topo. Tenta os dois.
    raw_score = description.get("score")
    if raw_score is None:
        raw_score = (description.get("validation") or {}).get("score")
    try:
        winner_score = float(raw_score)
    except (TypeError, ValueError):
        winner_score = float("nan")
    winner = {
        "smape": float(row.get("smape", float("nan"))),
        "pocid": float(row.get("pocid", float("nan"))),
        "rmse": float(row.get("rmse", float("nan"))),
    }
    winner_pred = np.asarray(json.loads(row["predictions"]), dtype=float)

    ingested = ingest_mod.load_series(
        models=models, dataset=dataset, dataset_index=idx, source=source,
        config=cfg, results_dir=results_dir, frames=frames, drop_models=drop,
    )
    state = ingested.state
    if meta_models:
        state.pooled_meta_model = meta_models.get(idx)
    pool_mod.run_phase2(state, cfg)
    seeds = [a for a in state.attempts if a.origin == "baseline" and np.isfinite(a.score)]
    if not seeds:
        return None
    best_seed = min(seeds, key=lambda a: a.score)
    seed_pred = _apply(state, best_seed.spec)
    seed_metrics = compute_metrics(seed_pred, ingested.test_values)

    same_forecast = bool(
        winner_pred.size == seed_pred.size
        and np.allclose(winner_pred, seed_pred, rtol=1e-6, atol=1e-6, equal_nan=True)
    )

    rel_gain = (
        float((best_seed.score - winner_score) / abs(best_seed.score))
        if np.isfinite(winner_score) and np.isfinite(best_seed.score) and best_seed.score != 0
        else float("nan")
    )
    verdict = _artifact_verdict(row) if use_artifacts else ""

    picks = {"argmin": "winner", "seed": "seed", "verdict_separated": "seed"}
    for m in margins:
        picks[f"margin_{m:g}"] = (
            "winner" if np.isfinite(rel_gain) and rel_gain >= float(m) else "seed"
        )
    if verdict == "separated":
        picks["verdict_separated"] = "winner"

    out = {
        "dataset": dataset,
        "dataset_index": idx,
        "winner_score": winner_score,
        "seed_score": float(best_seed.score),
        "rel_gain": rel_gain,
        "verdict": verdict,
        "same_forecast": same_forecast,
        "winner_smape": winner["smape"],
        "winner_pocid": winner["pocid"],
        "winner_rmse": winner["rmse"],
        "seed_smape": seed_metrics["smape"],
        "seed_pocid": seed_metrics["pocid"],
        "seed_rmse": seed_metrics["rmse"],
        "seed_strategy": json.dumps(best_seed.spec, ensure_ascii=False, default=str),
    }
    for rule, pick in picks.items():
        out[f"{rule}_pick"] = pick
        out[f"{rule}_smape"] = winner["smape"] if pick == "winner" else seed_metrics["smape"]
        out[f"{rule}_pocid"] = winner["pocid"] if pick == "winner" else seed_metrics["pocid"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Replay offline vencedor vs semente.")
    p.add_argument("--run", default="orchestrator_react_v5", help="pasta do run no results-dir")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--margins", default="0.005,0.01,0.02,0.05")
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-artifacts", action="store_true",
                   help="nao le o veredito de selecao dos artefatos")
    p.add_argument("--no-source", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    margins = [float(x) for x in args.margins.split(",") if x.strip()]
    cfg = v5_config(no_pooled=False, name=f"replay_{args.run}")

    out_path = args.out or os.path.join(
        "outputs", "resultados", "aprovacao_solida", f"replay_gate_{args.run}.csv"
    )

    all_rows = []
    for dataset in args.datasets:
        df = load_dataset_csv(args.run, dataset, args.results_dir)
        if df is None or df.empty:
            if not args.quiet:
                print(f"{dataset}: sem CSV em {args.run}, pulando")
            continue
        frames, _n_series, source = load_context(
            models, dataset, args.results_dir, args.source_dir,
            use_source=not args.no_source,
        )
        todo = df.copy()
        if args.indices is not None:
            todo = todo[todo["dataset_index"].astype(int).isin(args.indices)]
        if args.limit:
            todo = todo.head(int(args.limit))
        if todo.empty:
            continue
        first = int(todo["dataset_index"].astype(int).iloc[0])
        drop = detect_drop(models, dataset, first, cfg, args.results_dir, frames)
        meta_models = {}
        if cfg.pooled_meta_model:
            meta_models = pipeline_mod._build_pooled_meta_models(
                models, dataset, [int(i) for i in todo["dataset_index"].astype(int)],
                config=cfg, source=source, results_dir=args.results_dir,
                frames=frames, drop_models=drop,
            )
            if not meta_models and not args.quiet:
                print(f"{dataset}: pooled meta-model nao treinado; piso efetivo com 9 sementes")
        rows, anchor = [], time.perf_counter()
        for _pos, row in todo.iterrows():
            try:
                out = process_series(
                    row, models, dataset, cfg, args.results_dir, frames, source,
                    drop, margins, not args.no_artifacts, meta_models=meta_models,
                )
                if out:
                    rows.append(out)
                    if not args.quiet:
                        print(
                            f"[{dataset} {out['dataset_index']:>4}] "
                            f"w={out['winner_smape']:.4f} s={out['seed_smape']:.4f} "
                            f"rel_gain={out['rel_gain']:+.3f} same={out['same_forecast']} "
                            f"verdict={out['verdict']} ({time.perf_counter()-anchor:.0f}s)",
                            flush=True,
                        )
                        anchor = time.perf_counter()
            except Exception as exc:
                print(f"[{dataset} {row.get('dataset_index')}] FALHOU: {type(exc).__name__}: {exc}", flush=True)
        if not rows:
            continue
        all_rows.extend(rows)
        sub = pd.DataFrame(rows)
        summary = {
            "dataset": dataset,
            "n": len(sub),
            "same_forecast": int(sub["same_forecast"].sum()),
            "seed_beats_winner_test": int((sub["seed_smape"] < sub["winner_smape"] - 1e-6).sum()),
            "winner_beats_seed_test": int((sub["winner_smape"] < sub["seed_smape"] - 1e-6).sum()),
        }
        for rule in ["argmin", "seed"] + [f"margin_{m:g}" for m in margins] + ["verdict_separated"]:
            if f"{rule}_smape" in sub.columns:
                summary[f"{rule}_smape"] = float(sub[f"{rule}_smape"].mean())
                summary[f"{rule}_pocid"] = float(sub[f"{rule}_pocid"].mean())
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if all_rows:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_path, sep=";", index=False)
        print(f"\npor serie: {out_path}")
        full = pd.DataFrame(all_rows)
        agg = full.groupby("dataset")[
            [c for c in full.columns if c.endswith("_smape") or c.endswith("_pocid")]
        ].mean()
        agg_path = out_path.replace(".csv", "_por_dataset.csv")
        agg.to_csv(agg_path, sep=";")
        print(f"por dataset: {agg_path}")
        print("\n=== SMAPE medio por regra de decisao ===")
        print(agg[[c for c in agg.columns if c.endswith("_smape")]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
