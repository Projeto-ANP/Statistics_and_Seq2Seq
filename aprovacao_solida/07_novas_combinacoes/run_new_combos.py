#!/usr/bin/env python3
"""Testa as novas combinacoes offline, antes de entrarem no catalogo do agente.

Candidatas, todas deterministicas e so com validacao:

    best_per_horizon        pesos one-hot por passo do horizonte
    topk_mean_per_horizon   media dos k melhores por passo
    forward_selection       pesos de Caruana (selecao gulosa com reposicao)
    forward_mean            subconjunto de Caruana combinado por media
    best_single_validation  melhor modelo individual escolhido na validacao

Cada candidata e comparada com o piso (melhor das sementes): score de validacao
(para ver se ela seria selecionada) e metricas de TESTE (para confirmar). Nada
aqui altera o catalogo, o prompt ou as sementes; as receitas novas vivem em
`orchestrator_react/weighting.py` e `selection.py` e so entram no catalogo se
passarem neste teste.

Uso:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    nohup python3 aprovacao_solida/07_novas_combinacoes/run_new_combos.py \\
        > logs/new_combos.log 2>&1 &

A saida e gravada por dataset, nao acumulada em memoria ate o fim. Com
`--windows-out` o script tambem salva o score de cada candidato em cada janela
de validacao, usado nas analises de consistencia entre janelas. Com
`--ensemble-m` avalia ainda a media (igual e softmax) das top-m candidatas por
validacao, como o `final_strategy=ensemble` do harness.
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

from _common import DATASETS, DEFAULT_RESULTS_DIR, detect_drop, load_context, v5_config  # noqa: E402

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from orchestrator_react.selection import PoolRecipe, forward_indices  # noqa: E402
from orchestrator_react.state import FULL_POOL  # noqa: E402
from orchestrator_react.weighting import per_model_error, WeightsRecipe  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402


def _append_csv(path, frame):
    """Grava por dataset, com cabecalho na primeira escrita.

    O script salva a cada dataset em vez de acumular tudo em memoria: uma
    falha no meio do run perde apenas o dataset corrente.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not (os.path.isfile(path) and os.path.getsize(path) > 0)
    frame.to_csv(path, sep=";", index=False, mode="a", header=header)


def _label(spec):
    bits = [str(spec.get("combine"))]
    if spec.get("pool") and spec["pool"] != FULL_POOL:
        bits.append(str(spec["pool"]))
    if spec.get("weights"):
        bits.append(str(spec["weights"]))
    if spec.get("model"):
        bits.append(str(spec["model"]))
    return "_".join(bits)


def build_candidates(state):
    """Avalia cada nova combinacao no state e devolve [(nome, attempt, spec)]."""
    out = []

    recipes = {
        "best_per_horizon": WeightsRecipe(
            method="best_per_horizon", pool_handle=FULL_POOL, per_horizon=True,
            params={"metric": "rmse"},
        ),
        "topk_mean_per_horizon_k3": WeightsRecipe(
            method="topk_mean_per_horizon", pool_handle=FULL_POOL, per_horizon=True,
            params={"k": 3, "metric": "rmse"},
        ),
        "forward_selection": WeightsRecipe(
            method="forward_selection", pool_handle=FULL_POOL,
            params={"max_steps": 20, "metric": "rmse"},
        ),
    }
    for name, recipe in recipes.items():
        handle = state.register_weights(recipe)
        attempt, _ = state.evaluate(
            {"combine": "weighted", "pool": FULL_POOL, "weights": handle},
            rationale=f"nova combinacao: {name}", origin="baseline",
        )
        out.append((name, attempt))

    indices = forward_indices(state.y_true, state.y_preds, max_steps=20, metric="rmse")
    pool_handle = state.register_pool(
        indices, origin="forward",
        recipe=PoolRecipe(method="forward", params={"max_steps": 20, "metric": "rmse"}),
    )
    attempt, _ = state.evaluate(
        {"combine": "mean", "pool": pool_handle},
        rationale="nova combinacao: forward_mean (Caruana)", origin="baseline",
    )
    out.append(("forward_mean", attempt))

    err = per_model_error(state.y_true, state.y_preds, metric="rmse")
    best_model = state.model_names[int(np.nanargmin(err))]
    attempt, _ = state.evaluate(
        {"combine": "best_single", "model": best_model},
        rationale="nova combinacao: best_single por validacao", origin="baseline",
    )
    out.append(("best_single_validation", attempt))

    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Novas combinacoes, teste offline.")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-pooled-meta-model", action="store_true")
    p.add_argument("--no-source", action="store_true")
    p.add_argument("--score-preset", default="balanced")
    p.add_argument("--windows", type=int, default=None,
                   help="numero de janelas de validacao (default: 3, o protocolo da "
                        "bateria). O ANP tem 31 origens por serie, entao aceita ate 30; "
                        "NN5/M4/ETT tem so 4 origens (3 validacao + 1 teste)")
    p.add_argument("--out", default="outputs/resultados/aprovacao_solida/novas_combinacoes.csv")
    p.add_argument("--windows-out", default=None)
    p.add_argument("--ensemble-m", nargs="+", type=int, default=None,
                   help="avalia tambem a media (igual e softmax, eta=5) das top-m "
                        "candidatas por validacao, como o final_strategy=ensemble")
    p.add_argument("--mcs-alpha", type=float, default=None,
                   help="avalia tambem o comite MCS (DM do lider contra cada tentativa, "
                        "p >= alpha fica no conjunto, media igual das previsoes)")
    p.add_argument("--mcs-max-members", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    cfg = v5_config(no_pooled=args.no_pooled_meta_model, name="new_combos")
    cfg.score_preset = args.score_preset
    if args.windows is not None:
        cfg.n_validation_windows = int(args.windows)
    for path in [p for p in (args.out, args.windows_out) if p]:
        if os.path.isfile(path):
            os.remove(path)

    for dataset in args.datasets:
        frames, n_series, source = load_context(
            models, dataset, args.results_dir, args.source_dir, use_source=not args.no_source
        )
        if args.indices is not None:
            todo = list(args.indices)
        elif args.limit:
            todo = list(range(min(int(args.limit), n_series)))
        else:
            todo = list(range(n_series))
        drop = detect_drop(models, dataset, todo[0], cfg, args.results_dir, frames) if todo else []
        meta_models = {}
        if cfg.pooled_meta_model:
            meta_models = pipeline_mod._build_pooled_meta_models(
                models, dataset, todo, config=cfg, source=source,
                results_dir=args.results_dir, frames=frames, drop_models=drop,
            )

        rows = []
        ds_window_rows = []
        for idx in todo:
            try:
                ingested = ingest_mod.load_series(
                    models=models, dataset=dataset, dataset_index=int(idx), source=source,
                    config=cfg, results_dir=args.results_dir, frames=frames, drop_models=drop,
                )
                state = ingested.state
                state.pooled_meta_model = meta_models.get(int(idx))
                pool_mod.run_phase2(state, cfg)
                if args.score_preset == "auto_loo":
                    state.apply_score_weights(state.auto_score_weights()["weights"])

                seeds = [a for a in state.attempts if a.origin == "baseline"]
                floor = min(seeds, key=lambda a: a.score)
                candidates = build_candidates(state)

                fc, _ = state.apply_to_test(floor.spec)
                floor_m = compute_metrics(fc, ingested.test_values)
                fcs = {"floor": np.asarray(fc, dtype=float)}
                scores = {"floor": float(floor.score)}
                rows.append({
                    "dataset": dataset, "dataset_index": int(idx), "method": "floor",
                    "val_score": float(floor.score), "beats_floor_validation": 0,
                    "smape": floor_m["smape"], "pocid": floor_m["pocid"], "rmse": floor_m["rmse"],
                    "spec": json.dumps(floor.spec, ensure_ascii=False, default=str),
                })
                if args.windows_out is not None:
                    for w, window in enumerate(floor.per_window):
                        ds_window_rows.append({
                            "dataset": dataset, "dataset_index": int(idx), "method": "floor",
                            "window": w,
                            "score": float(floor.per_window_scores[w]) if w < len(floor.per_window_scores) else np.nan,
                            "smape": window.get("SMAPE"), "rmse": window.get("RMSE"),
                            "pocid": window.get("POCID"),
                        })

                cand_preview = []
                for name, attempt in candidates:
                    fc, _ = state.apply_to_test(attempt.spec)
                    fcs[name] = np.asarray(fc, dtype=float)
                    scores[name] = float(attempt.score)
                    m = compute_metrics(fc, ingested.test_values)
                    rows.append({
                        "dataset": dataset, "dataset_index": int(idx), "method": name,
                        "val_score": float(attempt.score),
                        "beats_floor_validation": int(attempt.score < floor.score),
                        "smape": m["smape"], "pocid": m["pocid"], "rmse": m["rmse"],
                        "spec": json.dumps(attempt.spec, ensure_ascii=False, default=str),
                    })
                    if args.windows_out is not None:
                        for w, window in enumerate(attempt.per_window):
                            ds_window_rows.append({
                                "dataset": dataset, "dataset_index": int(idx), "method": name,
                                "window": w,
                                "score": float(attempt.per_window_scores[w]) if w < len(attempt.per_window_scores) else np.nan,
                                "smape": window.get("SMAPE"), "rmse": window.get("RMSE"),
                                "pocid": window.get("POCID"),
                            })
                    cand_preview.append(f"{name}={m['smape']:.4f}")

                for m in (args.ensemble_m or []):
                    top = sorted(scores, key=lambda k: scores[k])[: max(1, int(m))]
                    stacked = np.vstack([fcs[k] for k in top])
                    sc = np.array([scores[k] for k in top], dtype=float)
                    scale = float(np.median(sc))
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0
                    z = -5.0 * (sc / scale)
                    z -= z.max()
                    w = np.exp(z)
                    w = w / w.sum() if w.sum() > 0 else np.ones(len(top)) / len(top)
                    for label, forecast in (
                        ("equal", stacked.mean(axis=0)),
                        ("softmax", np.tensordot(w, stacked, axes=1)),
                    ):
                        em = compute_metrics(forecast, ingested.test_values)
                        rows.append({
                            "dataset": dataset, "dataset_index": int(idx),
                            "method": f"ens_{label}_m{int(m)}",
                            "val_score": float(np.mean(sc)),
                            "beats_floor_validation": int(float(np.mean(sc)) < floor.score),
                            "smape": em["smape"], "pocid": em["pocid"], "rmse": em["rmse"],
                            "spec": json.dumps({"members": top, "weights": [float(x) for x in w]}
                                               if label == "softmax" else {"members": top}),
                        })
                        cand_preview.append(f"ens_{label}_m{int(m)}={em['smape']:.4f}")
                if args.mcs_alpha is not None:
                    fc_m, dbg_m = state.apply_mcs(
                        alpha=float(args.mcs_alpha), max_members=args.mcs_max_members
                    )
                    mcs_m = compute_metrics(fc_m, ingested.test_values)
                    leader = state.best_attempt()
                    rows.append({
                        "dataset": dataset, "dataset_index": int(idx),
                        "method": f"mcs_a{args.mcs_alpha}",
                        "val_score": float(leader.score),
                        "beats_floor_validation": int(leader.score < floor.score),
                        "smape": mcs_m["smape"], "pocid": mcs_m["pocid"], "rmse": mcs_m["rmse"],
                        "spec": json.dumps(
                            {"members": [m["attempt_id"] for m in dbg_m["members"]],
                             "n_members": dbg_m["n_members"]},
                            ensure_ascii=False,
                        ),
                    })
                    cand_preview.append(f"mcs={mcs_m['smape']:.4f}")
                if not args.quiet:
                    print(
                        f"[{dataset} {idx:>4}] floor={floor_m['smape']:.4f} | "
                        + " ".join(cand_preview),
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{dataset} {idx}] FALHOU: {type(exc).__name__}: {exc}", flush=True)

        if rows:
            _append_csv(args.out, pd.DataFrame(rows))
            if args.windows_out is not None and ds_window_rows:
                _append_csv(args.windows_out, pd.DataFrame(ds_window_rows))
            sub = pd.DataFrame(rows)
            print(f"\n=== {dataset}: n={sub['dataset_index'].nunique()} series ===")
            floor = sub[sub.method == "floor"].set_index("dataset_index")["smape"]
            floor_mean = float(floor.mean())
            for name in sub.method.unique():
                if name == "floor":
                    continue
                s = sub[sub.method == name].set_index("dataset_index")
                wins = int((s["smape"] < floor.reindex(s.index) - 1e-6).sum())
                losses = int((s["smape"] > floor.reindex(s.index) + 1e-6).sum())
                print(
                    f"  {name:<26} smape={s.smape.mean():.4f} (floor {floor_mean:.4f}) "
                    f"pocid={s.pocid.mean():.1f} | melhor que floor no teste em "
                    f"{wins}, pior em {losses} | selecionada na validacao em "
                    f"{int(s.beats_floor_validation.sum())}/{len(s)}"
                )

    print(f"\nsalvo incrementalmente em: {args.out}")
    if args.windows_out is not None:
        print(f"janelas salvas incrementalmente em: {args.windows_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
