#!/usr/bin/env python3
"""Enumeracao deterministica do catalogo v2: o teto alcancavel pelo agente.

Responde a pergunta de banca "qual a versao determinística das acoes do agente?".
O braco `01_busca_nao_llm` cobre o mesmo catalogo com o MESMO orcamento do
agente (12 avaliacoes) e politicas fixas (random, guloso, UCB). Este script vai
alem: enumera TODO o espaco de acoes do catalogo v2, por serie, aplica o mesmo
critério de decisao do harness (argmin do score de validacao) e mede o teste.

Espaco enumerado por serie:

    pools    : full, select_stable k em {3,5,7,9,11,13,15}, select_top_k k em
               {3,5,8}, prune_redundant, select_forward (Caruana)
    combiner : mean, median, trimmed_mean (20%), dba, best_single (todo modelo)
    pesos    : inverse_error, softmax_neg_error, error_trend, feature_based,
               pooled_meta_model (se disponivel), ols (se janelas suficientes),
               e, sem --no-extra-actions, best_per_horizon, topk_mean_per_horizon
               e forward_selection

Tudo determinístico, sem LLM. O vencedor e o argmin da validacao entre as
sementes (Phase 2) e os specs enumerados; o teste do vencedor e gravado ao lado
do teste do piso (melhor semente), para a comparacao direta.

Custo: ~150-190 backtests por serie, 10 a 15x o braco de busca. Rode por partes:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    nohup python3 aprovacao_solida/09_catalogo_deterministico/run_catalog_enum.py \\
        --datasets ANP_MONTHLY NN5_WEEKLY_DATASET ETTH1 ETTH2 ETTM1 ETTM2 \\
        > logs/catalog_enum_sem_m4.log 2>&1 &

A saida e gravada por dataset, nao acumulada em memoria ate o fim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PARENT)
_SEARCH = os.path.join(_PARENT, "01_busca_nao_llm")
for _p in (_SEARCH, _PARENT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import DATASETS, DEFAULT_RESULTS_DIR, detect_drop, load_context, v5_config  # noqa: E402
from search_lib import SpecFactory, allowed_weight_methods  # noqa: E402

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402


def _key(spec):
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)


def _append_csv(path, frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not (os.path.isfile(path) and os.path.getsize(path) > 0)
    frame.to_csv(path, sep=";", index=False, mode="a", header=header)


def _forward_pool(state):
    try:
        return T.select_forward(state)["pool"]
    except Exception:
        return None


def enumerate_specs(state, factory, extra_actions=True):
    """Todos os specs alcancaveis pelo catalogo v2 para esta serie."""
    specs = [{"combine": "best_single", "model": name} for name in state.model_names]

    pools = [(label, handle) for label, handle in factory.pools]
    forward = _forward_pool(state)
    if forward is not None:
        pools.append(("forward", forward))

    methods = list(allowed_weight_methods(state))
    extras = (
        ("best_per_horizon", T.weights_best_per_horizon),
        ("topk_per_horizon", T.weights_topk_per_horizon),
        ("forward_weights", T.weights_forward),
    )
    for _label, pool in pools:
        for combine in ("mean", "median", "trimmed_mean", "dba"):
            spec = {"combine": combine, "pool": pool}
            if combine == "trimmed_mean":
                spec["trim_pct"] = 0.2
            specs.append(spec)
        for method in methods:
            specs.append(
                {"combine": "weighted", "pool": pool, "weights": factory.weight_handle(method, pool)}
            )
        if extra_actions:
            for _name, tool in extras:
                specs.append(
                    {"combine": "weighted", "pool": pool, "weights": tool(state, pool=pool)["weights"]}
                )
    return specs


def evaluate_specs(state, specs):
    """Avalia os specs ineditos; devolve (avaliados, falhas)."""
    seen = {_key(a.spec) for a in state.attempts}
    n_ok = 0
    n_fail = 0
    for spec in specs:
        key = _key(spec)
        if key in seen:
            continue
        seen.add(key)
        try:
            attempt, _ = state.evaluate(
                spec, rationale="enumeracao deterministica do catalogo", origin="enum"
            )
            if np.isfinite(attempt.score):
                n_ok += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1
    return n_ok, n_fail


def _test_metrics(state, spec, test_values):
    fc, _ = state.apply_to_test(spec)
    return compute_metrics(fc, test_values)


def main(argv=None):
    p = argparse.ArgumentParser(description="Enumeracao deterministica do catalogo v2.")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-pooled-meta-model", action="store_true")
    p.add_argument("--no-source", action="store_true")
    p.add_argument("--no-extra-actions", action="store_true",
                   help="enumera so o catalogo v2 de 10 entradas (sem best_per_horizon, "
                        "topk_mean_per_horizon e forward weights)")
    p.add_argument("--score-preset", default="balanced")
    p.add_argument("--out", default="outputs/resultados/aprovacao_solida/catalogo_deterministico.csv")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    cfg = v5_config(no_pooled=args.no_pooled_meta_model, name="catalog_enum")
    cfg.score_preset = args.score_preset
    if os.path.isfile(args.out):
        os.remove(args.out)

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

                factory = SpecFactory(state)
                specs = enumerate_specs(state, factory, extra_actions=not args.no_extra_actions)
                n_ok, n_fail = evaluate_specs(state, specs)

                winner = state.best_attempt()
                floor_m = _test_metrics(state, floor.spec, ingested.test_values)
                winner_m = _test_metrics(state, winner.spec, ingested.test_values)
                scale = abs(float(floor.score)) or 1.0
                rows.append({
                    "dataset": dataset, "dataset_index": int(idx),
                    "n_specs": int(n_ok), "n_failed": int(n_fail),
                    "floor_score": float(floor.score), "enum_score": float(winner.score),
                    "rel_gain": float((float(floor.score) - float(winner.score)) / scale),
                    "winner_is_floor": int(_key(winner.spec) == _key(floor.spec)),
                    "floor_smape": floor_m["smape"], "floor_pocid": floor_m["pocid"], "floor_rmse": floor_m["rmse"],
                    "enum_smape": winner_m["smape"], "enum_pocid": winner_m["pocid"], "enum_rmse": winner_m["rmse"],
                    "delta_smape": winner_m["smape"] - floor_m["smape"],
                    "winner_spec": _key(winner.spec), "floor_spec": _key(floor.spec),
                })
                if not args.quiet:
                    print(
                        f"[{dataset} {idx:>4}] specs={n_ok} falhas={n_fail} "
                        f"floor={floor_m['smape']:.4f} enum={winner_m['smape']:.4f} "
                        f"delta={winner_m['smape'] - floor_m['smape']:+.4f}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{dataset} {idx}] FALHOU: {type(exc).__name__}: {exc}", flush=True)

        if rows:
            _append_csv(args.out, pd.DataFrame(rows))
            sub = pd.DataFrame(rows)
            d = sub.delta_smape
            print(f"\n=== {dataset}: n={len(sub)} series | specs/serie={sub.n_specs.mean():.0f} ===")
            print(
                f"  piso smape={sub.floor_smape.mean():.4f} | enumeracao smape={sub.enum_smape.mean():.4f} "
                f"| vencedor=piso em {int(sub.winner_is_floor.sum())}/{len(sub)} series "
                f"| melhora={int((d < -1e-6).sum())} piora={int((d > 1e-6).sum())} empata={int((d.abs() <= 1e-6).sum())}"
            )

    print(f"\nsalvo incrementalmente em: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
