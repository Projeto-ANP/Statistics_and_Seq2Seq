"""Oráculo amplo do CARTE sobre um universo fixo de estratégias.

Para cada série, enumera o catálogo determinístico usado pelas buscas sem LLM
e calcula dois oráculos:

  - teto de validação: estratégia de menor score de validação; o teste dela é
    reportado como o resultado que a melhor escolha possível na validação
    teria obtido;
  - oráculo de teste: estratégia de menor SMAPE (e a de maior POCID) na janela
    de teste, sem uso de validação. É um teto retrospectivo, não implantável.

O universo U é fixo e reproduzível: pool completo, seleções por estabilidade
(k = 3, 5, 7, 9, 11, 13, 15), seleções pelos melhores (k = 3, 5, 8), seleção
forward e as combinações média, mediana, média aparada (trim de 0 a 0,4 em
passos de 0,1), DBA e média ponderada com as receitas inverse (shrinkage de 0
a 0,5 em passos de 0,1), error trend, pooled/meta, best per horizon, top-k per
horizon e forward selection.

Saídas:
  outputs/resultados/ablacao_oraculo_u_carte/por_serie.csv
  outputs/resultados/ablacao_oraculo_u_carte/resumo.csv
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
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_APROV = os.path.join(_REPO, "aprovacao_solida")
for _p in (_REPO, _APROV, os.path.join(_APROV, "01_busca_nao_llm"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import (  # noqa: E402
    DATASETS,
    DEFAULT_RESULTS_DIR,
    detect_drop,
    load_context,
)
from ablacao_search_carte import carte_config  # noqa: E402
from search_lib_carte import SpecFactory, allowed_weight_methods  # noqa: E402

from orchestrator_react import ingest as ingest_mod  # noqa: E402
from orchestrator_react import pipeline as pipeline_mod  # noqa: E402
from orchestrator_react import pool as pool_mod  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.csv_writer import compute_metrics  # noqa: E402
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402

OUT_DIR = os.path.join(_REPO, "outputs", "resultados", "ablacao_oraculo_u_carte")
TRIM_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)
SHRINKAGE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def spec_key(spec) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)


def enumerate_specs(state, factory: SpecFactory) -> list[dict]:
    specs = []
    seen = set()

    def add(spec: dict) -> None:
        key = spec_key(spec)
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for _label, pool in factory.pools:
        add({"combine": "mean", "pool": pool})
        add({"combine": "median", "pool": pool})
        add({"combine": "dba", "pool": pool})
        for trim_pct in TRIM_GRID:
            add({"combine": "trimmed_mean", "pool": pool, "trim_pct": float(trim_pct)})
        for shrinkage in SHRINKAGE_GRID:
            try:
                handle = T.weights_inverse_error(
                    state, pool=pool, shrinkage=float(shrinkage)
                )["weights"]
            except Exception:
                continue
            add({"combine": "weighted", "pool": pool, "weights": handle})
        for method in allowed_weight_methods(state):
            if method == "inverse":
                continue
            handle = factory.weight_handle(method, pool)
            if handle is None:
                continue
            add({"combine": "weighted", "pool": pool, "weights": handle})
    return specs


def evaluate_specs(state, specs, test_values):
    records = []
    failures = 0
    for spec in specs:
        try:
            attempt, _ = state.evaluate(spec, rationale="oráculo U", origin="oracle")
            forecast, _ = state.apply_to_test(spec)
            metrics = compute_metrics(forecast, test_values)
            records.append(
                {
                    "validation": float(attempt.score),
                    "smape": float(metrics["smape"]) * 100.0,
                    "pocid": float(metrics["pocid"]),
                    "spec": spec,
                }
            )
        except Exception:
            failures += 1
    return records, failures


def build_rows(
    datasets: list[str],
    out_csv: str,
    indices: set[int] | None = None,
    resume: bool = False,
) -> None:
    models = list(DEFAULT_MODELS)
    cfg = carte_config(evals=12, no_pooled_meta_model=False)
    done = set()
    if resume and os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0:
        old = pd.read_csv(out_csv, sep=";", usecols=["dataset", "dataset_index"])
        done = {(str(r.dataset), int(r.dataset_index)) for r in old.itertuples()}
        print(f"retomando: {len(done)} séries já gravadas em {out_csv}", flush=True)
    elif os.path.isfile(out_csv):
        os.remove(out_csv)

    for dataset in datasets:
        print(f"\n=== {dataset} ===", flush=True)
        frames, n_series, source = load_context(
            models, dataset, DEFAULT_RESULTS_DIR, DEFAULT_SOURCE_DIR, use_source=True
        )
        todo = sorted(indices) if indices is not None else list(range(n_series))
        todo = [i for i in todo if 0 <= i < n_series]
        if todo and all((dataset, int(idx)) in done for idx in todo):
            print(f"{dataset}: já concluído, pulando", flush=True)
            continue
        drop = detect_drop(models, dataset, todo[0], cfg, DEFAULT_RESULTS_DIR, frames) if todo else []
        if drop:
            print(f"WARNING: modelos desalinhados descartados: {drop}", flush=True)

        meta_models = {}
        if todo:
            meta_models = pipeline_mod._build_pooled_meta_models(
                models,
                dataset,
                todo,
                config=cfg,
                source=source,
                results_dir=DEFAULT_RESULTS_DIR,
                frames=frames,
                drop_models=drop,
            )
            if not meta_models:
                print("WARNING: pooled meta-model nao treinado; weights_meta_model indisponivel.", flush=True)

        for idx in todo:
            if (dataset, int(idx)) in done:
                continue
            t0 = time.perf_counter()
            try:
                ingested = ingest_mod.load_series(
                    models=models,
                    dataset=dataset,
                    dataset_index=int(idx),
                    source=source,
                    config=cfg,
                    results_dir=DEFAULT_RESULTS_DIR,
                    frames=frames,
                    drop_models=drop,
                )
                state = ingested.state
                state.pooled_meta_model = meta_models.get(int(idx))
                pool_mod.run_phase2(state, cfg)
                factory = SpecFactory(state)
                specs = enumerate_specs(state, factory)
                records, failures = evaluate_specs(state, specs, ingested.test_values)
                if not records:
                    print(f"[{dataset} {idx:>4}] sem candidatos", flush=True)
                    continue

                val_pool = [r for r in records if np.isfinite(r["validation"])]
                test_pool = [r for r in records if np.isfinite(r["smape"]) and np.isfinite(r["pocid"])]
                val = min(val_pool, key=lambda r: r["validation"]) if val_pool else None
                test_smape = min(test_pool, key=lambda r: r["smape"]) if test_pool else None
                test_pocid = max(test_pool, key=lambda r: r["pocid"]) if test_pool else None
                row = {
                    "dataset": dataset,
                    "dataset_index": int(idx),
                    "n_specs": len(records),
                    "n_failures": failures,
                    "u_val_smape": val["smape"] if val else np.nan,
                    "u_val_pocid": val["pocid"] if val else np.nan,
                    "u_val_score": val["validation"] if val else np.nan,
                    "u_test_smape": test_smape["smape"] if test_smape else np.nan,
                    "u_test_smape_pocid": test_smape["pocid"] if test_smape else np.nan,
                    "u_test_pocid": test_pocid["pocid"] if test_pocid else np.nan,
                    "u_test_pocid_smape": test_pocid["smape"] if test_pocid else np.nan,
                    "u_val_spec": spec_key(val["spec"]) if val else None,
                    "u_test_smape_spec": spec_key(test_smape["spec"]) if test_smape else None,
                    "u_test_pocid_spec": spec_key(test_pocid["spec"]) if test_pocid else None,
                }
                header = not (os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0)
                pd.DataFrame([row]).to_csv(
                    out_csv, sep=";", index=False, mode="a", header=header
                )
                print(
                    f"[{dataset} {idx:>4}] specs={len(records):>3} falhas={failures:>2} "
                    f"val={(val['smape'] if val else float('nan')):.2f} "
                    f"teste={(test_smape['smape'] if test_smape else float('nan')):.2f} "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{dataset} {idx:>4}] FALHOU: {type(exc).__name__}: {exc}", flush=True)


def summarize(rows: pd.DataFrame, n_datasets: int) -> pd.DataFrame:
    metrics = [
        "u_val_smape",
        "u_val_pocid",
        "u_test_smape",
        "u_test_pocid",
        "n_specs",
        "n_failures",
    ]
    out = []
    for dataset, group in rows.groupby("dataset"):
        item = {"dataset": dataset, "n_series": len(group)}
        item.update({m: group[m].mean() for m in metrics})
        out.append(item)
    summary = pd.DataFrame(out)
    total_label = f"Total {len(rows)}" if n_datasets == 7 else f"Total {len(rows)} (parcial)"
    media_label = "Media 7" if n_datasets == 7 else f"Media {n_datasets}"
    for label, group in ((total_label, rows), (media_label, None)):
        item = {"dataset": label, "n_series": len(rows) if group is not None else n_datasets}
        for m in metrics:
            item[m] = group[m].mean() if group is not None else summary[m].mean()
        out.append(item)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Oráculo amplo do CARTE sobre U.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--resume", action="store_true",
                        help="pula séries já presentes em por_serie.csv")
    args = parser.parse_args()

    indices = None
    if args.indices is not None:
        indices = set(int(i) for i in args.indices)
    elif args.limit is not None:
        indices = set(range(int(args.limit)))

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "por_serie.csv")
    build_rows(args.datasets, out_csv, indices, resume=args.resume)
    rows = pd.read_csv(out_csv, sep=";")
    summary = summarize(rows, len(args.datasets))
    summary.to_csv(os.path.join(args.out_dir, "resumo.csv"), index=False)
    print("\n", summary.to_string(index=False), sep="")


if __name__ == "__main__":
    main()
