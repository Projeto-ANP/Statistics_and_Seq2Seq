"""Runner do agente LAYA (classificador System One) por dataset.

Espelho do `run_tsf_orchestrator.py` para a família JEV/: mesma ingestão, mesmas
sementes da Fase 2, mesmo contrato de avaliação (backtest nas janelas de
validação, melhor tentativa aplicada ao teste). A única diferença é a Fase 3:
em vez do loop ReAct com LLM gerador, o classificador LAYA escolhe entre ações
concretas (menu), turno a turno.

CLI no estilo do run_tsf_batch.py:

    python3 JEV/run_laya.py --datasets ETTM2 NN5_WEEKLY_DATASET --version laya_v0
    python3 JEV/run_laya.py --datasets ETTM2 --version laya_v0_en --checkpoint english
    python3 JEV/run_laya.py --datasets ETTM2 --version smoke --indices 0

Saída: `./timeseries/mestrado/resultados/orchestrator_laya_<version>/<DATASET>.csv`
(13 colunas core + description/origin/iterations/stop_reason/seed_floor_smape).

O DATASET SUMMARY imprime, além do "this run" e dos baselines externos, a linha
`baseline` — o piso das sementes (braço determinístico, o mesmo número do
`orchestrator_baseline_v1`), para a comparação "agente vs semente" ficar no
próprio log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd  # noqa: E402

from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402
from orchestrator_react import ingest as I  # noqa: E402
from orchestrator_react import pipeline as PL  # noqa: E402
from orchestrator_react import pool as POOL  # noqa: E402
from orchestrator_react import prompts as PR  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.config import ReactConfig  # noqa: E402
from orchestrator_react.csv_writer import CORE_COLUMNS, compute_metrics  # noqa: E402
from orchestrator_react.data_source import load_series_source  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "JEV"))
from laya_loop import LayaAgent, run_laya_loop  # noqa: E402

# Mesmo mapa do run_tsf_batch.py (nomes MAIÚSCULOS para ETTh/ETTm são outro arquivo!).
DATASET_SOURCES: Dict[str, str] = {
    "ETTH1": "ETTH1.tsf",
    "ETTH2": "ETTH2.tsf",
    "ETTM1": "ETTM1.tsf",
    "ETTM2": "ETTM2.tsf",
    "ANP_MONTHLY": "mes_11_venda_mensal.tsf",
    "NN5_WEEKLY_DATASET": "nn5_weekly_dataset.tsf",
    "NN5_DAILY_DATASET_WITHOUT_MISSING_VALUES": "nn5_daily_dataset_without_missing_values.tsf",
    "M4_WEEKLY_DATASET": "m4_weekly_dataset.tsf",
    "M4_HOURLY_DATASET": "m4_hourly_dataset.tsf",
    "PEDESTRIAN_COUNTS_DATASET": "pedestrian_counts_dataset.tsf",
    "US_BIRTHS_DATASET": "us_births_dataset.tsf",
}

DEFAULT_RESULTS_DIR = "./timeseries/mestrado/resultados"
DEFAULT_SOURCE_DIR = "../forecasting_datasets"

COLS: List[str] = CORE_COLUMNS + [
    "description", "origin", "react_iterations_used", "react_stop_reason",
    "seed_floor_smape", "seed_floor_rmse", "ablation_config",
]

_SUMMARY_KEYS = ("mape", "pocid", "smape", "rmse", "msmape", "mae")


def _summarise(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    out: Dict[str, float] = {}
    for key in _SUMMARY_KEYS:
        values = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out


def _seed_floor_metrics(state: Any, test_values: List[float]) -> Optional[Dict[str, float]]:
    """Piso determinístico desta série: a melhor SEMENTE (por score de validação)
    aplicada ao teste. É o braço `use_llm=False` do orquestrador."""
    seeds = [a for a in state.attempts if getattr(a, "origin", "") == "baseline"]
    if not seeds:
        return None
    best = min(seeds, key=lambda a: a.score if a.score == a.score else float("inf"))
    try:
        forecast, _ = state.apply_to_test(best.spec)
        return compute_metrics(forecast, test_values)
    except Exception:
        return None


def run_dataset(
    dataset: str,
    source_file: str,
    version: str,
    checkpoint: str = "english",
    max_len: Optional[int] = None,
    max_iterations: int = 12,
    no_seeds: bool = False,
    state_budget: int = 1400,
    option_format: str = "text",
    dataset_card: bool = True,
    indices: Optional[List[int]] = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    models = [m for m in DEFAULT_MODELS if os.path.exists(I.model_csv_path(m, dataset, results_dir))]
    frames = I.load_dataset_frames(models, dataset, results_dir)
    n_series = I.count_series(dataset, models[0], results_dir)

    # mesma detecção de desalinhamento do run_dataset original
    drop: List[str] = []
    try:
        bad = I.find_misaligned_models(
            models, dataset, 0, results_dir=results_dir, frames=frames, n_windows=3,
        )
    except Exception:
        bad = {}
    drop = sorted(bad)
    models_eff = [m for m in models if m not in drop]
    if drop:
        print(
            f"WARNING: {len(drop)} model(s) do not share the pool's windows on {dataset} "
            f"and are being DROPPED for every series: {drop}",
            flush=True,
        )

    cfg = ReactConfig()
    experiment = f"orchestrator_laya_{version}"
    out_dir = os.path.join(results_dir, experiment)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset}.csv")

    agent = LayaAgent(checkpoint=checkpoint, max_len=max_len)

    todo = indices if indices is not None else list(range(n_series))

    # ── pré-pass do DATASET CARD (prior cross-series LOO, só validação) ─────
    priors: Dict[int, Dict[str, float]] = {}
    if dataset_card:
        try:
            source = load_series_source(
                source_file, n_expected_series=n_series, source_dir=source_dir,
            )
            priors = PL._build_strategy_priors(
                models_eff, dataset, todo, cfg, source, results_dir, frames, {},
            )
        except Exception as exc:
            print(f"WARNING: dataset card pre-pass falhou ({exc}); sem card",
                  flush=True)
    print(f"dataset      : {dataset}")
    print(f"source       : {source_file}")
    print(f"models       : {len(models_eff)}")
    print(f"agent        : {agent.name} (max_len={agent.max_len})")
    print(f"writing to   : {out_dir}")
    print("-" * 74, flush=True)

    rows: List[Dict[str, Any]] = []
    per_series: List[Dict[str, float]] = []
    floor_rows: List[Dict[str, float]] = []
    external: Dict[str, List[Dict[str, float]]] = {}
    started = time.perf_counter()
    ok = failed = 0

    for idx in todo:
        try:
            ing = I.load_series(
                models=models_eff, dataset=dataset, dataset_index=idx, config=cfg,
                results_dir=results_dir, source_file=source_file, source_dir=source_dir,
                frames=frames,
            )
            state = ing.state
            phase2 = POOL.run_phase2(state, cfg)
            if no_seeds:
                # braço "sem sementes": o agente parte de histórico vazio e não
                # recebe os pools estáveis da Fase 2 (pool_full permanece — é a
                # base, não uma semente). O pool card continua no estado.
                state.attempts.clear()
                for handle in [h for h in list(state.pools) if h != "pool_full"]:
                    del state.pools[handle]
            series_card = T.series_profile(state)
            pool_card = phase2["report"]
            card = None
            if priors.get(idx):
                state.strategy_prior = priors[idx]
                try:
                    card = PR.build_dataset_card(state)
                except Exception:
                    card = None
            loop = run_laya_loop(
                state, agent, series_card, pool_card, max_iterations=max_iterations,
                no_seeds=no_seeds, state_budget=state_budget, fmt=option_format,
                dataset_card=card,
            )
            attempt = loop.final_attempt
            if attempt is None:
                raise RuntimeError("no strategy was selected")
            forecast, _ = state.apply_to_test(attempt.spec)
            metrics = compute_metrics(forecast, ing.test_values)
            floor = _seed_floor_metrics(state, ing.test_values)

            # ── artifact de telemetria por série (debug completo) ───────────
            art_dir = os.path.join(out_dir, "llm_artifacts", dataset)
            os.makedirs(art_dir, exist_ok=True)
            art_path = os.path.join(art_dir, f"dataset_{idx}.json")
            try:
                with open(art_path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "dataset": dataset, "series": int(idx),
                        "config": {
                            "checkpoint": checkpoint, "max_len": agent.max_len,
                            "option_format": option_format,
                            "max_iterations": max_iterations,
                            "state_budget": state_budget, "no_seeds": no_seeds,
                        },
                        "final": {
                            "strategy": attempt.spec, "origin": attempt.origin,
                            "score": round(float(attempt.score), 6),
                        },
                        "stop_reason": loop.stop_reason,
                        "iterations_used": loop.iterations_used,
                        "floor_smape_test": floor["smape"] if floor else None,
                        "final_smape_test": metrics["smape"],
                        "trace": loop.trace,
                        "step_details": loop.step_details,
                        "errors": loop.errors,
                    }, fh, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass

            rows.append({
                "dataset_index": str(idx),
                "horizon": ing.horizon,
                "regressor": experiment,
                **metrics,
                "test": [list(ing.test_values)],
                "predictions": [list(forecast)],
                "start_test": str(ing.start_test),
                "final_test": str(ing.final_test),
                "description": json.dumps({
                    "strategy": attempt.spec,
                    "origin": attempt.origin,
                    "loop": loop.summary(),
                    "score": round(float(attempt.score), 4),
                }, ensure_ascii=False, default=str),
                "origin": attempt.origin,
                "react_iterations_used": loop.iterations_used,
                "react_stop_reason": loop.stop_reason,
                "seed_floor_smape": floor["smape"] if floor else None,
                "seed_floor_rmse": floor["rmse"] if floor else None,
                "ablation_config": f"laya_{checkpoint}_{version}{'_noseeds' if no_seeds else ''}_{option_format}",
            })
            per_series.append(metrics)
            if floor:
                floor_rows.append(floor)
            ok += 1
            print(
                f"[{idx:>4}] {attempt.spec['combine']:<14} score={attempt.score:7.4f} "
                f"origin={attempt.origin:<8} iters={loop.iterations_used} stop={loop.stop_reason}"
            )
            print(
                f"         TEST  smape={metrics['smape']:.4f}  rmse={metrics['rmse']:.4f}"
                f"  pocid={metrics['pocid']:.2f}  mape={metrics['mape']:.4f}"
                + (f"  | FLOOR smape={floor['smape']:.4f}" if floor else "")
            )
            for entry in loop.trace:
                detail = (
                    f"rank={entry['rank']}" if "rank" in entry
                    else f"obs={entry.get('observation', '')}"
                )
                print(
                    f"         laya iter {entry['iteration']} -> {entry['action']}"
                    f" conf={entry['confidence']} {detail}"
                )
            ext = I.read_external_baselines(dataset, idx, results_dir=results_dir)
            for name, stats in ext.items():
                if isinstance(stats, dict) and stats.get("available"):
                    external.setdefault(name, []).append(stats)
        except Exception as exc:
            failed += 1
            rows.append({
                "dataset_index": str(idx), "horizon": 0, "regressor": experiment,
                **{k: float("nan") for k in _SUMMARY_KEYS},
                "test": [[]], "predictions": [[]],
                "start_test": "", "final_test": "",
                "description": json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                "origin": "", "react_iterations_used": 0, "react_stop_reason": "error",
                "seed_floor_smape": None, "seed_floor_rmse": None,
                "ablation_config": f"laya_{checkpoint}_{version}{'_noseeds' if no_seeds else ''}_{option_format}",
            })
            print(f"[{idx:>4}] FAILED: {type(exc).__name__}: {exc}")

    frame = pd.DataFrame(rows).reindex(columns=COLS)
    frame.to_csv(csv_path, sep=";", index=False)

    elapsed = time.perf_counter() - started
    summary = _summarise(per_series)
    floor_sum = _summarise(floor_rows)
    print("-" * 74)
    print(f"done in {elapsed:.1f}s | ok: {ok} | failed: {failed}")
    if summary:
        print(f"DATASET SUMMARY over {len(per_series)} series (mean across series):")
        print(
            f"  this run   smape={summary['smape']:.4f}  rmse={summary['rmse']:.4f}"
            f"  pocid={summary['pocid']:.2f}  mape={summary['mape']:.4f}"
            f"  mae={summary['mae']:.4f}"
        )
        if no_seeds:
            print("  baseline   (n/a - no-seeds arm: o agente partiu de histórico vazio)")
        elif floor_sum:
            delta = summary["rmse"] - floor_sum["rmse"]
            if abs(delta) < 1e-12:
                note = "  (== seed floor: no agent contribution this run)"
            elif delta < 0:
                note = "  <- this run is better"
            else:
                note = "  <- this run is WORSE than the seed floor"
            print(
                f"  baseline   smape={floor_sum['smape']:.4f}  rmse={floor_sum['rmse']:.4f}"
                f"  pocid={floor_sum['pocid']:.2f}  mape={floor_sum['mape']:.4f}{note}"
            )
        for name, rows_ in external.items():
            stats = _summarise(rows_)
            better = "  <- this run is better" if stats["rmse"] > summary["rmse"] else ""
            print(
                f"  {name:<10} smape={stats['smape']:.4f}  rmse={stats['rmse']:.4f}"
                f"  pocid={stats['pocid']:.2f}  mape={stats['mape']:.4f}{better}"
            )
    print(f"csv: {csv_path}")
    return {
        "dataset": dataset, "experiment": experiment, "n_ok": ok, "n_failed": failed,
        "csv_path": csv_path, "elapsed_s": elapsed,
        "summary": summary, "baseline": floor_sum,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LAYA classifier agent over forecast combination.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--version", default="laya_v0")
    p.add_argument("--checkpoint", choices=["english", "multilingual"], default="english")
    p.add_argument("--max-len", type=int, default=None,
                   help="max_len do laya (english: 512; multilingual: até 8192)")
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--option-format", choices=["text", "raw"], default="text",
                   help="evidência dos modelos nas opções: 'text' (resumo "
                        "comparativo) ou 'raw' (números crus) — o A/B")
    p.add_argument("--state-budget", type=int, default=1400,
                   help="orçamento do estado do classificador em chars "
                        "(english: ~1400; multilingual 8192: pode subir p/ ~8000)")
    p.add_argument("--no-dataset-card", action="store_true",
                   help="não montar o DATASET CARD (prior cross-series LOO)")
    p.add_argument("--no-seeds", action="store_true",
                   help="ablação: o agente parte de histórico VAZIO — sem as "
                        "sementes da Fase 2 (o menu ganha ações de construção de pool)")
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = p.parse_args(argv)

    failures = 0
    for dataset in args.datasets:
        source = DATASET_SOURCES.get(dataset)
        if source is None:
            print(f"dataset '{dataset}' não está em JEV/run_laya.py DATASET_SOURCES")
            failures += 1
            continue
        try:
            run_dataset(
                dataset=dataset,
                source_file=source,
                version=args.version,
                checkpoint=args.checkpoint,
                max_len=args.max_len,
                max_iterations=args.max_iterations,
                no_seeds=args.no_seeds,
                state_budget=args.state_budget,
                option_format=args.option_format,
                dataset_card=not args.no_dataset_card,
                indices=args.indices,
                source_dir=args.source_dir,
                results_dir=args.results_dir,
            )
        except Exception as exc:
            print(f"\nDATASET {dataset} ABORTOU: {type(exc).__name__}: {exc}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
