"""Orchestrator sem agente: so a parte deterministica (Fases 0, 1a, 1b, 2 e 4).

Roda o mesmo pipeline de `run_tsf_orchestrator.py` (ingestao dos 19 modelos base,
perfil da serie, meta-modelo pooled entre series, semeadura de baselines, diagnostico
deterministico), mas SEM nenhum LLM: os tres papeis (combinator, diagnostician,
reporter) ficam desligados, entao nao ha acesso ao Ollama nem preflight. A estrategia
final de cada serie e a melhor baseline semeada (menor score de validacao) -- e o mesmo
que `run_tsf_orchestrator.py --no-llm`, num script enxuto.

Sai em `<output-dir>/orchestrator_baseline_<version>/<DATASET>.csv` (mesmas colunas
`COLS_SERIE`), sem sobrescrever `orchestrator_react_*`. Colunas que dependem do agente
ficam vazias por construcao (react_*, accept_confidence, tools_called, agent_model_*="none").

Exemplos:

    python run_tsf_baselines.py --dataset ANP_MONTHLY --source mes_11_venda_mensal.tsf
    python run_tsf_baselines.py --dataset ETTH1 --source ETTH1.tsf --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator_react import pipeline as _pipeline
from orchestrator_react.config import LLMRole, ReactConfig
from orchestrator_react.csv_writer import COLS_SERIE, ResultWriter, compute_metrics
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR, SeriesAlignmentError
from orchestrator_react.ingest import DEFAULT_RESULTS_DIR, count_series
from run_tsf_orchestrator import DEFAULT_MODELS, _external_summary, _summarise


def exec_dataset_baselines(
    models: Optional[Sequence[str]] = None,
    dataset: str = "ANP_MONTHLY",
    source_file: Optional[str] = None,
    *,
    source_dir: str = DEFAULT_SOURCE_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    output_dir: Optional[str] = None,
    n_windows: int = 3,
    backtest_mode: str = "expanding",
    nested_selection: bool = True,
    min_windows_for_ols: int = 5,
    seed_stable_pools: bool = True,
    pooled_meta_model: bool = True,
    pooled_meta_model_min_series: int = 20,
    pooled_meta_model_objective: str = "fforma",
    seed_pooled_meta_model: bool = True,
    dataset_card: bool = False,
    pool_mode: str = "full",
    pool_k: int = 8,
    score_preset: str = "balanced",
    indices: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
    stop_on_error: bool = False,
    save_artifacts: bool = True,
    dry_run: bool = False,
    version: str = "v1",
    config: Optional[ReactConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Roda a parte deterministica sobre toda `dataset` e devolve um resumo."""
    models = list(models) if models else list(DEFAULT_MODELS)
    if not models:
        raise ValueError("empty model pool")

    cfg = config or ReactConfig()
    cfg.name = version
    cfg.n_validation_windows = int(n_windows)
    cfg.backtest_mode = backtest_mode
    cfg.nested_selection = bool(nested_selection)
    cfg.min_windows_for_ols = int(min_windows_for_ols)
    cfg.final_strategy = "argmin"
    cfg.seed_stable_pools = bool(seed_stable_pools)
    cfg.pooled_meta_model = bool(pooled_meta_model)
    cfg.pooled_meta_model_min_series = int(pooled_meta_model_min_series)
    cfg.pooled_meta_model_objective = pooled_meta_model_objective
    cfg.seed_pooled_meta_model = bool(seed_pooled_meta_model)
    cfg.dataset_card = bool(dataset_card)
    cfg.pool_mode = pool_mode
    cfg.pool_k = int(pool_k)
    cfg.score_preset = score_preset
    cfg.calibration_gate = False

    # Sem LLM em nenhuma fase: build_client(LLMRole(model=None)) devolve None.
    cfg.combinator = LLMRole(model=None)
    cfg.diagnostician = LLMRole(model=None)
    cfg.reporter = LLMRole(model=None)

    experiment = f"orchestrator_baseline_{version}"
    out_dir = output_dir or results_dir

    todo = list(indices) if indices is not None else None
    if todo is None and limit:
        total = count_series(dataset, models[0], results_dir)
        todo = list(range(min(int(limit), total)))

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(f"dataset      : {dataset}")
    log(f"reading from : {results_dir}")
    log(f"writing to   : {os.path.join(out_dir, experiment)}")
    log(f"source       : {source_file or '(none - profile falls back to the windows)'}")
    log(f"models       : {len(models)}")
    log(f"ablation     : {cfg.fingerprint()}")
    log(
        f"pool mode    : {cfg.pool_mode} | windows: {cfg.n_validation_windows}"
        f" | backtest: {cfg.backtest_mode} | stable seeds: {cfg.seed_stable_pools}"
        f" | pooled: {cfg.pooled_meta_model}"
        f"{' seeded' if cfg.pooled_meta_model and cfg.seed_pooled_meta_model else ''}"
    )
    log("llm          : none (deterministic arm)")
    log(f"series       : {'all' if todo is None else todo}")
    if cfg.pooled_meta_model and todo is not None and len(todo) < cfg.pooled_meta_model_min_series:
        log(
            f"WARNING      : this call covers {len(todo)} series but "
            f"pooled_meta_model_min_series={cfg.pooled_meta_model_min_series}, so the pooled "
            f"meta-model will NOT be trained. Run the whole dataset in one call to keep it."
        )
    if dry_run:
        log("dry run      : nothing will be written")
    log("-" * 74)

    writer = (
        None
        if dry_run
        else ResultWriter(
            dataset=dataset,
            experiment=experiment,
            results_dir=out_dir,
            save_artifacts=save_artifacts,
        )
    )
    if writer is not None and os.path.getsize(writer.csv_path) > 0:
        try:
            with open(writer.csv_path, encoding="utf-8") as fh:
                n_existing = sum(1 for _ in fh) - 1
        except OSError:
            n_existing = 0
        if n_existing > 0:
            log(
                f"WARNING      : {writer.csv_path} already has {n_existing} rows and new rows "
                f"are APPENDED. Use a new --version or delete the file to avoid duplicates."
            )

    started = time.perf_counter()
    outcomes: List[Any] = []
    per_series: List[Dict[str, float]] = []
    ok = failed = 0
    failures: List[str] = []

    for outcome in _pipeline.run_dataset(
        models=models,
        dataset=dataset,
        source_file=source_file,
        config=cfg,
        source_dir=source_dir,
        results_dir=results_dir,
        indices=todo,
    ):
        outcomes.append(outcome)
        if outcome.success:
            ok += 1
            attempt = outcome.react.final_attempt
            metrics = compute_metrics(outcome.forecast, outcome.test_values)
            per_series.append(metrics)
            log(
                f"[{outcome.dataset_index:>4}] {attempt.spec['combine']:<14}"
                f" pool={len(outcome.selected_models()):>2}"
                f" score={attempt.score:7.4f} | TEST smape={metrics['smape']:.4f}"
                f" rmse={metrics['rmse']:.4f} pocid={metrics['pocid']:.2f}"
            )
        else:
            failed += 1
            failures.append(f"{outcome.dataset_index}: {outcome.error}")
            log(f"[{outcome.dataset_index:>4}] FAILED: {outcome.error}")

        if writer is not None:
            writer.write(outcome, regressor=experiment)

        if not outcome.success and stop_on_error:
            raise RuntimeError(f"stopped at dataset_index={outcome.dataset_index}: {outcome.error}")

    elapsed = time.perf_counter() - started
    log("-" * 74)
    log(f"done in {elapsed:.1f}s | ok: {ok} | failed: {failed}")

    summary_metrics = _summarise(per_series)
    if summary_metrics:
        log("")
        log(f"DATASET SUMMARY over {len(per_series)} series (mean across series):")
        log(
            f"  this run   smape={summary_metrics['smape']:.4f}"
            f"  rmse={summary_metrics['rmse']:.4f}"
            f"  pocid={summary_metrics['pocid']:.2f}"
            f"  mape={summary_metrics['mape']:.4f}"
            f"  mae={summary_metrics['mae']:.4f}"
        )
        for name, stats in _external_summary(outcomes).items():
            log(
                f"  {name:<10} smape={stats['smape']:.4f}  rmse={stats['rmse']:.4f}"
                f"  pocid={stats['pocid']:.2f}  mape={stats['mape']:.4f}"
            )
    if writer is not None:
        log(f"csv: {writer.csv_path}")
        if save_artifacts:
            log(f"artifacts: {writer.artifacts_dir}")
    for line in failures[:20]:
        log(f"  failure {line}")

    return {
        "dataset": dataset,
        "experiment": experiment,
        "ablation_config": cfg.fingerprint(),
        "n_ok": ok,
        "n_failed": failed,
        "failures": failures,
        "csv_path": writer.csv_path if writer else None,
        "artifacts_dir": writer.artifacts_dir if (writer and save_artifacts) else None,
        "elapsed_s": elapsed,
        "columns": COLS_SERIE,
        "outcomes": outcomes,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orchestrator without the agent: deterministic seeding only, best seed wins.",
    )
    p.add_argument("--dataset", required=True, help="results dataset name, e.g. ANP_MONTHLY")
    p.add_argument("--source", default=None, help=".tsf file, e.g. mes_11_venda_mensal.tsf")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                   help="where the per-model result CSVs are read from")
    p.add_argument("--output-dir", default=None, help="where to write (default: --results-dir)")
    p.add_argument("--version", default="v1", help="experiment folder suffix")
    p.add_argument("--models", nargs="+", default=None)

    p.add_argument("--config", default=None, help="JSON file with a ReactConfig")
    p.add_argument("--windows", type=int, default=3, help="validation windows (default 3)")
    p.add_argument("--backtest-mode", choices=["expanding", "loo"], default="expanding")
    p.add_argument("--no-nested-selection", action="store_true")
    p.add_argument("--min-windows-for-ols", type=int, default=5)
    p.add_argument("--no-stable-seeds", action="store_true",
                   help="seed only the three full-pool baselines")
    p.add_argument("--no-pooled-meta-model", action="store_true",
                   help="skip the cross-series meta-model pre-pass")
    p.add_argument("--pooled-meta-model-min-series", type=int, default=20)
    p.add_argument("--pooled-objective", choices=["fforma", "per_model"], default="fforma")
    p.add_argument("--no-seed-pooled", action="store_true",
                   help="do not seed weighted(pooled_meta_model)")
    p.add_argument("--with-dataset-card", action="store_true",
                   help="also run the cross-series strategy-prior pre-pass (only useful for the agent prompt)")
    p.add_argument("--pool-mode", choices=["full", "top_k_error", "top_k_stable"], default="full")
    p.add_argument("--pool-k", type=int, default=8)
    p.add_argument("--score-preset", default="balanced")

    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N series")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--no-artifacts", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ReactConfig.from_json_file(args.config) if args.config else None
    try:
        summary = exec_dataset_baselines(
            models=args.models,
            dataset=args.dataset,
            source_file=args.source,
            source_dir=args.source_dir,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            n_windows=args.windows,
            backtest_mode=args.backtest_mode,
            nested_selection=not args.no_nested_selection,
            min_windows_for_ols=args.min_windows_for_ols,
            seed_stable_pools=not args.no_stable_seeds,
            pooled_meta_model=not args.no_pooled_meta_model,
            pooled_meta_model_min_series=args.pooled_meta_model_min_series,
            pooled_meta_model_objective=args.pooled_objective,
            seed_pooled_meta_model=not args.no_seed_pooled,
            dataset_card=args.with_dataset_card,
            pool_mode=args.pool_mode,
            pool_k=args.pool_k,
            score_preset=args.score_preset,
            indices=args.indices,
            limit=args.limit,
            stop_on_error=args.stop_on_error,
            save_artifacts=not args.no_artifacts,
            dry_run=args.dry_run,
            version=args.version,
            config=config,
            verbose=not args.quiet,
        )
    except SeriesAlignmentError as exc:
        print(f"\nALIGNMENT ERROR - the .tsf does not match the results:\n  {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 1 if summary["n_failed"] and args.stop_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
