"""Entry point for the forecast-combination orchestrator (ReAct architecture).

Same job as before, same calling convention: `exec_dataset_orchestrator(...)` loops
over every series of a dataset, decides how to combine the pool, and appends one row
per series to `<results>/<experiment>/<DATASET>.csv`.

What changed under the hood
---------------------------
The Proposer / Skeptic / Statistician / PatternAnalyst debate is gone. It is
replaced by a **single agent** cycling Thought -> Action -> Observation over a
closed catalog of 22 deterministic tools, where every strategy it proposes is
backtested on the validation windows before it can be accepted.

    Phase 0  ingestion            per-model CSVs + the original .tsf series
    Phase 1  diagnosis            deterministic profile, optional LLM reading
    Phase 2  pool evaluation      error table, stability, redundancy, and the
                                  mean/median/DBA baselines seeded first
    Phase 3  ReAct loop           the single combiner agent
    Phase 4  application          winning strategy applied to the test forecasts
    Phase 5  report               optional natural-language justification

The previous version of this file is preserved in git:

    git show 3fcfc122:run_tsf_orchestrator.py

Signature changes from the old call
-----------------------------------
    proposer_model / skeptic_model / statistician_model / pattern_analyst_model
        -> combinator_model      (one agent instead of four)

    train_window=3               -> n_windows=3
        The old name counted rows including the test row and sliced
        `iloc[-train_window:-1]`, so `train_window=3` actually gave TWO validation
        windows. `n_windows` is the number of validation windows directly, so
        `n_windows=3` matches the three windows the mean/median/dba/ADE/FFORMA
        baselines on disk were computed with.

    rolling="expanding"          -> backtest_mode="expanding" | "loo"

    (new)                        -> source_file="ETTh1.tsf"
        The .tsf the base models were trained on, looked up in `source_dir`. Each
        series is checked against the forecast CSVs at load time, so a wrong file
        fails loudly instead of silently pairing the wrong series.

Examples
--------
    from orchestrator_react.config import LLMRole

    exec_dataset_orchestrator(
        models,
        dataset="ANP_MONTHLY",
        source_file="mes_11_venda_mensal.tsf",
        combinator_model=LLMRole(model="gpt-oss:20b", temperature=0.2),
        version="react_v1",
    )

Command line:

    python run_tsf_orchestrator.py --dataset ANP_MONTHLY \\
        --source mes_11_venda_mensal.tsf --combinator gpt-oss:20b --version react_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator_react import pipeline as _pipeline
from orchestrator_react.config import LLMRole, ReactConfig
from orchestrator_react.csv_writer import COLS_SERIE, ResultWriter
from orchestrator_react.data_source import DEFAULT_SOURCE_DIR, SeriesAlignmentError
from orchestrator_react.ingest import DEFAULT_RESULTS_DIR, count_series
from orchestrator_react.llm import build_client, check_client


#: The 19 models used by the previous runs. Kept as the default so a call that only
#: passes `models` behaves as before.
DEFAULT_MODELS: List[str] = [
    "ARIMA",
    "ETS",
    "THETA",
    # "ridge",
    "rf",
    "catboost",
    # "CWT_ridge",
    # "DWT_ridge",
    # "FT_ridge",
    "CWT_rf",
    "DWT_rf",
    "FT_rf",
    "CWT_catboost",
    "DWT_catboost",
    "FT_catboost",
    "ONLY_CWT_catboost",
    "ONLY_CWT_rf",
    # "ONLY_CWT_ridge",
    "ONLY_DWT_catboost",
    "ONLY_DWT_rf",
    # "ONLY_DWT_ridge",
    "ONLY_FT_catboost",
    "ONLY_FT_rf",
    # "ONLY_FT_ridge",
    "NaiveSeasonal",
    "NaiveMovingAverage",
]

#: Parameters of the old debate architecture. Passing one raises with an
#: explanation rather than a bare TypeError, because an old call pasted from a
#: notebook is the most likely way to hit this.
_RETIRED = {
    "proposer_model": "the four debate agents became one combinator; use combinator_model",
    "skeptic_model": "the four debate agents became one combinator; use combinator_model",
    "statistician_model": "the four debate agents became one combinator; use combinator_model",
    "pattern_analyst_model": (
        "the PatternAnalyst is gone; trend/seasonality champions are now computed "
        "deterministically inside series_profile()"
    ),
    "debate": "there is no debate any more",
    "debate_auto": "there is no debate any more",
    "debate_margin": "there is no debate any more",
    "require_tool_call": "tool use is structural now: the agent can only act through the catalog",
    "ollama_model": "use combinator_model",
    "train_window": (
        "renamed to n_windows, and it now counts validation windows directly "
        "(old train_window=4 == new n_windows=3)"
    ),
    "rolling": "renamed to backtest_mode, with values 'expanding' or 'loo'",
    "debug": "use llm_logs",
}


def _as_role(value: Any, default_temperature: float = 0.2) -> LLMRole:
    """Accepts an `LLMRole`, the old `ModelConfig`, a plain string, or None."""
    if value is None:
        return LLMRole(model=None)
    if isinstance(value, LLMRole):
        return value
    if isinstance(value, str):
        return LLMRole(
            model=None if value.strip().lower() in {"", "none"} else value,
            temperature=default_temperature,
        )
    model = getattr(value, "model", None)  # duck-types orchestrator.utils.ModelConfig
    if model is None:
        raise TypeError(f"cannot read a model name from {value!r}")
    return LLMRole(
        model=str(model),
        temperature=float(getattr(value, "temperature", default_temperature)),
    )


def exec_dataset_orchestrator(
    models: Optional[Sequence[str]] = None,
    dataset: str = "ANP_MONTHLY",
    source_file: Optional[str] = None,
    *,
    use_llm: bool = True,
    combinator_model: Any = "gpt-oss:20b",
    diagnostician_model: Any = None,
    reporter_model: Any = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    output_dir: Optional[str] = None,
    n_windows: int = 3,
    backtest_mode: str = "expanding",
    max_iterations: int = 8,
    early_stop_patience: int = 2,
    pool_mode: str = "full",
    pool_k: int = 8,
    score_preset: str = "balanced",
    show_attempt_history: bool = True,
    calibration_gate: bool = False,
    indices: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
    llm_logs: bool = True,
    log_agent_steps: bool = True,
    stop_on_error: bool = False,
    allow_baseline_fallback: bool = False,
    save_artifacts: bool = True,
    dry_run: bool = False,
    version: str = "react_v1",
    config: Optional[ReactConfig] = None,
    **retired: Any,
) -> Dict[str, Any]:
    """Runs the orchestrator over every series of `dataset`.

    Args:
        models: the pool. Defaults to the 19 models used so far.
        dataset: results dataset name, e.g. "ANP_MONTHLY".
        source_file: the `.tsf` under `source_dir`, e.g. "ETTh1.tsf". Without it
            there is no historical series and `series_profile()` falls back to the
            validation windows.
        use_llm: False runs the deterministic arm — the best seeded baseline, no
            agent. That is the control arm of the ablations.
        combinator_model: the Phase 3 agent. Accepts an `LLMRole`, the old
            `ModelConfig`, or a plain model name.
        diagnostician_model: enables Phase 1 with an LLM (ablation 2).
        reporter_model: enables Phase 5.
        n_windows: validation windows. 3 matches the baselines already on disk.
        indices / limit: run a subset, for smoke runs.
        log_agent_steps: print every Thought / Action / Observation as it happens.
            A run with an agent is mostly waiting, and the interesting part is what
            it decided to look at; without this the log only shows the conclusion.
        allow_baseline_fallback: what to do when a configured agent does not answer.
            Default False: raise, with the underlying error. The point of this
            architecture is the agent, so a run that silently degrades to the
            deterministic baseline is a run that answers a different question while
            the log still reads `ok`. Set True only when you deliberately want the
            baseline arm on an unreliable server; `use_llm=False` is the clean way
            to ask for the deterministic arm.
        output_dir: write somewhere other than `results_dir`, so a smoke run does
            not touch the results tree.
        config: a fully built `ReactConfig`; the keyword arguments are applied on
            top of it.

    Returns:
        A summary dict with the counts, the CSV path and the per-series outcomes.

    Raises:
        TypeError: a parameter of the old debate architecture was passed.
        SeriesAlignmentError: the `.tsf` is not the file the forecasts came from.
    """
    for name in retired:
        raise TypeError(
            f"`{name}` no longer exists: "
            f"{_RETIRED.get(name, 'removed together with the debate architecture')}"
        )

    models = list(models) if models else list(DEFAULT_MODELS)
    if not models:
        raise ValueError("empty model pool")

    cfg = config or ReactConfig()
    cfg.name = version
    cfg.n_validation_windows = int(n_windows)
    cfg.backtest_mode = backtest_mode
    cfg.max_iterations = int(max_iterations)
    cfg.early_stop_patience = int(early_stop_patience)
    cfg.pool_mode = pool_mode
    cfg.pool_k = int(pool_k)
    cfg.score_preset = score_preset
    cfg.show_attempt_history = bool(show_attempt_history)
    cfg.calibration_gate = bool(calibration_gate)

    cfg.combinator = _as_role(combinator_model)
    cfg.diagnostician = _as_role(diagnostician_model)
    cfg.reporter = _as_role(reporter_model)
    cfg.diagnostic_llm = cfg.diagnostician.enabled

    # Environment variables win, so a model can be swapped without editing code.
    cfg = ReactConfig.from_env(cfg)

    if not use_llm:
        cfg.combinator = LLMRole(model=None)
        cfg.diagnostician = LLMRole(model=None)
        cfg.reporter = LLMRole(model=None)
        cfg.diagnostic_llm = False

    experiment = f"orchestrator_react_{version}"
    out_dir = output_dir or results_dir

    todo = list(indices) if indices is not None else None
    if todo is None and limit:
        total = count_series(dataset, models[0], results_dir)
        todo = list(range(min(int(limit), total)))

    def log(message: str) -> None:
        if llm_logs:
            print(message, flush=True)

    log(f"dataset      : {dataset}")
    log(f"reading from : {results_dir}")
    log(f"writing to   : {os.path.join(out_dir, experiment)}")
    log(f"source       : {source_file or '(none - profile falls back to the windows)'}")
    log(f"models       : {len(models)}")
    log(f"ablation     : {cfg.fingerprint()}")
    log(
        f"pool mode    : {cfg.pool_mode} | windows: {cfg.n_validation_windows}"
        f" | backtest: {cfg.backtest_mode}"
    )
    log(
        f"llm          : combinator={cfg.combinator.label()} "
        f"diagnostician={cfg.diagnostician.label()} reporter={cfg.reporter.label()}"
    )
    log(f"series       : {'all' if todo is None else todo}")
    if dry_run:
        log("dry run      : nothing will be written")
    log("-" * 74)

    # ── preflight ────────────────────────────────────────────────────────────
    # A server that is down, or a model that was never pulled, fails identically on
    # every series. Catching it here costs one call; not catching it costs a whole
    # run of silent deterministic fallback.
    if cfg.combinator.enabled:
        ok_llm, detail = check_client(build_client(cfg.combinator))
        if ok_llm:
            log(f"llm preflight: OK ({detail!r})")
        else:
            message = (
                "THE COMBINATOR LLM DID NOT ANSWER\n"
                f"  error    : {detail}\n"
                f"  model    : {cfg.combinator.label()}\n"
                f"  base_url : {cfg.combinator.base_url}\n"
                "  check    : is ollama up?  curl " + cfg.combinator.base_url + "/api/tags\n"
                f"             is the model pulled?  ollama list | grep {str(cfg.combinator.model).split(':')[0]}\n"
                "             is the port right?  export REACT_OLLAMA_URL=http://127.0.0.1:11434\n"
                "  note     : to run the deterministic arm on purpose use use_llm=False;\n"
                "             to run anyway and fall back per series use allow_baseline_fallback=True."
            )
            # printed as well as raised, so it lands in a log redirected with `>` alone
            print(message, flush=True)
            if not allow_baseline_fallback:
                raise RuntimeError(message)
            log("WARNING: continuing with the deterministic baseline per series")
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

    def on_step(index: Optional[int], entry: Dict[str, Any]) -> None:
        """One line per Thought, one per Action, one per Observation."""
        if not (llm_logs and log_agent_steps):
            return
        tag = f"[{index if index is not None else '?':>4}]"
        args = entry.get("action_args") or {}
        rendered = json.dumps(args, ensure_ascii=False, default=str)
        if len(rendered) > 110:
            rendered = rendered[:107] + "..."
        thought = " ".join(str(entry.get("thought") or "").split())
        if len(thought) > 200:
            thought = thought[:197] + "..."

        print(f"{tag}   iter {entry['iteration']} | {entry['action']} {rendered}", flush=True)
        if thought:
            print(f"{tag}     think: {thought}", flush=True)
        print(f"{tag}     obs  : {entry.get('observation_summary', '')}", flush=True)

    started = time.perf_counter()
    outcomes: List[Any] = []
    ok = failed = llm_failures = 0
    failures: List[str] = []

    for outcome in _pipeline.run_dataset(
        models=models,
        dataset=dataset,
        source_file=source_file,
        config=cfg,
        source_dir=source_dir,
        results_dir=results_dir,
        indices=todo,
        on_step=on_step,
    ):
        outcomes.append(outcome)
        if outcome.success:
            ok += 1
            attempt = outcome.react.final_attempt
            log(
                f"[{outcome.dataset_index:>4}] {attempt.spec['combine']:<14}"
                f" score={attempt.score:7.4f} origin={attempt.origin:<8}"
                f" iters={outcome.react.iterations_used} stop={outcome.react.stop_reason}"
            )
            if outcome.react.stop_reason == "llm_error":
                llm_failures += 1
                detail = outcome.react.errors[-1] if outcome.react.errors else "no detail recorded"
                message = (
                    f"THE AGENT FAILED ON dataset_index={outcome.dataset_index} and the row "
                    f"below is a DETERMINISTIC BASELINE, not an agent result.\n"
                    f"  error: {detail}\n"
                    f"  model: {cfg.combinator.label()} at {cfg.combinator.base_url}"
                )
                print(message, flush=True)
                if not allow_baseline_fallback:
                    if writer is not None:
                        writer.write(outcome, regressor=experiment)
                    raise RuntimeError(message)
        else:
            failed += 1
            failures.append(f"{outcome.dataset_index}: {outcome.error}")
            log(f"[{outcome.dataset_index:>4}] FAILED: {outcome.error}")

        if writer is not None:
            writer.write(outcome, regressor=experiment)

        if not outcome.success and stop_on_error:
            raise RuntimeError(
                f"stopped at dataset_index={outcome.dataset_index}: {outcome.error}"
            )

    elapsed = time.perf_counter() - started
    log("-" * 74)
    log(f"done in {elapsed:.1f}s | ok: {ok} | failed: {failed}")
    if llm_failures:
        log(
            f"WARNING: the agent failed on {llm_failures}/{ok + failed} series; those rows "
            "are deterministic baselines, not agent results."
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
        "n_llm_failures": llm_failures,
        "failures": failures,
        "csv_path": writer.csv_path if writer else None,
        "artifacts_dir": writer.artifacts_dir if (writer and save_artifacts) else None,
        "elapsed_s": elapsed,
        "columns": COLS_SERIE,
        "outcomes": outcomes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# command line
# ──────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Forecast combination with a single ReAct agent over deterministic tools.",
    )
    p.add_argument("--dataset", required=True, help="results dataset name, e.g. ANP_MONTHLY")
    p.add_argument("--source", default=None, help=".tsf file, e.g. mes_11_venda_mensal.tsf")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                   help="where the per-model result CSVs are read from")
    p.add_argument("--output-dir", default=None, help="where to write (default: --results-dir)")
    p.add_argument("--version", default="react_v1", help="experiment folder suffix")
    p.add_argument("--models", nargs="+", default=None)

    p.add_argument("--config", default=None, help="JSON file with a ReactConfig")
    p.add_argument("--windows", type=int, default=3, help="validation windows (default 3)")
    p.add_argument("--backtest-mode", choices=["expanding", "loo"], default="expanding")
    p.add_argument("--max-iterations", type=int, default=8)
    p.add_argument("--pool-mode", choices=["full", "top_k_error", "top_k_stable"], default="full")
    p.add_argument("--pool-k", type=int, default=8)
    p.add_argument("--score-preset", default="balanced")
    p.add_argument("--no-history", action="store_true", help="ablation 4: hide the attempt history")
    p.add_argument("--calibration-gate", action="store_true")

    p.add_argument("--combinator", default="gpt-oss:20b", help="Phase 3 agent")
    p.add_argument("--diagnostician", default=None, help="Phase 1 LLM (ablation 2)")
    p.add_argument("--reporter", default=None, help="Phase 5 LLM")
    p.add_argument("--no-llm", action="store_true", help="deterministic arm: no agent at all")

    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N series")
    p.add_argument("--quiet-agent", action="store_true",
                   help="do not print the agent's Thought/Action/Observation per turn")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--allow-baseline-fallback", action="store_true",
                   help="do not abort when the agent fails; fall back to the baseline "
                        "per series (off by default: a silent fallback is worse than a crash)")
    p.add_argument("--no-artifacts", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ReactConfig.from_json_file(args.config) if args.config else None
    try:
        summary = exec_dataset_orchestrator(
            models=args.models,
            dataset=args.dataset,
            source_file=args.source,
            use_llm=not args.no_llm,
            combinator_model=args.combinator,
            diagnostician_model=args.diagnostician,
            reporter_model=args.reporter,
            source_dir=args.source_dir,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            n_windows=args.windows,
            backtest_mode=args.backtest_mode,
            max_iterations=args.max_iterations,
            pool_mode=args.pool_mode,
            pool_k=args.pool_k,
            score_preset=args.score_preset,
            show_attempt_history=not args.no_history,
            calibration_gate=args.calibration_gate,
            indices=args.indices,
            limit=args.limit,
            log_agent_steps=not args.quiet_agent,
            stop_on_error=args.stop_on_error,
            allow_baseline_fallback=args.allow_baseline_fallback,
            save_artifacts=not args.no_artifacts,
            dry_run=args.dry_run,
            version=args.version,
            config=config,
        )
    except SeriesAlignmentError as exc:
        print(f"\nALIGNMENT ERROR - the .tsf does not match the results:\n  {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 1 if summary["n_failed"] and args.stop_on_error else 0


if __name__ == "__main__":
    models = DEFAULT_MODELS

    dataset = "NN5_WEEKLY_DATASET"
    exec_dataset_orchestrator(
        models,
        dataset=dataset,
        source_file="nn5_weekly_dataset.tsf",
        use_llm=True,
        combinator_model=LLMRole(model="gpt-oss:20b", temperature=0.2),
        # diagnostician_model=LLMRole(model="qwen3:8b"),   # ablation 2: Phase 1 with an LLM
        # reporter_model=LLMRole(model="qwen3:8b"),        # Phase 5: prose justification
        n_windows=3,
        max_iterations=8,
        llm_logs=True,
        version="react_v1",
    )
