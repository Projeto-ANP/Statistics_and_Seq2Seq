"""Entry point for the forecast-combination orchestrator (ReAct architecture).

Same job as before, same calling convention: `exec_dataset_orchestrator(...)` loops
over every series of a dataset, decides how to combine the pool, and appends one row
per series to `<results>/<experiment>/<DATASET>.csv`.

What changed under the hood
---------------------------
The Proposer / Skeptic / Statistician / PatternAnalyst debate is gone. It is
replaced by a **single agent** cycling Thought -> Action -> Observation over a
closed catalog of 24 deterministic tools, where every strategy it proposes is
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
from orchestrator_react.csv_writer import COLS_SERIE, ResultWriter, compute_metrics
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
    nested_selection: bool = True,
    min_windows_for_ols: int = 5,
    max_iterations: int = 12,
    early_stop_patience: int = 4,
    final_strategy: str = "argmin",
    final_top_m: int = 3,
    seed_stable_pools: bool = True,
    pooled_meta_model: bool = True,
    pooled_meta_model_min_series: int = 20,
    pooled_meta_model_objective: str = "fforma",
    seed_pooled_meta_model: bool = True,
    dataset_card: bool = True,
    final_prior_alpha: float = 0.0,
    combinator_reasoning: Optional[bool] = None,
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
    max_llm_failures: int = 5,
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
        nested_selection: re-chooses pool membership (select_top_k / select_stable /
            prune_redundant) inside each backtest fold instead of once over all
            windows. Default True. Measured on 111 NN5 series: with this off, the
            validation score is *anti-predictive* of the blind test (Spearman
            -0.468 between in-sample validation rank and test rank across 16 fixed
            rules); with it on, Spearman is +0.547. Leave this on for every run
            that will be reported; the off state exists for the ablation itself,
            not as an alternative default. See `orchestrator_react/ARQUITETURA.md`
            section 2.1.
        max_iterations / early_stop_patience: the loop budget. Raised from 8/2 to
            12/4 because the v2 run stopped early in 83 of 111 series and the agent
            beat the seeded floor in only 43 — with a richer seed set the bar is
            higher, so it needs more room to clear it before giving up.
        final_strategy: how the attempt history becomes one forecast. "argmin"
            (default) applies the single best-scoring attempt; "ensemble"
            softmax-averages the top `final_top_m`. The ensemble is implemented and
            tested but is not the default: its gain overlaps `seed_stable_pools`
            and does not survive alongside it (0.11536 -> 0.11595, p=0.62). Kept
            as an ablation arm.
        seed_stable_pools: seed stability-selected combinations alongside the three
            full-pool baselines. This is the single largest measured lever — the
            deterministic floor goes from 0.12036 to 0.11500 mean sMAPE on NN5,
            past ADE (0.11780). False restores the pre-v3 seed set for the ablation.
        pooled_meta_model: trains `weights_pooled_meta_model` once for the whole
            dataset run — a gradient-boosted regressor per pool model, predicting
            its error from a series' historical shape (trend/seasonal strength,
            entropy, autocorrelation), fit leave-one-series-out across every other
            series. This is the classical-ML piece ADE and FFORMA both use; the
            existing `weights_feature_based` tries the same idea per series on 3
            windows, which can never clear its own "enough samples" guard — pooling
            across series is what gives it enough samples to be worth trying. False
            skips the pre-pass entirely (also the correct control arm to ablate
            against).
        pooled_meta_model_min_series: below this many series in the run, pooling
            has too little signal to be worth it and the tool is withheld for
            every series, the same way weights_ols is withheld under too few
            validation windows.
        pooled_meta_model_objective: "fforma" (default) trains ONE multi-class
            booster whose softmax output IS the weight vector, using Montero-Manso
            et al.'s custom gradient to minimise the COMBINED error — measured
            0.21596 sMAPE on the 182 ANP series, past the real FFORMA baseline
            (0.21659). "per_model" is the earlier design (N independent regressors
            predicting each model's own error, weights post-hoc); it scores 0.22047
            on ANP and slightly better than fforma on NN5, and stays as the
            ablation arm.
        seed_pooled_meta_model: seed `weighted(pooled_meta_model)` as a Phase 2
            baseline. Without it the agent essentially never reaches the tool: on
            the ANP v4 run it called it once in 182 series. Seeded, it wins the
            deterministic floor in 28% of series.
        dataset_card: inject a DATASET CARD in every turn prompt — how each seeded
            strategy scored on VALIDATION across the other N-1 series
            (leave-one-series-out). Cross-series context the agent never had, which
            is what ADE and FFORMA get by construction. Recommendation only; the
            catalog stays open.
        combinator_reasoning: passed to Ollama only when set. False asks gpt-oss to
            stop spending its budget in the harmony reasoning channel, which is the
            documented source of both failure modes seen on real runs: 90 empty
            replies across the 182-series ANP run, and the "error parsing tool call"
            transport errors where Ollama's template JSON-parses prose written into
            gpt-oss's tool-call channel (ollama/ollama#11781, #11800). Leave unset to
            keep the server default that every result so far used, so an A/B is a
            deliberate change. Never use format="json" instead: on gpt-oss it returns
            an empty response every time (#11867).
        final_prior_alpha: >0 switches Phase 4 to shrink each attempt's 3-window
            score toward that dataset prior before the argmin. This is the largest
            ANP lever found (0.21904 -> 0.21453 at alpha 0.8, past FFORMA) and it
            hurts NN5 monotonically over the same sweep (0.11539 -> 0.11879). No
            honest way to pick alpha was found — fixed, validation-selected and
            stability-gated all fail — so it defaults to 0.0 (off) and is an
            explicit opt-in per dataset.
        min_windows_for_ols: `weights_ols` needs more independent equations than a
            3-window backtest gives it — below this threshold the simplex
            projection collapses to a vertex, i.e. `weights_ols` silently turns
            into model *selection* whose winner need not be the lowest-error
            model. Below the threshold the tool is withheld from the catalog
            before the prompt is built, so the agent never spends an iteration on
            it. See `orchestrator_react/ARQUITETURA.md` section 4.1.
        indices / limit: run a subset, for smoke runs.
        log_agent_steps: print every Thought / Action / Observation as it happens.
            A run with an agent is mostly waiting, and the interesting part is what
            it decided to look at; without this the log only shows the conclusion.
        max_llm_failures: how many series may fail before the run aborts. The
            failure mode this exists for is real: an isolated gpt-oss glitch on ONE
            series used to end a 182-series run, and resuming the remainder in a
            small chunk silently disabled the pooled meta-model for that chunk (see
            EXTRA_DEPENDENCIES.txt 7b). With a budget, one bad series costs one row
            — clearly marked in the log and the CSV — while a genuinely broken
            server still stops the run within a few series. Ignored when
            `allow_baseline_fallback=True` (never aborts).
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
    cfg.nested_selection = bool(nested_selection)
    cfg.min_windows_for_ols = int(min_windows_for_ols)
    cfg.max_iterations = int(max_iterations)
    cfg.early_stop_patience = int(early_stop_patience)
    cfg.final_strategy = final_strategy
    cfg.final_top_m = int(final_top_m)
    cfg.seed_stable_pools = bool(seed_stable_pools)
    cfg.pooled_meta_model = bool(pooled_meta_model)
    cfg.pooled_meta_model_min_series = int(pooled_meta_model_min_series)
    cfg.pooled_meta_model_objective = pooled_meta_model_objective
    cfg.seed_pooled_meta_model = bool(seed_pooled_meta_model)
    cfg.dataset_card = bool(dataset_card)
    cfg.final_prior_alpha = float(final_prior_alpha)
    cfg.pool_mode = pool_mode
    cfg.pool_k = int(pool_k)
    cfg.score_preset = score_preset
    cfg.show_attempt_history = bool(show_attempt_history)
    cfg.calibration_gate = bool(calibration_gate)

    cfg.combinator = _as_role(combinator_model)
    if combinator_reasoning is not None:
        cfg.combinator.reasoning = bool(combinator_reasoning)
    cfg.diagnostician = _as_role(diagnostician_model)
    cfg.reporter = _as_role(reporter_model)

    # Environment variables win, so a model can be swapped without editing code.
    # (This is also why there is no separate `diagnostic_llm` flag: it used to be
    # computed here, one line above where an env var could still change
    # `cfg.diagnostician.model` — so it could go stale relative to the thing it
    # was supposed to summarise. `cfg.diagnostician.enabled` is now checked live,
    # everywhere, instead of snapshotted.)
    cfg = ReactConfig.from_env(cfg)

    if not use_llm:
        cfg.combinator = LLMRole(model=None)
        cfg.diagnostician = LLMRole(model=None)
        cfg.reporter = LLMRole(model=None)

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
        f"final        : {cfg.final_strategy}"
        f"{' top_m=' + str(cfg.final_top_m) if cfg.final_strategy == 'ensemble' else ''}"
        f" | stable seeds: {cfg.seed_stable_pools}"
        f" | pooled: {cfg.pooled_meta_model}"
        f"{'/' + cfg.pooled_meta_model_objective if cfg.pooled_meta_model else ''}"
        f"{' seeded' if cfg.seed_pooled_meta_model else ''}"
        f" | card: {cfg.dataset_card}"
        f"{f' | prior_alpha={cfg.final_prior_alpha}' if cfg.final_prior_alpha else ''}"
        f"{f' (min {cfg.pooled_meta_model_min_series} series)' if cfg.pooled_meta_model else ''}"
        f" | budget: {cfg.max_iterations} iters, patience {cfg.early_stop_patience}"
    )
    log(
        f"protocol     : nested_selection={cfg.nested_selection}"
        f" | min_windows_for_ols={cfg.min_windows_for_ols}"
        f"{' (weights_ols WITHHELD this run)' if cfg.n_validation_windows < cfg.min_windows_for_ols else ''}"
    )
    log(
        f"llm          : combinator={cfg.combinator.label()}"
        f"{' reasoning=' + str(cfg.combinator.reasoning) if cfg.combinator.reasoning is not None else ''} "
        f"diagnostician={cfg.diagnostician.label()} reporter={cfg.reporter.label()}"
    )
    log(f"series       : {'all' if todo is None else todo}")
    # The pooled meta-model trains on the series of THIS call, so a resume chunk is
    # a smaller dataset as far as the pre-pass is concerned. Say it up front rather
    # than let it be discovered in the artifacts afterwards (which is how the ANP v4
    # run ended up with the tool withheld on its last 17 series).
    if cfg.pooled_meta_model and todo is not None and len(todo) < cfg.pooled_meta_model_min_series:
        log(
            f"WARNING      : this call covers {len(todo)} series but "
            f"pooled_meta_model_min_series={cfg.pooled_meta_model_min_series}, so the "
            f"pooled meta-model will NOT be trained and weights_pooled_meta_model "
            f"will be withheld for every series in this run. Run the whole dataset in "
            f"one call, or lower the minimum, to keep the architecture identical "
            f"across series."
        )
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
    per_series: List[Dict[str, float]] = []
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
            metrics = compute_metrics(outcome.forecast, outcome.test_values)
            per_series.append(metrics)
            prov = outcome.state.verify_provenance() if outcome.state else {}
            log(
                f"[{outcome.dataset_index:>4}] {attempt.spec['combine']:<14}"
                f" score={attempt.score:7.4f} origin={attempt.origin:<8}"
                f" iters={outcome.react.iterations_used} stop={outcome.react.stop_reason}"
            )
            if cfg.diagnostician.enabled:
                diag = outcome.diagnosis or {}
                if diag.get("source") == "llm":
                    log(
                        f"         DIAGNOSIS llm({diag.get('model', '?')})"
                        f" regime={diag.get('regime')} predictability={diag.get('predictability')}"
                        f" hint={diag.get('combination_hint')}"
                    )
                else:
                    # Phase 1 degrades instead of failing (a missing reading must
                    # never cost a series), so a fallback here is expected sometimes
                    # — but if EVERY series falls back, the ablation ran without the
                    # LLM ever actually answering, and the log needs to say so
                    # instead of looking identical to a real llm-source run.
                    log(
                        f"         DIAGNOSIS deterministic (fallback: "
                        f"{diag.get('fallback_reason', 'no llm configured')})"
                    )
            log(
                f"         TEST  smape={metrics['smape']:.4f}  rmse={metrics['rmse']:.4f}"
                f"  pocid={metrics['pocid']:.2f}  mape={metrics['mape']:.4f}"
            )
            log(f"         FORECAST {[round(v, 3) for v in outcome.forecast]}")
            log(
                f"         PROVENANCE tools={prov.get('n_tool_calls', 0)}"
                f" evaluate_calls={prov.get('n_evaluate_calls', 0)}"
                f" pool={len(outcome.selected_models())}"
                f" effective={len(outcome.effective_models())}"
                f" ok={prov.get('provenance_ok')}"
            )
            red = outcome.reducibility()
            if red.get("equivalent_to_pool_mean"):
                log(
                    "         NOTE this strategy is numerically the MEAN of its own pool"
                    f" (max relative difference {red['pool_mean_relative_diff']})"
                )
            if not prov.get("provenance_ok", True):
                log("         WARNING: provenance check FAILED for this series")
            if outcome.react.stop_reason == "llm_error":
                llm_failures += 1
                detail = outcome.react.errors[-1] if outcome.react.errors else "no detail recorded"
                budget = "unlimited" if allow_baseline_fallback else str(max_llm_failures)
                message = (
                    f"THE AGENT FAILED ON dataset_index={outcome.dataset_index} and the row "
                    f"below is a DETERMINISTIC BASELINE, not an agent result.\n"
                    f"  error: {detail}\n"
                    f"  model: {cfg.combinator.label()} at {cfg.combinator.base_url}\n"
                    f"  failures so far: {llm_failures} (tolerated: {budget})"
                )
                print(message, flush=True)
                # Abort only once the failures stop looking isolated. Aborting on the
                # FIRST one — the old behaviour — meant a single flaky series ended a
                # 182-series run and forced a resume, and resuming in a chunk smaller
                # than `pooled_meta_model_min_series` silently changes the
                # architecture for that chunk. A budget keeps the original guarantee
                # (a systematically broken server still stops the run, loudly) while
                # letting one bad series cost one row instead of the whole dataset.
                if not allow_baseline_fallback and llm_failures >= int(max_llm_failures):
                    if writer is not None:
                        writer.write(outcome, regressor=experiment)
                    raise RuntimeError(
                        message
                        + f"\n  ABORTING: {llm_failures} series have now failed, which is "
                        f"the max_llm_failures budget. This no longer looks like an "
                        f"isolated glitch. Raise max_llm_failures to tolerate more, or "
                        f"use allow_baseline_fallback=True to never abort."
                    )
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
            better = "  <- this run is better" if stats["rmse"] > summary_metrics["rmse"] else ""
            log(
                f"  {name:<10} smape={stats['smape']:.4f}  rmse={stats['rmse']:.4f}"
                f"  pocid={stats['pocid']:.2f}  mape={stats['mape']:.4f}{better}"
            )
        wins = sum(1 for o in outcomes if o.success and o.react
                   and o.react.final_attempt and o.react.final_attempt.origin == "agent")
        log(f"  strategy chosen by the agent in {wins}/{ok} series")
        verdicts: Dict[str, int] = {}
        for o in outcomes:
            if o.success and o.state:
                v = o.state.selection_confidence().get("verdict", "?")
                verdicts[v] = verdicts.get(v, 0) + 1
        if verdicts:
            log(f"  selection verdict: {verdicts}")
        if cfg.diagnostician.enabled:
            n_llm = sum(1 for o in outcomes if o.success and (o.diagnosis or {}).get("source") == "llm")
            log(f"  diagnosis answered by the llm in {n_llm}/{ok} series"
                f"{' (WARNING: 0 — check the model/server, the ablation ran with no llm reading at all)' if ok and n_llm == 0 else ''}")
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


def _summarise(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across series, ignoring the ones that could not be computed."""
    if not rows:
        return {}
    keys = ("mape", "pocid", "smape", "rmse", "msmape", "mae")
    out: Dict[str, float] = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out


def _external_summary(outcomes: Sequence[Any]) -> Dict[str, Dict[str, float]]:
    """The same means for the baselines already on disk, over the same series.

    Comparable because every method is evaluated on the identical blind window.
    """
    collected: Dict[str, List[Dict[str, float]]] = {}
    for outcome in outcomes:
        for name, stats in (outcome.external_baselines or {}).items():
            if isinstance(stats, dict) and stats.get("available"):
                collected.setdefault(name, []).append(stats)
    return {name: _summarise(rows) for name, rows in collected.items() if rows}


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
    p.add_argument("--no-nested-selection", action="store_true",
                   help="ablation only: disable per-fold pool re-selection. Leaving this "
                        "on (the default) is what makes the validation score predictive "
                        "of the test window; see ARQUITETURA.md section 2.1")
    p.add_argument("--min-windows-for-ols", type=int, default=5,
                   help="weights_ols is withheld from the catalog below this many "
                        "validation windows (default 5; --windows 3 withholds it)")
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--final-strategy", choices=["argmin", "ensemble"], default="argmin",
                   help="argmin (default) applies the best-scoring attempt; ensemble "
                        "averages the top --final-top-m (ablation arm)")
    p.add_argument("--final-top-m", type=int, default=3)
    p.add_argument("--no-stable-seeds", action="store_true",
                   help="ablation: seed only the three full-pool baselines")
    p.add_argument("--no-pooled-meta-model", action="store_true",
                   help="ablation: skip the cross-series meta-model pre-pass "
                        "(weights_pooled_meta_model is withheld for every series)")
    p.add_argument("--pooled-meta-model-min-series", type=int, default=20,
                   help="withhold weights_pooled_meta_model below this many series "
                        "in the run (default 20)")
    p.add_argument("--pooled-objective", choices=["fforma", "per_model"], default="fforma",
                   help="fforma (default): one booster, softmax output IS the weights, "
                        "trained on the combined error. per_model: the earlier design "
                        "(ablation arm)")
    p.add_argument("--no-seed-pooled", action="store_true",
                   help="ablation: do not seed weighted(pooled_meta_model) in Phase 2")
    p.add_argument("--no-dataset-card", action="store_true",
                   help="ablation: do not show the cross-series dataset card in the prompt")
    p.add_argument("--no-reasoning", action="store_true",
                   help="ask gpt-oss to skip the harmony reasoning channel (the source "
                        "of the empty replies and tool-call parse errors). A/B only: "
                        "every result so far used the server default")
    p.add_argument("--final-prior-alpha", type=float, default=0.0,
                   help="shrink each attempt's score toward the dataset prior before "
                        "the final pick. 0=off. Measured best on ANP around 0.6-0.8; "
                        "hurts NN5 at any value >0 (see ARQUITETURA.md)")
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
    p.add_argument("--max-llm-failures", type=int, default=5,
                   help="abort after this many series fail with an LLM error "
                        "(default 5). One flaky series no longer ends the run; a dead "
                        "server still stops it quickly")
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
            nested_selection=not args.no_nested_selection,
            min_windows_for_ols=args.min_windows_for_ols,
            max_iterations=args.max_iterations,
            early_stop_patience=args.early_stop_patience,
            final_strategy=args.final_strategy,
            final_top_m=args.final_top_m,
            seed_stable_pools=not args.no_stable_seeds,
            pooled_meta_model=not args.no_pooled_meta_model,
            pooled_meta_model_min_series=args.pooled_meta_model_min_series,
            pooled_meta_model_objective=args.pooled_objective,
            seed_pooled_meta_model=not args.no_seed_pooled,
            dataset_card=not args.no_dataset_card,
            final_prior_alpha=args.final_prior_alpha,
            combinator_reasoning=False if args.no_reasoning else None,
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
            max_llm_failures=args.max_llm_failures,
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
    # Command-line arguments win. Without this dispatch the block below ran no
    # matter what was typed, so every flag `build_parser` defines — including
    # --dataset itself — was silently ignored and the hardcoded NN5 run happened
    # instead. Running with no arguments still uses the editable block, which is
    # how this file has always been driven on the server.
    if len(sys.argv) > 1:
        raise SystemExit(main())

    models = DEFAULT_MODELS

    dataset = "NN5_WEEKLY_DATASET"

    # The cheapest ablation-2 experiment: does a dedicated Phase 1 reading (a
    # smaller/faster model, since it only has to read two cards and answer one
    # categorical judgement, not run the combination loop) change anything versus
    # the deterministic profile? Nothing else below should differ between the two
    # runs, or the comparison stops being apples-to-apples.
    #
    #   baseline (today's default): DIAGNOSTICIAN = None, version = "v4_baseline"
    #   this experiment:            DIAGNOSTICIAN = "qwen3:8b" (or whatever you
    #                                have pulled), version = "v4_diagnostician"
    #
    # Run both with the same everything else, then compare the two CSVs' mean
    # sMAPE/rmse/pocid and, per series, `outcome.diagnosis["source"]` (the log
    # below already prints a DIAGNOSIS line per series, and a dataset-level
    # "diagnosis answered by the llm in N/ok series" line at the end — if that
    # number is 0, the ablation ran without the LLM ever actually answering, and
    # the comparison would be meaningless).
    DIAGNOSTICIAN: Optional[str] = None  # None = baseline; set e.g. "qwen3:8b" for the ablation

    exec_dataset_orchestrator(
        models,
        dataset=dataset,
        source_file="nn5_weekly_dataset.tsf",
        use_llm=True,
        combinator_model=LLMRole(model="gpt-oss:20b", temperature=0.2),
        diagnostician_model=LLMRole(model=DIAGNOSTICIAN) if DIAGNOSTICIAN else None,
        # reporter_model=LLMRole(model="qwen3:8b"),        # Phase 5: prose justification
        n_windows=3,
        max_iterations=12,
        llm_logs=True,
        version="v4_diagnostician" if DIAGNOSTICIAN else "v4_baseline",
    )
