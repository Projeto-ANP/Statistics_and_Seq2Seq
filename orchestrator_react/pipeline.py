"""Per-series orchestration: Phase 0 -> Phase 4, and the CSV payload.

Phase 4 itself needs no new code — `ReactState.apply_to_test` runs the winning
strategy through `combiners.apply_combination`, the very function the backtest used.
What this module adds is the wiring and the traceability package: given a
`dataset_index`, it must be possible to reconstruct which profile was computed,
which tools were called in which order, which weights were used, and why the agent
accepted that strategy.

Phases 1 and 5 (the optional LLM diagnosis and report) are accepted here as
injectable callables so Step 6.6 can fill them in without touching this file.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from orchestrator_react import ingest as ingest_mod
from orchestrator_react import meta_model as meta_model_mod
from orchestrator_react import phases as phases_mod
from orchestrator_react import pool as pool_mod
from orchestrator_react import tools as T
from orchestrator_react.config import ReactConfig
from orchestrator_react.data_source import (
    DEFAULT_SOURCE_DIR,
    SeriesAlignmentError,
    SeriesSource,
    load_series_source,
)
from orchestrator_react.llm import LLMClient, build_client
from orchestrator_react.react_loop import ReactResult, run_react_loop
from orchestrator_react.state import FULL_POOL, ReactState


#: Phase 1 / Phase 5 hooks. Overridable for testing; the defaults live in `phases`.
#: A `None` client inside the hook means "run the deterministic variant".
DiagnosisHook = Callable[..., Dict[str, Any]]
ReportHook = Callable[[ReactState, "SeriesOutcome", Optional[LLMClient]], str]


# ──────────────────────────────────────────────────────────────────────────────
# effective weights
# ──────────────────────────────────────────────────────────────────────────────


def effective_weights(state: ReactState, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Per-horizon weights actually implied by the winning strategy.

    Feeds `weights_by_horizon`. Every strategy is expressed as a weight vector per
    horizon so the CSV stays uniformly analysable:

        mean          uniform over the pool
        median        indicator of the element(s) the median selects at that horizon
        trimmed_mean  uniform over the models kept after trimming, per horizon
        weighted      the resolved weights (broadcast when they are not per-horizon)
        best_single   one-hot
        dba           uniform, and flagged nominal - a DTW barycentre is not a
                      weighted average, so these weights describe participation,
                      not arithmetic

    `nominal=True` marks the case where the weights are descriptive rather than the
    literal arithmetic, so nobody reads them as a reproducible recipe.
    """
    method = spec["combine"]
    horizon = state.horizon
    names = state.model_names

    def _empty() -> Dict[str, Dict[str, float]]:
        return {str(h): {n: 0.0 for n in names} for h in range(horizon)}

    if method == "best_single":
        chosen = str(spec["model"])
        w = _empty()
        for h in range(horizon):
            w[str(h)][chosen] = 1.0
        return {"weights": w, "nominal": False}

    idx = state.get_pool(spec["pool"])
    pool_names = [names[i] for i in idx]
    preds = state.test_preds[idx, :]
    w = _empty()

    if method == "mean":
        share = 1.0 / len(idx)
        for h in range(horizon):
            for n in pool_names:
                w[str(h)][n] = share
        return {"weights": w, "nominal": False}

    if method == "dba":
        share = 1.0 / len(idx)
        for h in range(horizon):
            for n in pool_names:
                w[str(h)][n] = share
        return {"weights": w, "nominal": True, "note": "DBA is not a weighted average"}

    if method == "median":
        for h in range(horizon):
            col = preds[:, h]
            finite = np.where(np.isfinite(col))[0]
            if finite.size == 0:
                continue
            order = finite[np.argsort(col[finite])]
            mid = order.size // 2
            picked = [order[mid]] if order.size % 2 else [order[mid - 1], order[mid]]
            for p in picked:
                w[str(h)][pool_names[int(p)]] = 1.0 / len(picked)
        return {"weights": w, "nominal": False}

    if method == "trimmed_mean":
        trim = float(spec.get("trim_pct", 0.2))
        m = len(idx)
        k = int(np.floor(m * trim))
        for h in range(horizon):
            col = preds[:, h]
            finite = np.where(np.isfinite(col))[0]
            if finite.size == 0:
                continue
            order = finite[np.argsort(col[finite])]
            kept = order if (k <= 0 or 2 * k >= order.size) else order[k : order.size - k]
            share = 1.0 / len(kept)
            for p in kept:
                w[str(h)][pool_names[int(p)]] = share
        return {"weights": w, "nominal": False}

    if method == "weighted":
        recipe = state.get_weights_recipe(spec["weights"])
        resolved = np.asarray(recipe.resolved, dtype=float)
        for h in range(horizon):
            col = resolved[:, h] if resolved.ndim == 2 else resolved
            for j, n in enumerate(pool_names):
                w[str(h)][n] = float(col[j])
        return {"weights": w, "nominal": False}

    return {"weights": w, "nominal": True, "note": f"unmapped method {method}"}


# ──────────────────────────────────────────────────────────────────────────────
# outcome
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SeriesOutcome:
    """Result of one series, with everything the CSV writer needs."""

    dataset: str
    dataset_index: int
    horizon: int
    success: bool = True
    error: str = ""

    forecast: List[float] = field(default_factory=list)
    test_values: List[float] = field(default_factory=list)
    start_test: Any = None
    final_test: Any = None

    state: Optional[ReactState] = None
    react: Optional[ReactResult] = None
    series_card: Dict[str, Any] = field(default_factory=dict)
    pool_card: Dict[str, Any] = field(default_factory=dict)
    phase2: Dict[str, Any] = field(default_factory=dict)
    predict_debug: Dict[str, Any] = field(default_factory=dict)
    sanity: Dict[str, Any] = field(default_factory=dict)
    external_baselines: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    report_text: str = ""
    config: Optional[ReactConfig] = None
    warnings: List[str] = field(default_factory=list)

    # ── the decision, structured (the `decision.json` of Section 3.1) ─────────

    def decision(self) -> Dict[str, Any]:
        if not self.react or not self.react.final_attempt:
            return {"error": self.error or "no decision"}
        attempt = self.react.final_attempt
        spec = attempt.spec
        state = self.state
        assert state is not None

        weights_block: Optional[Dict[str, Any]] = None
        if spec["combine"] == "weighted":
            recipe = state.get_weights_recipe(spec["weights"])
            weights_block = {
                "handle": spec["weights"],
                "computed_by": recipe.method,
                "effective_mode": recipe.meta.get("mode"),
                "fit_windows": recipe.meta.get("fit_windows"),
                "params": recipe.params,
            }

        return {
            "strategy": spec,
            "attempt_id": attempt.attempt_id,
            "origin": attempt.origin,
            "models": self.selected_models(),
            "effective_models": self.effective_models(),
            "reducibility": self.reducibility(),
            "weights": weights_block,
            "validation": {
                "score": _round(attempt.score),
                "aggregate": {k: _round(v) for k, v in attempt.aggregate.items()},
                "per_window_rmse": [_round(w["RMSE"]) for w in attempt.per_window],
                "backtest_mode": state.config.backtest_mode,
                "n_windows": state.n_windows,
            },
            "loop": self.react.summary(),
            "provenance": state.verify_provenance(),
            "selection_confidence": state.selection_confidence(),
            "diagnosis": self.diagnosis or None,
            "config": {
                "ablation": state.config.fingerprint(),
                "pool_mode": self.phase2.get("pool_composition_mode"),
                "score_preset": state.config.score_preset,
            },
        }

    def effective_models(self, threshold: float = 0.01) -> List[str]:
        """Models that actually carry weight in the winning strategy.

        `selected_models` reports the pool the strategy was built on, which is not
        the same thing: OLS on three windows routinely collapses onto one model, so
        a "weighted combination of 9 models" can be a single model wearing a
        weighted label. Reporting only the pool size would overstate how much
        combination the agent is really doing.
        """
        if self.state is None or not self.react or not self.react.final_attempt:
            return []
        weights = effective_weights(self.state, self.react.final_attempt.spec)["weights"]
        if not weights:
            return []
        first = weights[next(iter(weights))]
        return sorted(name for name, value in first.items() if abs(float(value)) > threshold)

    def reducibility(self, tolerance: float = 0.01) -> Dict[str, Any]:
        """Is the winning strategy just a simpler one wearing a fancier label?

        Fitted weights on three validation windows come out close to uniform: on
        real NN5 series, inverse-error weights over five models landed between
        0.195 and 0.208, within 4% of the 0.200 equal weight. A strategy like that
        is arithmetically the mean of its pool, and calling it a weighted
        combination overstates what the weighting contributed.

        This compares the winning forecast against the plain mean of the *same*
        pool and reports the largest relative difference. It isolates the question
        that matters for the claim: did the weighting do anything, or is the whole
        gain in the choice of subset?
        """
        blank = {
            "equivalent_to_pool_mean": None,
            "pool_mean_relative_diff": None,
            "weights_concentration": None,
        }
        if self.state is None or not self.react or not self.react.final_attempt:
            return blank
        spec = self.react.final_attempt.spec
        if spec["combine"] == "best_single":
            return {**blank, "equivalent_to_pool_mean": False}

        concentration = None
        if spec["combine"] == "weighted":
            concentration = self.state.weights_summary(spec["weights"]).get("concentration")

        try:
            # Compare what was actually reported, not what the winner alone would
            # have produced: under `final_strategy="ensemble"` those differ, and
            # measuring the winner would answer a question nobody asked.
            chosen = (
                np.asarray(self.forecast, dtype=float)
                if self.forecast
                else self.state.apply_to_test(spec)[0]
            )
            plain, _ = self.state.apply_to_test({"combine": "mean", "pool": spec["pool"]})
        except Exception:
            return {**blank, "weights_concentration": concentration}
        if chosen.shape != plain.shape:
            return {**blank, "weights_concentration": concentration}

        scale = float(np.nanmax(np.abs(plain))) or 1.0
        diff = float(np.nanmax(np.abs(chosen - plain))) / scale
        return {
            "equivalent_to_pool_mean": bool(diff <= tolerance),
            "pool_mean_relative_diff": round(diff, 6),
            "weights_concentration": concentration,
        }

    def selected_models(self) -> List[str]:
        if not self.react or not self.react.final_attempt or self.state is None:
            return []
        spec = self.react.final_attempt.spec
        if spec["combine"] == "best_single":
            return [str(spec["model"])]
        return self.state.pool_names(spec["pool"])

    def decision_report(self) -> str:
        """One-line human-readable summary, the successor of the old field."""
        if not self.success or not self.react or not self.react.final_attempt:
            return f"failed: {self.error}"
        r = self.react
        a = r.final_attempt
        return (
            f"strategy={a.spec['combine']} | pool={len(self.selected_models())} "
            f"effective={len(self.effective_models())} "
            f"| score={_round(a.score)} | origin={a.origin} "
            f"| iterations={r.iterations_used} | stop={r.stop_reason} "
            f"| llm={r.llm_model} | ablation={self.config.fingerprint() if self.config else ''}"
        )

    # ── CSV payload ──────────────────────────────────────────────────────────

    def csv_fields(self) -> Dict[str, Any]:
        """The non-metric CSV columns of Step 4.

        Deliberately excludes `mape/pocid/smape/rmse/msmape/mae`: those must keep
        being computed by `all_functions` in the writer, byte-for-byte as today
        (Section 4.1). This method supplies `predictions` and `test`; the writer
        turns them into metrics.

        Column names follow Section 4.4 of the specification verbatim, including
        the Portuguese ones (`justificativa_final`, `agent_model_diagnostico`,
        `agent_model_relato`) — they are the contract with the analysis code.

        `llm_artifacts_path` is filled by the writer, which is the only component
        that knows where the artifacts land.
        """
        cfg = self.config or ReactConfig()
        react = self.react
        attempt = react.final_attempt if react else None
        stability = self.pool_card.get("ranking_stability", {})
        corr = self.pool_card.get("error_correlation", {})
        tools = react.tools if react else {}

        weights_map: Dict[str, Any] = {}
        resolved: Dict[str, Any] = {}
        if attempt and self.state is not None:
            weights_map = effective_weights(self.state, attempt.spec)
            if attempt.spec["combine"] == "weighted":
                resolved = self.state.resolved_weights_map(attempt.spec["weights"])

        baselines = {
            "seeded": pool_mod.baseline_scores(self.state) if self.state else {},
            "external": self.external_baselines,
        }

        return {
            # -- kept from the old format, re-pointed at the new architecture ---
            "description": _dumps(self.decision()),
            "decision_report": self.decision_report(),
            "score_preset": cfg.score_preset,
            "tool_missing": bool(tools.get("tool_missing", False)),
            "tools_called": _dumps(tools.get("tools_called", [])),
            **_provenance_columns(self.state),
            # Re-pointed: used to be the post-debate candidate ranking, now the
            # ranked attempt history, which plays the same role.
            "final_candidate_names": _dumps(
                [
                    {"id": a.attempt_id, "strategy": _strategy_name(a), "origin": a.origin}
                    for a in (self.state.ranked_attempts() if self.state else [])
                ]
            ),
            "final_candidate_count": len(self.state.attempts) if self.state else 0,
            "best_strategy_name": _strategy_name(attempt),
            "best_strategy_method": attempt.spec["combine"] if attempt else "",
            "best_strategy_params": _dumps(attempt.spec if attempt else {}),
            "predict_debug": _dumps(self.predict_debug),
            "selected_base_models": _dumps(self.selected_models()),
            # The pool the strategy was built on vs. the models that actually carry
            # weight. They differ whenever a weighting scheme concentrates, which
            # OLS on three windows does almost every time.
            "n_pool_models": len(self.selected_models()),
            "effective_models": _dumps(self.effective_models()),
            "n_effective_models": len(self.effective_models()),
            **self.reducibility(),
            "weights_by_horizon": _dumps(weights_map.get("weights", {})),
            # -- new, Section 4.4 -----------------------------------------------
            "series_profile_json": _dumps(self.series_card),
            "ranking_stability_score": stability.get("mean_kendall_tau"),
            "error_correlation_groups": _dumps(corr.get("redundant_groups", [])),
            "pool_composition_mode": self.phase2.get("pool_composition_mode", ""),
            "react_iterations_used": react.iterations_used if react else 0,
            "react_early_stopped": bool(react.early_stopped) if react else False,
            "react_trajectory_json": _dumps(react.trajectory if react else []),
            "baseline_results_json": _dumps(baselines),
            "weights_handle_resolved": _dumps(resolved),
            "agent_model_combinator": cfg.combinator.label(),
            "agent_model_diagnostico": cfg.diagnostician.label(),
            "agent_model_relato": cfg.reporter.label(),
            "accept_confidence": react.accept_confidence if react else None,
            **_selection_columns(self.state),
            "calibration_gate_triggered": bool(
                self.phase2.get("calibration_gate", {}).get("triggered", False)
            ),
            "ablation_config": cfg.fingerprint(),
            "justificativa_final": self.report_text or (react.justification if react else ""),
        }


# ──────────────────────────────────────────────────────────────────────────────
# the run
# ──────────────────────────────────────────────────────────────────────────────


def run_series(
    models: Sequence[str],
    dataset: str,
    dataset_index: int,
    config: Optional[ReactConfig] = None,
    source: Optional[SeriesSource] = None,
    source_file: Optional[str] = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
    results_dir: str = ingest_mod.DEFAULT_RESULTS_DIR,
    frames: Optional[Dict[str, Any]] = None,
    client: Optional[LLMClient] = None,
    diagnosis_hook: DiagnosisHook = phases_mod.run_diagnosis,
    report_hook: ReportHook = phases_mod.run_report,
    read_external_baselines: bool = True,
    on_step: Optional[Callable[[Optional[int], Dict[str, Any]], None]] = None,
    pooled_meta_model: Optional[meta_model_mod.PooledMetaModel] = None,
) -> SeriesOutcome:
    """Runs Phases 0 to 5 for one series.

    Args:
        client: the combiner LLM. `None` builds one from `config.combinator`, and
            a disabled role means the deterministic arm (best seeded baseline).
        diagnosis_hook / report_hook: Phases 1 and 5, overridable for testing.
        pooled_meta_model: this series' leave-one-series-out model from
            `meta_model.build_pooled_meta_models`, or `None` when the run has too
            few series to pool, or xgboost is unavailable. Attached to `state`
            before Phase 3 so `weights_pooled_meta_model` can find it; `run_dataset`
            computes the whole dataset's models once and passes each series its own.

    Phase ordering note: the series profile is computed first, but its
    *interpretation* runs after Phase 2, because principle 4 puts both the series
    characteristics and the pool performance in the pre-loop summary — and the
    reading is much weaker without knowing how the models actually behaved.
    """
    config = config or ReactConfig()

    # ── Phase 0 — ingestion ──────────────────────────────────────────────────
    ingested = ingest_mod.load_series(
        models=models,
        dataset=dataset,
        dataset_index=dataset_index,
        source=source,
        config=config,
        results_dir=results_dir,
        source_file=source_file,
        source_dir=source_dir,
        frames=frames,
    )
    state = ingested.state
    state.pooled_meta_model = pooled_meta_model
    outcome = SeriesOutcome(
        dataset=dataset,
        dataset_index=int(dataset_index),
        horizon=ingested.horizon,
        state=state,
        config=config,
        start_test=ingested.start_test,
        final_test=ingested.final_test,
        warnings=list(ingested.warnings),
        test_values=list(ingested.test_values),
    )

    # ── Phase 1a — series profile (deterministic, always) ───────────────────
    outcome.series_card = T.series_profile(state)

    # ── Phase 2 — pool evaluation and baseline seeding ──────────────────────
    outcome.phase2 = pool_mod.run_phase2(state, config)
    outcome.pool_card = outcome.phase2["report"]

    # ── Phase 1b — interpretation of both cards (LLM only under the ablation) ─
    diag_client = build_client(config.diagnostician)
    try:
        outcome.diagnosis = diagnosis_hook(
            state, outcome.series_card, diag_client, outcome.pool_card
        )
    except Exception as exc:  # non-fatal: the cards alone are enough
        outcome.warnings.append(f"diagnosis failed: {type(exc).__name__}: {exc}")
        outcome.diagnosis = {}

    # ── Phase 3 — the ReAct loop ────────────────────────────────────────────
    if client is None:
        client = build_client(config.combinator)
    gate = outcome.phase2.get("calibration_gate", {})
    skip = "calibration_gate" if gate.get("triggered") else ""
    outcome.react = run_react_loop(
        state=state,
        client=client,
        series_card=outcome.series_card,
        pool_card=outcome.pool_card,
        config=config,
        skip_reason=skip,
        diagnosis=outcome.diagnosis,
        on_step=on_step,
    )

    # ── Phase 4 — deterministic application to the test forecasts ───────────
    attempt = outcome.react.final_attempt
    if attempt is None:
        outcome.success = False
        outcome.error = "no strategy was selected"
        return outcome

    if config.final_strategy == "ensemble":
        forecast, debug = state.apply_ensemble(
            top_m=config.final_top_m, eta=config.final_eta
        )
    else:
        forecast, debug = state.apply_to_test(attempt.spec)
    if forecast.size != state.horizon or not np.all(np.isfinite(forecast)):
        outcome.warnings.append(
            f"final forecast has {int(np.sum(~np.isfinite(forecast)))} non-finite points"
        )
    outcome.forecast = [float(v) for v in forecast]
    outcome.predict_debug = _slim_debug(debug)
    outcome.sanity = T.sanity_check(state, attempt.spec)
    if not outcome.sanity.get("ok"):
        outcome.warnings.extend(outcome.sanity.get("warnings", []))

    if read_external_baselines:
        outcome.external_baselines = ingest_mod.read_external_baselines(
            dataset, dataset_index, results_dir=results_dir
        )

    # ── Phase 5 — natural-language report (optional) ────────────────────────
    report_client = build_client(config.reporter)
    if report_client is not None:
        try:
            outcome.report_text = report_hook(state, outcome, report_client)
        except Exception as exc:  # non-fatal: the causal justification stands
            outcome.warnings.append(f"report failed: {type(exc).__name__}: {exc}")

    return outcome


def run_dataset(
    models: Sequence[str],
    dataset: str,
    source_file: Optional[str] = None,
    config: Optional[ReactConfig] = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
    results_dir: str = ingest_mod.DEFAULT_RESULTS_DIR,
    indices: Optional[Sequence[int]] = None,
    client: Optional[LLMClient] = None,
    **kwargs: Any,
) -> Iterator[SeriesOutcome]:
    """Runs every series of a dataset, yielding one outcome at a time.

    The result CSVs and the `.tsf` are read once and reused across series.

    A `SeriesAlignmentError` is re-raised immediately: it means the `.tsf` on disk
    is not the one the forecasts came from, so every remaining series would fail
    the same way and the run should stop rather than emit 182 identical failures.
    Any other per-series error is captured into a failed outcome and the loop
    continues.
    """
    config = config or ReactConfig()
    frames = ingest_mod.load_dataset_frames(models, dataset, results_dir)
    n_series = ingest_mod.count_series(dataset, models[0], results_dir)

    source: Optional[SeriesSource] = None
    if source_file:
        source = load_series_source(source_file, n_expected_series=n_series, source_dir=source_dir)

    todo = list(indices) if indices is not None else list(range(n_series))

    meta_models: Dict[int, meta_model_mod.PooledMetaModel] = {}
    if config.pooled_meta_model:
        meta_models = _build_pooled_meta_models(
            models, dataset, todo, config=config, source=source,
            results_dir=results_dir, frames=frames,
        )

    for idx in todo:
        try:
            yield run_series(
                models=models,
                dataset=dataset,
                dataset_index=idx,
                config=config,
                source=source,
                results_dir=results_dir,
                frames=frames,
                client=client,
                pooled_meta_model=meta_models.get(idx),
                **kwargs,
            )
        except SeriesAlignmentError:
            raise
        except Exception as exc:
            yield SeriesOutcome(
                dataset=dataset,
                dataset_index=int(idx),
                horizon=0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                config=config,
                warnings=[traceback.format_exc(limit=3)],
            )


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_pooled_meta_models(
    models: Sequence[str],
    dataset: str,
    todo: Sequence[int],
    config: ReactConfig,
    source: Optional[SeriesSource],
    results_dir: str,
    frames: Dict[str, Any],
) -> Dict[int, meta_model_mod.PooledMetaModel]:
    """One pre-pass over the dataset, before any series' Phase 3 opens.

    Deterministic, no LLM: for each series this only runs Phase 0 (ingestion) plus
    `series_profile`, to get the same features and validation errors Phase 2 would
    have exposed anyway, then discards the state — keeping ~200 small feature
    vectors in memory costs nothing; keeping ~200 series' full forecast tensors
    would. Returns `{}` (every series gets `pooled_meta_model=None`, and the tool
    is withheld) if there are too few series or xgboost is unavailable, decided
    once by `meta_model.build_pooled_meta_models` rather than duplicated here.

    A `SeriesAlignmentError` here means the same thing it means in the main loop —
    the `.tsf` on disk is not the one the forecasts came from — and is re-raised
    for the same reason: every remaining series would fail identically. Any other
    per-series ingestion failure just drops that series from the training rows;
    the main loop below will independently hit and report the same failure again
    when it reaches that series for real.
    """
    rows: List[meta_model_mod.MetaRow] = []
    for idx in todo:
        try:
            ingested = ingest_mod.load_series(
                models=models, dataset=dataset, dataset_index=idx,
                source=source, config=config, results_dir=results_dir, frames=frames,
            )
        except SeriesAlignmentError:
            raise
        except Exception:
            continue
        profile = T.series_profile(ingested.state)
        rows.append(
            meta_model_mod.build_meta_row(
                idx, profile, ingested.state.y_true, ingested.state.y_preds,
                ingested.state.model_names,
            )
        )
    return meta_model_mod.build_pooled_meta_models(
        rows, models, min_series=config.pooled_meta_model_min_series
    )


def _provenance_columns(state: Optional[ReactState]) -> Dict[str, Any]:
    """Audit trail proving the numbers came from executed tools, not from text."""
    if state is None:
        return {"n_tool_calls": 0, "n_evaluate_calls": 0, "provenance_ok": False}
    checks = state.verify_provenance()
    return {
        "n_tool_calls": checks["n_tool_calls"],
        "n_evaluate_calls": checks["n_evaluate_calls"],
        "provenance_ok": checks["provenance_ok"],
    }


def _selection_columns(state: Optional[ReactState]) -> Dict[str, Any]:
    """Deterministic answer to "is this choice defensible?", for the CSV."""
    if state is None:
        return {
            "selection_margin": None,
            "selection_bootstrap_pvalue": None,
            "selection_dm_pvalue": None,
            "selection_verdict": "no_comparison",
        }
    conf = state.selection_confidence()
    return {
        "selection_margin": conf.get("margin"),
        "selection_bootstrap_pvalue": conf.get("bootstrap_pvalue"),
        "selection_dm_pvalue": conf.get("dm_pvalue"),
        "selection_verdict": conf.get("verdict"),
    }


def _round(x: Any, nd: int = 6) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _dumps(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"error": "not serialisable"})


def _strategy_name(attempt: Any) -> str:
    if attempt is None:
        return ""
    spec = attempt.spec
    parts = [str(spec.get("combine"))]
    if spec.get("pool") and spec["pool"] != FULL_POOL:
        parts.append(spec["pool"])
    if spec.get("weights"):
        parts.append(spec["weights"])
    if spec.get("model"):
        parts.append(str(spec["model"]))
    return "_".join(parts)


def _slim_debug(debug: Dict[str, Any]) -> Dict[str, Any]:
    """Drops the bulky resolved-weights block; it has its own CSV column.

    For an ensemble the per-member debug blocks are dropped too: the members and
    their shares are kept, which is what makes the row auditable, but their
    individual weight vectors would multiply the column size by `final_top_m`.
    """
    slim = {k: v for k, v in debug.items() if k not in ("weights_resolved", "member_debug")}
    return slim
