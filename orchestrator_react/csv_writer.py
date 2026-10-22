"""Result CSV writer — the Step 4 output contract.

Column policy, straight from the specification:

* **Section 4.1, untouchable.** The 13 evaluation columns keep their names, their
  order and their exact values. The six metrics are computed here with
  `all_functions`, using the same reshape convention the project has always used,
  so a row written by this architecture is directly comparable with every row
  written by the old one.
* **Section 4.2, removed.** The 23 debate columns are gone.
* **Section 4.3, kept.** Re-pointed at the new architecture.
* **Section 4.4, added.** 16 traceability columns.

`final_candidate_names` / `final_candidate_count` are the two columns the
specification never classified. They are kept and re-pointed: they used to hold the
post-debate candidate ranking, and now hold the ranked attempt history, which plays
the same role.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from orchestrator_react.config import FINAL_SCORE_PRESETS, SEED_TEST_PRESETS


#: Section 4.1 — never rename, never reorder, never change how these are computed.
CORE_COLUMNS: List[str] = [
    "dataset_index",
    "horizon",
    "regressor",
    "mape",
    "pocid",
    "smape",
    "rmse",
    "msmape",
    "mae",
    "test",
    "predictions",
    "start_test",
    "final_test",
]

#: Section 4.3 — kept from the old format, re-pointed at the new architecture.
KEPT_COLUMNS: List[str] = [
    "description",
    "decision_report",
    "llm_artifacts_path",
    "score_preset",
    "tool_missing",
    "tools_called",
    "n_tool_calls",
    "n_evaluate_calls",
    "provenance_ok",
    "final_candidate_names",
    "final_candidate_count",
    "best_strategy_name",
    "best_strategy_method",
    "best_strategy_params",
    "predict_debug",
    "selected_base_models",
    "n_pool_models",
    # Did the weighting actually do anything? With three validation windows fitted
    # weights land within a few percent of uniform, which makes a "weighted
    # combination" arithmetically the mean of its own pool.
    "weights_concentration",
    "equivalent_to_pool_mean",
    "pool_mean_relative_diff",
    "weights_by_horizon",
]

#: Section 4.4 — new traceability columns.
NEW_COLUMNS: List[str] = [
    "series_profile_json",
    "ranking_stability_score",
    "error_correlation_groups",
    "pool_composition_mode",
    "react_iterations_used",
    "react_early_stopped",
    "react_trajectory_json",
    "baseline_results_json",
    "weights_handle_resolved",
    "agent_model_combinator",
    "agent_model_diagnostico",
    "agent_model_relato",
    "accept_confidence",
    # Beyond Section 4.4: the agent's self-reported confidence turned out to be a
    # constant (0.9 on every accept of a 19-series gpt-oss:20b run), so it cannot
    # support any claim. These four are the deterministic replacement — whether the
    # selected strategy is statistically distinguishable from the runner-up.
    "selection_margin",
    "selection_bootstrap_pvalue",
    "selection_dm_pvalue",
    "selection_verdict",
    "calibration_gate_triggered",
    "ablation_config",
    "justificativa_final",
    # Beyond Section 4.4: sMAPE is unstable when an actual test value is at or
    # near zero — a single such point can push a whole series' sMAPE past 100%
    # for every method alike (ANP_MONTHLY series 18 and 85: agent and all five
    # external baselines all score 1.34-1.44 sMAPE, because the test window
    # contains literal zeros, not because any method forecast badly). Nothing
    # here changes the metric — smape/rmse/... stay byte-identical via
    # `all_functions`, as Section 4.1 requires — this only makes the artifact
    # visible instead of silently distorting an aggregate mean.
    "test_has_zero_actual",
    "test_min_abs_actual",
]

#: Compact per-series telemetry of the decision, one value per column. The full
#: per-turn trace (observations, leader before/after, timings, attempt registry)
#: lives in the artifacts under `agent_trace`; these columns exist so the CSV
#: alone can answer "how long did the agent take, how did it split its time, and
#: how often did the leader change" without opening every JSON.
AGENT_COLUMNS: List[str] = [
    "agent_started_at",
    "agent_finished_at",
    "agent_wall_clock_s",
    "agent_llm_s",
    "agent_tool_s",
    "agent_llm_time_share",
    "agent_turns",
    "agent_scoring_calls",
    "agent_non_scoring_calls",
    "agent_best_changes",
    "agent_time_to_first_best_s",
    "agent_time_after_last_improvement_s",
    "agent_score_weights_name",
]
NEW_COLUMNS = NEW_COLUMNS + AGENT_COLUMNS

COLS_SERIE: List[str] = CORE_COLUMNS + KEPT_COLUMNS + NEW_COLUMNS

#: Section 4.2 — must never reappear.
REMOVED_COLUMNS: List[str] = [
    "debate_ran", "debate_trigger", "approach_pre_debate", "approach_post_debate",
    "debate_explanation", "proposer_selected_names", "proposer_params_overrides",
    "proposer_force_debate", "proposer_debate_margin", "skeptic_remove_names",
    "skeptic_add_names", "skeptic_params_overrides", "statistician_remove_names",
    "statistician_add_names", "statistician_params_overrides", "proposer_think",
    "skeptic_think", "statistician_think", "pattern_analyst_think",
    "pattern_analyst_trend_champion", "pattern_analyst_seas_champion",
    "pattern_analyst_method_hint", "pattern_analyst_narrative",
    # collapsed into `justificativa_final`
    "selection_explanation", "when_good",
]


# ──────────────────────────────────────────────────────────────────────────────
# metrics — Section 4.1, computed exactly as the project always has
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(forecast: Sequence[float], actual: Sequence[float]) -> Dict[str, float]:
    """The six evaluation metrics, byte-for-byte compatible with the old writer.

    Uses `all_functions` with the `(1, -1)` reshape and `sklearn`'s MAPE, exactly as
    `run_tsf_orchestrator.py` did. Vectors are truncated to their common length
    first, which is also what the old code did.
    """
    from sklearn.metrics import mean_absolute_percentage_error as sk_mape

    from all_functions import (
        calculate_mae,
        calculate_msmape,
        calculate_rmse,
        calculate_smape,
        pocid,
    )

    nan = {k: float("nan") for k in ("mape", "pocid", "smape", "rmse", "msmape", "mae")}
    preds = np.asarray(forecast, dtype=float) if len(forecast) else np.array([])
    test = np.asarray(actual, dtype=float) if len(actual) else np.array([])
    n = int(min(preds.size, test.size))
    if n == 0:
        return nan

    preds_cut, test_cut = preds[:n], test[:n]
    p2, t2 = preds_cut.reshape(1, -1), test_cut.reshape(1, -1)
    return {
        "mape": float(sk_mape(test_cut, preds_cut)),
        "pocid": float(pocid(test_cut, preds_cut)),
        "smape": float(np.asarray(calculate_smape(p2, t2)).ravel()[0]),
        "rmse": float(np.asarray(calculate_rmse(p2, t2)).ravel()[0]),
        "msmape": float(np.asarray(calculate_msmape(p2, t2)).ravel()[0]),
        "mae": float(np.asarray(calculate_mae(p2, t2)).ravel()[0]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# row assembly
# ──────────────────────────────────────────────────────────────────────────────


def zero_actual_diagnostics(actual: Sequence[float]) -> Dict[str, Any]:
    """Flags a test window where sMAPE cannot be trusted at face value.

    sMAPE's denominator is `(|forecast| + |actual|) / 2`. When `actual` is at or
    near zero, that denominator collapses regardless of how close the forecast
    is in absolute terms, and the point's contribution saturates toward 200%.
    This is a property of the metric, not of any forecaster — on ANP_MONTHLY
    series 85 (test window `[20, 45, 15, 20, 5, 30, 0, 0, 10, 0, 15, 5]`) every
    method scored 1.34-1.44 sMAPE alike.

    Reads only `test_values`, after Phase 4 — the same data the metrics
    themselves are computed from, at the same point in the pipeline. It cannot
    leak anything the metrics do not already use.
    """
    arr = np.asarray(actual, dtype=float) if len(actual) else np.array([])
    if arr.size == 0:
        return {"test_has_zero_actual": None, "test_min_abs_actual": None}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"test_has_zero_actual": None, "test_min_abs_actual": None}
    min_abs = float(np.min(np.abs(finite)))
    return {
        "test_has_zero_actual": bool(min_abs < 1e-9),
        "test_min_abs_actual": round(min_abs, 6),
    }


def _finite(value: Any) -> Optional[float]:
    """Float for the artifacts, `None` when not finite (JSON has no NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _agent_columns(outcome: Any) -> Dict[str, Any]:
    """The compact telemetry columns, filled from the ReactResult and its derived block."""
    empty = {name: None for name in AGENT_COLUMNS}
    react = getattr(outcome, "react", None)
    if react is None:
        return empty
    derived = getattr(react, "derived", None) or {}
    return {
        "agent_started_at": getattr(react, "started_at", "") or None,
        "agent_finished_at": getattr(react, "finished_at", "") or None,
        "agent_wall_clock_s": derived.get("wall_clock_s"),
        "agent_llm_s": derived.get("llm_s"),
        "agent_tool_s": derived.get("tool_s"),
        "agent_llm_time_share": derived.get("llm_time_share"),
        "agent_turns": derived.get("turns"),
        "agent_scoring_calls": derived.get("scoring_calls"),
        "agent_non_scoring_calls": derived.get("non_scoring_calls"),
        "agent_best_changes": derived.get("best_changes"),
        "agent_time_to_first_best_s": derived.get("time_to_first_best_s"),
        "agent_time_after_last_improvement_s": derived.get("time_after_last_improvement_s"),
        "agent_score_weights_name": getattr(react, "score_weights_name", "") or None,
    }


def build_row(
    outcome: Any,
    regressor: str,
    horizon: Optional[int] = None,
    final_test: Any = None,
    start_test: str = "INICIO",
    llm_artifacts_path: str = "",
) -> Dict[str, Any]:
    """One CSV row from one `SeriesOutcome`.

    A failed series still produces a row: NaN metrics, empty forecast, and the
    error in `description`/`decision_report`. Losing the row entirely would hide
    the failure from the analysis.
    """
    forecast = list(outcome.forecast or [])
    actual = list(outcome.test_values or [])
    metrics = compute_metrics(forecast, actual) if (outcome.success and forecast) else compute_metrics([], [])
    zero_diag = zero_actual_diagnostics(actual)

    row: Dict[str, Any] = {
        "dataset_index": f"{outcome.dataset_index}",
        "horizon": horizon if horizon is not None else outcome.horizon,
        "regressor": regressor,
        **metrics,
        # `test`/`predictions` are wrapped in a list so `pd.DataFrame` treats each
        # as a single cell rather than as a column of rows — the old convention.
        "test": [list(actual)],
        "predictions": [list(forecast)],
        "start_test": start_test,
        "final_test": final_test if final_test is not None else outcome.final_test,
        **zero_diag,
    }

    fields = outcome.csv_fields()
    fields["llm_artifacts_path"] = llm_artifacts_path
    row.update(fields)
    row.update(_agent_columns(outcome))
    return row


def _cross_series_block(outcome: Any) -> Dict[str, Any]:
    """The dataset-level context this series was given, for post-run analysis.

    All of it is validation-only and leave-one-series-out by construction (see
    `pipeline._build_strategy_priors` and `meta_model.build_pooled_meta_models`);
    recording it here does not put anything in the artifact that was not already
    available to the run.
    """
    state = outcome.state
    if state is None:
        return {}

    out: Dict[str, Any] = {}
    prior = getattr(state, "strategy_prior", None)
    if prior:
        ranked = sorted(prior.items(), key=lambda kv: kv[1])
        out["strategy_prior"] = {k: round(float(v), 5) for k, v in ranked}
        out["prior_best"] = ranked[0][0]
        out["prior_worst"] = ranked[-1][0]

    try:
        from orchestrator_react.prompts import build_dataset_card

        card = build_dataset_card(state)
        if card:
            out["dataset_card_shown"] = card
    except Exception:  # never let bookkeeping break a written artifact
        pass

    meta = getattr(state, "pooled_meta_model", None)
    if meta is not None:
        out["pooled_meta_model"] = {
            "objective": getattr(meta, "objective", None),
            "n_train_series": getattr(meta, "n_train_series", None),
            "n_features": len(getattr(meta, "feature_names", ()) or ()),
            # True means the fit produced identical margins for every model, i.e.
            # its weights are uniform and it is indistinguishable from the mean.
            "degenerate": getattr(meta, "degenerate", None),
        }
    return out


def _agent_trace(outcome: Any) -> Dict[str, Any]:
    """Full per-series record of what the agent did, for the artifacts.

    Validation only. Each turn carries the full observation of the tool call, the
    leader before and after it (id, score, origin) and the timing; the attempt
    registry carries every strategy, its rationale, its validation aggregate and
    its per-window metrics; the decision and stop blocks close the record. The
    test actuals never appear here: they live in the pipeline outcome and are only
    turned into metrics by the writer, after the decision is over.

    This block is what makes the behaviour section reproducible end to end: any
    number quoted about "why the agent chose X" can be traced back to a turn and
    an attempt id.
    """
    react = getattr(outcome, "react", None)
    state = getattr(outcome, "state", None)
    if react is None or state is None:
        return {}

    ranking = {a.attempt_id: i + 1 for i, a in enumerate(state.ranked_attempts())}
    attempts: List[Dict[str, Any]] = []
    for a in state.attempts:
        attempts.append({
            "id": a.attempt_id,
            "rank": ranking.get(a.attempt_id),
            "origin": a.origin,
            "spec": a.spec,
            "rationale": a.rationale,
            "iteration": a.iteration,
            "n_models": a.n_models,
            "score": _finite(a.score),
            "aggregate": {k: _finite(v) for k, v in a.aggregate.items()},
            "per_window": a.per_window,
            "per_window_scores": [_finite(v) for v in a.per_window_scores],
            "preset_scores": (
                {k: _finite(v) for k, v in state.preset_scores(a, FINAL_SCORE_PRESETS).items()}
                if getattr(state.config, "multi_preset_final", False) else {}
            ),
            "agent_converged": bool(a.agent_converged),
            "agent_rationale": a.agent_rationale,
        })

    final = react.final_attempt
    config = getattr(outcome, "config", None)
    return {
        "meta": {
            "dataset": outcome.dataset,
            "dataset_index": outcome.dataset_index,
            "horizon": outcome.horizon,
            "n_windows": state.n_windows,
            "n_models": state.n_models,
            "score_preset": getattr(config, "score_preset", None),
            "ablation_config": config.fingerprint() if config is not None else None,
        },
        "turns": list(react.step_details),
        "attempts": attempts,
        "decision": {
            "final_attempt_id": final.attempt_id if final else None,
            "final_spec": final.spec if final else None,
            "agent_accepted_id": react.agent_accepted_id,
            "accept_confidence": react.accept_confidence,
            "overridden": react.overridden,
            "justification": react.justification,
            "score_weights_name": react.score_weights_name,
            "score_weights": dict(react.score_weights),
            "score_weights_held_out": react.score_weights_held_out,
            "score_weights_skipped": react.score_weights_skipped,
            "final_score_preset": react.final_score_preset,
            "preset_board": dict(react.preset_board),
            "selection_confidence": state.selection_confidence(),
        },
        "stop": {
            "stop_reason": react.stop_reason,
            "iterations_used": react.iterations_used,
            "early_stopped": react.early_stopped,
            "elapsed_s": react.elapsed_s,
            "empty_responses": react.empty_responses,
            "llm_error_retries": react.llm_error_retries,
            "withheld_tools": dict(react.withheld_tools),
            "llm_model": react.llm_model,
        },
        "derived": dict(react.derived),
    }


def _avg_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and np.isclose(arr[order[j]], arr[order[i]], equal_nan=True):
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _ranking_trace(outcome: Any) -> Dict[str, Any]:
    state = getattr(outcome, "state", None)
    react = getattr(outcome, "react", None)
    if state is None or react is None:
        return {}
    actual = list(outcome.test_values or [])
    if not actual:
        return {}
    rows: List[Dict[str, Any]] = []
    for attempt in state.attempts:
        try:
            forecast, _ = state.apply_to_test(attempt.spec)
            metrics = compute_metrics(forecast, actual)
        except Exception as exc:
            rows.append({
                "id": attempt.attempt_id,
                "origin": attempt.origin,
                "spec": attempt.spec,
                "validation_score": _finite(attempt.score),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        rows.append({
            "id": attempt.attempt_id,
            "origin": attempt.origin,
            "spec": attempt.spec,
            "validation_score": _finite(attempt.score),
            "validation_aggregate": {k: _finite(v) for k, v in attempt.aggregate.items()},
            "per_window": attempt.per_window,
            "test_metrics": metrics,
            "test_smape": metrics.get("smape"),
            "test_pocid": metrics.get("pocid"),
            "test_forecast": [float(v) for v in np.asarray(forecast, dtype=float).ravel()],
        })
    valid = [r for r in rows if r.get("validation_score") is not None]
    valid.sort(key=lambda r: (float(r["validation_score"]), str(r["id"])))
    for rank, row in enumerate(valid, start=1):
        row["validation_rank"] = rank
    testable = [r for r in rows if r.get("test_smape") is not None and np.isfinite(r["test_smape"])]
    testable.sort(key=lambda r: (float(r["test_smape"]), -float(r.get("test_pocid") or 0.0), str(r["id"])))
    for rank, row in enumerate(testable, start=1):
        row["test_rank"] = rank
    final = react.final_attempt
    final_row = next((r for r in rows if final and r.get("id") == final.attempt_id), None)
    seed_row = next((r for r in valid if r.get("origin") == "baseline"), None)
    top3_val = [r.get("id") for r in valid[:3]]
    top3_test = [r.get("id") for r in testable[:3]]
    val_rank = {r.get("id"): r.get("validation_rank") for r in valid}
    test_rank = {r.get("id"): r.get("test_rank") for r in testable}
    common = [r for r in rows if r.get("id") in val_rank and r.get("id") in test_rank]
    spearman = None
    if len(common) >= 2:
        rv = _avg_ranks([val_rank[r["id"]] for r in common])
        rt = _avg_ranks([test_rank[r["id"]] for r in common])
        if np.std(rv) > 0 and np.std(rt) > 0:
            spearman = float(np.corrcoef(rv, rt)[0, 1])
    return {
        "criterion": str(getattr(state.config, "score_preset", "")),
        "window_mode": str(getattr(state.config, "score_window_mode", "")),
        "n_attempts": len(rows),
        "top3_validation": top3_val,
        "top3_test": top3_test,
        "top1_same": bool(top3_val[:1] and top3_test[:1] and top3_val[0] == top3_test[0]),
        "top2_same": bool(len(top3_val) > 1 and len(top3_test) > 1 and top3_val[1] == top3_test[1]),
        "top3_same": bool(len(top3_val) > 2 and len(top3_test) > 2 and top3_val[2] == top3_test[2]),
        "validation_test_spearman": spearman,
        "final": (
            {
                "id": final_row.get("id") if final_row else None,
                "validation_rank": final_row.get("validation_rank") if final_row else None,
                "test_rank": final_row.get("test_rank") if final_row else None,
                "test_smape": final_row.get("test_smape") if final_row else None,
                "test_pocid": final_row.get("test_pocid") if final_row else None,
                "origin": final_row.get("origin") if final_row else None,
            }
        ),
        "seed": (
            {
                "id": seed_row.get("id") if seed_row else None,
                "validation_rank": seed_row.get("validation_rank") if seed_row else None,
                "test_rank": seed_row.get("test_rank") if seed_row else None,
                "test_smape": seed_row.get("test_smape") if seed_row else None,
                "test_pocid": seed_row.get("test_pocid") if seed_row else None,
            }
        ),
        "oracle": (
            {
                "id": testable[0].get("id") if testable else None,
                "test_smape": testable[0].get("test_smape") if testable else None,
                "test_pocid": testable[0].get("test_pocid") if testable else None,
                "origin": testable[0].get("origin") if testable else None,
            }
        ),
        "attempts": rows,
    }


def _seed_preset_tests(outcome: Any) -> Dict[str, Any]:
    state = getattr(outcome, "state", None)
    react = getattr(outcome, "react", None)
    if state is None or react is None:
        return {}
    actual = list(outcome.test_values or [])
    if not actual:
        return {}
    seeds = [a for a in state.attempts if a.origin == "baseline"]
    out: Dict[str, Any] = {"presets": {}, "agent": {}}
    for preset in SEED_TEST_PRESETS:
        seed = state.best_attempt_for_preset(preset, attempts=seeds)
        if seed is None:
            out["presets"][preset] = None
            continue
        try:
            forecast, _ = state.apply_to_test(seed.spec)
            metrics = compute_metrics(forecast, actual)
        except Exception as exc:
            out["presets"][preset] = {
                "attempt_id": seed.attempt_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        out["presets"][preset] = {
            "attempt_id": seed.attempt_id,
            "spec": seed.spec,
            "validation_score": _finite(state.preset_score(seed, preset)),
            "test_metrics": metrics,
            "forecast": [float(v) for v in np.asarray(forecast, dtype=float).ravel()],
        }
    final = react.final_attempt
    if final is not None:
        out["agent"] = {
            "attempt_id": final.attempt_id,
            "final_score_preset": getattr(react, "final_score_preset", "") or None,
            "spec": final.spec,
            "test_metrics": compute_metrics(outcome.forecast, actual),
            "forecast": [float(v) for v in np.asarray(outcome.forecast, dtype=float).ravel()],
        }
    return out


def artifacts_payload(outcome: Any) -> Dict[str, Any]:
    """Full per-series audit trail, written next to the CSV.

    The CSV keeps the compact trajectory; this keeps everything, including the raw
    cards and the warnings, for the cases where a row looks odd and needs digging.
    """
    payload = {
        "dataset": outcome.dataset,
        "dataset_index": outcome.dataset_index,
        "success": outcome.success,
        "error": outcome.error,
        "config": outcome.config.to_dict() if outcome.config else None,
        "decision": outcome.decision(),
        "series_card": outcome.series_card,
        "pool_card": outcome.pool_card,
        "diagnosis": outcome.diagnosis,
        # Everything the cross-series machinery decided for this series. Without it
        # a v5 run cannot be analysed the way v3/v4 were: which strategies the
        # dataset prior favoured, what the agent was shown in the DATASET CARD, and
        # whether the pooled meta-model actually learned anything for this series.
        "cross_series": _cross_series_block(outcome),
        "phase2": {
            "pool": outcome.phase2.get("pool"),
            "baselines": outcome.phase2.get("baselines"),
            "calibration_gate": outcome.phase2.get("calibration_gate"),
        },
        "react": {
            "trajectory": outcome.react.trajectory if outcome.react else [],
            "summary": outcome.react.summary() if outcome.react else {},
            "errors": outcome.react.errors if outcome.react else [],
            "parse_failures": outcome.react.parse_failures if outcome.react else [],
            "step_details": outcome.react.step_details if outcome.react else [],
            # `tools` is `registry.tools_called_summary(state)`: the same
            # `tool_missing`/`tools_called` the CSV row reports, but with the
            # `kind`/`detail` of each failed call attached. Before this, finding
            # out WHY a series had `tool_missing=True` meant re-parsing the
            # `tools_called` CSV column by hand — this is that lookup, saved once
            # instead of redone per series.
            "tools": outcome.react.tools if outcome.react else {},
        },
        # The complete decision record of the series, turn by turn. Everything in
        # it is validation-only; see `_agent_trace`.
        "agent_trace": _agent_trace(outcome),
        "seed_preset_tests": _seed_preset_tests(outcome),
        "ranking_trace": _ranking_trace(outcome),
        "predict_debug": outcome.predict_debug,
        "sanity": outcome.sanity,
        "report_text": outcome.report_text,
        "warnings": outcome.warnings,
    }
    # v11 only: the blind final-selection audit. Kept out of the dict entirely
    # for the other strategies, so a v10/`argmin` artifact retains its shape.
    final_choice = getattr(outcome, "final_choice", None)
    if final_choice:
        payload["final_choice"] = final_choice
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# writer
# ──────────────────────────────────────────────────────────────────────────────


class ResultWriter:
    """Appends rows to `<results_dir>/<experiment>/<DATASET>.csv`.

    Mirrors the old writer's behaviour: creates the file with the header when it is
    missing, and reindexes an existing file onto the current schema. The reindex
    now also **drops** the Section 4.2 columns, so re-running over an old file does
    not resurrect the debate schema.
    """

    def __init__(
        self,
        dataset: str,
        experiment: str,
        results_dir: str = "./timeseries/mestrado/resultados",
        save_artifacts: bool = True,
    ) -> None:
        self.dataset = dataset
        self.experiment = experiment
        self.base_dir = os.path.join(results_dir, experiment)
        self.csv_path = os.path.join(self.base_dir, f"{dataset}.csv")
        self.artifacts_dir = os.path.join(self.base_dir, "llm_artifacts", dataset)
        self.save_artifacts = save_artifacts
        self.rows_written = 0

        os.makedirs(self.base_dir, exist_ok=True)
        if self.save_artifacts:
            os.makedirs(self.artifacts_dir, exist_ok=True)
        self._prepare_file()

    def _prepare_file(self) -> None:
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            pd.DataFrame(columns=COLS_SERIE).to_csv(self.csv_path, sep=";", index=False)
            return
        try:
            existing = pd.read_csv(self.csv_path, sep=";")
        except Exception:
            # A malformed file is left alone; new rows still append.
            return
        stale = [c for c in existing.columns if c in REMOVED_COLUMNS]
        missing = [c for c in COLS_SERIE if c not in existing.columns]
        if stale or missing:
            for col in missing:
                existing[col] = np.nan
            existing.reindex(columns=COLS_SERIE).to_csv(self.csv_path, sep=";", index=False)

    def write(self, outcome: Any, regressor: Optional[str] = None, **kwargs: Any) -> str:
        """Writes one outcome. Returns the artifacts path (empty when disabled)."""
        artifacts_path = ""
        if self.save_artifacts:
            artifacts_path = os.path.abspath(
                os.path.join(self.artifacts_dir, f"dataset_{outcome.dataset_index}.json")
            )
            try:
                with open(artifacts_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        artifacts_payload(outcome), fh, ensure_ascii=False, indent=2, default=str
                    )
            except Exception:
                artifacts_path = ""

        row = build_row(
            outcome,
            regressor=regressor or self.experiment,
            llm_artifacts_path=artifacts_path,
            **kwargs,
        )
        frame = pd.DataFrame(row).reindex(columns=COLS_SERIE)
        frame.to_csv(self.csv_path, sep=";", mode="a", header=False, index=False)
        self.rows_written += 1
        return artifacts_path


# ──────────────────────────────────────────────────────────────────────────────
# dataset-level trace
# ──────────────────────────────────────────────────────────────────────────────


def dataset_summary(outcomes: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate the per-series traces of one dataset run, for the artifacts.

    Counts cover every outcome, including the failed ones; the means cover the
    series that produced a trace. Nothing here reads a test value: the summary is
    about the DECISION (how long it took, how it was split, how the leader moved,
    why it stopped), not about accuracy, which the per-series CSV already carries.
    """
    n_series = len(outcomes)
    n_ok = sum(1 for o in outcomes if getattr(o, "success", False))
    stop_reasons: Counter = Counter()
    verdicts: Counter = Counter()
    origins: Counter = Counter()
    score_weights: Counter = Counter()
    tools_called: Counter = Counter()
    overridden = 0
    derived_values: Dict[str, List[float]] = {}

    for o in outcomes:
        react = getattr(o, "react", None)
        if react is None:
            continue
        stop_reasons[str(react.stop_reason or "unknown")] += 1
        origins[str(react.final_attempt.origin) if react.final_attempt else "none"] += 1
        score_weights[str(react.score_weights_name or "config_preset")] += 1
        if react.overridden:
            overridden += 1
        for entry in (react.tools or {}).get("tools_called", []) or []:
            if entry.get("ok"):
                tools_called[str(entry.get("tool"))] += 1

        derived = getattr(react, "derived", None) or {}
        verdict = (derived.get("selection") or {}).get("verdict")
        if verdict:
            verdicts[str(verdict)] += 1
        for key in (
            "wall_clock_s", "llm_s", "tool_s", "llm_time_share", "turns",
            "scoring_calls", "non_scoring_calls", "best_changes",
            "time_to_first_best_s", "time_after_last_improvement_s",
        ):
            value = derived.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                derived_values.setdefault(key, []).append(float(value))

    means = {
        key: round(float(np.mean(values)), 4)
        for key, values in derived_values.items()
        if values
    }
    return {
        "n_series": n_series,
        "n_ok": n_ok,
        "n_failed": n_series - n_ok,
        "stop_reasons": dict(stop_reasons),
        "selection_verdicts": dict(verdicts),
        "final_origin": dict(origins),
        "overridden_accepted": overridden,
        "score_weights_used": dict(score_weights),
        "tool_calls_ok": dict(tools_called.most_common()),
        "total_wall_hours": round(sum(derived_values.get("wall_clock_s", [])) / 3600.0, 3),
        "mean_derived": means,
    }


def write_dataset_summary(outcomes: Sequence[Any], path: str) -> str:
    """Write `dataset_summary` next to the CSV. Never raises into the run."""
    payload = dataset_summary(outcomes)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return ""
    return path
