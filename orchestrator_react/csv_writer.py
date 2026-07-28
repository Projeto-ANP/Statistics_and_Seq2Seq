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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


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
    "effective_models",
    "n_effective_models",
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
]

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
    }

    fields = outcome.csv_fields()
    fields["llm_artifacts_path"] = llm_artifacts_path
    row.update(fields)
    return row


def artifacts_payload(outcome: Any) -> Dict[str, Any]:
    """Full per-series audit trail, written next to the CSV.

    The CSV keeps the compact trajectory; this keeps everything, including the raw
    cards and the warnings, for the cases where a row looks odd and needs digging.
    """
    return {
        "dataset": outcome.dataset,
        "dataset_index": outcome.dataset_index,
        "success": outcome.success,
        "error": outcome.error,
        "config": outcome.config.to_dict() if outcome.config else None,
        "decision": outcome.decision(),
        "series_card": outcome.series_card,
        "pool_card": outcome.pool_card,
        "diagnosis": outcome.diagnosis,
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
        },
        "predict_debug": outcome.predict_debug,
        "sanity": outcome.sanity,
        "report_text": outcome.report_text,
        "warnings": outcome.warnings,
    }


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
