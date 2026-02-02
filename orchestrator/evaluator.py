from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from orchestrator.data_contract import ValidationData
from orchestrator.metrics import MetricConfig, compute_metrics_1d, pocid_within_sequence
from orchestrator.schemas import CandidateStrategy
from orchestrator.strategies import RollingConfig, generate_combined_predictions


@dataclass
class ScoreConfig:
    # score = a*RMSE_norm + b*SMAPE_norm + c*MAPE_norm - d*POCID_norm
    a_rmse: float = 0.4
    b_smape: float = 0.3
    c_mape: float = 0.3
    d_pocid: float = 0.1


@dataclass
class EvaluationConfig:
    rolling: RollingConfig = field(
        default_factory=lambda: RollingConfig(mode="expanding", train_window=5)
    )
    metrics: MetricConfig = field(
        default_factory=lambda: MetricConfig(mape_zero="skip", mape_epsilon=1e-8)
    )
    score: ScoreConfig = field(default_factory=ScoreConfig)


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(np.asarray(x, dtype=float)))


def evaluate_candidate(
    data: ValidationData,
    candidate: CandidateStrategy,
    cfg: EvaluationConfig,
) -> Dict[str, Any]:
    combined_preds, debug = generate_combined_predictions(
        data.y_true, data.y_preds, data.model_names, candidate, cfg.rolling
    )

    y_true = data.y_true
    n_windows, horizon = y_true.shape

    # Metrics per horizon across windows
    per_horizon: Dict[str, List[float]] = {"MAPE": [], "SMAPE": [], "RMSE": [], "POCID": []}

    for h in range(horizon):
        m = compute_metrics_1d(y_true[:, h], combined_preds[:, h], cfg.metrics)
        per_horizon["MAPE"].append(m["MAPE"])
        per_horizon["SMAPE"].append(m["SMAPE"])
        per_horizon["RMSE"].append(m["RMSE"])

        # POCID across windows for a fixed horizon (direction between origins)
        if n_windows >= 2:
            try:
                from all_functions import pocid

                per_horizon["POCID"].append(float(pocid(y_true[:, h], combined_preds[:, h])))
            except Exception:
                per_horizon["POCID"].append(float("nan"))
        else:
            per_horizon["POCID"].append(float("nan"))

    # Window-level (sequence) POCID matches existing convention
    pocid_window = [pocid_within_sequence(y_true[i, :], combined_preds[i, :]) for i in range(n_windows)]

    # Aggregate metrics (mean over horizons)
    aggregate = {
        "MAPE": _nanmean(per_horizon["MAPE"]),
        "SMAPE": _nanmean(per_horizon["SMAPE"]),
        "RMSE": _nanmean(per_horizon["RMSE"]),
        "POCID": _nanmean(pocid_window),
    }

    # Stability: std across windows for aggregate errors
    # (compute per-window error metrics on the horizon-vector)
    window_metrics: List[Dict[str, float]] = []
    for i in range(n_windows):
        m = compute_metrics_1d(y_true[i, :], combined_preds[i, :], cfg.metrics)
        m["POCID"] = pocid_within_sequence(y_true[i, :], combined_preds[i, :])
        window_metrics.append(m)

    stability = {
        "MAPE_std": float(np.nanstd([m["MAPE"] for m in window_metrics])),
        "SMAPE_std": float(np.nanstd([m["SMAPE"] for m in window_metrics])),
        "RMSE_std": float(np.nanstd([m["RMSE"] for m in window_metrics])),
        "POCID_std": float(np.nanstd([m["POCID"] for m in window_metrics])),
    }

    return {
        "candidate": candidate.to_dict(),
        "aggregate": aggregate,
        "per_horizon": per_horizon,
        "pocid_window_mean": aggregate["POCID"],
        "stability": stability,
        "debug": debug,
    }


def _score_from_normed(
    rmse_n: float,
    smape_n: float,
    mape_n: float,
    pocid: float,
    score_cfg: ScoreConfig,
) -> float:
    # POCID is already in [0, 100]; normalize to [0, 1]
    pocid_n = float(pocid) / 100.0 if np.isfinite(pocid) else 0.0
    return (
        score_cfg.a_rmse * rmse_n
        + score_cfg.b_smape * smape_n
        + score_cfg.c_mape * mape_n
        - score_cfg.d_pocid * pocid_n
    )


def evaluate_all(
    data: ValidationData,
    candidates: List[CandidateStrategy],
    cfg: Optional[EvaluationConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EvaluationConfig()

    if not candidates:
        raise ValueError("No candidates provided")

    # Always compute baseline mean for normalization
    baseline = CandidateStrategy(
        name="baseline_mean",
        type="baseline",
        description="Mean across models",
        formula="mean across models per horizon",
        learns_weights=False,
        params={"method": "mean"},
    )
    baseline_eval = evaluate_candidate(data, baseline, cfg)
    b = baseline_eval["aggregate"]

    results: List[Dict[str, Any]] = []
    for c in candidates:
        res = evaluate_candidate(data, c, cfg)
        a = res["aggregate"]

        rmse_n = float(a["RMSE"]) / float(b["RMSE"]) if b["RMSE"] and np.isfinite(b["RMSE"]) else float("inf")
        smape_n = float(a["SMAPE"]) / float(b["SMAPE"]) if b["SMAPE"] and np.isfinite(b["SMAPE"]) else float("inf")
        mape_n = float(a["MAPE"]) / float(b["MAPE"]) if b["MAPE"] and np.isfinite(b["MAPE"]) else float("inf")

        score = _score_from_normed(rmse_n, smape_n, mape_n, float(a["POCID"]), cfg.score)

        res["normalized_vs_baseline"] = {"RMSE": rmse_n, "SMAPE": smape_n, "MAPE": mape_n, "POCID": float(a["POCID"]) / 100.0}
        res["score"] = float(score)
        results.append(res)

    ranked = sorted(results, key=lambda r: r.get("score", float("inf")))

    return {
        "baseline": baseline_eval,
        "ranking": [
            {
                "name": r["candidate"]["name"],
                "type": r["candidate"]["type"],
                "score": r["score"],
                "aggregate": r["aggregate"],
                "stability": r["stability"],
            }
            for r in ranked
        ],
        "details": ranked,
        "best": ranked[0] if ranked else None,
        "config": {
            "rolling": {"mode": cfg.rolling.mode, "train_window": cfg.rolling.train_window},
            "mape_zero": cfg.metrics.mape_zero,
            "score": {
                "a_rmse": cfg.score.a_rmse,
                "b_smape": cfg.score.b_smape,
                "c_mape": cfg.score.c_mape,
                "d_pocid": cfg.score.d_pocid,
            },
        },
    }
