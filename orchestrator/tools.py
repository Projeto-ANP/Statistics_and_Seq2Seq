from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from agno.tools import tool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.context import get_context, set_context

from orchestrator.data_contract import load_validation_from_context
from orchestrator.evaluator import EvaluationConfig, evaluate_all
from orchestrator.metrics import MetricConfig, mape_safe, pocid_within_sequence, rmse_safe, smape_safe
from orchestrator.schemas import parse_candidates
from orchestrator.utils import extract_json_object


SCORE_PRESETS: Dict[str, Dict[str, float]] = {
    # Lower score is better; these are weights for the combined score.
    "balanced": {"a_rmse": 0.3, "b_smape": 0.3, "c_mape": 0.2, "d_pocid": 0.2},
    "rmse_focus": {"a_rmse": 0.5, "b_smape": 0.2, "c_mape": 0.2, "d_pocid": 0.1},
    "direction_focus": {"a_rmse": 0.25, "b_smape": 0.25, "c_mape": 0.1, "d_pocid": 0.4},
    "robust_smape": {"a_rmse": 0.2, "b_smape": 0.5, "c_mape": 0.1, "d_pocid": 0.2},
}


@tool
def proposer_brief_tool() -> str:
    """Returns a compact, deterministic brief for the Proposer.

    This is meant to be the ONLY tool the Proposer uses.
    It provides real, dataset-specific context so the Proposer can:
      - select a subset of strategies (whitelist selection)
      - choose a score preset
      - optionally recommend forcing debate
    """
    max_candidates = get_context("proposer_max_candidates", 12)
    try:
        max_candidates = int(max_candidates)
    except Exception:
        max_candidates = 12
    max_candidates = max(3, min(max_candidates, 30))

    config_json = get_context("config_json_for_proposer", "")
    summary = _build_validation_summary(config_json=config_json)
    recommended = _recommended_knobs(summary)

    # Candidate universe is deterministic and dataset-conditioned (whitelist).
    # Keep it bigger than the recommended shortlist so LLM can add/try alternatives in debate.
    universe = _candidate_universe_from_summary(summary)
    recommended = _suggest_candidates_from_summary(summary, max_candidates=int(max_candidates))

    out = {
        "validation_summary": summary,
        "recommended_knobs": recommended,
        "candidate_library": universe,
        "recommended_candidates": recommended,
        "score_presets": SCORE_PRESETS,
        "output_schema": {
            "selected_names": ["baseline_mean"],
            "params_overrides": {"topk_mean_per_horizon_k3": {"top_k": 3}},
            "score_preset": "balanced",
            "force_debate": False,
            "debate_margin": 0.02,
            "rationale": "short text",
        },
    }

    set_context("orchestrator_proposer_brief", out)
    tools_called = get_context("tools_called", [])
    if not isinstance(tools_called, list):
        tools_called = []
    tools_called.append("proposer_brief_tool")
    set_context("tools_called", tools_called)

    return json.dumps(out, indent=2)


def _nanmean(x: np.ndarray) -> float:
    v = float(np.nanmean(x))
    return v


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _compute_model_aggregates(data, metric_cfg: MetricConfig) -> List[Dict[str, Any]]:
    y_true = np.asarray(data.y_true, dtype=float)
    y_preds = np.asarray(data.y_preds, dtype=float)

    y_true_flat = y_true.reshape(-1)
    out: List[Dict[str, Any]] = []
    for j, name in enumerate(list(data.model_names)):
        y_pred_flat = y_preds[:, j, :].reshape(-1)
        rmse = rmse_safe(y_true_flat, y_pred_flat)
        smape = smape_safe(y_true_flat, y_pred_flat)
        mape = mape_safe(y_true_flat, y_pred_flat, metric_cfg)

        # Window-level POCID averaged across windows (project convention).
        pocids: List[float] = []
        for w in range(data.n_windows):
            pocids.append(pocid_within_sequence(y_true[w, :], y_preds[w, j, :]))
        pocid_mean = float(np.nanmean(np.asarray(pocids, dtype=float)))

        out.append(
            {
                "model": name,
                "RMSE": rmse,
                "SMAPE": smape,
                "MAPE": mape,
                "POCID": pocid_mean,
            }
        )

    out.sort(key=lambda d: (np.inf if np.isnan(d["RMSE"]) else d["RMSE"]))
    return out


def _compute_best_model_per_horizon(data) -> Dict[str, Any]:
    y_true = np.asarray(data.y_true, dtype=float)
    y_preds = np.asarray(data.y_preds, dtype=float)

    best_names: List[str] = []
    best_rmses: List[float] = []
    for h in range(data.horizon):
        rmses: List[float] = []
        for j in range(data.n_models):
            rmses.append(rmse_safe(y_true[:, h], y_preds[:, j, h]))
        rmses_arr = np.asarray(rmses, dtype=float)
        if np.all(np.isnan(rmses_arr)):
            best_names.append("unknown")
            best_rmses.append(float("nan"))
            continue
        j_best = int(np.nanargmin(rmses_arr))
        best_names.append(data.model_names[j_best])
        best_rmses.append(float(rmses_arr[j_best]))
    unique = sorted(set([n for n in best_names if n != "unknown"]))
    return {
        "best_model_per_horizon": best_names,
        "best_rmse_per_horizon": best_rmses,
        "unique_winners": unique,
        "n_unique_winners": len(unique),
    }


def _compute_disagreement_score(data) -> Dict[str, Any]:
    """Quantifies how much models disagree (robustly) across horizons.

    Higher => more outliers / spread => median/trimmed_mean tends to help.
    """

    y_true = np.asarray(data.y_true, dtype=float)
    y_preds = np.asarray(data.y_preds, dtype=float)

    eps = 1e-8
    rel_spreads: List[float] = []
    for h in range(data.horizon):
        # Robust spread across models per window at horizon h
        p10, p90 = np.nanpercentile(y_preds[:, :, h], [10, 90], axis=1)
        spread = p90 - p10
        scale = np.abs(np.nanmedian(y_true[:, h])) + eps
        rel_spreads.append(float(np.nanmean(spread / scale)))

    rel_spreads_arr = np.asarray(rel_spreads, dtype=float)
    return {
        "relative_spread_per_horizon": [float(x) for x in rel_spreads_arr.tolist()],
        "relative_spread_mean": float(np.nanmean(rel_spreads_arr)),
        "relative_spread_p90": float(np.nanpercentile(rel_spreads_arr, 90)) if np.any(~np.isnan(rel_spreads_arr)) else float("nan"),
    }


def _build_validation_summary(config_json: str = "") -> Dict[str, Any]:
    """Internal (non-tool) builder used by multiple tools."""

    data = load_validation_from_context()
    metric_cfg = MetricConfig()
    if config_json:
        cfg_obj = extract_json_object(config_json)
        if isinstance(cfg_obj, dict):
            if cfg_obj.get("mape_zero") in {"skip", "epsilon"}:
                metric_cfg.mape_zero = cfg_obj["mape_zero"]
            if "mape_epsilon" in cfg_obj:
                metric_cfg.mape_epsilon = float(cfg_obj["mape_epsilon"])

    aggregates = _compute_model_aggregates(data, metric_cfg)
    best_per_h = _compute_best_model_per_horizon(data)
    disagreement = _compute_disagreement_score(data)

    return {
        "n_windows": data.n_windows,
        "horizon": data.horizon,
        "n_models": data.n_models,
        "model_names": list(data.model_names),
        "models": aggregates,
        "best_per_horizon": best_per_h,
        "disagreement": disagreement,
    }


def _eval_config_from_json(config_json: str) -> EvaluationConfig:
    cfg = EvaluationConfig()
    if not config_json:
        return cfg
    cfg_obj = extract_json_object(config_json)
    if not isinstance(cfg_obj, dict):
        return cfg

    rolling = cfg_obj.get("rolling", {})
    if isinstance(rolling, dict):
        mode = rolling.get("mode")
        if mode in {"expanding", "rolling"}:
            cfg.rolling.mode = mode
        if "train_window" in rolling:
            cfg.rolling.train_window = int(rolling["train_window"])

    metrics = cfg_obj.get("metrics", {})
    if isinstance(metrics, dict):
        if metrics.get("mape_zero") in {"skip", "epsilon"}:
            cfg.metrics.mape_zero = metrics["mape_zero"]
        if "mape_epsilon" in metrics:
            cfg.metrics.mape_epsilon = float(metrics["mape_epsilon"])

    score = cfg_obj.get("score", {})
    if isinstance(score, dict):
        for k in ["a_rmse", "b_smape", "c_mape", "d_pocid"]:
            if k in score:
                setattr(cfg.score, k, float(score[k]))

    return cfg


def _extract_candidates_any(obj: Any) -> Any:
    if isinstance(obj, dict) and "candidates" in obj:
        return obj.get("candidates")
    return obj


def _per_horizon_winner_by_key(details: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not details:
        return {"winner": [], "counts": {}}
    per_h = details[0].get("per_horizon", {}).get(key, [])
    if not isinstance(per_h, list):
        return {"winner": [], "counts": {}}

    horizon = len(per_h)
    winners: List[str] = []
    for h in range(horizon):
        best_name = ""
        best_val = float("inf")
        for d in details:
            name = str(d.get("candidate", {}).get("name", ""))
            vals = d.get("per_horizon", {}).get(key, [])
            if not isinstance(vals, list) or h >= len(vals):
                continue
            v = _safe_float(vals[h])
            if not np.isfinite(v):
                continue
            # For POCID, higher is better; for error metrics, lower is better.
            if key.upper() == "POCID":
                if best_name == "" or v > best_val:
                    best_val = v
                    best_name = name
            else:
                if v < best_val:
                    best_val = v
                    best_name = name
        winners.append(best_name or "unknown")

    counts: Dict[str, int] = {}
    for w in winners:
        counts[w] = counts.get(w, 0) + 1
    return {"winner": winners, "counts": counts}


def _stability_flags(r: Dict[str, Any]) -> Dict[str, Any]:
    agg = r.get("aggregate", {}) if isinstance(r.get("aggregate"), dict) else {}
    stab = r.get("stability", {}) if isinstance(r.get("stability"), dict) else {}
    rmse = _safe_float(agg.get("RMSE"))
    rmse_std = _safe_float(stab.get("RMSE_std"))
    ratio = float("nan")
    if np.isfinite(rmse) and rmse > 0 and np.isfinite(rmse_std):
        ratio = rmse_std / rmse
    return {
        "rmse_std_over_rmse": ratio,
        "unstable": bool(np.isfinite(ratio) and ratio >= 0.25),
    }


def _recommended_knobs(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    n_windows = int(data_summary.get("n_windows", 0) or 0)
    n_models = int(data_summary.get("n_models", 0) or 0)
    rel_spread_mean = _safe_float(data_summary.get("disagreement", {}).get("relative_spread_mean"))
    high_disagreement = bool(np.isfinite(rel_spread_mean) and rel_spread_mean >= 0.25)

    base_top_k = max(2, min(int(round(np.sqrt(max(n_models, 1)))), max(n_models, 2)))
    if n_windows <= 4:
        return {
            "top_k": int(min(base_top_k, 5)),
            "shrinkage": 0.35,
            "l2": 50.0,
            "trim_ratio": 0.2 if high_disagreement else 0.1,
            "rationale": "few_windows => stronger regularization/shrinkage",
        }
    if n_windows <= 8:
        return {
            "top_k": int(min(base_top_k, 7)),
            "shrinkage": 0.25,
            "l2": 20.0,
            "trim_ratio": 0.2 if high_disagreement else 0.1,
            "rationale": "medium_windows => moderate regularization",
        }
    return {
        "top_k": int(min(base_top_k, 10)),
        "shrinkage": 0.15,
        "l2": 10.0,
        "trim_ratio": 0.2 if high_disagreement else 0.1,
        "rationale": "many_windows => allow more flexibility",
    }


def _suggest_candidates_from_summary(summary: Dict[str, Any], max_candidates: int = 10) -> Dict[str, Any]:
    """Deterministic heuristic: generate a candidate list conditioned on validation summary."""

    n_models = int(summary.get("n_models", 0) or 0)
    n_windows = int(summary.get("n_windows", 0) or 0)
    n_unique_winners = int(summary.get("best_per_horizon", {}).get("n_unique_winners", 0) or 0)
    rel_spread_mean = _safe_float(summary.get("disagreement", {}).get("relative_spread_mean"))

    aggregates = summary.get("models", [])
    best = aggregates[0] if isinstance(aggregates, list) and aggregates else {}
    second = aggregates[1] if isinstance(aggregates, list) and len(aggregates) > 1 else {}

    best_rmse = _safe_float(best.get("RMSE"))
    second_rmse = _safe_float(second.get("RMSE"))
    dominance = 0.0
    if np.isfinite(best_rmse) and np.isfinite(second_rmse) and second_rmse > 0:
        dominance = float((second_rmse - best_rmse) / second_rmse)

    # Hyperparams chosen deterministically from dataset characteristics
    top_k = int(round(np.sqrt(max(n_models, 1))))
    top_k = max(2, min(top_k, max(n_models, 2)))
    if n_windows <= 4:
        shrinkage = 0.35
        l2 = 50.0
    elif n_windows <= 8:
        shrinkage = 0.25
        l2 = 20.0
    else:
        shrinkage = 0.15
        l2 = 10.0

    trim_ratio = 0.2 if n_models >= 10 else 0.1

    # Disagreement threshold (relative)
    high_disagreement = bool(np.isfinite(rel_spread_mean) and rel_spread_mean >= 0.25)
    strong_single = bool(dominance >= 0.05)
    horizon_heterogeneous = bool(n_unique_winners >= 2)

    candidates: List[Dict[str, Any]] = []

    def _add(c: Dict[str, Any]) -> None:
        if len(candidates) >= int(max_candidates):
            return
        names = {x.get("name") for x in candidates}
        if c.get("name") not in names:
            candidates.append(c)

    _add(
        {
            "name": "baseline_mean",
            "type": "baseline",
            "description": "Simple mean across all models (baseline)",
            "formula": "y_hat(h)=mean_m y_m(h)",
            "learns_weights": False,
            "constraints": "No leakage (no fitting)",
            "risks": ["Sensitive to outliers"],
            "validation_plan": "rolling",
            "params": {"method": "mean"},
        }
    )

    if high_disagreement:
        _add(
            {
                "name": "robust_median",
                "type": "baseline",
                "description": "Robust median across models (helps when models disagree)",
                "formula": "y_hat(h)=median_m y_m(h)",
                "learns_weights": False,
                "constraints": "No leakage (no fitting)",
                "risks": ["May under-use strong models"],
                "validation_plan": "rolling",
                "params": {"method": "median"},
            }
        )
        _add(
            {
                "name": f"robust_trimmed_mean_{trim_ratio:.2f}",
                "type": "baseline",
                "description": "Trimmed mean across models (robust to outliers)",
                "formula": "Drop top/bottom fraction per horizon, average remaining",
                "learns_weights": False,
                "constraints": "No leakage (no fitting)",
                "risks": ["Can remove good extreme signals"],
                "validation_plan": "rolling",
                "params": {"method": "trimmed_mean", "trim_ratio": float(trim_ratio)},
            }
        )

    if strong_single:
        _add(
            {
                "name": "best_single_by_validation",
                "type": "selection",
                "description": "Select single best model by past validation RMSE (rolling)",
                "formula": "Pick argmin RMSE on past windows; use its forecast",
                "learns_weights": False,
                "constraints": "anti-leakage: selection uses only past windows",
                "risks": ["Can switch too often if few windows"],
                "validation_plan": "rolling",
                "params": {"method": "best_single"},
            }
        )

    if horizon_heterogeneous:
        _add(
            {
                "name": "best_per_horizon_by_validation",
                "type": "selection",
                "description": "Per-horizon best model selection by past validation RMSE (rolling)",
                "formula": "For each horizon h pick best model on past windows",
                "learns_weights": False,
                "constraints": "anti-leakage: selection uses only past windows",
                "risks": ["Can be unstable with few windows"],
                "validation_plan": "rolling",
                "params": {"method": "best_per_horizon"},
            }
        )

    _add(
        {
            "name": f"topk_mean_per_horizon_k{top_k}",
            "type": "selection",
            "description": "Top-k mean per horizon (rolling selection)",
            "formula": "For each horizon h: choose top-k models by past RMSE and average their preds",
            "learns_weights": False,
            "constraints": "anti-leakage: selection uses only past windows",
            "risks": ["Instability if few windows"],
            "validation_plan": "rolling",
            "params": {"method": "topk_mean_per_horizon", "top_k": int(top_k)},
        }
    )

    _add(
        {
            "name": f"inverse_rmse_weights_k{top_k}_sh{shrinkage:.2f}",
            "type": "weighted",
            "description": "Inverse-RMSE weights per horizon with shrinkage (rolling)",
            "formula": "w(h) \u221d 1/(RMSE+eps), shrunk toward uniform; apply per horizon",
            "learns_weights": True,
            "constraints": "anti-leakage: weights learned only from past windows",
            "risks": ["May overfit if too many models or few windows"],
            "validation_plan": "rolling",
            "params": {
                "method": "inverse_rmse_weights_per_horizon",
                "top_k": int(top_k),
                "shrinkage": float(shrinkage),
            },
        }
    )

    _add(
        {
            "name": f"ridge_stacking_l2{l2:.0f}_topk{min(top_k, 5)}",
            "type": "stacking",
            "description": "Ridge stacking per horizon (project to simplex), rolling training",
            "formula": "Fit ridge per horizon on past windows; project weights to simplex",
            "learns_weights": True,
            "constraints": "anti-leakage: ridge fit uses only past windows",
            "risks": ["May be unstable with very few windows"],
            "validation_plan": "rolling",
            "params": {
                "method": "ridge_stacking_per_horizon",
                "l2": float(l2),
                "top_k": int(min(top_k, 5)),
            },
        }
    )

    return {
        "candidates": candidates,
        "meta": {
            "n_models": n_models,
            "n_windows": n_windows,
            "top_k": top_k,
            "trim_ratio": float(trim_ratio),
            "shrinkage": float(shrinkage),
            "l2": float(l2),
            "flags": {
                "high_disagreement": high_disagreement,
                "strong_single": strong_single,
                "horizon_heterogeneous": horizon_heterogeneous,
            },
        },
    }


def _candidate_universe_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic whitelist of candidate strategies (dataset-conditioned ranges).

    This is larger than the recommended shortlist so the LLM can:
      - start with a focused subset (Proposer)
      - later add candidates that look good for specific objectives (e.g., POCID)
    without inventing new strategies.
    """

    n_models = int(summary.get("n_models", 0) or 0)
    n_windows = int(summary.get("n_windows", 0) or 0)

    # Top-k grid (bounded by n_models)
    base = int(round(np.sqrt(max(n_models, 1))))
    base = max(2, min(base, max(n_models, 2)))
    k_grid = sorted({2, min(3, max(n_models, 2)), min(5, max(n_models, 2)), base, min(10, max(n_models, 2))})
    k_grid = [int(k) for k in k_grid if 2 <= int(k) <= max(2, int(n_models or 2))]

    # Robustness grid (include recommended knobs so the universe matches the brief).
    rec = _recommended_knobs(summary)
    rec_trim = _safe_float(rec.get("trim_ratio"))
    rec_shrink = _safe_float(rec.get("shrinkage"))
    rec_l2 = _safe_float(rec.get("l2"))

    trim_grid = [0.05, 0.1, 0.2, 0.3]
    if np.isfinite(rec_trim):
        trim_grid.append(float(rec_trim))
    trim_grid = sorted({float(x) for x in trim_grid if 0.0 <= float(x) <= 0.4})

    shrink_grid = [0.0, 0.15, 0.25, 0.3, 0.35, 0.5]
    if np.isfinite(rec_shrink):
        shrink_grid.append(float(rec_shrink))
    shrink_grid = sorted({float(x) for x in shrink_grid if 0.0 <= float(x) <= 0.9})

    if n_windows <= 4:
        l2_grid = [20.0, 50.0, 100.0]
    elif n_windows <= 8:
        l2_grid = [10.0, 20.0, 50.0]
    else:
        l2_grid = [5.0, 10.0, 20.0]
    if np.isfinite(rec_l2):
        l2_grid.append(float(rec_l2))
    l2_grid = sorted({float(x) for x in l2_grid if 0.1 <= float(x) <= 1000.0})

    candidates: List[Dict[str, Any]] = []

    def _add(c: Dict[str, Any]) -> None:
        name = str(c.get("name", "")).strip()
        if not name:
            return
        if any(str(x.get("name")) == name for x in candidates):
            return
        candidates.append(c)

    _add(
        {
            "name": "baseline_mean",
            "type": "baseline",
            "description": "Simple mean across all models (baseline)",
            "formula": "y_hat(h)=mean_m y_m(h)",
            "learns_weights": False,
            "constraints": "No leakage (no fitting)",
            "risks": ["Sensitive to outliers"],
            "validation_plan": "rolling",
            "params": {"method": "mean"},
        }
    )

    _add(
        {
            "name": "robust_median",
            "type": "baseline",
            "description": "Robust median across models",
            "formula": "y_hat(h)=median_m y_m(h)",
            "learns_weights": False,
            "constraints": "No leakage (no fitting)",
            "risks": ["May under-use strong models"],
            "validation_plan": "rolling",
            "params": {"method": "median"},
        }
    )

    for tr in trim_grid:
        _add(
            {
                "name": f"trimmed_mean_r{tr:.2f}",
                "type": "baseline",
                "description": "Trimmed mean across models (robust to outliers)",
                "formula": "Drop top/bottom fraction per horizon, average remaining",
                "learns_weights": False,
                "constraints": "No leakage (no fitting)",
                "risks": ["Can remove good extreme signals"],
                "validation_plan": "rolling",
                "params": {"method": "trimmed_mean", "trim_ratio": float(tr)},
            }
        )

    _add(
        {
            "name": "best_single_by_validation",
            "type": "selection",
            "description": "Select single best model by past validation RMSE (rolling)",
            "formula": "Pick argmin RMSE on past windows; use its forecast",
            "learns_weights": False,
            "constraints": "anti-leakage: selection uses only past windows",
            "risks": ["Can switch too often if few windows"],
            "validation_plan": "rolling",
            "params": {"method": "best_single"},
        }
    )

    _add(
        {
            "name": "best_per_horizon_by_validation",
            "type": "selection",
            "description": "Per-horizon best model selection by past validation RMSE (rolling)",
            "formula": "For each horizon h pick best model on past windows",
            "learns_weights": False,
            "constraints": "anti-leakage: selection uses only past windows",
            "risks": ["Can be unstable with few windows"],
            "validation_plan": "rolling",
            "params": {"method": "best_per_horizon"},
        }
    )

    for k in k_grid:
        _add(
            {
                "name": f"topk_mean_per_horizon_k{k}",
                "type": "selection",
                "description": "Top-k mean per horizon (rolling selection)",
                "formula": "For each horizon h: choose top-k models by past RMSE and average their preds",
                "learns_weights": False,
                "constraints": "anti-leakage: selection uses only past windows",
                "risks": ["Instability if few windows"],
                "validation_plan": "rolling",
                "params": {"method": "topk_mean_per_horizon", "top_k": int(k)},
            }
        )
        for sh in shrink_grid:
            _add(
                {
                    "name": f"inverse_rmse_weights_k{k}_sh{sh:.2f}",
                    "type": "weighted",
                    "description": "Inverse-RMSE weights per horizon with shrinkage (rolling)",
                    "formula": "w(h) ∝ 1/(RMSE+eps), shrunk toward uniform; apply per horizon",
                    "learns_weights": True,
                    "constraints": "anti-leakage: weights learned only from past windows",
                    "risks": ["May overfit if too many models or few windows"],
                    "validation_plan": "rolling",
                    "params": {
                        "method": "inverse_rmse_weights_per_horizon",
                        "top_k": int(k),
                        "shrinkage": float(sh),
                    },
                }
            )

    for l2 in l2_grid:
        for k in sorted({min(3, max(n_models, 2)), min(5, max(n_models, 2))}):
            k = int(max(2, min(int(k), max(2, int(n_models or 2)))))
            _add(
                {
                    "name": f"ridge_stacking_l2{l2:.0f}_topk{k}",
                    "type": "stacking",
                    "description": "Ridge stacking per horizon (project to simplex), rolling training",
                    "formula": "Fit ridge per horizon on past windows; project weights to simplex",
                    "learns_weights": True,
                    "constraints": "anti-leakage: ridge fit uses only past windows",
                    "risks": ["May be unstable with very few windows"],
                    "validation_plan": "rolling",
                    "params": {
                        "method": "ridge_stacking_per_horizon",
                        "l2": float(l2),
                        "top_k": int(k),
                    },
                }
            )

    return {
        "candidates": candidates,
        "meta": {
            "n_models": n_models,
            "n_windows": n_windows,
            "counts": {
                "total": len(candidates),
                "k_grid": k_grid,
                "trim_grid": trim_grid,
                "shrink_grid": shrink_grid,
                "l2_grid": l2_grid,
            },
        },
    }


@tool
def list_candidate_schema_tool() -> str:
    """Returns the strict JSON schema expected for candidate strategies."""

    schema = {
        "candidates": [
            {
                "name": "topk_mean_per_horizon_k3",
                "type": "selection",
                "description": "Top-k mean per horizon (rolling selection)",
                "formula": "For each horizon h: choose top-k models by past RMSE and average their preds",
                "learns_weights": False,
                "constraints": "anti-leakage: selection uses only past windows",
                "risks": ["instability if few windows"],
                "validation_plan": "rolling",
                "params": {"method": "topk_mean_per_horizon", "top_k": 3},
            }
        ]
    }
    return json.dumps(schema, indent=2)


@tool
def summarize_validation_tool(config_json: str = "") -> str:
    """Deterministically summarizes validation behavior for data-driven proposals.

    Returns JSON with per-model aggregates and per-horizon winners.
    """

    summary = _build_validation_summary(config_json=config_json)

    set_context("orchestrator_validation_summary", summary)
    tools_called = get_context("tools_called", [])
    tools_called.append("summarize_validation_tool")
    set_context("tools_called", tools_called)

    return json.dumps(summary, indent=2)


@tool
def generate_data_driven_candidates_tool(max_candidates: int = 10, config_json: str = "") -> str:
    """Deterministically generates a candidate set conditioned on validation data.

    This avoids random exploration while still adapting per dataset (not fixed).
    """

    summary = _build_validation_summary(config_json=config_json)
    out = _suggest_candidates_from_summary(summary, max_candidates=int(max_candidates))

    set_context("orchestrator_suggested_candidates", out)
    tools_called = get_context("tools_called", [])
    tools_called.append("generate_data_driven_candidates_tool")
    set_context("tools_called", tools_called)

    return json.dumps(out, indent=2)


@tool
def build_debate_packet_tool(candidates_json: str = "", config_json: str = "", top_n: int = 5) -> str:
    """Builds a deterministic debate packet with real numbers.

    The packet is meant to be consumed by LLM agents so they can interpret,
    critique and propose *bounded* knob edits without inventing metrics.
    """

    # Prefer context-provided inputs so LLM can't break the tool call by passing bad args.
    ctx_candidates = get_context("candidates_json_for_debate", "")
    ctx_config = get_context("config_json_for_debate", "")
    ctx_top_n = get_context("debate_top_n", None)

    if isinstance(ctx_candidates, str) and ctx_candidates.strip():
        candidates_json = ctx_candidates
    if isinstance(ctx_config, str) and ctx_config.strip():
        config_json = ctx_config
    if ctx_top_n is not None:
        try:
            top_n = int(ctx_top_n)
        except Exception:
            pass

    data = load_validation_from_context()
    cfg = _eval_config_from_json(config_json)

    obj = extract_json_object(candidates_json)
    candidates_obj = _extract_candidates_any(obj)
    candidates = parse_candidates(candidates_obj)
    if not candidates:
        return json.dumps({"error": "No valid candidates parsed"})

    summary = _build_validation_summary(config_json="")

    eval_result = evaluate_all(data, candidates, cfg)
    details = eval_result.get("details", []) if isinstance(eval_result, dict) else []
    if not isinstance(details, list):
        details = []

    ranking = eval_result.get("ranking", []) if isinstance(eval_result, dict) else []
    if not isinstance(ranking, list):
        ranking = []

    top_n = int(top_n)
    top_n = max(1, min(top_n, 10))
    top_rank = ranking[:top_n]

    # Margin between top-1 and top-2 (lower score is better)
    margin = None
    if len(ranking) >= 2:
        s1 = _safe_float(ranking[0].get("score"))
        s2 = _safe_float(ranking[1].get("score"))
        if np.isfinite(s1) and np.isfinite(s2):
            margin = float(s2 - s1)

    per_h_rmse = _per_horizon_winner_by_key(details, "RMSE")
    per_h_smape = _per_horizon_winner_by_key(details, "SMAPE")
    per_h_mape = _per_horizon_winner_by_key(details, "MAPE")

    # Stability flags for the current best
    best_detail = eval_result.get("best") if isinstance(eval_result, dict) else None
    best_flags = _stability_flags(best_detail) if isinstance(best_detail, dict) else {}

    def _top_by_aggregate(details_list: List[Dict[str, Any]], key: str, higher_is_better: bool, n: int = 5) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for d in details_list:
            if not isinstance(d, dict):
                continue
            cand = d.get("candidate", {}) if isinstance(d.get("candidate"), dict) else {}
            name = str(cand.get("name", ""))
            agg = d.get("aggregate", {}) if isinstance(d.get("aggregate"), dict) else {}
            v = _safe_float(agg.get(key))
            if not name:
                continue
            out_rows.append({"name": name, key: v})
        out_rows = [r for r in out_rows if np.isfinite(_safe_float(r.get(key)))]
        out_rows.sort(key=lambda r: _safe_float(r.get(key)), reverse=bool(higher_is_better))
        return out_rows[: int(max(1, min(int(n), 10)))]

    # Optional: evaluate a larger candidate universe so debate agents can add candidates
    # without inventing names.
    universe_payload = get_context("candidate_universe_json_for_debate", "")
    universe_eval = None
    universe_top = []
    universe_leaderboards = {}
    universe_names: List[str] = []
    if isinstance(universe_payload, str) and universe_payload.strip():
        u_obj = extract_json_object(universe_payload)
        u_candidates_obj = _extract_candidates_any(u_obj)
        u_candidates = parse_candidates(u_candidates_obj)
        if u_candidates:
            try:
                universe_eval = evaluate_all(data, u_candidates, cfg)
                u_details = universe_eval.get("details", []) if isinstance(universe_eval, dict) else []
                if not isinstance(u_details, list):
                    u_details = []
                u_ranking = universe_eval.get("ranking", []) if isinstance(universe_eval, dict) else []
                if not isinstance(u_ranking, list):
                    u_ranking = []

                universe_top = u_ranking[: int(max(1, min(int(top_n), 10)))]
                universe_leaderboards = {
                    "RMSE": _top_by_aggregate(u_details, "RMSE", higher_is_better=False, n=int(top_n)),
                    "SMAPE": _top_by_aggregate(u_details, "SMAPE", higher_is_better=False, n=int(top_n)),
                    "MAPE": _top_by_aggregate(u_details, "MAPE", higher_is_better=False, n=int(top_n)),
                    "POCID": _top_by_aggregate(u_details, "POCID", higher_is_better=True, n=int(top_n)),
                }
                universe_names = [str(x.get("candidate", {}).get("name")) for x in u_details if isinstance(x, dict)]
                universe_names = sorted({n for n in universe_names if n})
            except Exception:
                universe_eval = None

    packet = {
        "validation_summary": summary,
        "eval_config": eval_result.get("config", {}),
        "candidate_ranking_top": top_rank,
        "score_margin_top2": margin,
        "per_horizon_winners": {
            "RMSE": per_h_rmse,
            "SMAPE": per_h_smape,
            "MAPE": per_h_mape,
        },
        "universe": {
            "enabled": bool(universe_eval is not None),
            "candidate_names": universe_names,
            "ranking_top": universe_top,
            "leaderboards": universe_leaderboards,
        },
        "best": {
            "name": (best_detail or {}).get("candidate", {}).get("name") if isinstance(best_detail, dict) else None,
            "aggregate": (best_detail or {}).get("aggregate") if isinstance(best_detail, dict) else None,
            "stability": (best_detail or {}).get("stability") if isinstance(best_detail, dict) else None,
            "stability_flags": best_flags,
        },
        "recommended_knobs": _recommended_knobs(summary),
        "allowed_edits": {
            "params_allowed": ["top_k", "trim_ratio", "shrinkage", "l2"],
            "ranges": {
                "top_k": {"min": 2, "max": max(2, int(summary.get("n_models", 2)))},
                "trim_ratio": {"min": 0.0, "max": 0.4},
                "shrinkage": {"min": 0.0, "max": 0.9},
                "l2": {"min": 0.1, "max": 1000.0},
            },
            "can_remove_candidates": True,
            "can_add_candidates": True,
            "cannot_change_method": True,
        },
    }

    set_context("orchestrator_debate_packet", packet)
    tools_called = get_context("tools_called", [])
    tools_called.append("build_debate_packet_tool")
    set_context("tools_called", tools_called)

    return json.dumps(packet, indent=2)


@tool
def evaluate_strategies_tool(candidates_json: str, config_json: str = "") -> str:
    """Deterministically evaluates candidate strategies on the validation windows.

    Args:
        candidates_json: JSON list OR JSON object with key "candidates".
        config_json: optional JSON to override evaluation config.

    Returns:
        JSON with baseline, ranking, and details.
    """

    data = load_validation_from_context()

    obj = extract_json_object(candidates_json)
    if isinstance(obj, dict) and "candidates" in obj:
        candidates_obj = obj.get("candidates")
    else:
        candidates_obj = obj

    candidates = parse_candidates(candidates_obj)
    if not candidates:
        return json.dumps({"error": "No valid candidates parsed"})

    cfg = EvaluationConfig()
    if config_json:
        cfg_obj = extract_json_object(config_json)
        if isinstance(cfg_obj, dict):
            rolling = cfg_obj.get("rolling", {})
            if isinstance(rolling, dict):
                mode = rolling.get("mode")
                if mode in {"expanding", "rolling"}:
                    cfg.rolling.mode = mode
                if "train_window" in rolling:
                    cfg.rolling.train_window = int(rolling["train_window"])

            metrics = cfg_obj.get("metrics", {})
            if isinstance(metrics, dict):
                if metrics.get("mape_zero") in {"skip", "epsilon"}:
                    cfg.metrics.mape_zero = metrics["mape_zero"]
                if "mape_epsilon" in metrics:
                    cfg.metrics.mape_epsilon = float(metrics["mape_epsilon"])

            score = cfg_obj.get("score", {})
            if isinstance(score, dict):
                for k in ["a_rmse", "b_smape", "c_mape", "d_pocid"]:
                    if k in score:
                        setattr(cfg.score, k, float(score[k]))

    result = evaluate_all(data, candidates, cfg)

    set_context("orchestrator_last_eval", result)
    tools_called = get_context("tools_called", [])
    tools_called.append("evaluate_strategies_tool")
    set_context("tools_called", tools_called)

    return json.dumps(result, indent=2)
