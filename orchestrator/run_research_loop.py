from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.context import init_context, generate_all_validations_context, get_context

from orchestrator.agents import (
    create_orchestrator_agent,
    create_proposer_agent,
    create_skeptic_agent,
    create_statistician_agent,
)
from orchestrator.utils import extract_json_object


DEFAULT_CANDIDATES = {
    "candidates": [
        {
            "name": "baseline_mean",
            "type": "baseline",
            "description": "Mean across all models for each horizon.",
            "formula": "y_hat(h) = mean_m preds_m(h)",
            "learns_weights": False,
            "constraints": "none",
            "risks": ["sensitive to outliers"],
            "validation_plan": "rolling",
            "params": {"method": "mean"},
        },
        {
            "name": "baseline_median",
            "type": "baseline",
            "description": "Median across models per horizon.",
            "formula": "y_hat(h) = median_m preds_m(h)",
            "learns_weights": False,
            "constraints": "none",
            "risks": ["may underperform if most models biased"],
            "validation_plan": "rolling",
            "params": {"method": "median"},
        },
        {
            "name": "trimmed_mean_20",
            "type": "baseline",
            "description": "Trimmed mean per horizon (robust).",
            "formula": "trim top/bottom then mean",
            "learns_weights": False,
            "constraints": "trim_ratio in [0,0.4]",
            "risks": ["too aggressive trim if few models"],
            "validation_plan": "rolling",
            "params": {"method": "trimmed_mean", "trim_ratio": 0.2},
        },
        {
            "name": "best_single_rolling",
            "type": "selection",
            "description": "Select best single model using only past windows (aggregate RMSE).",
            "formula": "m* = argmin_m RMSE_past(m); y_hat = preds_{m*}",
            "learns_weights": False,
            "constraints": "anti-leakage rolling selection",
            "risks": ["unstable with few windows"],
            "validation_plan": "rolling",
            "params": {"method": "best_single", "selection_metric": "rmse"},
        },
        {
            "name": "best_per_horizon_rolling",
            "type": "selection",
            "description": "Select best model per horizon using only past windows.",
            "formula": "for each h: m*(h)=argmin_m RMSE_past(m,h); y_hat(h)=preds_{m*(h)}(h)",
            "learns_weights": False,
            "constraints": "anti-leakage rolling selection",
            "risks": ["pointwise overfit"],
            "validation_plan": "rolling",
            "params": {"method": "best_per_horizon", "selection_metric": "rmse"},
        },
        {
            "name": "topk_mean_per_horizon_k3",
            "type": "selection",
            "description": "Top-k mean per horizon (k=3) with rolling selection.",
            "formula": "for each h: pick top-k by past RMSE, average",
            "learns_weights": False,
            "constraints": "anti-leakage rolling selection",
            "risks": ["depends on k"],
            "validation_plan": "rolling",
            "params": {"method": "topk_mean_per_horizon", "top_k": 3, "selection_metric": "rmse"},
        },
        {
            "name": "inv_rmse_weights_per_horizon_k3_shrink02",
            "type": "weighted",
            "description": "Inverse-RMSE weights per horizon (top-k=3) with shrinkage to uniform.",
            "formula": "w_m(h) ∝ 1/RMSE_past(m,h); y_hat(h)=Σ w_m(h) pred_m(h)",
            "learns_weights": True,
            "constraints": "w>=0, sum(w)=1; weights learned from past windows only",
            "risks": ["weight instability"],
            "validation_plan": "rolling",
            "params": {"method": "inverse_rmse_weights_per_horizon", "top_k": 3, "shrinkage": 0.2, "eps": 1e-8},
        },
        {
            "name": "ridge_stacking_per_horizon_l2_10",
            "type": "stacking",
            "description": "Ridge stacking per horizon with simplex projection (anti-leakage rolling fit).",
            "formula": "w(h)=argmin ||Xw-y||^2 + λ||w||^2, projected to simplex",
            "learns_weights": True,
            "constraints": "w>=0, sum(w)=1; fit uses past windows only",
            "risks": ["needs enough windows"],
            "validation_plan": "rolling",
            "params": {"method": "ridge_stacking_per_horizon", "l2": 10.0, "top_k": 5},
        },
    ]
}


def _ensure_validation(models: List[str], dataset_index: int) -> None:
    init_context()
    generate_all_validations_context(models, dataset_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--ollama-model", type=str, default="qwen3:14b")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-llm", action="store_true", help="If set, run proposer/skeptic/statistician. Otherwise evaluate defaults.")
    parser.add_argument("--rolling", type=str, default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--train-window", type=int, default=5)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        # fallback: use whatever already exists in context
        existing = get_context("models_available", [])
        models = existing if existing else []

    if models:
        _ensure_validation(models, args.dataset_index)

    candidates_payload: Any = DEFAULT_CANDIDATES

    if args.use_llm:
        proposer = create_proposer_agent(args.ollama_model, debug=args.debug)
        skeptic = create_skeptic_agent(args.ollama_model, debug=args.debug)
        statistician = create_statistician_agent(args.ollama_model, debug=args.debug)

        prompt = (
            "Propose candidate multi-step combination strategies for forecasting given that predictions are already computed. "
            "Return STRICT JSON only. Ensure anti-leakage: any selection/weights must be learned using only past windows."
        )
        prop_out = proposer.run(prompt)
        obj = extract_json_object(str(prop_out))
        candidates_payload = obj or DEFAULT_CANDIDATES

        sk_out = skeptic.run(json.dumps(candidates_payload))
        obj2 = extract_json_object(str(sk_out))
        candidates_payload = obj2 or candidates_payload

        st_out = statistician.run(json.dumps(candidates_payload))
        obj3 = extract_json_object(str(st_out))
        candidates_payload = obj3 or candidates_payload

    orchestrator = create_orchestrator_agent(args.ollama_model, debug=args.debug)

    eval_config = {
        "rolling": {"mode": args.rolling, "train_window": args.train_window},
        "metrics": {"mape_zero": "skip", "mape_epsilon": 1e-8},
        "score": {"a_rmse": 0.4, "b_smape": 0.3, "c_mape": 0.3, "d_pocid": 0.1},
    }

    prompt = json.dumps(
        {
            "candidates": candidates_payload.get("candidates", candidates_payload),
            "eval_config": eval_config,
        }
    )

    # The Orchestrator agent must call evaluate_strategies_tool.
    # We pass candidates as JSON string; config is passed separately in the tool call.
    orchestrator_prompt = (
        "You will receive JSON with keys candidates and eval_config. "
        "Call evaluate_strategies_tool(candidates_json=<candidates>, config_json=<eval_config>) and then return final JSON. "
        f"INPUT: {prompt}"
    )

    result = orchestrator.run(orchestrator_prompt)
    print(result)


if __name__ == "__main__":
    main()
