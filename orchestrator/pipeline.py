from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from agent.context import get_context, set_context

from orchestrator.evaluator import EvaluationConfig, evaluate_all
from orchestrator.final_predictor import predict_final_from_context
from orchestrator.schemas import CandidateStrategy
from orchestrator.strategies import RollingConfig
from orchestrator.utils import extract_json_object
from orchestrator.schemas import parse_candidates
from orchestrator.tools import SCORE_PRESETS
from orchestrator.agents import (
    create_proposer_agent,
    create_skeptic_agent,
    create_statistician_agent,
)


ALLOWED_PARAM_EDITS = {"top_k", "trim_ratio", "shrinkage", "l2"}


def _run_agent_with_retry(
    agent_func: Callable[[], str],
    agent_name: str,
    max_retries: int = 3,
    log_func: Optional[Callable[[str], None]] = None,
) -> tuple[str, Dict[str, Any]]:
    """Run an agent with retry logic for JSON parsing failures.
    
    Args:
        agent_func: Callable that returns agent output string
        agent_name: Name of agent (for logging)
        max_retries: Maximum number of retry attempts (default 3)
        log_func: Optional logging function
    
    Returns:
        (raw_output, parsed_json_object)
    
    Raises:
        RuntimeError if all retries fail
    """
    if log_func is None:
        log_func = print
    
    for attempt in range(1, max_retries + 1):
        try:
            log_func(f"{agent_name}: attempt {attempt}/{max_retries}")
            t0 = time.perf_counter()
            output = agent_func()
            elapsed = time.perf_counter() - t0
            log_func(f"{agent_name}: response received in {elapsed:.1f}s")
            log_func(f"{agent_name} raw (first 2000 chars): {str(output)[:2000]}")
            
            parsed_obj = extract_json_object(str(output))
            if parsed_obj is None or not isinstance(parsed_obj, dict):
                if attempt < max_retries:
                    log_func(f"{agent_name}: invalid JSON (attempt {attempt}, retrying...)")
                    continue
                else:
                    raise RuntimeError(
                        f"{agent_name} did not return valid JSON after {max_retries} attempts (hard-stop). "
                        f"Raw (first 2000 chars): {str(output)[:2000]}"
                    )
            
            log_func(f"{agent_name}: successfully parsed JSON")
            return output, parsed_obj
        
        except Exception as e:
            if attempt < max_retries:
                log_func(f"{agent_name}: error on attempt {attempt}: {e} (retrying...)")
                continue
            else:
                raise RuntimeError(
                    f"{agent_name} failed after {max_retries} attempts: {e} (hard-stop)"
                )
    
    raise RuntimeError(f"{agent_name} exhausted all {max_retries} retry attempts")



def _validate_actions_against_universe(
    actions: Dict[str, Any],
    universe_names: List[str],
    current_names: Optional[List[str]],
    who: str,
) -> Dict[str, Any]:
    """Validate that LLM actions only reference real candidates.

    We hard-stop on unknown names to avoid silent hallucinations.
    """

    if not isinstance(actions, dict):
        raise RuntimeError(f"{who} returned non-dict actions (hard-stop)")

    valid_set = {str(n) for n in universe_names if str(n)}
    current_set = {str(n) for n in (current_names or []) if str(n)}

    add_names = actions.get("add_names", [])
    if isinstance(add_names, str):
        add_names = [add_names]
    if not isinstance(add_names, list):
        raise RuntimeError(f"{who}.add_names must be a list (hard-stop)")
    add_names_norm = [str(x) for x in add_names if str(x)]
    unknown_add = [n for n in add_names_norm if n not in valid_set]
    if unknown_add:
        raise RuntimeError(
            f"{who} tried to add unknown candidates: {unknown_add}. "
            f"Valid candidates: {sorted(valid_set)} (hard-stop)"
        )

    remove_names = actions.get("remove_names", [])
    if isinstance(remove_names, str):
        remove_names = [remove_names]
    if not isinstance(remove_names, list):
        raise RuntimeError(f"{who}.remove_names must be a list (hard-stop)")
    remove_names_norm = [str(x) for x in remove_names if str(x)]
    # Disallow removing candidates that are neither currently present nor being added.
    allowed_remove = set(current_set) | set(add_names_norm)
    unknown_remove = [n for n in remove_names_norm if n not in allowed_remove]
    if unknown_remove:
        raise RuntimeError(
            f"{who} tried to remove candidates not in current set: {unknown_remove}. "
            f"Current candidates: {sorted(current_set)} (hard-stop)"
        )

    overrides_raw = actions.get("params_overrides", {})
    if overrides_raw is None:
        overrides_raw = {}
    if not isinstance(overrides_raw, dict):
        raise RuntimeError(f"{who}.params_overrides must be an object/dict (hard-stop)")

    allowed_override = set(current_set) | set(add_names_norm)
    override_keys = [str(k) for k in overrides_raw.keys()]
    cand_override_keys = [k for k in override_keys if k in allowed_override]
    knob_override_keys = [k for k in override_keys if k in ALLOWED_PARAM_EDITS]
    unknown_override_keys = [k for k in override_keys if k not in allowed_override and k not in ALLOWED_PARAM_EDITS]
    if unknown_override_keys:
        raise RuntimeError(
            f"{who} tried to override params for candidates not in current/add set: {unknown_override_keys}. "
            f"Current candidates: {sorted(current_set)} (hard-stop)"
        )

    # Normalize overrides: allow either per-candidate overrides or a flat "knob override" that
    # applies the same params to every candidate in current/add set. This prevents false hard-stops
    # when agents return {{"trim_ratio": 0.1}} instead of {{"cand": {"trim_ratio": 0.1}}}.
    overrides: Dict[str, Dict[str, Any]] = {}

    # Candidate-specific overrides (classic path)
    for cand in cand_override_keys:
        ov = overrides_raw.get(cand, {})
        if not isinstance(ov, dict):
            raise RuntimeError(f"{who}.params_overrides['{cand}'] must be a dict (hard-stop)")
        overrides[cand] = dict(ov)

    # Global knob overrides (apply to all current/add candidates)
    if knob_override_keys:
        global_override = {k: overrides_raw[k] for k in knob_override_keys}
        for cand in allowed_override or current_set:
            if cand not in overrides:
                overrides[cand] = {}
            overrides[cand].update(global_override)

    # Validate allowed keys for each override payload
    for cand, ov in overrides.items():
        if not isinstance(ov, dict):
            raise RuntimeError(f"{who}.params_overrides['{cand}'] must be a dict (hard-stop)")
        bad_keys = [k for k in ov.keys() if str(k) not in ALLOWED_PARAM_EDITS and str(k) != "method"]
        if bad_keys:
            raise RuntimeError(
                f"{who} used unsupported override keys for '{cand}': {bad_keys}. "
                f"Allowed: {sorted(ALLOWED_PARAM_EDITS)} (hard-stop)"
            )

    # Return normalized copy
    return {
        "add_names": add_names_norm,
        "remove_names": remove_names_norm,
        "params_overrides": overrides,
        "rationale": actions.get("rationale"),
        "changes": actions.get("changes"),
        "when_good": actions.get("when_good"),
    }


def _apply_actions_to_payload(
    payload: Dict[str, Any],
    actions: Dict[str, Any],
    universe_by_name: Dict[str, Dict[str, Any]],
    n_models: int,
) -> Dict[str, Any]:
    """Apply LLM actions while keeping changes bounded and audit-friendly."""

    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        return payload

    add_names = actions.get("add_names", [])
    if isinstance(add_names, str):
        add_names = [add_names]
    if not isinstance(add_names, list):
        add_names = []
    add_names = [str(x) for x in add_names if str(x)]

    remove_names = actions.get("remove_names", [])
    if isinstance(remove_names, str):
        remove_names = [remove_names]
    if not isinstance(remove_names, list):
        remove_names = []
    remove_set = {str(x) for x in remove_names if str(x)}

    overrides = actions.get("params_overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}

    def _apply_overrides_to_params(base_params: Dict[str, Any], override_obj: Dict[str, Any]) -> Dict[str, Any]:
        o = override_obj if isinstance(override_obj, dict) else {}
        if "method" in o:
            o = {k: v for k, v in o.items() if k != "method"}

        new_params = dict(base_params)
        for k, v in o.items():
            if k not in ALLOWED_PARAM_EDITS:
                continue
            if k == "top_k":
                vv = _clamp_int(v, 2, max(2, int(n_models)))
                if vv is not None:
                    new_params[k] = vv
            elif k == "trim_ratio":
                vv = _clamp_float(v, 0.0, 0.4)
                if vv is not None:
                    new_params[k] = vv
            elif k == "shrinkage":
                vv = _clamp_float(v, 0.0, 0.9)
                if vv is not None:
                    new_params[k] = vv
            elif k == "l2":
                vv = _clamp_float(v, 0.1, 1000.0)
                if vv is not None:
                    new_params[k] = vv
        return new_params

    out_candidates: List[Dict[str, Any]] = []
    current_by_name: Dict[str, Dict[str, Any]] = {}
    for c in payload.get("candidates", []):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", ""))
        if not name:
            continue
        current_by_name[name] = c
        if name in remove_set:
            continue

        base = dict(c)
        base_params = base.get("params", {}) if isinstance(base.get("params"), dict) else {}
        o = overrides.get(name, {})
        base["params"] = _apply_overrides_to_params(base_params, o)
        out_candidates.append(base)

    # Add requested candidates from the universe (if not already present).
    names_out = {str(x.get("name")) for x in out_candidates if isinstance(x, dict)}
    for n in add_names:
        if n in remove_set:
            continue
        if n in names_out:
            continue
        cand = universe_by_name.get(n)
        if isinstance(cand, dict):
            base = dict(cand)
            base_params = base.get("params", {}) if isinstance(base.get("params"), dict) else {}
            o = overrides.get(n, {})
            base["params"] = _apply_overrides_to_params(base_params, o)
            out_candidates.append(base)
            names_out.add(n)

    # Safety: if edits removed too much, keep at least 2 candidates.
    if len(out_candidates) < 2:
        base = universe_by_name.get("baseline_mean")
        if isinstance(base, dict) and "baseline_mean" not in names_out:
            out_candidates.insert(0, dict(base))
            names_out.add("baseline_mean")

    if len(out_candidates) < 2:
        return payload
    return {"candidates": out_candidates, "meta": payload.get("meta")}


def _clamp_float(x: Any, lo: float, hi: float) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v == v):
        return None
    return float(max(lo, min(hi, v)))


def _clamp_int(x: Any, lo: int, hi: int) -> Optional[int]:
    try:
        v = int(x)
    except Exception:
        return None
    return int(max(lo, min(hi, v)))


def _sanitize_candidate_payload(
    original_payload: Dict[str, Any],
    revised_text: str,
    n_models: int,
) -> Dict[str, Any]:
    """Keeps LLM freedom bounded: only allows knob edits and candidate removal.

    - No new candidates are allowed.
    - params.method cannot change.
    - Only params in ALLOWED_PARAM_EDITS are accepted (clamped).
    - All other fields are kept from the original candidate definitions.
    """

    original_list = original_payload.get("candidates", [])
    if not isinstance(original_list, list) or not original_list:
        return original_payload

    original_by_name: Dict[str, Dict[str, Any]] = {}
    for c in original_list:
        if isinstance(c, dict) and c.get("name"):
            original_by_name[str(c.get("name"))] = c

    revised_obj = extract_json_object(str(revised_text))
    if isinstance(revised_obj, dict) and isinstance(revised_obj.get("candidates"), list):
        revised_list = revised_obj.get("candidates")
    elif isinstance(revised_obj, list):
        revised_list = revised_obj
    else:
        return original_payload

    sanitized: List[Dict[str, Any]] = []
    seen = set()
    for item in revised_list:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name or name not in original_by_name or name in seen:
            continue

        base = dict(original_by_name[name])
        base_params = base.get("params", {}) if isinstance(base.get("params"), dict) else {}
        item_params = item.get("params", {}) if isinstance(item.get("params"), dict) else {}

        # Do not allow method changes
        if "method" in item_params and "method" in base_params and item_params.get("method") != base_params.get("method"):
            item_params = {k: v for k, v in item_params.items() if k != "method"}

        new_params = dict(base_params)
        for k in ALLOWED_PARAM_EDITS:
            if k not in item_params:
                continue
            if k == "top_k":
                v = _clamp_int(item_params.get(k), 2, max(2, int(n_models)))
                if v is not None:
                    new_params[k] = v
            elif k == "trim_ratio":
                v = _clamp_float(item_params.get(k), 0.0, 0.4)
                if v is not None:
                    new_params[k] = v
            elif k == "shrinkage":
                v = _clamp_float(item_params.get(k), 0.0, 0.9)
                if v is not None:
                    new_params[k] = v
            elif k == "l2":
                v = _clamp_float(item_params.get(k), 0.1, 1000.0)
                if v is not None:
                    new_params[k] = v

        base["params"] = new_params
        sanitized.append(base)
        seen.add(name)

    # Safety: if edits removed too much, keep at least 2 candidates.
    if "baseline_mean" in original_by_name and "baseline_mean" not in seen:
        sanitized.insert(0, original_by_name["baseline_mean"])

    if len(sanitized) < 2:
        return original_payload

    out: Dict[str, Any] = {"candidates": sanitized}
    if isinstance(original_payload.get("meta"), dict):
        out["meta"] = original_payload.get("meta")
    return out


DEFAULT_CANDIDATES: List[CandidateStrategy] = [
    CandidateStrategy(
        name="baseline_mean",
        type="baseline",
        description="Mean across all models for each horizon.",
        formula="y_hat(h)=mean_m pred_m(h)",
        learns_weights=False,
        constraints="none",
        risks=["sensitive to outliers"],
        validation_plan="rolling",
        params={"method": "mean"},
    ),
    CandidateStrategy(
        name="baseline_median",
        type="baseline",
        description="Median across models per horizon.",
        formula="y_hat(h)=median_m pred_m(h)",
        learns_weights=False,
        constraints="none",
        risks=["may underperform if most models biased"],
        validation_plan="rolling",
        params={"method": "median"},
    ),
    CandidateStrategy(
        name="trimmed_mean_20",
        type="baseline",
        description="Trimmed mean per horizon (robust).",
        formula="trim top/bottom then mean",
        learns_weights=False,
        constraints="trim_ratio in [0,0.4]",
        risks=["too aggressive trim if few models"],
        validation_plan="rolling",
        params={"method": "trimmed_mean", "trim_ratio": 0.2},
    ),
    CandidateStrategy(
        name="best_single_rolling",
        type="selection",
        description="Select best single model using only past windows (aggregate RMSE).",
        formula="m*=argmin_m RMSE_past(m); y_hat=pred_{m*}",
        learns_weights=False,
        constraints="anti-leakage rolling selection",
        risks=["unstable with few windows"],
        validation_plan="rolling",
        params={"method": "best_single", "selection_metric": "rmse"},
    ),
    CandidateStrategy(
        name="best_per_horizon_rolling",
        type="selection",
        description="Select best model per horizon using only past windows.",
        formula="for each h: m*(h)=argmin_m RMSE_past(m,h)",
        learns_weights=False,
        constraints="anti-leakage rolling selection",
        risks=["pointwise overfit"],
        validation_plan="rolling",
        params={"method": "best_per_horizon", "selection_metric": "rmse"},
    ),
    CandidateStrategy(
        name="topk_mean_per_horizon_k3",
        type="selection",
        description="Top-k mean per horizon (k=3) with rolling selection.",
        formula="for each h: pick top-k by past RMSE, average",
        learns_weights=False,
        constraints="anti-leakage rolling selection",
        risks=["depends on k"],
        validation_plan="rolling",
        params={"method": "topk_mean_per_horizon", "top_k": 3, "selection_metric": "rmse"},
    ),
    CandidateStrategy(
        name="inv_rmse_weights_per_horizon_k3_shrink02",
        type="weighted",
        description="Inverse-RMSE weights per horizon (top-k=3) with shrinkage.",
        formula="w_m(h)∝1/RMSE_past(m,h); y_hat(h)=Σw_m(h)pred_m(h)",
        learns_weights=True,
        constraints="w>=0,sum(w)=1; learned from past windows only",
        risks=["weight instability"],
        validation_plan="rolling",
        params={"method": "inverse_rmse_weights_per_horizon", "top_k": 3, "shrinkage": 0.2, "eps": 1e-8},
    ),
    CandidateStrategy(
        name="ridge_stacking_per_horizon_l2_10",
        type="stacking",
        description="Ridge stacking per horizon with simplex projection.",
        formula="argmin ||Xw-y||^2+λ||w||^2 then project to simplex",
        learns_weights=True,
        constraints="w>=0,sum(w)=1; fit uses past windows only",
        risks=["needs enough windows"],
        validation_plan="rolling",
        params={"method": "ridge_stacking_per_horizon", "l2": 10.0, "top_k": 5},
    ),
]


def run_deterministic_pipeline(
    candidates: Optional[List[CandidateStrategy]] = None,
    eval_cfg: Optional[EvaluationConfig] = None,
) -> Dict[str, Any]:
    """Runs deterministic evaluation on validation windows and produces final prediction.

    Requires context to already contain:
      - all_validations (windows) and predictions (final preds)

    Returns:
      dict with keys: success, best, ranking, description, result, eval
    """

    candidates = candidates or DEFAULT_CANDIDATES
    eval_cfg = eval_cfg or EvaluationConfig()

    eval_result = evaluate_all(load_validation_from_context(), candidates, eval_cfg)  # type: ignore
    best = eval_result.get("best")
    if not best:
        return {"success": False, "error": "No best candidate"}

    best_candidate = CandidateStrategy(**best["candidate"])  # reconstruct

    pred = predict_final_from_context(best_candidate, eval_cfg.rolling)

    description = {
        "best": best_candidate.to_dict(),
        "score": best.get("score"),
        "aggregate": best.get("aggregate"),
        "stability": best.get("stability"),
        "predict_debug": pred.get("debug", {}),
    }

    out = {
        "success": True,
        "best": best_candidate.to_dict(),
        "ranking": eval_result.get("ranking", []),
        "description": json.dumps(description, ensure_ascii=False),
        "result": [float(x) for x in pred["result"]],
        "eval": eval_result,
    }

    set_context("orchestrator_last_pipeline", out)
    return out


def run_llm_pipeline(
    model_id: str = "qwen3:14b",
    debug: bool = False,
    rolling_mode: str = "expanding",
    train_window: int = 5,
    require_tool_call: bool = True,
    llm_logs: bool = True,
    debate: bool = False,
    debate_auto: bool = True,
    debate_margin: float = 0.02,
) -> Dict[str, Any]:
    """Mode 2: multi-agent proposal/debate + deterministic evaluator tool decides.

        Requirements:
            - context must already contain all_validations + predictions
            - final selection is deterministic via in-code evaluation (no LLM tool-calling dependency)

    Returns same structure as run_deterministic_pipeline.
    """

    def _log(msg: str) -> None:
        if llm_logs:
            print(f"[ORCH|LLM] {msg}", flush=True)

    proposer = create_proposer_agent(model_id, debug=debug)
    skeptic = create_skeptic_agent(model_id, debug=debug)
    statistician = create_statistician_agent(model_id, debug=debug)

    llm_artifacts: Dict[str, Any] = {
        "model_id": model_id,
        "prompts": {},
        "raw": {},
        "parsed": {},
    }

    _log(f"Starting LLM pipeline | model_id={model_id} | rolling={rolling_mode} | train_window={train_window}")

    eval_config = {
        "rolling": {"mode": rolling_mode, "train_window": int(train_window)},
        "metrics": {"mape_zero": "skip", "mape_epsilon": 1e-8},
        "score": {"a_rmse": 0.4, "b_smape": 0.3, "c_mape": 0.3, "d_pocid": 0.1},
    }

    proposer_prompt = (
        "Call proposer_brief_tool() FIRST. "
        "Then select candidates by name and choose a score_preset. "
        "Return ONLY JSON per instructions. "
        "IMPORTANT: you MUST ONLY reference candidate names that appear in the tool output candidate_library; "
        "unknown names will hard-stop.\n\n"
        # f"config_json: {json.dumps(eval_config, ensure_ascii=False)}"
    )

    # Tool inputs are provided via context so the LLM doesn't need to pass parameters.
    # proposer_brief_tool expects MAPE config at top-level keys.
    set_context("config_json_for_proposer", json.dumps(eval_config.get("metrics", {}), ensure_ascii=False))
    set_context("proposer_max_candidates", 12)
    
    _log("Proposer: waiting for LLM response...")
    pr_out, pr_obj = _run_agent_with_retry(
        lambda: proposer.run(proposer_prompt).content,
        "Proposer",
        max_retries=3,
        log_func=_log,
    )
    llm_artifacts["prompts"]["proposer"] = proposer_prompt
    llm_artifacts["raw"]["proposer"] = str(pr_out)
    llm_artifacts["parsed"]["proposer"] = pr_obj
    set_context("orchestrator_llm_artifacts", llm_artifacts)

    # Read proposer brief generated by the tool (stored in context by proposer_brief_tool).
    brief = get_context("orchestrator_proposer_brief")
    if not isinstance(brief, dict):
        raise RuntimeError("Proposer tool did not populate orchestrator_proposer_brief (hard-stop)")

    library = brief.get("candidate_library")
    if not isinstance(library, dict) or not isinstance(library.get("candidates"), list):
        raise RuntimeError("Proposer brief missing candidate_library.candidates (hard-stop)")

    summary = brief.get("validation_summary") if isinstance(brief.get("validation_summary"), dict) else {}
    candidates_all: List[Dict[str, Any]] = [c for c in library.get("candidates", []) if isinstance(c, dict)]
    by_name = {str(c.get("name")): c for c in candidates_all if c.get("name")}
    universe_names = sorted(by_name.keys())

    raw_selected_names = pr_obj.get("selected_names", [])
    selected_names = raw_selected_names
    if isinstance(selected_names, str):
        selected_names = [selected_names]
    if not isinstance(selected_names, list):
        selected_names = []
    selected_names_norm = [str(x) for x in selected_names if str(x)]
    dropped_selected_names = [n for n in selected_names_norm if n not in by_name]
    selected_names = [n for n in selected_names_norm if n in by_name]
    if dropped_selected_names:
        _log(
            "Proposer selected names not in candidate_library; they will be ignored: "
            f"{dropped_selected_names}"
        )

    # Ensure at least 2 candidates (safety). Do NOT silently inject baseline_mean
    # unless we can't keep a minimal set.
    if len(selected_names) < 2:
        selected_names = [str(c.get("name")) for c in candidates_all[:4] if c.get("name")]
        selected_names = [n for n in selected_names if n in by_name]

    candidates_payload = {
        "candidates": [by_name[n] for n in selected_names if n in by_name],
        "meta": {"selected_by": "proposer", "score_preset": pr_obj.get("score_preset")},
    }

    def _candidate_names_from_payload(p: Dict[str, Any]) -> List[str]:
        if not isinstance(p, dict) or not isinstance(p.get("candidates"), list):
            return []
        return [
            str(c.get("name"))
            for c in p.get("candidates", [])
            if isinstance(c, dict) and c.get("name")
        ]

    models_available = get_context("models_available", [])
    n_models = len(models_available) if isinstance(models_available, list) and models_available else int(summary.get("n_models", 1) or 1)

    proposer_candidate_names = [str(c.get("name")) for c in candidates_payload.get("candidates", []) if isinstance(c, dict) and c.get("name")]
    proposer_actions = _validate_actions_against_universe(
        {"add_names": [], "remove_names": [], "params_overrides": pr_obj.get("params_overrides", {})},
        proposer_candidate_names,
        current_names=proposer_candidate_names,
        who="Proposer",
    )
    candidates_payload = _apply_actions_to_payload(candidates_payload, proposer_actions, universe_by_name=by_name, n_models=n_models)
    candidates_after_proposer = _candidate_names_from_payload(candidates_payload)

    score_preset = str(pr_obj.get("score_preset", "balanced"))
    if score_preset not in SCORE_PRESETS:
        score_preset = "balanced"

    proposer_force_debate = bool(pr_obj.get("force_debate", False))
    proposer_debate_margin = _clamp_float(pr_obj.get("debate_margin", debate_margin), 0.0, 0.1)
    if proposer_debate_margin is None:
        proposer_debate_margin = float(debate_margin)
    # Do not allow the Proposer to accidentally disable debate_auto by setting 0.0.
    effective_debate_margin = float(max(float(debate_margin), float(proposer_debate_margin)))

    debate_trace: Dict[str, Any] = {
        "debate_ran": False,
        "debate_trigger": "disabled",
        "debate_margin_top2": None,
        "debate_margin_threshold": float(effective_debate_margin),
        "best_pre_debate": None,
        "best_post_debate": None,
    }

    # Evaluate once pre-debate to (1) compute gating margin and
    # (2) record what would have been chosen without debate.
    pre_eval = None
    try:
        data = load_validation_from_context()
        candidates_for_eval = parse_candidates(candidates_payload.get("candidates"))
        if candidates_for_eval:
            pre_cfg = EvaluationConfig()
            pre_cfg.rolling.mode = rolling_mode
            pre_cfg.rolling.train_window = int(train_window)
            # apply score preset
            sp = SCORE_PRESETS.get(score_preset, SCORE_PRESETS["balanced"])
            pre_cfg.score.a_rmse = float(sp["a_rmse"])
            pre_cfg.score.b_smape = float(sp["b_smape"])
            pre_cfg.score.c_mape = float(sp["c_mape"])
            pre_cfg.score.d_pocid = float(sp["d_pocid"])
            pre_eval = evaluate_all(data, candidates_for_eval, pre_cfg)
            pre_best = pre_eval.get("best") if isinstance(pre_eval, dict) else None
            if isinstance(pre_best, dict):
                debate_trace["best_pre_debate"] = pre_best.get("candidate")
            ranking = pre_eval.get("ranking", []) if isinstance(pre_eval, dict) else []
            if isinstance(ranking, list) and len(ranking) >= 2:
                s1 = float(ranking[0].get("score"))
                s2 = float(ranking[1].get("score"))
                debate_trace["debate_margin_top2"] = float(s2 - s1)
    except Exception as e:
        _log(f"Pre-debate eval skipped due to error: {e}")

    # Gating: debate if forced OR proposer requested OR ambiguous margin.
    should_debate = bool(debate) or proposer_force_debate
    if bool(debate):
        debate_trace["debate_trigger"] = "forced"
    elif proposer_force_debate:
        debate_trace["debate_trigger"] = "proposer_forced"
    if not should_debate and debate_auto:
        m = debate_trace.get("debate_margin_top2")
        if isinstance(m, (int, float)) and m == m and m < float(effective_debate_margin):
            should_debate = True
            debate_trace["debate_trigger"] = "auto_margin"
            _log(f"Debate auto-triggered: small margin top2 ({float(m):.4f})")

    if should_debate:
        _log("Debate enabled: running Skeptic + Statistician (tool-grounded)")
        debate_trace["debate_ran"] = True
        # Provide tool inputs via context so the LLM doesn't need to pass parameters.
        config_json = json.dumps(
            {
                "rolling": {"mode": rolling_mode, "train_window": int(train_window)},
                "metrics": {"mape_zero": "skip", "mape_epsilon": 1e-8},
                "score": SCORE_PRESETS.get(score_preset, SCORE_PRESETS["balanced"]),
            },
            ensure_ascii=False,
        )
        candidates_json = json.dumps(candidates_payload, ensure_ascii=False)
        universe_json = json.dumps({"candidates": candidates_all, "meta": {"source": "proposer_brief_universe"}}, ensure_ascii=False)

        set_context("config_json_for_debate", config_json)
        set_context("candidates_json_for_debate", candidates_json)
        set_context("candidate_universe_json_for_debate", universe_json)
        set_context("debate_top_n", 5)

        # Give agents a non-blind view of the candidate universe (names only).
        current_names = sorted(
            {
                str(c.get("name"))
                for c in candidates_payload.get("candidates", [])
                if isinstance(c, dict) and c.get("name")
            }
        )
        universe_names_hint = json.dumps(universe_names, ensure_ascii=False)
        current_names_hint = json.dumps(current_names, ensure_ascii=False)

        skeptic_prompt = (
            "Chame build_debate_packet_tool() PRIMEIRO (inputs via context). "
            "Depois retorne APENAS JSON (sem markdown) com add_names, remove_names, params_overrides, rationale, changes, when_good.\n"
            "IMPORTANT: you MUST ONLY use candidate names from valid_candidate_names; unknown names hard-stop.\n"
            "You may only remove/override candidates that are in current_candidate_names (or candidates you are adding).\n"
            f"valid_candidate_names: {universe_names_hint}\n"
            f"current_candidate_names: {current_names_hint}"
        )

        _log("Skeptic: waiting for LLM response...")
        sk_out, sk_obj = _run_agent_with_retry(
            lambda: skeptic.run(skeptic_prompt).content,
            "Skeptic",
            max_retries=3,
            log_func=_log,
        )
        llm_artifacts["prompts"]["skeptic"] = skeptic_prompt
        llm_artifacts["raw"]["skeptic"] = str(sk_out)
        llm_artifacts["parsed"]["skeptic"] = sk_obj
        set_context("orchestrator_llm_artifacts", llm_artifacts)
        sk_current_names = [
            str(c.get("name"))
            for c in candidates_payload.get("candidates", [])
            if isinstance(c, dict) and c.get("name")
        ]
        sk_actions = _validate_actions_against_universe(sk_obj, universe_names, current_names=sk_current_names, who="Skeptic")
        candidates_payload = _apply_actions_to_payload(candidates_payload, sk_actions, universe_by_name=by_name, n_models=n_models)
        candidates_after_skeptic = _candidate_names_from_payload(candidates_payload)

        # Refresh tool inputs for the Statistician after Skeptic edits.
        set_context("candidates_json_for_debate", json.dumps(candidates_payload, ensure_ascii=False))

        current_names = sorted(
            {
                str(c.get("name"))
                for c in candidates_payload.get("candidates", [])
                if isinstance(c, dict) and c.get("name")
            }
        )
        current_names_hint = json.dumps(current_names, ensure_ascii=False)

        statistician_prompt = (
            "Chame build_debate_packet_tool() PRIMEIRO (inputs via context). "
            "Depois retorne APENAS JSON (sem markdown) com add_names, remove_names, params_overrides, rationale, changes, when_good.\n"
            "IMPORTANT: you MUST ONLY use candidate names from valid_candidate_names; unknown names hard-stop.\n"
            "You may only remove/override candidates that are in current_candidate_names (or candidates you are adding).\n"
            f"valid_candidate_names: {universe_names_hint}\n"
            f"current_candidate_names: {current_names_hint}"
        )

        _log("Statistician: waiting for LLM response...")
        st_out, st_obj = _run_agent_with_retry(
            lambda: statistician.run(statistician_prompt).content,
            "Statistician",
            max_retries=3,
            log_func=_log,
        )
        llm_artifacts["prompts"]["statistician"] = statistician_prompt
        llm_artifacts["raw"]["statistician"] = str(st_out)
        llm_artifacts["parsed"]["statistician"] = st_obj
        set_context("orchestrator_llm_artifacts", llm_artifacts)
        st_current_names = [
            str(c.get("name"))
            for c in candidates_payload.get("candidates", [])
            if isinstance(c, dict) and c.get("name")
        ]
        st_actions = _validate_actions_against_universe(st_obj, universe_names, current_names=st_current_names, who="Statistician")
        candidates_payload = _apply_actions_to_payload(candidates_payload, st_actions, universe_by_name=by_name, n_models=n_models)
        candidates_after_statistician = _candidate_names_from_payload(candidates_payload)
    else:
        _log("Debate disabled: skipping Skeptic + Statistician (lower randomness)")
        candidates_after_skeptic = None
        candidates_after_statistician = None

    # Ensure structure is {"candidates": [...]}
    if isinstance(candidates_payload, list):
        candidates_payload = {"candidates": candidates_payload}
    if not isinstance(candidates_payload, dict) or "candidates" not in candidates_payload:
        raise RuntimeError("Candidates payload malformed after proposal/debate")

    if not isinstance(candidates_payload.get("candidates"), list) or len(candidates_payload.get("candidates")) == 0:
        raise RuntimeError("No candidates provided after proposal/debate")

    n_candidates = len(candidates_payload.get("candidates", [])) if isinstance(candidates_payload, dict) else 0
    _log(f"Candidates ready: {n_candidates} candidate(s)")

    # If debate ran, record the best candidate under the revised set BEFORE evaluation.
    if debate_trace.get("debate_ran"):
        try:
            data = load_validation_from_context()
            post_candidates = parse_candidates(candidates_payload.get("candidates"))
            if post_candidates:
                post_cfg = EvaluationConfig()
                post_cfg.rolling.mode = rolling_mode
                post_cfg.rolling.train_window = int(train_window)
                post_eval = evaluate_all(data, post_candidates, post_cfg)
                post_best = post_eval.get("best") if isinstance(post_eval, dict) else None
                if isinstance(post_best, dict):
                    debate_trace["best_post_debate"] = post_best.get("candidate")
        except Exception as e:
            _log(f"Post-debate eval skipped due to error: {e}")

    # Deterministic evaluation (anti-leakage) executed in-code.
    eval_cfg = EvaluationConfig()
    eval_cfg.rolling.mode = rolling_mode
    eval_cfg.rolling.train_window = int(train_window)
    eval_cfg.metrics.mape_zero = "skip"
    eval_cfg.metrics.mape_epsilon = 1e-8
    sp = SCORE_PRESETS.get(score_preset, SCORE_PRESETS["balanced"])
    eval_cfg.score.a_rmse = float(sp["a_rmse"])
    eval_cfg.score.b_smape = float(sp["b_smape"])
    eval_cfg.score.c_mape = float(sp["c_mape"])
    eval_cfg.score.d_pocid = float(sp["d_pocid"])

    parsed_candidates = parse_candidates(candidates_payload.get("candidates"))
    if not parsed_candidates:
        raise RuntimeError("No valid candidates parsed after proposal/debate (hard-stop)")

    eval_result: Dict[str, Any] = evaluate_all(load_validation_from_context(), parsed_candidates, eval_cfg)
    set_context("orchestrator_last_eval", eval_result)
    tools_called = get_context("tools_called", [])
    if not isinstance(tools_called, list):
        tools_called = []
    tools_called.append("evaluate_strategies_tool")
    set_context("tools_called", tools_called)

    if not isinstance(eval_result, dict) or not eval_result.get("best"):
        raise RuntimeError("Deterministic evaluation produced no best candidate (hard-stop)")

    _log("Evaluation result ready")

    best = eval_result["best"]
    best_candidate = CandidateStrategy(**best["candidate"])  # reconstruct
    _log(f"Best strategy: {best_candidate.name}")
    pred = predict_final_from_context(best_candidate, RollingConfig(mode=rolling_mode, train_window=int(train_window)))

    _log("Final prediction generated from context['predictions']")

    description = {
        "mode": "llm",
        "candidates_trace": {
            "after_proposer": candidates_after_proposer,
            "after_skeptic": candidates_after_skeptic,
            "after_statistician": candidates_after_statistician,
            "dropped_selected_names": dropped_selected_names,
        },
        "tool_validation": {
            "tools_called": tools_called,
            "require_tool_call": bool(require_tool_call),
            "tool_missing": bool(
                require_tool_call
                and (
                    "proposer_brief_tool" not in tools_called
                    or (
                        bool(debate_trace.get("debate_ran"))
                        and "build_debate_packet_tool" not in tools_called
                    )
                )
            ),
        },
        "debate": debate_trace,
        "score_preset": score_preset,
        "best": best_candidate.to_dict(),
        "score": best.get("score"),
        "aggregate": best.get("aggregate"),
        "stability": best.get("stability"),
        "predict_debug": pred.get("debug", {}),
        "llm": {
            "proposer": llm_artifacts.get("parsed", {}).get("proposer"),
            "skeptic": llm_artifacts.get("parsed", {}).get("skeptic") if debate_trace.get("debate_ran") else None,
            "statistician": llm_artifacts.get("parsed", {}).get("statistician") if debate_trace.get("debate_ran") else None,
        },
    }

    def _short_text(x: Any, max_len: int = 600) -> str:
        s = "" if x is None else str(x)
        s = " ".join(s.split())
        return s[:max_len]

    # Human-friendly short explanations for CSV.
    explanations: Dict[str, Any] = {
        "before": debate_trace.get("best_pre_debate", {}).get("name") if isinstance(debate_trace.get("best_pre_debate"), dict) else None,
        "after": debate_trace.get("best_post_debate", {}).get("name") if isinstance(debate_trace.get("best_post_debate"), dict) else None,
        "debate_trigger": debate_trace.get("debate_trigger"),
        "debate_margin_top2": debate_trace.get("debate_margin_top2"),
        "skeptic_rationale": None,
        "statistician_rationale": None,
        "orchestrator_reasoning": None,
        "orchestrator_when_good": None,
        "orchestrator_debate_notes": None,
    }
    sk_parsed = llm_artifacts.get("parsed", {}).get("skeptic")
    if isinstance(sk_parsed, dict):
        explanations["skeptic_rationale"] = _short_text(sk_parsed.get("rationale"))
        explanations["skeptic_when_good"] = _short_text(sk_parsed.get("when_good"))
    st_parsed = llm_artifacts.get("parsed", {}).get("statistician")
    if isinstance(st_parsed, dict):
        explanations["statistician_rationale"] = _short_text(st_parsed.get("rationale"))
        explanations["statistician_when_good"] = _short_text(st_parsed.get("when_good"))
    # Orchestrator reasoning is not generated by LLM in this pipeline variant.

    out = {
        "success": True,
        "best": best_candidate.to_dict(),
        "ranking": eval_result.get("ranking", []),
        "description": json.dumps(description, ensure_ascii=False),
        "result": [float(x) for x in pred["result"]],
        "eval": eval_result,
        "debate": debate_trace,
        "explanations": explanations,
        "llm_artifacts": llm_artifacts,
    }

    set_context("orchestrator_last_pipeline", out)
    set_context("orchestrator_last_candidates", candidates_payload)
    set_context("orchestrator_debate_trace", debate_trace)
    _log("LLM pipeline completed")
    return out


# Local import to avoid circular import at module import time
from orchestrator.data_contract import load_validation_from_context  # noqa: E402
