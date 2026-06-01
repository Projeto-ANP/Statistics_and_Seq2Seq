from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from orchestrator_langchain.context import get_context, set_context

from orchestrator.evaluator import EvaluationConfig, evaluate_all, evaluate_candidate
from orchestrator.final_predictor import predict_final_from_context
from orchestrator.schemas import CandidateStrategy
from orchestrator.strategies import RollingConfig
from orchestrator.utils import extract_json_object, strip_think_blocks as _strip_think_blocks_util
from orchestrator.schemas import parse_candidates
from orchestrator.diagnostics import diebold_mariano
from orchestrator.data_contract import load_validation_from_context
from orchestrator.tools import (
    SCORE_PRESETS,
    proposer_brief_tool as _proposer_brief_tool,
    build_fold_cot_context_tool as _build_fold_cot_context_tool,
    build_debate_packet_tool as _build_debate_packet_tool,
    resolve_unknown_candidate as _resolve_unknown_candidate,
    series_analysis_brief_tool as _series_analysis_brief_tool,
    model_critic_brief_tool as _model_critic_brief_tool,
    combination_architect_brief_tool as _combination_architect_brief_tool,
    _per_model_diagnostics,
)
from orchestrator.agents import (
    create_pattern_analyst_agent,
    create_proposer_agent,
    create_skeptic_agent,
    create_statistician_agent,
    create_series_annotator_agent,
    create_strategy_selector_agent,
    create_series_analyst_agent,
    create_model_critic_agent,
    create_combination_architect_agent,
)


ALLOWED_PARAM_EDITS = {"top_k", "trim_ratio", "shrinkage", "l2", "period"}


def _strip_think_blocks(text: str) -> str:
    return _strip_think_blocks_util(text)


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
            raw_str = str(output)
            log_func(f"{agent_name} raw (first 2000 chars): {raw_str[:2000]}")

            cleaned = _strip_think_blocks(raw_str)
            # Explicit empty-output detection: when the model fits its whole budget inside
            # <think>...</think> (typical when num_predict is too small or context overflows),
            # the cleaned text is empty and `extract_json_object` would return None silently.
            # Surfacing this distinctly helps diagnose num_ctx / num_predict tuning issues.
            if not cleaned.strip():
                log_func(
                    f"{agent_name}: EMPTY content after stripping <think> blocks "
                    f"(raw_len={len(raw_str)}, elapsed={elapsed:.1f}s). "
                    "Likely num_predict exhausted inside thinking or num_ctx overflow. "
                    "Check ChatOllama num_ctx/num_predict in agents.py."
                )
                if attempt < max_retries:
                    continue
                raise RuntimeError(
                    f"{agent_name} returned empty content after {max_retries} attempts (hard-stop). "
                    f"Tune num_ctx/num_predict in LangchainAgent."
                )

            parsed_obj = extract_json_object(cleaned)
            if parsed_obj is None or not isinstance(parsed_obj, dict):
                if attempt < max_retries:
                    log_func(f"{agent_name}: invalid JSON (attempt {attempt}, retrying...)")
                    continue
                else:
                    raise RuntimeError(
                        f"{agent_name} did not return valid JSON after {max_retries} attempts (hard-stop). "
                        f"Raw (first 2000 chars): {raw_str[:2000]}"
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


import re
import difflib

def _resolve_candidate_name(name: str, valid_set: set) -> str:
    """Auto-corrects candidate names by fixing float suffixes or minor typos."""
    if name in valid_set:
        return name
        
    # Strip trailing zeros from decimals to match things like "0.2" with "0.20"
    def strip_trailing_zeros(s: str) -> str:
        return re.sub(r'(\.\d*?[1-9])0+(?=[^\d]|$)|(\.)0+(?=[^\d]|$)', r'\1', s)
        
    normalized_valid_map = {strip_trailing_zeros(v): v for v in valid_set}
    stripped_target = strip_trailing_zeros(name)
    
    if stripped_target in normalized_valid_map:
        return normalized_valid_map[stripped_target]
        
    # Fallback to fuzzy string matching
    matches = difflib.get_close_matches(name, valid_set, n=1, cutoff=0.85)
    if matches:
        return matches[0]
        
    return name


def _validate_actions_against_universe(
    actions: Dict[str, Any],
    universe_names: List[str],
    current_names: Optional[List[str]],
    who: str,
    by_name_registry: Optional[Dict[str, Any]] = None,
    n_models: int = 2,
    n_windows: int = 3,
) -> Dict[str, Any]:
    """Validate that LLM actions only reference real candidates.

    Unknown add_names are first tried against resolve_unknown_candidate (pattern
    parser) before hard-stopping, so the LLM can propose variants with slightly
    different naming conventions without triggering a crash.
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
    add_names_norm = [_resolve_candidate_name(n, valid_set) for n in add_names_norm]

    # Try pattern-based resolution for any name still not in the universe.
    resolved_add_names: List[str] = []
    for _an in add_names_norm:
        if _an in valid_set:
            resolved_add_names.append(_an)
            continue
        _rc = _resolve_unknown_candidate(_an, n_models, n_windows)
        if _rc is not None:
            _rn = str(_rc["name"])
            if _rn not in valid_set:
                valid_set.add(_rn)
                universe_names.append(_rn)
                if by_name_registry is not None and _rn not in by_name_registry:
                    by_name_registry[_rn] = _rc
            _log(f"[ORCH|LLM] {who} add_names: resolved '{_an}' → '{_rn}'")
            resolved_add_names.append(_rn)
        else:
            resolved_add_names.append(_an)  # keep for hard-stop check below
    add_names_norm = resolved_add_names

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
    remove_names_norm = [_resolve_candidate_name(n, current_set) for n in remove_names_norm]

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

    overrides_raw = {_resolve_candidate_name(str(k), valid_set): v for k, v in overrides_raw.items()}

    allowed_override = set(current_set) | set(add_names_norm)
    override_keys = [str(k) for k in overrides_raw.keys()]
    cand_override_keys = [k for k in override_keys if k in allowed_override]
    knob_override_keys = [k for k in override_keys if k in ALLOWED_PARAM_EDITS]
    unknown_override_keys = [k for k in override_keys if k not in allowed_override and k not in ALLOWED_PARAM_EDITS]

    # Auto-promote: if an unknown override key is a valid universe candidate, the LLM simply
    # forgot to add it to add_names. Silently promote it instead of hard-stopping.
    auto_promoted = [k for k in unknown_override_keys if k in valid_set]
    still_unknown = [k for k in unknown_override_keys if k not in valid_set]

    if auto_promoted:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"{who}: auto-promoting candidates found in params_overrides but missing from "
            f"add_names: {auto_promoted}. Adding them to add_names automatically."
        )
        add_names_norm = add_names_norm + auto_promoted
        allowed_override = set(current_set) | set(add_names_norm)
        cand_override_keys = [k for k in override_keys if k in allowed_override]

    if still_unknown:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"{who}: params_overrides references names not found anywhere in the universe "
            f"(likely hallucinated by the LLM): {still_unknown}. "
            f"These overrides will be silently dropped."
        )
        # Drop the unknown keys so they don't pollute the override dict
        overrides_raw = {k: v for k, v in overrides_raw.items() if str(k) not in still_unknown}
        override_keys = [k for k in override_keys if k not in still_unknown]
        cand_override_keys = [k for k in override_keys if k in allowed_override]

    # Normalize overrides: allow either per-candidate overrides or a flat "knob override" that
    # applies the same params to every candidate in current/add set. This prevents false hard-stops
    # when agents return {{"trim_ratio": 0.1}} instead of {{"cand": {"trim_ratio": 0.1}}}.
    overrides: Dict[str, Dict[str, Any]] = {}

    # Candidate-specific overrides (classic path)
    for cand in cand_override_keys:
        ov = overrides_raw.get(cand, {})
        if not isinstance(ov, dict):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"{who}.params_overrides['{cand}'] is not a dict (got {type(ov).__name__}: {ov}). "
                f"Gracefully ignoring this specific override to prevent hard-stop."
            )
            continue
        overrides[cand] = dict(ov)

    # Global knob overrides (apply to all current/add candidates)
    if knob_override_keys:
        global_override = {k: overrides_raw[k] for k in knob_override_keys}
        for cand in allowed_override or current_set:
            if cand not in overrides:
                overrides[cand] = {}
            overrides[cand].update(global_override)

    # Validate allowed keys for each override payload
    valid_overrides = {}
    for cand, ov in overrides.items():
        if not isinstance(ov, dict):
            continue
        
        bad_keys = [k for k in ov.keys() if str(k) not in ALLOWED_PARAM_EDITS and str(k) != "method"]
        if bad_keys:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"{who} used unsupported override keys for '{cand}': {bad_keys}. "
                f"Allowed: {sorted(ALLOWED_PARAM_EDITS)}. Ignoring bad keys instead of hard-stopping."
            )
            ov = {k: v for k, v in ov.items() if k not in bad_keys}
            
        if ov:
            valid_overrides[cand] = ov

    # Return normalized copy
    return {
        "add_names": add_names_norm,
        "remove_names": remove_names_norm,
        "params_overrides": valid_overrides,
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
            elif k == "period":
                vv = _clamp_int(v, 2, 24)
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
            elif k == "period":
                v = _clamp_int(item_params.get(k), 2, 24)
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

import orchestrator.utils as _utils
def run_llm_pipeline(
    # model_id: str = "qwen3:14b",
    proposer_model: _utils.ModelConfig,
    skeptic_model: _utils.ModelConfig,
    statistician_model: _utils.ModelConfig,
    pattern_analyst_model: _utils.ModelConfig,
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
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts} ORCH|LLM] {msg}", flush=True)

    proposer = create_proposer_agent(proposer_model.model, debug=debug)
    skeptic = create_skeptic_agent(skeptic_model.model, debug=debug)
    statistician = create_statistician_agent(statistician_model.model, debug=debug)
    pattern_analyst = create_pattern_analyst_agent(pattern_analyst_model.model, debug=debug)

    llm_artifacts: Dict[str, Any] = {
        "proposer": f"{proposer_model.model} + {proposer_model.temperature}",
        "skeptic": f"{skeptic_model.model} + {skeptic_model.temperature}",
        "statistician": f"{statistician_model.model} + {statistician_model.temperature}",
        "pattern_analyst": f"{pattern_analyst_model.model} + {pattern_analyst_model.temperature}",
        "prompts": {},
        "raw": {},
        "parsed": {},
    }

    _log(f"Starting LLM pipeline | proposer={proposer_model.model} | skeptic={skeptic_model.model} | statistician={statistician_model.model} | pattern_analyst={pattern_analyst_model.model} | rolling={rolling_mode} | train_window={train_window}")

    eval_config = {
        "rolling": {"mode": rolling_mode, "train_window": int(train_window)},
        "metrics": {"mape_zero": "skip", "mape_epsilon": 1e-8},
        "score": {"a_rmse": 0.2, "b_smape": 0.4, "c_mape": 0.3, "d_pocid": 0.1},
    }

    # ── Step 0: PatternAnalyst — fold decomposition CoT (non-fatal) ───────────
    _log("PatternAnalyst: analyzing validation folds (trend/seasonality)...")
    pattern_analyst_prompt = (
        "Call build_fold_cot_context_tool() to analyze the validation folds. "
        "Then return ONLY JSON with your insights."
    )
    try:
        pa_out, pa_obj = _run_agent_with_retry(
            lambda: pattern_analyst.run(pattern_analyst_prompt).content,
            "PatternAnalyst",
            max_retries=2,
            log_func=_log,
        )
        set_context("pattern_analyst_insights", pa_obj)
        llm_artifacts["raw"]["pattern_analyst"] = str(pa_out)
        llm_artifacts["parsed"]["pattern_analyst"] = pa_obj
        _log(f"PatternAnalyst: trend_champion={pa_obj.get('trend_champion')} | seas_champion={pa_obj.get('seasonality_champion')} | hint={pa_obj.get('recommended_method_hint')}")
    except Exception as e:
        _log(f"PatternAnalyst: failed (non-fatal, continuing) — {e}")
        set_context("pattern_analyst_insights", {})

    # Ensure build_fold_cot_context_tool was actually called (writes pattern_analyst_cot_context).
    # If the LLM skipped the tool, invoke it programmatically so the Proposer brief has fold data.
    if not get_context("pattern_analyst_cot_context"):
        _log("PatternAnalyst did not call build_fold_cot_context_tool; invoking it automatically...")
        try:
            _build_fold_cot_context_tool.entrypoint()
            _log("Fallback build_fold_cot_context_tool succeeded.")
        except Exception as _fcot_err:
            _log(f"Fallback build_fold_cot_context_tool failed (non-fatal): {_fcot_err}")

    set_context("orchestrator_llm_artifacts", llm_artifacts)
    # ─────────────────────────────────────────────────────────────────────────

    pa_insights = get_context("pattern_analyst_insights", {})
    pattern_hint_text = ""
    if isinstance(pa_insights, dict) and pa_insights:
        trend_champ = pa_insights.get("trend_champion", "")
        seas_champ = pa_insights.get("seasonality_champion", "")
        method_hint = pa_insights.get("recommended_method_hint", "")
        narrative = pa_insights.get("cot_narrative", "")
        pattern_hint_text = (
            f"\nPATTERN ANALYST INSIGHTS: trend_champion={trend_champ!r}, "
            f"seasonality_champion={seas_champ!r}, recommended_method={method_hint!r}. "
            f"Narrative: {narrative}"
        )

    proposer_prompt = (
        "Call proposer_brief() FIRST. "
        "The tool output includes pattern_analyst_insights — use trend_champion, seasonality_champion, "
        "and insights.recommended_method_hint to bias your candidate selection toward non-mean methods. "
        "MUST propose at least 3 candidates including at least 1 non-baseline type. "
        "Return ONLY JSON per instructions. "
        "IMPORTANT: you MUST ONLY reference candidate names that appear in the tool output candidate_library; "
        "unknown names will hard-stop."
        + pattern_hint_text
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
        # LLM skipped the tool call — invoke it programmatically as a recovery step.
        _log("Proposer did not call proposer_brief_tool; invoking it automatically as fallback...")
        try:
            _proposer_brief_tool.entrypoint()
            brief = get_context("orchestrator_proposer_brief")
        except Exception as _pbt_err:
            _log(f"Fallback proposer_brief_tool call failed: {_pbt_err}")
            brief = None
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

    # ── Resolve unknown names before dropping them ────────────────────────────
    # Instead of hard-ignoring LLM-invented names, try to parse them via regex
    # and register dynamically into by_name / universe_names so evaluation can
    # proceed.  Only truly unrecognisable names are silently dropped.
    models_available_early = get_context("models_available", [])
    _nm_early = len(models_available_early) if isinstance(models_available_early, list) and models_available_early else int(summary.get("n_models", 2) or 2)
    _nw_early = int(summary.get("n_windows", 3) or 3)
    _resolved_names: List[str] = []
    _truly_dropped: List[str] = []
    for _unk in selected_names_norm:
        if _unk in by_name:
            _resolved_names.append(_unk)
            continue
        _resolved = _resolve_unknown_candidate(_unk, _nm_early, _nw_early)
        if _resolved is not None:
            _rname = str(_resolved["name"])
            if _rname not in by_name:
                by_name[_rname] = _resolved
                candidates_all.append(_resolved)
                if _rname not in universe_names:
                    universe_names.append(_rname)
            _log(f"[ORCH|LLM] Resolved unknown Proposer name '{_unk}' → '{_rname}'")
            _resolved_names.append(_rname)
        else:
            _truly_dropped.append(_unk)
    if _truly_dropped:
        _log(
            "Proposer selected names not in candidate_library and could not be resolved; "
            f"they will be ignored: {_truly_dropped}"
        )
    selected_names = _resolved_names

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

    # Pre-filter params_overrides: try to resolve unknown keys via the same resolver
    # before dropping them.  This lets the LLM say params_overrides: {"trimmed_mean_tr0.2": {...}}
    # and have it remapped to the canonical name transparently.
    _raw_overrides: Dict[str, Any] = pr_obj.get("params_overrides") or {}
    _remapped_overrides: Dict[str, Any] = {}
    _dropped_override_keys: List[str] = []
    for _ok, _ov in _raw_overrides.items():
        if str(_ok) in by_name or str(_ok) in ALLOWED_PARAM_EDITS:
            _remapped_overrides[str(_ok)] = _ov
        else:
            _resolved_ov = _resolve_unknown_candidate(str(_ok), _nm_early, _nw_early)
            if _resolved_ov is not None:
                _rk = str(_resolved_ov["name"])
                _remapped_overrides[_rk] = _ov
                _log(f"[ORCH|LLM] Proposer params_overrides key '{_ok}' remapped → '{_rk}'")
            else:
                _dropped_override_keys.append(str(_ok))
    if _dropped_override_keys:
        _log(
            f"[ORCH|LLM] Proposer params_overrides references unresolvable names "
            f"(will be ignored): {_dropped_override_keys}"
        )
    _raw_overrides = _remapped_overrides

    proposer_actions = _validate_actions_against_universe(
        {"add_names": [], "remove_names": [], "params_overrides": _raw_overrides},
        universe_names,  # full library so auto-promotion works for real candidates
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
        "statistical_tie_break": None,
        "best_pre_debate": None,
        "best_post_debate": None,
    }

    # Evaluate once pre-debate to (1) compute gating margin and
    # (2) record what would have been chosen without debate.
    pre_eval = None
    try:
        from orchestrator.diagnostics import tie_break_analysis as _tie_break_analysis

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

            # A2 — Statistical tie-break (Diebold-Mariano + paired bootstrap).
            details_pre = pre_eval.get("details", []) if isinstance(pre_eval, dict) else []
            pw_scores: Dict[str, Any] = {}
            pw_errors: Dict[str, Any] = {}
            for d in details_pre:
                if not isinstance(d, dict):
                    continue
                cand = d.get("candidate", {}) if isinstance(d.get("candidate"), dict) else {}
                name = str(cand.get("name", ""))
                if not name:
                    continue
                pws = d.get("per_window_scores")
                if isinstance(pws, list) and pws:
                    pw_scores[name] = pws
                rfl = d.get("residuals_flat")
                if isinstance(rfl, list) and rfl:
                    pw_errors[name] = rfl
            try:
                tb = _tie_break_analysis(
                    top_ranking=ranking[:2],
                    per_window_scores=pw_scores,
                    per_window_errors=pw_errors,
                    alpha=0.10,
                )
                debate_trace["statistical_tie_break"] = tb
            except Exception as e:
                _log(f"Tie-break skipped: {e}")
    except Exception as e:
        _log(f"Pre-debate eval skipped due to error: {e}")

    # Gating: debate if forced OR proposer requested OR statistically tied OR ambiguous margin.
    should_debate = bool(debate) or proposer_force_debate
    if bool(debate):
        debate_trace["debate_trigger"] = "forced"
    elif proposer_force_debate:
        debate_trace["debate_trigger"] = "proposer_forced"

    # A2 — primary automatic trigger: statistical tie between top-1 and top-2.
    tie_info = debate_trace.get("statistical_tie_break")
    statistically_tied = bool(
        isinstance(tie_info, dict)
        and tie_info.get("available")
        and tie_info.get("statistically_tied")
    )
    if not should_debate and debate_auto and statistically_tied:
        should_debate = True
        debate_trace["debate_trigger"] = "auto_statistical_tie"
        _log("Debate auto-triggered: statistical tie (DM + paired bootstrap cannot separate top-1 vs top-2)")

    # Fallback: narrow score margin (kept for back-compat when tie-break is unavailable).
    if not should_debate and debate_auto:
        m = debate_trace.get("debate_margin_top2")
        if isinstance(m, (int, float)) and m == m and m < float(effective_debate_margin):
            should_debate = True
            debate_trace["debate_trigger"] = "auto_margin"
            _log(f"Debate auto-triggered: small margin top2 ({float(m):.4f})")

    if should_debate:
        _log("Debate enabled: running 2-round Skeptic↔Statistician (Du et al. 2023 style)")
        debate_trace["debate_ran"] = True
        debate_trace["debate_rounds"] = 2
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

        current_names_r1 = sorted(
            {
                str(c.get("name"))
                for c in candidates_payload.get("candidates", [])
                if isinstance(c, dict) and c.get("name")
            }
        )
        universe_names_hint = json.dumps(universe_names, ensure_ascii=False)
        current_names_hint_r1 = json.dumps(current_names_r1, ensure_ascii=False)

        # ── Pre-invoke deterministic packet once so both agents see identical numbers.
        try:
            _build_debate_packet_tool()
            _log("Pre-invoked build_debate_packet_tool for Round 1.")
        except Exception as _dpt_err:
            _log(f"Pre-invoke debate_packet (R1) failed (non-fatal): {_dpt_err}")

        def _round_prompt(role: str, round_num: int, peer_json: Optional[str], peer_role: Optional[str]) -> str:
            header = (
                "Chame build_debate_packet_tool() PRIMEIRO (inputs via context). "
                "Depois retorne APENAS JSON (sem markdown) com add_names, remove_names, params_overrides, rationale, changes, when_good.\n"
                "IMPORTANT: you MUST ONLY use candidate names from valid_candidate_names; unknown names hard-stop.\n"
                "You may only remove/override candidates that are in current_candidate_names (or candidates you are adding)."
            )
            body = (
                f"\nvalid_candidate_names: {universe_names_hint}\n"
                f"current_candidate_names: {current_names_hint_r1}\n"
                f"debate_round: {round_num}"
            )
            if peer_json is not None and peer_role is not None:
                body += (
                    f"\n{peer_role}_round1_actions: {peer_json}\n"
                    "Rodada 2: leia as acoes do par (acima), identifique pontos de concordancia/discordancia "
                    "e responda com JSON final revisado. Se concordar com o par, mantenha/apoie; "
                    "se discordar, explique em `rationale` e proponha um plano alternativo dentro dos knobs permitidos."
                )
            return header + body

        # ── Round 1 ─────────────────────────────────────────────────────────
        # Both agents respond independently (blind to the peer's output).
        skeptic_prompt_r1 = _round_prompt("Skeptic", 1, peer_json=None, peer_role=None)
        statistician_prompt_r1 = _round_prompt("Statistician", 1, peer_json=None, peer_role=None)

        _log("Round 1 — Skeptic: waiting for LLM response...")
        sk_out_r1, sk_obj_r1 = _run_agent_with_retry(
            lambda: skeptic.run(skeptic_prompt_r1).content,
            "Skeptic-R1",
            max_retries=3,
            log_func=_log,
        )
        llm_artifacts["prompts"]["skeptic_r1"] = skeptic_prompt_r1
        llm_artifacts["raw"]["skeptic_r1"] = str(sk_out_r1)
        llm_artifacts["parsed"]["skeptic_r1"] = sk_obj_r1

        _log("Round 1 — Statistician: waiting for LLM response...")
        st_out_r1, st_obj_r1 = _run_agent_with_retry(
            lambda: statistician.run(statistician_prompt_r1).content,
            "Statistician-R1",
            max_retries=3,
            log_func=_log,
        )
        llm_artifacts["prompts"]["statistician_r1"] = statistician_prompt_r1
        llm_artifacts["raw"]["statistician_r1"] = str(st_out_r1)
        llm_artifacts["parsed"]["statistician_r1"] = st_obj_r1
        set_context("orchestrator_llm_artifacts", llm_artifacts)

        # ── Round 2 ─────────────────────────────────────────────────────────
        # Each agent sees the peer's Round 1 JSON and can revise.
        def _compact_peer(obj: Any) -> str:
            if not isinstance(obj, dict):
                return "{}"
            keep = {
                "add_names": obj.get("add_names"),
                "remove_names": obj.get("remove_names"),
                "params_overrides": obj.get("params_overrides"),
                "rationale": obj.get("rationale"),
            }
            return json.dumps(keep, ensure_ascii=False, default=str)

        peer_stat_r1 = _compact_peer(st_obj_r1)
        peer_sk_r1 = _compact_peer(sk_obj_r1)

        skeptic_prompt_r2 = _round_prompt("Skeptic", 2, peer_json=peer_stat_r1, peer_role="Statistician")
        statistician_prompt_r2 = _round_prompt("Statistician", 2, peer_json=peer_sk_r1, peer_role="Skeptic")

        _log("Round 2 — Skeptic: revising with peer visibility...")
        sk_out_r2, sk_obj_r2 = _run_agent_with_retry(
            lambda: skeptic.run(skeptic_prompt_r2).content,
            "Skeptic-R2",
            max_retries=3,
            log_func=_log,
        )
        llm_artifacts["prompts"]["skeptic_r2"] = skeptic_prompt_r2
        llm_artifacts["raw"]["skeptic_r2"] = str(sk_out_r2)
        llm_artifacts["parsed"]["skeptic_r2"] = sk_obj_r2

        _log("Round 2 — Statistician: revising with peer visibility...")
        st_out_r2, st_obj_r2 = _run_agent_with_retry(
            lambda: statistician.run(statistician_prompt_r2).content,
            "Statistician-R2",
            max_retries=3,
            log_func=_log,
        )
        llm_artifacts["prompts"]["statistician_r2"] = statistician_prompt_r2
        llm_artifacts["raw"]["statistician_r2"] = str(st_out_r2)
        llm_artifacts["parsed"]["statistician_r2"] = st_obj_r2

        # Expose Round-2 JSON under the legacy keys so downstream logging keeps working.
        llm_artifacts["prompts"]["skeptic"] = skeptic_prompt_r2
        llm_artifacts["raw"]["skeptic"] = str(sk_out_r2)
        llm_artifacts["parsed"]["skeptic"] = sk_obj_r2
        llm_artifacts["prompts"]["statistician"] = statistician_prompt_r2
        llm_artifacts["raw"]["statistician"] = str(st_out_r2)
        llm_artifacts["parsed"]["statistician"] = st_obj_r2
        set_context("orchestrator_llm_artifacts", llm_artifacts)

        # ── Apply Round-2 actions sequentially: Skeptic first, then Statistician.
        sk_current_names = [
            str(c.get("name"))
            for c in candidates_payload.get("candidates", [])
            if isinstance(c, dict) and c.get("name")
        ]
        sk_actions = _validate_actions_against_universe(
            sk_obj_r2, universe_names, current_names=sk_current_names, who="Skeptic",
            by_name_registry=by_name, n_models=n_models, n_windows=_nw_early,
        )
        candidates_payload = _apply_actions_to_payload(candidates_payload, sk_actions, universe_by_name=by_name, n_models=n_models)
        candidates_after_skeptic = _candidate_names_from_payload(candidates_payload)

        # Refresh tool inputs between agents so Statistician's action is evaluated on the post-Skeptic payload.
        set_context("candidates_json_for_debate", json.dumps(candidates_payload, ensure_ascii=False))

        # Statistician-R2 was prompted against the pre-round candidate list, so
        # validate removals against that reference set. Any names already removed
        # by the Skeptic become harmless no-ops when applied to the post-Skeptic payload.
        st_actions = _validate_actions_against_universe(
            st_obj_r2, universe_names, current_names=current_names_r1, who="Statistician",
            by_name_registry=by_name, n_models=n_models, n_windows=_nw_early,
        )
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
            "dropped_selected_names": _truly_dropped,
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


# ─────────────────────────────────────────────────────────────────────────────
# V2 Pipeline: SeriesAnnotator → StrategySelector → Deterministic eval + Oracle
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_pipeline_v2(
    series_annotator_model: _utils.ModelConfig,
    strategy_selector_model: _utils.ModelConfig,
    debug: bool = False,
    rolling_mode: str = "expanding",
    train_window: int = 3,
    require_tool_call: bool = True,
    llm_logs: bool = True,
) -> Dict[str, Any]:
    """V2 pipeline: structured annotation → strategy selection → deterministic eval + oracle.

    Key design decisions:
    - temperature=0 on both agents for full reproducibility.
    - Oracle (all candidates from the universe) always runs for comparison.
    - Both oracle and LLM-guided eval use the 'balanced' preset for fair delta measurement.
    - No debate: LLM annotates and selects; deterministic evaluator decides.
    """

    def _log(msg: str) -> None:
        if llm_logs:
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts} ORCH|V2] {msg}", flush=True)

    _log(
        f"Starting V2 pipeline | annotator={series_annotator_model.model} "
        f"| selector={strategy_selector_model.model} | rolling={rolling_mode} | train_window={train_window}"
    )

    series_annotator = create_series_annotator_agent(series_annotator_model.model, debug=debug)
    strategy_selector = create_strategy_selector_agent(strategy_selector_model.model, debug=debug)

    llm_artifacts: Dict[str, Any] = {
        "series_annotator": series_annotator_model.model,
        "strategy_selector": strategy_selector_model.model,
        "prompts": {},
        "raw": {},
        "parsed": {},
    }

    # Balanced preset used consistently for both LLM-guided and oracle evals.
    _BALANCED = {"a_rmse": 0.3, "b_smape": 0.3, "c_mape": 0.2, "d_pocid": 0.2}

    eval_cfg = EvaluationConfig()
    eval_cfg.rolling.mode = rolling_mode
    eval_cfg.rolling.train_window = int(train_window)
    eval_cfg.metrics.mape_zero = "skip"
    eval_cfg.metrics.mape_epsilon = 1e-8
    eval_cfg.score.a_rmse = _BALANCED["a_rmse"]
    eval_cfg.score.b_smape = _BALANCED["b_smape"]
    eval_cfg.score.c_mape = _BALANCED["c_mape"]
    eval_cfg.score.d_pocid = _BALANCED["d_pocid"]

    set_context("config_json_for_proposer", json.dumps({"mape_zero": "skip", "mape_epsilon": 1e-8}))
    set_context("proposer_max_candidates", 12)

    # ── Step 0: Pre-build fold CoT context (shared by both agents) ───────────
    if not get_context("pattern_analyst_cot_context"):
        try:
            _build_fold_cot_context_tool()
            _log("build_fold_cot_context_tool: pre-built successfully.")
        except Exception as _e:
            _log(f"build_fold_cot_context_tool pre-build failed (non-fatal): {_e}")

    set_context("orchestrator_llm_artifacts", llm_artifacts)

    # ── Step 1: SeriesAnnotator → SeriesProfile ───────────────────────────────
    _log("SeriesAnnotator: analyzing series patterns...")
    sa_prompt = (
        "Call build_fold_cot_context() FIRST to analyze the validation folds. "
        "Then return ONLY JSON with the SeriesProfile per the output schema in your instructions. "
        "No markdown, no explanation — only the JSON object."
    )
    llm_artifacts["prompts"]["series_annotator"] = sa_prompt

    series_profile: Dict[str, Any] = {}
    try:
        sa_out, sa_obj = _run_agent_with_retry(
            lambda: series_annotator.run(sa_prompt).content,
            "SeriesAnnotator",
            max_retries=2,
            log_func=_log,
        )
        series_profile = sa_obj if isinstance(sa_obj, dict) else {}
        set_context("series_profile", series_profile)
        llm_artifacts["raw"]["series_annotator"] = str(sa_out)
        llm_artifacts["parsed"]["series_annotator"] = series_profile
        _log(
            f"SeriesAnnotator: strategy_type={series_profile.get('combination_recommendation', {}).get('strategy_type')} "
            f"| confidence={series_profile.get('confidence')}"
        )
    except Exception as _e:
        _log(f"SeriesAnnotator failed (non-fatal, continuing without profile): {_e}")
        set_context("series_profile", {})

    set_context("orchestrator_llm_artifacts", llm_artifacts)

    # ── Step 2: StrategySelector → selected candidates ───────────────────────
    ss_prompt = (
        "Call strategy_brief() FIRST — it contains the SeriesProfile and the full candidate library. "
        "Use series_profile fields to justify every selection. "
        "Return ONLY JSON per the output schema. No markdown, no explanation."
    )
    llm_artifacts["prompts"]["strategy_selector"] = ss_prompt

    _log("StrategySelector: selecting candidates from library...")
    ss_out, ss_obj = _run_agent_with_retry(
        lambda: strategy_selector.run(ss_prompt).content,
        "StrategySelector",
        max_retries=3,
        log_func=_log,
    )
    llm_artifacts["raw"]["strategy_selector"] = str(ss_out)
    llm_artifacts["parsed"]["strategy_selector"] = ss_obj
    set_context("orchestrator_llm_artifacts", llm_artifacts)

    # ── Step 3: Resolve candidate library from context ────────────────────────
    brief = get_context("orchestrator_strategy_brief")
    if not isinstance(brief, dict):
        _log("StrategySelector did not call strategy_brief_tool; invoking fallback...")
        try:
            from orchestrator.tools import strategy_brief_tool as _sbt
            _sbt()
            brief = get_context("orchestrator_strategy_brief")
        except Exception as _e:
            raise RuntimeError(f"strategy_brief_tool fallback failed: {_e} (hard-stop)")
    if not isinstance(brief, dict):
        raise RuntimeError("strategy_brief_tool did not populate orchestrator_strategy_brief (hard-stop)")

    library = brief.get("candidate_library")
    if not isinstance(library, dict) or not isinstance(library.get("candidates"), list):
        raise RuntimeError("strategy_brief missing candidate_library.candidates (hard-stop)")

    summary = brief.get("validation_summary", {})
    candidates_all: List[Dict[str, Any]] = [c for c in library.get("candidates", []) if isinstance(c, dict)]
    by_name: Dict[str, Dict[str, Any]] = {str(c.get("name")): c for c in candidates_all if c.get("name")}
    universe_names: List[str] = sorted(by_name.keys())

    models_available = get_context("models_available", [])
    n_models = len(models_available) if isinstance(models_available, list) and models_available else int(summary.get("n_models", 2) or 2)
    n_windows = int(summary.get("n_windows", train_window) or train_window)

    # ── Step 4: Parse + validate StrategySelector output ─────────────────────
    raw_selected = ss_obj.get("selected_names", [])
    if isinstance(raw_selected, str):
        raw_selected = [raw_selected]
    if not isinstance(raw_selected, list):
        raw_selected = []
    selected_names: List[str] = [str(x) for x in raw_selected if str(x)]

    # Resolve unknown names via pattern parser
    resolved_names: List[str] = []
    dropped_names: List[str] = []
    for name in selected_names:
        if name in by_name:
            resolved_names.append(name)
            continue
        resolved = _resolve_unknown_candidate(name, n_models, n_windows)
        if resolved is not None:
            rn = str(resolved["name"])
            if rn not in by_name:
                by_name[rn] = resolved
                candidates_all.append(resolved)
                if rn not in universe_names:
                    universe_names.append(rn)
            _log(f"Resolved unknown name '{name}' → '{rn}'")
            resolved_names.append(rn)
        else:
            dropped_names.append(name)
            _log(f"Dropped unknown name '{name}'")
    selected_names = resolved_names

    if dropped_names:
        _log(f"StrategySelector dropped {len(dropped_names)} unknown names: {dropped_names}")

    # Safety: at least 2 candidates
    if len(selected_names) < 2:
        fallback = [str(c.get("name")) for c in candidates_all[:4] if c.get("name")]
        selected_names = [n for n in fallback if n in by_name][:4]
        _log(f"Safety fallback: using top-4 from library: {selected_names}")

    # Confidence-gated safety: when the series is hard to predict (confidence=low),
    # force-include robust fallback candidates so the evaluator always has a safe option
    # even if the LLM made a poor selection on noisy validation data.
    _sp_confidence = series_profile.get("confidence", "medium") if isinstance(series_profile, dict) else "medium"
    if _sp_confidence == "low":
        _force_names = ["baseline_mean", "inv_rmse_weights_per_horizon_k3_shrink02"]
        for _fn in _force_names:
            if _fn in by_name and _fn not in selected_names:
                selected_names.append(_fn)
                _log(f"confidence=low: force-adding '{_fn}' to candidate set")

    # Apply any params_overrides
    params_overrides = ss_obj.get("params_overrides") or {}
    candidates_payload: Dict[str, Any] = {
        "candidates": [dict(by_name[n]) for n in selected_names if n in by_name]
    }
    try:
        validated_actions = _validate_actions_against_universe(
            {"add_names": [], "remove_names": [], "params_overrides": params_overrides},
            universe_names,
            current_names=selected_names,
            who="StrategySelector",
            by_name_registry=by_name,
            n_models=n_models,
            n_windows=n_windows,
        )
        candidates_payload = _apply_actions_to_payload(
            candidates_payload, validated_actions, universe_by_name=by_name, n_models=n_models
        )
    except Exception as _e:
        _log(f"params_overrides validation failed (non-fatal, using unmodified candidates): {_e}")

    # ── Step 5: LLM-guided deterministic evaluation ───────────────────────────
    data = load_validation_from_context()
    llm_candidates = parse_candidates(candidates_payload.get("candidates", []))
    if not llm_candidates:
        raise RuntimeError("No valid candidates after StrategySelector processing (hard-stop)")

    _log(f"Evaluating {len(llm_candidates)} LLM-selected candidates...")
    llm_eval = evaluate_all(data, llm_candidates, eval_cfg)
    if not isinstance(llm_eval, dict) or not llm_eval.get("best"):
        raise RuntimeError("LLM-guided evaluation produced no best candidate (hard-stop)")

    set_context("orchestrator_last_eval", llm_eval)

    # ── Step 6: Oracle evaluation (ALL candidates from universe) ─────────────
    oracle_info: Dict[str, Any] = {
        "best_name": "",
        "best_score": float("nan"),
        "best_method": "",
        "n_candidates": len(candidates_all),
        "llm_selected_in_oracle_top5": False,
    }
    try:
        oracle_candidates = parse_candidates(candidates_all)
        _log(f"Oracle: evaluating {len(oracle_candidates)} candidates from full universe...")
        oracle_eval = evaluate_all(data, oracle_candidates, eval_cfg)
        ob = oracle_eval.get("best") if isinstance(oracle_eval, dict) else None
        if isinstance(ob, dict):
            oracle_info["best_name"] = str(ob.get("candidate", {}).get("name", ""))
            oracle_info["best_score"] = float(ob.get("score", float("nan")))
            oracle_info["best_method"] = str(ob.get("candidate", {}).get("params", {}).get("method", ""))
            # Check if any of the LLM-selected candidates appear in oracle top-5
            oracle_top5 = [r.get("name") for r in (oracle_eval.get("ranking") or [])[:5]]
            oracle_info["llm_selected_in_oracle_top5"] = any(n in oracle_top5 for n in selected_names)
            _log(
                f"Oracle best: {oracle_info['best_name']} (score={oracle_info['best_score']:.4f}) "
                f"| LLM in oracle top-5: {oracle_info['llm_selected_in_oracle_top5']}"
            )
    except Exception as _oe:
        _log(f"Oracle eval failed (non-fatal): {_oe}")

    # ── Step 6.5: Fixed baselines — always deterministic, independent of LLM ────
    # These are the publication baselines: any LLM-guided combination that fails to
    # beat equal_weights is not adding value. References: Stock & Watson (2004),
    # Timmermann (2006) — the 'forecast combination puzzle'.
    baselines_info: Dict[str, Any] = {
        "equal_weights_score": float("nan"),
        "best_single_score": float("nan"),
        "best_single_model": "",
        "llm_vs_equal_weights_delta": float("nan"),
        "llm_vs_best_single_delta": float("nan"),
    }
    try:
        _ew_cand = CandidateStrategy(
            name="equal_weights",
            type="baseline",
            description="Simple equal-weights average of all models (canonical baseline).",
            formula="y_hat(h)=mean_m pred_m(h)",
            learns_weights=False,
            constraints="none",
            risks=["sensitive to outlier models"],
            validation_plan="rolling",
            params={"method": "mean"},
        )
        _bs_cand = CandidateStrategy(
            name="best_single_rolling",
            type="selection",
            description="Best single model by past-window RMSE (anti-leakage).",
            formula="m*=argmin_m RMSE_past(m)",
            learns_weights=False,
            constraints="anti-leakage rolling",
            risks=["unstable with few windows"],
            validation_plan="rolling",
            params={"method": "best_single", "selection_metric": "rmse"},
        )
        _log("Fixed baselines: evaluating equal_weights + best_single_rolling...")
        _ew_eval = evaluate_all(data, [_ew_cand], eval_cfg)
        _bs_eval = evaluate_all(data, [_bs_cand], eval_cfg)
        _ew_best = _ew_eval.get("best") if isinstance(_ew_eval, dict) else None
        _bs_best = _bs_eval.get("best") if isinstance(_bs_eval, dict) else None
        if _ew_best:
            baselines_info["equal_weights_score"] = float(_ew_best.get("score", float("nan")))
        if _bs_best:
            baselines_info["best_single_score"] = float(_bs_best.get("score", float("nan")))
            _bs_debug = _bs_best.get("predict_debug") or {}
            baselines_info["best_single_model"] = str(_bs_debug.get("chosen_model", ""))
        _log(
            f"Baselines: equal_weights={baselines_info['equal_weights_score']:.4f} "
            f"| best_single={baselines_info['best_single_score']:.4f} "
            f"({baselines_info['best_single_model']})"
        )
    except Exception as _be:
        _log(f"Fixed baselines eval failed (non-fatal): {_be}")

    # ── Step 7: Final prediction using LLM-selected best ─────────────────────
    best = llm_eval["best"]
    best_candidate = CandidateStrategy(**best["candidate"])
    _log(f"LLM-selected best: {best_candidate.name} (score={best['score']:.4f})")

    pred = predict_final_from_context(
        best_candidate, RollingConfig(mode=rolling_mode, train_window=int(train_window))
    )

    llm_score = float(best.get("score", float("nan")))
    oracle_score_val = oracle_info["best_score"]
    import math

    def _safe_delta(a: float, b: float) -> float:
        return a - b if (math.isfinite(a) and math.isfinite(b)) else float("nan")

    llm_vs_oracle_delta = _safe_delta(llm_score, oracle_score_val)

    # Baseline deltas: negative means LLM beats the baseline (lower score = better)
    _ew_score = baselines_info["equal_weights_score"]
    _bs_score = baselines_info["best_single_score"]
    baselines_info["llm_vs_equal_weights_delta"] = _safe_delta(llm_score, _ew_score)
    baselines_info["llm_vs_best_single_delta"] = _safe_delta(llm_score, _bs_score)

    # ── Step 8: Tool call traceability ───────────────────────────────────────
    tools_called = get_context("tools_called", [])
    if not isinstance(tools_called, list):
        tools_called = []
    tools_called.append("evaluate_strategies_v2")
    set_context("tools_called", tools_called)

    description = {
        "mode": "llm_v2",
        "agents": {
            "series_annotator": series_annotator_model.model,
            "strategy_selector": strategy_selector_model.model,
        },
        "candidates_trace": {
            "llm_selected": selected_names,
            "n_llm_selected": len(llm_candidates),
            "dropped_names": dropped_names,
            "n_oracle": oracle_info["n_candidates"],
        },
        "tool_validation": {
            "tools_called": tools_called,
            "require_tool_call": bool(require_tool_call),
            "tool_missing": bool(
                require_tool_call
                and "build_fold_cot_context_tool" not in tools_called
                and "strategy_brief_tool" not in tools_called
            ),
        },
        "score_preset": "balanced",
        "best": best_candidate.to_dict(),
        "score": best.get("score"),
        "aggregate": best.get("aggregate"),
        "stability": best.get("stability"),
        "predict_debug": pred.get("debug", {}),
        "oracle": oracle_info,
        "llm_vs_oracle_delta": llm_vs_oracle_delta,
        "baselines": baselines_info,
        "series_profile": series_profile,
        "strategy_reasoning": ss_obj.get("reasoning", {}),
    }

    out = {
        "success": True,
        "best": best_candidate.to_dict(),
        "ranking": llm_eval.get("ranking", []),
        "description": json.dumps(description, ensure_ascii=False),
        "result": [float(x) for x in pred["result"]],
        "eval": llm_eval,
        "oracle": oracle_info,
        "llm_vs_oracle_delta": llm_vs_oracle_delta,
        "baselines": baselines_info,
        "llm_artifacts": llm_artifacts,
        "series_profile": series_profile,
        "strategy_reasoning": ss_obj.get("reasoning", {}),
    }

    set_context("orchestrator_last_pipeline", out)
    set_context("orchestrator_last_candidates", candidates_payload)
    _log("V2 pipeline completed.")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# V3 PIPELINE — SeriesAnalyst → ModelCritic (prune) → CombinationArchitect → robust core
# ══════════════════════════════════════════════════════════════════════════════

_V3_REGIME_METHODS = {
    "robust": "double_shrinkage_per_horizon",
    "adaptive": "ade_dynamic_error_per_horizon",
    "structured": "stl_hierarchical_stacking",
    "selection": "topk_mean_per_horizon",
}


def _v3_apply_pruning_floor(
    prune_models: List[str],
    model_names: List[str],
    diag: Dict[str, Any],
    min_keep: int,
) -> Dict[str, Any]:
    """Enforce the statistical floor over the agent's prune decision.

    - Never prune a model in the MCS superior set unless it is the worse member of a
      redundant pair (corr > 0.95).
    - Always keep at least `min_keep` models (best by rmse_mean).
    Returns {survivors, pruned, blocked_by_mcs}.
    """

    import numpy as _np

    superior = set(diag.get("model_confidence_set", {}).get("superior_set", []) or [])
    redundant_worse = {
        str(p.get("worse_model")) for p in diag.get("redundant_pairs", []) if isinstance(p, dict)
    }
    per_model = diag.get("per_model", {})

    requested = {str(m) for m in (prune_models or []) if str(m) in set(model_names)}

    # Floor 1: protect MCS-superior models unless redundant.
    blocked_by_mcs = sorted([m for m in requested if m in superior and m not in redundant_worse])
    effective_prune = requested - set(blocked_by_mcs)

    survivors = [m for m in model_names if m not in effective_prune]

    # Floor 2: keep at least min_keep, restoring the best pruned models if needed.
    if len(survivors) < min_keep:
        pruned_sorted = sorted(
            effective_prune,
            key=lambda m: float(per_model.get(m, {}).get("rmse_mean", _np.inf)),
        )
        for m in pruned_sorted:
            if len(survivors) >= min_keep:
                break
            survivors.append(m)
            effective_prune.discard(m)
        survivors = [m for m in model_names if m in set(survivors)]

    return {
        "survivors": survivors,
        "pruned": sorted(effective_prune),
        "blocked_by_mcs": blocked_by_mcs,
    }


def run_llm_pipeline_v3(
    series_analyst_model: _utils.ModelConfig,
    model_critic_model: _utils.ModelConfig,
    combination_architect_model: _utils.ModelConfig,
    debug: bool = False,
    rolling_mode: str = "expanding",
    train_window: int = 3,
    require_tool_call: bool = True,
    llm_logs: bool = True,
    gate_alpha: float = 0.10,
) -> Dict[str, Any]:
    """V3 pipeline: structured analysis → model pruning → robust combination with a DM gate.

    Design (see ARCHITECTURE_V3_PROPOSAL.md):
    - The LLM makes the STRUCTURAL decisions (prune which models, which regime, how strongly to
      shrink); weights come from low-variance robust estimators.
    - The final combiner is ANCHORED to pruned-equal-weights. The chosen regime is only used if it
      beats pruned-equal-weights with a statistically significant Diebold-Mariano margin, else we
      fall back to pruned-equal-weights. This guarantees consistency (cannot do much worse than the
      robust anchor, which itself tends to beat full-pool mean and FFORMA/ADE on short samples).
    - temperature=0 on all three agents → reproducible.
    """

    import math
    import numpy as np

    def _log(msg: str) -> None:
        if llm_logs:
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts} ORCH|V3] {msg}", flush=True)

    _log(
        f"Starting V3 | analyst={series_analyst_model.model} | critic={model_critic_model.model} "
        f"| architect={combination_architect_model.model} | rolling={rolling_mode} | train_window={train_window}"
    )

    analyst = create_series_analyst_agent(series_analyst_model.model, debug=debug)
    critic = create_model_critic_agent(model_critic_model.model, debug=debug)
    architect = create_combination_architect_agent(combination_architect_model.model, debug=debug)

    llm_artifacts: Dict[str, Any] = {
        "series_analyst": series_analyst_model.model,
        "model_critic": model_critic_model.model,
        "combination_architect": combination_architect_model.model,
        "prompts": {},
        "raw": {},
        "parsed": {},
    }

    _BALANCED = {"a_rmse": 0.3, "b_smape": 0.3, "c_mape": 0.2, "d_pocid": 0.2}
    eval_cfg = EvaluationConfig()
    eval_cfg.rolling.mode = rolling_mode
    eval_cfg.rolling.train_window = int(train_window)
    eval_cfg.metrics.mape_zero = "skip"
    eval_cfg.metrics.mape_epsilon = 1e-8
    eval_cfg.score.a_rmse = _BALANCED["a_rmse"]
    eval_cfg.score.b_smape = _BALANCED["b_smape"]
    eval_cfg.score.c_mape = _BALANCED["c_mape"]
    eval_cfg.score.d_pocid = _BALANCED["d_pocid"]

    set_context("config_json_for_proposer", json.dumps({"mape_zero": "skip", "mape_epsilon": 1e-8}))
    set_context("orchestrator_llm_artifacts", llm_artifacts)

    data = load_validation_from_context()
    model_names = list(data.model_names)
    n_models = len(model_names)

    # ── Step 1: SeriesAnalyst → SeriesProfile ─────────────────────────────────
    sa_prompt = (
        "Call series_analysis_brief() FIRST, then return ONLY the SeriesProfile JSON per your schema. "
        "No markdown, no explanation."
    )
    llm_artifacts["prompts"]["series_analyst"] = sa_prompt
    series_profile: Dict[str, Any] = {}
    try:
        sa_out, sa_obj = _run_agent_with_retry(
            lambda: analyst.run(sa_prompt).content, "SeriesAnalyst", max_retries=2, log_func=_log
        )
        series_profile = sa_obj if isinstance(sa_obj, dict) else {}
        llm_artifacts["raw"]["series_analyst"] = str(sa_out)
        llm_artifacts["parsed"]["series_analyst"] = series_profile
    except Exception as _e:
        _log(f"SeriesAnalyst failed (non-fatal, continuing): {_e}")
        series_profile = {}
    set_context("series_profile", series_profile)
    set_context("orchestrator_llm_artifacts", llm_artifacts)
    _log(
        f"SeriesProfile: strategy_type={series_profile.get('combination_recommendation', {}).get('strategy_type')} "
        f"| confidence={series_profile.get('confidence')}"
    )

    # ── Step 2: ModelCritic → prune (with statistical floor) ──────────────────
    mc_prompt = (
        "Call model_critic_brief() FIRST, then return ONLY the pruning JSON per your schema "
        "(prune_models, reasoning, confidence). No markdown."
    )
    llm_artifacts["prompts"]["model_critic"] = mc_prompt
    prune_models: List[str] = []
    mc_obj: Dict[str, Any] = {}
    try:
        mc_out, mc_obj = _run_agent_with_retry(
            lambda: critic.run(mc_prompt).content, "ModelCritic", max_retries=2, log_func=_log
        )
        raw_prune = mc_obj.get("prune_models", []) if isinstance(mc_obj, dict) else []
        if isinstance(raw_prune, str):
            raw_prune = [raw_prune]
        prune_models = [str(m) for m in raw_prune if str(m)]
        llm_artifacts["raw"]["model_critic"] = str(mc_out)
        llm_artifacts["parsed"]["model_critic"] = mc_obj
    except Exception as _e:
        _log(f"ModelCritic failed (non-fatal, no pruning): {_e}")
        prune_models = []

    diag = _per_model_diagnostics(data)
    min_keep = min(max(3, int(round(np.sqrt(max(n_models, 1))))), n_models)
    floor = _v3_apply_pruning_floor(prune_models, model_names, diag, min_keep)
    survivors = floor["survivors"]
    set_context("survivors", survivors)
    _log(
        f"Pruning: requested={prune_models} | blocked_by_MCS={floor['blocked_by_mcs']} "
        f"| pruned={floor['pruned']} | survivors={len(survivors)}/{n_models}"
    )

    # ── Step 3: CombinationArchitect → regime + shrinkage ─────────────────────
    ca_prompt = (
        "Call combination_architect_brief() FIRST, then return ONLY the regime JSON per your schema "
        "(regime, shrinkage_lambda, score_preset, reasoning, confidence). No markdown."
    )
    llm_artifacts["prompts"]["combination_architect"] = ca_prompt
    regime = "robust"
    shrinkage_lambda = 0.7 if data.n_windows <= 3 else (0.5 if data.n_windows <= 6 else 0.3)
    ca_obj: Dict[str, Any] = {}
    try:
        ca_out, ca_obj = _run_agent_with_retry(
            lambda: architect.run(ca_prompt).content, "CombinationArchitect", max_retries=2, log_func=_log
        )
        if isinstance(ca_obj, dict):
            r = str(ca_obj.get("regime", "robust")).strip().lower()
            if r in _V3_REGIME_METHODS:
                regime = r
            try:
                shrinkage_lambda = float(ca_obj.get("shrinkage_lambda", shrinkage_lambda))
            except Exception:
                pass
            shrinkage_lambda = min(max(shrinkage_lambda, 0.0), 1.0)
        llm_artifacts["raw"]["combination_architect"] = str(ca_out)
        llm_artifacts["parsed"]["combination_architect"] = ca_obj
    except Exception as _e:
        _log(f"CombinationArchitect failed (non-fatal, using robust default): {_e}")
    set_context("orchestrator_llm_artifacts", llm_artifacts)
    _log(f"Architect: regime={regime} | shrinkage_lambda={shrinkage_lambda}")

    # ── Step 4: Build candidates (regime, anchor, full baselines) ─────────────
    brief = get_context("orchestrator_architect_brief", {})
    regime_knobs: Dict[str, Any] = {}
    if isinstance(brief, dict):
        regimes = brief.get("regimes", {})
        if isinstance(regimes, dict) and regime in regimes:
            regime_knobs = dict(regimes[regime].get("knobs", {}) or {})
    rec_top_k = regime_knobs.get("top_k") or max(2, min(int(round(np.sqrt(max(len(survivors), 1)))), len(survivors)))
    rec_l2 = float(regime_knobs.get("l2", 50.0))
    rec_period = int(regime_knobs.get("period", max(2, data.horizon // 2)))

    def _mk(name: str, method: str, extra: Dict[str, Any], use_survivors: bool) -> Dict[str, Any]:
        params: Dict[str, Any] = {"method": method}
        params.update(extra)
        if use_survivors:
            params["survivors"] = survivors
        return {
            "name": name,
            "type": "weighted",
            "description": name,
            "formula": "",
            "learns_weights": method not in {"mean", "median"},
            "constraints": "anti-leakage rolling",
            "risks": [],
            "validation_plan": "rolling",
            "params": params,
        }

    regime_method = _V3_REGIME_METHODS[regime]
    if regime_method == "double_shrinkage_per_horizon":
        regime_extra = {"shrinkage": shrinkage_lambda, "l2": rec_l2, "top_k": int(rec_top_k)}
    elif regime_method == "ade_dynamic_error_per_horizon":
        regime_extra = {"beta": 0.5, "eta": 1.0, "trim_ratio": 1.0}
    elif regime_method == "stl_hierarchical_stacking":
        regime_extra = {"period": rec_period, "shrinkage": shrinkage_lambda}
    else:  # topk_mean_per_horizon
        regime_extra = {"top_k": int(rec_top_k)}

    regime_cand = _mk("llm_regime", regime_method, regime_extra, use_survivors=True)
    anchor_cand = _mk("pruned_equal_weights", "mean", {}, use_survivors=True)
    full_mean_cand = _mk("full_mean", "mean", {}, use_survivors=False)
    full_median_cand = _mk("full_median", "median", {}, use_survivors=False)

    eval_list = [regime_cand, anchor_cand, full_mean_cand, full_median_cand]
    # Oracle-over-regimes: evaluate the other regimes on survivors to report whether the
    # LLM picked the best regime (paper ablation), reusing the same eval call.
    for rname, rmethod in _V3_REGIME_METHODS.items():
        if rname == regime:
            continue
        if rmethod == "double_shrinkage_per_horizon":
            ex = {"shrinkage": shrinkage_lambda, "l2": rec_l2, "top_k": int(rec_top_k)}
        elif rmethod == "ade_dynamic_error_per_horizon":
            ex = {"beta": 0.5, "eta": 1.0, "trim_ratio": 1.0}
        elif rmethod == "stl_hierarchical_stacking":
            ex = {"period": rec_period, "shrinkage": shrinkage_lambda}
        else:
            ex = {"top_k": int(rec_top_k)}
        eval_list.append(_mk(f"regime_{rname}", rmethod, ex, use_survivors=True))

    # ── Step 5: Evaluate everything on the validation windows ─────────────────
    ev = evaluate_all(data, parse_candidates(eval_list), eval_cfg)
    details_by_name: Dict[str, Dict[str, Any]] = {}
    for d in ev.get("details", []) or []:
        nm = str(d.get("candidate", {}).get("name", ""))
        details_by_name[nm] = d

    def _score(name: str) -> float:
        d = details_by_name.get(name)
        return float(d.get("score", float("nan"))) if isinstance(d, dict) else float("nan")

    def _resid(name: str) -> np.ndarray:
        d = details_by_name.get(name)
        if not isinstance(d, dict):
            return np.array([])
        return np.asarray(d.get("residuals_flat", []), dtype=float)

    regime_score = _score("llm_regime")
    anchor_score = _score("pruned_equal_weights")

    # ── Step 6: Diebold-Mariano significance gate vs pruned-equal-weights ─────
    dm = diebold_mariano(_resid("llm_regime"), _resid("pruned_equal_weights"), loss="squared", h=1)
    p_val = dm.get("p_value")
    dm_stat = dm.get("dm_stat")
    regime_better = bool(math.isfinite(regime_score) and math.isfinite(anchor_score) and regime_score < anchor_score)
    significant = bool(
        p_val is not None and math.isfinite(p_val) and p_val < float(gate_alpha)
        and dm_stat is not None and math.isfinite(dm_stat) and dm_stat < 0
    )
    if regime_better and significant:
        chosen_name, chosen_cand, fellback = "llm_regime", regime_cand, False
        gate_reason = f"regime beats anchor (DM p={p_val:.3f} < {gate_alpha}, stat={dm_stat:.2f})"
    elif regime_better and (p_val is None or not math.isfinite(p_val)):
        rel = (anchor_score - regime_score) / (abs(anchor_score) + 1e-9)
        if rel > 0.02:
            chosen_name, chosen_cand, fellback = "llm_regime", regime_cand, False
            gate_reason = f"DM unavailable; regime beats anchor by {rel:.1%} > 2% margin"
        else:
            chosen_name, chosen_cand, fellback = "pruned_equal_weights", anchor_cand, True
            gate_reason = f"DM unavailable; regime margin {rel:.1%} <= 2% → fallback to anchor"
    else:
        chosen_name, chosen_cand, fellback = "pruned_equal_weights", anchor_cand, True
        gate_reason = (
            f"regime not significantly better (score {regime_score:.4f} vs anchor {anchor_score:.4f}, "
            f"DM p={p_val}) → fallback to pruned-equal-weights"
        )
    _log(f"Gate: chosen={chosen_name} | fellback={fellback} | {gate_reason}")

    # ── Step 7: Final prediction with the chosen candidate ────────────────────
    best_candidate = CandidateStrategy(**chosen_cand)
    pred = predict_final_from_context(
        best_candidate, RollingConfig(mode=rolling_mode, train_window=int(train_window))
    )

    # ── Step 8: Baselines + deltas (publication evidence) ─────────────────────
    full_mean_score = _score("full_mean")
    full_median_score = _score("full_median")
    chosen_score = _score(chosen_name)

    def _delta(a: float, b: float) -> float:
        return a - b if (math.isfinite(a) and math.isfinite(b)) else float("nan")

    # Oracle-over-regimes: best regime by validation score (for "did LLM pick best regime").
    regime_scores = {"robust": None, "adaptive": None, "structured": None, "selection": None}
    for rname in regime_scores:
        nm = "llm_regime" if rname == regime else f"regime_{rname}"
        regime_scores[rname] = _score(nm) if nm in details_by_name else float("nan")
    oracle_regime = min(
        (r for r in regime_scores if math.isfinite(regime_scores[r])),
        key=lambda r: regime_scores[r],
        default=regime,
    )

    baselines_info = {
        "full_mean_score": full_mean_score,
        "full_median_score": full_median_score,
        "pruned_equal_weights_score": anchor_score,
        "llm_regime_score": regime_score,
        "chosen_score": chosen_score,
        "delta_chosen_vs_full_mean": _delta(chosen_score, full_mean_score),
        "delta_chosen_vs_full_median": _delta(chosen_score, full_median_score),
        "delta_chosen_vs_pruned_mean": _delta(chosen_score, anchor_score),
        "delta_pruned_mean_vs_full_mean": _delta(anchor_score, full_mean_score),
        "regime_scores": regime_scores,
        "oracle_regime": oracle_regime,
        "llm_picked_best_regime": bool(oracle_regime == regime),
    }

    tools_called = get_context("tools_called", [])
    if not isinstance(tools_called, list):
        tools_called = []
    tools_called.append("evaluate_strategies_v3")
    set_context("tools_called", tools_called)

    prune_report = {
        "requested_by_llm": prune_models,
        "blocked_by_mcs": floor["blocked_by_mcs"],
        "pruned": floor["pruned"],
        "survivors": survivors,
        "min_keep": min_keep,
        "mcs_superior_set": diag.get("model_confidence_set", {}).get("superior_set", []),
        "reasoning": mc_obj.get("reasoning", {}) if isinstance(mc_obj, dict) else {},
    }

    description = {
        "mode": "llm_v3",
        "agents": {
            "series_analyst": series_analyst_model.model,
            "model_critic": model_critic_model.model,
            "combination_architect": combination_architect_model.model,
        },
        "tool_validation": {
            "tools_called": tools_called,
            "require_tool_call": bool(require_tool_call),
            "tool_missing": bool(
                require_tool_call and "series_analysis_brief_tool" not in tools_called
            ),
        },
        "score_preset": "balanced",
        "best": best_candidate.to_dict(),
        "score": chosen_score,
        "chosen_name": chosen_name,
        "fellback_to_pruned_mean": fellback,
        "gate": {"alpha": gate_alpha, "dm": dm, "reason": gate_reason},
        "regime": regime,
        "shrinkage_lambda": shrinkage_lambda,
        "survivors": survivors,
        "prune_report": prune_report,
        "predict_debug": pred.get("debug", {}),
        "baselines": baselines_info,
        "series_profile": series_profile,
        "architect_reasoning": ca_obj.get("reasoning", "") if isinstance(ca_obj, dict) else "",
    }

    out = {
        "success": True,
        "best": best_candidate.to_dict(),
        "ranking": ev.get("ranking", []),
        "description": json.dumps(description, ensure_ascii=False),
        "result": [float(x) for x in pred["result"]],
        "eval": ev,
        "baselines": baselines_info,
        "prune_report": prune_report,
        "survivors": survivors,
        "regime": regime,
        "shrinkage_lambda": shrinkage_lambda,
        "fellback_to_pruned_mean": fellback,
        "series_profile": series_profile,
        "llm_artifacts": llm_artifacts,
    }

    set_context("orchestrator_last_pipeline", out)
    _log("V3 pipeline completed.")
    return out
