"""Prompt construction for the Phase 3 ReAct loop.

Section 3.2, principle 1: raw data never reaches the prompt. Everything here is a
compact card produced by the deterministic tools — the series profile, the pool
card and the ranked attempt history. The full series and the
`n_models x n_windows x horizon` forecast tensor stay in `ReactState`.

Section 3.2, principle 6: the agent is steered toward structured comparisons —
choose a subset, weight it, test it — and away from "assign a weight to each of the
25 models in one go".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from orchestrator_react.registry import TERMINAL_ACTION, describe_tools
from orchestrator_react.state import ReactState


def _compact(payload: Any, limit: int = 1800) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def build_system_prompt(include_history_rules: bool = True) -> str:
    """System prompt: the role, the closed action space and the output contract."""
    catalog = "\n".join(
        f"  {t['name']}({', '.join(t['args'])}) - {t['description']}" for t in describe_tools()
    )
    rules = [
        "You are a forecast COMBINATION AGENT.",
        "",
        "A pool of already-trained forecasting models produced predictions for one time",
        "series. Your job is to decide HOW TO COMBINE them. You never write forecast",
        "numbers and you never write weights: you call tools, and the tools compute.",
        "",
        "AVAILABLE TOOLS (this list is exhaustive - anything else is an error):",
        catalog,
        f"  {TERMINAL_ACTION}(attempt_id, confidence, justification) - stop and take an attempt",
        "",
        "OUTPUT FORMAT - exactly three lines, nothing else:",
        "Thought: <one or two sentences on what you want to learn or test next>",
        "Action: <tool name from the list above>",
        'Action Input: {"arg": value}',
        "",
        "HOW TO WORK:",
        "1. Read the series profile and the pool card you are given.",
        "2. Form a hypothesis about WHY a combination should work for THIS series,",
        "   grounded in what you observe: trend strength, seasonal strength, how stable",
        "   the model ranking is across windows, how redundant the models are.",
        "3. Test it with evaluate_strategy. Only evaluate_strategy scores anything.",
        "4. Read the result against the attempt history and decide: test another",
        f"   hypothesis, or {TERMINAL_ACTION} the best attempt.",
        "",
        "EVALUATING A STRATEGY - one call, flat arguments:",
        '  Action Input: {"combine": "mean", "pool": "pool_full", "rationale": "..."}',
        '  Action Input: {"combine": "median", "pool": "pool1", "rationale": "..."}',
        '  Action Input: {"combine": "trimmed_mean", "pool": "pool1", "trim_pct": 0.2}',
        '  Action Input: {"combine": "weighted", "pool": "pool1", "weights": "w1"}',
        '  Action Input: {"combine": "best_single", "model": "<a model name>"}',
        '  Action Input: {"combine": "dba", "pool": "pool1"}',
        "  You do NOT need combine_* first. It only builds the same object, so going",
        "  through it costs you an iteration for nothing.",
        "",
        "A TYPICAL SEQUENCE:",
        '  select_stable      {"k": 5}                  -> pool1',
        '  weights_inverse_error {"pool": "pool1"}      -> w1',
        '  evaluate_strategy  {"combine": "weighted", "pool": "pool1", "weights": "w1"}',
        f'  {TERMINAL_ACTION}  {{"attempt_id": "aN", "confidence": 0.7, "justification": "..."}}',
        "",
        "RULES:",
        "- Prefer small structured comparisons over one sweeping decision. Pick a",
        "  subset, weight it, test it. Do not try to reason about every model at once.",
        "- Weight tools return a HANDLE (w1, w2, ...). Pass the handle to",
        "  combine_weighted; you will never see the numbers, and you do not need to.",
        "- A handle is only valid with the pool it was computed on.",
        "- Repeating a strategy already in the history wastes an iteration.",
        "- Computing weights scores NOTHING on its own: every weights_* call must be",
        "  followed by evaluate_strategy. Do not spend your last turns building",
        "  handles you will not evaluate.",
        "- Handles start empty for every series. Never assume w1 or pool1 exists:",
        "  create it in this run before you refer to it.",
        "- If a SEEDED BASELINE is leading, the most direct improvement is usually the",
        "  SAME method on a better pool - dba on a pruned pool, median on a stable",
        "  subset - not a different method entirely. The baselines run on all models,",
        "  so a smaller pool is the variable you have not tried yet.",
        "- If a call is rejected, read the error: it lists the arguments that ARE",
        "  accepted. Do not retry the same shape.",
        f"- When you {TERMINAL_ACTION}, justification must explain the choice in terms of",
        "  OBSERVABLE SERIES CHARACTERISTICS, not just 'it had the lowest error'.",
        "  Example: 'ranking is unstable across windows (tau=0.17) and the models are",
        "  redundant, so a robust equal-weight combination of the stable subset beats",
        "  fitted weights, which would overfit 3 windows.'",
    ]
    if include_history_rules:
        rules += [
            "- The attempt history is ranked best to worst and is given to you every turn.",
            "  Use it: if two attempts differ only slightly, look for a different KIND of",
            "  strategy rather than tuning the winner.",
        ]
    return "\n".join(rules)


def build_turn_prompt(
    state: ReactState,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    scratchpad: Sequence[Dict[str, Any]],
    iteration: int,
    max_iterations: int,
    last_observation: Optional[Dict[str, Any]] = None,
    show_history: bool = True,
    show_rationales: bool = True,
    diagnosis: Optional[Dict[str, Any]] = None,
) -> str:
    """Per-turn message: cards, diagnosis, ranked history, scratchpad and budget."""
    parts: List[str] = []

    parts.append(f"ITERATION {iteration} of {max_iterations}.")
    parts.append("")
    parts.append("SERIES PROFILE:")
    parts.append(_compact(_slim_series_card(series_card)))
    parts.append("")
    parts.append("MODEL POOL:")
    parts.append(_compact(_slim_pool_card(pool_card)))

    if diagnosis:
        parts.append("")
        parts.append(f"DIAGNOSIS ({diagnosis.get('source', 'unknown')}) - a reading, not a rule:")
        parts.append(_compact(_slim_diagnosis(diagnosis), limit=700))

    if show_history:
        ranked = state.ranked_attempts()
        parts.append("")
        parts.append(f"ATTEMPT HISTORY ({len(ranked)}), best first - lower score is better:")
        if ranked:
            for a in ranked[:10]:
                parts.append("  " + _compact(a.brief(include_rationale=show_rationales), limit=320))
        else:
            parts.append("  (empty)")

    handles = _handles_summary(state)
    parts.append("")
    if handles:
        parts.append("HANDLES YOU HAVE CREATED:")
        parts.append(_compact(handles, limit=600))
    else:
        # Saying "none" matters: handles reset per series, and a model that just
        # finished another one will otherwise reach for a `w1` that does not exist.
        parts.append("HANDLES YOU HAVE CREATED: none yet (pools and weights start empty)")

    if scratchpad:
        parts.append("")
        parts.append("WHAT YOU DID SO FAR:")
        for entry in scratchpad[-6:]:
            parts.append(
                f"  [{entry['iteration']}] {entry['action']}({_compact(entry['action_args'], 120)})"
                f" -> {entry['observation_summary']}"
            )

    if last_observation is not None:
        parts.append("")
        parts.append("LAST OBSERVATION (full):")
        parts.append(_compact(last_observation, limit=1400))

    remaining = max_iterations - iteration + 1
    pending = _unscored_weights(state)
    parts.append("")
    if pending and remaining <= 3:
        parts.append(
            f"NOTE: {pending} were computed but never scored. Weights score nothing "
            "until evaluate_strategy runs on them, so that work is lost unless you "
            "evaluate now."
        )
    if remaining <= 1:
        parts.append(
            f"This is your LAST iteration. Use {TERMINAL_ACTION} to take the best attempt."
        )
    else:
        parts.append(
            f"{remaining} iterations left. Respond with Thought / Action / Action Input."
        )
    return "\n".join(parts)


def _slim_series_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """Keeps the fields that inform a combination decision; drops the rest.

    catch22 is 22 numbers the agent cannot act on directly, so it is reduced to a
    marker. The full profile is still written to `series_profile_json` in the CSV.
    """
    keep = (
        "n_points", "frequency", "seasonal_period", "seasonal_period_source", "horizon",
        "n_validation_windows", "n_models", "trend_strength", "seasonal_strength",
        "component_method", "trend_champion", "seasonality_champion",
    )
    slim = {k: card[k] for k in keep if k in card}
    stat = card.get("stationarity")
    if isinstance(stat, dict):
        slim["stationarity"] = {
            "verdict": stat.get("verdict"),
            "reliable": stat.get("reliable"),
        }
    out = card.get("outliers")
    if isinstance(out, dict):
        slim["outliers_pct"] = out.get("pct")
    feats = card.get("features")
    if isinstance(feats, dict):
        slim["features"] = {
            k: feats[k]
            for k in ("acf1", "acf_seasonal", "spectral_entropy", "coef_variation", "crosses_zero")
            if k in feats
        }
    slim["catch22"] = "computed" if isinstance(card.get("catch22"), dict) else card.get("catch22")
    return slim


def _slim_pool_card(card: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("n_models", "n_windows", "horizon", "error_table", "best_model_per_window")
    slim = {k: card[k] for k in keep if k in card}
    stab = card.get("ranking_stability")
    if isinstance(stab, dict):
        slim["ranking_stability"] = {
            "mean_kendall_tau": stab.get("mean_kendall_tau"),
            "verdict": stab.get("verdict"),
            "always_top3": stab.get("always_top3"),
        }
    corr = card.get("error_correlation")
    if isinstance(corr, dict):
        slim["error_correlation"] = {
            "mean_corr": corr.get("mean_corr"),
            "n_groups": corr.get("n_groups"),
            "n_independent": corr.get("n_independent"),
            "redundant_groups": corr.get("redundant_groups"),
        }
    return slim


def _slim_diagnosis(diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    """The Phase 1 reading, without its provenance bookkeeping."""
    return {
        k: diagnosis[k]
        for k in ("regime", "predictability", "combination_hint", "risks", "narrative")
        if k in diagnosis
    }


def _handles_summary(state: ReactState) -> Dict[str, Any]:
    """Pools and weight handles the agent can still refer to."""
    pools = {
        h: {"k": len(idx), "models": [state.model_names[i] for i in idx][:6]}
        for h, idx in state.pools.items()
        if h != "pool_full"
    }
    used = {a.spec.get("weights") for a in state.attempts if a.spec.get("weights")}
    weights = {
        h: {
            "method": r.method,
            "pool": r.pool_handle,
            "concentration": state.weights_summary(h).get("concentration"),
            "scored": h in used,
        }
        for h, r in state.weights.items()
    }
    out: Dict[str, Any] = {}
    if pools:
        out["pools"] = pools
    if weights:
        out["weights"] = weights
    return out


def _unscored_weights(state: ReactState) -> List[str]:
    """Weight handles that cost a turn to build and have never been scored."""
    used = {a.spec.get("weights") for a in state.attempts if a.spec.get("weights")}
    return [h for h in state.weights if h not in used]


def summarize_observation(action: str, ok: bool, observation: Dict[str, Any]) -> str:
    """One-line digest of a tool result, for the scratchpad and the trajectory.

    Kept short on purpose: `react_trajectory_json` has to stay parseable and small
    enough to sit in a CSV cell for every series.
    """
    if not ok:
        return f"ERROR {observation.get('error', 'unknown')}: {str(observation.get('detail', ''))[:120]}"

    if action == "evaluate_strategy":
        leader = observation.get("current_best", {})
        tail = ""
        if not observation.get("is_best"):
            tail = f", leader is {leader.get('id')} ({leader.get('strategy', '?')})"
        return (
            f"rank {observation.get('rank')}/{observation.get('total_attempts')}"
            f" score={observation.get('score')}"
            f" rmse={observation.get('metrics', {}).get('rmse')}"
            + (" (best so far)" if observation.get("is_best") else tail)
            + (" [already tested]" if observation.get("already_tested") else "")
        )
    if action.startswith("weights_"):
        s = observation.get("summary", {})
        return (
            f"handle {observation.get('weights')} mode={observation.get('effective_mode')}"
            f" active={s.get('n_active')}/{s.get('n_models')} conc={s.get('concentration')}"
        )
    if action == "prune_redundant":
        return (
            f"pool {observation.get('pool')} {observation.get('n_before')}->"
            f"{observation.get('n_after')} models, dropped {observation.get('removed', [])[:4]}"
        )
    if action.startswith("select_"):
        models = observation.get("models", [])
        names = [m["model"] if isinstance(m, dict) else m for m in models][:5]
        return f"pool {observation.get('pool')} k={observation.get('k')} {names}"
    if action.startswith("combine_"):
        strategy = observation.get("strategy", {})
        return (
            f"strategy built: {json.dumps(strategy, separators=(',', ':'))}"
            " -> pass it to evaluate_strategy"
        )
    if action == "series_profile":
        return (
            f"trend={observation.get('trend_strength')} seasonal={observation.get('seasonal_strength')}"
            f" period={observation.get('seasonal_period')}"
        )
    if action == "stl_summary":
        return f"dominant={observation.get('dominant_component')} trend%={observation.get('trend_pct')}"
    if action == "error_summary":
        top = observation.get("top", [])
        return f"best={top[0]['model'] if top else '?'} spread={observation.get('relative_spread')}"
    if action == "ranking_stability":
        return f"tau={observation.get('mean_kendall_tau')} ({observation.get('verdict')})"
    if action == "error_correlation":
        return f"{observation.get('n_groups')} groups, {observation.get('n_independent')} independent"
    if action == "dm_test":
        return f"p={observation.get('p_value')} -> {observation.get('verdict')}"
    if action == "sanity_check":
        return "ok" if observation.get("ok") else f"warnings: {observation.get('warnings')}"
    if action == "list_attempts":
        return f"{observation.get('total')} attempts, best={observation.get('best')}"
    return _compact(observation, limit=160)
