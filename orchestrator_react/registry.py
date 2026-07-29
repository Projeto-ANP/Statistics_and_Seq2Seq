"""Closed action space: dispatch, validation and trace.

The ReAct agent may only emit `Action: <name> | Action Input: <json>`. This module
is the single place where a tool name becomes execution. Names outside the catalog,
unknown arguments and exceptions all become a **structured** error observation that
the agent reads and can correct on the next turn — and that feeds the `tools_called`
and `tool_missing` CSV fields.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from orchestrator_react import tools as T
from orchestrator_react.state import ReactState


#: Catalog — the order mirrors Section 3.4 of the specification.
TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    # 3.4.1 diagnostics
    "series_profile": T.series_profile,
    "stl_summary": T.stl_summary,
    "error_summary": T.error_summary,
    "ranking_stability": T.ranking_stability,
    "error_correlation": T.error_correlation,
    "dm_test": T.dm_test,
    # 3.4.2 pool selection
    "select_top_k": T.select_top_k,
    "select_stable": T.select_stable,
    "prune_redundant": T.prune_redundant,
    # 3.4.3 weights
    "weights_inverse_error": T.weights_inverse_error,
    "weights_softmax_neg_error": T.weights_softmax_neg_error,
    "weights_error_trend": T.weights_error_trend,
    "weights_ols": T.weights_ols,
    "weights_feature_based": T.weights_feature_based,
    "weights_pooled_meta_model": T.weights_pooled_meta_model,
    # 3.4.4 combination
    "combine_mean": T.combine_mean,
    "combine_median": T.combine_median,
    "combine_trimmed_mean": T.combine_trimmed_mean,
    "combine_weighted": T.combine_weighted,
    "combine_dba": T.combine_dba,
    "combine_best_single": T.combine_best_single,
    # 3.4.5 validation
    "evaluate_strategy": T.evaluate_strategy,
    "sanity_check": T.sanity_check,
    "list_attempts": T.list_attempts,
}

#: There is no terminal tool — accepting is the agent's decision, emitted as
#: `Action: accept` and handled by the Phase 3 loop, not here.
TERMINAL_ACTION = "accept"

#: Tools where an unknown argument is dropped instead of rejected. Only the loop's
#: central tool qualifies: models routinely decorate the call with a descriptive
#: field ("origin", "note"), and failing the whole turn over one stray key costs an
#: iteration and teaches nothing. What was dropped comes back in the observation, so
#: the tolerance is visible rather than silent.
PERMISSIVE_TOOLS = {"evaluate_strategy"}

#: Failures meaning "the agent asked for something outside the catalog contract".
#: These are the ones that switch on the `tool_missing` CSV field.
SPEC_ERROR_KINDS = {
    "unknown_tool",
    "unknown_argument",
    "missing_required_argument",
    "invalid_action_input",
}


#: Tools that need more validation windows than the protocol always provides.
#: The value is the config field holding that tool's minimum.
WINDOW_GATED_TOOLS: Dict[str, str] = {"weights_ols": "min_windows_for_ols"}


def withheld_tools(config: Any, n_windows: int, state: Optional[Any] = None) -> Dict[str, str]:
    """Catalog entries unavailable for this run, mapped to the reason why.

    A tool that cannot produce a trustworthy answer is removed from the catalog
    instead of being offered and then failing: the agent never sees it, so it
    cannot spend an iteration on it, and the prompt stays honest about what the
    action space actually is.

    `state` is optional, and only used for the one gate that cannot be decided
    from `config`/`n_windows` alone: `weights_pooled_meta_model` needs a model
    that was fit once for the whole dataset run and attached to this series'
    `state` before Phase 3 opened (see `meta_model.py`) — `config` has no way to
    know whether that happened. Callers without a `state` (existing tests, or
    code built before this tool existed) keep working exactly as before; they
    just cannot withhold this one entry.
    """
    out: Dict[str, str] = {}
    for name, field_name in WINDOW_GATED_TOOLS.items():
        minimum = int(getattr(config, field_name, 0) or 0)
        if int(n_windows) < minimum:
            out[name] = (
                f"needs at least {minimum} validation windows, this run has {int(n_windows)}"
            )
    if state is not None and getattr(state, "pooled_meta_model", None) is None:
        out["weights_pooled_meta_model"] = (
            "no pooled meta-model was trained for this run "
            "(too few series in the dataset, or xgboost unavailable)"
        )
    return out


def tool_names(withheld: Optional[Dict[str, str]] = None) -> List[str]:
    return [n for n in TOOLS if not withheld or n not in withheld]


def describe_tools(withheld: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Compact signatures to inject into the system prompt."""
    out: List[Dict[str, Any]] = []
    for name, fn in TOOLS.items():
        if withheld and name in withheld:
            continue
        sig = inspect.signature(fn)
        params = []
        for pname, p in sig.parameters.items():
            if pname == "state":
                continue
            if p.default is inspect.Parameter.empty:
                params.append(pname)
            else:
                params.append(f"{pname}={p.default!r}")
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        out.append({"name": name, "args": params, "description": doc})
    return out


def call_tool(
    state: ReactState, name: str, args: Any = None,
    withheld: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Runs a catalog tool. Returns `(ok, observation)`.

    Never raises: any failure becomes an error observation and is recorded in
    `state.tool_errors`, so the loop keeps going and the CSV keeps the trace.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            obs = {
                "error": "invalid_action_input",
                "detail": f"not valid JSON: {exc}",
                "received": args[:200],
            }
            state.log_tool(str(name), {"raw": str(args)[:200]}, ok=False,
                           error=obs["detail"], kind=obs["error"])
            return False, obs
    if args is None:
        args = {}
    if not isinstance(args, dict):
        obs = {"error": "invalid_action_input", "detail": "expected a JSON object"}
        state.log_tool(str(name), {"raw": str(args)[:200]}, ok=False,
                       error=obs["detail"], kind=obs["error"])
        return False, obs

    fn = TOOLS.get(str(name))
    if fn is None or (withheld and str(name) in withheld):
        detail = (
            f"{name!r} is unavailable for this run: {withheld[str(name)]}"
            if withheld and str(name) in withheld
            else f"{name!r} is not in the catalog"
        )
        obs = {
            "error": "unknown_tool",
            "detail": detail,
            "available": tool_names(withheld),
        }
        state.log_tool(str(name), args, ok=False, error=obs["detail"], kind=obs["error"])
        return False, obs

    sig = inspect.signature(fn)
    accepted = {p for p in sig.parameters if p != "state"}
    unknown = [k for k in args if k not in accepted]
    ignored: List[str] = []
    if unknown:
        if str(name) in PERMISSIVE_TOOLS:
            ignored = sorted(unknown)
            args = {k: v for k, v in args.items() if k in accepted}
        else:
            obs = {
                "error": "unknown_argument",
                "detail": f"{name} does not accept {unknown}",
                "accepted": sorted(accepted),
            }
            state.log_tool(str(name), args, ok=False, error=obs["detail"], kind=obs["error"])
            return False, obs

    missing = [
        p
        for p, spec in sig.parameters.items()
        if p != "state" and spec.default is inspect.Parameter.empty and p not in args
    ]
    if missing:
        obs = {
            "error": "missing_required_argument",
            "detail": f"{name} requires {missing}",
            "accepted": sorted(accepted),
        }
        state.log_tool(str(name), args, ok=False, error=obs["detail"], kind=obs["error"])
        return False, obs

    try:
        result = fn(state, **args)
    except (ValueError, KeyError) as exc:
        obs = {"error": "invalid_argument", "detail": str(exc)}
        state.log_tool(str(name), args, ok=False, error=str(exc), kind=obs["error"])
        return False, obs
    except Exception as exc:  # pragma: no cover - loop safety net
        obs = {"error": "internal_failure", "detail": f"{type(exc).__name__}: {exc}"}
        state.log_tool(str(name), args, ok=False, error=str(exc), kind=obs["error"])
        return False, obs

    state.log_tool(str(name), args, ok=True)
    if ignored and isinstance(result, dict):
        result = {**result, "ignored_args": ignored}
    return True, result


def tools_called_summary(state: ReactState) -> Dict[str, Any]:
    """Feeds the `tools_called` and `tool_missing` CSV fields."""
    return {
        "tools_called": [
            {"tool": c["tool"], "ok": c["ok"], "args": c["args"]} for c in state.tools_called
        ],
        "n_calls": len(state.tools_called),
        "n_failures": len(state.tool_errors),
        "tool_missing": any(e.get("kind") in SPEC_ERROR_KINDS for e in state.tool_errors),
        "errors": state.tool_errors[:10],
    }
