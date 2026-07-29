"""Phase 3 — the ReAct decision loop (the core).

A single agent, cycling Thought -> Action -> Observation over the closed catalog of
Section 3.4. Every turn it receives the series card, the pool card and the ranked
attempt history, and it either calls a tool or accepts an attempt.

Guarantees this module enforces, so the loop cannot damage the result:

* **The applied strategy is always the best entry in the whole history**, baselines
  included (Section 3.2, principle 5). If the agent accepts something worse, its
  choice is recorded in `agent_accepted_id` and overridden — that is visible in the
  output, never silent.
* A malformed answer, an unknown tool or a bad argument becomes an observation the
  agent can read and correct, instead of aborting the series. The old pipeline
  hard-stopped on these; here they are just information.
* A fixed iteration budget plus early stopping after `early_stop_patience`
  consecutive proposals that fail to improve on the best (principle 8).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from orchestrator_react import prompts as P
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import AgentStep, LLMClient, LLMError, parse_agent_step
from orchestrator_react.registry import (
    TERMINAL_ACTION,
    call_tool,
    tools_called_summary,
    withheld_tools,
)
from orchestrator_react.state import Attempt, ReactState


#: How many times to re-ask when the model returns nothing at all.
EMPTY_RESPONSE_RETRIES = 2


@dataclass
class ReactResult:
    """Everything Phase 3 hands to Phase 4 and to the CSV writer."""

    final_attempt: Optional[Attempt]
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    iterations_used: int = 0
    early_stopped: bool = False
    stop_reason: str = ""
    agent_accepted_id: Optional[str] = None
    accept_confidence: Optional[float] = None
    justification: str = ""
    overridden: bool = False
    llm_model: str = "none"
    elapsed_s: float = 0.0
    tools: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    #: Generations that came back empty and were re-asked rather than charged to
    #: the iteration budget.
    empty_responses: int = 0
    #: Catalog entries this run could not support, mapped to the reason. Recorded
    #: so a row states which action space produced it.
    withheld_tools: Dict[str, str] = field(default_factory=dict)
    #: Raw model output for every turn the parser could not read. Kept out of the
    #: CSV, which it would bloat, and written to the per-series artifacts instead —
    #: that is where to look when a model keeps missing the output format.
    parse_failures: List[Dict[str, Any]] = field(default_factory=list)
    #: Raw model output for every turn the parser could not read. Kept out of the
    #: CSV, which would bloat `react_trajectory_json`, and written to the per-series
    #: artifacts instead — that is where you look when a model keeps missing the
    #: output format and you need to see what it actually said.
    parse_failures: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "final_attempt": self.final_attempt.attempt_id if self.final_attempt else None,
            "strategy": self.final_attempt.spec if self.final_attempt else None,
            "iterations_used": self.iterations_used,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "agent_accepted_id": self.agent_accepted_id,
            "accept_confidence": self.accept_confidence,
            "overridden": self.overridden,
            "llm_model": self.llm_model,
            "n_trajectory_steps": len(self.trajectory),
            "empty_responses": self.empty_responses,
            "withheld_tools": dict(self.withheld_tools),
            "elapsed_s": round(self.elapsed_s, 2),
        }


def run_react_loop(
    state: ReactState,
    client: Optional[LLMClient],
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    config: Optional[ReactConfig] = None,
    skip_reason: str = "",
    diagnosis: Optional[Dict[str, Any]] = None,
    on_step: Optional[Callable[[Optional[int], Dict[str, Any]], None]] = None,
) -> ReactResult:
    """Runs the decision loop and returns the winning attempt plus the full trace.

    Args:
        state: application state, already seeded with the Phase 2 baselines.
        client: the LLM. `None` runs the deterministic path — the best baseline is
            taken as-is, which is the "no agent" arm of the ablations.
        series_card: output of `series_profile()`.
        pool_card: output of `pool.pool_report()`.
        skip_reason: set by the calibration gate to bypass the loop.
        diagnosis: the Phase 1 reading, injected into every turn when present.
        on_step: called with `(dataset_index, trajectory_entry)` after every turn,
            so a long run can show what the agent is doing instead of only what it
            concluded.
    """
    config = config or state.config
    started = time.perf_counter()
    result = ReactResult(final_attempt=state.best_attempt())
    result.llm_model = getattr(client, "name", "none") if client else "none"

    if not state.attempts:
        raise RuntimeError(
            "the attempt history is empty: run pool.seed_baselines() before the loop, "
            "otherwise there is no floor to guarantee the result against"
        )

    if client is None or skip_reason:
        result.stop_reason = skip_reason or "no_llm_client"
        result.justification = _fallback_justification(state, series_card, pool_card)
        result.elapsed_s = time.perf_counter() - started
        result.tools = tools_called_summary(state)
        return result

    withheld = withheld_tools(config, state.n_windows)
    result.withheld_tools = dict(withheld)
    system = P.build_system_prompt(
        include_history_rules=config.show_attempt_history, withheld_tools=withheld
    )
    scratchpad: List[Dict[str, Any]] = []
    last_observation: Optional[Dict[str, Any]] = None
    best_score = _score(state.best_attempt())
    stale = 0
    max_iterations = max(1, int(config.max_iterations))

    for iteration in range(1, max_iterations + 1):
        result.iterations_used = iteration
        user = P.build_turn_prompt(
            state=state,
            series_card=series_card,
            pool_card=pool_card,
            scratchpad=scratchpad,
            iteration=iteration,
            max_iterations=max_iterations,
            last_observation=last_observation,
            show_history=config.show_attempt_history,
            show_rationales=config.show_attempt_rationales,
            diagnosis=diagnosis,
        )

        # An EMPTY answer is a failed generation, not a decision: gpt-oss emits a
        # <think> block and sometimes stops before writing anything after it.
        # Retrying costs a call; spending one of eight iterations on it costs a
        # hypothesis. A malformed but non-empty answer is different — that one goes
        # back to the agent as an observation, because it can learn from it.
        raw = ""
        step = None
        for attempt_no in range(1, EMPTY_RESPONSE_RETRIES + 2):
            try:
                raw = client.complete(system, user)
            except LLMError as exc:
                result.errors.append(str(exc))
                result.stop_reason = "llm_error"
                step = None
                break
            step = parse_agent_step(raw)
            if step.ok or step.parse_error != "empty response":
                break
            result.empty_responses += 1
            if attempt_no <= EMPTY_RESPONSE_RETRIES:
                result.errors.append(
                    f"iteration {iteration}: empty response, retrying "
                    f"({attempt_no}/{EMPTY_RESPONSE_RETRIES})"
                )
        if step is None:
            break
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "thought": _clip(step.thought, 600),
            "action": step.action or "",
            "action_args": step.action_input,
            "observation_summary": "",
        }

        if not step.ok:
            entry["action"] = step.action or "unparsed"
            entry["observation_summary"] = f"ERROR parse: {step.parse_error}"
            observation = {
                "error": "unparsed_response",
                "detail": step.parse_error,
                "reminder": "answer with three lines: Thought:, Action:, Action Input: {json}",
            }
            state.log_tool(
                step.action or "unparsed", {}, ok=False,
                error=step.parse_error, kind="invalid_action_input",
            )
            result.errors.append(f"iteration {iteration}: {step.parse_error}")
            result.parse_failures.append(
                {"iteration": iteration, "reason": step.parse_error, "raw": str(step.raw)[:4000]}
            )
            result.trajectory.append(entry)
            scratchpad.append(entry)
            last_observation = observation
            _emit(on_step, state, entry)
            continue

        # ── terminal action ──────────────────────────────────────────────────
        if step.action.strip().lower() == TERMINAL_ACTION:
            accepted, confidence, justification, problem = _read_accept(state, step)
            if problem:
                entry["observation_summary"] = f"ERROR accept: {problem}"
                last_observation = {"error": "invalid_accept", "detail": problem}
                state.log_tool(TERMINAL_ACTION, step.action_input, ok=False,
                               error=problem, kind="invalid_argument")
                result.trajectory.append(entry)
                scratchpad.append(entry)
                _emit(on_step, state, entry)
                continue

            result.agent_accepted_id = accepted.attempt_id
            result.accept_confidence = confidence
            result.justification = justification
            entry["observation_summary"] = (
                f"accepted {accepted.attempt_id} (confidence={confidence})"
            )
            result.trajectory.append(entry)
            _emit(on_step, state, entry)
            result.stop_reason = "agent_accepted"
            break

        # ── tool call ────────────────────────────────────────────────────────
        args = dict(step.action_input)
        if step.action == "evaluate_strategy":
            args.setdefault("rationale", step.thought or "")
            args["iteration"] = iteration

        ok, observation = call_tool(state, step.action, args, withheld=withheld)
        entry["observation_summary"] = P.summarize_observation(step.action, ok, observation)
        result.trajectory.append(entry)
        scratchpad.append(entry)
        last_observation = observation
        _emit(on_step, state, entry)

        if not ok:
            result.errors.append(f"iteration {iteration}: {observation.get('detail', '')}")
            continue

        # ── early stopping, counted on proposals only ────────────────────────
        if step.action == "evaluate_strategy":
            current = _score(state.best_attempt())
            improved = _improved(current, best_score, config.min_improvement)
            if improved:
                best_score = current
                stale = 0
            else:
                stale += 1
            if stale >= max(1, int(config.early_stop_patience)):
                result.early_stopped = True
                result.stop_reason = (
                    f"no improvement in {stale} consecutive proposals"
                )
                break
    else:
        result.stop_reason = "iteration_budget_exhausted"

    # ── principle 5: the applied strategy is the best of the whole history ───
    best = state.best_attempt()
    result.final_attempt = best
    if result.agent_accepted_id and best and result.agent_accepted_id != best.attempt_id:
        result.overridden = True
        result.errors.append(
            f"agent accepted {result.agent_accepted_id} but {best.attempt_id} scores better; "
            "applied the better one"
        )
    if not result.justification:
        result.justification = _fallback_justification(state, series_card, pool_card)

    result.elapsed_s = time.perf_counter() - started
    result.tools = tools_called_summary(state)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _emit(
    on_step: Optional[Callable[[Optional[int], Dict[str, Any]], None]],
    state: ReactState,
    entry: Dict[str, Any],
) -> None:
    """Reporting must never break the run."""
    if on_step is None:
        return
    try:
        on_step(state.dataset_index, entry)
    except Exception:
        pass


def _score(attempt: Optional[Attempt]) -> float:
    if attempt is None or not np.isfinite(attempt.score):
        return float("inf")
    return float(attempt.score)


def _improved(current: float, previous: float, min_gain: float) -> bool:
    if not np.isfinite(previous):
        return np.isfinite(current)
    if not np.isfinite(current):
        return False
    scale = abs(previous) or 1.0
    return (previous - current) / scale > float(min_gain)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _read_accept(state: ReactState, step: AgentStep):
    """Validates an `accept` action. Returns `(attempt, confidence, text, problem)`."""
    args = step.action_input or {}
    attempt_id = str(
        args.get("attempt_id") or args.get("id") or args.get("attempt") or ""
    ).strip()

    if not attempt_id:
        best = state.best_attempt()
        if best is None:
            return None, None, "", "no attempt to accept"
        attempt = best  # accepting without naming one means "the best"
    else:
        match = [a for a in state.attempts if a.attempt_id == attempt_id]
        if not match:
            known = [a.attempt_id for a in state.ranked_attempts()][:10]
            return None, None, "", f"unknown attempt_id {attempt_id!r}; known: {known}"
        attempt = match[0]

    confidence: Optional[float] = None
    raw_conf = args.get("confidence", args.get("accept_confidence"))
    if raw_conf is not None:
        try:
            confidence = float(np.clip(float(raw_conf), 0.0, 1.0))
        except (TypeError, ValueError):
            confidence = None

    justification = str(
        args.get("justification") or args.get("rationale") or step.thought or ""
    ).strip()
    return attempt, confidence, _clip(justification, 1200), ""


def _fallback_justification(
    state: ReactState, series_card: Dict[str, Any], pool_card: Dict[str, Any]
) -> str:
    """Deterministic causal justification, in terms of observable characteristics.

    Used when there is no LLM (ablation), when the loop ends on budget without an
    accept, or when the agent accepted without writing one. Principle 7 requires a
    causal explanation to exist either way — Phase 5 can later rewrite it in prose,
    but the run must never ship without one.
    """
    best = state.best_attempt()
    if best is None:
        return "no attempt was evaluated"

    bits: List[str] = []
    trend = series_card.get("trend_strength")
    seasonal = series_card.get("seasonal_strength")
    if trend is not None and seasonal is not None:
        bits.append(f"trend strength {trend} and seasonal strength {seasonal}")

    stability = pool_card.get("ranking_stability", {})
    tau, verdict = stability.get("mean_kendall_tau"), stability.get("verdict")
    if tau is not None:
        bits.append(f"model ranking is {verdict} across windows (Kendall tau {tau})")

    corr = pool_card.get("error_correlation", {})
    if isinstance(corr, dict) and corr.get("n_groups") is not None:
        bits.append(
            f"{corr.get('n_independent')} of {pool_card.get('n_models')} models carry "
            f"independent error"
        )

    spread = pool_card.get("error_table", {}).get("relative_spread")
    if spread is not None:
        bits.append(f"relative error spread across the pool is {spread}")

    context = "; ".join(bits) if bits else "no distinguishing series characteristics"
    origin = "a seeded deterministic baseline" if best.origin == "baseline" else "an agent proposal"
    return (
        f"Selected '{_label(best)}' ({origin}) as the best of {len(state.attempts)} "
        f"backtested attempts, score {round(float(best.score), 4)}. "
        f"Series context: {context}."
    )


def _label(attempt: Attempt) -> str:
    spec = attempt.spec
    parts = [str(spec.get("combine"))]
    if spec.get("pool") and spec["pool"] != "pool_full":
        parts.append(f"on {spec['pool']}")
    if spec.get("weights"):
        parts.append(f"weighted by {spec['weights']}")
    if spec.get("model"):
        parts.append(str(spec["model"]))
    return " ".join(parts)
