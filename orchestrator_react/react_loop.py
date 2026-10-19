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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from orchestrator_react import prompts as P
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import AgentStep, LLMClient, LLMError, parse_agent_step, split_think
from orchestrator_react.registry import (
    TERMINAL_ACTION,
    call_tool,
    tools_called_summary,
    withheld_tools,
)
from orchestrator_react.state import Attempt, ReactState


#: How many times to re-ask when the model returns nothing at all. gpt-oss stops
#: after its reasoning with no final message in roughly 1 call in 5 (measured, all
#: `done_reason=stop`, prompts far below the context window), independently across
#: seeds. An empty call costs ~100-350 tokens and no iteration, so the ladder is
#: long: with p~0.2 and a fresh seed/temperature each time, 9 asks leave ~5e-7.
EMPTY_RESPONSE_RETRIES = 8

#: How many times to re-ask when the API call itself fails transiently.
#:
#: Observed cause on a real run: Ollama's chat template for gpt-oss recognises a
#: "tool call" channel in the model's own harmony format and tries to parse it
#: JSON-side, server-side, before the response ever reaches this client. When the
#: model writes plain text into that channel instead of JSON — e.g. it wrote
#: "We need to output exactly three lines: Thought, Action, Action Input." —
#: Ollama's own parser fails and returns an error instead of the text. That is a
#: sampling hiccup on one turn, not evidence the server or model is broken: the
#: preflight already ran before the loop opened, and the same server had answered
#: every prior turn correctly. Retrying costs a call, exactly like the empty-
#: response case, and for the same reason: spending one of `max_iterations` on it
#: would charge the series for a glitch instead of a decision. Exhausting the
#: retries still sets `stop_reason="llm_error"` and still lets
#: `exec_dataset_orchestrator` raise under `allow_baseline_fallback=False` — this
#: only removes the false positives where one bad token would have killed an
#: otherwise-healthy run.
#:
#: Raised from 2 to 4 after a second real occurrence (ANP_MONTHLY series 80)
#: exhausted 2 retries with the SAME narrated-plan-instead-of-JSON pattern three
#: times in a row ("We will call weights_softmax_neg_error on pool1. We need to
#: specify arguments: ..."), not three unrelated glitches. That is weak evidence
#: this failure mode is somewhat prompt-triggered rather than purely random
#: sampling noise, which means a small retry budget is more likely to exhaust on
#: exactly the turns that need it most. Doubling the budget is a cheap hedge
#: either way: on a true one-off it costs nothing extra (the first retry already
#: succeeds), and on a semi-systematic case it roughly doubles the chance of
#: escaping before the whole run stops. It does not fix the underlying cause —
#: Ollama's own tool-call channel detection on gpt-oss's harmony format — which
#: would need a server-side change to address properly.
LLM_ERROR_RETRIES = 4


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
    #: Transient API-level failures (see `LLM_ERROR_RETRIES`) that were re-asked
    #: and recovered from, rather than immediately ending the loop.
    llm_error_retries: int = 0
    #: Catalog entries this run could not support, mapped to the reason. Recorded
    #: so a row states which action space produced it.
    withheld_tools: Dict[str, str] = field(default_factory=dict)
    #: Scoring weights actually used to rank the history at the end of the loop.
    #: Filled only with `config.score_preset == "auto_loo"`; the choice is made by
    #: leave-one-window-out on validation (see `ReactState.auto_score_weights`).
    score_weights: Dict[str, float] = field(default_factory=dict)
    score_weights_name: str = ""
    score_weights_held_out: Optional[float] = None
    score_weights_skipped: str = ""
    #: Per-turn record that would bloat the CSV (full raw response, full thought,
    #: Ollama token counters, think-block size); written to the artifacts only.
    step_details: List[Dict[str, Any]] = field(default_factory=list)
    #: LLM time of a turn that produced no trajectory entry (API failure that ended the loop).
    llm_failed_turn_s: float = 0.0
    #: Raw model output for every turn the parser could not read. Kept out of the
    #: CSV, which it would bloat, and written to the per-series artifacts instead —
    #: that is where you look when a model keeps missing the output format and you
    #: need to see what it actually said.
    parse_failures: List[Dict[str, Any]] = field(default_factory=list)
    #: ISO timestamps of the whole decision, for the trace.
    started_at: str = ""
    finished_at: str = ""
    #: Timing metrics derived from the turn records (time to first best, leader
    #: changes, LLM/tool split). Filled by `_finalize_trace`; artifacts only.
    derived: Dict[str, Any] = field(default_factory=dict)

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
            "llm_error_retries": self.llm_error_retries,
            "withheld_tools": dict(self.withheld_tools),
            "score_weights": dict(self.score_weights),
            "score_weights_name": self.score_weights_name,
            "score_weights_held_out": self.score_weights_held_out,
            "score_weights_skipped": self.score_weights_skipped,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "derived": dict(self.derived),
            "elapsed_s": round(self.elapsed_s, 2),
            "llm_call_s_total": round(
                sum(e.get("llm_call_s", 0.0) for e in self.trajectory) + self.llm_failed_turn_s, 4),
            "tool_exec_s_total": round(sum(e.get("tool_exec_s", 0.0) for e in self.trajectory), 4),
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
    result.started_at = _utcnow()
    result.llm_model = getattr(client, "name", "none") if client else "none"

    if not state.attempts:
        raise RuntimeError(
            "the attempt history is empty: run pool.seed_baselines() before the loop, "
            "otherwise there is no floor to guarantee the result against"
        )

    # With `score_preset="auto_loo"` the weights are chosen once, from the seeded
    # history, before the loop opens: exploration, early stopping and the final
    # ranking all then use the same score. Validation only; see
    # `ReactState.auto_score_weights`.
    _apply_auto_score_weights(state, config, result)
    result.final_attempt = state.best_attempt()

    if client is None or skip_reason:
        result.stop_reason = skip_reason or "no_llm_client"
        result.justification = _fallback_justification(state, series_card, pool_card)
        result.elapsed_s = time.perf_counter() - started
        _finalize_trace(state, result, started)
        result.tools = tools_called_summary(state)
        return result

    withheld = withheld_tools(config, state.n_windows, state=state)
    result.withheld_tools = dict(withheld)
    scoring_budget = bool(getattr(config, "scoring_budget", False))
    no_accept = bool(
        getattr(config, "no_early_stop", False) or scoring_budget
    )
    if getattr(config, "prompt_v2", False):
        system = P.build_system_prompt_v2(
            include_history_rules=config.show_attempt_history, withheld_tools=withheld,
            prompt_format=config.prompt_format,
            reorder_weight_tools=config.reorder_weight_tools,
            no_early_stop=no_accept,
            direct_models=scoring_budget,
        )
    else:
        system = P.build_system_prompt(
            include_history_rules=config.show_attempt_history, withheld_tools=withheld,
            prompt_format=config.prompt_format,
            reorder_weight_tools=config.reorder_weight_tools,
            no_early_stop=no_accept,
            direct_models=scoring_budget,
        )
    scratchpad: List[Dict[str, Any]] = []
    last_observation: Optional[Dict[str, Any]] = None
    best_score = _score(state.best_attempt())
    stale = 0
    max_iterations = max(1, int(config.max_iterations))
    seeded_attempts = len(state.attempts)
    target_scores = max_iterations
    last_duplicate = False
    total_turns = max_iterations
    if scoring_budget:
        total_turns = max(
            1,
            int(getattr(config, "max_total_turns", 0) or 2 * max_iterations),
        )

    for iteration in range(1, total_turns + 1):
        result.iterations_used = iteration
        user = P.build_turn_prompt(
            state=state,
            series_card=series_card,
            pool_card=pool_card,
            scratchpad=scratchpad,
            iteration=iteration,
            max_iterations=total_turns,
            last_observation=last_observation,
            show_history=config.show_attempt_history,
            show_rationales=config.show_attempt_rationales,
            diagnosis=diagnosis,
            prompt_format=config.prompt_format,
            leader=(
                _leader_diagnostics(state)
                if getattr(config, "prompt_v2", False)
                else None
            ),
            untried=P.untried_candidates(state, pool_card),
            repeat_notice=last_duplicate,
            no_accept=no_accept,
        )

        # Two different things can go wrong asking for one turn, and both are
        # failed generations, not decisions — retrying costs a call; spending one
        # of `max_iterations` on either would charge the series for a glitch:
        #   - EMPTY answer: gpt-oss emits a <think> block and sometimes stops
        #     before writing anything after it.
        #   - LLMError: the API call itself failed (see `LLM_ERROR_RETRIES` for
        #     the observed cause). Unlike the empty case there is no `raw` text to
        #     fall back to, so exhausting these retries ends the whole loop, same
        #     as before this retry existed.
        # A malformed but non-empty PARSED answer is different from both: that one
        # goes back to the agent as an observation, because it can learn from it.
        turn_started_at = _utcnow()
        raw = ""
        step = None
        llm_error: Optional[LLMError] = None
        empty_left = EMPTY_RESPONSE_RETRIES
        error_left = LLM_ERROR_RETRIES
        if hasattr(client, "seed_offset"):
            client.seed_offset = 0
        llm_call_s = 0.0
        llm_calls = 0
        llm_meta: Dict[str, Any] = {}
        empty_metas: List[Dict[str, Any]] = []
        while True:
            t_call = time.perf_counter()
            llm_calls += 1
            try:
                raw = client.complete(system, user)
                llm_call_s += time.perf_counter() - t_call
                llm_meta = dict(getattr(client, "last_meta", None) or {})
                llm_error = None
            except LLMError as exc:
                llm_call_s += time.perf_counter() - t_call
                llm_error = exc
                if error_left <= 0:
                    result.errors.append(str(exc))
                    result.stop_reason = "llm_error"
                    step = None
                    break
                error_left -= 1
                result.llm_error_retries += 1
                if hasattr(client, "seed_offset"):
                    # Same seed + same prompt fails the same way (the "error parsing
                    # tool call" 500 is deterministic), so retries must vary it.
                    client.seed_offset = (LLM_ERROR_RETRIES - error_left) + (EMPTY_RESPONSE_RETRIES - empty_left)
                result.errors.append(
                    f"iteration {iteration}: transient LLM error, retrying "
                    f"({LLM_ERROR_RETRIES - error_left}/{LLM_ERROR_RETRIES}): {exc}"
                )
                continue

            step = parse_agent_step(raw)
            if step.ok or step.parse_error != "empty response":
                break
            if empty_left <= 0:
                break
            empty_left -= 1
            result.empty_responses += 1
            if hasattr(client, "seed_offset"):
                client.seed_offset = (LLM_ERROR_RETRIES - error_left) + (EMPTY_RESPONSE_RETRIES - empty_left)
            empty_metas.append(dict(llm_meta))
            why = ", ".join(
                f"{k}={llm_meta[k]}" for k in ("done_reason", "eval_count", "thinking_chars", "num_ctx")
                if k in llm_meta
            )
            result.errors.append(
                f"iteration {iteration}: empty response, retrying "
                f"({EMPTY_RESPONSE_RETRIES - empty_left}/{EMPTY_RESPONSE_RETRIES})"
                + (f" [{why}]" if why else "")
            )
        if llm_error is not None:
            result.llm_failed_turn_s += llm_call_s
            # A turn that died on the API still belongs in the trace: otherwise the
            # artifacts would show a gap where the loop actually stopped.
            result.step_details.append({
                "iteration": iteration,
                "action": "",
                "llm_error": str(llm_error),
                "llm_call_s": round(llm_call_s, 4),
                "turn_finished_at": _utcnow(),
            })
            break
        if step is None:
            break
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "thought": _clip(step.thought, 600),
            "action": step.action or "",
            "action_args": step.action_input,
            "observation_summary": "",
            "llm_call_s": round(llm_call_s, 4),
            "tool_exec_s": 0.0,
        }
        think_text = split_think(raw)[0]
        detail: Dict[str, Any] = {
            "iteration": iteration,
            "turn_started_at": turn_started_at,
            "action": step.action or "",
            "llm_call_s": round(llm_call_s, 4),
            "llm_calls_in_turn": llm_calls,
            "thought_full": step.thought,
            "raw_response": raw,
            "has_think_block": bool(think_text),
            "think_chars": len(think_text),
        }
        detail.update({f"ollama_{k}": v for k, v in llm_meta.items()})
        if empty_metas:
            detail["empty_retries_meta"] = empty_metas
        result.step_details.append(detail)

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
            detail.update({
                "ok": False,
                "parse_error": step.parse_error,
                "observation": observation,
                "turn_finished_at": _utcnow(),
            })
            result.trajectory.append(entry)
            scratchpad.append(entry)
            last_observation = observation
            _emit(on_step, state, entry)
            continue

        # ── terminal action ──────────────────────────────────────────────────
        if step.action.strip().lower() == TERMINAL_ACTION and no_accept:
            reason = (
                "no_early_stop" if getattr(config, "no_early_stop", False)
                else "scoring_budget"
            )
            result.errors.append(
                f"iteration {iteration}: accept disabled by {reason}; "
                "use the remaining turns and let the final history decide"
            )
            entry["observation_summary"] = f"ERROR accept: disabled by {reason}"
            last_observation = {"error": "accept_disabled", "detail": reason}
            state.log_tool(TERMINAL_ACTION, step.action_input, ok=False,
                           error=f"disabled by {reason}", kind="terminal_disabled")
            detail.update({
                "ok": False,
                "accept_error": f"disabled by {reason}",
                "observation": last_observation,
                "turn_finished_at": _utcnow(),
            })
            result.trajectory.append(entry)
            scratchpad.append(entry)
            _emit(on_step, state, entry)
            continue

        if step.action.strip().lower() == TERMINAL_ACTION:
            accepted, confidence, justification, problem = _read_accept(state, step)
            if problem:
                entry["observation_summary"] = f"ERROR accept: {problem}"
                last_observation = {"error": "invalid_accept", "detail": problem}
                state.log_tool(TERMINAL_ACTION, step.action_input, ok=False,
                               error=problem, kind="invalid_argument")
                detail.update({
                    "ok": False,
                    "accept_error": problem,
                    "observation": last_observation,
                    "turn_finished_at": _utcnow(),
                })
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
            detail.update({
                "ok": True,
                "accepted_attempt_id": accepted.attempt_id,
                "confidence": confidence,
                "justification": justification,
                "observation": {
                    "accepted": accepted.attempt_id,
                    "confidence": confidence,
                },
                "best_before": _attempt_ref(state.best_attempt()),
                "turn_finished_at": _utcnow(),
            })
            result.trajectory.append(entry)
            _emit(on_step, state, entry)
            result.stop_reason = "agent_accepted"
            break

        # ── tool call ────────────────────────────────────────────────────────
        args = dict(step.action_input)
        if step.action == "evaluate_strategy":
            args.setdefault("rationale", step.thought or "")
            args["iteration"] = iteration

        best_before = state.best_attempt()
        t_tool = time.perf_counter()
        ok, observation = call_tool(state, step.action, args, withheld=withheld)
        tool_exec_s = round(time.perf_counter() - t_tool, 4)
        best_after = state.best_attempt()
        entry["tool_exec_s"] = tool_exec_s
        entry["observation_summary"] = P.summarize_observation(step.action, ok, observation)
        detail.update({
            "ok": bool(ok),
            "observation": observation,
            "tool_exec_s": tool_exec_s,
            "attempt_id": observation.get("id") if (ok and step.action == "evaluate_strategy") else None,
            "attempt_rank": observation.get("rank") if (ok and step.action == "evaluate_strategy") else None,
            "best_before": _attempt_ref(best_before),
            "best_after": _attempt_ref(best_after),
            "turn_finished_at": _utcnow(),
        })
        result.trajectory.append(entry)
        scratchpad.append(entry)
        last_observation = observation
        _emit(on_step, state, entry)

        if not ok:
            result.errors.append(f"iteration {iteration}: {observation.get('detail', '')}")
            continue

        if step.action == "evaluate_strategy":
            last_duplicate = bool(observation.get("already_tested"))

        # ── early stopping, counted on proposals only ────────────────────────
        if step.action == "evaluate_strategy":
            current = _score(state.best_attempt())
            improved = _improved(current, best_score, config.min_improvement)
            if improved:
                best_score = current
                stale = 0
            else:
                stale += 1
            if not no_accept and stale >= max(1, int(config.early_stop_patience)):
                result.early_stopped = True
                result.stop_reason = (
                    f"no improvement in {stale} consecutive proposals"
                )
                break
            if scoring_budget:
                scores_used = max(0, len(state.attempts) - seeded_attempts)
                if scores_used >= target_scores:
                    result.stop_reason = "scoring_budget_exhausted"
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
    _finalize_trace(state, result, started)
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


def _utcnow() -> str:
    """ISO timestamp for the trace; seconds are enough for a per-series record."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _leader_diagnostics(state: ReactState) -> Dict[str, Any]:
    """Deterministic answer to "how safe is stopping now?".

    Reuses `state.selection_confidence()`: it compares the leader against the
    first genuinely different runner-up (numerical twins are skipped) with the
    paired bootstrap and Diebold-Mariano. Everything in it is validation-only.
    """
    best = state.best_attempt()
    if best is None:
        return {}
    confidence = state.selection_confidence()
    return {
        "leader_id": best.attempt_id,
        "leader_strategy": _label(best),
        "leader_origin": best.origin,
        "leader_score": round(float(best.score), 4) if np.isfinite(best.score) else None,
        "runner_up_margin": confidence.get("margin"),
        "verdict": confidence.get("verdict"),
        "note": (
            "a small margin or 'indistinguishable' means the runner-up is not "
            "statistically worse; accepting the leader is usually right"
        ),
    }


def _attempt_ref(attempt: Optional[Attempt]) -> Optional[Dict[str, Any]]:
    """Compact reference to the current leader, for the per-turn trace."""
    if attempt is None:
        return None
    return {
        "id": attempt.attempt_id,
        "score": round(float(attempt.score), 6) if attempt.score == attempt.score else None,
        "origin": attempt.origin,
        "strategy": _label(attempt),
    }


def _finalize_trace(state: ReactState, result: "ReactResult", started: float) -> None:
    """Close the trace with the finish timestamp and the derived timing metrics.

    Everything here is computed from `result.step_details`, which only contains
    what the loop already recorded: LLM time per turn, tool time, the leader
    before and after each call. The test window is not involved in any of it.
    """
    result.finished_at = _utcnow()
    details = result.step_details
    llm_s = sum(float(d.get("llm_call_s") or 0.0) for d in details)
    tool_s = sum(float(d.get("tool_exec_s") or 0.0) for d in details)
    scoring = sum(1 for d in details if d.get("action") == "evaluate_strategy" and d.get("ok"))
    calls = sum(1 for d in details if d.get("action"))

    # Leader sequence, in turn order: when the best attempt changed and at what
    # point of the decision (accumulated LLM + tool time).
    leader_changes: List[Dict[str, Any]] = []
    current_id = result.final_attempt.attempt_id if result.final_attempt else None
    elapsed = 0.0
    for d in details:
        elapsed += float(d.get("llm_call_s") or 0.0) + float(d.get("tool_exec_s") or 0.0)
        after = d.get("best_after") or {}
        attempt_id = after.get("id")
        if attempt_id and attempt_id != current_id:
            current_id = attempt_id
            leader_changes.append({
                "iteration": d.get("iteration"),
                "attempt_id": attempt_id,
                "score": after.get("score"),
                "at_s": round(elapsed, 3),
            })

    total = llm_s + tool_s
    result.derived = {
        "wall_clock_s": round(time.perf_counter() - started, 3),
        "llm_s": round(llm_s, 3),
        "tool_s": round(tool_s, 3),
        "llm_time_share": round(llm_s / total, 4) if total > 0 else None,
        "turns": len(details),
        "tool_calls": calls,
        "scoring_calls": scoring,
        "non_scoring_calls": max(0, calls - scoring),
        "parse_failures": len(result.parse_failures),
        "empty_responses": result.empty_responses,
        "llm_error_retries": result.llm_error_retries,
        "best_changes": len(leader_changes),
        "time_to_first_best_s": leader_changes[0]["at_s"] if leader_changes else None,
        "time_after_last_improvement_s": (
            round(elapsed - leader_changes[-1]["at_s"], 3) if leader_changes else None
        ),
        "leader_changes": leader_changes,
        # Runner-up margin and verdict, so the trace says how defensible the pick
        # was without anyone having to recompute it downstream.
        "selection": state.selection_confidence(),
    }


def _score(attempt: Optional[Attempt]) -> float:
    if attempt is None or not np.isfinite(attempt.score):
        return float("inf")
    return float(attempt.score)


def _apply_auto_score_weights(state: ReactState, config: ReactConfig, result: "ReactResult") -> None:
    """Per-series choice of the scoring weights, when `score_preset == "auto_loo"`.

    Called once, right after the seeds and before the loop opens: the harness
    picks the preset favoured by the leave-one-window-out protocol on the three
    validation windows, applies it, and every evaluation from then on is scored
    under it. No test data is read: the decision uses `Attempt.per_window` and
    the mean anchor only. Recorded on the result so the artifacts and the CSV can
    state which weights were actually used.
    """
    if str(getattr(config, "score_preset", "")) != "auto_loo":
        return
    choice = state.auto_score_weights(getattr(config, "score_auto_candidates", None))
    state.apply_score_weights(choice["weights"])
    result.score_weights = dict(choice["weights"])
    result.score_weights_name = str(choice.get("name", ""))
    held_out = choice.get("held_out")
    result.score_weights_held_out = float(held_out) if isinstance(held_out, (int, float)) else None
    result.score_weights_skipped = str(choice.get("skipped", ""))


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
