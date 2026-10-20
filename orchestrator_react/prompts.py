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
import random

import numpy as np
from typing import Any, Dict, List, Optional, Sequence

from orchestrator_react.config import FINAL_SCORE_PRESETS, SCORE_PRESETS
from orchestrator_react.registry import TERMINAL_ACTION, describe_tools
from orchestrator_react.state import FULL_POOL, ReactState
from orchestrator_react import metrics as M


def _compact(payload: Any, limit: int = 1800) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(text) <= limit:
        return text
    obj = payload

    def _shrink(o: Any) -> Any:
        if isinstance(o, dict):
            keys = list(o.keys())
            if len(keys) > 1:
                return {k: o[k] for k in keys[: max(1, len(keys) - 1)]}
            k = keys[0]
            return {k: _shrink(o[k])}
        if isinstance(o, list) and len(o) > 1:
            return o[: max(1, len(o) - 1)]
        if isinstance(o, str) and len(o) > 40:
            return o[:40] + "..."
        return o

    shrunk = obj
    for _ in range(200):
        text = json.dumps(shrunk, ensure_ascii=False, default=str, separators=(",", ":"))
        if len(text) <= limit:
            return text + " [truncated: some fields omitted]"
        new_shrunk = _shrink(shrunk)
        if new_shrunk == shrunk:
            break
        shrunk = new_shrunk
    return json.dumps({"note": "payload too large to include"}, separators=(",", ":"))


PROMPT_FORMATS = ("text", "xml")

_SYSTEM_TAGS = (
    ("AVAILABLE TOOLS", "tools"),
    ("OUTPUT FORMAT", "output_format"),
    ("HOW TO WORK", "workflow"),
    ("EVALUATING A STRATEGY", "evaluate_strategy"),
    ("A TYPICAL SEQUENCE", "typical_sequence"),
    ("THE WEIGHT METHODS", "weight_methods"),
    ("RULES", "rules"),
    # Tags used by the v2 prompt (`build_system_prompt_v2`).
    ("WHAT YOU RECEIVE", "context"),
    ("PROCEDURE", "procedure"),
    ("WORKED EXAMPLES", "examples"),
    ("WHEN TO STOP", "stopping"),
)


def _check_format(prompt_format: str) -> None:
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}, got {prompt_format!r}")


def _public_tools(
    withheld_tools: Optional[Dict[str, str]],
    reorder_weight_tools: bool,
    no_early_stop: bool,
    allow_agent_score_preset: bool,
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool in describe_tools(withheld_tools, reorder_weight_tools):
        if no_early_stop and tool["name"] == TERMINAL_ACTION:
            continue
        tool = dict(tool)
        if tool["name"] == "evaluate_strategy" and not allow_agent_score_preset:
            tool["args"] = [a for a in tool["args"] if not a.startswith("score_preset")]
        tools.append(tool)
    return tools


def _system_as_xml(lines: List[str], merge_first_two: bool = True) -> str:
    """Wraps each blank-line-separated block of the system prompt in a tag.

    Wording is untouched: only the delimiters change, so the `text` and `xml`
    variants are an A/B on structure alone. `merge_first_two` joins the first
    two blocks into the `role` tag, for prompts whose role statement spans two
    paragraphs.
    """
    blocks: List[List[str]] = [[]]
    for line in lines:
        if line == "":
            blocks.append([])
        else:
            blocks[-1].append(line)
    blocks = [b for b in blocks if b]
    if merge_first_two and len(blocks) >= 2:
        blocks[:2] = [blocks[0] + [""] + blocks[1]]  # the role statement spans two paragraphs
    out: List[str] = []
    for i, block in enumerate(blocks):
        tag = "role" if i == 0 else "section"
        for prefix, name in _SYSTEM_TAGS:
            if block[0].startswith(prefix):
                tag = name
                break
        out += [f"<{tag}>", *block, f"</{tag}>"]
    return "\n".join(out)


def build_system_prompt(
    include_history_rules: bool = True,
    withheld_tools: Optional[Dict[str, str]] = None,
    prompt_format: str = "text",
    reorder_weight_tools: bool = False,
    no_early_stop: bool = False,
    direct_models: bool = False,
    score_window_mode: str = "mean",
    multi_preset_final: bool = False,
    no_seed_attempts: bool = False,
    allow_agent_score_preset: bool = True,
    score_preset: str = "balanced",
    allow_best_single: bool = True,
) -> str:
    """System prompt: the role, the closed action space and the output contract.

    `withheld_tools` drops entries from the advertised catalog. A tool the run
    cannot support is never shown, so the agent cannot waste an iteration on it.
    """
    catalog = "\n".join(
        f"  {t['name']}({', '.join(t['args'])}) - {t['description']}"
        for t in _public_tools(
            withheld_tools, reorder_weight_tools, no_early_stop, allow_agent_score_preset
        )
    )
    no_combine_mean = bool(withheld_tools) and "combine_mean" in withheld_tools
    weight_pair = (
        "weights_softmax_neg_error / weights_inverse_error"
        if reorder_weight_tools else "weights_inverse_error / weights_softmax_neg_error"
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
        *([] if no_early_stop else [
            f"  {TERMINAL_ACTION}(attempt_id, confidence, justification) - stop and take an attempt",
        ]),
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
        *([
            "4. Read the result against the attempt history; accept is disabled, so",
            "   use the turn to test another hypothesis.",
        ] if no_early_stop else [
            "4. Read the result against the attempt history and decide: test another",
            f"   hypothesis, or {TERMINAL_ACTION} the best attempt.",
        ]),
        "",
        "EVALUATING A STRATEGY - one call, flat arguments:",
        '  Action Input: {"combine": "mean", "pool": "pool_full", "rationale": "..."}',
        '  Action Input: {"combine": "median", "pool": "pool1", "rationale": "..."}',
        '  Action Input: {"combine": "trimmed_mean", "pool": "pool1", "trim_pct": 0.2}',
        '  Action Input: {"combine": "weighted", "pool": "pool1", "weights": "w1"}',
        *([
            '  Action Input: {"combine": "best_single", "model": "<a model name>"}',
        ] if allow_best_single else []),
        '  Action Input: {"combine": "dba", "pool": "pool1"}',
        *(
            [
                "  You do NOT need combine_trimmed_mean / combine_dba first. They only",
                "  build the same object, so going through them costs you an iteration",
                "  for nothing.",
            ]
            if no_combine_mean else [
                "  You do NOT need combine_* first. It only builds the same object, so going",
                "  through it costs you an iteration for nothing.",
            ]
        ),
        "",
        *([
            "A TYPICAL SEQUENCE (one evaluate per turn):",
            '  evaluate_strategy {"combine": "mean", "models": ["..."], "rationale": "..."}',
            *([
                '  evaluate_strategy {"combine": "best_single", "model": "<model>"}',
            ] if allow_best_single else []),
            "",
        ] if direct_models else [
            "A TYPICAL SEQUENCE (the weights_* step is a CHOICE, not a fixed step):",
            '  select_stable      {"k": 5}                  -> pool1',
            '  weights_<method>   {"pool": "pool1"}         -> w1',
            '  evaluate_strategy  {"combine": "weighted", "pool": "pool1", "weights": "w1"}',
            *([] if no_early_stop else [
                f'  {TERMINAL_ACTION}  {{"attempt_id": "aN", "confidence": 0.7, "justification": "..."}}',
            ]),
            "",
        ]),
        "THE WEIGHT METHODS READ DIFFERENT SIGNALS - they are not interchangeable:",
        f"  {weight_pair} - each model's AVERAGE",
        "     error over the windows. Similar to each other; if one gives you",
        "     concentration near 0, so will the other.",
        "  weights_error_trend      - where each model's error is HEADING, read from",
        "     the pointwise grid, not from the average. Separates a model that is",
        "     improving from one that is degrading at the same mean error.",
        "  weights_pooled_meta_model - learned across the OTHER series of this",
        "     dataset, from this series' shape. The only method whose signal does",
        "     not come from these three windows.",
        "",
        "RULES:",
        "- Prefer small structured comparisons over one sweeping decision. Pick a",
        "  subset, weight it, test it. Do not try to reason about every model at once.",
        "- 'weights' is either an existing handle or a known method name (for",
        "  example inverse_error); evaluate_strategy resolves it. You never see",
        "  the numbers, and you do not need to.",
        "- A handle is only valid with the pool it was computed on.",
        "- Repeating a strategy already in the history wastes an iteration.",
        "- Computing weights scores NOTHING on its own: every weights_* call must be",
        "  followed by evaluate_strategy. Do not spend your last turns building",
        "  handles you will not evaluate.",
        "- Handles start empty for every series. Never assume w1 or pool1 exists:",
        "  create it in this run before you refer to it.",
        "- The seeded baselines are part of the history and are a reference, not",
        "  a required starting point. Test whichever combination the series",
        "  characteristics and the evidence justify.",
        "- If a weights_* handle comes back with concentration near 0, the weights",
        "  are effectively UNIFORM, and that strategy gives the same forecasts as the",
        "  plain mean of the same pool. Another error-based weights_* method on the",
        "  SAME pool will usually do the same; consider changing the pool or the kind",
        "  of combination.",
        "- If an observation says SAME forecasts as an earlier attempt, that turn",
        "  bought nothing. Do not follow it with a variation of the same idea.",
        "- If a call is rejected, read the error: it lists the arguments that ARE",
        "  accepted. Do not retry the same shape.",
        *([] if no_early_stop else [
            f"- When you {TERMINAL_ACTION}, justification must explain the choice in terms of",
            "  OBSERVABLE SERIES CHARACTERISTICS, not just 'it had the lowest error'.",
            "  Example: 'ranking is unstable across windows (tau=0.17) and the models are",
            "  redundant, so a robust equal-weight combination of the stable subset beats",
            "  fitted weights, which would overfit 3 windows.'",
        ]),
    ]
    if include_history_rules:
        rules += [
            "- The attempt history is ranked best to worst and is given to you every turn.",
            "  Use it: if two attempts differ only slightly, look for a different KIND of",
            "  strategy rather than tuning the winner.",
        ]
    if no_early_stop:
        rules += [
            "- ACCEPT IS DISABLED in this run. Do not call accept. Use every turn",
            "  to test a genuinely different combination; the final strategy is",
            "  chosen from the full attempt history after the iteration budget.",
            "- One tested approach = one evaluate_strategy call. select_*/prune_*/",
            "  weights_* are preparation, not an approach; preparation must be scored",
            "  on the next turn. The seeds are already in the history; repeating them",
            "  adds nothing.",
            "- The scoring budget counts evaluate_strategy calls; preparation does not",
            "  consume it. A separate cap on total turns also applies.",
            "- The last validation window is the closest to the blind test; it is a",
            "  useful signal, but do not treat it as a rule.",
            "- If an observation says already_tested, change the model subset or the",
            "  combination next; do not resubmit the same call.",
            *([
                "- For best_single, only 'model' matters. Changing the pool/models is a",
                "  no-op; change the model, not the pool.",
            ] if allow_best_single else []),
            "- Use exact model names from the pool; weights must be a known method",
            "  name or an existing handle. Do not call accept.",
        ]
    if direct_models and multi_preset_final:
        rules += [
            "- DIRECT MODE: the only action is evaluate_strategy. Each turn pass",
            "  'models' (the subset) and the combination in the same call; the",
            "  harness builds the pool/weights. There are no select/weights actions.",
            "- MULTI-PRESET FINAL: every evaluate_strategy returns three scores for",
            "  the same attempt: " + ", ".join(FINAL_SCORE_PRESETS) + ".",
            f"  The window aggregation is fixed to '{score_window_mode}'.",
            "- Do NOT pass score_preset during the loop. After the scoring budget",
            "  you choose ONE of the three criteria; the best attempt under it is applied.",
        ]
    elif direct_models and allow_agent_score_preset:
        rules += [
            "- DIRECT MODE: the only action is evaluate_strategy. Each turn pass",
            "  'models' (the subset) and the combination in the same call; the",
            "  harness builds the pool/weights. There are no select/weights actions.",
            "- You may pass score_preset=<one of: " + ", ".join(SCORE_PRESETS) + ">",
            "  to choose how SMAPE/POCID are weighted for this series.",
        ]
    elif direct_models:
        rules += [
            "- DIRECT MODE: the only action is evaluate_strategy. Each turn pass",
            "  'models' (the subset) and the combination in the same call; the",
            "  harness builds the pool/weights. There are no select/weights actions.",
            f"- The scoring criterion is FIXED to '{score_preset}' (normalized SMAPE).",
            "  POCID is reported separately and does not affect the ranking. Do not pass",
            "  score_preset.",
            *([] if allow_best_single else [
                "- COMBINATION-ONLY: best_single is disabled; combine at least two models.",
            ]),
            *([
                "- CLASSIC STARTING POINTS (optional, not a required order):",
                "  mean(pool_full), median(pool_full), dba(pool_full),",
                "  trimmed_mean(pool_full, trim_pct=0.2),",
                "  weighted(pool_full, weights='inverse_error').",
                "  They are simple references; use or ignore them freely.",
            ] if no_seed_attempts else []),
        ]
    _check_format(prompt_format)
    if prompt_format == "xml":
        return _system_as_xml(rules)
    return "\n".join(rules)


def build_system_prompt_v2(
    include_history_rules: bool = True,
    withheld_tools: Optional[Dict[str, str]] = None,
    prompt_format: str = "xml",
    reorder_weight_tools: bool = False,
    no_early_stop: bool = False,
    direct_models: bool = False,
    score_window_mode: str = "mean",
    multi_preset_final: bool = False,
    no_seed_attempts: bool = False,
    allow_agent_score_preset: bool = True,
    score_preset: str = "balanced",
    allow_best_single: bool = True,
) -> str:
    """Structured system prompt: procedure, grouped catalog, worked examples.

    Keeps the closed action space and the three-line output contract. The
    organisation is deliberate: a short four-step procedure, the catalog grouped
    by phase with each entry's trigger in its own description, two worked
    examples of different KINDS of strategy (robust and selective), and an
    explicit stop rule that reads the LEADER DIAGNOSTICS block. Local models
    follow explicit structure and concrete examples better than dense prose.
    """
    tools = _public_tools(
        withheld_tools, reorder_weight_tools, no_early_stop, allow_agent_score_preset
    )
    groups: Dict[str, List[str]] = {
        "diagnostics": [], "pool": [], "weights": [], "combination": [], "validation": [],
    }
    for tool in tools:
        name = tool["name"]
        if name.startswith("select_"):
            key = "pool"
        elif name.startswith("weights_"):
            key = "weights"
        elif name.startswith("combine_"):
            key = "combination"
        elif name in ("evaluate_strategy", "sanity_check", "list_attempts"):
            key = "validation"
        else:
            key = "diagnostics"
        groups[key].append(f"  {name}({', '.join(tool['args'])}) - {tool['description']}")

    catalog: List[str] = []
    for key, label in (
        ("diagnostics", "DIAGNOSTICS"),
        ("pool", "POOL SELECTION"),
        ("weights", "WEIGHTS"),
        ("combination", "COMBINATION"),
        ("validation", "VALIDATION"),
    ):
        if groups[key]:
            catalog.append(f"{label}:")
            catalog.extend(groups[key])

    # The catalog lines themselves say WHEN to use each tool: every advertised
    # tool's first docstring line starts with "Use when ...". That keeps a single
    # source of truth for the trigger (Anthropic's tool-writing guidance) instead
    # of a separate block duplicating the descriptions.
    lines: List[str] = [
        "You are a forecast COMBINATION AGENT.",
        "",
        "A pool of already-trained forecasting models produced predictions for one",
        "time series. Your job is to decide HOW TO COMBINE them for the blind test",
        "window. You never write forecast numbers and you never write weights: you",
        "call tools, and the tools compute. Every strategy is scored by the harness",
        "on validation windows, and the strategy applied at the end is the",
        "best-scoring one of the whole history, so testing a hypothesis cannot",
        "make the final validation score worse.",
        "",
        "WHAT YOU RECEIVE EVERY TURN:",
        "  - SERIES PROFILE: shape of the series (trend, seasonality, stationarity, outliers).",
        "  - MODEL POOL: error table, ranking stability across windows, correlated-error groups.",
        "  - ATTEMPT HISTORY: every strategy tested so far, best first, with its score.",
        "  - LEADER DIAGNOSTICS: the current leader, how far the runner-up is, and whether",
        "    that gap is statistically distinguishable.",
        "  - FORECAST SHAPE: normalized last-validation actual + top-model forecasts;",
        "    blind-test model forecasts and agreement. Test actual is never shown.",
        *([
            "  - SEEDED POOL HANDLES: optional Phase-2 references; no new handles",
            "    are created in direct mode.",
        ] if direct_models else [
            "  - HANDLES: pools and weight vectors created in this run, with concentration.",
        ]),
        "",
        *([
            "DIRECT COMBINATION MODE:",
            "  The only action is evaluate_strategy. Each turn is ONE tested combination.",
            "  Pass 'models' (the subset) and 'combine' in the same call; the harness",
            "  builds the pool and any weights internally. There are no separate",
            "  pool/weight/diagnostic actions. Phase-2 pool handles such as",
            "  'pool_full' may also be passed as 'pool'.",
            "  Examples:",
            '    evaluate_strategy {"combine":"trimmed_mean","models":["ARIMA","ETS","rf"],"trim_pct":0.2}',
            *([
                '    evaluate_strategy {"combine":"best_single","model":"rf"}',
            ] if allow_best_single else []),
            '    evaluate_strategy {"combine":"mean","models":["ARIMA","ETS","rf","catboost"]}',
            '    evaluate_strategy {"combine":"weighted","models":["..."],"weights":"inverse_error"}',
            *([
                "  MULTI-PRESET FINAL: every evaluate_strategy returns the same",
                "  attempt scored under three criteria:",
                f"    {', '.join(FINAL_SCORE_PRESETS)}.",
                f"  The window aggregation is fixed to '{score_window_mode}'.",
                "  Do NOT pass score_preset during the loop. After the budget you",
                "  choose ONE criterion; the best attempt under it is applied.",
                "",
            ] if multi_preset_final else ([
                "  You may also choose a scoring preset on the same call:",
                "    score_preset=<one of the presets below>.",
                f"  Score presets: {', '.join(SCORE_PRESETS)}. The harness aggregates",
                f"  the validation windows by '{score_window_mode}'.",
                "  Pick the preset that matches the series; changing it re-scores the history.",
                "",
            ] if allow_agent_score_preset else [
                f"  The scoring criterion is FIXED to '{score_preset}' (normalized SMAPE).",
                "  The 3 windows are aggregated as 0.2/0.3/0.5, the most recent highest.",
                "  POCID is reported separately and does not affect the ranking. Do not",
                "  pass score_preset.",
                *([] if allow_best_single else [
                    "  COMBINATION-ONLY: best_single is disabled; combine at least two models.",
                ]),
                "",
            ])),
        ] if direct_models else []),
        "PROCEDURE:",
        "  1. Read the diagnostics and form ONE hypothesis about this series.",
        *([
            "  2. Test one combination per turn with evaluate_strategy: pass the",
            "     model subset as 'models' and the combination; the harness builds",
            "     the pool/weights. No separate pool/weight action is available.",
        ] if direct_models else [
            "  2. Test it with evaluate_strategy, building a pool or weights handle first",
            "     only when the hypothesis needs a subset or a fitted weighting.",
        ]),
        "  3. Read the result against the attempt history and the leader diagnostics.",
        (
            "  4. Test a DIFFERENT kind of strategy; accept is disabled in this run."
            if no_early_stop else
            "  4. Either test a DIFFERENT kind of strategy, or accept the leader."
        ),
        "  If the leader is narrow (median, DBA, or a small pool) and the next",
        "  validation scores are close, a broad softmax/weighted over the active",
        "  pool is a useful comparison. This is a check, not a rule.",
        "  When choosing the model subset for a mean/weighted combination, individual",
        "  error is not the only criterion: models from different correlation groups",
        "  or forecast clusters can complement each other, and a slightly weaker",
        "  model can help. This is a hint, not a rule.",
        "",
        "CLASSIC STARTING POINTS (optional, not a required order):",
        "  mean(pool_full), median(pool_full), trimmed_mean(pool_full, trim_pct=0.2),",
        "  dba(pool_full), weighted(pool_full, weights='inverse_error'),",
        "  weighted(pool_full, weights='softmax_neg_error').",
        "  A simple baseline is often a useful reference; none is required.",
        "",
        "AVAILABLE TOOLS (this list is exhaustive - anything else is an error):",
        *catalog,
        *([] if no_early_stop else [
            f"  {TERMINAL_ACTION}(attempt_id, confidence, justification) - stop and take an attempt",
        ]),
        "",
        "OUTPUT FORMAT - exactly three lines, nothing else:",
        "Thought: <one or two sentences on what you want to learn or test next>",
        "Action: <tool name from the list above>",
        'Action Input: {"arg": value}',
        "",
        *([] if direct_models else [
            "WORKED EXAMPLES (the harness prints the observation back after each action):",
            "  Example 1, a baseline first:",
            "    Thought: a simple full-pool mean is a useful floor; I will compare later",
            "      hypotheses with it.",
            "    Action: evaluate_strategy",
            '    Action Input: {"combine": "mean", "pool": "pool_full"}',
            "  Example 2, a diversity check, when several models are plausible:",
            "    Thought: validation scores are close and the leader uses only a few",
            "      models, so a broad softmax over the active pool is a useful comparison.",
            "    Action: weights_softmax_neg_error",
            '    Action Input: {"pool": "pool_full"}',
            "    Action: evaluate_strategy",
            '    Action Input: {"combine": "weighted", "pool": "pool_full", "weights": "w1"}',
            "",
        ]),
        *([] if no_early_stop else [
            "WHEN TO STOP (read LEADER DIAGNOSTICS before deciding):",
            "  - If the leader is a SEED and the verdict is 'indistinguishable', accept it.",
            "    The runner-up is not statistically better and more turns rarely turn",
            "    noise into a real gain.",
            "  - If a strategy you tested leads with a 'separated' margin, accept it.",
            "  - If your last proposals returned 'SAME forecasts' notes or near-uniform",
            "    weights, change the POOL or the KIND of combination, not the weighting.",
            "",
        ]),
        *([
            "NO ACCEPT IN THIS RUN:",
            "  - Accept is disabled. Do not call accept.",
            "  - Use every turn to test a genuinely different combination.",
            "  - The final strategy is chosen from the full attempt history after the",
            "    iteration budget, so a failed test cannot make the final validation",
            "    score worse.",
            "",
            "BUDGET RULES (no accept):",
            *([
                "  - One tested approach = one evaluate_strategy call; pass the model",
                "    subset and the combination in the same call.",
            ] if direct_models else [
                "  - One tested approach = one evaluate_strategy call. select_*, prune_*,",
                "    and weights_* are preparation, not an approach.",
                "  - A pool/weight handle is optional and not a commitment: reuse it",
                "    while it still fits; build another when the subset changes.",
                "  - Never spend two turns in a row without a score; never build a handle",
                "    you will not score.",
            ]),
            *([
                "  - There are NO pre-evaluated seeded baselines in this run; your first",
                "    evaluate_strategy is the floor.",
            ] if no_seed_attempts else [
                "  - The seeded baselines are already scored and are part of the final",
                "    history. They are a reference, not a required starting point; test",
                "    whichever combination the series characteristics justify.",
            ]),
            "  - The diagnostics are already in the turn context; do not spend turns on them.",
            *([
                "  - The scoring budget counts evaluate_strategy calls; the separate",
                "    cap on total turns also applies.",
            ] if direct_models else [
                "  - The scoring budget counts evaluate_strategy calls; preparation actions",
                "    do not consume it. A separate cap on total turns also applies.",
            ]),
            "  - The last validation window is the closest to the blind test; it is a",
            "    useful signal, but do not treat it as a rule.",
            "  - If the observation says 'already_tested', change the model subset or",
            "    the combination next; do not resubmit the same call.",
            "  - In each Thought, name one SERIES DECISION CARD characteristic",
            "    (trend, seasonality, outliers, stationarity, ranking tau, or the",
            "    diagnosis hint) that motivates the test; if none does, say why.",
            "  - Use FORECAST SHAPE as evidence for model/pool selection: how the",
            "    top models tracked the last validation actual, and how much they",
            "    agree on the blind test. Do not treat it as a rule; the test actual",
            "    is not shown.",
            *([
                "  - For best_single, only 'model' matters. Changing the pool/models is a",
                "    no-op; change the model, not the pool.",
            ] if allow_best_single else []),
            "  - Use exact model names from the MODEL POOL; 'weights' is a known",
            "    method name (see RULES).",
            "",
        ] if no_early_stop else []),
        "RULES:",
        *([
            "- 'weights' must be a known method name (for example inverse_error);",
            "  evaluate_strategy resolves it. Phase-2 pool handles may be reused, but",
            "  the agent creates no new handles in direct mode.",
            "- Repeating a strategy already in the history wastes an iteration.",
        ] if direct_models else [
            "- 'weights' is either an existing handle or a known method name (for",
            "  example inverse_error); evaluate_strategy resolves it. You never see",
            "  the numbers, and you do not need to.",
            "- A handle is only valid with the pool it was computed on.",
            "- Repeating a strategy already in the history wastes an iteration.",
            "- Never assume w1 or pool1 exists: create it in this run before you refer to it.",
        ]),
        *([] if no_early_stop else [
            "- When you accept, justify the choice in terms of OBSERVABLE SERIES",
            "  CHARACTERISTICS (trend, seasonality, ranking stability, redundancy), not",
            "  just 'it had the lowest error'.",
        ]),
    ]
    if include_history_rules:
        lines += [
            "- The attempt history is ranked best to worst and is given every turn;",
            "  when two attempts differ only slightly, look for a different KIND of",
            "  strategy rather than tuning the winner.",
        ]
    _check_format(prompt_format)
    if prompt_format == "xml":
        return _system_as_xml(lines, merge_first_two=False)
    return "\n".join(lines)


def build_dataset_card(state: ReactState, top_n: int = 5) -> Optional[Dict[str, Any]]:
    """What worked on the OTHER series of this dataset, from validation only.

    Cross-series experience, computed by the Phase-2 pre-pass and held in
    `state.strategy_prior` (leave-one-series-out, so this series' own result is not
    in it). ADE and FFORMA both get this kind of dataset-wide view before they
    combine anything; the agent had none, and it showed — on the 182-series ANP run
    it used 4-5 of ~10 usable tools and reached for whichever one the prompt's
    worked example happened to name.

    Deliberately a RECOMMENDATION: the catalog stays fully open. Ranking strategies
    by dataset-level validation is informative but not decisive — the same ranking
    reverses between our two datasets — so it is offered as context, and the
    caution line says how far to trust it.
    """
    prior = state.strategy_prior
    if not prior:
        return None
    ranked = sorted(prior.items(), key=lambda kv: kv[1])
    return {
        "n_other_series": None,
        "best_on_this_dataset": [
            {"strategy": lbl, "mean_validation_score": round(float(v), 4)}
            for lbl, v in ranked[:top_n]
        ],
        "worst_on_this_dataset": [
            {"strategy": lbl, "mean_validation_score": round(float(v), 4)}
            for lbl, v in ranked[-2:]
        ],
        "how_to_use": (
            "validation-only averages over the OTHER series of this dataset. A "
            "strategy near the top is a good place to START, not a rule: which "
            "strategy wins is series-specific, and this ordering does not always "
            "hold. Your own attempt history for THIS series outranks it."
        ),
    }


def _untried_key(spec: Dict[str, Any]) -> str:
    if str(spec.get("combine", "")).strip().lower() == "best_single":
        spec = {"combine": "best_single", "model": spec.get("model")}
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)


def untried_candidates(
    state: ReactState,
    pool_card: Dict[str, Any],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Deterministic menu of valid combinations that are not in the history."""
    tried = {_untried_key(a.spec) for a in state.attempts}
    top = [
        str(item.get("model"))
        for item in ((pool_card.get("error_table") or {}).get("top") or [])
        if item.get("model") is not None
    ]
    candidates: List[Dict[str, Any]] = []
    seen_singles: set[str] = set()
    pools = sorted(state.pools, key=lambda h: (h != FULL_POOL, h))
    for pool in pools:
        idx = state.get_pool(pool)
        models = [state.model_names[int(i)] for i in idx]
        if not models:
            continue
        best = next((model for model in top if model in models), models[0])
        specs = [
            {"combine": "mean", "pool": pool},
            {"combine": "median", "pool": pool},
            {"combine": "trimmed_mean", "pool": pool, "trim_pct": 0.2},
            {"combine": "dba", "pool": pool, "dba_max_iter": 30},
            {"combine": "weighted", "pool": pool, "weights": "inverse_error"},
            *([{"combine": "weighted", "pool": pool, "weights": "softmax_neg_error"}]
              if "softmax_neg_error" in _available_weight_methods(state) else []),
        ]
        for handle, recipe in state.weights.items():
            if recipe.pool_handle == pool:
                specs.append({"combine": "weighted", "pool": pool, "weights": handle})
        if getattr(state.config, "allow_best_single", True):
            single = {"combine": "best_single", "model": best}
            if _untried_key(single) not in tried and best not in seen_singles:
                specs.append(single)
                seen_singles.add(best)
        random.Random(int(getattr(state, "dataset_index", 0) or 0) + 17).shuffle(specs)
        for spec in specs:
            key = _untried_key(spec)
            if key not in tried:
                candidates.append(spec)
                if len(candidates) >= limit:
                    return candidates
    return candidates


def _zscore(values: Any) -> List[float]:
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    center = float(np.mean(finite))
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale < 1e-12:
        return [round(float(v - center), 2) if np.isfinite(v) else None for v in arr]
    return [round(float((v - center) / scale), 2) if np.isfinite(v) else None for v in arr]


def _forecast_shape_card(state: ReactState, top_n: int = 3, threshold: float = 0.9) -> Dict[str, Any]:
    y_true = np.asarray(state.y_true, dtype=float)
    y_preds = np.asarray(state.y_preds, dtype=float)
    test = np.asarray(state.test_preds, dtype=float)
    if y_true.ndim != 2 or y_preds.ndim != 3 or test.ndim != 2:
        return {}
    names = [str(n) for n in state.model_names]
    last = y_true[-1]

    def window_smape(j: int, w: int) -> Optional[float]:
        try:
            v = float(M.smape(y_true[w], y_preds[w, j]))
            return round(v, 4) if np.isfinite(v) else None
        except Exception:
            return None

    def corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
            return None
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    parent = list(range(len(names)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            c = corr(test[a], test[b])
            if c is not None and c >= threshold:
                union(a, b)
    roots = sorted({find(i) for i in range(len(names))})
    cluster_of = {i: roots.index(find(i)) for i in range(len(names))}
    clusters: Dict[str, List[str]] = {}
    for i, name in enumerate(names):
        clusters.setdefault(str(cluster_of[i]), []).append(name)

    models: Dict[str, Any] = {}
    last_scores: List[tuple[float, int, str]] = []
    for j, name in enumerate(names):
        ws = [window_smape(j, w) for w in range(y_true.shape[0])]
        ls = ws[-1] if ws else None
        if ls is not None:
            last_scores.append((ls, j, name))
        vec = np.asarray(test[j], dtype=float)
        finite = vec[np.isfinite(vec)]
        if finite.size >= 2:
            mean = float(np.mean(finite)); std = float(np.std(finite))
            slope = float(finite[-1] - finite[0])
            test_summary = {
                "slope_z": round(slope / (std or 1.0), 2),
                "volatility": round(std / (abs(mean) or 1.0), 3),
                "last_z": round((float(finite[-1]) - mean) / (std or 1.0), 2),
            }
        else:
            test_summary = {}
        models[name] = {
            "window_smape": ws,
            "last_smape": ls,
            "test": test_summary,
            "cluster": cluster_of[j],
        }

    last_scores.sort(key=lambda x: x[0])
    detail_names: List[str] = [name for _, _, name in last_scores[:top_n]]
    for root, members in clusters.items():
        best = next((m for _, _, m in last_scores if m in members), None)
        if best and best not in detail_names:
            detail_names.append(best)
    detail_names = detail_names[: max(1, top_n + len(clusters))]

    detail_hist: Dict[str, Any] = {}
    detail_test: Dict[str, Any] = {}
    for name in detail_names:
        j = names.index(name)
        detail_hist[name] = {
            "last_smape": models[name]["last_smape"],
            "forecast_z": _zscore(y_preds[-1, j]),
        }
        detail_test[name] = _zscore(test[j])

    pair_corr: Dict[str, float] = {}
    for a in range(len(detail_names)):
        for b in range(a + 1, len(detail_names)):
            c = corr(test[names.index(detail_names[a])], test[names.index(detail_names[b])])
            if c is not None:
                pair_corr[f"{detail_names[a]} vs {detail_names[b]}"] = round(c, 3)

    stacked = np.asarray([test[j] for j in range(len(names)) if np.any(np.isfinite(test[j]))], dtype=float)
    dispersion = None
    if stacked.size:
        d = np.nanstd(stacked, axis=0)
        if np.any(np.isfinite(d)):
            dispersion = round(float(np.nanmean(d)), 4)

    return {
        "last_validation_actual_z": _zscore(last),
        "models": models,
        "clusters": clusters,
        "detail_models": detail_names,
        "last_validation_detail": detail_hist,
        "test_detail_z": detail_test,
        "test_pairwise_corr": pair_corr,
        "test_forecast_dispersion": dispersion,
        "note": "all models are summarized; detailed vectors are top3 + one per cluster; test actual is NOT shown",
    }

def _decision_card(
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    diagnosis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    feats = series_card.get("features") or {}
    stability = pool_card.get("ranking_stability") or {}
    card = {
        "trend_strength": series_card.get("trend_strength"),
        "seasonal_strength": series_card.get("seasonal_strength"),
        "outliers_pct": (series_card.get("outliers") or {}).get("pct"),
        "crosses_zero": feats.get("crosses_zero"),
        "stationarity": (series_card.get("stationarity") or {}).get("verdict"),
        "ranking_tau": stability.get("mean_kendall_tau"),
        "ranking_verdict": stability.get("verdict"),
    }
    if diagnosis:
        card.update({
            "regime": diagnosis.get("regime"),
            "predictability": diagnosis.get("predictability"),
            "combination_hint": diagnosis.get("combination_hint"),
        })
    return {k: v for k, v in card.items() if v is not None}


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
    prompt_format: str = "text",
    leader: Optional[Dict[str, Any]] = None,
    untried: Optional[Sequence[Dict[str, Any]]] = None,
    repeat_notice: bool = False,
    no_accept: bool = False,
    score_presets: Optional[Sequence[str]] = None,
    score_window_mode: str = "weighted",
    score_target: int = 0,
    scores_used: Optional[int] = None,
) -> str:
    """Per-turn message: cards, diagnosis, ranked history, scratchpad and budget.

    `prompt_format="xml"` wraps each section in a tag and changes nothing else.

    `leader` is the deterministic leader diagnostics block (leader, margin to
    the runner-up, statistical verdict), computed by the harness from the
    attempt history on validation only.
    """
    _check_format(prompt_format)
    xml = prompt_format == "xml"
    parts: List[str] = []

    def open_(tag: str) -> None:
        if xml:
            parts.append(f"<{tag}>")

    def close_(tag: str) -> None:
        if xml:
            parts.append(f"</{tag}>")

    parts.append(f"ITERATION {iteration} of {max_iterations}.")
    parts.append("")
    open_("series_profile")
    parts.append("SERIES PROFILE:")
    parts.append(_compact(_slim_series_card(series_card)))
    close_("series_profile")
    parts.append("")
    open_("model_pool")
    parts.append("MODEL POOL:")
    parts.append(_compact(_slim_pool_card(pool_card)))
    close_("model_pool")

    card = build_dataset_card(state)
    if card:
        parts.append("")
        open_("dataset_card")
        parts.append("DATASET CARD - what worked on the OTHER series (validation only):")
        parts.append(_compact(card, limit=900))
        close_("dataset_card")

    if diagnosis:
        parts.append("")
        open_("diagnosis")
        parts.append(f"DIAGNOSIS ({diagnosis.get('source', 'unknown')}) - a reading, not a rule:")
        parts.append(_compact(_slim_diagnosis(diagnosis), limit=700))
        close_("diagnosis")

    if show_history:
        ranked = state.ranked_attempts()
        parts.append("")
        open_("attempt_history")
        parts.append(f"ATTEMPT HISTORY ({len(ranked)}), best first - lower score is better:")
        if ranked:
            for a in ranked[:10]:
                parts.append("  " + _compact(a.brief(include_rationale=show_rationales), limit=320))
        else:
            parts.append("  (empty)")
        close_("attempt_history")

    if leader:
        parts.append("")
        open_("leader_diagnostics")
        parts.append("LEADER DIAGNOSTICS (validation only):")
        parts.append(_compact(leader, limit=600))
        close_("leader_diagnostics")

    if score_presets:
        board = state.preset_board(score_presets)
        parts.append("")
        open_("score_rankings")
        parts.append(
            f"SCORE RANKINGS (fixed window '{score_window_mode}'; lower is better):"
        )
        for name in score_presets:
            row = board.get(name, {})
            seed = row.get("best_seed") or {}
            best = row.get("best") or {}
            seed_text = (
                f"best seed {seed.get('id')} ({seed.get('score')}), "
                if seed else "no seeded baseline; "
            )
            parts.append(
                f"  {name}: {seed_text}"
                f"best overall {best.get('id')} ({best.get('score')}, "
                f"{best.get('origin')})"
            )
        parts.append(
            "  Every evaluate_strategy returns all three scores. The preset is "
            "chosen only at the final decision."
        )
        close_("score_rankings")

    handles = _handles_summary(state)
    parts.append("")
    open_("handles")
    if handles:
        # Phase 2 seeds stability pools before the loop opens. Under scoring_budget
        # all other actions are withheld, so the agent can reuse those pools but
        # cannot create new handles; say that instead of inviting a tool call.
        if getattr(state.config, "scoring_budget", False):
            parts.append("HANDLES AVAILABLE (seeded by Phase 2; no new handles in direct mode):")
        else:
            parts.append("HANDLES AVAILABLE (seeded by Phase 2, plus anything you create):")
        parts.append(_compact(handles, limit=600))
        if not state.weights:
            if getattr(state.config, "scoring_budget", False):
                parts.append(
                    "  weights are passed as method names in evaluate_strategy; "
                    "no separate weight handle is created in this mode"
                )
            else:
                parts.append(
                    "  no weight handles exist yet - create one with a weights_* tool "
                    "before any combine='weighted' strategy"
                )
    else:
        # Saying "none" matters: handles reset per series, and a model that just
        # finished another one will otherwise reach for a `w1` that does not exist.
        if getattr(state.config, "scoring_budget", False):
            parts.append(
                "HANDLES AVAILABLE: none; direct mode creates no pool/weight handles"
            )
        else:
            parts.append("HANDLES AVAILABLE: none yet (pools and weights start empty)")
    close_("handles")

    methods = _available_weight_methods(state)
    parts.append("")
    open_("weight_methods")
    parts.append(
        "WEIGHT METHODS available for evaluate_strategy['weights'] "
        "(optional; defaults):"
    )
    parts.append("  " + ", ".join(methods))
    if state.weights:
        parts.append(
            "  Existing handles can also be passed directly: "
            + ", ".join(sorted(state.weights))
        )
    close_("weight_methods")

    if untried:
        parts.append("")
        open_("untried")
        parts.append(
            "NEXT UNTRIED COMBINATIONS (not in the history; order arbitrary, NOT a ranking):"
        )
        for i, spec in enumerate(untried):
            parts.append(
                f"  {chr(65 + i)}. "
                f"{json.dumps(spec, ensure_ascii=False, default=str)}"
            )
        close_("untried")

    if repeat_notice:
        parts.append("")
        open_("repeat_notice")
        parts.append(
            "YOU REPEATED THE LAST COMBINATION. It was already tested and did not "
            "add a new attempt. Choose one of the untried combinations above, or "
            "change the model subset/combine; do not resubmit the same call. "
            + (
                " For best_single, change the model; its pool/models argument is ignored."
                if getattr(state.config, "allow_best_single", True) else
                " COMBINATION-ONLY: use at least two models."
            )
        )
        close_("repeat_notice")

    if scratchpad:
        parts.append("")
        open_("scratchpad")
        parts.append("WHAT YOU DID SO FAR:")
        for entry in scratchpad[-4:]:
            parts.append(
                f"  [{entry['iteration']}] {entry['action']}({_compact(entry['action_args'], 120)})"
                f" -> {entry['observation_summary']}"
            )
        close_("scratchpad")

    if last_observation is not None:
        parts.append("")
        open_("last_observation")
        parts.append("LAST OBSERVATION (full):")
        parts.append(_compact(last_observation, limit=900))
        close_("last_observation")

    try:
        shape = _forecast_shape_card(state)
    except Exception:
        shape = {}
    if shape:
        parts.append("")
        open_("forecast_shape")
        parts.append(
            "FORECAST SHAPE (evidence, not a rule): all models summarized; detailed "
            "vectors are top3 + one per test cluster. Last-validation actual is shown; "
            "test actual is NOT shown."
        )
        parts.append(_compact(shape, limit=3200))
        close_("forecast_shape")

    card = _decision_card(series_card, pool_card, diagnosis)
    if card:
        parts.append("")
        open_("decision_card")
        parts.append(
            "SERIES DECISION CARD (repeat; use or ignore, but name one feature in your Thought):"
        )
        parts.append(_compact(card, limit=500))
        close_("decision_card")

    remaining = max_iterations - iteration + 1
    pending = _unscored_weights(state)
    parts.append("")
    open_("budget")
    if score_target:
        used = int(scores_used or 0)
        parts.append(
            f"SCORES: {used}/{score_target} new evaluations used. "
            f"TURNS: {iteration - 1}/{max_iterations} used, {remaining} left."
        )
    if pending and remaining <= 3:
        parts.append(
            f"NOTE: {pending} were computed but never scored. Weights score nothing "
            "until evaluate_strategy runs on them, so that work is lost unless you "
            "evaluate now."
        )
    if no_accept:
        if remaining <= 1:
            parts.append(
                "This is your LAST turn. Test one more combination; the final "
                "strategy is chosen from the whole history."
            )
        else:
            parts.append(
                f"{remaining} turns left. Test another combination; do not call "
                f"{TERMINAL_ACTION}. The final strategy is chosen from the whole history."
            )
    elif remaining <= 1:
        parts.append(
            f"This is your LAST iteration. Use {TERMINAL_ACTION} to take the best attempt."
        )
    else:
        parts.append(
            f"{remaining} iterations left. You may {TERMINAL_ACTION} now if the current "
            "best attempt already looks solid for this series, or respond with "
            "Thought / Action / Action Input to test another hypothesis."
        )
    close_("budget")
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


def _available_weight_methods(state: ReactState) -> List[str]:
    methods = [
        "inverse_error", "softmax_neg_error", "error_trend", "feature_based",
        "meta_model", "best_per_horizon", "topk_per_horizon", "forward",
    ]
    if state.n_windows >= int(getattr(state.config, "min_windows_for_ols", 5)):
        methods.insert(3, "ols")
    if getattr(state, "pooled_meta_model", None) is not None:
        methods.insert(4, "pooled_meta_model")
    return methods


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


def build_final_preset_prompt(
    state: ReactState,
    presets: Optional[Sequence[str]] = None,
    score_window_mode: str = "weighted",
) -> str:
    names = list(presets or FINAL_SCORE_PRESETS)
    board = state.preset_board(names, top_n=3)
    lines = [
        "FINAL SCORE CHOICE.",
        "",
        "The scoring budget is over. Validation is complete; the blind test is not visible.",
        f"The three criteria share the fixed window aggregation '{score_window_mode}'.",
        "Choose exactly ONE criterion for this series. The best attempt under that",
        "criterion is the one that will be applied.",
        "",
        "RANKINGS (lower score is better):",
    ]
    for name in names:
        row = board.get(name, {})
        lines.append(f"  {name}:")
        for item in (row.get("top") or []):
            lines.append(
                f"    {item.get('id')} {item.get('origin')} score={item.get('score')} "
                f"{item.get('strategy')}"
            )
        seed = row.get("best_seed") or {}
        lines.append(
            f"    best seed: {seed.get('id')} score={seed.get('score')}"
            if seed else "    best seed: none"
        )
    lines += [
        "",
        "Return exactly one JSON object and nothing else:",
        '  {"score_preset": "<one of: ' + ", ".join(names) + '>"}',
        "The preset must be one of the three names exactly.",
    ]
    return "\n".join(lines)


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
        scores = observation.get("scores_by_preset") or {}
        score_tail = (
            " | " + " ".join(f"{k}={v}" for k, v in scores.items())
            if scores else ""
        )
        ignored = observation.get("ignored_arguments") or []
        ignored_tail = (
            " [ignored: " + "/".join(str(x) for x in ignored) + "]" if ignored else ""
        )
        leaders = observation.get("best_by_preset") or {}
        leader_tail = ""
        if leaders:
            rows = []
            for name, item in leaders.items():
                item = item or {}
                short = {
                    "smape_pocid_50_50": "50/50",
                    "smape60_pocid40": "60/40",
                    "smape40_pocid60": "40/60",
                }.get(name, name)
                rows.append(f"{short}={item.get('id')}({item.get('score')})")
            leader_tail = " | leaders " + " ".join(rows)
        return (
            f"rank {observation.get('rank')}/{observation.get('total_attempts')}"
            f" score={observation.get('score')}"
            f" rmse={observation.get('metrics', {}).get('rmse')}"
            + score_tail
            + (" (best so far)" if observation.get("is_best") else tail)
            + leader_tail
            + ignored_tail
            + (" [already tested]" if observation.get("already_tested") else "")
            + (
                f" [SAME forecasts as {observation['numerically_identical_to']}]"
                if observation.get("numerically_identical_to") else ""
            )
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
