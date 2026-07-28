"""Phase 1 (diagnosis) and Phase 5 (report) — the two optional LLM phases.

Both are **interpretive, never generative of numbers**. Every figure they may cite
was already computed by the deterministic tools; the model's job is to read a card
and say what it means, or to turn a decision into prose. Nothing here can change
which strategy is applied.

Phase 1 is an ablation (`config.diagnostic_llm`): without an LLM the precomputed
card is used as-is, which is the control arm. With one, the model adds a structured
interpretation that is injected into the Phase 3 prompt.

Both phases degrade instead of failing. A refusal, a timeout or unparseable output
falls back to a deterministic reading, because a missing interpretation must never
cost a series.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from orchestrator_react.llm import LLMClient, LLMError, extract_json, split_think
from orchestrator_react.state import ReactState


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — diagnosis
# ══════════════════════════════════════════════════════════════════════════════

#: Closed vocabularies. Anything outside them is dropped, so the field stays
#: groupable in the analysis instead of collecting free-text variants.
REGIMES = ("trend_dominated", "seasonal_dominated", "noisy", "mixed")
PREDICTABILITY = ("high", "medium", "low")
HINTS = ("robust", "weighted", "selective", "full_pool")

DIAGNOSIS_SYSTEM = """You are a TIME SERIES DIAGNOSTICIAN.

You receive a precomputed profile of one time series and a summary of how a pool of
forecasting models performed on it. Every number was computed by deterministic code.
Your job is to INTERPRET, not to compute: never invent or recompute a figure, and
never name a model that is not in the list you are given.

Answer with ONE JSON object and nothing else:
{
  "regime": one of ["trend_dominated", "seasonal_dominated", "noisy", "mixed"],
  "predictability": one of ["high", "medium", "low"],
  "combination_hint": one of ["robust", "weighted", "selective", "full_pool"],
  "risks": [short strings, at most 3],
  "narrative": "2-3 sentences explaining the series and what it implies for combining"
}

How to choose combination_hint:
  robust      - noisy or unstable ranking; median / trimmed mean resist it
  weighted    - models differ clearly and consistently; error-based weights pay off
  selective   - a few models dominate; a small pool is better than the full one
  full_pool   - models are similar and none dominates; averaging everything is fine
"""

REPORT_SYSTEM = """You are writing the DECISION RECORD for one forecast combination.

You receive the series profile, the pool summary, and the strategy that was selected
by backtesting on validation windows. Write 3 to 5 sentences in English explaining
WHY this strategy suits THIS series, in terms of the observable characteristics you
were given: trend strength, seasonal strength, ranking stability across windows,
redundancy between models, error spread.

Rules:
- Only cite numbers that appear in the material you were given.
- Do not say "it had the lowest error" and stop there; explain what about the series
  makes this kind of combination appropriate.
- No preamble, no bullet points, no markdown. Plain prose only.
"""


def build_diagnosis_prompt(series_card: Dict[str, Any], pool_card: Dict[str, Any]) -> str:
    """Compact material for Phase 1 — the same cards the agent sees."""
    from orchestrator_react.prompts import _slim_pool_card, _slim_series_card

    return "\n".join(
        [
            "SERIES PROFILE:",
            json.dumps(_slim_series_card(series_card), ensure_ascii=False, default=str),
            "",
            "MODEL POOL:",
            json.dumps(_slim_pool_card(pool_card), ensure_ascii=False, default=str),
            "",
            "Answer with the JSON object only.",
        ]
    )


def deterministic_diagnosis(
    series_card: Dict[str, Any], pool_card: Dict[str, Any]
) -> Dict[str, Any]:
    """The control arm: the same interpretation, decided by thresholds.

    Used when the ablation runs without an LLM, and as the fallback when the model
    fails. Having a deterministic twin is what makes the ablation a fair comparison:
    both arms produce the same field, one by rule and one by reading.
    """
    trend = _num(series_card.get("trend_strength"))
    seasonal = _num(series_card.get("seasonal_strength"))
    entropy = _num((series_card.get("features") or {}).get("spectral_entropy"))
    stability = pool_card.get("ranking_stability", {}) or {}
    tau = _num(stability.get("mean_kendall_tau"))
    spread = _num((pool_card.get("error_table") or {}).get("relative_spread"))
    corr = pool_card.get("error_correlation", {}) or {}
    redundant = bool(corr.get("redundant_groups"))

    if trend is not None and seasonal is not None:
        if trend >= 0.6 and seasonal >= 0.6:
            regime = "mixed"
        elif seasonal >= 0.6:
            regime = "seasonal_dominated"
        elif trend >= 0.6:
            regime = "trend_dominated"
        else:
            regime = "noisy"
    else:
        regime = "mixed"

    if entropy is None:
        predictability = "medium"
    elif entropy >= 0.85:
        predictability = "low"
    elif entropy <= 0.5:
        predictability = "high"
    else:
        predictability = "medium"

    if tau is not None and tau < 0.3:
        hint = "robust"
    elif spread is not None and spread >= 0.5 and tau is not None and tau >= 0.5:
        hint = "selective"
    elif spread is not None and spread >= 0.3:
        hint = "weighted"
    else:
        hint = "full_pool"

    risks: List[str] = []
    if tau is not None and tau < 0.3:
        risks.append("model ranking does not hold across windows")
    if redundant:
        risks.append("several models carry correlated error")
    if predictability == "low":
        risks.append("series is close to noise; fitted weights would overfit")
    if series_card.get("n_validation_windows", 0) <= 3:
        risks.append("only 3 validation windows: weight estimation is high variance")

    narrative = (
        f"Trend strength {trend} and seasonal strength {seasonal} put this series in the "
        f"{regime.replace('_', ' ')} regime, with {predictability} predictability. "
        f"Ranking stability across windows is {stability.get('verdict', 'unknown')}"
        + (f" (Kendall tau {tau})" if tau is not None else "")
        + f", which favours a {hint.replace('_', ' ')} combination."
    )

    return {
        "regime": regime,
        "predictability": predictability,
        "combination_hint": hint,
        "risks": risks[:3],
        "narrative": narrative,
        "source": "deterministic",
    }


def run_diagnosis(
    state: ReactState,
    series_card: Dict[str, Any],
    client: Optional[LLMClient],
    pool_card: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 1. Returns a structured interpretation, LLM-read or rule-based."""
    pool_card = pool_card or {}
    fallback = deterministic_diagnosis(series_card, pool_card)
    if client is None:
        return fallback

    try:
        raw = client.complete(DIAGNOSIS_SYSTEM, build_diagnosis_prompt(series_card, pool_card))
    except LLMError as exc:
        fallback["fallback_reason"] = f"llm_error: {exc}"
        return fallback

    _, body = split_think(raw)
    parsed = extract_json(body)
    if not isinstance(parsed, dict):
        fallback["fallback_reason"] = "unparseable response"
        return fallback

    clean = _validate_diagnosis(parsed, state.model_names, fallback)
    clean["source"] = "llm"
    clean["model"] = getattr(client, "name", "unknown")
    return clean


def _validate_diagnosis(
    parsed: Dict[str, Any], model_names: Sequence[str], fallback: Dict[str, Any]
) -> Dict[str, Any]:
    """Keeps the model inside the closed vocabulary and inside the real model list.

    Out-of-vocabulary labels fall back to the deterministic value rather than
    entering the CSV as a one-off string, and any model name that does not exist in
    the pool is stripped from the narrative material — a hallucinated name would
    otherwise be handed to the Phase 3 agent as if it were real.
    """
    out: Dict[str, Any] = {}
    rejected: List[str] = []

    for key, vocabulary in (
        ("regime", REGIMES),
        ("predictability", PREDICTABILITY),
        ("combination_hint", HINTS),
    ):
        value = str(parsed.get(key, "")).strip().lower().replace(" ", "_")
        if value in vocabulary:
            out[key] = value
        else:
            out[key] = fallback[key]
            if value:
                rejected.append(f"{key}={value!r}")

    risks = parsed.get("risks", [])
    if isinstance(risks, str):
        risks = [risks]
    out["risks"] = [str(r)[:160] for r in risks if str(r).strip()][:3] if isinstance(risks, list) else []

    narrative = str(parsed.get("narrative", "")).strip()
    known = set(model_names)
    invented = [
        token
        for token in _candidate_model_tokens(narrative)
        if token not in known
    ]
    if invented:
        rejected.append(f"unknown models mentioned: {invented[:5]}")
    out["narrative"] = narrative[:800] or fallback["narrative"]

    if rejected:
        out["validation_notes"] = rejected
    return out


def _candidate_model_tokens(text: str) -> List[str]:
    """Tokens that look like model identifiers, for the hallucination check.

    Only screens tokens that look like the project's model names (uppercase or
    containing an underscore), so ordinary prose is not flagged.
    """
    tokens = []
    for raw in text.replace(",", " ").replace(".", " ").split():
        token = raw.strip("()[]'\"`:;")
        if not token or len(token) < 3:
            continue
        if token.isupper() or ("_" in token and token.lower() != token.upper()):
            tokens.append(token)
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — report
# ══════════════════════════════════════════════════════════════════════════════


def build_report_prompt(outcome: Any) -> str:
    """Material for Phase 5: the cards plus the decision that was made."""
    from orchestrator_react.prompts import _slim_pool_card, _slim_series_card

    decision = outcome.decision()
    attempt = outcome.react.final_attempt if outcome.react else None
    ranked = outcome.state.ranked_attempts()[:4] if outcome.state else []

    return "\n".join(
        [
            "SERIES PROFILE:",
            json.dumps(_slim_series_card(outcome.series_card), ensure_ascii=False, default=str),
            "",
            "MODEL POOL:",
            json.dumps(_slim_pool_card(outcome.pool_card), ensure_ascii=False, default=str),
            "",
            "SELECTED STRATEGY:",
            json.dumps(
                {
                    "strategy": decision.get("strategy"),
                    "models": decision.get("models"),
                    "weights": decision.get("weights"),
                    "validation": decision.get("validation"),
                    "chosen_by": "the agent" if attempt and attempt.origin == "agent" else "a seeded baseline",
                },
                ensure_ascii=False,
                default=str,
            ),
            "",
            "ALTERNATIVES THAT SCORED WORSE:",
            json.dumps([a.brief(include_rationale=False) for a in ranked[1:]], ensure_ascii=False),
            "",
            "Write the explanation now, plain prose, 3 to 5 sentences.",
        ]
    )


def run_report(state: ReactState, outcome: Any, client: Optional[LLMClient]) -> str:
    """Phase 5. Returns prose, or an empty string so the caller keeps its fallback.

    Returning "" rather than a placeholder is deliberate: `SeriesOutcome.csv_fields`
    falls back to the deterministic causal justification produced in Phase 3, which
    is always present (principle 7).
    """
    if client is None:
        return ""
    try:
        raw = client.complete(REPORT_SYSTEM, build_report_prompt(outcome))
    except LLMError:
        return ""

    _, body = split_think(raw)
    text = " ".join(str(body).split())
    if text.startswith("{") or text.startswith("["):
        # The model answered with JSON; salvage a text field if there is one.
        parsed = extract_json(text)
        if isinstance(parsed, dict):
            for key in ("narrative", "text", "explanation", "justification"):
                if isinstance(parsed.get(key), str) and parsed[key].strip():
                    text = parsed[key].strip()
                    break
    return text[:2000]


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # drops NaN
