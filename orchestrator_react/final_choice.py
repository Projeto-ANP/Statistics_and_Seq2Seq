"""Final selection among family winners, blind to names/scores/ranks.

The ReAct loop ends with the best validation score (the "leader"). This module
adds an optional final phase that builds a shortlist of *different* strategy
families, hides everything that could bias a choice (model names, combination
types, validation scores, ranks, origins), and asks the LLM to choose one
candidate or KEEP the leader.

Leakage rules, enforced by construction:

* the prompt builder only receives ``BlindCandidate`` objects, which carry a
  forecast trajectory and derived features — no attempt id, spec, score, rank or
  origin;
* the test actual values are never passed to this module; only the future
  forecasts produced by the base models (available before the test) are shown;
* the series profile is slimmed to shape/statistics fields, which excludes the
  deterministic trend/seasonality "champions" (those carry model names);
* the diagnosis block keeps only ``regime``, ``predictability`` and
  ``combination_hint`` — not the free-text narrative/risks, which could mention a
  model.

The mapping from A/B/C/... back to the real attempts lives only in the debug
artifact, never in the prompt.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import LLMClient, LLMError, split_think
from orchestrator_react.state import ReactState


LABELS = "ABCDEFGHIJ"
MAX_PROMPT_POINTS = 12


# ──────────────────────────────────────────────────────────────────────────────
# data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BlindCandidate:
    """What the selector is allowed to see about one candidate.

    Deliberately has no ``attempt_id``, ``spec``, ``score``, ``rank`` or
    ``origin``. The prompt builder only reads this object.
    """

    label: str
    forecast_scaled: List[float]
    features: Dict[str, Optional[float]]


@dataclass
class ShortlistEntry:
    """Internal mapping between a real attempt and its blind view."""

    attempt: Any
    forecast: np.ndarray
    label: str = ""
    blind: Optional[BlindCandidate] = None


@dataclass
class ParsedChoice:
    choice: str
    confidence: str
    reason: str
    raw: str
    parse_error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────────────────────


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _fmt(value: Any, nd: int = 3) -> str:
    number = _num(value)
    if number is None:
        return "n/a"
    text = f"{number:.{nd}f}"
    return "0." + "0" * nd if text == "-0." + "0" * nd else text


def _fmt_list(values: Sequence[Any], nd: int = 3) -> str:
    return "[" + ", ".join(_fmt(v, nd) for v in values) + "]"


def _fmt_pct(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{100.0 * number:.1f}%"


def _fmt_band(low: Sequence[Any], high: Sequence[Any], nd: int = 3) -> str:
    pairs = [
        f"[{_fmt(lo, nd)}, {_fmt(hi, nd)}]"
        for lo, hi in zip(low, high)
    ]
    return "[" + ", ".join(pairs) + "]"


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return np.unique(idx)


def _same_forecast(a: np.ndarray, b: np.ndarray, corr_threshold: float = 0.9999) -> bool:
    """True when two candidates are effectively the same trajectory.

    Requires both a high correlation AND the same standardized shape, so a
    candidate that is merely a rescaled version of another is not silently
    dropped. Exact near-equality is handled first.
    """
    if a.shape != b.shape:
        return False
    if np.allclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True):
        return True
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 3:
        return False
    aa, bb = a[finite], b[finite]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return False
    corr = float(np.corrcoef(aa, bb)[0, 1])
    if not np.isfinite(corr) or corr < float(corr_threshold):
        return False
    za = (aa - aa.mean()) / (aa.std() + 1e-12)
    zb = (bb - bb.mean()) / (bb.std() + 1e-12)
    return bool(np.allclose(za, zb, rtol=1e-6, atol=1e-6))


def _percentile_columns(pool: np.ndarray, q: float) -> np.ndarray:
    out = np.full(pool.shape[1], np.nan, dtype=float)
    for j in range(pool.shape[1]):
        col = pool[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            out[j] = float(np.percentile(col, q))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# shortlist and context
# ──────────────────────────────────────────────────────────────────────────────


def build_family_shortlist(
    attempts: Sequence[Any],
    forecast_for: Callable[[Any], np.ndarray],
    max_candidates: int = 5,
    dedup_corr: float = 0.9999,
    max_abs: Optional[float] = None,
) -> List[ShortlistEntry]:
    """Best *plausible* validation attempt of each combination family.

    The global leader is included when it is plausible: it is the best member of
    its own family. ``max_abs`` is a deterministic sanity bound (computed from the
    historical scale) that skips exploding forecasts, including those of a family
    whose best-validation member is degenerate. No test actual is used.

    Families are hidden from the prompt; they are only used here to make the
    shortlist diverse.
    """
    ranked = sorted(
        [a for a in attempts if _num(getattr(a, "score", None)) is not None],
        key=lambda a: float(a.score),
    )
    by_family: Dict[str, List[Any]] = {}
    family_order: List[str] = []
    for attempt in ranked:
        spec = getattr(attempt, "spec", {}) or {}
        family = str(spec.get("combine", "?"))
        if family not in by_family:
            by_family[family] = []
            family_order.append(family)
        by_family[family].append(attempt)

    limit = min(max(1, int(max_candidates)), len(LABELS))
    kept: List[ShortlistEntry] = []
    for family in family_order:
        for attempt in by_family[family]:
            try:
                forecast = np.asarray(forecast_for(attempt), dtype=float).ravel()
            except Exception:
                continue
            if forecast.size == 0 or not np.any(np.isfinite(forecast)):
                continue
            if max_abs is not None and float(np.nanmax(np.abs(forecast))) > float(max_abs):
                continue
            if any(_same_forecast(forecast, entry.forecast, dedup_corr) for entry in kept):
                continue
            kept.append(ShortlistEntry(attempt=attempt, forecast=forecast))
            break
        if len(kept) >= limit:
            break
    return kept


def build_series_context(
    state: ReactState,
    series_card: Dict[str, Any],
    max_points: int = MAX_PROMPT_POINTS,
) -> Dict[str, Any]:
    """Past context plus the pool's blind-window forecasts, in one common scale.

    No test actual is available here, by design: ``ReactState`` only carries the
    validation actuals and the base models' forecasts for the blind window.
    """
    y_true = np.asarray(state.y_true, dtype=float)
    if y_true.ndim != 2 or y_true.size == 0:
        raise ValueError("state.y_true must be (n_windows, horizon)")
    horizon = int(y_true.shape[1])
    last_actual = float(y_true[-1, -1])

    hist = state.train_series
    if hist is None:
        hist = y_true.reshape(-1)
    hist = np.asarray(hist, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        hist = np.asarray([last_actual], dtype=float)

    try:
        period = int(series_card.get("seasonal_period") or 0)
    except (TypeError, ValueError):
        period = 0

    look = max(2, min(int(hist.size), max(2 * period, 2 * horizon, 2)))
    scale = float(np.nanstd(hist[-look:]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.nanstd(hist))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(abs(last_actual), 1.0)

    if period >= 2 and hist.size >= period:
        seasonal = np.resize(hist[-period:], horizon)
        has_seasonal = True
    else:
        seasonal = np.full(horizon, last_actual, dtype=float)
        has_seasonal = False

    pool = np.asarray(state.test_preds, dtype=float)
    if pool.ndim != 2 or pool.shape[1] != horizon:
        raise ValueError("state.test_preds must be (n_models, horizon)")
    pool_median = _percentile_columns(pool, 50.0)
    pool_q10 = _percentile_columns(pool, 10.0)
    pool_q90 = _percentile_columns(pool, 90.0)
    width = pool_q90 - pool_q10
    dispersion = float(np.nanmean(width) / scale) if np.any(np.isfinite(width)) else None

    points = _downsample_indices(horizon, max_points)
    tail = hist[-min(8, int(hist.size)):]
    return {
        "horizon": horizon,
        "last_actual": last_actual,
        "scale": scale,
        "period": period,
        "has_seasonal": has_seasonal,
        "seasonal_forecast": seasonal,
        "pool_median": pool_median,
        "pool_q10": pool_q10,
        "pool_q90": pool_q90,
        "pool_dispersion": dispersion,
        "points": points,
        "last_path_scaled": (tail - last_actual) / scale,
        "pool_median_path_scaled": (pool_median[points] - last_actual) / scale,
        "pool_q10_path_scaled": (pool_q10[points] - last_actual) / scale,
        "pool_q90_path_scaled": (pool_q90[points] - last_actual) / scale,
    }


def _candidate_features(forecast: np.ndarray, ctx: Dict[str, Any]) -> Dict[str, Optional[float]]:
    fc = np.asarray(forecast, dtype=float).ravel()
    mask = np.isfinite(fc)
    values = fc[mask]
    if values.size == 0:
        return {}
    scale = float(ctx["scale"]) or 1.0
    last = float(ctx["last_actual"])
    horizon = int(values.size)
    pool_median = np.asarray(ctx["pool_median"], dtype=float)[mask]
    pool_q10 = np.asarray(ctx["pool_q10"], dtype=float)[mask]
    pool_q90 = np.asarray(ctx["pool_q90"], dtype=float)[mask]

    seasonal_dist: Optional[float] = None
    if ctx.get("has_seasonal"):
        seasonal = np.asarray(ctx["seasonal_forecast"], dtype=float)[mask]
        seasonal_dist = float(np.nanmean(np.abs(values - seasonal)) / scale)

    if np.any(np.isfinite(pool_q10)) and np.any(np.isfinite(pool_q90)):
        outside = float(np.nanmean((values < pool_q10) | (values > pool_q90)))
        outside = outside if np.isfinite(outside) else None
    else:
        outside = None
    return {
        "level_1": float((values[0] - last) / scale),
        "slope": float((values[-1] - values[0]) / max(horizon - 1, 1) / scale),
        "range": float((np.nanmax(values) - np.nanmin(values)) / scale),
        "volatility": float(np.nanstd(values) / scale),
        "naive_dist": float(np.nanmean(np.abs(values - last)) / scale),
        "seasonal_dist": seasonal_dist,
        "pool_dist": float(np.nanmean(np.abs(values - pool_median)) / scale),
        "outside_band": outside,
    }


def blind_shortlist(
    shortlist: Sequence[ShortlistEntry],
    ctx: Dict[str, Any],
    dataset: str,
    dataset_index: int,
    seed: int,
) -> Tuple[List[ShortlistEntry], Dict[str, ShortlistEntry]]:
    """Shuffle candidates and attach A/B/C labels plus the blind view.

    The shuffle is deterministic per (dataset, series, seed), so a replay can
    reproduce the same labels and mapping.
    """
    order = list(range(len(shortlist)))
    random.Random(f"{dataset}|{int(dataset_index)}|{int(seed)}").shuffle(order)

    points = ctx["points"]
    scale = float(ctx["scale"]) or 1.0
    last = float(ctx["last_actual"])
    blinded: List[ShortlistEntry] = []
    by_label: Dict[str, ShortlistEntry] = {}
    for position, index in enumerate(order):
        entry = shortlist[index]
        label = LABELS[position]
        scaled = (np.asarray(entry.forecast, dtype=float)[points] - last) / scale
        entry.label = label
        entry.blind = BlindCandidate(
            label=label,
            forecast_scaled=[float(v) for v in scaled],
            features=_candidate_features(entry.forecast, ctx),
        )
        by_label[label] = entry
        blinded.append(entry)
    return blinded, by_label


# ──────────────────────────────────────────────────────────────────────────────
# prompt
# ──────────────────────────────────────────────────────────────────────────────


def build_system_prompt(allow_keep: bool = True, labels: str = "ABCDE") -> str:
    choices = "|".join(list(labels)) + ("|KEEP" if allow_keep else "")
    if allow_keep:
        choice_line = f"CHOICE: <{choices}>"
        keep_rule = (
            "5. If the candidates are nearly indistinguishable, KEEP. Do not override "
            "the leader just to change something."
        )
    else:
        choice_line = f"CHOICE: <{choices}>"
        keep_rule = (
            "5. If the candidates are nearly indistinguishable, choose the one that is "
            "least implausible. KEEP is not available in this run."
        )
    return f"""You are a forecast SELECTION agent.

A pool of already-trained forecasting models produced predictions for one time
series. A validation protocol shortlisted a few candidate COMBINATIONS. Each
candidate is a complete forecast for the same blind future window. The test
actual is NOT shown and must not be guessed.

Your job is to choose ONE candidate that is the most structurally plausible for
THIS series, or to KEEP the current leader if no candidate is clearly better.

Model names, validation scores, ranks, and combination types are deliberately
hidden: they are noisy and would bias the choice. Judge only the candidate
trajectories against the series' recent behavior and the pool context.

SELECTION RUBRIC (apply in this order):
1. Continuity: the forecast should start near the last observed value. A large
   unexplained jump is a warning sign.
2. Shape: the trajectory should be consistent with the series' recent trend and
   seasonal pattern, given the SERIES PROFILE. Do not mechanically extrapolate:
   a break in the trend can be legitimate, but it needs support in the profile.
3. Scale: the level, range and volatility must be plausible for this series and
   inside (or near) the pool's forecast band. An off-scale or exploding
   candidate is a strong reason to reject it.
4. Robustness: do not prefer a candidate just because it is the most central or
   the most extreme. Use the profile: strongly seasonal -> check the seasonal
   phase/amplitude; strongly trending -> check direction and magnitude; noisy ->
   prefer a stable trajectory unless the profile justifies a break.
{keep_rule}

Answer exactly three lines, nothing else:
{choice_line}
CONFIDENCE: <low|medium|high>
REASON: <one or two sentences citing the series characteristics you used>
"""


def _slim_profile(series_card: Dict[str, Any]) -> Dict[str, Any]:
    features = series_card.get("features") or {}
    stationarity = series_card.get("stationarity") or {}
    outliers = series_card.get("outliers") or {}
    profile = {
        "n_points": series_card.get("n_points"),
        "frequency": series_card.get("frequency"),
        "horizon": series_card.get("horizon"),
        "seasonal_period": series_card.get("seasonal_period"),
        "trend_strength": _num(series_card.get("trend_strength")),
        "trend_direction": series_card.get("trend_direction"),
        "trend_relative_change": _num(series_card.get("trend_relative_change")),
        "seasonal_strength": _num(series_card.get("seasonal_strength")),
        "stationarity": stationarity.get("verdict"),
        "outliers_pct": _num(outliers.get("pct")),
        "acf1": _num(features.get("acf1")),
        "acf_seasonal": _num(features.get("acf_seasonal")),
        "spectral_entropy": _num(features.get("spectral_entropy")),
        "hurst": _num(features.get("hurst")),
        "coef_variation": _num(features.get("coef_variation")),
        "crosses_zero": (
            bool(features["crosses_zero"])
            if features.get("crosses_zero") is not None
            else None
        ),
    }
    return {key: value for key, value in profile.items() if value is not None}


def _slim_diagnosis(diagnosis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not diagnosis:
        return {}
    allowed = ("regime", "predictability", "combination_hint")
    return {key: diagnosis[key] for key in allowed if diagnosis.get(key) is not None}


def build_final_prompt(
    series_card: Dict[str, Any],
    diagnosis: Optional[Dict[str, Any]],
    ctx: Dict[str, Any],
    entries: Sequence[ShortlistEntry],
    allow_keep: bool = True,
) -> Tuple[str, str]:
    """Builds the blind prompt. Only ``entry.blind`` is read, never ``entry.attempt``."""
    labels = "".join(entry.label for entry in entries if entry.blind)
    system = build_system_prompt(allow_keep=allow_keep, labels=labels or "ABCDE")
    profile = _slim_profile(series_card)
    diag = _slim_diagnosis(diagnosis)

    lines: List[str] = [
        "SERIES PROFILE:",
        json.dumps(profile, ensure_ascii=False, sort_keys=True),
        "",
        "LAST OBSERVED VALUE:",
        f"last = {_fmt(ctx['last_actual'], 6)}",
        f"recent scaled path (last = 0): {_fmt_list(ctx['last_path_scaled'])}",
        "",
        "POOL CONTEXT (all base models, blind window, same scale):",
        f"scaled median: {_fmt_list(ctx['pool_median_path_scaled'])}",
        f"q10-q90 band: {_fmt_band(ctx['pool_q10_path_scaled'], ctx['pool_q90_path_scaled'])}",
        f"dispersion: {_fmt(ctx['pool_dispersion'], 3)}",
        "",
    ]
    if diag:
        lines += [
            "DIAGNOSIS (a reading, not a rule):",
            json.dumps(diag, ensure_ascii=False, sort_keys=True),
            "",
        ]

    lines += [
        "CANDIDATES (same scale; each is the best of a different hidden family on "
        "validation; the test actual is NOT shown):",
        "",
    ]
    for entry in entries:
        blind = entry.blind
        if blind is None:
            continue
        features = blind.features
        lines += [
            f"{blind.label}.",
            f"  scaled forecast: {_fmt_list(blind.forecast_scaled)}",
            f"  level at h=1: {_fmt(features.get('level_1'))}",
            f"  slope: {_fmt(features.get('slope'))}",
            f"  range: {_fmt(features.get('range'))}",
            f"  volatility: {_fmt(features.get('volatility'))}",
            f"  dist to last: {_fmt(features.get('naive_dist'))}",
            f"  dist to seasonal naive: {_fmt(features.get('seasonal_dist'))}",
            f"  dist to pool median: {_fmt(features.get('pool_dist'))}",
            f"  outside band: {_fmt_pct(features.get('outside_band'))}",
            "",
        ]
    lines.append(
        "REMINDER: choose the candidate that is most plausible for THIS series, or "
        "KEEP if none is clearly better. Do not use hidden names, scores, or ranks."
    )
    return system, "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# parsing and decision
# ──────────────────────────────────────────────────────────────────────────────


_CHOICE_RE = re.compile(r"CHOICE\s*:\s*([^\n]+)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*:\s*([^\n]+)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON\s*:\s*([^\n]+)", re.IGNORECASE)


def parse_choice(raw: str, labels: Sequence[str], allow_keep: bool = True) -> ParsedChoice:
    text = str(raw or "")
    _, body = split_think(text)
    body = body or text

    choice_match = _CHOICE_RE.search(body)
    if not choice_match:
        return ParsedChoice("", "medium", "", text, "missing CHOICE line")

    value = choice_match.group(1).strip().strip("`*_ .;:'\"")
    token = value.upper()
    if token.startswith("KEEP"):
        choice = "KEEP"
    else:
        match = re.match(r"^(?:CANDIDATE\s+)?([A-Z])\b", token)
        choice = match.group(1) if match else ""

    label_set = set(str(label).upper() for label in labels)
    if choice != "KEEP" and choice not in label_set:
        return ParsedChoice(choice, "medium", "", text, f"unknown label: {value[:40]!r}")
    # KEEP is parsed even when the run disallows it; `decide` turns it into
    # `keep_disallowed` so the artifact records the reason instead of a parse error.

    confidence_match = _CONFIDENCE_RE.search(body)
    confidence = "medium"
    if confidence_match:
        candidate = confidence_match.group(1).strip().strip("`*_ .;:'\"").lower()
        if candidate in {"low", "medium", "high"}:
            confidence = candidate

    reason_match = _REASON_RE.search(body)
    reason = reason_match.group(1).strip() if reason_match else ""
    return ParsedChoice(choice, confidence, reason[:500], text)


def _same_attempt(a: Any, b: Any) -> bool:
    if a is b:
        return True
    a_id = getattr(a, "attempt_id", None)
    b_id = getattr(b, "attempt_id", None)
    return a_id is not None and a_id == b_id


def decide(
    parsed: ParsedChoice,
    entries: Sequence[ShortlistEntry],
    leader: Any,
    ctx: Dict[str, Any],
    config: ReactConfig,
    forecast_for: Callable[[Any], np.ndarray],
    allow_keep: bool = True,
) -> Tuple[Any, str, Optional[str]]:
    """Applies the conservative decision policy.

    Returns ``(chosen_attempt, decision, fallback_reason)``. ``decision`` is one
    of ``keep``, ``override`` or ``fallback``.
    """
    if parsed.parse_error or not parsed.choice:
        return leader, "fallback", "parse_error"
    if parsed.choice == "KEEP":
        if not allow_keep:
            return leader, "fallback", "keep_disallowed"
        return leader, "keep", None

    by_label = {entry.label: entry for entry in entries if entry.label}
    entry = by_label.get(parsed.choice)
    if entry is None:
        return leader, "fallback", "unknown_label"
    chosen = entry.attempt
    if not _same_attempt(chosen, leader):
        if config.final_selector_low_confidence_keep and parsed.confidence == "low":
            return leader, "fallback", "low_confidence"
        try:
            forecast = np.asarray(forecast_for(chosen), dtype=float).ravel()
        except Exception:
            return leader, "fallback", "forecast_error"
        if forecast.size == 0 or not np.all(np.isfinite(forecast)):
            return leader, "fallback", "non_finite_forecast"
        limit = float(config.final_selector_degenerate_factor) * max(
            abs(float(ctx["last_actual"])), float(ctx["scale"]), 1.0
        )
        if float(np.nanmax(np.abs(forecast))) > limit:
            return leader, "fallback", "degenerate_forecast"
        return chosen, "override", None
    return chosen, "keep", None


def choose_final(
    state: ReactState,
    series_card: Dict[str, Any],
    diagnosis: Optional[Dict[str, Any]],
    client: Optional[LLMClient],
    config: ReactConfig,
    dataset: str,
    dataset_index: int,
    forecast_for: Optional[Callable[[Any], np.ndarray]] = None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Runs the final selection phase.

    The loop and the validation ranking are already finished. This function never
    reads the test actual; it only applies each candidate to the blind window to
    show its forecast.
    """
    leader = state.best_attempt()
    debug: Dict[str, Any] = {
        "mode": "agent_family",
        "leader_id": getattr(leader, "attempt_id", None),
        "leader_spec": getattr(leader, "spec", None),
        "leader_validation_score": (
            float(leader.score)
            if leader is not None and _num(getattr(leader, "score", None)) is not None
            else None
        ),
        "decision": "fallback",
        "fallback": None,
        "shortlist": [],
        "shortlist_size": 0,
        "prompt": {},
        "response": "",
        "parsed": {},
        "chosen_id": getattr(leader, "attempt_id", None),
    }
    if leader is None:
        debug["fallback"] = "no_leader"
        return None, debug
    if client is None:
        debug["fallback"] = "no_client"
        return leader, debug

    if forecast_for is None:
        forecast_for = lambda attempt: state.apply_to_test(attempt.spec)[0]

    try:
        ctx = build_series_context(state, series_card)
    except Exception as exc:
        debug["fallback"] = f"context_error:{type(exc).__name__}"
        return leader, debug

    # Deterministic sanity bound: a family whose best-validation member explodes
    # is not a legitimate candidate. This is computed from the history/scale, not
    # from the test actual.
    max_abs = float(config.final_selector_degenerate_factor) * max(
        abs(float(ctx["last_actual"])), float(ctx["scale"]), 1.0
    )
    shortlist = build_family_shortlist(
        state.attempts,
        forecast_for,
        max_candidates=config.final_selector_max_candidates,
        max_abs=max_abs,
    )
    debug["shortlist_size"] = len(shortlist)
    if not shortlist:
        debug["fallback"] = "all_candidates_degenerate"
        return leader, debug

    # If the validation leader itself is degenerate, KEEP must not mean "apply the
    # exploding forecast". Use the best plausible family winner as the effective
    # leader and record the replacement. This is rare, but it is exactly the case
    # the deterministic guard exists for.
    if not any(_same_attempt(entry.attempt, leader) for entry in shortlist):
        debug["original_leader_id"] = getattr(leader, "attempt_id", None)
        leader = shortlist[0].attempt
        debug["leader_replaced_with"] = getattr(leader, "attempt_id", None)
        debug["leader_id"] = getattr(leader, "attempt_id", None)
        debug["leader_spec"] = getattr(leader, "spec", None)
        debug["leader_validation_score"] = (
            float(leader.score) if _num(getattr(leader, "score", None)) is not None else None
        )
        debug["chosen_id"] = getattr(leader, "attempt_id", None)

    if len(shortlist) < 2:
        debug["fallback"] = "shortlist_too_small"
        return leader, debug

    entries, _ = blind_shortlist(
        shortlist, ctx, dataset, dataset_index, config.final_selector_shuffle_seed
    )
    system, user = build_final_prompt(
        series_card, diagnosis, ctx, entries, allow_keep=config.final_selector_allow_keep
    )
    debug["shortlist"] = [
        {
            "label": entry.label,
            "attempt_id": getattr(entry.attempt, "attempt_id", None),
            "spec": getattr(entry.attempt, "spec", None),
            "validation_score": (
                float(entry.attempt.score)
                if _num(getattr(entry.attempt, "score", None)) is not None
                else None
            ),
            "features": dict(entry.blind.features) if entry.blind else {},
            "forecast_scaled": (
                list(entry.blind.forecast_scaled) if entry.blind else []
            ),
        }
        for entry in entries
    ]
    debug["leader_label"] = next(
        (entry.label for entry in entries if _same_attempt(entry.attempt, leader)), None
    )
    debug["prompt"] = {"system": system, "user": user}
    debug["context"] = {
        "last_actual": ctx["last_actual"],
        "scale": ctx["scale"],
        "pool_dispersion": ctx["pool_dispersion"],
    }

    try:
        raw = client.complete(system, user)
    except LLMError as exc:
        debug["fallback"] = "llm_error"
        debug["error"] = str(exc)[:400]
        return leader, debug
    except Exception as exc:
        debug["fallback"] = "llm_error"
        debug["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return leader, debug

    debug["response"] = str(raw)
    parsed = parse_choice(
        raw, LABELS[: len(entries)], allow_keep=config.final_selector_allow_keep
    )
    debug["parsed"] = {
        "choice": parsed.choice,
        "confidence": parsed.confidence,
        "reason": parsed.reason,
        "parse_error": parsed.parse_error,
    }
    chosen, decision, fallback = decide(
        parsed,
        entries,
        leader,
        ctx,
        config,
        forecast_for,
        allow_keep=config.final_selector_allow_keep,
    )
    debug["decision"] = decision
    debug["fallback"] = fallback
    debug["chosen_id"] = getattr(chosen, "attempt_id", None)
    if decision == "override":
        debug["chosen_spec"] = getattr(chosen, "spec", None)
        debug["chosen_validation_score"] = (
            float(chosen.score) if _num(getattr(chosen, "score", None)) is not None else None
        )
    return chosen, debug
