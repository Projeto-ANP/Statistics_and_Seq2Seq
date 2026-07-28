"""Phase 2 — pool evaluation and baseline seeding (pure code, no LLM).

Runs before the ReAct loop opens. Three jobs:

1. Build the working pool according to the ablation (`full`, `top_k_error`,
   `top_k_stable`) and record which mode was used — the `pool_composition_mode`
   CSV field.
2. Compute the pool card the agent will see every turn: ranked error table,
   ranking stability across windows, and correlated-error groups.
3. **Seed the attempt history with deterministic baselines** — mean, median and
   DBA over all models — before the agent proposes anything (Section 3.2,
   principle 5). Because the final strategy is always the best entry of the whole
   history, the result can never be worse than these baselines by construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from orchestrator_react import tools as T
from orchestrator_react.config import ReactConfig
from orchestrator_react.state import FULL_POOL, Attempt, ReactState


#: Baselines seeded before the loop. Order matters only for readability.
SEED_BASELINES = ("mean", "median", "dba")


def build_pool(state: ReactState, config: Optional[ReactConfig] = None) -> Dict[str, Any]:
    """Selects the working pool for this run, per `config.pool_mode`."""
    config = config or state.config
    mode = str(config.pool_mode)
    k = int(config.pool_k)

    if mode == "full":
        return {
            "pool": FULL_POOL,
            "mode": "full",
            "k": state.n_models,
            "models": list(state.model_names),
        }
    if mode == "top_k_error":
        r = T.select_top_k(state, k=k)
        return {"pool": r["pool"], "mode": mode, "k": r["k"], "models": r["models"]}
    if mode == "top_k_stable":
        r = T.select_stable(state, k=k)
        return {
            "pool": r["pool"],
            "mode": mode,
            "k": r["k"],
            "models": [m["model"] for m in r["models"]],
        }
    raise ValueError(
        f"unknown pool_mode: {mode!r} (valid: full, top_k_error, top_k_stable)"
    )


def seed_baselines(
    state: ReactState,
    pool: str = FULL_POOL,
    methods: Sequence[str] = SEED_BASELINES,
) -> List[Attempt]:
    """Evaluates the deterministic baselines and puts them in the history first.

    They are recorded with `origin="baseline"` so the analysis can tell apart what
    the agent contributed from what was there by construction.
    """
    seeded: List[Attempt] = []
    for method in methods:
        spec: Dict[str, Any] = {"combine": str(method), "pool": pool}
        rationale = {
            "mean": "deterministic baseline: plain mean of the full pool",
            "median": "deterministic baseline: median, robust to outlying models",
            "dba": "deterministic baseline: DTW barycenter of the full pool",
        }.get(str(method), f"deterministic baseline: {method}")
        attempt, _ = state.evaluate(spec, rationale=rationale, origin="baseline", iteration=0)
        seeded.append(attempt)
    return seeded


def pool_report(state: ReactState, top_n: int = 8, corr_threshold: float = 0.9) -> Dict[str, Any]:
    """Pool card injected into the agent's prompt every turn.

    Compact on purpose: top-N table plus an aggregate of the rest, the stability
    verdict, and only the redundant groups — never the full error matrix.
    """
    report: Dict[str, Any] = {
        "n_models": state.n_models,
        "n_windows": state.n_windows,
        "horizon": state.horizon,
        "error_table": T.error_summary(state, top_n=top_n),
        "ranking_stability": T.ranking_stability(state),
    }
    try:
        report["error_correlation"] = T.error_correlation(state, threshold=corr_threshold)
    except ValueError as exc:  # single-model pool
        report["error_correlation"] = {"unavailable": str(exc)}

    per_window = []
    for w in range(state.n_windows):
        top = T.error_summary(state, window=w, top_n=1)["top"]
        per_window.append({"window": w, "best_model": top[0]["model"] if top else None})
    report["best_model_per_window"] = per_window
    return report


def calibration_gate(state: ReactState, config: Optional[ReactConfig] = None) -> Dict[str, Any]:
    """Should the ReAct loop be skipped because the ranking is already settled?

    When the per-window rankings agree almost perfectly, there is little for the
    agent to discover: the top models are the top models in every window, and the
    seeded baselines already cover the sensible combinations. Off by default;
    `calibration_gate_triggered` records whether it fired.
    """
    config = config or state.config
    stability = T.ranking_stability(state)
    tau = stability.get("mean_kendall_tau")
    triggered = bool(
        config.calibration_gate
        and tau is not None
        and float(tau) >= float(config.calibration_gate_kendall)
    )
    return {
        "enabled": bool(config.calibration_gate),
        "triggered": triggered,
        "mean_kendall_tau": tau,
        "threshold": float(config.calibration_gate_kendall),
    }


def run_phase2(
    state: ReactState, config: Optional[ReactConfig] = None, top_n: int = 8
) -> Dict[str, Any]:
    """Runs the whole of Phase 2 and returns what Phase 3 and the CSV need."""
    config = config or state.config
    pool = build_pool(state, config)
    seeded = seed_baselines(state, pool=pool["pool"])
    report = pool_report(state, top_n=top_n)
    gate = calibration_gate(state, config)

    ranked = state.ranked_attempts()
    best = ranked[0] if ranked else None
    return {
        "pool": pool,
        "pool_composition_mode": pool["mode"],
        "report": report,
        "baselines": [a.brief() for a in seeded],
        "best_baseline": best.attempt_id if best else None,
        "calibration_gate": gate,
        "attempts_seeded": len(seeded),
    }


def baseline_scores(state: ReactState) -> Dict[str, Any]:
    """Metrics of the seeded baselines, keyed by strategy — for the CSV.

    Complements `ingest.read_external_baselines`, which brings FFORMA/ADE numbers
    from disk. Together they fill `baseline_results_json`.
    """
    out: Dict[str, Any] = {}
    for a in state.attempts:
        if a.origin != "baseline":
            continue
        out[str(a.spec.get("combine"))] = {
            "attempt_id": a.attempt_id,
            "score": round(float(a.score), 4) if np.isfinite(a.score) else None,
            "rmse": _r(a.aggregate.get("RMSE")),
            "smape": _r(a.aggregate.get("SMAPE")),
            "mape": _r(a.aggregate.get("MAPE")),
            "pocid": _r(a.aggregate.get("POCID"), 1),
        }
    return out


def _r(x: Any, nd: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None
