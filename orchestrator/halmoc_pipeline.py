"""End-to-end HALMOC pipeline orchestrator.

`run_halmoc_pipeline()` chains all Tier-D modules:

    Feature Extractor + Memory Retrieval
    └─► (optional) Diagnostician → Council × 3 → Verifier-Judge
            │
            ▼
        Selected proposals (combiner_name + params)
            │
            ▼
        For each proposal:
          - PerHorizonEnsemble.fit on expanding-origin folds (no leakage)
          - Composite score over OOF predictions
        │
        ▼
        Model Confidence Set on the proposal-level losses
        │
        ▼
        Final ensemble = mean of MCS-surviving combined predictions
        │
        ▼
        Conformal wrapper → prediction interval
        │
        ▼
        memory.append_run(record)

The function returns a structured `Dict[str, Any]` with all intermediate
results so the caller can audit, plot, or persist any layer.

LLM-free mode (`use_llm=False`) is used for unit tests and ablations:
the Diagnostician/Council/Judge are replaced by a deterministic
heuristic that proposes 3 default combiners (`ridge`, `random_forest`,
`simple_average`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from orchestrator.conformal import make_conformal
from orchestrator.data_contract import load_validation_from_context
from orchestrator.feature_extractor import (
    FeatureExtractorConfig,
    compute_dataset_id,
    extract_features,
)
from orchestrator.mcs import MCSConfig, model_confidence_set, squared_error_loss
from orchestrator.memory import (
    MemoryRecord,
    append_run,
    build_record,
    retrieve_similar,
)
from orchestrator.meta_combiner import (
    PerHorizonEnsemble,
    PerHorizonEnsembleConfig,
    list_meta_combiners,
    make_meta_combiner,
)
from orchestrator.utils import extract_json_object, strip_think_blocks
from orchestrator_langchain.context import get_context, set_context


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class HALMOCConfig:
    """Knobs for the HALMOC pipeline.

    Attributes:
        use_llm: if False, skip Diagnostician/Council/Judge and use the
            deterministic default proposals.
        diagnostician_model_id, judge_model_id: Ollama (or compatible)
            model identifiers.
        council_model_ids: list of *3 distinct* model identifiers — the
            heterogeneous diversity is the point.
        memory_path: override `runs.jsonl` location.
        memory_k, memory_min_sim: retrieval knobs.
        mcs_alpha, mcs_n_boot: Model Confidence Set knobs.
        conformal_name, conformal_alpha: prediction interval.
        score_weights: composite-score weights forwarded to evaluator.
        max_proposals: cap on Council proposal count fed to Φ + MCS.
        default_combiners: fallback proposals when LLM is off.
        log: optional logger callable.
    """

    use_llm: bool = True
    diagnostician_model_id: str = "qwen3:14b"
    council_model_ids: List[str] = field(
        default_factory=lambda: ["qwen3:14b", "llama3.1:8b", "gemma3:12b"]
    )
    judge_model_id: str = "qwen3:14b"
    memory_path: Optional[str] = None
    memory_k: int = 3
    memory_min_sim: float = 0.3
    mcs_alpha: float = 0.10
    mcs_n_boot: int = 999
    conformal_name: str = "aci"
    conformal_alpha: float = 0.10
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {"a_rmse": 0.3, "b_smape": 0.3, "c_mape": 0.2, "d_pocid": 0.2}
    )
    max_proposals: int = 6
    default_combiners: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"combiner_name": "ridge", "params": {"alpha": 1.0}, "share_combiner_across_horizons": False},
            {"combiner_name": "random_forest", "params": {"n_estimators": 200}, "share_combiner_across_horizons": False},
            {"combiner_name": "simple_average", "params": {}, "share_combiner_across_horizons": True},
        ]
    )
    log: Optional[Callable[[str], None]] = None


# ──────────────────────────────────────────────────────────────────────────
# OOF (out-of-fold) prediction via expanding-origin scheme
# ──────────────────────────────────────────────────────────────────────────


def _oof_predict(
    y_true: np.ndarray,
    y_preds: np.ndarray,
    proposal: Dict[str, Any],
    min_train_windows: int = 2,
) -> np.ndarray:
    """Anti-leakage out-of-fold combined prediction.

    For each window i ≥ min_train_windows, fit `PerHorizonEnsemble` on
    windows [0, i) and predict on window i.  Returns an array of shape
    `(n_windows, horizon)`; rows < min_train_windows are NaN.

    Contract: the project's base-model CSVs are laid out as
    ``N validation rows + 1 blind-test row``.  With the default
    ``train_window=4`` (3 validation + 1 blind → ``n_windows=3``), this
    gives exactly one OOF fold; ``train_window ≥ 5`` gives more.  The
    caller is expected to enforce ``n_windows ≥ 3`` upfront, so at least
    one fold always exists here.
    """

    n_w, n_m, h = y_preds.shape
    out = np.full((n_w, h), np.nan, dtype=float)
    for i in range(min_train_windows, n_w):
        cfg = PerHorizonEnsembleConfig(
            combiner_name=str(proposal.get("combiner_name", "ridge")),
            combiner_params=dict(proposal.get("params", {})),
            share_combiner_across_horizons=bool(
                proposal.get("share_combiner_across_horizons", False)
            ),
            project_predictions_simplex=bool(
                proposal.get("project_predictions_simplex", False)
            ),
        )
        try:
            ens = PerHorizonEnsemble(cfg)
            ens.fit(y_preds[:i], y_true[:i])
            out[i] = ens.predict(y_preds[i : i + 1])[0]
        except Exception as e:
            # If a fit fails (e.g. too few rows for that combiner), fall
            # back to simple average for that fold.
            out[i] = np.nanmean(y_preds[i], axis=0)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Composite score (mirrors orchestrator.evaluator presets)
# ──────────────────────────────────────────────────────────────────────────


def _composite_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: Dict[str, float]
) -> Dict[str, float]:
    """Compute RMSE, SMAPE, MAPE, POCID and a weighted composite."""

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"rmse": float("nan"), "smape": float("nan"), "mape": float("nan"),
                "pocid": float("nan"), "composite": float("nan")}
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    denom = np.abs(yt) + np.abs(yp)
    smape = float(np.mean(np.where(denom > 1e-12, 2 * np.abs(yt - yp) / denom, 0.0)))
    mape = float(np.mean(np.where(np.abs(yt) > 1e-12, np.abs(yt - yp) / np.abs(yt), 0.0)))
    if yt.size >= 2:
        pocid = float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))))
    else:
        pocid = 0.0
    a, b, c, d = (weights.get(k, 0.0) for k in ("a_rmse", "b_smape", "c_mape", "d_pocid"))
    # Simple normalization (no min-max here; the relative ordering is
    # what matters for MCS + final selection).
    composite = a * rmse + b * smape + c * mape - d * pocid
    return {"rmse": rmse, "smape": smape, "mape": mape, "pocid": pocid,
            "composite": float(composite)}


# ──────────────────────────────────────────────────────────────────────────
# Optional LLM stages
# ──────────────────────────────────────────────────────────────────────────


def _preview(text: str, n: int = 180) -> str:
    """Return a single-line preview (≤ n chars) for logging."""

    if not isinstance(text, str):
        text = str(text)
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def _extract_think(raw: str) -> str:
    """Best-effort extraction of <think>...</think> reasoning blocks."""

    if not isinstance(raw, str) or "<think>" not in raw:
        return ""
    out = []
    i = 0
    while True:
        s = raw.find("<think>", i)
        if s < 0:
            break
        e = raw.find("</think>", s)
        if e < 0:
            break
        out.append(raw[s + len("<think>") : e].strip())
        i = e + len("</think>")
    return "\n\n".join(x for x in out if x)


def _run_llm_stage(
    agent_factory,
    model_id: str,
    user_prompt: str,
    log,
    label: str = "agent",
) -> Dict[str, Any]:
    """Invoke a LangChain agent, parse its JSON output, log a preview.

    Returns a dict with *additional* bookkeeping keys so callers can
    record what the agent produced:
        - `_raw`       : untouched content string
        - `_think`     : extracted <think>...</think> reasoning
        - `_model_id`  : which model answered
        - `_duration_s`: wall-clock time for the call
    Parsed JSON keys (if any) are merged at the top level of the dict.
    """

    agent = agent_factory(model_id=model_id)
    t0 = time.perf_counter()
    response = agent.run(user_prompt)
    dt = time.perf_counter() - t0
    raw = str(response.content)
    think = _extract_think(raw)
    cleaned = strip_think_blocks(raw)
    parsed = extract_json_object(cleaned)

    log(f"  ↳ [{label}] {model_id} responded in {dt:.1f}s")
    if think:
        log(f"    think:  {_preview(think, 220)}")
    if isinstance(parsed, dict):
        keys_preview = ", ".join(list(parsed.keys())[:8])
        log(f"    keys:   {keys_preview}")
        log(f"    output: {_preview(cleaned, 220)}")
        out = dict(parsed)
    else:
        log(f"  ⚠ [{label}] JSON parse failed; keeping raw content")
        log(f"    output: {_preview(cleaned, 260)}")
        out = {"_raw": cleaned}

    out.setdefault("_raw", cleaned)
    out["_think"] = think
    out["_model_id"] = model_id
    out["_duration_s"] = round(dt, 2)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def run_halmoc_pipeline(config: Optional[HALMOCConfig] = None) -> Dict[str, Any]:
    """Run HALMOC end-to-end.

    Pre-condition: the global `CONTEXT_MEMORY` must already contain
    `all_validations` (call `generate_all_validations_context` first).
    """

    cfg = config or HALMOCConfig()
    log = cfg.log or (lambda m: print(f"[HALMOC] {m}"))

    val = load_validation_from_context()
    y_true = val.y_true
    y_preds = val.y_preds
    names = val.model_names
    log("═══════════════════════════════════════════════════════════════")
    log(
        f"[Step 0/7] Validation loaded: n_windows={val.n_windows}, "
        f"n_models={val.n_models}, horizon={val.horizon}"
    )
    # HALMOC's OOF / MCS machinery needs at least one expanding-origin
    # fold to evaluate proposals.  The project's base-model CSVs are
    # designed as "N validation rows + 1 blind-test row" with
    # n_windows = train_window - 1, so n_windows ≥ 3 (i.e.
    # train_window ≥ 4) is the contract.  Fail fast with a clear message
    # — otherwise all composites come back NaN and the ranking is
    # meaningless.
    if val.n_windows < 3:
        raise ValueError(
            f"HALMOC requires n_windows ≥ 3 validation windows (got "
            f"n_windows={val.n_windows}).  The project convention is "
            f"'N validation rows + 1 blind-test row'; n_windows = "
            f"train_window - 1.  Re-run with train_window ≥ 4 (ideally "
            f"5 or more for robust MCS/composite scoring)."
        )

    # ------------------------------------------------------------------
    # Step 1 — Feature extraction + memory retrieval
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log("[Step 1/7] Feature extraction + memory retrieval")
    features = extract_features(y_true, y_preds, names, FeatureExtractorConfig())
    dataset_id = compute_dataset_id(y_true, names, explicit_id=get_context("dataset_index"))
    set_context("halmoc_packet", {"ready": False})  # invalidate cache
    log(f"  dataset_id: {dataset_id}")
    log(f"  extracted {len(features)} features")

    memory_path = cfg.memory_path
    few_shots = []
    try:
        few_shots = [
            r.to_few_shot()
            for r in retrieve_similar(
                features,
                k=cfg.memory_k,
                min_similarity=cfg.memory_min_sim,
                exclude_dataset_ids={dataset_id},
                path=memory_path,
            )
        ]
    except Exception as e:
        log(f"  memory retrieval skipped: {e}")
    log(f"  retrieved {len(few_shots)} memory exemplars")

    # ------------------------------------------------------------------
    # Step 2 — Diagnostician / Council / Judge (optional LLM)
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log("[Step 2/7] LLM Council stage (Diagnostician → Council × N → Judge)")
    diagnosis: Dict[str, Any] = {}
    council_proposals: List[Dict[str, Any]] = []
    judge_decision: Dict[str, Any] = {}
    # Retain agent artefacts for the return dict (fed into CSV).
    diagnostician_raw: str = ""
    diagnostician_think: str = ""
    council_raws: List[Dict[str, Any]] = []
    judge_raw: str = ""
    judge_think: str = ""
    # Track failures to report once at the end of Step 2.
    failed_stages: List[Dict[str, str]] = []

    def _describe_llm_error(exc: Exception, mid: str) -> str:
        """Render a friendly diagnosis for common Ollama failure modes."""

        msg = f"{type(exc).__name__}: {exc}"
        text = str(exc).lower()
        if "not found" in text or "404" in text:
            return (
                f"{msg}\n"
                f"      → model '{mid}' not available on the Ollama server.  "
                f"Run `ollama pull {mid}` (or fix the id) and retry."
            )
        if "connection" in text or "refused" in text or "econnrefused" in text:
            return (
                f"{msg}\n"
                f"      → cannot reach Ollama server.  Is `ollama serve` "
                f"running and reachable at the default URL?"
            )
        return msg

    if cfg.use_llm:
        # Late import to keep LLM dependency optional.
        from orchestrator_langchain.agents import (
            create_council_member_agent,
            create_diagnostician_agent,
            create_judge_agent,
        )

        # Preflight: show every model that will be called before firing so
        # the user immediately sees a typo / missing model.
        log(
            f"  preflight: diagnostician={cfg.diagnostician_model_id!r}  "
            f"council={list(cfg.council_model_ids)!r}  "
            f"judge={cfg.judge_model_id!r}"
        )

        log(f"  [2a] Diagnostician  (model={cfg.diagnostician_model_id})")
        try:
            diagnosis = _run_llm_stage(
                create_diagnostician_agent,
                cfg.diagnostician_model_id,
                "Diagnose this dataset.",
                log,
                label="diagnostician",
            )
            diagnostician_raw = str(diagnosis.get("_raw", ""))
            diagnostician_think = str(diagnosis.get("_think", ""))
            regime = diagnosis.get("regime_signature") or diagnosis.get("regime")
            if regime:
                log(f"    regime_signature: {regime}")
            set_context("halmoc_diagnosis", diagnosis)
        except Exception as e:
            msg = _describe_llm_error(e, cfg.diagnostician_model_id)
            log(f"  ⚠ [diagnostician] FAILED — {msg}")
            failed_stages.append(
                {"stage": "diagnostician", "model_id": cfg.diagnostician_model_id,
                 "error": str(e)}
            )
            diagnosis = {}

        log(f"  [2b] Council ({len(cfg.council_model_ids)} members, serial)")
        for idx, mid in enumerate(cfg.council_model_ids):
            log(f"    member #{idx+1}: {mid}")
            try:
                cm = _run_llm_stage(
                    create_council_member_agent,
                    mid,
                    "Propose strategies.",
                    log,
                    label=f"council#{idx+1}",
                )
            except Exception as e:
                msg = _describe_llm_error(e, mid)
                log(f"    ⚠ [council#{idx+1}] FAILED — {msg}")
                failed_stages.append(
                    {"stage": f"council#{idx+1}", "model_id": mid,
                     "error": str(e)}
                )
                council_raws.append({
                    "member_idx": idx, "model_id": mid, "raw": "", "think": "",
                    "duration_s": None, "proposals": [], "error": str(e),
                })
                continue
            council_raws.append({
                "member_idx": idx,
                "model_id": mid,
                "raw": str(cm.get("_raw", "")),
                "think": str(cm.get("_think", "")),
                "duration_s": cm.get("_duration_s"),
                "proposals": cm.get("proposals", []),
            })
            proposals = cm.get("proposals", []) or []
            names_preview = [str(p.get("combiner_name", "?")) for p in proposals]
            log(f"    proposed combiners: {names_preview}")
            for p in proposals:
                council_proposals.append({**p, "_proposed_by": mid})
        # Cap at max_proposals (preserve order; could be smarter)
        if len(council_proposals) > cfg.max_proposals:
            log(
                f"    ⚠ {len(council_proposals)} proposals collected; capping "
                f"to max_proposals={cfg.max_proposals}"
            )
        council_proposals = council_proposals[: cfg.max_proposals]
        surviving_members = sum(1 for cr in council_raws if not cr.get("error"))
        log(
            f"    council summary: {surviving_members}/"
            f"{len(cfg.council_model_ids)} members responded, "
            f"{len(council_proposals)} proposals kept"
        )
        set_context("halmoc_council_proposals", council_proposals)

        log(f"  [2c] Judge  (model={cfg.judge_model_id})")
        try:
            judge_decision = _run_llm_stage(
                create_judge_agent,
                cfg.judge_model_id,
                "Rank the proposals.",
                log,
                label="judge",
            )
            judge_raw = str(judge_decision.get("_raw", ""))
            judge_think = str(judge_decision.get("_think", ""))
            ranked = judge_decision.get("ranked_proposals", [])
            keep_idx = [int(rid.split("_")[-1]) - 1 for rid in ranked
                        if isinstance(rid, str) and rid.startswith("p_")]
            keep_idx = [i for i in keep_idx if 0 <= i < len(council_proposals)]
            if keep_idx:
                council_proposals = [council_proposals[i] for i in keep_idx]
            log(
                f"    judge.confidence = {judge_decision.get('confidence', 'n/a')!r}, "
                f"kept {len(council_proposals)}/{len(ranked) or len(council_proposals)} proposals"
            )
        except Exception as e:
            msg = _describe_llm_error(e, cfg.judge_model_id)
            log(f"  ⚠ [judge] FAILED — {msg}  (keeping all council proposals)")
            failed_stages.append(
                {"stage": "judge", "model_id": cfg.judge_model_id,
                 "error": str(e)}
            )
            judge_decision = {}

        if failed_stages:
            log(
                f"  Step 2 finished with {len(failed_stages)} LLM failure(s): "
                f"{[f['stage']+':'+f['model_id'] for f in failed_stages]}"
            )
    else:
        log("  LLM stages skipped (use_llm=False); using default combiners")
        council_proposals = list(cfg.default_combiners)

    if not council_proposals:
        log("  ⚠ No proposals from Judge; falling back to defaults")
        council_proposals = list(cfg.default_combiners)

    # ------------------------------------------------------------------
    # Step 3 — Out-of-fold predictions per proposal
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log(
        f"[Step 3/7] Out-of-fold evaluation of {len(council_proposals)} proposals "
        f"(expanding-origin, min_train_windows=2)"
    )
    proposal_oof: List[np.ndarray] = []
    proposal_scores: List[Dict[str, float]] = []
    for i, p in enumerate(council_proposals):
        oof = _oof_predict(y_true, y_preds, p)
        proposal_oof.append(oof)
        sc = _composite_score(y_true, oof, cfg.score_weights)
        proposal_scores.append(sc)
        params_str = json.dumps(p.get("params", {}), separators=(",", ":"))[:60]
        log(
            f"  [{i+1}/{len(council_proposals)}] {p.get('combiner_name','?'):16s} "
            f"params={params_str:60s}  composite={sc['composite']:.4f}  "
            f"rmse={sc['rmse']:.4f}"
        )
    # ------------------------------------------------------------------
    # Step 4 — Model Confidence Set on proposal-level losses
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log(
        f"[Step 4/7] Model Confidence Set  (α={cfg.mcs_alpha}, "
        f"n_boot={cfg.mcs_n_boot}, stationary bootstrap)"
    )
    proposal_names = [
        f"{i+1}:{p.get('combiner_name','?')}" for i, p in enumerate(council_proposals)
    ]
    # Build a (T, K) loss matrix aligned by valid rows
    losses_per_proposal = []
    for oof in proposal_oof:
        diff = oof - y_true  # (n_w, h)
        losses_per_proposal.append((diff ** 2).reshape(-1))
    L = np.vstack(losses_per_proposal).T  # (n_w*h, K)
    mcs = model_confidence_set(
        L,
        proposal_names,
        MCSConfig(alpha=cfg.mcs_alpha, n_boot=cfg.mcs_n_boot, random_state=0),
    )
    log(f"  surviving ({len(mcs.surviving)}): {mcs.surviving}")
    if mcs.eliminated_order:
        log(f"  eliminated order: {mcs.eliminated_order}")

    # ------------------------------------------------------------------
    # Step 5 — Final ensemble: mean of MCS-surviving combined predictions
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log("[Step 5/7] Final ensemble on OOF (mean of MCS survivors)")
    survivor_idx = [
        proposal_names.index(name) for name in mcs.surviving if name in proposal_names
    ] or [int(np.argmin([s["composite"] for s in proposal_scores]))]
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            final_oof = np.nanmean(
                np.stack([proposal_oof[i] for i in survivor_idx]), axis=0
            )
    final_score = _composite_score(y_true, final_oof, cfg.score_weights)
    log(
        f"  survivors used: {[proposal_names[i] for i in survivor_idx]}  "
        f"→ composite={final_score['composite']:.4f}  rmse={final_score['rmse']:.4f}"
    )

    # ------------------------------------------------------------------
    # Step 5b — Final blind-window prediction
    #   • read context['predictions'] = {model: [h1,h2,...]}
    #   • build final_X shape (1, n_models, horizon)
    #   • for each MCS-surviving proposal: fit on FULL validation,
    #     predict on final_X, average across survivors
    #   • predictions per-model stay tracked for diagnostics
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log(f"[Step 5b/7] Final blind-window prediction (horizon={val.horizon})")
    final_test_predictions: List[float] = []
    winning_weights_per_horizon: Optional[List[List[float]]] = None
    selected_base_models: List[str] = list(names)
    try:
        final_preds_dict = get_context("predictions") or {}
        # Build final_X in the SAME model order as validation (names).
        final_X_rows = []
        missing_models: List[str] = []
        for m in names:
            if m in final_preds_dict and final_preds_dict[m] is not None:
                arr = np.asarray(final_preds_dict[m], dtype=float)
                if arr.size < val.horizon:
                    # Pad with NaN so downstream code still sees a valid slot.
                    pad = np.full(val.horizon - arr.size, np.nan)
                    arr = np.concatenate([arr, pad])
                final_X_rows.append(arr[: val.horizon])
            else:
                missing_models.append(m)
                final_X_rows.append(np.full(val.horizon, np.nan))
        if missing_models:
            log(f"  ⚠ final_test preds missing for: {missing_models}")
        final_X = np.array(final_X_rows, dtype=float).reshape(1, val.n_models, val.horizon)

        # Fit each surviving proposal on ALL validation windows and predict.
        survivor_blind: List[np.ndarray] = []
        for idx in survivor_idx:
            p = council_proposals[idx]
            cfg_e = PerHorizonEnsembleConfig(
                combiner_name=str(p.get("combiner_name", "ridge")),
                combiner_params=dict(p.get("params", {})),
                share_combiner_across_horizons=bool(
                    p.get("share_combiner_across_horizons", False)
                ),
                project_predictions_simplex=bool(
                    p.get("project_predictions_simplex", False)
                ),
            )
            try:
                ens = PerHorizonEnsemble(cfg_e)
                ens.fit(y_preds, y_true)
                pred = ens.predict(final_X)[0]  # (horizon,)
                survivor_blind.append(pred)
                # Prefer to expose weights from the first successful linear combiner.
                if winning_weights_per_horizon is None:
                    w = ens.get_weights_per_horizon()
                    if w is not None:
                        winning_weights_per_horizon = w.tolist()
            except Exception as e:
                log(f"  survivor #{idx} blind-fit failed: {e}")
        if survivor_blind:
            blind_stack = np.vstack(survivor_blind)  # (K_survivors, horizon)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                final_test_predictions = np.nanmean(blind_stack, axis=0).tolist()
            first6 = ", ".join(f"{v:.2f}" for v in final_test_predictions[:6])
            log(
                f"  mean of {len(survivor_blind)} survivor combiner(s); "
                f"first6={first6}{'…' if val.horizon > 6 else ''}"
            )
        else:
            # Last-resort fallback: simple average of base preds (never leaks).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                final_test_predictions = np.nanmean(final_X[0], axis=0).tolist()
            log("  ⚠ all survivor combiners failed; using simple average of base preds")
    except Exception as e:
        log(f"  ⚠ final blind-window step failed: {e}")
        final_test_predictions = []

    # ------------------------------------------------------------------
    # Step 6 — Conformal interval (calibrated on OOF, applied to OOF)
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log(
        f"[Step 6/7] Conformal interval  (method={cfg.conformal_name}, "
        f"α={cfg.conformal_alpha})"
    )
    intervals: Dict[str, Any] = {}
    try:
        wrapper = make_conformal(cfg.conformal_name, alpha=cfg.conformal_alpha)
        # Use the second half of OOF as calibration; first half as fit
        valid_rows = np.where(np.all(np.isfinite(final_oof), axis=1))[0]
        if valid_rows.size >= 4:
            mid = valid_rows.size // 2
            cal_idx = valid_rows[:mid]
            test_idx = valid_rows[mid:]
            wrapper.calibrate(
                y_true[cal_idx].reshape(-1), final_oof[cal_idx].reshape(-1)
            )
            lo, hi = wrapper.interval(final_oof[test_idx].reshape(-1))
            cov = float(
                np.mean(
                    (y_true[test_idx].reshape(-1) >= lo)
                    & (y_true[test_idx].reshape(-1) <= hi)
                )
            )
            intervals = {
                "method": cfg.conformal_name,
                "alpha": cfg.conformal_alpha,
                "empirical_coverage": cov,
                "mean_width": float(np.mean(hi - lo)),
            }
            log(
                f"  empirical coverage={cov:.3f} "
                f"(nominal {1.0 - cfg.conformal_alpha:.2f})  "
                f"mean_width={intervals['mean_width']:.3f}"
            )
        else:
            log(
                f"  ⚠ skipped — only {valid_rows.size} valid OOF rows "
                f"(need ≥ 4 for split calibration)"
            )
    except Exception as e:
        log(f"  ⚠ conformal step failed: {e}")
        intervals = {"error": str(e)}

    # ------------------------------------------------------------------
    # Step 7 — Persist to memory
    # ------------------------------------------------------------------
    log("───────────────────────────────────────────────────────────────")
    log("[Step 7/7] Persist run to experiential memory")
    winner_proposal = council_proposals[survivor_idx[0]]
    record = build_record(
        dataset_id=dataset_id,
        features=features,
        n_models=val.n_models,
        n_windows=val.n_windows,
        horizon=val.horizon,
        winning_strategy=str(winner_proposal.get("combiner_name", "?")),
        winning_strategy_type="halmoc_meta_combiner",
        winning_params=dict(winner_proposal.get("params", {})),
        score=float(final_score["composite"]),
        score_vs_baseline_mean_pct=float(
            100.0
            * (
                _composite_score(y_true, np.nanmean(y_preds, axis=1), cfg.score_weights)["composite"]
                - final_score["composite"]
            )
            / max(
                abs(_composite_score(y_true, np.nanmean(y_preds, axis=1), cfg.score_weights)["composite"]),
                1e-12,
            )
        ),
        debate_triggered=False,
        notes=json.dumps(
            {"mcs_surviving": list(mcs.surviving),
             "judge_confidence": judge_decision.get("confidence", "n/a"),
             "regime": diagnosis.get("regime_signature", "n/a")},
            separators=(",", ":"),
        ),
    )
    try:
        path_used = append_run(record, memory_path)
        log(f"  persisted to {path_used}")
    except Exception as e:
        log(f"  ⚠ memory persist failed: {e}")
        path_used = None
    log(f"  winning_proposal: {winner_proposal}")
    log("═══════════════════════════════════════════════════════════════")

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    return {
        "dataset_id": dataset_id,
        "features": features,
        "memory_few_shots": few_shots,
        "diagnosis": diagnosis,
        "council_proposals": council_proposals,
        "judge": judge_decision,
        "proposal_scores": proposal_scores,
        "mcs": mcs.to_dict(),
        "final_score": final_score,
        "final_oof_predictions": final_oof.tolist(),
        "final_test_predictions": final_test_predictions,
        "winning_weights_per_horizon": winning_weights_per_horizon,
        "selected_base_models": selected_base_models,
        "conformal": intervals,
        "memory_path": str(path_used) if path_used else None,
        "winner_proposal": winner_proposal,
        "n_models": int(val.n_models),
        "n_windows": int(val.n_windows),
        "horizon": int(val.horizon),
        "model_names": list(names),
        # Per-agent artefacts for CSV auditability.
        "diagnostician_raw": diagnostician_raw,
        "diagnostician_think": diagnostician_think,
        "diagnostician_model_id": cfg.diagnostician_model_id if cfg.use_llm else None,
        "council_raws": council_raws,
        "council_model_ids": list(cfg.council_model_ids) if cfg.use_llm else [],
        "judge_raw": judge_raw,
        "judge_think": judge_think,
        "judge_model_id": cfg.judge_model_id if cfg.use_llm else None,
        "failed_stages": failed_stages if cfg.use_llm else [],
    }
