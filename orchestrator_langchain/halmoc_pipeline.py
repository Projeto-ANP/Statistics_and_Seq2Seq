"""Adapter that exposes the HALMOC pipeline with the result schema
expected by `run_tsf_orchestrator.py`.

The rest of the codebase (CSV-writer, diagnostics dumping) was built
around the older Pattern-Analyst/Proposer/Skeptic/Statistician agents
and keys like `result["debate"]`, `result["eval"]["ranking"]`,
`result["best"]`, `result["llm_artifacts"]`, `result["explanations"]`.

HALMOC replaces all those agents with Diagnostician / Council / Judge +
a deterministic Meta-Combiner + MCS.  To avoid touching the whole
`run_tsf_orchestrator.py`, we produce a *superset* result dict here:

    - HALMOC-native keys (halmoc: {...}) carry the new artefacts.
    - Legacy keys are synthesised from the HALMOC output so the CSV
      writer does not need special-casing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from orchestrator.halmoc_pipeline import HALMOCConfig, run_halmoc_pipeline


def _as_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def run_halmoc_langchain_pipeline(
    model_id: str = "mychen76/qwen3_cline_roocode:14b",
    debug: bool = False,
    rolling_mode: str = "expanding",
    train_window: int = 5,
    require_tool_call: bool = True,
    llm_logs: bool = True,
    use_llm: bool = True,
    diagnostician_model_id: Optional[str] = None,
    council_model_ids: Optional[List[str]] = None,
    judge_model_id: Optional[str] = None,
    memory_path: Optional[str] = None,
    mcs_alpha: float = 0.10,
    mcs_n_boot: int = 999,
    conformal_name: str = "aci",
    conformal_alpha: float = 0.10,
    max_proposals: int = 6,
) -> Dict[str, Any]:
    """Run HALMOC and emit a result dict compatible with the CSV writer.

    Keyword args are a permissive superset of the legacy
    `run_langchain_pipeline` signature so the caller can drop-in replace
    the old function without editing its call-site.  `rolling_mode`,
    `train_window`, `require_tool_call` and `debug` are accepted for
    call-site compatibility but the HALMOC pipeline does its own
    anti-leakage expanding-origin scheme; they are recorded in the
    description only.
    """

    logs: List[str] = []

    def _log(msg: str) -> None:
        if llm_logs:
            print(f"[HALMOC] {msg}")
        logs.append(str(msg))

    council_ids = council_model_ids or [model_id, model_id, model_id]
    cfg = HALMOCConfig(
        use_llm=bool(use_llm),
        diagnostician_model_id=diagnostician_model_id or model_id,
        council_model_ids=list(council_ids),
        judge_model_id=judge_model_id or model_id,
        memory_path=memory_path,
        mcs_alpha=float(mcs_alpha),
        mcs_n_boot=int(mcs_n_boot),
        conformal_name=str(conformal_name),
        conformal_alpha=float(conformal_alpha),
        max_proposals=int(max_proposals),
        log=_log,
    )

    try:
        halmoc_out = run_halmoc_pipeline(cfg)
    except Exception as e:
        return {
            "success": False,
            "description": {
                "mode": "halmoc",
                "error": "HALMOC pipeline failed",
                "exception": str(e),
                "exception_type": type(e).__name__,
                "rolling_mode": rolling_mode,
                "train_window": train_window,
                "logs": logs,
            },
            "result": [],
            "debate": {"debate_ran": False, "debate_trigger": "exception"},
            "halmoc": None,
        }

    winner = halmoc_out.get("winner_proposal", {}) or {}
    winner_name = str(winner.get("combiner_name", "?"))
    winner_params = dict(winner.get("params", {}))

    # Proposal ranking → legacy eval.ranking (for CSV final_candidate_names).
    council_proposals = halmoc_out.get("council_proposals", []) or []
    proposal_scores = halmoc_out.get("proposal_scores", []) or []
    ranking: List[Dict[str, Any]] = []
    for p, sc in zip(council_proposals, proposal_scores):
        ranking.append(
            {
                "name": str(p.get("combiner_name", "?")),
                "params": dict(p.get("params", {})),
                "score": float(sc.get("composite", float("nan"))),
                "rmse": float(sc.get("rmse", float("nan"))),
                "smape": float(sc.get("smape", float("nan"))),
                "mape": float(sc.get("mape", float("nan"))),
                "pocid": float(sc.get("pocid", float("nan"))),
            }
        )
    ranking.sort(key=lambda r: (r["score"] if r["score"] == r["score"] else float("inf")))

    final_preds = list(halmoc_out.get("final_test_predictions") or [])

    # Build per-horizon weights dict for the CSV predict_debug column.
    weights_by_horizon_dict: Dict[str, Dict[str, float]] = {}
    wph = halmoc_out.get("winning_weights_per_horizon")
    model_names = halmoc_out.get("model_names", []) or []
    if isinstance(wph, list) and model_names:
        for h_idx, row in enumerate(wph):
            if not isinstance(row, list):
                continue
            mapping = {}
            for m, w in zip(model_names, row):
                try:
                    mapping[str(m)] = float(w)
                except Exception:
                    continue
            weights_by_horizon_dict[f"h{h_idx+1}"] = mapping

    predict_debug = {
        "method": winner_name,
        "horizon_count": int(halmoc_out.get("horizon", 0) or 0),
        "mcs_surviving": list((halmoc_out.get("mcs") or {}).get("surviving", [])),
        "weights_by_horizon": weights_by_horizon_dict or None,
    }

    judge = halmoc_out.get("judge", {}) or {}
    diagnosis = halmoc_out.get("diagnosis", {}) or {}
    conformal = halmoc_out.get("conformal", {}) or {}

    description_obj = {
        "mode": "halmoc",
        "dataset_id": halmoc_out.get("dataset_id"),
        "score_preset": "halmoc_composite",
        "rolling_mode": rolling_mode,
        "train_window": train_window,
        "best": {"name": winner_name, "params": winner_params},
        "halmoc": {
            "n_models": halmoc_out.get("n_models"),
            "n_windows": halmoc_out.get("n_windows"),
            "horizon": halmoc_out.get("horizon"),
            "diagnosis_regime": diagnosis.get("regime_signature"),
            "mcs_surviving": list((halmoc_out.get("mcs") or {}).get("surviving", [])),
            "mcs_eliminated_order": list(
                (halmoc_out.get("mcs") or {}).get("eliminated_order", [])
            ),
            "judge_confidence": judge.get("confidence"),
            "conformal": conformal,
            "winning_combiner": {
                "name": winner_name,
                "params": winner_params,
                "share_horizon": bool(
                    winner.get("share_combiner_across_horizons", False)
                ),
            },
            "final_score": halmoc_out.get("final_score"),
            "memory_path": halmoc_out.get("memory_path"),
            "agents": {
                "diagnostician_model_id": halmoc_out.get("diagnostician_model_id"),
                "council_model_ids": list(halmoc_out.get("council_model_ids", []) or []),
                "judge_model_id": halmoc_out.get("judge_model_id"),
            },
            "failed_stages": list(halmoc_out.get("failed_stages", []) or []),
        },
        "predict_debug": predict_debug,
        "tool_validation": {"tool_missing": False, "tools_called": [
            "diagnostician_brief", "council_brief", "judge_brief"
        ] if use_llm else []},
    }

    # Synthesise legacy keys so existing CSV writer keeps working.
    explanations = {
        "orchestrator_reasoning": (
            diagnosis.get("narrative")
            or diagnosis.get("reasoning")
            or f"HALMOC picked {winner_name} (MCS α={mcs_alpha}, survivors="
            f"{len(predict_debug['mcs_surviving'])})."
        ),
        "orchestrator_when_good": diagnosis.get("when_good", ""),
        "orchestrator_debate_notes": judge.get("tie_resolution_notes", ""),
        "skeptic_rationale": "",
        "statistician_when_good": "",
    }

    diagnostician_raw = str(halmoc_out.get("diagnostician_raw", "") or "")
    diagnostician_think = str(halmoc_out.get("diagnostician_think", "") or "")
    diagnostician_model = halmoc_out.get("diagnostician_model_id")
    council_raws = halmoc_out.get("council_raws", []) or []
    council_model_ids = halmoc_out.get("council_model_ids", []) or []
    judge_raw = str(halmoc_out.get("judge_raw", "") or "")
    judge_think = str(halmoc_out.get("judge_think", "") or "")
    judge_model = halmoc_out.get("judge_model_id")

    llm_artifacts = {
        "raw": {
            "diagnostician": diagnostician_raw or _as_json(diagnosis),
            "judge": judge_raw or _as_json(judge),
            "council": _as_json(
                [
                    {
                        "member_idx": cr.get("member_idx"),
                        "model_id": cr.get("model_id"),
                        "raw": cr.get("raw", ""),
                        "think": cr.get("think", ""),
                        "duration_s": cr.get("duration_s"),
                        "proposals": cr.get("proposals", []),
                    }
                    for cr in council_raws
                ]
            )
            if council_raws
            else _as_json([{"proposal": p} for p in council_proposals]),
            "halmoc_logs": "\n".join(logs[-200:]),
        },
        "parsed": {
            "diagnostician": diagnosis,
            "diagnostician_think": diagnostician_think,
            "diagnostician_model_id": diagnostician_model,
            "judge": judge,
            "judge_think": judge_think,
            "judge_model_id": judge_model,
            "council_proposals": council_proposals,
            "council_raws": council_raws,
            "council_model_ids": council_model_ids,
            "mcs": halmoc_out.get("mcs"),
            "conformal": conformal,
            "memory_few_shots": halmoc_out.get("memory_few_shots", []),
        },
    }

    return {
        "success": True,
        "description": description_obj,
        "result": final_preds,
        "debate": {
            "debate_ran": False,
            "debate_trigger": "halmoc_no_debate",
            "best_pre_debate": {"name": winner_name, "params": winner_params},
            "best_post_debate": {"name": winner_name, "params": winner_params},
        },
        "eval": {"ranking": ranking},
        "best": {"name": winner_name, "params": winner_params},
        "explanations": explanations,
        "llm_artifacts": llm_artifacts,
        "halmoc": halmoc_out,
    }
