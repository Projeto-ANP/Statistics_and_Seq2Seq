"""Deterministic context tools for the HALMOC LLM agents.

These functions are the *only* way the Diagnostician / Council / Judge
LLMs receive ground-truth numbers.  Every value comes from
deterministic Python (`feature_extractor`, `diagnostics`, `mcs`,
`memory`, `meta_combiner.list_meta_combiners`) — the LLMs never compute
statistics themselves.  This is the architectural firewall that
defends against LLM-as-judge bias (Zheng et al. 2023 NeurIPS).

Each tool returns a JSON string (LangChain expects scalar tool
outputs).  Each tool is keyed to a single agent role:

| Tool                       | Used by      |
| -------------------------- | ------------ |
| `diagnostician_brief_tool` | Diagnostician|
| `council_brief_tool`       | Council × 3  |
| `judge_brief_tool`         | Judge        |
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from orchestrator.data_contract import load_validation_from_context
from orchestrator.diagnostics import (
    bias_per_horizon,
    drift_signal,
    error_similarity_matrix,
    heteroscedasticity_ratio,
    hurst_rs,
    ljung_box_pvalue,
    rank_stability_kendall,
    spectral_entropy,
)
from orchestrator.feature_extractor import (
    FeatureExtractorConfig,
    compute_dataset_id,
    extract_features,
)
from orchestrator.mcs import MCSConfig, model_confidence_set, squared_error_loss
from orchestrator.memory import RetrievedExample, retrieve_similar
from orchestrator.meta_combiner import list_meta_combiners
from orchestrator_langchain.context import get_context, set_context


def _round_dict(d: Dict[str, Any], digits: int = 4) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, digits)
        elif isinstance(v, np.floating):
            out[k] = round(float(v), digits)
        elif isinstance(v, dict):
            out[k] = _round_dict(v, digits)
        else:
            out[k] = v
    return out


# ──────────────────────────────────────────────────────────────────────────
# Shared computation: cache features, diagnostics, retrieved memory in
# context to avoid recomputing across agents.
# ──────────────────────────────────────────────────────────────────────────


def _build_or_get_packet() -> Dict[str, Any]:
    """Build (or fetch from cache) the deterministic diagnostic packet."""

    cached = get_context("halmoc_packet")
    if isinstance(cached, dict) and cached.get("ready"):
        return cached

    val = load_validation_from_context()
    y_true = val.y_true
    y_preds = val.y_preds
    names = val.model_names

    cfg = FeatureExtractorConfig()
    features = extract_features(y_true, y_preds, names, cfg)
    dataset_id = compute_dataset_id(y_true, names, explicit_id=get_context("dataset_index"))

    # Diagnostics (best-effort: failures degrade to None instead of raising)
    diags: Dict[str, Any] = {}
    try:
        diags["bias_per_horizon"] = {
            n: [round(float(x), 4) for x in arr.tolist()]
            for n, arr in bias_per_horizon(y_true, y_preds, names).items()
        }
    except Exception as e:
        diags["bias_per_horizon"] = f"<error: {e}>"
    try:
        diags["ljung_box_p"] = {
            n: round(float(p), 4)
            for n, p in ljung_box_pvalue(y_true, y_preds, names).items()
        }
    except Exception as e:
        diags["ljung_box_p"] = f"<error: {e}>"
    try:
        diags["heteroscedasticity"] = {
            n: round(float(r), 4)
            for n, r in heteroscedasticity_ratio(y_true, y_preds, names).items()
        }
    except Exception as e:
        diags["heteroscedasticity"] = f"<error: {e}>"
    try:
        diags["drift_signal"] = round(float(drift_signal(y_true)), 4)
    except Exception as e:
        diags["drift_signal"] = f"<error: {e}>"
    try:
        diags["spectral_entropy"] = round(float(spectral_entropy(y_true)), 4)
    except Exception as e:
        diags["spectral_entropy"] = f"<error: {e}>"
    try:
        diags["hurst"] = round(float(hurst_rs(y_true)), 4)
    except Exception as e:
        diags["hurst"] = f"<error: {e}>"
    try:
        diags["kendall_rank_stability"] = round(
            float(rank_stability_kendall(y_true, y_preds, names)), 4
        )
    except Exception as e:
        diags["kendall_rank_stability"] = f"<error: {e}>"
    try:
        sim = error_similarity_matrix(y_true, y_preds, names)
        if isinstance(sim, dict) and "matrix" in sim:
            diags["error_similarity"] = {
                "models": sim.get("models"),
                "matrix": [
                    [round(float(v), 3) for v in row] for row in sim.get("matrix", [])
                ],
            }
        else:
            diags["error_similarity"] = sim
    except Exception as e:
        diags["error_similarity"] = f"<error: {e}>"

    # MCS loose pass over base models (gives Council + Judge cheap evidence)
    try:
        losses = squared_error_loss(y_true, y_preds)
        mcs_loose = model_confidence_set(
            losses, names, MCSConfig(alpha=0.25, n_boot=399, random_state=0)
        ).to_dict()
    except Exception as e:
        mcs_loose = {"error": str(e)}

    # Memory retrieval (skip if no log yet)
    try:
        few_shots = [
            r.to_few_shot()
            for r in retrieve_similar(
                features,
                k=int(get_context("memory_k", 3)),
                min_similarity=float(get_context("memory_min_sim", 0.3)),
                exclude_dataset_ids={dataset_id},
            )
        ]
    except Exception as e:
        few_shots = []

    packet = {
        "ready": True,
        "dataset_id": dataset_id,
        "n_models": int(val.n_models),
        "n_windows": int(val.n_windows),
        "horizon": int(val.horizon),
        "model_names": list(names),
        "feature_vector": _round_dict(features, 4),
        "diagnostics": diags,
        "mcs_loose": mcs_loose,
        "memory_few_shots": few_shots,
        "combiner_registry": list_meta_combiners(),
    }
    set_context("halmoc_packet", packet)
    return packet


# ──────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────


def diagnostician_brief_tool() -> str:
    """Return the diagnostic packet for the Diagnostician agent."""

    pkt = _build_or_get_packet()
    out = {
        "dataset_id": pkt["dataset_id"],
        "n_models": pkt["n_models"],
        "n_windows": pkt["n_windows"],
        "horizon": pkt["horizon"],
        "model_names": pkt["model_names"],
        "feature_vector": pkt["feature_vector"],
        "diagnostics": pkt["diagnostics"],
        "mcs_loose": pkt["mcs_loose"],
        "memory_few_shots": pkt["memory_few_shots"],
    }
    return json.dumps(out, separators=(",", ":"), default=str)


def council_brief_tool() -> str:
    """Brief for each Council member.

    Returns the Diagnostician output (set in context by the orchestrator
    after the Diagnostician runs) plus the deterministic packet plus
    the available combiner registry.
    """

    pkt = _build_or_get_packet()
    diagnosis = get_context("halmoc_diagnosis", {})
    out = {
        "diagnosis": diagnosis,
        "feature_vector": pkt["feature_vector"],
        "diagnostics": pkt["diagnostics"],
        "model_names": pkt["model_names"],
        "n_models": pkt["n_models"],
        "n_windows": pkt["n_windows"],
        "horizon": pkt["horizon"],
        "memory_few_shots": pkt["memory_few_shots"],
        "combiner_registry": pkt["combiner_registry"],
    }
    return json.dumps(out, separators=(",", ":"), default=str)


def judge_brief_tool() -> str:
    """Brief for the Verifier-Judge.

    Returns the Diagnostician output, the union of council proposals
    (anonymised IDs), the loose MCS, and the DM matrix.
    """

    pkt = _build_or_get_packet()
    proposals = get_context("halmoc_council_proposals", [])
    diagnosis = get_context("halmoc_diagnosis", {})
    # Anonymise / re-id proposals for positional bias mitigation
    anon = []
    for i, p in enumerate(proposals):
        anon.append({"id": f"p_{i+1}", **{k: v for k, v in p.items() if k != "id"}})

    # DM matrix from diagnostics if present
    dm = pkt.get("diagnostics", {}).get("error_similarity", None)

    out = {
        "diagnosis": diagnosis,
        "proposals": anon,
        "mcs_loose": pkt["mcs_loose"],
        "error_similarity": dm,
        "model_names": pkt["model_names"],
        "memory_few_shots": pkt["memory_few_shots"],
    }
    return json.dumps(out, separators=(",", ":"), default=str)
