from __future__ import annotations

from typing import List, Optional

from agno.agent import Agent
from agno.models.ollama import Ollama

from orchestrator.tools import build_debate_packet_tool, evaluate_strategies_tool, proposer_brief_tool


DEFAULT_MODEL_ID = "mychen76/qwen3_cline_roocode:4b"


PROPOSER_DESCRIPTION = """You are a forecasting strategy PROPOSER.
You propose candidate multi-step combination strategies, as STRICT JSON only."""

SKEPTIC_DESCRIPTION = """You are an anti-leakage SKEPTIC/AUDITOR.
You reject or rewrite any strategy that uses y_true from the same window being evaluated."""

STATISTICIAN_DESCRIPTION = """You are a STATISTICIAN.
You improve robustness: regularization, shrinkage, top-k, stability constraints."""

ORCHESTRATOR_DESCRIPTION = """You are the ORCHESTRATOR.
You must evaluate candidates using the deterministic evaluator tool and return the final ranked decision."""


CANDIDATE_RULES = [
    "Output MUST be valid JSON (no markdown).",
    "Output MUST be either a JSON list of candidates or an object {\"candidates\": [...]}.",
    "Each candidate MUST include: name, type, description, formula, learns_weights, constraints, risks, validation_plan, params.",
    "params.method MUST be one of: mean, median, trimmed_mean, best_single, best_per_horizon, topk_mean_per_horizon, inverse_rmse_weights_per_horizon, ridge_stacking_per_horizon, exp_weighted_average_per_horizon, poly_weighted_average_per_horizon, ade_dynamic_error_per_horizon.",
    "If method=trimmed_mean include params.trim_ratio (0..0.4).",
    "If method=topk_mean_per_horizon include params.top_k.",
    "If method=inverse_rmse_weights_per_horizon include params.top_k and optionally shrinkage.",
    "If method=ridge_stacking_per_horizon include params.l2 and optionally top_k.",
    "Never propose strategies that fit weights using ALL windows and then report on those same windows (leakage).",
]


def _ollama(model_id: str, temperature: float = 0.15) -> Ollama:
    # These options follow the recommended "Optimizing for Tool Calling" settings.
    # Note: specific Ollama builds/models may ignore unsupported keys.
    return Ollama(
        id=model_id,
        options={
            "num_ctx": 65536,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_keep": 1024,
            "num_predict": 16384,
            "keep_alive": "5m",
        },
    )


def create_proposer_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> Agent:
    return Agent(
        tool_choice="required",
        model=_ollama(model_id, temperature=0.25),
        tools=[proposer_brief_tool],
        description=PROPOSER_DESCRIPTION,
        instructions=[
            "FIRST call your own tool function proposer_brief_tool() to continue",
            "CRITICAL: Output MUST be VALID JSON (no markdown, no extra text before/after JSON).",
            "FIRST call proposer_brief_tool() to get validation_summary + candidate_library + score_presets.",
            "Then output ONLY a JSON object with EXACTLY these keys:",
            "  \"selected_names\": [], \"params_overrides\": {}, \"score_preset\": \"\", \"force_debate\": false, \"debate_margin\": 0.02, \"rationale\": \"\"",
            "Valid JSON example: {\"selected_names\": [\"baseline_mean\", \"topk_mean_per_horizon_k5\"], \"params_overrides\": {}, \"score_preset\": \"balanced\", \"force_debate\": false, \"debate_margin\": 0.02, \"rationale\": \"text\"}",
            "Rules:",
            "- Do not create new candidates. Select by name only.",
            "- Do not change params.method.",
            "- Prefer returning at least 3 candidates.",
            "- ALWAYS return valid JSON with all 6 required keys. If uncertain: {\"selected_names\": [\"baseline_mean\"], \"params_overrides\": {}, \"score_preset\": \"balanced\", \"force_debate\": false, \"debate_margin\": 0.02, \"rationale\": \"Conservative selection\"}",
        ],
        markdown=False,
        debug_mode=debug,
    )


def create_skeptic_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> Agent:
    return Agent(
        tool_choice="required",
        model=_ollama(model_id, temperature=0.25),
        tools=[build_debate_packet_tool],
        description=SKEPTIC_DESCRIPTION,
        instructions=[
            "CRITICAL: Output MUST be VALID JSON (no markdown, no extra text before/after JSON).",
            "FIRST call build_debate_packet_tool() to see real evaluation numbers (inputs are provided via context).",
            "Then output ONLY a JSON object with EXACTLY these keys:",
            "  \"add_names\": [], \"remove_names\": [], \"params_overrides\": {}, \"rationale\": \"\", \"changes\": [], \"when_good\": \"\"",
            "Valid JSON example: {\"add_names\": [], \"remove_names\": [\"bad_model\"], \"params_overrides\": {}, \"rationale\": \"text\", \"changes\": [], \"when_good\": \"text\"}",
            "Rules:",
            "- You MAY add candidates, but ONLY by name from the tool-provided universe (no invented names).",
            "- Do not change params.method.",
            "- If you cite numbers, they must come from the tool output.",
            "- ALWAYS return valid JSON. If uncertain, return: {\"add_names\": [], \"remove_names\": [], \"params_overrides\": {}, \"rationale\": \"No changes\", \"changes\": [], \"when_good\": \"Stable\"}",
        ],
        markdown=False,
        debug_mode=debug,
    )


def create_statistician_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> Agent:
    return Agent(
        tool_choice="required",
        model=_ollama(model_id, temperature=0.25),
        tools=[build_debate_packet_tool],
        description=STATISTICIAN_DESCRIPTION,
        instructions=[
            "CRITICAL: Output MUST be VALID JSON (no markdown, no extra text before/after JSON).",
            "FIRST call build_debate_packet_tool() to see real evaluation numbers (inputs are provided via context).",
            "Then output ONLY a JSON object with EXACTLY these keys:",
            "  \"add_names\": [], \"remove_names\": [], \"params_overrides\": {}, \"rationale\": \"\", \"changes\": [], \"when_good\": \"\"",
            "Valid JSON example: {\"add_names\": [], \"remove_names\": [], \"params_overrides\": {\"topk_k5\": {\"top_k\": 7}}, \"rationale\": \"text\", \"changes\": [], \"when_good\": \"text\"}",
            "Rules:",
            "- You MAY add candidates, but ONLY by name from the tool-provided universe (no invented names).",
            "- Do not change params.method.",
            "- Prefer stability when n_windows is small; follow recommended_knobs when available.",
            "- If you cite numbers, they must come from the tool output.",
            "- ALWAYS return valid JSON. If uncertain, return: {\"add_names\": [], \"remove_names\": [], \"params_overrides\": {}, \"rationale\": \"Recommended knobs applied\", \"changes\": [], \"when_good\": \"Robust\"}",
        ],
        markdown=False,
        debug_mode=debug,
    )


def create_orchestrator_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> Agent:
    return Agent(
        tool_choice="required",
        model=_ollama(model_id, temperature=0.15),
        tools=[evaluate_strategies_tool],
        description=ORCHESTRATOR_DESCRIPTION,
        instructions=[
            "You MUST call evaluate_strategies_tool with the candidates JSON you receive.",
            "After the tool returns, summarize the top-3 and pick the best one.",
            "Return a final JSON with keys: best_name, reasoning, top3, config_used, when_good, debate_notes.",
            "- reasoning: why the best was selected, referencing ONLY tool results",
            "- when_good: when this selected strategy tends to be appropriate",
            "- debate_notes: if debate happened (margin small), explain why debate was warranted",
            "Do not invent metrics; if you cite a number, it must come from the tool output.",
        ],
        markdown=False,
        debug_mode=debug,
    )
