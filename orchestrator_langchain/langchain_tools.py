from __future__ import annotations

from langchain_core.tools import tool

from orchestrator import tools as orch_tools


@tool("proposer_brief")
def proposer_brief() -> str:
    """Deterministic brief for proposer (validation summary + candidate library + pattern insights)."""

    return orch_tools.proposer_brief_tool.entrypoint()


@tool("debate_packet")
def debate_packet() -> str:
    """Deterministic debate packet with evaluation numbers."""

    return orch_tools.build_debate_packet_tool.entrypoint()


@tool("evaluate_strategies")
def evaluate_strategies(candidates_json: str, config_json: str = "") -> str:
    """Deterministic evaluation of candidate strategies."""

    return orch_tools.evaluate_strategies_tool.entrypoint(
        candidates_json=candidates_json,
        config_json=config_json,
    )


@tool("build_fold_cot_context")
def build_fold_cot_context() -> str:
    """Build chain-of-thought context from validation folds (trend/seasonality decomposition per model)."""

    return orch_tools.build_fold_cot_context_tool.entrypoint()
