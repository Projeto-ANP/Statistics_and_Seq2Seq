from __future__ import annotations

from langchain_core.tools import tool

from orchestrator import tools as orch_tools


@tool("proposer_brief")
def proposer_brief() -> str:
    """Deterministic brief for proposer (validation summary + candidate library + pattern insights)."""

    return orch_tools.proposer_brief_tool()


@tool("debate_packet")
def debate_packet() -> str:
    """Deterministic debate packet with evaluation numbers."""

    return orch_tools.build_debate_packet_tool()


@tool("evaluate_strategies")
def evaluate_strategies(candidates_json: str, config_json: str = "") -> str:
    """Deterministic evaluation of candidate strategies."""

    return orch_tools.evaluate_strategies_tool(
        candidates_json=candidates_json,
        config_json=config_json,
    )


@tool("build_fold_cot_context")
def build_fold_cot_context() -> str:
    """Build chain-of-thought context from validation folds (trend/seasonality decomposition per model)."""

    return orch_tools.build_fold_cot_context_tool()


# ── V2 pipeline tools ────────────────────────────────────────────────────────

@tool("strategy_brief")
def strategy_brief() -> str:
    """V2: Strategy brief for StrategySelector — includes SeriesProfile + candidate library + strategy guide."""

    return orch_tools.strategy_brief_tool()


# ── V3 pipeline tools ────────────────────────────────────────────────────────

@tool("series_analysis_brief")
def series_analysis_brief() -> str:
    """V3: Series features (forecastability, trend/seasonal strength, stationarity) + validation summary for the SeriesAnalyst."""

    return orch_tools.series_analysis_brief_tool()


@tool("model_critic_brief")
def model_critic_brief() -> str:
    """V3: Per-model diagnostics + Model Confidence Set + redundancy pairs for the ModelCritic to decide which models to prune."""

    return orch_tools.model_critic_brief_tool()


@tool("combination_architect_brief")
def combination_architect_brief() -> str:
    """V3: Robust combination regimes over the surviving pool + recommended shrinkage for the CombinationArchitect."""

    return orch_tools.combination_architect_brief_tool()
