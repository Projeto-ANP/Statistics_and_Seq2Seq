from __future__ import annotations

from typing import Any, Dict

import orchestrator.pipeline as _base
from orchestrator_langchain import agents as lc_agents
import orchestrator.utils as _utils

def run_langchain_pipeline(
    proposer_model: _utils.ModelConfig,
    skeptic_model: _utils.ModelConfig,
    statistician_model: _utils.ModelConfig,
    pattern_analyst_model: _utils.ModelConfig,
    debug: bool = False,
    rolling_mode: str = "expanding",
    train_window: int = 5,
    require_tool_call: bool = True,
    llm_logs: bool = True,
    debate: bool = False,
    debate_auto: bool = True,
    debate_margin: float = 0.02,
) -> Dict[str, Any]:
    # Monkey-patch the agent factories used inside orchestrator.pipeline
    _base.create_pattern_analyst_agent = lc_agents.create_pattern_analyst_agent
    _base.create_proposer_agent = lc_agents.create_proposer_agent
    _base.create_skeptic_agent = lc_agents.create_skeptic_agent
    _base.create_statistician_agent = lc_agents.create_statistician_agent

    return _base.run_llm_pipeline(
        proposer_model,
        statistician_model,
        skeptic_model,
        pattern_analyst_model,
        debug=debug,
        rolling_mode=rolling_mode,
        train_window=train_window,
        require_tool_call=require_tool_call,
        llm_logs=llm_logs,
        debate=debate,
        debate_auto=debate_auto,
        debate_margin=debate_margin,
    )


def run_deterministic_pipeline(*args, **kwargs):
    return _base.run_deterministic_pipeline(*args, **kwargs)


def run_langchain_pipeline_v2(
    series_annotator_model: _utils.ModelConfig,
    strategy_selector_model: _utils.ModelConfig,
    debug: bool = False,
    rolling_mode: str = "expanding",
    train_window: int = 3,
    require_tool_call: bool = True,
    llm_logs: bool = True,
) -> Dict[str, Any]:
    """V2: SeriesAnnotator (temp=0) → StrategySelector (temp=0) → Deterministic eval + Oracle.

    Improvements over v1:
    - temperature=0 on both agents → fully reproducible given same model weights.
    - No debate loop → cleaner contribution isolation.
    - Oracle always runs on full candidate library → fair comparison baseline.
    - Structured SeriesProfile stored in context → auditable reasoning chain.
    - Both LLM-guided and oracle evals use 'balanced' preset → apples-to-apples delta.
    """
    _base.create_series_annotator_agent = lc_agents.create_series_annotator_agent
    _base.create_strategy_selector_agent = lc_agents.create_strategy_selector_agent

    return _base.run_llm_pipeline_v2(
        series_annotator_model=series_annotator_model,
        strategy_selector_model=strategy_selector_model,
        debug=debug,
        rolling_mode=rolling_mode,
        train_window=train_window,
        require_tool_call=require_tool_call,
        llm_logs=llm_logs,
    )


def run_langchain_pipeline_v3(
    series_analyst_model: _utils.ModelConfig,
    model_critic_model: _utils.ModelConfig,
    combination_architect_model: _utils.ModelConfig,
    debug: bool = False,
    rolling_mode: str = "expanding",
    train_window: int = 3,
    require_tool_call: bool = True,
    llm_logs: bool = True,
    gate_alpha: float = 0.20,
    pool_curation_threshold: float = 0.92,
    pool_curation_min_keep: int = 6,
) -> Dict[str, Any]:
    """V3 (post-Sprint-1): SeriesAnalyst → ModelCritic (prune) → CombinationArchitect →
    robust core (DM gate against the better of pruned_mean / pruned_median).

    Sprint-1 changes vs. original V3:
    - Upstream pool curation by error-correlation clustering (Cawood & van Zyl 2024).
    - Dual anchor: evaluate pruned_mean AND pruned_median, gate against the better one.
    - DM gate alpha relaxed 0.10 → 0.20 to recover power on 3×horizon residuals.
    - catch22 features (Lubba et al. 2019) appended to compute_series_features.
    """
    _base.create_series_analyst_agent = lc_agents.create_series_analyst_agent
    _base.create_model_critic_agent = lc_agents.create_model_critic_agent
    _base.create_combination_architect_agent = lc_agents.create_combination_architect_agent

    return _base.run_llm_pipeline_v3(
        series_analyst_model=series_analyst_model,
        model_critic_model=model_critic_model,
        combination_architect_model=combination_architect_model,
        debug=debug,
        rolling_mode=rolling_mode,
        train_window=train_window,
        require_tool_call=require_tool_call,
        llm_logs=llm_logs,
        gate_alpha=gate_alpha,
        pool_curation_threshold=pool_curation_threshold,
        pool_curation_min_keep=pool_curation_min_keep,
    )
