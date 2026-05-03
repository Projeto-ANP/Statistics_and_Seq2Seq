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
