from __future__ import annotations

# Replace Agno infrastructure directly with Langchain's adapted infrastructure.
from orchestrator_langchain.agents import (
    AgentResponse,
    DEFAULT_MODEL_ID,
    LangchainAgent as Agent,
    create_pattern_analyst_agent,
    create_proposer_agent,
    create_skeptic_agent,
    create_statistician_agent,
    create_orchestrator_agent
)
