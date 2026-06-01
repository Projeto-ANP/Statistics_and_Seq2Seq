"""V5 episodic + semantic + procedural memory for the Selector agent.

Inspired by Mem0 (arXiv:2504.19413), Contextual Experience Replay (arXiv:2506.06698) and
MemGPT/Letta. The memory grows as series are processed; the Selector queries past episodes
to inform its choice on each new series. See ARCHITECTURE_V5_PROPOSAL.md §3.3 / §3.4.
"""
from orchestrator.memory.episodic import EpisodicMemory  # noqa: F401
