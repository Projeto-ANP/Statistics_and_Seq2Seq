LangChain-based agents for the orchestrator pipeline.

Usage:
- Import run_llm_pipeline from orchestrator_langchain.pipeline
- The rest of the pipeline API is unchanged.

Notes:
- Tools are exposed with concise names: proposer_brief, debate_packet, evaluate_strategies.
- Prompts live in orchestrator_langchain/prompts/*.md
- Temperature is kept low (0.2) to improve tool-calling reliability.
