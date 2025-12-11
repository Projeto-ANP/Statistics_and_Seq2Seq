from agno.agent import Agent
from agno.models.ollama import Ollama
from context import CONTEXT_MEMORY, generate_all_validations_context, init_context
from analysis_tools import generate_ade_weighted_point_models_tool

def run_agent(
    models_available: List[str],
    dataset_index: int,
):
    init_context()
    CONTEXT_MEMORY["models_available"] = models_available
    generate_all_validations_context(models_available, dataset_index)
    
    temperature = 0.0
    model_id = "qwen3:14b"
    agent = Agent(
        tool_choice="required",
        model=Ollama(
            id=model_id,
            options={"temperature": temperature, "num_ctx": 65536, "keep_alive": "5m"},
        ),
        tools=[calculate_metrics_tool, selective_combine_tool],
        instructions=instructions,
        markdown=True,
    )

    print("=" * 80)
    print("AGENT EXECUTION")
    print("=" * 80)
    print(f"Model: {agent.model.id}")
    print(f"Models Available: {list(validation_predictions.keys())}")
    print("=" * 80 + "\n")
    
    print("Sending prompt to agent...")
    print("-" * 80)
