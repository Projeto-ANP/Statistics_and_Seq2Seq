"""
Forecast Pipeline with Multi-Agent Architecture
================================================
Two specialized agents working in sequence:
1.Analysis Agent - Analyzes model performance
2.Combination Agent - Executes the recommended combination strategy
"""

from typing import Dict, List, Optional, Any
from agno.agent import Agent
from agno.models.ollama import Ollama
from context import CONTEXT_MEMORY, generate_all_validations_context, init_context, get_context, set_context
from analysis_tools import (
    calculate_metrics_tool,
    generate_ade_point_models_tool,
    generate_ade_weighted_point_models_tool,
)
from combination_tools import (
    mean_combination_tool,
    weight_combination_tool,
    point_combination_tool,
    ade_point_selection_tool,
    ade_weighted_point_tool,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "qwen3:14b"  # ou "mychen76/qwen3_cline_roocode"

ANALYSIS_AGENT_DESCRIPTION = """You are a forecasting ANALYST.Your ONLY job is to analyze model performance.

AVAILABLE TOOLS:
1.calculate_metrics_tool() - Calculates MAPE, RMSE, SMAPE, POCID for each model
2.generate_ade_point_models_tool(top_k) - Finds best models for each forecast horizon
3.generate_ade_weighted_point_models_tool(top_k) - Calculates weights per model per horizon

WORKFLOW (follow exactly):
1.FIRST: Call calculate_metrics_tool to get overall metrics
2.THEN: Call the appropriate generation tool based on your strategy decision:
   - For "ade_point_selection": Call generate_ade_point_models_tool(top_k=3)
   - For "ade_weighted_point": Call generate_ade_weighted_point_models_tool(top_k=3)
3.IMPORTANT: The result from step 2 is your strategy_parameters - use it directly!

STRATEGIES:
- "mean_combination": All models have similar performance → parameters: {"model_names": ["model1", "model2", ...]}
- "weight_combination": Some models are clearly better → parameters: {"model_weights": {"model1": 0.5, "model2": 0.3, ...}}
- "point_combination": Each model specializes in ONE horizon → parameters: {"model_points": {"model1": 0, "model2": 1, ...}}
- "ade_point_selection": Different models excel at different horizons → parameters: output from generate_ade_point_models_tool
- "ade_weighted_point": Fine-grained weight control per horizon → parameters: output from generate_ade_weighted_point_models_tool

OUTPUT FORMAT (JSON):
{
    "metrics_summary": "Brief summary of model performance",
    "point_analysis": "Which models are best at which points",
    "recommended_strategy": "strategy_name",
    "strategy_parameters": <ACTUAL_PARAMETERS_DICT>,
    "reasoning": "Why this strategy is best for this data"
}

IMPORTANT: strategy_parameters must be the COMPLETE parameters dict ready to pass to the combination tool!
"""

COMBINATION_AGENT_DESCRIPTION = """You are a forecasting COMBINER. Your job is to execute combination tools.

AVAILABLE TOOLS:
1. mean_combination_tool(model_names: List[str]) - Average specified models
2. weight_combination_tool(model_weights: Dict[str, float]) - Weighted average
3. point_combination_tool(model_points: Dict[str, int]) - Each model predicts one point
4. ade_point_selection_tool(point_models: Dict[int, List[str]]) - Multiple models per point (average)
5. ade_weighted_point_tool(point_model_weights: Dict[int, Dict[str, float]]) - Weighted models per point

YOUR TASK:
You will receive analysis results with a recommended strategy and parameters.
You MUST call the appropriate tool with the provided parameters.

IMPORTANT:
- You must CALL A TOOL, not just describe what to do
- Extract parameters from strategy_parameters in the analysis
- Pass parameters directly to the tool function
- Return the tool's result
"""


# =============================================================================
# TOOL DEFINITIONS (import your actual tools here)
# =============================================================================

# Analysis Tools
ANALYSIS_TOOLS = [
    calculate_metrics_tool,
    generate_ade_point_models_tool,
    generate_ade_weighted_point_models_tool,
]

# Combination Tools
COMBINATION_TOOLS = [
    mean_combination_tool,
    weight_combination_tool,
    point_combination_tool,
    ade_point_selection_tool,
    ade_weighted_point_tool,
]


# =============================================================================
# AGENT FACTORIES
# =============================================================================

def create_analysis_agent(
    model_id: str = MODEL_ID,
    analysis_tools: List = None,
    debug: bool = False
) -> Agent:
    """
    Creates the Analysis Agent responsible for evaluating model performance.
    
    Args: 
        model_id: Ollama model identifier
        analysis_tools: List of analysis tool functions
        debug:  Enable debug mode
    
    Returns:
        Configured Analysis Agent
    """
    tools = analysis_tools if analysis_tools is not None else ANALYSIS_TOOLS
    
    agent = Agent(
        model=Ollama(id=model_id, options={"temperature": 0.0}),
        tools=tools,
        description=ANALYSIS_AGENT_DESCRIPTION,
        instructions=[
            "Always call calculate_metrics_tool first",
            "Then call generate_ade_point_models_tool to analyze per-point performance",
            "IMPORTANT: Use the dict returned by generate_ade_point_models_tool as your strategy_parameters",
            "The strategy_parameters must be wrapped: {'point_models': <result_from_tool>}",
            "Output your final recommendation in valid JSON format",
            "Be specific about which models to use and with what parameters",
        ],
        markdown=True,
        debug_mode=debug,
    )
    
    return agent


def create_combination_agent(
    model_id: str = MODEL_ID,
    combination_tools: List = None,
    debug: bool = False
) -> Agent:
    """
    Creates the Combination Agent responsible for executing combination strategies.
    
    Args:
        model_id: Ollama model identifier
        combination_tools: List of combination tool functions
        debug: Enable debug mode
    
    Returns:
        Configured Combination Agent
    """
    tools = combination_tools if combination_tools is not None else COMBINATION_TOOLS
    
    agent = Agent(
        model=Ollama(id=model_id, options={"temperature": 0.0}),
        tools=tools,
        description=COMBINATION_AGENT_DESCRIPTION,
        instructions=[
            "Read the analysis result JSON carefully",
            "Find the 'recommended_strategy' field",
            "Find the 'strategy_parameters' field", 
            "Call the matching tool with the parameters from strategy_parameters",
            "YOU MUST CALL A TOOL - do not just explain, actually execute the function",
        ],
        markdown=True,
        debug_mode=debug,
        )
    
    return agent


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class ForecastPipeline: 
    """
    Orchestrates the multi-agent forecasting pipeline.
    
    Attributes:
        analysis_agent: Agent for analyzing model performance
        combination_agent: Agent for combining predictions
        debug: Enable verbose logging
    """
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        analysis_tools: List = None,
        combination_tools: List = None,
        debug: bool = False
    ):
        """
        Initialize the forecast pipeline.
        
        Args:
            model_id:  Ollama model identifier
            analysis_tools:  Custom analysis tools (optional)
            combination_tools: Custom combination tools (optional)
            debug: Enable debug mode
        """
        self.debug = debug
        self.model_id = model_id
        
        self.analysis_agent = create_analysis_agent(
            model_id=model_id,
            analysis_tools=analysis_tools,
            debug=debug
        )
        self.combination_agent = create_combination_agent(
            model_id=model_id,
            combination_tools=combination_tools,
            debug=debug
        )
        
        # Store results
        self.last_analysis = None
        self.last_combination = None
        self.last_result = None
    
    def _log(self, message: str) -> None:
        """Print log message if debug mode is enabled."""
        if self.debug:
            print(f"[PIPELINE] {message}")
    
    def _print_phase_header(self, phase_name: str) -> None:
        """Print formatted phase header."""
        print("\n" + "=" * 60)
        print(f"  {phase_name}")
        print("=" * 60 + "\n")
    
    def run_analysis(self, user_query: str = "") -> str:
        """
        Execute Phase 1: Analysis.
        
        Args:
            user_query: Optional user context or request
        
        Returns: 
            Analysis result as string
        """
        self._print_phase_header("PHASE 1: ANALYSIS")
        self._log("Starting analysis agent...")
        
        analysis_prompt = f"""
Analyze the forecasting models and recommend the best combination strategy.

{f"User context: {user_query}" if user_query else ""}

Steps to follow:
1.Call calculate_metrics_tool() to get performance metrics
2.Based on metrics, decide which strategy to use
3.If you choose "ade_point_selection": Call generate_ade_point_models_tool(top_k=3) and use its output as strategy_parameters
4.If you choose "ade_weighted_point": Call generate_ade_weighted_point_models_tool(top_k=3) and use its output as strategy_parameters
5.For other strategies, construct the appropriate parameters dict

Provide your recommendation in JSON format with:
- metrics_summary
- point_analysis
- recommended_strategy
- strategy_parameters (MUST be the complete parameters dict, not just {{"top_k": 3}})
- reasoning

EXAMPLE for ade_point_selection:
{{
    "recommended_strategy": "ade_point_selection",
    "strategy_parameters": {{"point_models": {{0: ["model1", "model2"], 1: ["model3"], ...}}}}
}}
"""
        
        result = self.analysis_agent.run(analysis_prompt)
        self.last_analysis = result
        print("(-----------------------------ANALISES--------------------------------)\n\n\n")
        print(result.content)
        print("(-----------------------------ANALISES--------------------------------)\n\n\n")
        self._log("Analysis complete")
        return result.content
    
    def run_combination(self, analysis_result: str) -> str:
        """
        Execute Phase 2: Combination.
        
        Args:
            analysis_result: Output from the analysis phase
        
        Returns:
            Combination result with final forecast
        """
        self._print_phase_header("PHASE 2: COMBINATION")
        self._log("Starting combination agent...")
        
        combination_prompt = f"""
Based on the analysis below, execute the recommended combination strategy by calling the appropriate tool.

ANALYSIS RESULT:
{analysis_result}

INSTRUCTIONS:
1. Parse the JSON above to extract "recommended_strategy" and "strategy_parameters"
2. Identify which tool to call based on the strategy
3. Extract the parameters and call the tool

TOOL MAPPING:
- If recommended_strategy = "ade_point_selection":
  Extract point_models = strategy_parameters["point_models"]
  Call: ade_point_selection_tool(point_models=point_models)

- If recommended_strategy = "ade_weighted_point":
  Extract point_model_weights = strategy_parameters["point_model_weights"]
  Call: ade_weighted_point_tool(point_model_weights=point_model_weights)

- If recommended_strategy = "mean_combination":
  Extract model_names = strategy_parameters["model_names"]
  Call: mean_combination_tool(model_names=model_names)

- If recommended_strategy = "weight_combination":
  Extract model_weights = strategy_parameters["model_weights"]
  Call: weight_combination_tool(model_weights=model_weights)

- If recommended_strategy = "point_combination":
  Extract model_points = strategy_parameters["model_points"]
  Call: point_combination_tool(model_points=model_points)

YOU MUST CALL ONE OF THE TOOLS ABOVE. Do not just return text - execute the tool function.
"""
        
        result = self.combination_agent.run(combination_prompt)
        self.last_combination = result
        
        print("(-----------------------------COMBINATION--------------------------------)\n\n\n")
        print(result.content)
        print("(-----------------------------COMBINATION--------------------------------)\n\n\n")
        
        self._log("Combination complete")
        return result.content
    
    def run(self, user_query: str = "") -> Dict[str, Any]: 
        """
        Execute the complete forecasting pipeline.
        
        Args:
            user_query: Optional user context or request
        
        Returns:
            Dictionary containing analysis and combination results
        """
        self._print_phase_header("FORECAST PIPELINE STARTED")
        
        # Phase 1: Analysis (with retry if tools not called)
        max_analysis_retries = 3
        analysis_result = None
        
        for attempt in range(max_analysis_retries):
            analysis_result = self.run_analysis(user_query)
            
            # Validate that analysis tools were called
            tools_called = get_context("tools_called", [])
            analysis_tools_ok = "calculate_metrics_tool" in tools_called and (
                "generate_ade_point_models_tool" in tools_called or 
                "generate_ade_weighted_point_models_tool" in tools_called
            )
            
            if analysis_tools_ok:
                break
            
            print(f"\n[WARNING] Analysis agent did not call required tools. Retrying... Attempt {attempt + 1}/{max_analysis_retries}")
            print(f"[WARNING] Tools called so far: {tools_called}")
        else:
            # Analysis failed after all retries
            print("\n[ERROR] Analysis phase failed after all retries. Tools were not called correctly.")
            return {
                "description": "Pipeline failed - analysis agent did not execute tools correctly",
                "result": None,
                "success": False,
                "error": "Analysis tool execution validation failed"
            }
        
        # Phase 2: Combination (with retry if tools not called)
        max_combination_retries = 3
        combination_result = None
        
        for attempt in range(max_combination_retries):
            combination_result = self.run_combination(analysis_result)
            
            # Validate that combination tool was called
            tools_called = get_context("tools_called", [])
            combination_tools = ["mean_combination_tool", "weight_combination_tool", "point_combination_tool", 
                                "ade_point_selection_tool", "ade_weighted_point_tool"]
            combination_tools_ok = any(tool in tools_called for tool in combination_tools)
            
            if combination_tools_ok:
                break
            
            print(f"\n[WARNING] Combination agent did not call any combination tool. Retrying... Attempt {attempt + 1}/{max_combination_retries}")
            print(f"[WARNING] Tools called so far: {tools_called}")
        else:
            # Combination failed after all retries
            print("\n[ERROR] Combination phase failed after all retries. Tools were not called correctly.")
            return {
                "description": "Pipeline failed - combination agent did not execute tools correctly",
                "result": None,
                "success": False,
                "error": "Combination tool execution validation failed"
            }
        
        import json
        try:
            analysis_json = json.loads(analysis_result)
            strategy = analysis_json.get("recommended_strategy", "unknown")
            strategy_params = analysis_json.get("strategy_parameters", {})
            reasoning = analysis_json.get("reasoning", "")
            
            description = analysis_result
            
        except json.JSONDecodeError:
            description = f"Strategy execution completed. Analysis result was not valid JSON."
        
        # Parse combination result to extract the forecast
        try:
            # Try to find a list in the combination result
            if isinstance(combination_result, str):
                # Look for JSON list or Python list representation
                import re
                list_match = re.search(r'\[[\d\.,\s]+\]', combination_result)
                if list_match:
                    result_list = json.loads(list_match.group())
                else:
                    result_list = None
            else:
                result_list = combination_result
        except:
            result_list = None
        
        # Summary
        self._print_phase_header("PIPELINE COMPLETE")
        
        final_result = {
            "description": description,
            "result": result_list,
            "success": True
        }
        
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(json.dumps(final_result, indent=2))
        print("=" * 60 + "\n")
        
        self.last_result = final_result
        
        return final_result
    
    def run_with_strategy(
        self,
        strategy:  str,
        strategy_params: Dict = None
    ) -> Dict[str, Any]: 
        """
        Execute pipeline with a specific strategy (skip analysis recommendation).
        
        Args:
            strategy: Strategy name (e.g., "ade_weighted_point")
            strategy_params: Parameters for the strategy
        
        Returns: 
            Dictionary containing results
        """
        self._print_phase_header("FORECAST PIPELINE (DIRECT STRATEGY)")
        
        # Phase 1: Still run analysis for metrics
        analysis_result = self.run_analysis()
        
        # Phase 2: Force specific strategy
        self._print_phase_header("PHASE 2: COMBINATION (FORCED)")
        
        params_str = str(strategy_params) if strategy_params else "use parameters from analysis"
        
        override_prompt = f"""
Execute the "{strategy}" strategy with these parameters:
{params_str}

Analysis context: 
{analysis_result}

Call the appropriate tool and return the combined forecast.
"""
        
        combination_result = self.combination_agent.run(override_prompt)
        self.last_combination = combination_result
        
        self._print_phase_header("PIPELINE COMPLETE")
        
        self.last_result = {
            "analysis": analysis_result,
            "combination":  combination_result,
            "forced_strategy": strategy,
            "success":  True
        }
        
        return self.last_result
    
    def run_analysis_only(self, user_query: str = "") -> str:
        """
        Run only the analysis phase without combination.
        Useful for understanding model performance before deciding strategy.
        
        Args:
            user_query: Optional user context
        
        Returns: 
            Analysis result
        """
        return self.run_analysis(user_query)
    
    def run_combination_only(self, strategy: str, params: Dict) -> str:
        """
        Run only the combination phase with explicit parameters.
        Useful when you already know what strategy and parameters to use.
        
        Args: 
            strategy: Strategy name
            params:  Strategy parameters
        
        Returns:
            Combination result
        """
        self._print_phase_header("DIRECT COMBINATION")
        
        prompt = f"""
Execute the "{strategy}" strategy with these exact parameters:
{params}

Call the appropriate combination tool and return the forecast.
"""
        
        result = self.combination_agent.run(prompt)
        self.last_combination = result
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(
    model_id: str = MODEL_ID,
    debug: bool = False
) -> ForecastPipeline: 
    """
    Factory function to create a forecast pipeline with default configuration.
    
    Args:
        model_id: Ollama model identifier
        debug: Enable debug mode
    
    Returns:
        Configured ForecastPipeline instance
    """
    return ForecastPipeline(model_id=model_id, debug=debug)


def quick_forecast(debug: bool = False) -> Dict[str, Any]:
    """
    One-liner to run the complete forecast pipeline.
    
    Args: 
        debug: Enable debug mode
    
    Returns:
        Pipeline results
    """
    pipeline = create_pipeline(debug=debug)
    return pipeline.run()


def analyze_models(debug: bool = False) -> str:
    """
    Quick analysis without combination.
    
    Args:
        debug: Enable debug mode
    
    Returns: 
        Analysis result
    """
    pipeline = create_pipeline(debug=debug)
    return pipeline.run_analysis_only()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    
    models_available = [
        "ARIMA",
        "ETS",
        "THETA",
        "svr",
        "rf",
        "catboost",
        "CWT_svr",
        "DWT_svr",
        "FT_svr",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
        "ONLY_CWT_catboost",
        "ONLY_CWT_rf",
        "ONLY_CWT_svr",
        "ONLY_DWT_catboost",
        "ONLY_DWT_rf",
        "ONLY_DWT_svr",
        "ONLY_FT_catboost",
        "ONLY_FT_rf",
        "ONLY_FT_svr",
        "NaiveSeasonal",
        "NaiveMovingAverage",
    ]
    
    dataset_index = 0
    print("\n>>> Example 1: Full Pipeline")
    init_context()
    CONTEXT_MEMORY["models_available"] = models_available
    generate_all_validations_context(models_available, dataset_index)
    if "all_validations" in CONTEXT_MEMORY:
        print(f"[DEBUG] all_validations keys: {list(CONTEXT_MEMORY['all_validations'].keys())}")
    
    pipeline = create_pipeline(debug=False, model_id="mychen76/qwen3_cline_roocode:14b")
    result = pipeline.run()
    print("***************************")
    # print(result)
    
    # Example 2: Analysis only
    # print("\n>>> Example 2: Analysis Only")
    # analysis = analyze_models(debug=True)
    # print(analysis)
    
    # Example 3: Force specific strategy
    # print("\n>>> Example 3: Force Strategy")
    # pipeline = create_pipeline(debug=True)
    # result = pipeline.run_with_strategy(
    #     strategy="ade_weighted_point",
    #     strategy_params={
    #         0: {"rf": 0.6, "catboost": 0.4},
    #         1: {"catboost": 0.5, "ARIMA": 0.5},
    #     }
    # )
    
    # Example 4: Direct combination (skip analysis)
    # print("\n>>> Example 4: Direct Combination")
    # pipeline = create_pipeline(debug=True)
    # result = pipeline.run_combination_only(
    #     strategy="mean_combination",
    #     params={"model_names":  ["rf", "catboost", "ARIMA"]}
    # )