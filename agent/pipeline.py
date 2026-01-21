"""
Forecast Pipeline with Multi-Agent Architecture
================================================
Two specialized agents working in sequence:
1.Analysis Agent - Analyzes model performance
2.Combination Agent - Executes the recommended combination strategy
"""

from typing import Dict, List, Optional, Any
import os
import re
import json
import ast
from agno.agent import Agent
from agno.models.ollama import Ollama
from agent.context import CONTEXT_MEMORY, generate_all_validations_context, init_context, get_context, set_context
from agent.analysis_tools import (
    calculate_metrics_tool,
    generate_ade_point_models_tool,
    generate_ade_weighted_point_models_tool,
)
from agent.combination_tools import (
    mean_combination_tool,
    weight_combination_tool,
    point_combination_tool,
    ade_point_selection_tool,
    ade_weighted_point_tool,
)
from agent.tool_logging import log_tool_event


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

COMBINATION_AGENT_DESCRIPTION = """You are a tool executor. Your ONLY job is to call combination tools.

When you receive a message, you must IMMEDIATELY call the appropriate tool with the provided parameters.

DO NOT explain. DO NOT describe. CALL THE TOOL."""


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
            "Choose exactly ONE recommended_strategy from: mean_combination, weight_combination, point_combination, ade_point_selection, ade_weighted_point",
            "If you choose ade_point_selection OR point_combination: call generate_ade_point_models_tool(top_k=3) (use top_k=1 only if explicitly needed)",
            "If you choose ade_weighted_point: call generate_ade_weighted_point_models_tool(top_k=3)",
            "For mean_combination/weight_combination you may skip per-point tools if not needed",
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
            "Call the tool immediately with the provided parameters",
            "Do not explain or describe - execute the function",
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

    def _dataset_id(self) -> str:
        dataset_index = get_context("dataset_index", None)
        if dataset_index is None:
            return "dataset_unknown"
        return f"dataset_{dataset_index}"

    def _log_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "tool_logs")

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        try:
            log_tool_event(
                base_dir=self._log_dir(),
                dataset_id=self._dataset_id(),
                event=event,
                payload=payload,
            )
        except Exception as e:
            # Never fail the pipeline due to logging
            if self.debug:
                print(f"[PIPELINE] Logging failed: {e}")

    def _execute_combination_strategy(self, strategy: str, params: Dict[str, Any]) -> Any:
        """Deterministically execute the selected combination via Python tool call."""
        def _call_tool(tool_obj: Any, *args: Any, **kwargs: Any) -> Any:
            # agno.tools.function.Function is not callable; use `.entrypoint()`.
            entrypoint = getattr(tool_obj, "entrypoint", None)
            if callable(entrypoint):
                return entrypoint(*args, **kwargs)
            if callable(tool_obj):
                return tool_obj(*args, **kwargs)
            raise TypeError(f"Tool object is not callable and has no callable entrypoint: {tool_obj}")

        tool_map = {
            "ade_point_selection": ade_point_selection_tool,
            "ade_weighted_point": ade_weighted_point_tool,
            "mean_combination": mean_combination_tool,
            "weight_combination": weight_combination_tool,
            "point_combination": point_combination_tool,
        }
        if strategy not in tool_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Prepare context for combination tools (they read from context[\"point_parameter\"]).
        if strategy == "ade_point_selection":
            set_context("point_parameter", params.get("point_models", {}))
        elif strategy == "ade_weighted_point":
            set_context("point_parameter", params.get("point_model_weights", {}))
        elif strategy == "mean_combination":
            set_context("point_parameter", params.get("model_names", []))
        elif strategy == "weight_combination":
            set_context("point_parameter", params.get("model_weights", {}))
        elif strategy == "point_combination":
            set_context("point_parameter", params.get("model_points", {}))

        before_tools = list(get_context("tools_called", []))
        self._log_event(
            "combination_start",
            {
                "strategy": strategy,
                "strategy_parameters": params,
                "tools_called_before": before_tools,
            },
        )

        tool_obj = tool_map[strategy]
        result = _call_tool(tool_obj)

        after_tools = list(get_context("tools_called", []))
        self._log_event(
            "combination_end",
            {
                "strategy": strategy,
                "tools_called_after": after_tools,
                "tool_called": getattr(tool_obj, "name", None) or getattr(tool_obj, "__name__", None) or str(tool_obj),
            },
        )
        return result

    def _ensure_point_models(self, top_k: int = 1) -> Dict[int, List[str]]:
        """Deterministically compute point_models from validation context."""
        tool_obj = generate_ade_point_models_tool
        entrypoint = getattr(tool_obj, "entrypoint", None)
        if callable(entrypoint):
            return entrypoint(top_k=top_k)
        # Fallback if tool isn't wrapped
        return tool_obj(top_k=top_k)  # type: ignore[misc]

    def _ensure_point_model_weights(self, top_k: int = 3) -> Dict[int, Dict[str, float]]:
        """Deterministically compute point_model_weights from validation context."""
        tool_obj = generate_ade_weighted_point_models_tool
        entrypoint = getattr(tool_obj, "entrypoint", None)
        if callable(entrypoint):
            return entrypoint(top_k=top_k)
        return tool_obj(top_k=top_k)  # type: ignore[misc]

    def _parse_analysis_result(self, analysis_result: str) -> Dict[str, Any]:
        """Parse analysis agent output.

        The LLM sometimes returns almost-JSON (e.g., numeric keys without quotes), which breaks `json.loads`.
        We first try strict JSON, then fall back to Python-literal parsing.
        """
        if not isinstance(analysis_result, str):
            raise TypeError("analysis_result must be a string")

        text = analysis_result.strip()

        # Strip Markdown code fences if present
        if "```" in text:
            fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
            if fenced:
                text = fenced[0].strip()

        # Extract the outer-most object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON-like object found in analysis_result")

        blob = text[start : end + 1]

        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        parsed = ast.literal_eval(blob)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed analysis_result is not a dict")
        return parsed
    
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
1. Call calculate_metrics_tool() to get performance metrics.

2. Decide the strategy using this logic:

    A) Prefer "mean_combination" when models are similar overall:
        - There is no clear winner across windows (global RMSE/MAPE are close)
        - The best model is not consistently better than the rest

    B) Prefer "weight_combination" when there are clear overall winners:
        - A small set of models are consistently better across windows
        - Build global weights from overall performance (better models get higher weights)

    C) Prefer "point_combination" when ONE model is best for EACH horizon point (top-1 per point),
        and the best model changes across points:
        - Each horizon point has a clear single best model
        - Different points choose different best models
        - Parameters must map model_name -> horizon_index

    D) Prefer "ade_point_selection" when different models excel at different horizons and you want robustness:
        - Multiple good models per point
        - Use top_k models per point

    E) Prefer "ade_weighted_point" when per-horizon weighting is beneficial:
        - Different models excel at different horizons AND
        - There are meaningful gaps in per-point RMSE among top models
        - Use weights per point (normalized)

3. If you choose "ade_point_selection":
    - Call generate_ade_point_models_tool(top_k=3)
    - Use the returned dict as the core selection
    - strategy_parameters MUST include the key "point_models" containing the returned dict

4. If you choose "ade_weighted_point":
    - Call generate_ade_weighted_point_models_tool(top_k=3)
    - strategy_parameters MUST include the key "point_model_weights" containing the returned dict

5. If you choose "mean_combination":
    - Choose a reasonable subset of models (e.g., top N overall) OR all models if truly similar
    - strategy_parameters MUST include the key "model_names" as a list of model names

6. If you choose "weight_combination":
    - Build global weights from overall metrics (weights sum to 1)
    - strategy_parameters MUST include the key "model_weights" as a dict model->weight (sum to 1)

7. If you choose "point_combination":
    - You must infer a single best model per point (top-1 per horizon)
    - strategy_parameters MUST include the key "model_points" as a dict model->horizon_index

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
EXAMPLE for ade_weighted_point:
{{
    "recommended_strategy": "ade_weighted_point",
    "strategy_parameters": {{"point_model_weights": {{0: {{"model1": 0.6, "model2": 0.4}}, 1: {{"model3": 1.0}}, ...}}}}
}}
EXAMPLE for mean_combination:
{{
    "recommended_strategy": "mean_combination",
    "strategy_parameters": {{"model_names": ["model1", "model2", "model3"]}}
}}
EXAMPLE for weight_combination:
{{
    "recommended_strategy": "weight_combination",
    "strategy_parameters": {{"model_weights": {{"model1": 0.6, "model2": 0.3, "model3": 0.1}}}}
}}
EXAMPLE for point_combination:
{{
    "recommended_strategy": "point_combination",
    "strategy_parameters": {{"model_points": {{"modelA": 0, "modelB": 1, "modelC": 2}}}}
}}
"""
        
        result = self.analysis_agent.run(analysis_prompt)
        self.last_analysis = result
        print("(-----------------------------ANALISES--------------------------------)\n\n\n")
        print(result.content)
        print("(-----------------------------ANALISES--------------------------------)\n\n\n")
        self._log("Analysis complete")

        self._log_event(
            "analysis_end",
            {
                "tools_called": list(get_context("tools_called", [])),
                "analysis_output_is_str": isinstance(result.content, str),
            },
        )
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
        self._log("Executing combination deterministically via tools...")

        try:
            analysis_json = self._parse_analysis_result(analysis_result)
            strategy = analysis_json.get("recommended_strategy", "")
            params = analysis_json.get("strategy_parameters", {})
        except Exception as e:
            print(f"\n[ERROR] Failed to parse analysis result. Defaulting to ade_point_selection strategy. Error: {e}")
            self._log_event(
                "analysis_parse_error",
                {"error": str(e), "analysis_preview": analysis_result[:500]},
            )
            strategy = "ade_point_selection"
            params = {}

        # Coerce/auto-fill parameters deterministically to avoid LLM mistakes.
        # NOTE: `point_combination_tool` does not produce a full-horizon forecast; treat it as top-1-per-point selection.
        if strategy == "point_combination":
            self._log_event(
                "strategy_coerced",
                {"from": "point_combination", "to": "ade_point_selection", "reason": "ensure horizon-length forecast"},
            )
            strategy = "ade_point_selection"
            params = {"point_models": self._ensure_point_models(top_k=1)}

        if strategy == "ade_point_selection":
            point_models = params.get("point_models")
            if not point_models:
                params = {"point_models": self._ensure_point_models(top_k=3)}

        if strategy == "ade_weighted_point":
            point_model_weights = params.get("point_model_weights")
            if not point_model_weights:
                params = {"point_model_weights": self._ensure_point_model_weights(top_k=3)}

        combined = self._execute_combination_strategy(strategy, params)
        self.last_combination = combined

        print("(-----------------------------COMBINATION--------------------------------)\n\n\n")
        print(combined)
        print("(-----------------------------COMBINATION--------------------------------)\n\n\n")

        self._log("Combination complete")
        return json.dumps(combined)
    
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

            tools_called = get_context("tools_called", [])

            # Only require metrics; combination parameters are auto-filled deterministically if needed.
            analysis_tools_ok = "calculate_metrics_tool" in tools_called
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
        
        # Phase 2: Combination (deterministic tool call)
        combination_result = self.run_combination(analysis_result)

        description = analysis_result
        try:
            analysis_json = self._parse_analysis_result(analysis_result)
            strategy = analysis_json.get("recommended_strategy", "unknown")
            description = strategy
            strategy_params = analysis_json.get("strategy_parameters", {})
            reasoning = analysis_json.get("reasoning", "")
            
            
        except Exception:
            pass
            # description = f"Strategy execution completed. Analysis result was not valid JSON."
        
        # Parse combination result to extract the forecast
        try:
            if isinstance(combination_result, str):
                # We return json.dumps(list) from run_combination
                result_list = json.loads(combination_result)
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
        
        # Deterministic forced combination
        if strategy_params is None:
            try:
                import json
                analysis_json = json.loads(analysis_result)
                strategy_params = analysis_json.get("strategy_parameters", {})
            except Exception:
                strategy_params = {}

        combined = self._execute_combination_strategy(strategy, strategy_params)
        self.last_combination = combined
        
        self._print_phase_header("PIPELINE COMPLETE")
        
        self.last_result = {
            "analysis": analysis_result,
            "combination":  combined,
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
        
        import json
        combined = self._execute_combination_strategy(strategy, params)
        self.last_combination = combined
        return json.dumps(combined)


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