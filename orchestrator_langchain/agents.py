from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from orchestrator_langchain.langchain_tools import debate_packet, evaluate_strategies, proposer_brief


DEFAULT_MODEL_ID = "mychen76/qwen3_cline_roocode:4b"


@dataclass
class AgentResponse:
    content: str


def _load_prompt(name: str) -> str:
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    prompt_path = prompts_dir / name
    return prompt_path.read_text(encoding="utf-8")


class LangchainAgent:
    def __init__(
        self,
        model_id: str,
        tools: List[BaseTool],
        system_prompt: str,
        temperature: float = 0.2,
        force_tool_call: bool = True,
        max_tool_rounds: int = 3,
    ) -> None:
        self._tools = tools
        self._system_prompt = system_prompt
        self._force_tool_call = force_tool_call
        self._max_tool_rounds = max_tool_rounds
        self._llm = ChatOllama(model=model_id, temperature=temperature)
        self._bound_llm = self._bind_tools(self._llm, tools)
        self._tool_map = {t.name: t for t in tools}

    def _bind_tools(self, llm: ChatOllama, tools: List[BaseTool]) -> ChatOllama:
        if not tools:
            return llm
        try:
            return llm.bind_tools(tools, tool_choice="required")
        except Exception:
            return llm.bind_tools(tools)

    def _execute_tool(self, name: str, args: dict) -> str:
        tool = self._tool_map.get(name)
        if tool is None:
            return f"{{\"error\": \"unknown tool: {name}\"}}"
        return tool.invoke(args or {})

    def run(self, user_prompt: str) -> AgentResponse:
        messages = [SystemMessage(content=self._system_prompt), HumanMessage(content=user_prompt)]
        tool_called = False

        for step in range(self._max_tool_rounds):
            response = self._bound_llm.invoke(messages)
            if isinstance(response, AIMessage) and response.tool_calls:
                tool_called = True
                for tc in response.tool_calls:
                    name = tc.get("name")
                    args = tc.get("args", {})
                    tool_result = self._execute_tool(name, args)
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc.get("id", "")))
                continue

            if self._force_tool_call and self._tools and not tool_called and step == 0:
                messages.append(
                    HumanMessage(
                        content=(
                            "You MUST call the required tool before responding. "
                            "Return ONLY JSON after the tool output."
                        )
                    )
                )
                continue

            content = response.content if isinstance(response, AIMessage) else str(response)
            return AgentResponse(content=str(content))

        return AgentResponse(content="{}");


def create_proposer_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> LangchainAgent:
    _ = debug
    return LangchainAgent(
        model_id=model_id,
        tools=[proposer_brief],
        system_prompt=_load_prompt("proposer.md"),
        temperature=0.2,
        force_tool_call=True,
    )


def create_skeptic_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> LangchainAgent:
    _ = debug
    return LangchainAgent(
        model_id=model_id,
        tools=[debate_packet],
        system_prompt=_load_prompt("skeptic.md"),
        temperature=0.2,
        force_tool_call=True,
    )


def create_statistician_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> LangchainAgent:
    _ = debug
    return LangchainAgent(
        model_id=model_id,
        tools=[debate_packet],
        system_prompt=_load_prompt("statistician.md"),
        temperature=0.2,
        force_tool_call=True,
    )


def create_orchestrator_agent(model_id: str = DEFAULT_MODEL_ID, debug: bool = False) -> LangchainAgent:
    _ = debug
    return LangchainAgent(
        model_id=model_id,
        tools=[evaluate_strategies],
        system_prompt=_load_prompt("orchestrator.md"),
        temperature=0.15,
        force_tool_call=True,
    )
