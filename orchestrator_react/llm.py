"""LLM clients and ReAct output parsing.

The loop in `react_loop.py` talks to an LLM only through the tiny `LLMClient`
protocol below, so the whole Phase 3 can be exercised with a scripted client and no
server running. `OllamaClient` is the real one; it imports langchain lazily so this
module stays importable in a plain numpy environment.

Section 3.2, principle 9: no fine-tuning anywhere. These are prompting-only clients
over off-the-shelf models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from orchestrator_react.config import LLMRole


class LLMError(RuntimeError):
    """The model could not be reached or returned nothing usable."""


class LLMClient(Protocol):
    """Everything the ReAct loop needs from a language model."""

    name: str

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - protocol
        ...


# ──────────────────────────────────────────────────────────────────────────────
# clients
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ScriptedLLM:
    """Deterministic stand-in for tests: replays canned responses in order."""

    responses: Sequence[str]
    name: str = "scripted"
    calls: List[Dict[str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        idx = len(self.calls) - 1
        if idx >= len(self.responses):
            raise LLMError(
                f"scripted client exhausted after {len(self.responses)} responses; "
                "the loop asked for one more turn than the script provides"
            )
        return self.responses[idx]


@dataclass
class OllamaClient:
    """Real client over a local Ollama server (default http://127.0.0.1:11434)."""

    role: LLMRole
    num_ctx: int = 8192
    timeout: float = 600.0
    _chat: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.role.label()

    def _client(self) -> Any:
        if self._chat is None:
            try:
                from langchain_ollama import ChatOllama
            except Exception as exc:  # pragma: no cover - environment dependent
                raise LLMError(
                    "langchain-ollama is not installed; see EXTRA_DEPENDENCIES.txt"
                ) from exc
            kwargs: Dict[str, Any] = {
                "model": self.role.model,
                "temperature": float(self.role.temperature),
                "base_url": self.role.base_url,
                "num_ctx": int(self.num_ctx),
            }
            # Only pass these when configured: `None` is not the same as omitting
            # them for every Ollama build.
            if getattr(self.role, "seed", None) is not None:
                kwargs["seed"] = int(self.role.seed)
            if getattr(self.role, "reasoning", None) is not None:
                kwargs["reasoning"] = bool(self.role.reasoning)
            self._chat = ChatOllama(**kwargs)
        return self._chat

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = self._client().invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise LLMError(f"Ollama call failed ({self.role.label()}): {exc}") from exc
        content = getattr(response, "content", response)
        if isinstance(content, list):  # some models return content blocks
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return str(content)


def build_client(role: LLMRole) -> Optional[LLMClient]:
    """Returns a client for a role, or None when the role is disabled."""
    return OllamaClient(role=role) if role.enabled else None


def check_client(client: Optional[LLMClient]) -> tuple[bool, str]:
    """Preflight: one trivial call, to fail in seconds instead of per series.

    An unreachable server or a model that was never pulled fails identically on
    every series. Without this the whole dataset runs on the deterministic
    fallback and the log says `ok`, which is the worst possible outcome: a run
    that looks successful and answers a different question.
    """
    if client is None:
        return True, "no client configured"
    try:
        reply = client.complete(
            "Reply with the single word OK.", "Say OK."
        )
    except LLMError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    text = str(reply).strip()
    if not text:
        return False, "the model answered with an empty string"
    return True, text[:120]


# ──────────────────────────────────────────────────────────────────────────────
# parsing
# ──────────────────────────────────────────────────────────────────────────────


_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def split_think(text: str) -> tuple[str, str]:
    """Separates `<think>...</think>` reasoning from the answer body.

    Qwen3 and gpt-oss emit these blocks; they are the model's reasoning, which is
    exactly what belongs in the `thought` field of the trajectory. An orphan
    closing tag (thinking truncated) is handled too.
    """
    if not isinstance(text, str) or not text:
        return "", ""
    thoughts = [m.strip() for m in _THINK.findall(text)]
    body = _THINK.sub("", text)
    if "</think>" in body.lower():
        head, _, tail = body.partition("</think>")
        thoughts.append(head.strip())
        body = tail
    return "\n\n".join(t for t in thoughts if t), body.strip()


def extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction: fenced block, then outermost braces."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidates: List[str] = []
    fenced = _FENCE.findall(text)
    candidates.extend(f.strip() for f in fenced)
    candidates.append(text.strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            found = _scan_balanced(candidate, opener, closer)
            if found is not None:
                return found
    return None


def _scan_balanced(text: str, opener: str, closer: str) -> Optional[Any]:
    start = text.find(opener)
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


@dataclass
class AgentStep:
    """One parsed turn of the agent."""

    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    parse_error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.action) and not self.parse_error


_ACTION = re.compile(r"^\s*action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_INPUT = re.compile(r"action\s*_?\s*input\s*:", re.IGNORECASE)
_THOUGHT = re.compile(r"^\s*thought\s*:\s*(.*?)$", re.IGNORECASE | re.MULTILINE)


def parse_agent_step(text: str) -> AgentStep:
    """Parses `Thought: / Action: / Action Input:` leniently.

    Small local models are inconsistent: they fence the JSON, drop the
    `Action Input` label, wrap everything in a single JSON object, or emit the
    reasoning inside `<think>` tags. All of those are accepted — a rigid parser
    would turn a usable answer into a wasted iteration.
    """
    step = AgentStep(raw=text or "")
    thinking, body = split_think(text or "")
    step.thought = thinking

    if not body.strip():
        step.parse_error = "empty response"
        return step

    # Form 1: a single JSON object carrying the whole step.
    obj = extract_json(body)
    if isinstance(obj, dict) and ("action" in obj or "tool" in obj):
        step.action = str(obj.get("action") or obj.get("tool") or "").strip()
        args = obj.get("action_input", obj.get("args", obj.get("input", {})))
        if isinstance(args, str):
            args = extract_json(args) or {}
        step.action_input = args if isinstance(args, dict) else {}
        if not step.thought:
            step.thought = str(obj.get("thought") or obj.get("reasoning") or "").strip()
        if not step.action:
            step.parse_error = "json step without an 'action' field"
        return step

    # Form 2: the labelled ReAct layout.
    matches = _ACTION.findall(body)
    action = ""
    for candidate in matches:
        cleaned = candidate.strip().strip("`\"'").split()[0] if candidate.strip() else ""
        if cleaned and not _ACTION_INPUT.match(candidate.strip()):
            action = cleaned
            break
    if not action:
        step.parse_error = "no 'Action:' line found"
        return step
    step.action = action

    if not step.thought:
        thoughts = _THOUGHT.findall(body)
        step.thought = thoughts[0].strip() if thoughts else ""

    tail = body[body.lower().find("action input") :] if _ACTION_INPUT.search(body) else body
    args = extract_json(tail)
    step.action_input = args if isinstance(args, dict) else {}
    return step
