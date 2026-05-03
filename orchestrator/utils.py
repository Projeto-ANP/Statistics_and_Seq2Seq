from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass

@dataclass
class ModelConfig:
    model: str
    temperature: float

def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks and orphaned </think> tags."""
    if not isinstance(text, str):
        return ""
    # Remove complete <think>...</think> blocks (including nested/multiple).
    while True:
        start = text.find("<think>")
        end = text.find("</think>")
        if start != -1 and end != -1 and end > start:
            text = text[:start] + text[end + len("</think>"):]
        elif end != -1:
            # Orphaned </think> (no opening tag) — discard everything before it.
            text = text[end + len("</think>"):]
        else:
            break
    return text


def _extract_fenced_json(text: str) -> Optional[str]:
    """Return content of the LAST ```json ... ``` or ``` ... ``` block found."""
    # Try ```json first, then plain ```
    for pattern in (r"```json\s*([\s\S]+?)\s*```", r"```\s*([\s\S]+?)\s*```"):
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the last match (most likely the final answer)
            return matches[-1].strip()
    return None


def _strip_code_fences(text: str) -> str:
    """Legacy helper kept for call-sites that import it directly."""
    text = text.strip()
    # If there's a fenced block anywhere, extract it.
    fenced = _extract_fenced_json(text)
    if fenced:
        return fenced
    # Fallback: strip leading/trailing fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> Any:
    """Best-effort JSON extraction from LLM outputs (object or list).

    Handles:
    - Complete <think>...</think> and orphaned </think> tags.
    - JSON wrapped in ```json ... ``` or ``` ... ``` anywhere in the text.
    - Bare JSON objects/arrays.
    - Scans for the outermost matching { } pair with backtracking so a stray
      } inside a string value doesn't cause rfind to pick the wrong position.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # 1. Strip think blocks first.
    text = strip_think_blocks(text)

    # 2. Try extracting from fenced code blocks (anywhere in the text).
    fenced = _extract_fenced_json(text)
    candidates = [fenced] if fenced else []
    candidates.append(text.strip())

    for candidate in candidates:
        if not candidate:
            continue

        # Fast path: valid JSON as-is.
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

        # Scan for outermost { ... } using a bracket counter (handles nesting).
        result = _extract_outermost(candidate, "{", "}")
        if result is not None:
            return result

        # Scan for outermost [ ... ] in case the response is an array.
        result = _extract_outermost(candidate, "[", "]")
        if result is not None:
            return result

    return None


def _extract_outermost(text: str, open_ch: str, close_ch: str) -> Any:
    """Find the outermost matching bracket pair and try json.loads on it."""
    start = text.find(open_ch)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                frag = text[start:i + 1]
                try:
                    return json.loads(frag)
                except (json.JSONDecodeError, ValueError):
                    return None
    return None
