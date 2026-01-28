from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> Any:
    """Best-effort JSON extraction from LLM outputs (object or list)."""

    text = _strip_code_fences(text)
    if not text:
        return None

    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extract list
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        frag = text[start : end + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            pass

    # Try extract dict
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        frag = text[start : end + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            pass

    return None
