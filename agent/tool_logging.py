import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def log_tool_event(
    *,
    base_dir: str,
    dataset_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    """Append a JSONL log event to a per-dataset log file.

    Returns the log file path.
    """
    os.makedirs(base_dir, exist_ok=True)
    dataset_file = f"{_safe_filename(dataset_id)}.jsonl"
    path = os.path.join(base_dir, dataset_file)

    record = {
        "ts": _now_iso(),
        "dataset_id": dataset_id,
        "event": event,
        **payload,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path
