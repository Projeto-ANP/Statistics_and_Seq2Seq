from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ALLOWED_STRATEGY_TYPES = {
    "baseline",
    "selection",
    "weighted",
    "stacking",
    "regime",
}


@dataclass
class CandidateStrategy:
    """LLM-proposed candidate strategy, evaluated deterministically."""

    name: str
    type: str
    description: str
    formula: str
    learns_weights: bool = False
    constraints: str = ""
    risks: List[str] = field(default_factory=list)
    validation_plan: str = "rolling"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "formula": self.formula,
            "learns_weights": self.learns_weights,
            "constraints": self.constraints,
            "risks": self.risks,
            "validation_plan": self.validation_plan,
            "params": self.params,
        }


def candidate_from_dict(d: Dict[str, Any]) -> CandidateStrategy:
    name = str(d.get("name", "")).strip() or "unnamed"
    stype = str(d.get("type", "baseline")).strip() or "baseline"
    if stype not in ALLOWED_STRATEGY_TYPES:
        stype = "baseline"

    description = str(d.get("description", "")).strip()
    formula = str(d.get("formula", "")).strip()

    risks = d.get("risks", [])
    if isinstance(risks, str):
        risks = [risks]
    if not isinstance(risks, list):
        risks = []

    params = d.get("params", {})
    if not isinstance(params, dict):
        params = {}

    return CandidateStrategy(
        name=name,
        type=stype,
        description=description,
        formula=formula,
        learns_weights=bool(d.get("learns_weights", False)),
        constraints=str(d.get("constraints", "")),
        risks=[str(r).strip() for r in risks if str(r).strip()],
        validation_plan=str(d.get("validation_plan", "rolling")),
        params=params,
    )


def parse_candidates(obj: Any) -> List[CandidateStrategy]:
    if obj is None:
        return []
    if isinstance(obj, CandidateStrategy):
        return [obj]
    if isinstance(obj, dict):
        return [candidate_from_dict(obj)]
    if isinstance(obj, list):
        out: List[CandidateStrategy] = []
        for item in obj:
            if isinstance(item, CandidateStrategy):
                out.append(item)
            elif isinstance(item, dict):
                out.append(candidate_from_dict(item))
        return out
    return []
