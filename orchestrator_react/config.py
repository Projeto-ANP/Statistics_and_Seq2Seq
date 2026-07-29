"""Single configuration object per experiment run.

Covers the six ablations of Step 5 of the specification. A whole `ReactConfig` is
serialised into the `ablation_config` CSV field, so any result row can be
reconstructed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


# Composite-score weights (lower is better). Kept compatible with the historical
# presets in `orchestrator/tools.py` so versions remain comparable.
SCORE_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": {"a_rmse": 0.3, "b_smape": 0.3, "c_mape": 0.2, "d_pocid": 0.2},
    "rmse_focus": {"a_rmse": 0.5, "b_smape": 0.2, "c_mape": 0.2, "d_pocid": 0.1},
    "direction_focus": {"a_rmse": 0.25, "b_smape": 0.25, "c_mape": 0.1, "d_pocid": 0.4},
    "robust_smape": {"a_rmse": 0.2, "b_smape": 0.5, "c_mape": 0.1, "d_pocid": 0.2},
    # Series crossing zero (ETT) make MAPE/SMAPE unstable; this preset drops them.
    "scale_free_safe": {"a_rmse": 0.7, "b_smape": 0.1, "c_mape": 0.0, "d_pocid": 0.2},
}


PoolMode = Literal["full", "top_k_error", "top_k_stable"]
BacktestMode = Literal["expanding", "loo"]


@dataclass
class LLMRole:
    """LLM model assigned to a role (Section 3.5). `model=None` disables the role."""

    model: Optional[str] = None
    temperature: float = 0.2
    base_url: str = "http://127.0.0.1:11434"
    #: Sampling seed handed to Ollama. Without it the same series can be given the
    #: same prompt twice and produce two different strategies — measured on the
    #: three duplicate NN5 series (T1/T47, T11/T50, T79/T111), where the agent
    #: chose a different combination all three times. `None` leaves sampling free.
    seed: Optional[int] = 7

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    def label(self) -> str:
        return self.model or "none"


@dataclass
class ReactConfig:
    """Full run configuration. Everything that changes the result lives here."""

    # -- identity --------------------------------------------------------------
    name: str = "default"

    # -- ablation 1: pool composition -------------------------------------------
    pool_mode: PoolMode = "full"
    pool_k: int = 8

    # -- ablation 2: Phase 1 (diagnosis) with or without LLM ---------------------
    diagnostic_llm: bool = False

    # -- ablation 3: ReAct loop budget ------------------------------------------
    max_iterations: int = 8
    early_stop_patience: int = 2  # consecutive proposals without improvement
    min_improvement: float = 1e-4  # minimum relative gain to count as progress

    # -- ablation 4: prompt format ----------------------------------------------
    show_attempt_history: bool = True
    show_attempt_rationales: bool = True

    # -- ablations 5 and 6: LLM model per role -----------------------------------
    combinator: LLMRole = field(default_factory=lambda: LLMRole(model="gpt-oss:20b", temperature=0.2))
    diagnostician: LLMRole = field(default_factory=LLMRole)
    reporter: LLMRole = field(default_factory=LLMRole)

    # -- deterministic protocol (not an ablation: the evaluation contract) -------
    backtest_mode: BacktestMode = "expanding"
    #: Re-choose pool membership inside each backtest fold instead of once over all
    #: windows. Without it the same windows that pick the models also score them,
    #: and the validation score stops predicting the test window: ranking sixteen
    #: fixed rules on 111 NN5 series gives Spearman -0.718 between in-sample
    #: validation and blind test, against +0.288 with nesting on. Kept as a flag
    #: because the off state is the protocol every earlier result was produced
    #: under, and `ablation_config` has to be able to say which one ran.
    nested_selection: bool = True
    score_preset: str = "balanced"
    n_validation_windows: int = 3
    seasonal_period: Optional[int] = None  # None => inferred from the frequency
    mape_zero: Literal["skip", "epsilon"] = "skip"
    mape_epsilon: float = 1e-8

    # -- guardrails --------------------------------------------------------------
    # Granger-Ramanathan least squares needs more independent equations than a
    # 3-window backtest provides. Below this many windows the simplex projection
    # lands on a vertex, so `weights_ols` silently degenerates into a model
    # *selection* whose winner need not be the lowest-error model (that is what
    # `select_top_k`/`best_single` are for). Under the threshold the tool is
    # withheld from the catalog rather than left to mislead the agent.
    min_windows_for_ols: int = 5
    calibration_gate: bool = False  # skip the loop when the ranking is already stable
    calibration_gate_kendall: float = 0.85
    sanity_check_tolerance: float = 3.0  # multiples of the historical std

    def score_weights(self) -> Dict[str, float]:
        return dict(SCORE_PRESETS.get(self.score_preset, SCORE_PRESETS["balanced"]))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short, stable hash of the configuration — goes into `ablation_config`."""
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return f"{self.name}-{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:10]}"

    # -- external loading (Section 3.5: swap models without touching code) -------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReactConfig":
        data = dict(data or {})
        roles = {}
        for role in ("combinator", "diagnostician", "reporter"):
            raw = data.pop(role, None)
            if isinstance(raw, dict):
                roles[role] = LLMRole(**raw)
            elif isinstance(raw, str):
                roles[role] = LLMRole(model=raw)
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean, **roles)

    @classmethod
    def from_json_file(cls, path: str) -> "ReactConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_env(cls, base: Optional["ReactConfig"] = None) -> "ReactConfig":
        """Overrides each role's model from environment variables.

        REACT_MODEL_COMBINATOR, REACT_MODEL_DIAGNOSTICIAN, REACT_MODEL_REPORTER,
        REACT_OLLAMA_URL, REACT_CONFIG (path to a JSON applied first).
        """
        cfg = base
        if cfg is None:
            path = os.environ.get("REACT_CONFIG")
            cfg = cls.from_json_file(path) if path else cls()

        url = os.environ.get("REACT_OLLAMA_URL")
        for role, env_key in (
            ("combinator", "REACT_MODEL_COMBINATOR"),
            ("diagnostician", "REACT_MODEL_DIAGNOSTICIAN"),
            ("reporter", "REACT_MODEL_REPORTER"),
        ):
            current: LLMRole = getattr(cfg, role)
            model = os.environ.get(env_key)
            if model is not None:
                current.model = None if model.strip().lower() in {"", "none"} else model
            if url:
                current.base_url = url
        return cfg
