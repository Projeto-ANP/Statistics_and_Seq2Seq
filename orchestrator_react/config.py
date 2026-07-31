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
from typing import Any, Dict, Literal, Optional, Union


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
    #: Ollama's `reasoning` control, passed through only when set.
    #:
    #: `False` asks gpt-oss to stop spending its budget in the harmony reasoning
    #: channel. That channel is the documented cause of both failure modes seen on
    #: real runs: 90 EMPTY replies across the 182-series ANP run (the model
    #: reasons and then emits nothing in the final channel), and the
    #: "error parsing tool call: raw='We will call ...'" transport errors, where
    #: Ollama's own template tries to JSON-parse prose written into gpt-oss's
    #: tool-call channel (ollama/ollama#11781, #11800). `None` leaves the server
    #: default alone, which is what every result so far was produced under — so it
    #: stays the default here and turning it off is a deliberate A/B.
    #:
    #: Do NOT reach for `format="json"` as an alternative: on gpt-oss it makes the
    #: response empty every time (ollama/ollama#11867). The three-line text contract
    #: exists precisely to avoid depending on structured-output support.
    reasoning: Optional[bool] = None

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

    # -- ablation 3: ReAct loop budget ------------------------------------------
    # Raised from 8/2 after the v2 run: with only the full-pool seeds the agent
    # cleared the floor in 43 of 111 series and stopped early in 83. The stability
    # seeds raise that floor further, so it needs more room to find something that
    # clears it before the patience counter gives up on it.
    max_iterations: int = 12
    early_stop_patience: int = 4  # consecutive proposals without improvement
    min_improvement: float = 1e-4  # minimum relative gain to count as progress

    # -- ablation 4: prompt format ----------------------------------------------
    show_attempt_history: bool = True
    show_attempt_rationales: bool = True

    # -- ablation 2 (Phase 1 with/without an LLM) and ablations 5-6 (model per role)
    # `diagnostician.model = None` (the default) IS ablation 2's "off": Phase 1
    # then uses `deterministic_diagnosis` only. There used to be a second,
    # separate `diagnostic_llm: bool` flag here — removed because it could disagree
    # with whether `diagnostician.model` was actually set (e.g. an env-var override
    # via `REACT_MODEL_DIAGNOSTICIAN` landed on `diagnostician.model` but not on the
    # separate flag, so the LLM silently never ran while the config claimed it was
    # enabled). `LLMRole.enabled` is now the only signal, everywhere.
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
    #: Phase 4: how the attempt history becomes one forecast.
    #: "argmin" applies the single best-scoring strategy — the original contract.
    #: "ensemble" softmax-averages the top `final_top_m`.
    #:
    #: `ensemble` is NOT the default, and the reason is worth recording. The idea
    #: is sound in isolation: a three-window score orders strategies against the
    #: blind window at only Spearman +0.33, 98 of 111 NN5 series cannot separate
    #: first from second, and averaging over a noisy ranking beats betting on its
    #: top entry — worth 0.12036 -> 0.11948 sMAPE against the old seed set.
    #: But that gain is the *same* gain `seed_stable_pools` delivers, and it does
    #: not survive alongside it: with the stability seeds in place the ensemble is
    #: 0.11536 -> 0.11595 (p=0.62) on the deterministic arm and 0.11601 -> 0.11645
    #: (p=0.58) replayed over the agent's real trajectories. Both directions are
    #: inside the noise, so the simpler contract keeps the default and this stays
    #: as a measured, implemented ablation arm.
    #: "prior_blend" shrinks each attempt's 3-window score toward the dataset-level
    #: prior before taking the argmin (see `ReactState.blended_score`). It is the
    #: single largest ANP lever measured — 0.21904 -> 0.21453 at alpha 0.8, past
    #: FFORMA's 0.21659 — and it hurts NN5 monotonically over the same sweep
    #: (0.11539 -> 0.11879). Since no honest way to pick alpha was found (fixed,
    #: validation-selected, and stability-gated all fail — details in
    #: `blended_score`), it ships as an explicit opt-in rather than a default.
    final_strategy: Literal["argmin", "ensemble", "prior_blend"] = "argmin"
    #: Shrinkage weight for "prior_blend". 0.0 reproduces "argmin" exactly.
    final_prior_alpha: float = 0.0
    final_top_m: int = 3
    final_eta: float = 5.0
    #: Seed stability-selected combinations alongside the three full-pool
    #: baselines. The seeded set is the floor the whole run inherits whenever the
    #: agent finds nothing better — which was 68 of 111 series on the v2 NN5 run —
    #: so what goes in it matters more than its name suggests. See
    #: `pool.SEED_STABLE_POOLS`.
    seed_stable_pools: bool = True
    #: Trains `weights_pooled_meta_model` once per dataset run (see meta_model.py):
    #: a gradient-boosted regressor per pool model, predicting its error from THIS
    #: series' historical shape (trend/seasonal strength, entropy, autocorrelation),
    #: fit leave-one-series-out across every other series in the run. Exists
    #: because `weights_feature_based` fits the same kind of model per series on 3
    #: validation windows, which structurally can never clear its own "enough
    #: samples" guard (`n_fit < 2 * n_features` is true for every real feature
    #: count once `n_fit == 3`) — confirmed by that tool never once running its
    #: real path across every NN5/ANP run so far. Pooling across series is what
    #: gives the classical meta-learner (ADE's and FFORMA's own mechanism) enough
    #: samples to be worth trying. `False` skips the pre-pass entirely.
    pooled_meta_model: bool = True
    #: Also SEED `weighted(pooled_meta_model, pool)` as a Phase 2 baseline, so the
    #: cross-series model is exercised on every series instead of only when the
    #: agent happens to reach for it. Measured need: on the 182-series ANP v4 run
    #: the agent called that tool once in 182 series, and that single call failed —
    #: an unreachable tool cannot be evaluated. Same move that made
    #: `seed_stable_pools` pay off (see `pool.SEED_STABLE_POOLS`).
    seed_pooled_meta_model: bool = True
    #: Inject a DATASET CARD into every turn prompt: how each seeded strategy scored
    #: on VALIDATION across the other N-1 series (leave-one-series-out), plus which
    #: models most often rank top-3 and how often "weighted" collapsed to the pool
    #: mean here. Motivation, measured: the agent used 4-5 of ~10 usable tools on the
    #: 182-series ANP run, anchored on the one named in the prompt's worked example
    #: (462 -> 223 -> 10 -> 2 -> 1 uses down the weight family), and it had no way to
    #: know what works on THIS dataset. The card is a recommendation, never a
    #: restriction — the catalog stays open. Costs nothing extra: it reuses the
    #: pre-pass that `pooled_meta_model` already runs.
    dataset_card: bool = True
    #: Drop models whose windows do not match the rest of the pool, instead of
    #: failing the series. On ETTM1/ETTM2 five models (ONLY_CWT_catboost,
    #: ONLY_DWT_catboost, ONLY_DWT_rf, ONLY_FT_catboost, ONLY_FT_rf) were generated
    #: over a different period at a different sampling step than the other fourteen,
    #: so their actuals are different numbers — combining across them is adding two
    #: unrelated quantities, not averaging forecasts.
    #:
    #: The existing baselines do not fail there because they never look: `mean.py`
    #: takes the actuals from `models[0]` alone, and `ade.py`'s alignment check
    #: compares window COUNTS only, documenting that dates "podem divergir". So
    #: their ETTM numbers combine mismatched windows silently. Matching that
    #: behaviour would make our number comparable to theirs and equally
    #: meaningless; dropping the five gives a smaller pool that is actually
    #: coherent. Which models were dropped is recorded per series, so the smaller
    #: pool is never invisible in the comparison.
    drop_misaligned_models: bool = True
    #: Training objective for the pooled meta-model. Measured head-to-head (same
    #: 26 features, same LOSO folds, only the objective differing):
    #:   "fforma"    ANP 0.2159 — past the real FFORMA baseline (0.2166) — NN5 0.1197
    #:   "per_model" ANP 0.2205 (≈ plain mean),                          NN5 0.1188
    #: Neither transfers across datasets (the project's recurring finding), so the
    #: loser stays available as the ablation arm. "fforma" is the default because
    #: ANP is where the architecture is currently behind, and because on NN5 the
    #: seeded floor (stable pools) dominates either variant anyway.
    pooled_meta_model_objective: Literal["per_model", "fforma"] = "fforma"
    #: Below this many series in the run, pooling has too little signal to be
    #: worth it — `weights_pooled_meta_model` is withheld for every series, the
    #: same way `weights_ols` is withheld under too few validation windows.
    pooled_meta_model_min_series: int = 20
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
