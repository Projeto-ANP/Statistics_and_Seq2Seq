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
from typing import Any, Dict, List, Literal, Optional, Union


# Composite-score weights (lower is better). Kept compatible with the historical
# presets in `orchestrator/tools.py` so versions remain comparable.
SCORE_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": {"a_rmse": 0.3, "b_smape": 0.3, "c_mape": 0.2, "d_pocid": 0.2},
    "rmse_focus": {"a_rmse": 0.5, "b_smape": 0.2, "c_mape": 0.2, "d_pocid": 0.1},
    "direction_focus": {"a_rmse": 0.25, "b_smape": 0.25, "c_mape": 0.1, "d_pocid": 0.4},
    "robust_smape": {"a_rmse": 0.2, "b_smape": 0.5, "c_mape": 0.1, "d_pocid": 0.2},
    # Series crossing zero (ETT) make MAPE/SMAPE unstable; this preset drops them.
    "scale_free_safe": {"a_rmse": 0.7, "b_smape": 0.1, "c_mape": 0.0, "d_pocid": 0.2},
    # Error/direction only: the two metrics the thesis compares. MAPE and RMSE
    # are deliberately zero here, so scale and the zero-crossing MAPE artefact
    # cannot leak into the choice.
    "smape_pocid_50_50": {"a_rmse": 0.0, "b_smape": 0.5, "c_mape": 0.0, "d_pocid": 0.5},
    "smape60_pocid40": {"a_rmse": 0.0, "b_smape": 0.6, "c_mape": 0.0, "d_pocid": 0.4},
    "smape40_pocid60": {"a_rmse": 0.0, "b_smape": 0.4, "c_mape": 0.0, "d_pocid": 0.6},
    "smape_only": {"a_rmse": 0.0, "b_smape": 1.0, "c_mape": 0.0, "d_pocid": 0.0},
    "pocid_only": {"a_rmse": 0.0, "b_smape": 0.0, "c_mape": 0.0, "d_pocid": 1.0},
}

FINAL_SCORE_PRESETS: tuple[str, ...] = (
    "smape_pocid_50_50",
    "smape60_pocid40",
    "smape40_pocid60",
)


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
    #:
    #: Booleans are the universal control: `False` is "off" and `True` is "on" for
    #: models that support thinking. Any other string is a named level, defined per
    #: model and discovered with `/api/show`: gpt-oss low/medium/high (default
    #: medium), qwen3.8 low/medium/xhigh (default xhigh), muse-glimmer
    #: low/medium/high/max (default high). An unsupported name falls back to the
    #: model default inside Ollama instead of erroring, so check the level per model
    #: before comparing runs. Booleans and strings are NOT interchangeable to
    #: Ollama: `bool("low")` is True, which is not the same as "low".
    reasoning: Optional[Union[bool, str]] = None

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
    #: Experiment: ignore both terminal accept and the patience break, so the
    #: agent uses the full iteration budget. The final strategy is still the
    #: best-scoring attempt of the whole history.
    no_early_stop: bool = False
    #: Experiment: `max_iterations` is the number of scored combinations, not the
    #: number of tool turns. Preparation actions do not consume the budget; a
    #: separate hard cap on total turns prevents an agent that never scores.
    scoring_budget: bool = False
    #: Hard cap on total tool turns when `scoring_budget` is on. 0 means
    #: `2 * max_iterations`.
    max_total_turns: int = 0
    min_improvement: float = 1e-4  # minimum relative gain to count as progress

    # -- ablation 4: prompt format ----------------------------------------------
    show_attempt_history: bool = True
    #: How the three validation-window scores are aggregated into `Attempt.score`.
    #: "mean" is the historical mean; "last"/"last2" emphasise the window closest
    #: to the test; "weighted" uses 0.2/0.3/0.5 for 3 windows.
    score_window_mode: Literal["mean", "last", "last2", "weighted"] = "mean"
    #: How the agent's prompts are delimited: "xml" (default; each section wrapped in
    #: a tag) or "text" (headers such as `SERIES PROFILE:`, the format of every result
    #: before v1_gpt). Same wording either way: an A/B on structure only.
    prompt_format: str = "xml"
    #: Catalog and prompt of the v2 run. Both default OFF, so a default config
    #: reproduces the published behaviour and the old prompt stays available for
    #: the A/B. What they change:
    #:
    #: `catalog_v2` hides the diagnostics already present in the turn context,
    #: the duplicated `weights_softmax_neg_error`, the two split meta-model tools
    #: (replaced by `weights_meta_model`), the `combine_*` sugar tools that only
    #: rebuild what `evaluate_strategy` scores, and the two reporting tools whose
    #: content the turn already carries. It also adds `select_forward` (Caruana).
    #: The closed catalog keeps 10 entries.
    #:
    #: `extra_actions` additionally exposes the recipes tested offline in
    #: `aprovacao_solida/07_novas_combinacoes`: `weights_best_per_horizon`,
    #: `weights_topk_per_horizon` and `weights_forward`. On by default as agent
    #: options: the offline battery rejected them as forced picks (validation
    #: selects them, the test does not confirm), so what they add is pool and
    #: weighting diversity for the agent to choose per series, not a new floor.
    #: The CLI switch is `--no-extra-actions`, which restores the ten-entry
    #: catalog for the ablation.
    #: RESEARCH: tool selection degrades as the catalog grows, and overlapping
    #: or ambiguous tools make it worse; fewer, well-described actions are
    #: easier for local models to choose from.
    catalog_v2: bool = False
    #: Use the restructured system/turn prompt: role, four-step procedure,
    #: catalog grouped by phase, two worked examples with different kinds of
    #: strategy, and an explicit stop rule that reads the leader diagnostics.
    prompt_v2: bool = False
    extra_actions: bool = True
    #: Catalog experiments. Each is independent and OFF by default, so a default
    #: config reproduces the published behaviour; they exist to be turned on one at
    #: a time. See `RESUMO_*.md` under outputs/resultados/ablacao.
    #:
    #: List `weights_softmax_neg_error` before `weights_inverse_error` (catalog and
    #: the prompt paragraph that describes them as similar to each other).
    reorder_weight_tools: bool = False
    #: Withhold `combine_mean/median/weighted/best_single` (24 -> 20 actions): each
    #: only builds the spec that `evaluate_strategy(combine=...)` builds and scores.
    drop_redundant_combine_actions: bool = False
    #: Seed only dba + the six stability pools + the pooled meta-model (8 seeds
    #: instead of 10): no plain mean/median over the full pool.
    reduced_seeding: bool = False
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
    #: "seed_gate" keeps the best seeded baseline unless an agent proposal beats
    #: it on validation by a relative margin (`seed_gate_margin`). It exists to
    #: blunt the winner's curse: the final strategy is the max over every attempt
    #: tried, so more agent iterations also mean more chances to fit validation
    #: noise. A margin small enough to be real (1% by default) keeps the seed
    #: unless the agent's gain is larger than the 3-window noise usually is.
    #: "mcs" (Model Confidence Set, trimmed to the members the validation cannot
    #: separate from the leader by Diebold-Mariano) averages the confidence set
    #: instead of picking its top entry. It exists for the measured failure mode:
    #: first and second place are indistinguishable in most series, and betting on
    #: the top one keeps losing to the deterministic floor. See `ReactState.apply_mcs`.
    final_strategy: Literal["argmin", "ensemble", "prior_blend", "seed_gate", "mcs"] = "argmin"
    #: Shrinkage weight for "prior_blend". 0.0 reproduces "argmin" exactly.
    final_prior_alpha: float = 0.0
    #: Relative validation-score improvement an agent attempt must show over the
    #: best seeded baseline to be applied under `final_strategy="seed_gate"`.
    seed_gate_margin: float = 0.01
    final_top_m: int = 3
    final_eta: float = 5.0
    #: Significance level of the Diebold-Mariano test that decides MCS membership:
    #: an attempt with p < `mcs_alpha` against the leader is rejected from the
    #: committee. 0.10 matches the verdict threshold used everywhere else.
    mcs_alpha: float = 0.10
    #: Optional cap on the committee size (leader included). None keeps every
    #: attempt the test could not separate.
    mcs_max_members: Optional[int] = None
    #: Practical tie band, in normalized score units: a candidate more than this
    #: far behind the leader stays out of the committee even when the DM test has
    #: no power against heavy-tailed residuals (measured on the ETTM2 `dba` seed).
    mcs_score_margin: float = 0.10
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
    #: Composite-score weights used to rank candidate strategies. Any key of
    #: `SCORE_PRESETS`, or `"auto_loo"`: the harness then chooses, per series, the
    #: preset that a leave-one-window-out protocol on VALIDATION favours (see
    #: `ReactState.auto_score_weights`). `"auto_loo"` never reads the test window;
    #: the test actuals are not even reachable from the state the harness ranks.
    #:
    #: `score_auto_candidates` narrows the candidate presets for the automatic
    #: choice; `None` means all of them.
    score_preset: str = "balanced"
    score_auto_candidates: Optional[List[str]] = None
    multi_preset_final: bool = False
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
        d = self.to_dict()
        # An experiment flag left at False must not change the id of a run that
        # never heard of it.
        for flag in (
            "reorder_weight_tools",
            "drop_redundant_combine_actions",
            "reduced_seeding",
            "no_early_stop",
            "scoring_budget",
            "max_total_turns",
            "multi_preset_final",
        ):
            if not d.get(flag):
                d.pop(flag, None)
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
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
