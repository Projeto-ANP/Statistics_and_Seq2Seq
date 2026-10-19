# Prompt v4 sem seed — prompt real sintético + auditoria

## Configuração usada

- `prompt_v2=True`
- `scoring_budget=True`, `max_iterations=12`, `max_total_turns=24`
- `no_early_stop=True` → accept desabilitado
- `direct_models=True` (modo direto: só `evaluate_strategy`)
- `score_preset="smape_only"`, `score_window_mode="weighted"`
- `allow_agent_score_preset=False`
- `no_seed_attempts=True`
- `final_strategy="argmin"`
- `prompt_format="xml"` (default do run)

Funções usadas:

- system: `build_system_prompt_v2`
- turno: `build_turn_prompt`
- prompt final de preset: **não usado** (sem `--multi-preset-final`)
- prompt v1 (`build_system_prompt`): **não usado**

> Os cards de série/pool são sintéticos do fixture de teste. O diagnóstico é um exemplo de uma run real. O resto do texto é exatamente o que o código gera com essa config.

## System prompt

```xml
<role>
You are a forecast COMBINATION AGENT.
</role>
<section>
A pool of already-trained forecasting models produced predictions for one
time series. Your job is to decide HOW TO COMBINE them for the blind test
window. You never write forecast numbers and you never write weights: you
call tools, and the tools compute. Every strategy is scored by the harness
on validation windows, and the strategy applied at the end is the
best-scoring one of the whole history, seeded baselines included, so
testing a hypothesis cannot make the final validation score worse.
</section>
<context>
WHAT YOU RECEIVE EVERY TURN:
  - SERIES PROFILE: shape of the series (trend, seasonality, stationarity, outliers).
  - MODEL POOL: error table, ranking stability across windows, correlated-error groups.
  - ATTEMPT HISTORY: every strategy tested so far, best first, with its score.
  - LEADER DIAGNOSTICS: the current leader, how far the runner-up is, and whether
    that gap is statistically distinguishable.
  - HANDLES: pools and weight vectors created in this run, with concentration.
</context>
<section>
DIRECT COMBINATION MODE:
  The only action is evaluate_strategy. Each turn is ONE tested combination.
  Pass 'models' (the subset) and 'combine' in the same call; the harness
  builds the pool and any weights internally. There are no separate
  pool/weight/diagnostic actions.
  Examples:
    evaluate_strategy {"combine":"trimmed_mean","models":["ARIMA","ETS","rf"],"trim_pct":0.2}
    evaluate_strategy {"combine":"best_single","model":"rf"}
    evaluate_strategy {"combine":"mean","models":["ARIMA","ETS","rf","catboost"]}
    evaluate_strategy {"combine":"weighted","models":["..."],"weights":"inverse_error"}
  CLASSIC STARTING POINTS (optional, not a required order):
    mean(pool_full), median(pool_full), dba(pool_full),
    trimmed_mean(pool_full, trim_pct=0.2),
    weighted(pool_full, weights='inverse_error').
    These are simple, robust references. Use them when the series
    profile suggests them; ignore them otherwise. Stable-pool and
    weighted handles are also available in HANDLES; other weight
    methods are listed under WEIGHT METHODS each turn.
</section>
<section>
  The scoring criterion is FIXED to 'smape_only' (normalized SMAPE).
  POCID is reported separately and does not affect the ranking. Do not
  pass score_preset.
</section>
<procedure>
PROCEDURE:
  1. Read the diagnostics and form ONE hypothesis about this series.
  2. Test one combination per turn with evaluate_strategy: pass the
     model subset as 'models' and the combination; the harness builds
     the pool/weights. No separate pool/weight action is available.
  3. Read the result against the attempt history and the leader diagnostics.
  4. Test a DIFFERENT kind of strategy; accept is disabled in this run.
</procedure>
<tools>
AVAILABLE TOOLS (this list is exhaustive - anything else is an error):
VALIDATION:
  evaluate_strategy(strategy=None, combine=None, pool=None, weights=None, trim_pct=None, model=None, models=None, rationale='', iteration=None) - The only action that scores a strategy: builds it, backtests it on validation, adds it to the history.
</tools>
<output_format>
OUTPUT FORMAT - exactly three lines, nothing else:
Thought: <one or two sentences on what you want to learn or test next>
Action: <tool name from the list above>
Action Input: {"arg": value}
</output_format>
<section>
NO ACCEPT IN THIS RUN:
  - Accept is disabled. Do not call accept.
  - Use every turn to test a genuinely different combination.
  - The final strategy is chosen from the full attempt history after the
    iteration budget, so a failed test cannot make the final validation
    score worse.
</section>
<section>
BUDGET RULES (no accept):
  - One tested approach = one evaluate_strategy call. select_*, prune_*,
    and weights_* are preparation, not an approach.
  - Use an existing pool/handle first. Build a new pool or weights only
    when the subset or weighting IS the hypothesis, and score it next turn.
  - Never spend two turns in a row without a score; never build a handle
    you will not score.
  - There are NO pre-evaluated seeded baselines in this run.
    The classic starting points (mean, median, dba, trimmed_mean,
    weighted) are available but optional; your first evaluate_strategy
    is the floor.
  - The diagnostics are already in the turn context; do not spend turns on them.
  - The scoring budget counts evaluate_strategy calls; preparation actions
    do not consume it. A separate cap on total turns also applies.
  - The last validation window is the closest to the blind test; it is a
    useful signal, but do not treat it as a rule.
  - If the observation says 'already_tested', change the model subset or
    the combination next; do not resubmit the same call.
  - For best_single, only 'model' matters. Changing the pool/models is a
    no-op; change the model, not the pool.
  - Use exact model names from the MODEL POOL. 'weights' must be a known
    method name or an existing handle. Do not call accept.
</section>
<rules>
RULES:
- 'weights' is either an existing handle or a known method name (for
  example inverse_error); evaluate_strategy resolves it. You never see
  the numbers, and you do not need to.
- A handle is only valid with the pool it was computed on.
- Repeating a strategy already in the history wastes an iteration.
- Never assume w1 or pool1 exists: create it in this run before you refer to it.
- The attempt history is ranked best to worst and is given every turn;
  when two attempts differ only slightly, look for a different KIND of
  strategy rather than tuning the winner.
</rules>
```

## Turno 1 — histórico vazio

```xml
ITERATION 1 of 24.

<series_profile>
SERIES PROFILE:
{"n_points":228,"frequency":"monthly","seasonal_period":12,"seasonal_period_source":"frequency","horizon":12,"n_validation_windows":3,"n_models":6,"trend_strength":0.8746,"seasonal_strength":0.0,"component_method":"linear_detrend_surrogate","trend_champion":{"model":"redundant_b","score":0.986,"metric":"slope_agreement","tied":false,"informative":true},"seasonality_champion":{"model":"good_a","score":0.994,"metric":"detrended_residual_corr","tied":false,"informative":true},"stationarity":{"verdict":"statsmodels_unavailable","reliable":true},"outliers_pct":0.0,"features":{"acf1":0.9783,"acf_seasonal":0.9942,"spectral_entropy":0.3752,"coef_variation":0.1747,"crosses_zero":false},"catch22":"pycatch22 unavailable"}
</series_profile>

<model_pool>
MODEL POOL:
{"n_models":6,"n_windows":3,"horizon":12,"error_table":{"metric":"rmse","window":"all","top":[{"model":"good_a","error":0.9998,"rank":1},{"model":"good_b","error":1.0329,"rank":2},{"model":"redundant_b","error":2.6903,"rank":3},{"model":"redundant_a","error":2.6951,"rank":4},{"model":"mediocre","error":3.5166,"rank":5},{"model":"bad","error":13.8767,"rank":6}],"rest":null,"relative_spread":12.88},"best_model_per_window":[{"window":0,"best_model":"good_a"},{"window":1,"best_model":"good_b"},{"window":2,"best_model":"good_a"}],"ranking_stability":{"mean_kendall_tau":0.822,"verdict":"stable","always_top3":["good_a","good_b"]},"error_correlation":{"mean_corr":0.224,"n_groups":5,"n_independent":4,"redundant_groups":[{"models":["redundant_a","redundant_b"],"representative":"redundant_a"}]}}
</model_pool>

<diagnosis>
DIAGNOSIS (sample from a real run) - a reading, not a rule:
{"regime":"trend_dominated","predictability":"high","combination_hint":"weighted","risks":["several models carry correlated error","only 3 validation windows"],"narrative":"Strong trend, little seasonal structure, ranking moderately unstable across windows."}
</diagnosis>

<attempt_history>
ATTEMPT HISTORY (0), best first - lower score is better:
  (empty)
</attempt_history>

<handles>
HANDLES AVAILABLE (seeded by Phase 2, plus anything you create):
{"pools":{"pool1":{"k":5,"models":["good_a","good_b","redundant_a","redundant_b","mediocre"]}}}
  no weight handles exist yet - create one with a weights_* tool before any combine='weighted' strategy
</handles>

<weight_methods>
WEIGHT METHODS available for evaluate_strategy['weights'] (optional; defaults):
  inverse_error, softmax_neg_error, error_trend, feature_based, meta_model, best_per_horizon, topk_per_horizon, forward
</weight_methods>

<untried>
NEXT UNTRIED COMBINATIONS (not in the history; choose ONE):
  A. {"combine": "mean", "pool": "pool_full"}
  B. {"combine": "median", "pool": "pool_full"}
  C. {"combine": "trimmed_mean", "pool": "pool_full", "trim_pct": 0.2}
  D. {"combine": "dba", "pool": "pool_full", "dba_max_iter": 30}
  E. {"combine": "weighted", "pool": "pool_full", "weights": "inverse_error"}
  F. {"combine": "best_single", "model": "good_a"}
  G. {"combine": "mean", "pool": "pool1"}
  H. {"combine": "median", "pool": "pool1"}
</untried>

<budget>
SCORES: 0/12 new evaluations used. TURNS: 0/24 used, 24 left.
24 turns left. Test another combination; do not call accept. The final strategy is chosen from the whole history.
</budget>
```

## Turno 2 — depois de um `evaluate_strategy` de exemplo

O attempt usado foi `mean` sobre `good_a, good_b`, para mostrar leader/histórico/observação.

```xml
ITERATION 2 of 24.

<series_profile>
SERIES PROFILE:
{"n_points":228,"frequency":"monthly","seasonal_period":12,"seasonal_period_source":"frequency","horizon":12,"n_validation_windows":3,"n_models":6,"trend_strength":0.8746,"seasonal_strength":0.0,"component_method":"linear_detrend_surrogate","trend_champion":{"model":"redundant_b","score":0.986,"metric":"slope_agreement","tied":false,"informative":true},"seasonality_champion":{"model":"good_a","score":0.994,"metric":"detrended_residual_corr","tied":false,"informative":true},"stationarity":{"verdict":"statsmodels_unavailable","reliable":true},"outliers_pct":0.0,"features":{"acf1":0.9783,"acf_seasonal":0.9942,"spectral_entropy":0.3752,"coef_variation":0.1747,"crosses_zero":false},"catch22":"pycatch22 unavailable"}
</series_profile>

<model_pool>
MODEL POOL:
{"n_models":6,"n_windows":3,"horizon":12,"error_table":{"metric":"rmse","window":"all","top":[{"model":"good_a","error":0.9998,"rank":1},{"model":"good_b","error":1.0329,"rank":2},{"model":"redundant_b","error":2.6903,"rank":3},{"model":"redundant_a","error":2.6951,"rank":4},{"model":"mediocre","error":3.5166,"rank":5},{"model":"bad","error":13.8767,"rank":6}],"rest":null,"relative_spread":12.88},"best_model_per_window":[{"window":0,"best_model":"good_a"},{"window":1,"best_model":"good_b"},{"window":2,"best_model":"good_a"}],"ranking_stability":{"mean_kendall_tau":0.822,"verdict":"stable","always_top3":["good_a","good_b"]},"error_correlation":{"mean_corr":0.224,"n_groups":5,"n_independent":4,"redundant_groups":[{"models":["redundant_a","redundant_b"],"representative":"redundant_a"}]}}
</model_pool>

<diagnosis>
DIAGNOSIS (sample from a real run) - a reading, not a rule:
{"regime":"trend_dominated","predictability":"high","combination_hint":"weighted","risks":["several models carry correlated error","only 3 validation windows"],"narrative":"Strong trend, little seasonal structure, ranking moderately unstable across windows."}
</diagnosis>

<attempt_history>
ATTEMPT HISTORY (1), best first - lower score is better:
  {"id":"a1","strategy":"mean pool=pool2","n_models":2,"score":0.2385,"rmse":0.7156,"smape":0.0034,"pocid":97.0,"rmse_per_window":[0.9272,0.5546,0.6076],"origin":"agent"}
</attempt_history>

<leader_diagnostics>
LEADER DIAGNOSTICS (validation only):
{"leader_id":"a1","leader_strategy":"mean on pool2","leader_origin":"agent","leader_score":0.2385,"runner_up_margin":null,"verdict":"no_comparison","note":"accept is disabled in this run; the final history will take the best validation score. Use the remaining turns only if a new combination tests a real hypothesis."}
</leader_diagnostics>

<handles>
HANDLES AVAILABLE (seeded by Phase 2, plus anything you create):
{"pools":{"pool1":{"k":5,"models":["good_a","good_b","redundant_a","redundant_b","mediocre"]},"pool2":{"k":2,"models":["good_a","good_b"]}}}
  no weight handles exist yet - create one with a weights_* tool before any combine='weighted' strategy
</handles>

<weight_methods>
WEIGHT METHODS available for evaluate_strategy['weights'] (optional; defaults):
  inverse_error, softmax_neg_error, error_trend, feature_based, meta_model, best_per_horizon, topk_per_horizon, forward
</weight_methods>

<untried>
NEXT UNTRIED COMBINATIONS (not in the history; choose ONE):
  A. {"combine": "mean", "pool": "pool_full"}
  B. {"combine": "median", "pool": "pool_full"}
  C. {"combine": "trimmed_mean", "pool": "pool_full", "trim_pct": 0.2}
  D. {"combine": "dba", "pool": "pool_full", "dba_max_iter": 30}
  E. {"combine": "weighted", "pool": "pool_full", "weights": "inverse_error"}
  F. {"combine": "best_single", "model": "good_a"}
  G. {"combine": "mean", "pool": "pool1"}
  H. {"combine": "median", "pool": "pool1"}
</untried>

<scratchpad>
WHAT YOU DID SO FAR:
  [1] evaluate_strategy({"combine":"mean","models":["good_a","good_b"]}) -> rank 1/1 score=0.2385 rmse=0.7156 (best so far)
</scratchpad>

<last_observation>
LAST OBSERVATION (full):
{"id":"a1","strategy":{"combine":"mean","pool":"pool2"},"already_tested":false,"rank":1,"total_attempts":1,"metrics":{"rmse":0.7156,"smape":0.0034,"mape":0.0033,"pocid":97.0},"rmse_per_window":[0.9272,0.5546,0.6076],"score":0.2385,"current_best":{"id":"a1","score":0.2385,"strategy":"mean pool=pool2","origin":"agent"},"worse_than_best_by":null,"is_best":true}
</last_observation>

<budget>
SCORES: 1/12 new evaluations used. TURNS: 1/24 used, 23 left.
23 turns left. Test another combination; do not call accept. The final strategy is chosen from the whole history.
</budget>
```

## Auditoria dos ifs do prompt

| Branch | Estado no v4 | O que acontece |
|---|---|---|
| `prompt_v2` | True | usa `build_system_prompt_v2` |
| `no_early_stop` | True | remove accept, regras de aceite e “WHEN TO STOP” |
| `direct_models` | True | catálogo só `evaluate_strategy`; exemplos diretos |
| `multi_preset_final` | False | sem `SCORE RANKINGS`, sem chamada final de preset |
| `no_seed_attempts` | True | bloco `CLASSIC STARTING POINTS`, sem tentativas de seed |
| `allow_agent_score_preset` | False | `score_preset` some do schema e o critério fica fixo |
| `score_preset` | `smape_only` | linha “FIXED to smape_only (normalized SMAPE)” |
| `withheld_tools` | só `evaluate_strategy` | os outros tools não aparecem no catálogo |
| `no_accept` | True | budget no modo “do not call accept” |
| `score_target`/`scores_used` | 12/0 e 12/1 | linha `SCORES` no turno |
| `prompt_format` | XML | seções com tags |
| `include_history_rules` | True | regra de histórico presente |

## O que está coerente

- Não há `score_preset` no catálogo do `evaluate_strategy`; o texto diz que é fixo e o tool rejeita.
- Não há accept no catálogo/típico; o líder diz que accept está desabilitado.
- O turno mostra `SCORES: 0/12` e `TURNS: 0/24` (e 1/12, 1/24 depois).
- O modo sem seed mostra as clássicas como **opcionais**, sem ordem obrigatória.
- As stable pools/weighted handles são registradas como **preparação**, não como tentativas.
- O menu `NEXT UNTRIED` inclui mean/median/trimmed/dba/weighted no pool full e mean/median no pool estável.
- Não há mapa fixo “trend → modelo”; o perfil e o diagnóstico são informação, não regra.
- POCID não entra no score; o texto diz isso.

## O que ainda ficou esquecido / vale melhorar

> Atualização: o item 1 foi implementado; os prompts abaixo já contêm o bloco `WEIGHT METHODS`.

1. **Weight methods: implementado.**
   - O turno agora tem `WEIGHT METHODS available for evaluate_strategy['weights']`.
   - É state-aware: `ols` só aparece com janelas suficientes; `pooled_meta_model` só se o modelo pooled existir.
   - É opcional e com defaults; não força uso.

2. **O `score_window_mode` não aparece no texto.**
   - O score é weighted 0.2/0.3/0.5, mas o prompt não diz isso.
   - O agente vê `score` e `smape`, mas não sabe a ponderação das janelas.
   - Sugestão: uma linha “validation score: weighted 0.2/0.3/0.5”.

3. **O menu `NEXT UNTRIED` é limitado a 8.**
   - Full pool ocupa 6; sobram 2 para o primeiro pool estável.
   - As stable pools com trimmed/weighted não aparecem no menu; estão nos HANDLES.
   - Sugestão: aumentar o limite ou separar “full” e “stable” no menu.

4. **Os HANDLES só mostram `w1` se o pooled meta-model estiver disponível.**
   - No fixture não há w1; em run real, se o pre-pass rodar, aparece.
   - O prompt já avisa “stable-pool and weighted handles are also available in HANDLES”.

5. **O prompt v1 ainda existe e não é usado no v4.**
   - Ele já foi corrigido para não mostrar accept/direct quebrado, mas o caminho oficial é o v2. Vale marcar como legado nos docs.

## Resumo

O caminho ativo do v4 está coerente e sem contradições graves. As pendências são de **explicitação**, não de lógica: listar os weight methods disponíveis, dizer a ponderação das janelas e melhorar o menu de stable pools. Nenhuma delas força uma abordagem; são informações opcionais.
