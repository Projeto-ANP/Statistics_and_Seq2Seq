# Arquitetura de combinação de previsões — especificação de funcionamento

Descrição completa e mecânica do sistema: o que entra, o que cada etapa faz, o
catálogo inteiro de ferramentas, os algoritmos, e o que sai. Escrito no nível de
detalhe necessário para reimplementar.

---

## 1. Visão geral

O sistema recebe as previsões já produzidas por *N* modelos individuais para uma
série temporal e decide **como combiná-las** numa única previsão para a janela de
teste. Não treina modelos de previsão — consome previsões prontas.

```
ENTRADA                      PROCESSAMENTO                        SAÍDA
─────────────────────────────────────────────────────────────────────────────────
CSV de N modelos       ┌─ pré-passo por dataset ─┐         CSV de 58 colunas
(previsões + reais)    │  meta-modelo LOSO       │         (1 linha por série)
                       │  prior de estratégias   │
arquivo .tsf           └─────────┬───────────────┘         artifact JSON
(série histórica)                │                         (1 por série)
                       ┌─ por série ─────────────┐
                       │  Fase 0  ingestão       │
                       │  Fase 1a perfil         │
                       │  Fase 2  pool+sementes  │
                       │  Fase 1b diagnóstico    │
                       │  Fase 3  loop ReAct     │
                       │  Fase 4  aplicação      │
                       │  Fase 5  relato         │
                       └─────────────────────────┘
```

O componente de decisão é um agente ReAct (ciclo *Thought → Action →
Observation*) operando sobre um catálogo fechado de **24 ferramentas
determinísticas**. O agente emite apenas texto — nome de ferramenta e argumentos
JSON. Toda aritmética acontece no código; o agente recebe de volta um
identificador (*handle*) e um resumo qualitativo, nunca valores numéricos brutos
de peso ou previsão.

---

## 2. Entrada

### 2.1 Previsões dos modelos individuais

Um CSV por modelo, separador `;`:

```
<results_dir>/<MODELO>/normal/<DATASET>.csv
```

Colunas obrigatórias:

| coluna | conteúdo |
|---|---|
| `dataset_index` | inteiro: identificador da série dentro do dataset |
| `horizon` | inteiro: tamanho do horizonte de previsão |
| `test` | string com a lista dos valores reais da janela, ex. `"[12.3, 14.1, ...]"` |
| `predictions` | string com a lista das previsões daquela janela |
| `start_test`, `final_test` | timestamps da janela |
| `regressor`, `mape`, `pocid`, `smape`, `rmse`, `msmape`, `mae` | métricas do modelo individual (não usadas na decisão) |

Cada série (`dataset_index`) tem **W+1 linhas**, ordenadas por `start_test`. As
`W` primeiras são janelas de **validação**; a última é a janela de **teste cega**.
Default: `W = 3`, logo 4 linhas por série.

O parsing de `test`/`predictions` usa a regex
`[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?` — as células são `repr` de arrays numpy e
podem conter quebras de linha e notação científica.

### 2.2 Série histórica

Arquivo Monash `.tsf` em `<source_dir>` (default `../forecasting_datasets`). O
parser lê `@attribute`, `@frequency`, `@horizon`, `@relation` e as linhas de
dados após `@data`. `dataset_index` é o índice **posicional** da linha no arquivo
(possivelmente após filtragem, ver §4.1).

`@frequency` é a fonte autoritativa do período sazonal:

| `@frequency` | período |
|---|---|
| `yearly` | 1 |
| `quarterly` | 4 |
| `monthly` | 12 |
| `weekly` | 52 |
| `daily` | 7 |
| `hourly` | 24 |
| `half_hourly` / `30min` | 48 |
| `15min` | 96 |
| `10_minutes` | 144 |
| `minutely` | 60 |

Frequência desconhecida → fallback `max(2, min(horizon, 12))`.

### 2.3 O pool de modelos

Default: 19 modelos.

| família | modelos |
|---|---|
| estatísticos | `ARIMA`, `ETS`, `THETA` |
| ML direto | `rf`, `catboost` |
| ML + transformada concatenada | `CWT_rf`, `DWT_rf`, `FT_rf`, `CWT_catboost`, `DWT_catboost`, `FT_catboost` |
| ML só sobre a transformada | `ONLY_CWT_rf`, `ONLY_CWT_catboost`, `ONLY_DWT_rf`, `ONLY_DWT_catboost`, `ONLY_FT_rf`, `ONLY_FT_catboost` |
| ingênuos | `NaiveSeasonal`, `NaiveMovingAverage` |

`CWT` = wavelet contínua, `DWT` = wavelet discreta, `FT` = Fourier.

---

## 3. Estruturas de dados

### 3.1 `ReactState` — o estado de uma série

Detém todos os dados numéricos. **Nada daqui entra em prompt.**

| campo | forma | conteúdo |
|---|---|---|
| `y_true` | `(W, H)` | valores reais das janelas de validação |
| `y_preds` | `(W, M, H)` | previsões de cada modelo nas janelas de validação |
| `test_preds` | `(M, H)` | previsões de cada modelo na janela de teste (**sem** os reais) |
| `model_names` | lista de `M` strings | nomes, na ordem do eixo de modelos |
| `train_series` | vetor 1-D | série histórica sem o período de teste |
| `freq` | string | `@frequency` do `.tsf` |
| `pools` | dict | `handle -> lista de índices de modelo` |
| `pool_recipes` | dict | `handle -> PoolRecipe` |
| `weights` | dict | `handle -> WeightsRecipe` |
| `attempts` | lista | histórico de `Attempt` |
| `pooled_meta_model` | objeto ou `None` | modelo cross-series desta série (§4.2) |
| `strategy_prior` | dict ou `None` | prior de dataset desta série (§4.3) |
| `tools_called`, `tool_errors` | listas | traço de execução |

`W` = janelas de validação, `H` = horizonte, `M` = número de modelos.

Os valores reais da janela de teste **não existem** nesta estrutura. Só entram no
cálculo das métricas finais, depois da Fase 4.

### 3.2 Handles

Três espaços de nomes, todos reiniciados a cada série:

| tipo | formato | criado por |
|---|---|---|
| pool | `pool_full`, `pool1`, `pool2`, … | `select_top_k`, `select_stable`, `prune_redundant` |
| pesos | `w1`, `w2`, … | qualquer `weights_*` |
| tentativa | `a1`, `a2`, … | `evaluate_strategy` |

`pool_full` sempre existe e contém todos os `M` modelos.

**Deduplicação.** Registrar um pool cuja composição resolve igual em todos os
folds a um pool já existente devolve o handle existente. O mesmo para pesos: a
receita é canonicalizada em JSON e receitas idênticas compartilham handle. Isso
impede que duas estratégias numericamente idênticas apareçam como entradas
distintas no histórico.

### 3.3 `PoolRecipe` — como um pool foi escolhido

Armazena a **receita**, não os índices:

```python
PoolRecipe(
    method: str,               # "top_k" | "stable" | "prune_redundant"
    params: dict,              # {"k": 5, "metric": "rmse"} etc.
    base: tuple[int,...]|None, # pool de origem, para receitas que filtram outro
    resolved: tuple[int,...],  # composição ajustada em todas as janelas
)
```

`method in ("top_k", "stable", "prune_redundant")` é re-ajustável por fold. Uma
lista explícita de modelos nomeada pelo agente não é.

### 3.4 `WeightsRecipe` — como pesos são obtidos

```python
WeightsRecipe(
    method: str,               # um de WEIGHT_METHODS
    pool_handle: str,
    fit_windows: tuple|None,   # None = todas as disponíveis
    per_horizon: bool,
    params: dict,
    resolved: np.ndarray,      # (n_pool,) ou (n_pool, H)
    meta: dict,                # {"mode": ..., "fit_windows": [...]}
)
```

Também guarda a receita, não os números: durante o backtest ela é reajustada por
janela sob o protocolo anti-vazamento.

### 3.5 `Attempt` — uma estratégia avaliada

```python
Attempt(
    attempt_id: str,           # "a1", "a2", ...
    spec: dict,                # a estratégia normalizada
    origin: str,               # "baseline" | "agent"
    aggregate: dict,           # MAPE/SMAPE/RMSE/POCID/MSMAPE/MAE sobre toda a validação
    per_window: list[dict],    # as mesmas métricas por janela
    score: float,              # score composto (§7.4) — menor é melhor
    per_window_scores: list,   # score composto por janela (amostra pareada)
    residuals: list,           # (previsão - real) achatado, para Diebold-Mariano
    rationale: str,            # justificativa que o agente escreveu
    iteration: int|None,
    n_models: int,
    agent_converged: bool,     # o agente propôs algo que já era semente
    agent_rationale: str,
)
```

### 3.6 Especificação de estratégia (`spec`)

O objeto que descreve uma combinação. Normalizado e validado por
`normalize_spec`:

```python
{"combine": "mean",         "pool": "pool_full"}
{"combine": "median",       "pool": "pool1"}
{"combine": "trimmed_mean", "pool": "pool1", "trim_pct": 0.2}   # 0 <= trim < 0.5
{"combine": "weighted",     "pool": "pool1", "weights": "w1"}
{"combine": "dba",          "pool": "pool1", "dba_max_iter": 30}
{"combine": "best_single",  "model": "ETS"}                     # pool forçado p/ pool_full
```

Regras de validação:
- `combine` tem que estar em `("mean","median","trimmed_mean","weighted","dba","best_single")`;
- `weighted` exige `weights`, e o handle de pesos precisa ter sido calculado
  **sobre o mesmo pool** da estratégia;
- `best_single` exige `model`, validado contra `model_names`.

---

## 4. Pré-passo — roda uma vez por dataset, antes de qualquer série

### 4.1 Carregamento e alinhamento do `.tsf`

O número de linhas do `.tsf` precisa bater com o número de séries nos CSVs de
resultado. Quando não bate, filtros conhecidos são tentados em ordem:

| filtro | regra |
|---|---|
| `drop_zero_windows_24` | descarta a série se **alguma** janela de 24 pontos tiver >50% de zeros |
| `drop_zero_windows_12` | idem, janelas de 12 |

Se nenhum reconcilia as contagens, levanta `SeriesAlignmentError` — nunca
adivinha.

### 4.2 Meta-modelo cross-series

Só roda se `pooled_meta_model=True` **e** o número de séries da chamada
`>= pooled_meta_model_min_series` (default 20) **e** `xgboost` estiver
disponível. Caso contrário devolve `{}` e a ferramenta
`weights_pooled_meta_model` é retirada do catálogo para todas as séries.

**Passo 1 — montar uma linha por série.** Para cada série: roda a Fase 0
(ingestão) e `series_profile`, extrai **26 características** e o erro de validação
de cada modelo.

As 26 características, nesta ordem fixa:

| # | origem | nomes |
|---|---|---|
| 1–4 | `series_profile` | `trend_strength`, `seasonal_strength`, `spectral_entropy`, `acf1` |
| 5–26 | catch22 | `DN_HistogramMode_5`, `DN_HistogramMode_10`, `CO_f1ecac`, `CO_FirstMin_ac`, `CO_HistogramAMI_even_2_5`, `CO_trev_1_num`, `MD_hrv_classic_pnn40`, `SB_BinaryStats_mean_longstretch1`, `SB_TransitionMatrix_3ac_sumdiagcov`, `PD_PeriodicityWang_th0_01`, `CO_Embed2_Dist_tau_d_expfit_meandiff`, `IN_AutoMutualInfoStats_40_gaussian_fmmi`, `FC_LocalSimple_mean1_tauresrat`, `DN_OutlierInclude_p_001_mdrmd`, `DN_OutlierInclude_n_001_mdrmd`, `SP_Summaries_welch_rect_area_5_1`, `SB_BinaryStats_diff_longstretch0`, `SB_MotifThree_quantile_hh`, `SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1`, `SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1`, `SP_Summaries_welch_rect_centroid`, `FC_LocalSimple_mean3_stderr` |

Indexadas **por nome**, não por posição de iteração. Valores ausentes ou não
finitos viram `0.0`. Sem `pycatch22` instalado, os 22 slots ficam zerados e o
modelo degrada para as 4 básicas.

O erro por modelo é **sMAPE médio** nas janelas de validação (não RMSE: o
gradiente soma contribuições entre séries, e erro em escala bruta faria séries de
maior magnitude dominarem).

**Passo 2 — treinar, leave-one-series-out.** Para cada série *i*, treina-se um
modelo usando todas as séries **exceto** *i*.

Dois objetivos possíveis (`pooled_meta_model_objective`):

**`fforma`** (default) — um único booster multiclasse XGBoost, 100 rodadas,
`num_class = M`. O softmax da saída **é** o vetor de pesos. Gradiente e hessiana
customizados:

```
p       = softmax(margens)                 # (n_series, M)
weighted = Σ_m p[:,m] · contrib[y, m]      # erro da combinação
grad    = p · (contrib[y] - weighted)
hess    = contrib[y] · p · (1-p) - grad · p
```

`contrib` é a matriz `(n_series, M)` de sMAPE de validação. Contribuições não
finitas viram `2 × (pior erro finito da linha)`; uma linha inteiramente não finita
vira `1.0`. As contribuições entram **cruas**, sem normalização.

Depois do treino, verifica-se se todas as margens saíram idênticas
(`np.allclose(margens, margens[:, :1])`); se sim, o modelo é marcado
`degenerate=True` — ele não aprendeu nada e seus pesos são uniformes.

**Fallback automático:** se *todos* os folds saírem degenerados, o conjunto
inteiro é refeito com o objetivo `per_model`.

**`per_model`** — `M` regressores XGBoost independentes (40 árvores,
profundidade 2), cada um prevendo o erro do seu próprio modelo. Um modelo com
menos de 5 alvos finitos não recebe regressor (predição `None`).

**Passo 3 — consulta.** `predict_scores(features)` devolve
`(dict {nome_modelo: score}, kind)`:

| `kind` | significado | conversão em pesos |
|---|---|---|
| `"margin"` | margens brutas do booster fforma | `softmax(+margens)` sobre o subconjunto |
| `"error"` | erro previsto por modelo | `softmax(-eta · erro / mediana)` |

Chavear por **nome** (não por posição) é o que torna isso seguro sob composição de
pool variável por fold: o softmax de um subconjunto de margens é igual ao softmax
completo renormalizado àquele subconjunto.

### 4.3 Prior de estratégias por dataset

Só roda se `dataset_card=True` ou `final_strategy="prior_blend"`.

Para cada série do dataset: roda Fase 0 + Fase 2 (sem agente) e registra o score
de validação de cada estratégia semeada. Para a série *i*, o prior é a **média
desse score nas outras N−1 séries**:

```python
prior[i][label] = mean(score[j][label] for j != i)
```

Nada aqui lê a janela de teste.

### 4.4 Detecção de modelos desalinhados

Só se `drop_misaligned_models=True`. Compara a coluna `test` de cada modelo
contra um modelo de referência (o primeiro da lista), na janela de teste e nas
`W` de validação, com `rtol=1e-4, atol=1e-4`. Um modelo cujos valores reais
divergem está prevendo outro trecho da série e é removido do pool para **todas** as
séries do dataset.

Detectado uma vez, na primeira série — um modelo gerado sobre a janela errada está
errado para o dataset inteiro.

---

## 5. Fases por série

### 5.1 Fase 0 — ingestão

1. Remove os modelos marcados em §4.4.
2. Lê as `W+1` linhas de cada modelo para este `dataset_index`, ordenadas por
   `start_test`. Erro se algum modelo tiver menos que `W+1` linhas.
3. **Horizonte** `H` = menor comprimento comum entre todas as listas
   `predictions`/`test` de todos os modelos e janelas. Comprimentos divergentes
   geram aviso e truncamento.
4. Monta `y_true (W,H)`, `y_preds (W,M,H)`, `test_preds (M,H)`.
5. **Gate de valores.** Para cada janela, os valores reais de cada modelo são
   comparados com os do modelo de referência (`rtol=1e-4`). Divergência →
   `IngestionError`. Divergência apenas de *timestamp*, com valores idênticos, é
   apenas aviso (há arquivos com bug de rotulagem de frequência).
6. Carrega a série histórica do `.tsf` e chama `verify_alignment`: a cauda de `H`
   pontos da série tem que bater com a coluna `test`. Se não bater, tenta
   reamostragem por média em blocos de `(2, 3, 4, 6, 12)` antes de desistir com
   `SeriesAlignmentError`. O fator usado é registrado.
7. `train_series = série[: len(série) - H]`.

### 5.2 Fase 1a — perfil da série (sempre roda)

`series_profile(state)` devolve:

| campo | conteúdo |
|---|---|
| `source` | `"train_series"` ou `"validation_windows"` |
| `decomposition` | `"stl"` ou `"linear_fallback:<motivo>"` |
| `n_points`, `frequency`, `horizon`, `n_validation_windows`, `n_models` | dimensões |
| `seasonal_period`, `seasonal_period_declared`, `seasonal_period_source`, `seasonal_period_fits` | período sazonal e sua procedência |
| `trend_strength`, `seasonal_strength` | força de tendência/sazonalidade em `[0,1]` |
| `stationarity` | `{n, adf_pvalue, kpss_pvalue, verdict, reliable}` |
| `outliers` | `{n_outliers, pct, max_z}` — cerca IQR com `k=3` |
| `features` | `acf1`, `acf_seasonal`, `acf_diff1`, `spectral_entropy`, `hurst`, `skewness`, `kurtosis`, `coef_variation`, `fluctuation_scale`, `crosses_zero` |
| `catch22` | dict de 22 valores, ou a string `"pycatch22 unavailable"` |
| `trend_champion`, `seasonality_champion` | `{model, score, metric, tied, informative}` |
| `component_method`, `component_layout`, `component_n_points` | como os campeões foram computados |
| `mean_trend_score`, `mean_seasonality_score` | médias dos scores |

**Decomposição.** STL robusta (`statsmodels`) quando há pelo menos `2 × período`
pontos válidos e no mínimo 4; caso contrário, detrend linear com componente
sazonal identicamente zero — registrado em `decomposition`.

**Forças.** Para cada componente `C` com resíduo `R`:

```
strength(C) = clip(1 - Var(R) / Var(C + R), 0, 1)
```

**Campeões de componente.** Determinístico, sem LLM. As janelas de validação são
concatenadas (verificado que são contíguas comparando a concatenação com a cauda
de `train_series`), gerando `W × H` pontos. Dois regimes:

- período cabe na amostra (`n >= 2 × período`): decompõe STL a série real e a de
  cada modelo; o campeão é o `argmax` da correlação de Pearson entre as
  componentes correspondentes;
- não cabe: substitutos livres de escala — concordância de inclinação
  `1 / (1 + |slope_m − slope_real| / escala)` para tendência, e correlação dos
  resíduos destendenciados para o formato sazonal.

`informative=False` marca o caso em que todos os modelos obtêm o mesmo score, ou
seja, o sinal não carrega informação.

### 5.3 Fase 2 — pool de trabalho, card do pool e sementes

**Pool de trabalho** (`pool_mode`):

| modo | efeito |
|---|---|
| `full` (default) | `pool_full`, todos os `M` modelos |
| `top_k_error` | `select_top_k(k=pool_k)` |
| `top_k_stable` | `select_stable(k=pool_k)` |

**Card do pool** — injetado em todo turno do agente:

```python
{
  "n_models", "n_windows", "horizon",
  "error_table":        error_summary(top_n=8),
  "ranking_stability":  ranking_stability(),
  "error_correlation":  error_correlation(threshold=0.9),
  "best_model_per_window": [{"window": w, "best_model": nome}, ...]
}
```

**Sementes.** Avaliadas e inseridas no histórico com `origin="baseline"`, antes do
loop abrir:

1. `mean`, `median`, `dba` sobre o pool de trabalho — sempre;
2. se `seed_stable_pools=True`: `select_stable(k)` para `k ∈ {5,7,9}`, cada um com
   `mean` e `trimmed_mean` → 6 sementes. `k >= M` é pulado (seria o pool completo);
3. se `seed_pooled_meta_model=True` e o meta-modelo existe:
   `weighted(pool_trabalho, weights_pooled_meta_model)`. Falha aqui degrada para
   "sem semente", nunca mata a série.

**Consequência estrutural:** a estratégia final aplicada é sempre a de menor score
de todo o histórico. Como as sementes já estão no histórico antes de o agente
falar, o resultado nunca é pior que a melhor semente.

**Portão de calibração** (`calibration_gate`, default `False`): se o Kendall tau
médio entre os rankings por janela `>= calibration_gate_kendall` (0.85), o loop da
Fase 3 é pulado inteiramente.

### 5.4 Fase 1b — diagnóstico (opcional)

Lê o perfil da série e o card do pool, devolve uma leitura estruturada em
vocabulário fechado:

```python
{
  "regime":            "trend_dominated"|"seasonal_dominated"|"noisy"|"mixed",
  "predictability":    "high"|"medium"|"low",
  "combination_hint":  "robust"|"weighted"|"selective"|"full_pool",
  "risks":             [str, ...],   # no máximo 3
  "narrative":         str,          # 2-3 frases
  "source":            "llm"|"deterministic",
}
```

Sem LLM configurado (`diagnostician.model=None`, o default), a mesma estrutura é
preenchida por regras:

| campo | regra |
|---|---|
| `regime` | `trend>=0.6 e seasonal>=0.6` → mixed; `seasonal>=0.6` → seasonal_dominated; `trend>=0.6` → trend_dominated; senão noisy |
| `predictability` | `spectral_entropy>=0.85` → low; `<=0.5` → high; senão medium |
| `combination_hint` | `tau<0.3` → robust; `spread>=0.5 e tau>=0.5` → selective; `spread>=0.3` → weighted; senão full_pool |

Com LLM, a saída é validada: valores fora do vocabulário caem para o valor
determinístico, e todo token que pareça nome de modelo (maiúsculo ou com `_`) é
conferido contra `model_names` — nomes inexistentes são registrados em
`validation_notes` e não são propagados.

Este bloco **não decide nada**; é texto injetado no prompt.

### 5.5 Fase 3 — o loop ReAct

**Contrato de saída do agente.** Exatamente três linhas:

```
Thought: <uma ou duas frases>
Action: <nome de ferramenta do catálogo>
Action Input: {"arg": valor}
```

O parser aceita variações: JSON único contendo `action`/`tool` e
`action_input`/`args`/`input`; bloco cercado por ``` ```; ausência do rótulo
`Action Input`; e blocos `<think>...</think>` (extraídos para o campo `thought`).

**Conteúdo do prompt de turno**, nesta ordem:

1. `ITERATION i of max_iterations`
2. perfil da série (versão reduzida — catch22 vira só o marcador `"computed"`)
3. card do pool (versão reduzida)
4. card do dataset, se `dataset_card=True` e o prior existe
5. diagnóstico, se houver
6. histórico de tentativas ranqueado, melhor primeiro, top 10
7. handles disponíveis (pools e pesos, com concentração e se já foram pontuados)
8. o que foi feito nos últimos 6 turnos
9. a última observação completa
10. aviso de pesos calculados e nunca pontuados, se restarem ≤3 iterações
11. iterações restantes

**Ciclo por iteração:**

```
1. monta o prompt do turno
2. chama o LLM
   - resposta vazia    -> reexecuta, até 2 vezes, sem gastar iteração
   - erro de transporte -> reexecuta, até 4 vezes, sem gastar iteração
3. faz o parse
   - falha de parse -> vira observação de erro; o agente pode corrigir; gasta iteração
4. se Action == "accept": valida e encerra
5. senão despacha a ferramenta pelo registry
   - erro -> observação estruturada, gasta iteração
6. se a ferramenta foi evaluate_strategy: atualiza contador de parada antecipada
```

**Despacho e validação de argumentos** (`registry.call_tool`):

| situação | resultado |
|---|---|
| `Action Input` não é JSON válido | `{"error": "invalid_action_input"}` |
| nome fora do catálogo, ou ferramenta retida | `{"error": "unknown_tool", "available": [...]}` |
| argumento desconhecido | `{"error": "unknown_argument", "accepted": [...]}` — exceto em `evaluate_strategy`, onde é descartado e reportado em `ignored_args` |
| argumento obrigatório ausente | `{"error": "missing_required_argument"}` |
| `ValueError`/`KeyError` da ferramenta | `{"error": "invalid_argument", "detail": msg}` |
| qualquer outra exceção | `{"error": "internal_failure"}` |

Os tipos `unknown_tool`, `unknown_argument`, `missing_required_argument` e
`invalid_action_input` ligam a coluna `tool_missing` do CSV.

**Retenção de ferramentas.** Antes de o prompt de sistema ser montado, ferramentas
que não podem dar resposta confiável são removidas do catálogo — o agente nunca as
vê:

| ferramenta | condição de retenção |
|---|---|
| `weights_ols` | `W < min_windows_for_ols` (default 5; com `W=3` é sempre retida) |
| `weights_pooled_meta_model` | `state.pooled_meta_model is None` |

**Ação terminal.** `accept` não é uma ferramenta; é tratado pelo loop:

```
Action: accept
Action Input: {"attempt_id": "a14", "confidence": 0.9, "justification": "..."}
```

`attempt_id` ausente significa "a melhor". Um id inexistente devolve a lista dos
conhecidos e não encerra o loop. `confidence` é recortada em `[0,1]`.

**Parada.** O loop encerra por: `accept` válido; `early_stop_patience` (4)
propostas consecutivas sem melhorar o melhor score em pelo menos
`min_improvement` (1e-4 relativo); esgotamento de `max_iterations` (12); ou erro
de LLM após todas as retentativas.

**Garantia de resultado.** Ao final, `final_attempt = best_attempt()` — a de menor
score de todo o histórico. Se o agente aceitou outra, a escolha dele fica em
`agent_accepted_id`, `overridden=True` é registrado, e a melhor é a aplicada.

### 5.6 Fase 4 — aplicação à janela de teste

Três modos (`final_strategy`):

**`argmin`** (default) — aplica `best_attempt().spec` via `apply_to_test`.

**`ensemble`** — média ponderada das `final_top_m` (3) melhores:

```
scores = [score de cada tentativa escolhida]
z = -final_eta · (scores / mediana(scores))       # final_eta = 5.0
w = softmax(z)
previsão = Σ w_k · apply_to_test(spec_k)
```

**`prior_blend`** — reordena o histórico por score encolhido em direção ao prior
de dataset antes do argmin:

```
blended(a) = (1 - alpha) · score(a) + alpha · prior[label(a)]
```

com `alpha = final_prior_alpha` (default 0.0, o que reproduz `argmin` exatamente).
Sem prior para aquele rótulo, usa o score cru.

`apply_to_test` usa **a mesma função de combinação** do backtest. A única
diferença é que os pesos são ajustados em todas as janelas que a receita pedir,
já que não há janela alvo a excluir.

Depois: `sanity_check` compara a previsão contra os limites históricos e emite
avisos (não bloqueia nada), e os resultados das baselines externas
(`mean`, `median`, `dba`, `ADE`, `FFORMA`) são lidos do disco para a mesma série.

### 5.7 Fase 5 — relato (opcional)

Se `reporter.model` estiver configurado, uma chamada de LLM recebe o perfil, o
card do pool e a decisão, e escreve 3–5 frases de prosa. Falha ou ausência devolve
string vazia, e `justificativa_final` cai para a justificativa causal
determinística montada na Fase 3 a partir de tendência, sazonalidade, estabilidade
de ranking, independência de erro e dispersão de erro.

---

## 6. O catálogo — 24 ferramentas

Contrato comum: determinísticas; recebem `state` como primeiro argumento; nunca
escrevem fora dele; devolvem um dict compacto (ordem de 100–300 tokens), nunca
arrays brutos; levantam `ValueError`/`KeyError` com mensagem acionável.

### 6.1 Diagnóstico (6)

#### `series_profile()`
Ficha completa da série. Retorno em §5.2.

#### `stl_summary()`
Participação de cada componente na variância.

```python
{"period", "period_source", "frequency", "decomposition",
 "trend_pct", "seasonal_pct", "residual_pct",     # normalizados para somar 100
 "trend_strength", "seasonal_strength",
 "dominant_component"}                             # "trend"|"seasonality"|"residual"
```

#### `error_summary(window=None, top_n=8, metric='rmse')`
Tabela de erro por modelo, ranqueada. `window=None` agrega todas.

```python
{"metric": "rmse", "window": "all",
 "top":  [{"model": "THETA", "error": 2257.1152, "rank": 1}, ...],   # top_n entradas
 "rest": {"n_models": 17, "median_error": 3965.41, "worst_error": 5230.57},
 "relative_spread": 1.317}     # (max-min)/min sobre os erros finitos
```

#### `ranking_stability(metric='rmse')`
Concordância entre os rankings por janela.

```python
{"mean_kendall_tau": 0.228,          # média dos tau par-a-par entre janelas
 "verdict": "unstable",              # >=0.7 stable | >=0.3 moderate | senão unstable
 "biggest_movers": [{"model", "rank_spread", "ranks"}, ...],   # até 5
 "always_top3": [nomes que ficaram <=3 em TODAS as janelas]}
```

Com `W < 2` devolve `{"mean_kendall_tau": None, "reason": "fewer than 2 windows"}`.

#### `error_correlation(model_ids=None, threshold=0.9)`
Grupos de modelos cujos **erros** correlacionam acima do limiar. Correlaciona
resíduos, não previsões: dois modelos podem seguir a série de perto e ainda
falhar em janelas diferentes, que é exatamente a diversidade que a combinação
aproveita.

```python
{"threshold": 0.9, "n_models": 19,
 "mean_corr": 0.804,                  # média dos |corr| fora da diagonal
 "n_groups": 6,
 "redundant_groups": [{"models": [...], "representative": nome}, ...],  # até 6
 "n_independent": 3}                  # grupos de tamanho 1
```

Agrupamento guloso: parte do menor índice não atribuído e agrega todo modelo que
correlacione `>= threshold` com **todos** os já no grupo.

#### `dm_test(model_a, model_b, loss='squared')`
Diebold-Mariano entre dois modelos, com correção de amostra pequena
Harvey-Leybourne-Newbold.

```python
{"model_a", "model_b", "dm_stat", "p_value", "n_obs",
 "verdict": "statistical_tie" | "<a> better" | "<a> worse" | "undetermined"}
```

`p > 0.10` → empate estatístico.

### 6.2 Seleção de pool (3)

Todas registram uma `PoolRecipe` re-ajustável e devolvem um handle.

#### `select_top_k(k, metric='rmse', windows=None)`
Os `k` modelos de menor erro nas janelas dadas.

```python
{"pool": "pool1", "k": 5, "criterion": "lowest rmse",
 "models": [nomes...], "reused": False}
```

#### `select_stable(k, metric='rmse')`
Os `k` modelos mais consistentes: menor `média do rank + desvio do rank` entre
janelas. Um modelo 1º numa janela e 20º noutra perde para um que seja 4º em todas.
Com `W < 2` degenera para `select_top_k`.

```python
{"pool": "pool1", "k": 5, "criterion": "mean rank + std across windows",
 "models": [{"model": "ETS", "mean_rank": 3.0, "rank_std": 0.8}, ...],
 "reused": False}
```

#### `prune_redundant(pool='pool_full', corr_threshold=0.95, metric='rmse')`
Remove modelos redundantes, mantendo o de menor erro em cada grupo correlacionado.

```python
{"pool": "pool4", "base": "pool_full", "corr_threshold": 0.95,
 "n_before": 19, "n_after": 13, "removed": [nomes...], "reused": False}
```

Erro se a poda removeria todos os modelos.

**Nota sobre `reused`.** Quando a seleção pedida resolve identicamente a um pool
já registrado, o handle existente é devolvido e `reused=True` acompanha uma nota
explicando isso — o agente não fica adivinhando se recebeu um handle novo.

### 6.3 Pesos (6)

Todas devolvem um handle e um resumo. **Nunca os números.**

Retorno comum:

```python
{"weights": "w1", "method": "inverse_error", "pool": "pool_full",
 "reused": False,
 "effective_mode": "inverse_error",     # o modo REALMENTE usado (pode ser fallback)
 "summary": {"n_models": 19, "n_active": 19,     # n_active = peso > 0.01
             "concentration": 0.003,             # Herfindahl normalizado: 0=uniforme, 1=tudo num modelo
             "top3": [{"model", "share_pct"}, ...],
             "per_horizon": False},
 "note": "raw values stay in the state; pass this handle to combine_weighted"}
```

#### `weights_inverse_error(pool, windows=None, metric='rmse', shrinkage=0.0, per_horizon=False)`

```
w_m ∝ 1 / (erro_m + 1e-8)
w   = (1-shrinkage)·w + shrinkage·uniforme        # shrinkage recortado em [0, 0.9]
w   = projeção_no_simplex(w)
```

#### `weights_softmax_neg_error(pool, windows=None, metric='rmse', eta=1.0, per_horizon=False)`

```
w = softmax(-eta · erro / mediana(erro))          # eta recortado em [0.01, 20]
```

A normalização pela mediana torna `eta` livre de escala.

#### `weights_error_trend(pool, windows=None, metric='mae', eta=1.0, damping=None)`
Pesa por **para onde o erro de cada modelo está indo**, não pela média. Lê a grade
de erro ponto a ponto `(W, M, H)` em vez de um número por modelo.

```
1. slopes[m,h] = inclinação por mínimos quadrados do erro do modelo m
                 no passo de horizonte h, ao longo das janelas
2. model_slope[m] = mediana_h(slopes[m,h])        # neutraliza o perfil do horizonte
3. level[m] = erro médio do modelo m na ÚLTIMA janela
4. damping: None -> adaptativo, = clip(2·(concordância_de_sinal − 0.5), 0, 1)
            valor -> fixo para todos
5. previsto[m] = max(level[m] + damping[m]·model_slope[m], piso)
6. w = softmax(-eta · previsto / mediana(previsto)), projetado no simplex
```

Fitar a inclinação **por passo de horizonte** separa a degradação real do fato de
que o passo 8 é mais difícil que o passo 1 para todo mundo. Com `W < 3` cai para
`weights_softmax_neg_error` e `effective_mode` reporta o fallback.

Meta extra no retorno: `damping` (adaptive/fixed), `mean_damping`, `n_worsening`,
`n_improving`, `n_points_per_model`.

#### `weights_ols(pool, windows=None, l2=0.0, nonneg=True, per_horizon=False)`
Mínimos quadrados (Granger-Ramanathan). `l2 > 0` vira ridge. Com `nonneg=True`
(default) a solução é projetada no simplex.

```
A = XᵀX + l2·I ;  b = Xᵀy ;  w = solve(A, b)   (pinv se singular)
w = projeção_no_simplex(w)   se nonneg
```

Retida do catálogo quando `W < min_windows_for_ols`.

#### `weights_feature_based(pool, windows=None, metric='smape', eta=1.0)`
Meta-modelo **por série**: um regressor XGBoost por modelo, mapeando
características da janela (nível, dispersão, inclinação, acf1, tamanho) para o
erro daquele modelo, e depois `softmax(-erro previsto)`.

Cai para `softmax(-erro médio)` quando: `W < 3`; `xgboost` indisponível;
`W < 2 × n_features`; ou alvo não finito. O modo real fica em `effective_mode`.

#### `weights_pooled_meta_model(pool, eta=1.0)`
Consulta o meta-modelo cross-series de §4.2. Prediz para **todos** os modelos da
execução, não só os do pool pedido, porque um fold pode reselecionar membros
diferentes.

Retorno adicional:

```python
{"objective": "fforma", "n_train_series": 181, "n_models_with_a_fit": 19, ...}
```

Erro se nenhum meta-modelo foi treinado para a execução.

### 6.4 Combinação (6)

Estas apenas **montam** o objeto de estratégia; não pontuam nada. São açúcar
sintático opcional — `evaluate_strategy` aceita os mesmos campos diretamente.

| ferramenta | assinatura |
|---|---|
| `combine_mean` | `(pool='pool_full')` |
| `combine_median` | `(pool='pool_full')` |
| `combine_trimmed_mean` | `(pool='pool_full', trim_pct=0.2)` |
| `combine_weighted` | `(pool, weights)` |
| `combine_dba` | `(pool='pool_full')` |
| `combine_best_single` | `(model_id)` |

Retorno comum:

```python
{"strategy": {...},        # spec normalizada
 "n_models": 13,
 "next_step": "call evaluate_strategy with exactly this Action Input",
 "next_action_input": {"strategy": {...}, "rationale": "<why this should work>"}}
```

### 6.5 Validação (3)

#### `evaluate_strategy(strategy=None, combine=None, pool=None, weights=None, trim_pct=None, model=None, rationale='', iteration=None)`

**O único caminho para o histórico.** Monta a estratégia, roda o backtest e
ranqueia. Aceita todas as formas que um agente naturalmente tenta:

```python
{"combine": "weighted", "pool": "pool1", "weights": "w1"}       # plana
{"strategy": {"combine": "weighted", "pool": "pool1", ...}}     # aninhada
{"strategy": "weighted", "pool": "pool1", "weights": "w1"}      # método + irmãos
{"strategy": <o dict inteiro que combine_* devolveu>}           # passagem direta
```

Retorno:

```python
{"id": "a12", "strategy": {...}, "already_tested": False,
 "rank": 1, "total_attempts": 12,
 "metrics": {"rmse", "smape", "mape", "pocid"},
 "rmse_per_window": [3949.33, 4374.10, 1243.15],
 "score": 0.5311,
 "current_best": {"id", "score", "strategy", "origin"},
 "worse_than_best_by": 0.0842,        # gap relativo, None se for a melhor
 "is_best": True,
 # presente só quando aplicável:
 "numerically_identical_to": "a7",
 "note": "this produces the SAME forecasts as a7 — it adds nothing new. ..."}
```

Reenviar uma estratégia já testada não cria entrada nova: devolve a existente com
`already_tested=True`.

**Detecção de gêmeo numérico.** Antes de retornar, os resíduos da nova tentativa
são comparados com os de todas as anteriores (`np.allclose`, tolerância 1e-9). Se
alguma for idêntica, isso é dito explicitamente — pesos ajustados em 3 janelas
frequentemente ficam a poucos por cento do peso uniforme, o que torna uma
estratégia "ponderada" aritmeticamente igual à média do próprio pool.

#### `sanity_check(reference)`
Compara a previsão de teste contra os limites históricos. **Não bloqueia nada.**
`reference` é um id de tentativa (`"a3"`) ou uma spec.

```python
{"n_points", "forecast_range": [min, max], "historical_range": [min, max],
 "extrapolates_history": bool, "points_outside_history": int,
 "points_outside_band": int,     # banda = mediana ± tolerância·desvio (tolerância=3.0)
 "warnings": [...], "ok": bool}
```

Extrapolar o intervalo histórico é normal numa série com tendência, então isso é
informação; o aviso é a banda robusta em torno da mediana.

#### `list_attempts(top_n=10)`

```python
{"total": 14, "ranking": [brief de cada tentativa...], "best": "a14"}
```

---

## 7. Algoritmos centrais

### 7.1 Funções de combinação

Todas recebem uma matriz `(n_pool, H)` de uma janela e devolvem `(H,)`. Nenhuma
olha para `y_true`.

| método | operação |
|---|---|
| `mean` | `nanmean` no eixo dos modelos |
| `median` | `nanmedian` no eixo dos modelos |
| `trimmed_mean` | ordena por horizonte, remove `floor(m·trim_pct)` de cada cauda, tira a média; se `k<=0` ou `2k>=m`, vira média |
| `weighted` | média ponderada; peso de modelo com previsão NaN é redistribuído entre os demais para a soma continuar 1; horizonte sem peso válido cai para a média |
| `dba` | DTW Barycenter Averaging (tslearn), `max_iter=30`; RNG global do numpy semeado imediatamente antes da chamada; qualquer falha cai para a média |
| `best_single` | copia a linha do modelo escolhido |

### 7.2 Protocolo anti-vazamento

Duas disciplinas distintas, deliberadamente:

**Ajuste de pesos** — segue `backtest_mode`:

```python
_fit_windows(requested, exclude):
    pool = requested ou todas as janelas
    se exclude is None:        devolve pool
    se backtest_mode == "loo": devolve [i for i in pool if i != exclude]
    senão (expanding):         devolve [i for i in pool if i <  exclude]
```

**Escolha de quais modelos comparar** — sempre leave-one-out:

```python
_selection_windows(exclude):
    se exclude is None: devolve todas as janelas
    senão:              devolve [i for i in range(W) if i != exclude]
```

São problemas diferentes: ajustar peso é estimativa prospectiva (só o passado pode
informar o número aplicado ao futuro), escolher modelo é seleção de modelo (onde
LOO é o padrão e o uso eficiente de 3 janelas).

### 7.3 Backtest

```python
backtest(spec):
    spec = normalize_spec(spec)
    para cada janela i em 0..W-1:
        idx = pool_for_window(spec["pool"], exclude_window=i)
        # se nested_selection e a receita é re-ajustável, a composição é
        # recalculada em _selection_windows(i) — a janela i não votou em quem
        # está sendo pontuado nela
        janela = y_preds[i][idx, :]
        se spec é "weighted":
            fit = _fit_windows(recipe.fit_windows, exclude=i)
            se fit vazio: pesos uniformes
            senão: pesos = resolve_recipe(recipe, y_true[fit], y_preds[fit, idx])
        saída[i] = apply_combination(janela, método, pesos, ...)
    devolve saída (W, H)
```

`apply_to_test` é o mesmo caminho com `exclude_window=None` e a composição de pool
ajustada em todas as janelas.

### 7.4 Score composto

Menor é melhor. Normalizado contra uma âncora: a média simples de **todos** os
modelos, backtestada sob o mesmo protocolo.

```python
score = Σ_k peso_k · (métrica_k / âncora_k)  −  peso_pocid · (POCID / 100)
```

com `k ∈ {RMSE, SMAPE, MAPE}`. Métricas de peso zero são puladas mesmo se forem
NaN — é isso que faz o preset `scale_free_safe` funcionar em séries que cruzam o
zero. Denominador não finito ou ~0 → razão `inf` (ou `1.0` se o numerador também
for zero).

Presets:

| preset | RMSE | SMAPE | MAPE | POCID |
|---|---|---|---|---|
| `balanced` (default) | 0.3 | 0.3 | 0.2 | 0.2 |
| `rmse_focus` | 0.5 | 0.2 | 0.2 | 0.1 |
| `direction_focus` | 0.25 | 0.25 | 0.1 | 0.4 |
| `robust_smape` | 0.2 | 0.5 | 0.1 | 0.2 |
| `scale_free_safe` | 0.7 | 0.1 | 0.0 | 0.2 |

O score agregado usa as métricas sobre toda a validação achatada; POCID é a média
das POCIDs por janela (é direcional dentro de uma janela).

### 7.5 Métricas

| métrica | fórmula |
|---|---|
| MAPE | `mean(\|p−y\| / \|y\|)`, fração; `mape_zero="skip"` descarta pontos com `\|y\| <= 1e-8` |
| SMAPE | `mean(2·\|p−y\| / (\|p\|+\|y\|))`, em `[0,2]`; denominador zero → NaN |
| MSMAPE | `mean(2·\|p−y\| / max(0.5+ε, \|p\|+\|y\|+ε))`, `ε=0.1` |
| MAE | `mean(\|p−y\|)` |
| RMSE | `sqrt(mean((p−y)²))` |
| POCID | `100 · fração de passos consecutivos com direção correta` |

As seis métricas gravadas no CSV são recalculadas por `all_functions` com o
reshape `(1,-1)` e o MAPE do sklearn, preservando compatibilidade byte a byte com
o formato histórico do projeto.

### 7.6 Confiança de seleção

Resposta determinística a "esta escolha é defensável?". Compara a vencedora com a
**primeira alternativa numericamente distinta** (gêmeos são pulados; comparar com
uma cópia de si mesma daria margem zero por um motivo que não diz nada sobre os
dados).

```python
{"n_windows", "n_attempts", "winner", "runner_up", "twins_skipped",
 "margin",              # (score_2º − score_1º) / |score_1º|
 "bootstrap_pvalue",    # bootstrap pareado sobre per_window_scores
 "dm_pvalue",           # Diebold-Mariano sobre os resíduos, HLN-corrigido
 "alpha": 0.10, "bootstrap_reliable": bool,
 "verdict"}
```

| veredito | condição |
|---|---|
| `separated` | todos os testes considerados rejeitam em `alpha=0.10` |
| `indistinguishable` | nenhum rejeita |
| `weak` | discordam |
| `no_distinct_alternative` | toda alternativa é gêmea da vencedora |
| `undetermined` | nenhum p-valor disponível |

Com `W < 5` o bootstrap é marcado não confiável (reamostrar 3 valores só produz
um punhado de médias distintas) e o veredito segue apenas o DM, que trabalha
sobre `W × H` resíduos.

### 7.7 Verificação de procedência

```python
{"n_tool_calls", "n_successful", "n_failed",
 "n_evaluate_calls", "n_agent_attempts", "n_baseline_attempts",
 "agent_called_tools",    # houve ao menos uma chamada bem-sucedida
 "evaluated_via_tool",    # n_evaluate_calls >= número de tentativas do agente
 "all_backtested",        # toda tentativa tem W·H resíduos e W scores por janela
 "provenance_ok"}         # conjunção dos três
```

### 7.8 Redutibilidade

A estratégia vencedora é, na prática, a média simples do próprio pool?

```python
{"equivalent_to_pool_mean": bool,      # diferença relativa <= 0.01
 "pool_mean_relative_diff": float,     # max|escolhida − média| / max|média|
 "weights_concentration": float|None}
```

Compara **o que foi reportado** (que sob `ensemble` difere da vencedora isolada)
contra a média simples do mesmo pool.

---

## 8. Saída

### 8.1 CSV — 58 colunas

`<output_dir>/orchestrator_react_<version>/<DATASET>.csv`, separador `;`, uma
linha por série (séries que falham também geram linha, com métricas NaN e o erro
em `description`).

**Avaliação (13)** — nomes, ordem e cálculo preservados do formato histórico:

`dataset_index`, `horizon`, `regressor`, `mape`, `pocid`, `smape`, `rmse`,
`msmape`, `mae`, `test`, `predictions`, `start_test`, `final_test`

**Decisão e rastreabilidade (23):**

| coluna | conteúdo |
|---|---|
| `description` | JSON completo da decisão (estratégia, pesos, validação, procedência, confiança, config) |
| `decision_report` | resumo de uma linha |
| `llm_artifacts_path` | caminho absoluto do JSON do artefato |
| `score_preset` | preset usado |
| `tool_missing` | houve erro de contrato de ferramenta |
| `tools_called` | JSON `[{tool, ok, args}, ...]` |
| `n_tool_calls`, `n_evaluate_calls`, `provenance_ok` | procedência |
| `final_candidate_names`, `final_candidate_count` | histórico ranqueado |
| `best_strategy_name`, `best_strategy_method`, `best_strategy_params` | a vencedora |
| `predict_debug` | debug da aplicação final |
| `selected_base_models`, `n_pool_models` | o pool sobre o qual foi construída |
| `effective_models`, `n_effective_models` | modelos com peso > 1% |
| `weights_concentration`, `equivalent_to_pool_mean`, `pool_mean_relative_diff` | redutibilidade |
| `weights_by_horizon` | JSON `{horizonte: {modelo: peso}}` para toda estratégia |

**Diagnóstico e controle (22):**

| coluna | conteúdo |
|---|---|
| `series_profile_json` | o perfil completo, incluindo catch22 |
| `ranking_stability_score` | Kendall tau médio |
| `error_correlation_groups` | grupos redundantes |
| `pool_composition_mode` | `full` / `top_k_error` / `top_k_stable` |
| `react_iterations_used`, `react_early_stopped` | uso do orçamento |
| `react_trajectory_json` | a trajetória compacta, turno a turno |
| `baseline_results_json` | `{seeded: {...}, external: {mean, median, dba, ADE, FFORMA}}` |
| `weights_handle_resolved` | os pesos numéricos por nome de modelo |
| `agent_model_combinator`, `agent_model_diagnostico`, `agent_model_relato` | modelos por papel |
| `accept_confidence` | o número que o agente declarou |
| `selection_margin`, `selection_bootstrap_pvalue`, `selection_dm_pvalue`, `selection_verdict` | confiança medida |
| `calibration_gate_triggered` | o loop foi pulado |
| `ablation_config` | `<version>-<sha1[:10]>` da config inteira |
| `justificativa_final` | prosa da Fase 5, ou a justificativa causal determinística |
| `test_has_zero_actual`, `test_min_abs_actual` | flag de janela com valor real em zero (sMAPE satura ali independentemente da previsão) |

`weights_by_horizon` é preenchido para **toda** estratégia, não só as ponderadas,
para o CSV ficar uniformemente analisável:

| método | pesos equivalentes |
|---|---|
| `mean` | uniforme sobre o pool |
| `median` | indicador do(s) elemento(s) que a mediana seleciona naquele horizonte |
| `trimmed_mean` | uniforme sobre os que sobraram após o corte, por horizonte |
| `weighted` | os pesos resolvidos |
| `best_single` | one-hot |
| `dba` | uniforme, marcado `nominal=True` — um baricentro DTW não é média ponderada |

### 8.2 Artefato JSON

Um por série, em
`<output_dir>/orchestrator_react_<version>/llm_artifacts/<DATASET>/dataset_<i>.json`:

```python
{"dataset", "dataset_index", "success", "error",
 "config",            # ReactConfig inteira
 "decision",          # o mesmo JSON de `description`
 "series_card",       # perfil completo
 "pool_card",         # card do pool completo
 "diagnosis",
 "cross_series": {    # o contexto de dataset que esta série recebeu
    "strategy_prior", "prior_best", "prior_worst",
    "dataset_card_shown",
    "pooled_meta_model": {"objective", "n_train_series", "n_features", "degenerate"}},
 "phase2": {"pool", "baselines", "calibration_gate"},
 "react": {"trajectory", "summary", "errors", "parse_failures", "tools"},
 "predict_debug", "sanity", "report_text", "warnings"}
```

`parse_failures` guarda a saída bruta de todo turno que o parser não conseguiu
ler — é onde se olha quando um modelo insiste em errar o formato.

---

## 9. Configuração

Um objeto `ReactConfig` por execução. A configuração inteira é serializada em
`ablation_config` como `<name>-<sha1[:10]>`, então qualquer linha de resultado é
reconstruível.

| campo | default | efeito |
|---|---|---|
| `n_validation_windows` | `3` | janelas de validação por série |
| `backtest_mode` | `"expanding"` | `expanding` ou `loo`, para ajuste de pesos |
| `nested_selection` | `True` | reescolhe a composição do pool dentro de cada fold |
| `pool_mode` | `"full"` | `full`, `top_k_error`, `top_k_stable` |
| `pool_k` | `8` | usado pelos dois modos `top_k_*` |
| `score_preset` | `"balanced"` | pesos do score composto |
| `seasonal_period` | `None` | `None` = inferido da `@frequency` |
| `mape_zero` / `mape_epsilon` | `"skip"` / `1e-8` | tratamento de zero no MAPE |
| `max_iterations` | `12` | orçamento do loop |
| `early_stop_patience` | `4` | propostas consecutivas sem melhora antes de parar |
| `min_improvement` | `1e-4` | ganho relativo mínimo que conta como progresso |
| `show_attempt_history` | `True` | mostra o histórico ranqueado no prompt |
| `show_attempt_rationales` | `True` | inclui as justificativas no histórico |
| `seed_stable_pools` | `True` | semeia as 6 combinações por estabilidade |
| `pooled_meta_model` | `True` | roda o pré-passo cross-series |
| `pooled_meta_model_objective` | `"fforma"` | `fforma` ou `per_model` |
| `pooled_meta_model_min_series` | `20` | mínimo de séries para treinar |
| `seed_pooled_meta_model` | `True` | semeia `weighted(pooled_meta_model)` |
| `dataset_card` | `True` | injeta o prior de dataset no prompt |
| `drop_misaligned_models` | `True` | remove modelos com alvo divergente |
| `final_strategy` | `"argmin"` | `argmin`, `ensemble` ou `prior_blend` |
| `final_top_m` / `final_eta` | `3` / `5.0` | parâmetros do `ensemble` |
| `final_prior_alpha` | `0.0` | encolhimento do `prior_blend`; 0 reproduz `argmin` |
| `min_windows_for_ols` | `5` | abaixo disso `weights_ols` é retida |
| `calibration_gate` / `calibration_gate_kendall` | `False` / `0.85` | pula o loop com ranking já estável |
| `sanity_check_tolerance` | `3.0` | múltiplos do desvio histórico |
| `combinator` | `LLMRole("gpt-oss:20b", temperature=0.2, seed=7)` | agente da Fase 3 |
| `diagnostician` | `LLMRole(None)` | Fase 1b; `None` = regras determinísticas |
| `reporter` | `LLMRole(None)` | Fase 5 |

`LLMRole` tem `model`, `temperature`, `base_url`, `seed` (default 7, repassada ao
Ollama) e `reasoning` (repassada só quando definida).

Variáveis de ambiente sobrescrevem os modelos por papel:
`REACT_MODEL_COMBINATOR`, `REACT_MODEL_DIAGNOSTICIAN`, `REACT_MODEL_REPORTER`,
`REACT_OLLAMA_URL`, `REACT_CONFIG` (caminho de um JSON aplicado primeiro).

---

## 10. Execução

```bash
python run_tsf_orchestrator.py \
    --dataset ANP_MONTHLY \
    --source mes_11_venda_mensal.tsf \
    --combinator gpt-oss:20b \
    --version v5
```

Vários datasets numa chamada:

```bash
python run_tsf_batch.py \
    --datasets ETTH1 ETTH2 ANP_MONTHLY \
    --combinators gpt-oss:20b \
    --version v5
```

**Preflight.** Antes da primeira série, uma chamada trivial ao LLM verifica que o
servidor responde e o modelo existe. Um servidor fora do ar falha identicamente em
todas as séries; detectar isso custa uma chamada, não detectar custa uma execução
inteira de fallback determinístico silencioso.

**Tolerância a falha.** `max_llm_failures` (default 5) séries podem falhar por
erro de LLM antes de a execução abortar. Cada falha é impressa com o erro
subjacente e a linha correspondente é marcada. `allow_baseline_fallback=True`
nunca aborta (e a linha vira baseline determinística, o que é sempre dito no log).

**Braço determinístico.** `--no-llm` desliga os três papéis de LLM: as sementes da
Fase 2 são avaliadas e a melhor é aplicada, sem agente nenhum.
