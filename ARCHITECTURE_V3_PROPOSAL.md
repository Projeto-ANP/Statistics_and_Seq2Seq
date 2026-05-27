# Arquitetura V3 — Multi-Agente LLM para Combinação de Previsões com Poda e Encolhimento Robusto

> **Status:** EM IMPLEMENTAÇÃO. Este documento descreve o diagnóstico do sistema atual, a arquitetura
> proposta, exatamente onde alterar o código, por que cada alteração melhora o resultado, o protocolo
> experimental para publicação e as referências.

> ### Decisões resolvidas (confirmadas com o usuário + inspeção dos dados)
> 1. **Estrutura dos dados (confirmada):** cada `dataset_index` tem **exatamente 4 linhas** no CSV (4
>    rodadas do dataset encurtado, contíguas). A linha `iloc[-1]` é o **teste real** (para o qual
>    combinamos); as 3 anteriores (`iloc[-4:-1]`) são **validação**. Sempre será assim.
> 2. **Bug de contagem de janelas (confirmado):** `iloc[-train_window:-1]` com `train_window=3` retorna
>    só 2 linhas (posições -3,-2), perdendo a janela mais antiga. Correção: `iloc[-(train_window+1):-1]`
>    → 3 janelas de validação. Verificado em ARIMA/catboost NN5_WEEKLY (horizon=8, janelas sequenciais).
> 3. **Série histórica (resolvida):** `streamfuels` **não está disponível** no ambiente. A coluna `test`
>    é **idêntica entre todos os modelos** (é a verdade) e as janelas são **contíguas e não-sobrepostas**.
>    Logo, o histórico recente = **concatenação dos arrays `test` das 3 janelas de validação**
>    (3×horizon pontos; ex.: 24 para NN5_WEEKLY). Leakage-safe (não usa a janela de teste).
> 4. **FFORMA/ADE como baselines:** já existem como CSVs de resultado em
>    `./timeseries/mestrado/resultados/{FFORMA,ADE}/NN5_WEEKLY_DATASET.csv`. Comparação final é externa;
>    internamente o pipeline já tem `ade_dynamic_error_per_horizon` e `ridge_stacking` (FFORMA-like) como
>    candidatos para o Oracle.

---

## 1. Diagnóstico — por que o V2 perde para `mean` e `ADE`/`FFORMA`

No `NN5_WEEKLY` (111 séries) o `orchestrator_v2` ficou em **último** (SMAPE 0.1240), atrás de
ADE (0.1178), FFORMA (0.1197), `mean` (0.1199), `median` (0.1201) e DBA (0.1226). Isto não é
azar — é consequência direta de três falhas estruturais que encontrei ao reler o código.

### Falha A — Variância de seleção com pouquíssimas janelas (a causa dominante)

Em [`orchestrator_langchain/context.py:103`](orchestrator_langchain/context.py#L103):

```python
df_filtred_sample = df_sample.iloc[-train_window:-1]   # train_window=3  →  iloc[-3:-1]  →  2 LINHAS
n_windows = len(df_filtred_sample)                      # = 2 janelas de validação
```

Com `train_window=3` o orquestrador escolhe **1 vencedor entre ~13 estratégias** com base em
**apenas 2 janelas** de validação. O `evaluate_all` ([`orchestrator/evaluator.py:131`](orchestrator/evaluator.py#L131))
ordena candidatos por um score calculado sobre essas 2 janelas e pega o `ranking[0]` (winner-take-all).

Isto é overfitting de ruído por construção. O vencedor das 2 janelas raramente generaliza para a
janela de teste. A `mean` tem **variância de estimação zero** (não estima nada); qualquer método que
estima pesos/seleciona modelos a partir de 2 pontos tem variância enorme. Este é exatamente o
**"forecast combination puzzle"**: o erro de estimação dos pesos engole o ganho teórico da combinação
(Claeskens et al. 2016; Smith & Wallis 2009).

> **Consequência:** o desenho *winner-take-all* maximiza justamente a variância que estamos tentando
> evitar. É o pior desenho possível para amostras curtas.

### Falha B — Nenhuma poda de modelos ruins (limitação que você apontou)

Todos os combinadores em [`orchestrator/strategies.py`](orchestrator/strategies.py) operam sobre o
**pool completo** de modelos. `mean`/`median` incluem regressores que vão muito mal e poluem a
combinação. FFORMA implicitamente reduz o peso deles via XGBoost; ADE adapta dinamicamente. O V2 não
tem nenhum mecanismo explícito de **remover** o modelo ruim antes de combinar. A literatura mostra que
**podar o pool primeiro** (Kourentzes et al. 2019; Wang et al. 2023) é um dos jeitos mais baratos e
eficazes de melhorar a combinação — e frequentemente faz a média simples dos sobreviventes bater
métodos sofisticados sobre o pool inteiro.

### Falha C — O "annotator" não vê a série, só vê previsões

`load_validation_from_context` ([`orchestrator/data_contract.py:38`](orchestrator/data_contract.py#L38))
só carrega `predictions` (por modelo) e `test` (verdade) das janelas. **A série histórica bruta nunca
entra no contexto.** O `SeriesAnnotator` roda STL sobre segmentos de tamanho `horizon` do `y_true` de
2 janelas — base fraquíssima para "entender a série". A alegação central da tese ("usar LLM para extrair
conhecimento da série que modelos tradicionais não conseguem") fica sem sustentação empírica porque a
LLM literalmente não recebe a série.

### Resumo do diagnóstico

| Falha | Efeito | Métrica afetada |
|---|---|---|
| A. Winner-take-all com 2 janelas | Overfitting de ruído na seleção | Perde para `mean` (variância) |
| B. Sem poda de modelos ruins | Pool poluído por regressores fracos | Perde para FFORMA/ADE (que downweight) |
| C. LLM não vê a série | "Entendimento" sem base | Fragiliza a narrativa do paper |

---

## 2. Princípio da solução

> **A LLM decide o que a estatística não consegue decidir bem em amostra curta (estrutura: quais modelos
> podar, qual regime de combinação, quão agressivo encolher). Os pesos numéricos vêm de estimadores
> robustos de baixa variância. O combinador final é ancorado em "média-dos-sobreviventes" e só se afasta
> dela quando há evidência estatisticamente significativa.**

Isto ataca as três falhas de uma vez:

- **Contra A:** trocamos *winner-take-all* por um combinador **ancorado e encolhido** (double shrinkage,
  Liu 2024) — variância de estimação baixa por design. Nunca cai muito abaixo da média-dos-sobreviventes.
- **Contra B:** introduzimos um **agente Podador (ModelCritic)** que remove modelos ruins/redundantes,
  com piso estatístico no Model Confidence Set (Hansen et al. 2011; Samuels & Sekkel 2017).
- **Contra C:** passamos a **série histórica** para o contexto e calculamos features ricas (catch22 /
  tsfeatures) que a LLM lê junto com o raciocínio qualitativo.

A novidade publicável: **LLM como meta-controlador de um combinador robusto, guiado por entendimento
semântico + estatístico da série** — diferente de FFORMA (meta-learner XGBoost de caixa-preta sobre
features fixas) e de ADE (meta-learner de erro por modelo). A LLM faz escolhas estruturais
interpretáveis e auditáveis; a robustez vem da ancoragem + shrinkage.

---

## 3. Arquitetura V3 — três agentes + núcleo determinístico robusto

```
                ┌─────────────────────────────────────────────────────────────┐
                │  Contexto (NOVO: inclui série histórica + features catch22)   │
                └─────────────────────────────────────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                  ▼                                  ▼
┌──────────────────┐          ┌──────────────────────┐          ┌────────────────────────┐
│ Agente 1         │          │ Agente 2             │          │ Agente 3               │
│ SeriesAnalyst    │  ──────► │ ModelCritic (PODA)   │  ──────► │ CombinationArchitect   │
│ (semântico+stat) │ profile  │ remove ruins/redund. │ survivors│ escolhe REGIME+λ        │
└──────────────────┘          └──────────────────────┘          └────────────────────────┘
                                          │                                   │
                                          ▼                                   ▼
                              ┌────────────────────────────────────────────────────────┐
                              │  NÚCLEO DETERMINÍSTICO ROBUSTO                          │
                              │  • piso = pruned_equal_weights (sempre calculado)       │
                              │  • double-shrinkage (Liu 2024) sobre sobreviventes      │
                              │  • só escala p/ inverse-RMSE/stacking se DM-significativo│
                              │  • Oracle + baselines (mean, median, FFORMA, ADE)       │
                              └────────────────────────────────────────────────────────┘
```

### Agente 1 — SeriesAnalyst (substitui o SeriesAnnotator)

**Mudança principal:** recebe a **série histórica de treino completa** (não só as janelas de previsão) +
um bloco de features determinísticas (catch22 ou tsfeatures equivalentes, calculadas em Python). A LLM
produz um `SeriesProfile` combinando o que ela lê das features com raciocínio qualitativo (regime,
quebras estruturais, sazonalidade dominante, previsibilidade).

**Fonte da série histórica (anti-leakage):** a série crua é lida do `.tsf` original e truncada em
`series_value[:-horizon]` — exatamente o `train` que os modelos-base usaram para prever o teste final,
então **nunca vaza o alvo de teste**. É a escala real bruta (a mesma das colunas `test`/`predictions`
nos CSVs), logo as transformações internas de cada modelo (normalização, STL) são irrelevantes — elas já
foram desfeitas antes de virarem previsão. Para NN5 weekly isso dá **105 pontos** (113 − horizonte 8) em
vez dos 24 do proxy anterior, o que reativa as features sazonais (limiar `n ≥ 2·período`). Se o `.tsf` não
for encontrado, há fallback automático para o proxy (concat dos `test` das janelas de validação). O caminho
do `.tsf` é passado **explicitamente** na execução (`original_tsf_path`) — ver §11.

- **Entrada:** série histórica completa daquela `dataset_index`, features (entropia espectral, força de
  tendência/sazonalidade, Hurst, estacionariedade/ADF, nº de quebras), e nome do dataset (prior de domínio).
- **Saída:** `SeriesProfile` (mantém o schema do V2, acrescentando `forecastability` e `regime`).
- **Temperatura 0** → reprodutível.

### Agente 2 — ModelCritic (NOVO — resolve a Falha B)

Este é o agente que materializa a sua observação ("remover candidatos de regressores muito ruins").

- **Entrada por modelo:** RMSE/SMAPE por janela, viés por horizonte, drift (slope do RMSE entre folds),
  matriz de correlação de erros entre modelos, e **pertencimento ao Model Confidence Set** (calculado
  deterministicamente em Python via DM/bootstrap — já temos `tie_break_analysis` em
  [`orchestrator/diagnostics.py`](orchestrator/diagnostics.py)).
- **Decisão da LLM:** lista `prune_models` + justificativa por modelo. Regras (no prompt, com piso
  estatístico que o código garante):
  1. **Nunca podar** um modelo que está no MCS "superior set" a menos que seja redundante (corr > 0.95
     com outro melhor) — piso estatístico imposto pelo código, não pela LLM.
  2. **Podar** modelos consistentemente piores (RMSE > k× melhor) E instáveis entre janelas.
  3. **Podar redundância:** entre dois modelos com correlação de erro > 0.95, manter o de menor RMSE.
  4. **Manter diversidade mínima:** nunca deixar o pool com menos de `max(3, ceil(sqrt(n_models)))`.
- **Saída:** `survivors` (lista de modelos mantidos) + `prune_report` auditável.

> O valor da LLM aqui é raciocinar *por que* um modelo é ruim (ex.: "FT_rf tem variância explosiva nesta
> série volátil → podar") de um jeito que um limiar fixo não captura. Mas o **piso estatístico (MCS)**
> impede a LLM de podar um modelo comprovadamente bom.

### Agente 3 — CombinationArchitect (substitui o StrategySelector)

Em vez de escolher 1 entre 13 (winner-take-all), escolhe um **regime de combinação** sobre o pool
**podado**, com a intensidade de encolhimento `λ` como knob central:

- **Default (regime "robust"):** `double_shrinkage` sobre sobreviventes — encolhe pesos para
  equal-weights (via WLS) e para zero (regularização). Em `n_windows≤2`, `λ→` forte (quase equal-weight).
- **Escalda para "adaptive" (inverse-RMSE/EWA/ADE-like)** somente se `SeriesProfile.concept_drift=true`
  E o ganho na validação for **DM-significativo** vs. `pruned_equal_weights`.
- **Escalda para "structured" (STL-stacking/ridge)** somente se `models_redundant=true` E ganho
  DM-significativo.
- **Saída:** `regime`, `shrinkage_lambda`, `score_preset`, `reasoning` citando campos do profile.

### Núcleo determinístico robusto (a garantia de consistência)

Independente do que a LLM disser, o código:

1. Sempre calcula `pruned_equal_weights` (média dos sobreviventes) — esse é o **piso**.
2. Aplica o regime escolhido com shrinkage forte.
3. **Gate de significância:** se o regime escolhido **não** supera `pruned_equal_weights` por uma margem
   DM-significativa nas janelas de validação, **cai de volta para `pruned_equal_weights`**.
4. Sempre roda Oracle + baselines (`mean`, `median`, FFORMA-like, ADE) para o delta empírico.

> **Por que isto bate `mean`:** `pruned_equal_weights` ⪰ `mean` sempre que houver ≥1 modelo ruim no pool
> (poda). **Por que bate FFORMA/ADE:** esses estimam muitos pesos e sofrem erro de estimação em séries
> curtas; nosso combinador encolhido tem variância muito menor e ainda assim explora performance via o
> regime adaptativo *quando há evidência*.

---

## 4. Onde alterar — mapa concreto de arquivos

| # | Arquivo / função | Alteração | Resolve |
|---|---|---|---|
| 1 | [`orchestrator_langchain/context.py`](orchestrator_langchain/context.py) `generate_all_validations_context` | Corrigir a fatia `iloc[-train_window:-1]` (hoje dá 2 janelas p/ train_window=3) e **carregar a série histórica** (`history`) no contexto, além das janelas de previsão. | A, C |
| 2 | [`orchestrator/data_contract.py`](orchestrator/data_contract.py) | Adicionar `y_history` ao `ValidationData` e um loader `load_history_from_context()`. | C |
| 3 | **Novo** `orchestrator/features.py` | Features determinísticas da série (catch22/tsfeatures: entropia espectral, força tendência/sazonalidade, Hurst, ADF, nº quebras). | C |
| 4 | [`orchestrator/diagnostics.py`](orchestrator/diagnostics.py) | Adicionar `model_confidence_set(residuals_by_model, alpha=0.1)` (Hansen et al. 2011) e `error_correlation_matrix` (já existe `error_similarity_matrix` — reusar). | B |
| 5 | [`orchestrator/strategies.py`](orchestrator/strategies.py) `generate_combined_predictions` | Adicionar método `double_shrinkage` (encolhe p/ equal + p/ zero) e suporte a `survivors` (subconjunto de modelos) em **todos** os métodos. | A, B |
| 6 | [`orchestrator/tools.py`](orchestrator/tools.py) | 3 novas tools: `series_analysis_brief` (features+history), `model_critic_brief` (diagnósticos+MCS), `combination_architect_brief` (regime+λ sobre pool podado). | A, B, C |
| 7 | `orchestrator_langchain/prompts/` | 3 novos prompts: `series_analyst.md`, `model_critic.md`, `combination_architect.md`. | A, B, C |
| 8 | [`orchestrator_langchain/agents.py`](orchestrator_langchain/agents.py) + [`orchestrator/agents.py`](orchestrator/agents.py) | Factories `create_series_analyst_agent`, `create_model_critic_agent`, `create_combination_architect_agent` (temp=0). | — |
| 9 | [`orchestrator/pipeline.py`](orchestrator/pipeline.py) | `run_llm_pipeline_v3`: Analyst → Critic (poda) → Architect → núcleo robusto com **gate DM vs pruned_equal_weights** + baselines FFORMA/ADE. | A, B |
| 10 | [`orchestrator_langchain/pipeline.py`](orchestrator_langchain/pipeline.py) | `run_langchain_pipeline_v3` (monkey-patch dos 3 factories). | — |
| 11 | [`run_tsf_orchestrator.py`](run_tsf_orchestrator.py) | `version="v3_pruning"`, novas colunas: `survivors`, `pruned_models`, `prune_report`, `regime`, `shrinkage_lambda`, `fellback_to_pruned_mean`, `fforma_score`, `ade_score`, deltas vs cada baseline. | — |

> **Importante (consistência):** a alteração #1 (corrigir a contagem de janelas) precisa ser decidida com
> você — hoje `train_window=3` ⇒ 2 janelas. Se a intenção era 3 janelas, a fatia deveria ser
> `iloc[-(train_window+1):-1]` ou similar. Confirmaremos antes de implementar.

---

## 5. Por que cada alteração melhora (ligação com a literatura)

1. **Poda do pool (ModelCritic + MCS)** — Kourentzes, Barrow & Petropoulos (2019) e Wang, Hyndman et al.
   (2023) mostram que remover modelos ruins/instáveis antes de combinar melhora acurácia a custo
   computacional baixo. Samuels & Sekkel (2017) formalizam "trim para o superior set via MCS, depois
   equal-weight". → resolve a Falha B e ataca a perda para FFORMA/ADE.

2. **Double shrinkage (regime default)** — Liu (2024, OBES) e Frazier et al. (2023) mostram que encolher
   pesos para equal-weights + para zero **resolve o combination puzzle** justamente em regime de poucos
   dados. → resolve a Falha A (variância de estimação) e é o motivo de bater `mean`.

3. **Gate de significância (DM vs pruned-equal-weight)** — Diebold-Mariano (1995) com correção
   Harvey-Leybourne-Newbold (1997), que já existe em [`diagnostics.py`](orchestrator/diagnostics.py).
   Só nos afastamos do piso robusto quando há evidência. → garante que **nunca** ficamos muito abaixo do
   baseline robusto (consistência).

4. **Série histórica + features (SeriesAnalyst)** — sustenta empiricamente a tese "LLM extrai
   conhecimento da série". A combinação de features estatísticas (estilo FFORMA/tsfeatures, Montero-Manso
   et al. 2020) com raciocínio qualitativo da LLM é a contribuição metodológica. → resolve a Falha C.

5. **Anchoring + interpretabilidade** — diferente do XGBoost caixa-preta do FFORMA, cada decisão
   (poda, regime, λ) é textual e auditável, citando campos do profile. Isto endereça a crítica de
   interpretabilidade do FFORMA levantada na literatura recente.

---

## 6. A seleção de modelos influencia? (sua pergunta)

**Sim, fortemente — e é uma alavanca central da V3.** Dois eixos:

1. **Composição do pool base.** Hoje o pool tem ~19 modelos, vários correlacionados (variações
   CWT/DWT/FT de rf/catboost). Pools redundantes inflam a variância dos pesos e atrapalham a média. A
   poda (Agente 2) resolve isto *por série* — em vez de um pool fixo, cada série fica com o subconjunto
   que maximiza acurácia + diversidade.

2. **Diversidade > quantidade.** A literatura de combinação (e o "puzzle") é clara: o que importa é a
   diversidade dos erros, não o número de modelos. A matriz de correlação de erros guia a poda de
   redundância. Recomendo também **testar um pool reduzido e diverso** (ex.: 1 estatístico ARIMA/ETS/Theta
   + 1 ML rf/catboost + 1 transform CWT + Naive) como ablação — pode bater o pool grande.

---

## 7. Protocolo experimental para publicação

**Ablações (isolam cada contribuição):**

| Sistema | Poda | Shrinkage | LLM regime | Objetivo da comparação |
|---|---|---|---|---|
| `mean` (full pool) | ✗ | ✗ | ✗ | baseline ingênuo |
| `median` (full pool) | ✗ | ✗ | ✗ | baseline robusto |
| `pruned_mean` | ✓ | ✗ | ✗ | **efeito isolado da poda** |
| `pruned_shrinkage` | ✓ | ✓ | ✗ | **efeito do shrinkage** |
| `V3 completo` | ✓ | ✓ | ✓ | **valor agregado da LLM** |
| FFORMA | — | — | — | SOTA feature-based |
| ADE | — | — | — | SOTA dinâmico |

**Métricas:** SMAPE/MSMAPE/RMSE/MAE/POCID por série; agregado e por dataset. Reportar:
- **Win-rate** de V3 vs cada baseline (fração de séries onde ganha).
- **Teste de significância** entre séries: Wilcoxon signed-rank + diagrama de diferença crítica
  (Friedman + Nemenyi) — padrão em papers de combinação (Demšar 2006).
- **MCS final** entre todos os métodos (Hansen et al. 2011): V3 deve estar no superior set.
- **`fellback_to_pruned_mean` rate:** quantas vezes o gate caiu no piso — mede quão "agressiva" a LLM foi.

**Reprodutibilidade:** temp=0 nos 3 agentes; seeds fixas; modelos Ollama versionados.

**Datasets:** além de NN5_WEEKLY, rodar nos demais (ETTH1, ANP_MONTHLY, etc.) para generalização.

---

## 8. Modelos locais (Ollama, ≤24b)

Os agentes só emitem **JSON estruturado pequeno** (decisões), não geram previsões numéricas — então
modelos de 14b são suficientes e rápidos:

| Agente | Modelo recomendado | Temp |
|---|---|---|
| SeriesAnalyst | `qwen3:14b` (bom raciocínio numérico/estrutural) | 0.0 |
| ModelCritic | `qwen3:14b` | 0.0 |
| CombinationArchitect | `qwen3:14b` | 0.0 |

Alternativas ≤24b: `gemma2:27b` (borderline de tamanho/VRAM), `mistral-small:24b`. Evitar >24b pela
restrição. Como o piso estatístico (MCS, gate DM) protege contra alucinação, não precisamos de um modelo
gigante — o código garante o resultado mínimo.

---

## 9. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| LLM poda modelo bom | Piso MCS no código impede; diversidade mínima garantida |
| LLM escolhe regime ruim | Gate DM cai para `pruned_equal_weights` |
| Poucos dados p/ MCS confiável | Com `n_windows≤2`, MCS vira só correlação+RMSE ranking; shrinkage forte compensa |
| Custo de 3 chamadas LLM × N séries | Modelos 14b locais + JSON curto; cache de features determinísticas |
| FFORMA/ADE precisam ser baselines reais | Implementar/portar como candidatos no `evaluate_all` p/ comparação justa |

---

## 10. Referências

**Combinação de previsões — fundamentos**
- Bates, J.M. & Granger, C.W.J. (1969). *The Combination of Forecasts*. Oper. Res. Q. 20(4):451–468.
- Timmermann, A. (2006). *Forecast Combinations*. Handbook of Economic Forecasting, vol. 1.
- Stock, J.H. & Watson, M.W. (2004). *Combination forecasts of output growth*. J. Forecast. 23(6):405–430.

**Combination puzzle e shrinkage (núcleo da V3)**
- Claeskens, G., Magnus, J.R., Vasnev, A.L. & Wang, W. (2016). *The forecast combination puzzle: A simple
  theoretical explanation*. Int. J. Forecast. 32(3):754–762.
- Smith, J. & Wallis, K.F. (2009). *A simple explanation of the forecast combination puzzle*. Oxford Bull.
  Econ. Stat. 71(3):331–355.
- Liu, ... (2024). *Solving the Forecast Combination Puzzle Using Double Shrinkages*. Oxford Bull. Econ.
  Stat. — encolhe para equal-weights (WLS) + para zero (regularização). [arXiv:2308.05263]
- Frazier, D.T., Covey, ... (2023). *Solving the Forecast Combination Puzzle*. [arXiv:2308.05263 / Monash WP18-2023]
- Genre, V., Kenny, G., Meyler, A. & Timmermann, A. (2013). *Combining expert forecasts: Can anything beat
  the simple average?*. Int. J. Forecast. 29(1):108–121.

**Poda / trimming de pool (Agente ModelCritic)**
- Kourentzes, N., Barrow, D. & Petropoulos, F. (2019). *Treating and Pruning: New approaches to forecasting
  model selection and combination using prediction intervals*. Int. J. Forecast. — [ScienceDirect S0169207020301096]
- Wang, X., Hyndman, R.J., Li, F. & Kang, Y. (2023). *Another look at forecast trimming for combinations:
  robustness, accuracy and diversity*. [arXiv:2208.00139]
- Samuels, J.D. & Sekkel, R.M. (2017). *Model Confidence Sets and forecast combination*. Int. J. Forecast.
  33(1):48–60.
- Hansen, P.R., Lunde, A. & Nason, J.M. (2011). *The Model Confidence Set*. Econometrica 79(2):453–497.

**SOTA a bater**
- Montero-Manso, P., Athanasopoulos, G., Hyndman, R.J. & Talagala, T.S. (2020). *FFORMA: Feature-based
  forecast model averaging*. Int. J. Forecast. 36(1):86–92.
- Cerqueira, V., Torgo, L., Pinto, F. & Soares, C. (2017/2019). *Arbitrated Ensemble for Time Series
  Forecasting* (ADE). ECML-PKDD / Machine Learning.

**LLM-agents para séries temporais (estado da arte 2025)**
- Yeh, C.-C.M. et al. (2025). *Empowering Time Series Forecasting with LLM-Agents* (DCATS). [arXiv:2508.04231]
- *FLAIRR-TS: Forecasting LLM-Agents with Iterative Refinement* (2025). Findings of EMNLP 2025.

**Testes estatísticos / decomposição**
- Diebold, F.X. & Mariano, R.S. (1995). *Comparing Predictive Accuracy*. J. Bus. Econ. Stat. 13(3):253–263.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). *Testing the equality of prediction mean squared errors*.
  Int. J. Forecast. 13(2):281–291.
- Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990). *STL: A Seasonal-Trend Decomposition
  Procedure Based on Loess*. J. Off. Stat. 6(1):3–73.
- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets*. JMLR 7:1–30.

---

## 11. Status de implementação (CONCLUÍDA)

Toda a V3 foi implementada. Resumo do que mudou (arquivo → mudança):

- [`orchestrator_langchain/context.py`](orchestrator_langchain/context.py) — janelas corrigidas
  (`iloc[-(train_window+1):-1]` ⇒ 3 validações); grava `series_history` a partir da **série original do
  `.tsf` truncada em `[:-horizon]`** (105 pts no NN5) via `load_original_series_history()` +
  `resolve_tsf_path()` (resolução case-insensitive; aceita caminho explícito), com fallback ao proxy
  (concat dos `test`) e flag `series_history_source`; grava `dataset_name`. Correção de pandas 3.x:
  removido `infer_datetime_format` (descontinuado) em `read_model_preds`.
- [`orchestrator/data_contract.py`](orchestrator/data_contract.py) — `load_history_from_context()`.
- [`orchestrator/features.py`](orchestrator/features.py) **(novo)** — `compute_series_features`
  (forecastability, trend/seasonal strength, Hurst, ADF, variance-ratio, spectral entropy).
- [`orchestrator/diagnostics.py`](orchestrator/diagnostics.py) — `model_confidence_set()` (Hansen
  et al. 2011, bootstrap recentrado, range statistic).
- [`orchestrator/strategies.py`](orchestrator/strategies.py) + [`orchestrator/final_predictor.py`](orchestrator/final_predictor.py)
  — método `double_shrinkage_per_horizon` + suporte a `params["survivors"]` (poda) em ambos.
- [`orchestrator/tools.py`](orchestrator/tools.py) — `series_analysis_brief_tool`,
  `model_critic_brief_tool`, `combination_architect_brief_tool`, `_per_model_diagnostics`,
  `_infer_seasonal_period`.
- [`orchestrator_langchain/langchain_tools.py`](orchestrator_langchain/langchain_tools.py) — 3 wrappers `@tool`.
- `orchestrator_langchain/prompts/{series_analyst,model_critic,combination_architect}.md` **(novos)**.
- [`orchestrator_langchain/agents.py`](orchestrator_langchain/agents.py) + [`orchestrator/agents.py`](orchestrator/agents.py)
  — `create_{series_analyst,model_critic,combination_architect}_agent` (temp=0).
- [`orchestrator/pipeline.py`](orchestrator/pipeline.py) — `run_llm_pipeline_v3` + `_v3_apply_pruning_floor`
  + gate Diebold-Mariano vs `pruned_equal_weights`.
- [`orchestrator_langchain/pipeline.py`](orchestrator_langchain/pipeline.py) — `run_langchain_pipeline_v3`.
- [`run_tsf_orchestrator.py`](run_tsf_orchestrator.py) — `version="v3_pruning"`, modelos v3, parâmetro
  `original_tsf_path` (caminho do `.tsf` original repassado ao contexto), e **schema de colunas por versão**
  (`COLS_BASE` + `COLS_V3` para v3; `COLS_BASE` + `COLS_LEGACY` para v1/v2 via `cols_for_version()`). O CSV
  do V3 tem **48 colunas** (antes 88): mantém métricas + rastreabilidade da combinação (best_strategy_*,
  predict_debug, selected_base_models, weights_by_horizon, final_candidate_*) + as colunas próprias do V3
  (survivors, pruned_models, prune_blocked_by_mcs, mcs_superior_set, regime, shrinkage_lambda,
  fellback_to_pruned_mean, oracle_regime, llm_picked_best_regime, scores de baselines e deltas, think
  blocks, series_profile) e **remove** as de V1 (proposer/skeptic/statistician/pattern_analyst/debate) e
  V2 (oracle_best_*, baselines fixos, strategy_reasoning, annotator/selector think).

### Como rodar
O `__main__` já está configurado para V3 no `NN5_WEEKLY_DATASET` com `qwen3:14b` (temp 0) nos 3 agentes,
`version="v3_pruning"`, `train_window=3`. Basta executar `python run_tsf_orchestrator.py`.

**Identificação da série original (importante):** os nomes dos `.tsf` de origem **não** seguem a convenção
maiúscula das pastas de resultado — ex.: resultado `NN5_WEEKLY_DATASET` ↔ origem `nn5_weekly_dataset.tsf`;
`ETTH1` ↔ `ETTH1.tsf` (maiúsculo). Por isso o caminho é passado **explicitamente** no `__main__`:

```python
original_tsf_path = "../forecasting_datasets/nn5_weekly_dataset.tsf"
exec_dataset_orchestrator(
    models, dataset="NN5_WEEKLY_DATASET", use_llm=True,
    series_analyst_model=..., model_critic_model=..., combination_architect_model=...,
    version="v3_pruning", train_window=3,
    original_tsf_path=original_tsf_path,   # ← série original p/ o histórico anti-leakage
)
```

Para trocar de dataset, ajuste `dataset` **e** `original_tsf_path` para o `.tsf` correspondente
(`m4_weekly_dataset.tsf`, `us_births_dataset.tsf`, `ETTH1.tsf`, `ETTH2.tsf`, `ETTM1.tsf`, `ETTM2.tsf`, …).
Se `original_tsf_path=None`, o caminho é resolvido case-insensitive pelo nome do dataset; não achando, cai
no proxy de janelas. Confira a coluna/flag `series_history_source` (`tsf_original` vs `validation_proxy`).

### Decisões resolvidas (eram pendências)
1. Contagem de janelas: corrigida para 3 (`iloc[-(train_window+1):-1]`). ✓
2. Série histórica: **série original completa do `.tsf` truncada em `[:-horizon]`** (105 pts no NN5),
   passada via `original_tsf_path`; fallback ao proxy dos `test` se o `.tsf` faltar. ✓
3. FFORMA/ADE: já existem como CSVs de resultado; a comparação final é externa. Internamente o gate usa
   `pruned_equal_weights`, `full_mean`, `full_median` como baselines reproduzíveis. ✓

### Validação feita (sem LLM, núcleo determinístico)
- `load_original_series_history`/`resolve_tsf_path`: histórico limpo de 105 pts no NN5 (113−8), termina
  exatamente antes da janela de teste (sem vazamento); resolve ETTH1/ETTM2 case-insensitive; fallbacks
  (índice/arquivo ausentes) retornam `None`. ✓
- `features.py`: com 105 pts o flag `history_too_short_for_period` vira `False` → features sazonais ativam
  (proxy de 24 pts mantinha `True`). ✓
- `model_confidence_set`: elimina modelo ruim, mantém empatados (recentragem bootstrap correta). ✓
- `double_shrinkage_per_horizon` + `survivors`: poda aplicada, pesos regularizados. ✓
- `_v3_apply_pruning_floor`: protege modelos do MCS-superior mesmo quando a LLM tenta podá-los. ✓
- Schema por versão: V3 = 48 colunas (sem duplicatas), todas presentes em `data_serie`; V1/V2 preservados. ✓

> O caminho que envolve LLM (Ollama) e `evaluate_all` (que importa `aeon` via `all_functions`) só roda no
> seu ambiente — todos os arquivos passam em `py_compile`. Recomendo rodar primeiro 2–3 séries para
> inspecionar `series_history_source`, `regime`, `survivors`, `fellback_to_pruned_mean` e os deltas antes do
> batch completo.
