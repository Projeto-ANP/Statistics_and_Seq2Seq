# O método, em termos diretos

Documento de referência para escrita do artigo: o que a implementação faz, como
cada peça funciona, e de onde veio cada escolha. As referências estão numeradas
em §9 e citadas ao longo do texto como [1], [2], ...

---

## 1. O problema

Dada uma série temporal e as previsões já produzidas por *N* modelos individuais,
decidir **como combiná-las** para a janela de teste cega.

Não treinamos modelos de previsão. Recebemos as previsões prontas e decidimos a
combinação. É o problema de *forecast combination* clássico [1], com uma restrição
que domina tudo: **só existem 3 janelas de validação** por série.

---

## 2. O pool de modelos (19)

| família | modelos |
|---|---|
| estatísticos | `ARIMA`, `ETS`, `THETA` |
| ML direto | `rf`, `catboost` |
| ML + transformada | `CWT_rf`, `DWT_rf`, `FT_rf`, `CWT_catboost`, `DWT_catboost`, `FT_catboost` |
| ML só na transformada | `ONLY_CWT_rf`, `ONLY_CWT_catboost`, `ONLY_DWT_rf`, `ONLY_DWT_catboost`, `ONLY_FT_rf`, `ONLY_FT_catboost` |
| ingênuos | `NaiveSeasonal`, `NaiveMovingAverage` |

`CWT` = transformada wavelet contínua, `DWT` = discreta, `FT` = Fourier. Os
prefixados com `ONLY_` usam apenas os coeficientes da transformada; os demais
concatenam transformada + série original.

Incluir os ingênuos é deliberado: a literatura de combinação mostra que modelos
fracos mas **diversos** ainda contribuem, porque o ganho da combinação vem da
redução de variância, que cresce quando os erros são pouco correlacionados [1][2].

---

## 3. Protocolo de avaliação

### 3.1 As janelas

Cada série tem 4 janelas de previsão. As 3 primeiras são **validação** (usadas para
decidir), a última é **teste cego** (usada só para pontuar no fim).

### 3.2 Backtest anti-vazamento

Ao pontuar a janela *i*, qualquer coisa ajustada (pesos, seleção de modelos) é
reajustada **sem** a janela *i*. Duas disciplinas distintas, de propósito:

| passo | protocolo | por quê |
|---|---|---|
| ajustar **pesos** | expanding (só janelas anteriores) | é estimativa prospectiva; imita o uso real |
| escolher **quais modelos** | leave-one-out | é seleção de modelo, onde LOO é o padrão |

**Por que isso importa (medido).** Antes de implementar a seleção aninhada, o pool
era escolhido uma vez usando as 3 janelas e depois pontuado nas *mesmas* 3. Isso
tornava o score de validação **anticorrelacionado** com o teste: ranqueando 16
regras fixas nas 111 séries do NN5, Spearman(validação, teste) = **−0.47**. Com a
seleção aninhada, **+0.55**.

---

## 4. O agente

Um agente ReAct [3] — ciclo Thought → Action → Observation — sobre um **catálogo
fechado de 24 ferramentas determinísticas**.

Regra central: **o agente nunca escreve números.** Ele escolhe qual ferramenta
chamar; a ferramenta calcula e devolve um *handle* (`pool1`, `w2`) mais um resumo
qualitativo. Isso elimina por construção a possibilidade de peso ou previsão
alucinados.

Exemplo de um turno real:

```
Thought: ranking is unstable across windows and several models are redundant;
         pruning first should reduce overfitting
Action: prune_redundant
Action Input: {"pool": "pool_full", "corr_threshold": 0.95}
→ observação: pool pool1  19->11 models, dropped ['ARIMA','ETS','THETA','catboost']
```

### 4.1 Catálogo

**Diagnóstico (6):** `series_profile`, `stl_summary`, `error_summary`,
`ranking_stability`, `error_correlation`, `dm_test`

**Seleção de pool (3):** `select_top_k` (menor erro), `select_stable` (menor
`média do rank + desvio do rank`), `prune_redundant` (remove modelos com erros
correlacionados)

**Pesos (5):** `weights_inverse_error`, `weights_softmax_neg_error`,
`weights_error_trend`, `weights_ols`, `weights_pooled_meta_model`

**Combinação (6):** `combine_mean`, `combine_median`, `combine_trimmed_mean`,
`combine_weighted`, `combine_dba`, `combine_best_single`

**Validação (3):** `evaluate_strategy` (único caminho para entrar no histórico),
`sanity_check`, `list_attempts`

### 4.2 Ferramentas retiradas quando não são confiáveis

Uma ferramenta que não pode dar resposta confiável é **removida do catálogo antes
do prompt**, em vez de ser oferecida e falhar.

`weights_ols` (mínimos quadrados de Granger–Ramanathan [4]) exige mais equações
independentes do que 3 janelas fornecem — a projeção no simplex colapsa num
vértice e a "ponderação" vira seleção disfarçada. Com menos de 5 janelas, ela não
aparece para o agente.

---

## 5. O meta-aprendiz cross-series (a peça clássica)

Esta é a parte mais importante do método, e a que exige mais explicação.

### 5.1 O problema que ela resolve

As ferramentas de peso normais leem só as 3 janelas **desta** série. Com 3 pontos,
qualquer peso ajustado degenera para quase-uniforme — o *forecast combination
puzzle* [5][6]. Medido: em 62% das séries do ANP, uma estratégia "ponderada"
produzia previsões **aritmeticamente idênticas à média simples** do próprio pool.

FFORMA [2] e ADE [7] não têm esse problema porque não aprendem por série: eles
treinam **um modelo usando todas as séries do dataset** como amostras de treino.

### 5.2 Como funciona a nossa

Um pré-passo, uma vez por dataset, antes do agente rodar:

1. **Para cada série**, extrai 26 características: força de tendência e
   sazonalidade (via STL [8], fórmula de [9]), entropia espectral,
   autocorrelação lag-1, e as 22 do catch22 [10].
2. **Treina um booster multiclasse** (XGBoost) cujo *softmax da saída* **é** o
   vetor de pesos, usando o gradiente customizado do FFORMA [2] — que minimiza
   diretamente o **erro da combinação resultante**, não o erro de cada modelo
   isolado.
3. **Leave-one-series-out**: o modelo que responde sobre a série *i* nunca viu a
   série *i*. Mesma disciplina do backtest, um nível acima.

Exemplo real (NN5, duas séries com perfis opostos, cada uma consultando sua
própria versão LOSO):

| | série 64 (ruidosa, acf1=0.02) | série 25 (persistente, acf1=0.91) |
|---|---|---|
| top-3 previsto | ARIMA, THETA, ETS | NaiveMovingAverage, ARIMA, ETS |
| pior previsto | NaiveSeasonal, ONLY_FT_catboost | DWT_catboost, DWT_rf |

Numa série persistente o modelo penaliza pesado os baseados em wavelet; numa
ruidosa eles nem entram entre os piores. Esse padrão vem das **outras** séries do
dataset, não das 3 janelas desta.

### 5.3 Por que o objetivo importa mais que as features

Testado com features e folds idênticos, mudando só o objetivo (182 séries do ANP):

| desenho | sMAPE |
|---|---|
| N regressores independentes, cada um prevendo o erro do seu modelo | 0.2205 |
| **1 booster, softmax = pesos, gradiente do erro combinado [2]** | **0.2159** |

O segundo passa o FFORMA de referência (0.2166). **A vantagem do FFORMA está no
objetivo de treino, não nas features** — adicionar as 22 do catch22 teve efeito
estatisticamente nulo (p=0.81 e p=0.97 nos dois datasets).

### 5.4 Uma reconciliação que vale citar

Otimizar diretamente o erro combinado **por série** é a *pior* estratégia que
testamos (busca gulosa: último lugar nos dois datasets). O *mesmo* objetivo,
treinado **pooled entre séries**, é o melhor. A diferença é o tamanho da amostra:
24 pontos versus 182 séries. Essa é a fronteira do *forecast combination puzzle*
[5][6] medida nos nossos dados, e é coerente com o benchmark de 90 mil séries [11]
que conclui que stacking aprendido vence quando há dados suficientes.

---

## 6. O piso determinístico

Antes do agente abrir, a Fase 2 **semeia** estratégias já avaliadas. O resultado
final nunca é pior que a melhor semente, por construção.

Sementes: `mean`, `median`, `dba` sobre o pool completo; `mean` e `trimmed_mean`
sobre os *k* modelos mais estáveis (*k* = 5, 7, 9); e a estratégia do meta-aprendiz.

**Por que `select_stable` e não `select_top_k`:** `top_k` ranqueia pelo mesmo erro
que a estratégia depois é pontuada — ajusta o ruído da validação duas vezes.
`stable` ranqueia por consistência entre janelas, uma estatística diferente. Isso
é a mesma lógica do *trimming* por robustez de [12].

---

## 7. O card do dataset

O agente recebe, a cada turno, como cada estratégia semeada pontuou na validação
das **outras** *N*−1 séries (leave-one-series-out).

```
DATASET CARD (validation only, computed across the other 181 series):
  best_on_this_dataset: [{"strategy": "dba", "mean_validation_score": 0.6612}, ...]
  how_to_use: "a good place to START, not a rule ... your own attempt history
               for THIS series outranks it"
```

É **recomendação, nunca restrição** — o catálogo continua aberto. A motivação é a
mesma da literatura de reuso de experiência em agentes [13], mas aqui a memória é
determinística e auditável em vez de livre.

**Por que existe:** a forense das trajetórias mostrou que o agente usava 4–5 de ~10
ferramentas úteis, ancorado na que o exemplo do prompt citava (462 → 223 → 10 → 2 →
1 usos, decaindo por posição). É o viés de posição em seleção de ferramentas
documentado em [14], medido em produção.

---

## 8. Garantias e instrumentação

| garantia | como |
|---|---|
| a janela de teste nunca influencia a decisão | verificado estruturalmente, por teste comportamental (envenenar a janela cega não muda nada antes da Fase 4) e por varredura dos prompts |
| nenhum nome de modelo alucinado | espaço de ação fechado; nomes fora do catálogo viram erro estruturado |
| nenhum número alucinado | o agente recebe *handles*, nunca valores |
| a série é a certa | o `.tsf` é verificado contra a coluna `test` de cada modelo, série por série |
| procedência | cada linha do CSV registra quais ferramentas foram chamadas, com que argumentos |
| a confiança é medida, não declarada | margem + bootstrap pareado + Diebold–Mariano [15] com correção de amostra pequena [16] |

A confiança auto-reportada pelo agente foi descartada: era **0.9 em 59 de 61
aceites**, uma constante que não sustenta afirmação nenhuma. O substituto é
determinístico.

Comparações finais usam a Multi-Comparison Matrix [17], sobre séries do arquivo
Monash [18].

---

## 9. Referências

[1] Wang, X., Hyndman, R.J., Li, F., Kang, Y. (2023). *Forecast combinations: an
over 50-year review*. International Journal of Forecasting.
https://arxiv.org/abs/2205.04216

[2] Montero-Manso, P., Athanasopoulos, G., Hyndman, R.J., Talagala, T.S. (2020).
*FFORMA: Feature-based forecast model averaging*. International Journal of
Forecasting 36(1). — **origem do objetivo de treino do meta-aprendiz (§5)**

[3] Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language
Models*. ICLR. https://arxiv.org/abs/2210.03629 — **paradigma do agente (§4)**

[4] Granger, C.W.J., Ramanathan, R. (1984). *Improved methods of combining
forecasts*. Journal of Forecasting 3(2). — **`weights_ols` (§4.2)**

[5] Claeskens, G., Magnus, J.R., Vasnev, A.L., Wang, W. (2016). *The forecast
combination puzzle: A simple theoretical explanation*. International Journal of
Forecasting 32(3). — **por que pesos ajustados degeneram (§5.1)**

[6] Smith, J., Wallis, K.F. (2009). *A simple explanation of the forecast
combination puzzle*. Oxford Bulletin of Economics and Statistics 71(3).

[7] Cerqueira, V., Torgo, L., Pinto, F., Soares, C. (2017/2019). *Arbitrated
Ensemble for Time Series Forecasting* / *Arbitrage of forecasting experts*.
Machine Learning. — **baseline ADE**

[8] Cleveland, R.B., Cleveland, W.S., McRae, J.E., Terpenning, I. (1990).
*STL: A seasonal-trend decomposition procedure based on loess*. Journal of
Official Statistics 6(1). — **decomposição (§5.2)**

[9] Wang, X., Smith, K.A., Hyndman, R.J. (2006). *Characteristic-based clustering
for time series data*. Data Mining and Knowledge Discovery 13(3). — **fórmulas de
força de tendência e sazonalidade**

[10] Lubba, C.H. et al. (2019). *catch22: CAnonical Time-series CHaracteristics*.
Data Mining and Knowledge Discovery 33. — **22 das 26 features (§5.2)**

[11] *Multi-layer Stack Ensembles for Time Series Forecasting* (2025), benchmark
de 33 métodos de combinação em 50 datasets / 90 mil séries.
https://arxiv.org/abs/2511.15350 — **contraponto: stacking aprendido vence com
dados suficientes (§5.4)**

[12] Kourentzes, N., Barrow, D., Petropoulos, F. (2019). *Another look at forecast
selection and combination: Evidence from forecast pooling*. International Journal
of Production Economics 209. — **seleção de subconjunto por robustez (§6)**

[13] Zhao, A. et al. (2024). *ExpeL: LLM Agents Are Experiential Learners*. AAAI.
https://arxiv.org/abs/2308.10144 — **reuso de experiência entre tarefas (§7)**

[14] *BiasBusters: position bias in LLM tool selection* (2025).
https://arxiv.org/abs/2605.18857 — **viés de posição na escolha de ferramentas (§7)**

[15] Diebold, F.X., Mariano, R.S. (1995). *Comparing predictive accuracy*. Journal
of Business & Economic Statistics 13(3). — **teste de significância (§8)**

[16] Harvey, D., Leybourne, S., Newbold, P. (1997). *Testing the equality of
prediction mean squared errors*. International Journal of Forecasting 13(2). —
**correção de amostra pequena para o DM**

[17] Ismail-Fawaz, A. et al. (2023). *An Approach to Multiple Comparison Benchmark
Evaluations that is Stable Under Manipulation of the Comparate Set*.
https://arxiv.org/abs/2305.11921 — **a MCM usada nas comparações**

[18] Godahewa, R., Bergmeir, C., Webb, G.I., Hyndman, R.J., Montero-Manso, P.
(2021). *Monash Time Series Forecasting Archive*. NeurIPS Datasets and Benchmarks.
— **formato `.tsf` e os datasets NN5/M4**

[19] Petitjean, F., Ketterlin, A., Gançarski, P. (2011). *A global averaging method
for dynamic time warping, with applications to clustering*. Pattern Recognition
44(3). — **`combine_dba`**

[20] Li, W. et al. (2025). *Rethinking Mixture-of-Agents: Is Mixing Different LLMs
Beneficial?* (Self-MoA). https://arxiv.org/abs/2502.00674 — **pool menor generaliza
melhor; confirmado independentemente nos nossos dados**
