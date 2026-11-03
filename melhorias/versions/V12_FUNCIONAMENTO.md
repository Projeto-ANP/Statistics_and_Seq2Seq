# V12 — funcionamento de ponta a ponta

Documento da versão **v12** (`topk_guard_margin`): o loop ReAct do v10/v11 é
mantido; o agente pode construir pools e pesos **top-k por horizonte**; a decisão
final é o **seasonal guard determinístico** com margem mínima de `0.10`.

Status dos resultados:

- `gpt-oss:20b` **low** e **medium** concluídos nos 7 datasets;
- `qwen3.8:27b` medium ainda em execução (a pasta local tem apenas ANP);
- `gemma4:26b` on ainda não concluído.

Este documento descreve o algoritmo, não só a run do gpt-oss. O teste real nunca
entra no prompt nem na decisão; ele só é lido depois que a previsão final já foi
produzida.

---

## 1. Definição da v12

A v12 é a promoção da configuração v11 `topk_guard_margin`. Ela combina três
peças:

1. **`--direct-preparation`**: além de `evaluate_strategy`, o agente pode usar
   `select_*` e `weights_*` como preparação. Essas chamadas não consomem o
   orçamento de avaliações (mas contam no limite de turnos).
2. **Top-k por horizonte**: um pool com os `k` melhores modelos por SMAPE e um
   peso por horizonte que dá peso igual aos `k` melhores modelos **em cada passo
   do horizonte**. É a receita que atacou o ETTM1/ETTM2.
3. **`seasonal_guard` + `--seasonal-guard-min-gap 0.10`**: depois do loop, a
   decisão final é determinística. O líder de validação só é trocado por um
   candidato de outra família se o candidato estiver pelo menos `0.10` (em
   unidades de escala recente) mais próximo do **seasonal naive**, e somente se
   o gate sazonal estiver ativo.

A v12 **não obriga** o agente a usar top-k. O agente testa as combinações que
julgar adequadas; o guard final escolhe entre os melhores membros de cada
família. Na prática, a receita top-k aparece nos ETTs e o guard a seleciona pela
forma sazonal.

### 1.1 Configuração oficial

Flags do runner (`run_v12_all.sh`):

```text
--prompt-v2
--catalog-v2
--no-early-stop
--scoring-budget
--direct-preparation
--max-iterations 12
--max-total-turns 24
--temperature 0.7
--score-preset smape_only
--score-window-mode weighted
--no-agent-score-preset
--no-seed-attempts
--no-best-single
--llm-seed 11
--final-strategy seasonal_guard
--final-selector-max-candidates 10
--seasonal-guard-min-gap 0.10
```

Modelos/configurações da run completa:

| versão | modelo | reasoning |
|---|---|---|
| `v12_gptoss_low` | `gpt-oss:20b` | low |
| `v12_gptoss_medium` | `gpt-oss:20b` | medium |
| `v12_qwen_medium` | `qwen3.8:27b` | medium |
| `v12_gemma_on` | `gemma4:26b` | on |

Cada configuração roda os 7 datasets: ANP, ETTH1, ETTH2, ETTM1, ETTM2, NN5 e M4.
São 4 × 7 = 28 execuções de dataset no total.

---

## 2. Entrada: previsões dos N modelos e construção das janelas

### 2.1 Dados de entrada

Para cada série e cada modelo base, o harness lê um CSV em:

```text
timeseries/mestrado/resultados/<MODELO>/normal/<DATASET>.csv
```

Cada linha é uma janela de previsão e contém, entre outras colunas:

- `dataset_index`: identifica a série;
- `start_test` / `final_test`: início e fim da janela;
- `predictions`: vetor de previsões do modelo para a janela;
- `test`: vetor de valores reais da janela.

O `.tsf` original fica em `../forecasting_datasets`. Ele é usado para o perfil da
série e para verificar o alinhamento das previsões.

### 2.2 Os N modelos base

O pool padrão tem **19 modelos**:

```text
ARIMA, ETS, THETA, rf, catboost,
CWT_rf, DWT_rf, FT_rf,
CWT_catboost, DWT_catboost, FT_catboost,
ONLY_CWT_catboost, ONLY_CWT_rf,
ONLY_DWT_catboost, ONLY_DWT_rf,
ONLY_FT_catboost, ONLY_FT_rf,
NaiveSeasonal, NaiveMovingAverage
```

Nos ETTM1/ETTM2, cinco modelos `ONLY_*` são descartados porque foram gerados em
uma janela diferente (alvos que não são os mesmos números dos outros 14). O pool
do ETTM fica com **14 modelos**. O descarte é registrado por série; ele não é
silencioso.

| dataset | séries | horizonte H |
|---|---:|---:|
| ANP_MONTHLY | 182 | 12 |
| ETTH1 | 7 | 24 |
| ETTH2 | 7 | 24 |
| ETTM1 | 7 | 24 |
| ETTM2 | 7 | 24 |
| NN5_WEEKLY_DATASET | 111 | 8 |
| M4_WEEKLY_DATASET | 359 | 13 |
| **total** | **680** | — |

O horizonte `H` é o menor comprimento comum entre as previsões e os alvos das
janelas; se houver comprimentos diferentes, todas as séries são truncadas em
`H` e um warning é registrado.

### 2.3 Três janelas de validação + a janela de teste

Cada modelo precisa ter pelo menos `n_windows + 1 = 4` linhas para a série. As
linhas são ordenadas por `start_test`:

```text
posição -4  -> validação 0 (mais antiga)
posição -3  -> validação 1
posição -2  -> validação 2 (mais recente)
posição -1  -> teste (cega)
```

Matrizes montadas:

```text
y_true      : (3, H)       # valores reais das 3 janelas de validação
y_preds     : (3, N, H)    # previsões dos N modelos nas 3 janelas
test_preds  : (N, H)       # previsões dos N modelos na janela de teste
train_series: (T, )        # série histórica sem o período de teste
```

O `y_true` de teste **não é armazenado no `ReactState`**. O harness só tem
`test_preds`; o alvo de teste só é lido na fase de avaliação final, depois que a
previsão já foi gerada.

### 2.4 Alinhamento e segurança

A ingestão faz quatro verificações:

1. todos os modelos têm a mesma quantidade de janelas para a série;
2. as janelas de validação têm os mesmos valores reais entre os modelos
   (`np.allclose`, `rtol=1e-4`, `atol=1e-4`); se não tiverem, a série falha com
   `IngestionError` — não se combinam previsões de janelas diferentes;
3. a janela de teste também tem o mesmo alvo entre os modelos;
4. o `.tsf` é conferido contra a cauda dos valores reais. Se o `.tsf` estiver em
   resolução mais fina (caso ETT), ele pode ser reamostrado por média para casar
   com a resolução das previsões.

`train_series = full[:len(full) - H]`, que é o mesmo corte usado na geração das
previsões. Não há contaminação do teste.

### 2.5 Por que 3 janelas

O protocolo usa **3 janelas de validação** porque é o que os CSVs dos modelos
oferecem antes da janela cega. O custo é uma estimativa de score com alta
variância; por isso:

- o score usa pesos maiores na janela mais recente;
- a confiança da seleção é reportada;
- a decisão final v12 não usa o score sozinho; usa também a forma sazonal.

---

## 3. Visão geral do pipeline

```text
.tsf + previsões dos N modelos
            |
            v
Fase 0  ingestão, alinhamento e montagem de y_true/y_preds/test_preds
            |
            v
Fase 1  perfil da série + pool card + diagnóstico determinístico
            |
            v
Fase 2  pool de trabalho (full) e handles de pools estáveis
        v12: nenhuma tentativa semeada; o agente começa sem líder
            |
            v
Fase 3  agente ReAct
        - evaluates pontuados na validação
        - select_* e weights_* como preparação
        - top-k por horizonte é uma das receitas
            |
            v
Fase 3.5 seasonal guard determinístico
        - líder = menor score de validação
        - shortlist por família
        - gate sazonal
        - troca só se a distância ao seasonal naive melhorar >= 0.10
            |
            v
Fase 4  aplica a combinação escolhida ao test_preds, mede no teste
            |
            v
Fase 5  CSV, artefatos, relatório opcional
```

A ordem é importante: **o teste real não participa de nenhuma fase antes da
Fase 4**.

---

## 4. Fase 0 — ingestão em detalhe

Para cada série `(dataset, dataset_index)`:

1. lê os CSVs de todos os modelos do pool;
2. ordena as linhas por `start_test`;
3. exige `n_windows + 1` linhas; usa as 3 primeiras como validação e a última
   como teste;
4. verifica se os valores reais das janelas coincidem entre modelos;
5. verifica o `.tsf` e monta `train_series`;
6. remove modelos desalinhados do dataset, quando configurado
   (`drop_misaligned_models=True`);
7. constrói o `ReactState`.

O `ReactState` é o dono de:

- `y_true`, `y_preds`, `test_preds`, `model_names`;
- pools (`pool_full`, `pool1`, ...);
- receitas de pesos (`w1`, `w2`, ...);
- histórico de tentativas (`a1`, `a2`, ...);
- protocolo de backtest e de aplicação no teste.

---

## 5. Fase 1 — perfil da série

O `series_profile` é calculado sobre `train_series` (sem o período de teste). Ele
produz o `series_card` usado no prompt e pelo guard final.

### 5.1 Período sazonal

O período vem de:

1. `--seasonal-period`, se explícito;
2. da frequência do `.tsf` (`FREQ_TO_PERIOD`);
3. fallback para o horizonte, limitado a 12.

O perfil registra também:

- `seasonal_period_declared`: o que a frequência implica;
- `seasonal_period_source`: `explicit`, `frequency` ou `horizon_fallback`;
- `seasonal_period_fits`: se há pelo menos 2 ciclos completos.

O guard v12 usa `seasonal_period` e `n_points`; ele exige **10 ciclos completos**
para agir.

### 5.2 STL e forças de tendência/sazonalidade

Se a série tem pelo menos `2 * period` pontos, roda STL robusto
(`statsmodels.tsa.seasonal.STL`). Caso contrário, faz fallback de tendência
linear e sazonalidade zero.

As forças são as de Wang, Smith & Hyndman (2006):

```text
trend_strength    = clip(1 - Var(resid) / Var(trend + resid), 0, 1)
seasonal_strength = clip(1 - Var(resid) / Var(seasonal + resid), 0, 1)
```

`seasonal_strength` é o número que o gate do guard compara com `0.5`.

### 5.3 Estacionariedade, outliers e features

Além de tendência/sazonalidade, o perfil inclui:

- ADF e KPSS (p-valores e veredito `stationary` / `nonstationary` / `ambiguous`);
- outliers por z-score (quantidade, percentual, z máximo);
- `acf1`, `acf_seasonal`, entropia espectral, Hurst, coeficiente de variação,
  escala de flutuação, se cruza zero;
- `n_points`, frequência, horizonte.

O agente vê uma versão resumida no prompt; o guard usa apenas os campos
necessários.

---

## 6. Fase 1/2 — pool card e diagnóstico

### 6.1 Pool card

O `pool_report` contém:

- tabela de erro por modelo (top-N);
- melhor modelo por janela;
- estabilidade do ranking entre janelas (`Kendall tau` médio e veredito);
- grupos de erro correlacionado (threshold `0.9`);
- número de modelos e de janelas.

Não é a matriz completa de previsões: é um resumo para o agente.

### 6.2 Diagnóstico determinístico

Sem LLM diagnosticador, o diagnóstico produz:

- `regime`: `trend`, `seasonal`, `mixed` ou `noisy`;
- `predictability`: `high`, `medium` ou `low`;
- `combination_hint`: `robust`, `weighted`, `selective` ou `full_pool`;
- riscos e uma narrativa curta.

O diagnóstico é uma leitura, não uma regra: o agente pode testar qualquer
combinação válida.

### 6.3 Fase 2 na v12

A v12 usa:

```text
pool_mode           = full
no_seed_attempts    = True
no_best_single      = True
seed_stable_pools   = True (mas só registra handles; não pontua)
```

Consequências:

- o pool de trabalho é o `pool_full` (19 modelos, ou 14 nos ETTM);
- **não há tentativas semeadas** (mean, median, dba, pools estáveis etc.);
- pools estáveis da Phase 2 existem como handles, mas não são mostrados nem
  pontuados; o agente não começa ancorado neles;
- o histórico começa vazio; a primeira `evaluate_strategy` é o piso;
- o líder final é o melhor **agente**, não um baseline determinístico;
- `best_single` está desabilitado: toda combinação usa pelo menos 2 modelos;
- `score_preset` é fixo em `smape_only`;
- o agente não pode trocar o critério de score.

Essa escolha foi feita porque, na v11, o agente ancorava em pools estáveis da
Phase 2 e deixava de testar a receita top-k. A v12 esconde esses handles e exige
que o agente construa o que quiser.

---

## 7. Fase 3 — o loop ReAct

### 7.1 Modelo e amostragem

```text
combinator   = gpt-oss:20b | qwen3.8:27b | gemma4:26b
reasoning    = low | medium | on
temperature  = 0.7
llm_seed     = 11
prompt       = v2
catalog      = v2
```

O seed vai em `options.seed` do Ollama. Em retry de resposta vazia ou erro
transitório, o cliente aumenta o seed e a temperatura (até 0.9) para tentar sair
do mesmo ponto. O `retry_events` registra cada tentativa.

### 7.2 Orçamento

```text
max_iterations   = 12   # número de evaluate_strategy pontuados
max_total_turns  = 24   # teto de turnos de ferramenta, incluindo preparação
no_early_stop    = True
scoring_budget   = True
direct_preparation = True
```

- `evaluate_strategy` conta como uma tentativa pontuada;
- `select_*`, `weights_*`, `prune_*` são preparação e **não** consomem o
  orçamento de avaliações;
- o teto de 24 turnos vale para tudo;
- sem `accept` e sem parada antecipada: o agente usa o orçamento até acabar;
- `best_single` não está disponível.

### 7.3 O que o agente vê a cada turno

O prompt v2 entrega:

- `SERIES PROFILE`: perfil resumido;
- `MODEL POOL`: pool card;
- `DIAGNOSIS`: leitura determinística;
- `ATTEMPT HISTORY`: tentativas pontuadas, ordenadas por score;
- `LEADER DIAGNOSTICS`: líder, vice-líder, spread e veredito;
- `HANDLES`: pools e pesos criados pelo próprio agente;
- `WEIGHT METHODS`: métodos disponíveis;
- `NEXT UNTRIED`: sugestões do que ainda não foi tentado;
- `SCRATCHPAD` e `LAST OBSERVATION`;
- `FORECAST SHAPE`: forma das previsões de validação, sem o alvo de teste;
- `SERIES DECISION CARD`: características que devem motivar o próximo teste;
- orçamento restante.

O `FORECAST SHAPE` mostra a forma das previsões; o valor real da janela de teste
**não** aparece.

### 7.4 Formato da resposta

```text
Thought: ...
Action: <tool>
Action Input: {...}
```

Erros de parsing, nome de ferramenta inválido ou argumento inválido viram uma
observação estruturada para o agente corrigir; não matam a série, salvo quando o
loop não consegue mais avançar.

### 7.5 Catálogo efetivo

O catálogo é fechado. Com `catalog_v2` e `direct_preparation`, a v12 expõe o
essencial:

- **avaliação**: `evaluate_strategy`;
- **pool**: `select_top_k`, `select_stable`, `select_forward`,
  `prune_redundant`;
- **pesos**: `weights_inverse_error`, `weights_error_trend`, `weights_ols`
  (bloqueado quando há menos de 5 janelas), `weights_best_per_horizon`,
  `weights_topk_per_horizon`, `weights_forward`;
- **meta-modelo**: `weights_meta_model`, quando aplicável;
- `combine_*` e diagnósticos redundantes ficam escondidos no `catalog_v2` porque
  a mesma informação já está no contexto do turno ou pode ser passada diretamente
  para `evaluate_strategy`.

A v12 **não** usa o seletor final por LLM; o catálogo é só para o loop de
combinação.

### 7.6 Receita top-k por horizonte

A receita canônica que a v12 quer tornar alcançável é:

```text
select_top_k(k=7, metric="smape")
  -> handle de pool com os 7 modelos de menor SMAPE de validação

weights_topk_per_horizon(pool=<handle>, k=5, metric="smape")
  -> handle de pesos (N_pool, H): em cada passo h, peso 1/k para os k
     modelos de menor SMAPE naquele passo, 0 para os demais

evaluate_strategy(combine="weighted", pool=<handle>, weights=<weights_handle>)
  -> tenta a combinação e recebe o score de validação
```

O exemplo do prompt usa `k=7` e `k=5`, mas o agente decide os valores. O que a
v12 garante é que a família `weighted:topk_mean_per_horizon` exista no shortlist
final se o agente a tiver criado — o `family_key` separa `weighted` pelo método
de pesos, então ela não é engolida por um `weighted` de outro tipo.

O `per_horizon_tool_chain` do prompt só aparece quando o perfil indica série
sazonal com pelo menos 10 ciclos; é uma dica, não uma ordem.

### 7.7 Como uma tentativa é registrada

`evaluate_strategy` normaliza o spec, executa o backtest, calcula métricas e
score, e cria um `Attempt` com:

- `attempt_id` (`a1`, `a2`, ...);
- `spec` normalizado: `combine`, `pool`, `weights`, `trim_pct`, `model`, etc.;
- `origin = "agent"`;
- `aggregate`: métricas agregadas da validação;
- `per_window`: métricas de cada janela;
- `score`: score de validação;
- `per_window_scores`: score composto por janela;
- `residuals`: erros concatenados, para DM/bootstrap;
- `n_models`;
- `rationale` e `iteration`.

Specs numericamente idênticos são deduplicados; reavaliar a mesma coisa não cria
uma nova tentativa. O agente recebe `already_tested`.

---

## 8. Cálculo de uma tentativa

### 8.1 Especificação

Um spec válido é um objeto como:

```json
{"combine": "weighted", "pool": "pool4", "weights": "w3"}
```

ou:

```json
{"combine": "median", "pool": "pool_full"}
```

Métodos de combinação:

| método | cálculo |
|---|---|
| `mean` | média dos modelos no pool, por horizonte |
| `median` | mediana dos modelos no pool, por horizonte |
| `trimmed_mean` | ordena por horizonte, remove `floor(N*trim_pct)` de cada cauda, média do resto |
| `weighted` | média ponderada; pesos `(N,)` ou `(N,H)` |
| `dba` | DTW Barycenter Averaging; fallback para a média |
| `best_single` | um único modelo; **desabilitado na v12** |

O `trimmed_mean` usa `trim_pct=0.2` por padrão. Se `2*floor(N*trim_pct) >= N`,
cai para a média. O `weighted` redistribui o peso de modelos com previsão NaN no
horizonte e, se nenhum peso sobrar, cai para a média dos modelos disponíveis.

O DBA usa `tslearn.barycenters.dtw_barycenter_averaging`, com semente fixa
(`random_state=7`) para não variar entre séries idênticas. Se o `tslearn` faltar
ou falhar, cai para a média.

### 8.2 Pools

Handles de pool:

- `pool_full`: todos os modelos do dataset (19 ou 14);
- `poolN`: pool criado por `select_top_k`, `select_stable`,
  `select_forward` ou `prune_redundant`;
- lista explícita de modelos passada em `evaluate_strategy.models`.

`select_top_k(k, metric)`:

```text
para cada modelo m:
    erro_m = erro(metric) nas janelas de fit
ordena m por erro_m
pool = os k menores
```

`select_stable(k, metric)`:

```text
para cada janela de fit:
    rank_m = posição do modelo m pelo erro
score_m = mean(rank_m) + std(rank_m)
pool = os k menores scores
```

O pool de um fold pode ser **re-escolhido** (nested selection). Para avaliar a
janela `w`, a receita de pool é reajustada nas janelas `!= w` (leave-one-out),
não em todas as janelas. Isso evita que a janela pontuada tenha participado da
escolha dos modelos que estão sendo pontuados nela.

### 8.3 Pesos

Receitas de peso implementadas:

| método | cálculo |
|---|---|
| `inverse_error` | `w_m ∝ 1 / (erro_m + 1e-8)^power`, normalizado; opcionalmente encolhido para uniforme |
| `softmax_neg_error` | `softmax(-eta * erro_m / median(erro))`; `eta=1` por padrão |
| `error_trend` | peso pela tendência do erro entre janelas |
| `ols` | mínimos quadrados (ridge opcional), projetado no simplex; indisponível com 3 janelas |
| `feature_based` | modelo por série sobre features de validação; exige amostras suficientes |
| `pooled_meta_model` | modelo de erro treinado nas outras séries do dataset (leave-one-series-out) |
| `best_per_horizon` | one-hot para o melhor modelo em cada passo do horizonte |
| `topk_mean_per_horizon` | peso igual para os `k` melhores modelos em cada passo |
| `forward_selection` | Caruana: contagem de seleções normalizada |

O ponto importante da v12 é o **`topk_mean_per_horizon`**:

```text
para cada passo h do horizonte:
    erro_{m,h} = erro(metric) do modelo m no passo h, nas janelas de fit
    keep = os k menores erro_{m,h}
    w_{m,h} = 1/k se m ∈ keep, senão 0
```

Isso permite que modelos diferentes sejam usados em passos diferentes do
horizonte. O resultado é uma matriz `(N_pool, H)`, que o `combine_weighted`
aplica diretamente.

### 8.4 O protocolo de backtest: expanding + nested selection

Para cada janela de validação `w`:

1. **membros do pool**:
   - se o pool tem receita re-ajustável (`top_k`, `stable`, `forward`,
     `prune_redundant`) e `nested_selection=True`, os membros usados para
     pontuar `w` são escolhidos nas janelas `!= w` (leave-one-out);
   - se o pool é uma lista explícita ou não tem receita, ele é constante.
2. **pesos**:
   - a receita de pesos é ajustada nas janelas **anteriores** a `w`
     (`expanding`): `w=0` não tem janela anterior → pesos uniformes;
     `w=1` usa a janela 0; `w=2` usa as janelas 0 e 1.
3. **combinação**: aplica a combinação ao pool daquele fold e obtém
   `p_w = (H,)`.

No teste (`apply_to_test`):

- o pool é o handle completo (todos os membros escolhidos nas 3 janelas);
- os pesos são ajustados nas 3 janelas de validação (não há janela alvo para
  excluir);
- a combinação é aplicada a `test_preds`;
- o alvo de teste não é consultado nesse momento.

Essa assimetria é deliberada:

- **pool** é seleção de modelos, então usa leave-one-out para não escolher e
  pontuar na mesma janela;
- **pesos** são uma estimativa prospectiva, então usam apenas o passado
  (`expanding`), imitando o deploy.

### 8.5 Métricas por janela

Com previsão `p` e real `y`, ambos vetores de tamanho `H`:

```text
MAPE   = mean( |p_h - y_h| / |y_h| )        para |y_h| > 1e-8
SMAPE  = mean( 2|p_h - y_h| / (|p_h| + |y_h|) )
         pontos com |p_h|+|y_h| = 0 são ignorados
RMSE   = sqrt( mean( (p_h - y_h)^2 ) )
MAE    = mean( |p_h - y_h| )
MSMAPE = mean( 2|p_h - y_h| / max(0.6, |p_h| + |y_h| + 0.1) )
POCID  = 100 * #{ h : Δp_h * Δy_h > 0 } / (H - 1)
         Δp_h = p_h - p_{h-1}; Δy_h = y_h - y_{h-1}
```

Convenções:

- MAPE é fração, não percentual;
- SMAPE fica em `[0, 2]`;
- POCID fica em `[0, 100]` e **maior é melhor**; empates não contam;
- MAPE ignora `y=0` (`mape_zero="skip"`).

### 8.6 Métricas agregadas da validação

O `Attempt.aggregate` é calculado assim:

- MAPE, SMAPE, RMSE, MSMAPE e MAE: concatenam-se as 3 janelas
  (`y_true.reshape(-1)` e `combined.reshape(-1)`) e calcula-se a métrica uma vez;
- POCID: é calculado **dentro de cada janela** e depois é feita a média das 3
  janelas. POCID concatenado poderia inventar uma transição entre o fim de uma
  janela e o início da seguinte; a média por janela evita isso.

O POCID agregado da validação não entra no score (ver abaixo), mas é reportado.

### 8.7 Score de validação

A v12 usa:

```text
score_preset      = smape_only
score_window_mode = weighted
```

Âncora: em cada janela, o SMAPE do **mean do pool completo** (`y_preds` de todos
os modelos, média por horizonte; sem seleção).

Score composto em uma janela:

```text
score_w = SMAPE_tentativa_w / SMAPE_mean_pool_w
```

Como `smape_only` tem `b_smape = 1` e os outros pesos zero, o `composite_score`
é exatamente essa razão. Menor é melhor.

Score final da tentativa:

```text
score = 0.2 * score_0 + 0.3 * score_1 + 0.5 * score_2
```

A janela 2 (mais recente, mais próxima do teste) tem peso 0.5. O `score` não é o
SMAPE bruto; o SMAPE bruto é o que aparece no CSV e nos resumos.

POCID **não** entra no score da v12. Ele é reportado separadamente; o agente não
pode usar POCID para ranquear. Isso é explícito no prompt: "POCID is reported
separately and does not affect the ranking."

### 8.8 Confiança da seleção

Depois do loop, `selection_confidence` compara o líder com o vice-líder:

```text
margin = (score_vice - score_lider) / |score_lider|
bootstrap pareado sobre os 3 scores por janela
Diebold-Mariano sobre os resíduos concatenados (HLN)
alpha = 0.10
verdict = separated | indistinguishable | weak
```

Com 3 janelas, o bootstrap é degenerado (poucos valores possíveis); o veredito
segue o DM. O `bootstrap_reliable` fica `False` quando há menos de 5 janelas.
Essas estatísticas são **registradas** no CSV/artefato, mas a v12 não as usa para
decidir: a decisão é do guard sazonal.

---

## 9. Fase 3.5 — decisão final: seasonal guard

A decisão final é separada do loop. O objetivo é corrigir o caso em que o líder
de validação tem um score bom, mas a **forma** da previsão não é plausível para
uma série sazonal.

### 9.1 Líder

```text
leader = argmin_a score(a)
```

O líder é a tentativa de menor score de validação entre todas as tentativas do
agente. O teste não é consultado.

### 9.2 Shortlist por família

`build_family_shortlist`:

1. ordena as tentativas por score crescente;
2. agrupa por família:
   - `mean`, `median`, `trimmed_mean`, `dba`;
   - `weighted` é subdividido por método de pesos
     (`weighted:inverse_error`, `weighted:topk_mean_per_horizon`, ...);
3. para cada família, pega o primeiro membro **plausível**;
4. um membro é plausível se:
   - a previsão é finita e não vazia;
   - `max |previsão| <= 100 * max(|último real|, escala, 1)`;
   - não é duplicata de um membro já aceito (correlação `> 0.9999` **e** forma
     padronizada igual; quase igualdade exata também é descartada);
5. até `final_selector_max_candidates = 10` famílias.

O líder global é o melhor da sua própria família, então normalmente entra na
shortlist. Se o líder for degenerado e ficar de fora, ele é substituído pelo
primeiro membro plausível da shortlist (o "líder efetivo").

A shortlist é construída com as previsões de **teste** dos modelos base
(`apply_to_test`), mas apenas a forma; nenhum valor real de teste é lido.

### 9.3 Gate sazonal

O gate só ativa o guard quando o seasonal naive é estimável:

```text
period >= 2
seasonal_strength >= 0.5
n_points >= 10 * period
```

Se qualquer condição falhar, o guard **mantém o líder**. O gate é conservador:
não basta sazonalidade forte se a história tem poucos ciclos (NN5), e não basta
história longa se a série não é sazonal (ANP).

### 9.4 Seasonal naive e distância

Para uma série com período `p` e histórico `hist`:

```text
seasonal_naive = resize(hist[-p:], H)
```

Isto é, repete o último ciclo completo até o horizonte `H`.

A escala usada na normalização é o desvio-padrão dos últimos `look` pontos do
histórico:

```text
look = max(2, min(len(hist), max(2*p, 2*H, 2)))
scale = std(hist[-look:])
```

Se a escala for inválida ou zero, usa o desvio-padrão de todo o histórico; se
ainda for inválido, usa `max(|último valor|, 1)`.

Para cada candidato da shortlist, a distância ao seasonal naive é:

```text
seasonal_dist = mean_h( |forecast_h - seasonal_naive_h| ) / scale
```

Menor é melhor. É uma distância L1 normalizada pela escala recente; não usa o
alvo de teste.

### 9.5 Regra de decisão

Seja:

```text
best  = candidato com menor seasonal_dist
leader_dist = seasonal_dist do líder
best_dist   = seasonal_dist do best
gap = leader_dist - best_dist
```

Decisões, nesta ordem:

1. gate inativo → `keep` (mantém líder);
2. shortlist vazia → fallback;
3. `best` é o líder → `keep`;
4. `gap <= 0.10` → `keep` (a melhora é pequena; não vale trocar);
5. caso contrário → `switch` para `best`.

O limiar é `seasonal_guard_min_gap = 0.10`. Como a distância é dividida por
`scale`, o gap é em unidades de desvio-padrão recente: trocar exige que o
candidato esteja **10% de um desvio-padrão** mais perto do seasonal naive do que
o líder.

Pontos importantes:

- o guard pode escolher um candidato com **score de validação pior** que o
  líder, se a forma sazonal for muito melhor; a validação não é o único critério
  na decisão final;
- o guard não usa LLM;
- o guard não olha o teste real;
- o gap de 0.10 foi motivado pelo caso do ETTM2 série 5, em que um gap de 0.0507
  fez o guard trocar um bom `trimmed_mean` por um top-k pior. Com 0.10, essa
  troca não acontece.

### 9.6 Pseudocódigo do guard

```python
leader = min(state.attempts, key=score)

shortlist = []
for family in ordered_families(state.attempts):
    for attempt in family_by_score(family):
        forecast = apply_to_test(attempt.spec)
        if not finite(forecast): continue
        if max(abs(forecast)) > 100 * max(abs(last), scale, 1): continue
        if duplicate(forecast, shortlist): continue
        shortlist.append(attempt)
        break
    if len(shortlist) == 10: break

if not shortlist: return leader, "fallback"
if leader not in shortlist: leader = shortlist[0]

gate = seasonal_strength >= 0.5 and period >= 2 and n_points >= 10*period
if not gate: return leader, "keep:gate_inactive"

distances = {a: mean(abs(forecast(a) - seasonal_naive)) / scale
             for a in shortlist}
best = min(distances, key=distances.get)

if best is leader: return leader, "keep:leader_is_seasonal_best"
gap = distances[leader] - distances[best]
if gap <= 0.10: return leader, "keep:seasonal_gap_below_min"
return best, "switch:seasonal_switch"
```

### 9.7 Debug e justificativa

O artefato `final_choice` registra:

- `mode = "seasonal_guard"`;
- `leader_id`, `leader_spec`, `leader_validation_score`;
- `shortlist` com `attempt_id`, `spec`, `validation_score`, `seasonal_dist`,
  features;
- `gate` (active/reason/strength/n_points/period);
- `seasonal_id`, `seasonal_dist`, `leader_seasonal_dist`, `seasonal_gap`;
- `decision`: `keep`, `switch` ou `fallback`;
- `reason`: `gate_inactive`, `leader_is_seasonal_best`,
  `seasonal_gap_below_min`, `seasonal_switch`, etc.;
- `chosen_id`, `chosen_spec`, `chosen_validation_score`;
- `justification` textual.

A justificativa é copiada para o campo de texto do CSV; o prompt/resposta
completo do seletor por LLM não existe na v12 (o guard é determinístico).

---

## 10. Fase 4 — aplicação no teste e resultado final

### 10.1 Aplicação

Com a tentativa escolhida (`final_attempt`):

```text
forecast_test, debug = state.apply_to_test(final_attempt.spec)
```

Para cada método:

- `mean`, `median`, `trimmed_mean`, `dba`: aplica a mesma combinação aos
  `test_preds` do pool;
- `weighted`: reajusta os pesos nas 3 janelas de validação e aplica aos
  `test_preds` do pool; se o peso é por horizonte, usa a matriz `(N, H)`;
- `best_single`: indisponível na v12.

O teste real não entra nesse cálculo. A previsão final é um vetor `(H,)`.

### 10.2 Métricas de teste

Depois da previsão, o harness compara com o alvo de teste e calcula:

```text
MAPE, SMAPE, RMSE, POCID, MSMAPE, MAE
```

O POCID de teste é o POCID da única janela de teste. O SMAPE de teste é o SMAPE
bruto (não normalizado), que é o número reportado nas tabelas.

### 10.3 CSV e artefatos

Cada série gera uma linha no CSV do dataset com:

- métricas de teste;
- `test`, `predictions`, `start_test`, `final_test`;
- `decision_report`, `justification`;
- `final_candidate_names`, `final_candidate_count`;
- `best_strategy_name`, `best_strategy_method`, `best_strategy_params`;
- `final_origin` (na v12, `agent`);
- `weights_by_horizon`, `selected_base_models`, `n_pool_models`;
- `react_iterations_used`, `react_trajectory_json`;
- `selection_margin`, `selection_bootstrap_pvalue`, `selection_dm_pvalue`,
  `selection_verdict`;
- `agent_turns`, `agent_llm_s`, `agent_tool_s`, `agent_wall_clock_s`;
- `provenance_ok`, `tools_called`, `n_evaluate_calls`;
- `warnings`.

O artefato JSON da série contém:

- `series_card`, `pool_card`, `diagnosis`;
- `phase2` (pools, handles, baseline seeds — vazios na v12);
- `react.step_details`, `react.trajectory`, `react.summary`;
- `ranking_trace` com validação, teste e `test_forecast` de cada tentativa;
- `final_choice` com a decisão do guard;
- `predict_debug`, `baselines`, `warnings`.

O `ranking_trace` permite recalcular top-k, regret e outras análises sem rodar o
agente de novo.

---

## 11. Como os resultados são agregados

Para cada dataset, o número reportado é a média das métricas por série:

```text
SMAPE_dataset = mean_series( SMAPE_serie )
POCID_dataset = mean_series( POCID_serie )
```

Para o resumo:

- **média das médias** (`mean6`/`mean7`): média simples das médias por dataset;
  cada dataset pesa igual, independentemente do número de séries;
- **agregado** (`agg6`/`agg7`): média sobre todas as séries concatenadas;
  datasets com mais séries pesam mais (M4 tem 359, ANP 182, NN5 111).

Essas duas leituras podem discordar; por isso as duas são reportadas. O agregado
é dominado por M4/ANP/NN5; a média das médias dá o mesmo peso a ETTH1/ETTH2 (7
séries cada).

O `aggregate_metrics` do capítulo de resultados também usa:

- `POCID`: média das séries;
- `MAPE`: média apenas das séries cujo alvo de teste não tem nenhum valor com
  módulo abaixo de `1e-3` (o MAPE explode perto de zero);
- `SMAPE`, `RMSE`: média das séries.

---

## 12. Resultados de referência (gpt-oss)

Os resultados abaixo são da run v12 com **gpt-oss**, 7 datasets. O low repetiu
exatamente o v11_topk_guard_margin nos 6 datasets antigos; o M4 é novo.

### 12.1 SMAPE por dataset

| método | ANP | ETTH1 | ETTH2 | ETTM1 | ETTM2 | NN5 | M4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v12 gpt low** | 0.2193 | **0.2236** | **0.1731** | **0.4303** | 0.1642 | 0.1196 | 0.0916 |
| v12 gpt medium | 0.2245 | 0.2287 | 0.1794 | 0.4483 | **0.1642** | 0.1218 | 0.0916 |
| FFORMA | **0.2166** | 0.3001 | 0.2035 | 0.4396 | 0.1654 | 0.1197 | 0.0996 |
| ADE | 0.2177 | 0.3951 | 0.2261 | 0.4572 | 0.1872 | 0.1178 | 0.1069 |
| median | 0.2194 | 0.2602 | 0.1815 | 0.5443 | 0.1997 | 0.1201 | **0.0897** |
| mean † | 0.2206 | 2.0000 | 1.7197 | 2.0000 | 1.7405 | 0.1199 | 0.1560 |

### 12.2 Média das médias e agregado

| método | média6 | agregado6 | média7 | agregado7 |
|---|---:|---:|---:|---:|
| **v12 gpt low** | **0.221669** | **0.187275** | **0.203082** | 0.136745 |
| v12 gpt medium | 0.227800 | 0.191630 | 0.208343 | 0.138823 |
| FFORMA | 0.240803 | 0.188354 | 0.220624 | 0.141471 |
| ADE | 0.266871 | 0.191779 | 0.244017 | 0.146965 |
| median | 0.254212 | 0.192287 | 0.230704 | 0.139727 |
| mean † | 1.300128 | 0.364916 | 1.136677 | 0.255728 |

† O `mean` simples colapsa nas ETT (SMAPE ≈ 2), então não é um baseline
competitivo; a `median` é o baseline robusto.

### 12.3 POCID (maior melhor)

| método | média6 | agregado6 | média7 | agregado7 |
|---|---:|---:|---:|---:|
| **v12 gpt low** | 53.8701 | **56.9768** | 54.4215 | 57.3743 |
| FFORMA | 52.3569 | 56.8290 | 53.3699 | **58.2115** |
| ADE | 51.4458 | 55.3372 | 52.2805 | 56.3675 |
| median | **54.0484** | 54.7839 | **54.6771** | 56.6611 |
| mean | 49.4134 | 54.5797 | 49.9548 | 53.8605 |

Leitura:

- no SMAPE, o v12 low é o melhor na média6, média7 e agregado6; no agregado7 ele
  perde do v10 medium por ~0.0001;
- no POCID, o v12 low ganha o agregado6, mas não é o melhor nas médias
  (median) nem no agregado7 (FFORMA);
- o `v12 gpt medium` é pior que o low em todos os agregados; não compensa;
- o M4 da median é o melhor (0.0897); o M4 do v12 low (0.0916) é melhor que
  FFORMA/ADE, mas pior que o v10 (0.0910).

O `v12 gpt low` é o candidato atual. O qwen/gemma precisam terminar para a
comparação completa dos quatro.

---

## 13. Por que a v12 chegou nesse desenho

### 13.1 v10: argmin

O v10 termina com:

```text
final_strategy = argmin
```

Aplica a tentativa de menor score de validação. É simples, mas o score de 3
janelas é ruidoso; o melhor de validação não é necessariamente o melhor no teste.

### 13.2 v11: tentativas de decisão final

Foram testadas:

- **`agent_family`**: shortlist cega por família, LLM escolhe ou mantém o líder.
  Ficou pior que o argmin; a escolha livre por LLM introduziu variância e
  overrides ruins.
- **`seasonal_veto`**: o LLM veta ou mantém; também não se sustentou.
- **`seasonal_guard`**: decisão determinística pela forma sazonal; funcionou
  melhor que o seletor livre, mas a run inicial (sem top-k) ainda perdia no
  ETTM2 por ancoragem do agente.
- **top-k por horizonte**: a receita que melhora o ETTM1/ETTM2; o problema era
  que a validação não a colocava no topo. A solução foi permitir que o agente a
  construísse (`direct_preparation`) e que o guard a visse no shortlist
  (família `weighted:topk_mean_per_horizon`).
- **gap 0.10**: evita trocar o líder por uma diferença sazonal pequena; o caso
  ETTM2 série 5 (gap 0.0507) motivou o parâmetro.

Resumo dos 6 datasets (SMAPE):

| política | média6 | agregado6 |
|---|---:|---:|
| v10x_low (argmin) | 0.239556 | 0.188715 |
| v11_seasonal (guard sem top-k) | 0.242904 | 0.189846 |
| v11_topk_guard (sem gap) | 0.237294 | 0.191497 |
| **v11_topk_guard_margin (v12)** | **0.221669** | **0.187275** |
| FFORMA | 0.240803 | 0.188354 |

O v12 low com M4 fecha em `média7 = 0.203082` e `agregado7 = 0.136745`.

### 13.3 Estratégias finais disponíveis no código

O campo `final_strategy` aceita:

| estratégia | decisão |
|---|---|
| `argmin` | menor score de validação |
| `ensemble` | média softmax dos top-M por score |
| `prior_blend` | encolhe o score em direção ao prior do dataset |
| `seed_gate` | mantém o melhor seed se o agente não vencer por margem |
| `mcs` | média do Model Confidence Set (DM) |
| `agent_family` | shortlist cega + LLM |
| `seasonal_guard` | **usada na v12**; determinística |
| `seasonal_veto` | guard + veto binário por LLM |

A v12 usa só `seasonal_guard`. As outras ficam documentadas como ablações; não
fazem parte do resultado oficial.

---

## 14. Reprodução

### 14.1 Script

O runner da v12 está em `run_v12_all.sh`. Ele define:

- 7 datasets;
- as 4 configurações de modelo;
- as flags da seção 1.1;
- duas filas (`gpt-oss` e `qwen/gemma`) em `parallel`, ou uma depois da outra em
  `sequential`.

Comando típico:

```bash
cd ~/Statistics_and_Seq2Seq
export OLLAMA_NUM_CTX=16384
export CHECK_GPU=0
nohup bash run_v12_all.sh sequential > logs/v12_all.log 2>&1 &
```

O `OLLAMA_NUM_CTX=16384` é o contexto do runner no Ollama. O harness envia
`num_ctx` em cada chamada; o prompt medido no gpt-oss fica em torno de 6.1k
tokens (máximo ~6.7k na run v11), então 8192 caberia, mas 16384 dá margem para
qwen/gemma e para o `thinking`.

O `CHECK_GPU=0` pula o preflight de GPU quando os modelos foram descarregados;
o preflight normal só olha processos `llama-server` já carregados.

### 14.2 Uma GPU

Para rodar tudo na mesma GPU/servidor, usar `run_v12_all_onegpu.sh`: as duas
filas apontam para o mesmo `--gpu` e o modo `sequential` garante um
`run_tsf_batch.py` por vez. Isso evita a disputa de CPU no metalearner (o
pré-passo de meta-modelo é CPU-bound e roda antes do agente; dois batches em
paralelo podem saturar a máquina e atrasar a GPU).

### 14.3 Comando direto de um dataset

O comando de cada execução fica em `run_command.txt` e `run_commands.log` na
pasta de resultados. Exemplo (gpt-oss low, ETTM1):

```bash
python3 run_tsf_orchestrator.py \
  --dataset ETTM1 --source ETTM1.tsf --source-dir ../forecasting_datasets \
  --combinator gpt-oss:20b --version v12_gptoss_low --gpu 0 \
  --llm-seed 11 --prompt-v2 --catalog-v2 --no-early-stop --scoring-budget \
  --direct-preparation --max-iterations 12 --max-total-turns 24 \
  --temperature 0.7 --score-preset smape_only --score-window-mode weighted \
  --no-agent-score-preset --no-seed-attempts --no-best-single \
  --final-strategy seasonal_guard --final-selector-max-candidates 10 \
  --seasonal-guard-min-gap 0.10 --reasoning low
```

### 14.4 Onde ficam os resultados

```text
timeseries/mestrado/resultados/
├── orchestrator_react_v12_gptoss_low/
├── orchestrator_react_v12_gptoss_medium/
├── orchestrator_react_v12_qwen_medium/
└── orchestrator_react_v12_gemma_on/
```

Cada pasta tem os CSVs por dataset, `agent_summary.json`, `run_command.txt`,
`run_commands.log` e `llm_artifacts/`.

---

## 15. Limitações e cuidados

- **Uma run, um seed**: o agente é sensível à trajetória. O low reproduziu o
  v11, mas a confirmação com outro seed/reasoning ainda é necessária.
- **Top-k depende do agente**: a v12 dá as ferramentas e o exemplo; não força a
  receita. Se o agente não a criar, o guard não pode escolhê-la.
- **M4 pesa muito no agregado**: 359 das 680 séries. O agregado7 é dominado por
  M4; a média7 é a leitura balanceada por dataset.
- **POCID não é o critério**: o score é SMAPE; POCID é reportado. O v12 pode
  ganhar SMAPE e não ganhar POCID.
- **3 janelas**: a confiança da seleção é baixa; `selection_verdict` costuma ser
  `indistinguishable`. O guard não resolve a variância da validação; ele só
  corrige a forma sazonal.
- **Gate conservador**: séries com menos de 10 ciclos ou `seasonal_strength <
  0.5` nunca são trocadas; o líder é mantido. Isso protege NN5/ANP, mas também
  limita o ganho onde o seasonal naive não é estimável.
- **Sem teste no loop**: o teste real só é lido na Fase 4; qualquer análise de
  "test_forecast" no artefato é feita depois da decisão, para diagnóstico.
- **`mean` simples não é um baseline forte**: ele colapsa em séries que cruzam
  zero/escalas diferentes; a `median` é o baseline robusto.

---

## 16. Mapa de código

| arquivo | função |
|---|---|
| `orchestrator_react/ingest.py` | lê CSVs, monta `y_true/y_preds/test_preds`, alinha `.tsf` |
| `orchestrator_react/state.py` | `ReactState`, backtest, `apply_to_test`, score, tentativas |
| `orchestrator_react/combiners.py` | mean/median/trimmed/weighted/DBA/best_single |
| `orchestrator_react/weighting.py` | receitas de pesos e `topk_mean_per_horizon` |
| `orchestrator_react/selection.py` | `top_k`, `stable`, `forward`, `prune` |
| `orchestrator_react/tools.py` | catálogo de ferramentas do agente |
| `orchestrator_react/pool.py` | Phase 2, pool card, seeds |
| `orchestrator_react/pipeline.py` | fases 0–5, chamada do guard e aplicação no teste |
| `orchestrator_react/final_choice.py` | shortlist, gate, `seasonal_dist`, guard |
| `orchestrator_react/prompts.py` | prompt v2, dica top-k, catálogo |
| `orchestrator_react/config.py` | flags e presets de score |
| `run_v12_all.sh` | runner das 4 configurações × 7 datasets |
| `run_tsf_batch.py` | batch por modelo/dataset |
| `run_tsf_orchestrator.py` | CLI de uma execução |

### Glossário rápido

- **tentativa / attempt**: uma combinação avaliada (`combine`, `pool`, `weights`)
  com score de validação.
- **líder**: tentativa de menor score de validação.
- **shortlist**: melhores membros de cada família, após deduplicação e filtro de
  plausibilidade.
- **gate**: condição `seasonal_strength >= 0.5`, `period >= 2`,
  `n_points >= 10*period`.
- **seasonal naive**: repetição do último ciclo completo da série.
- **seasonal_dist**: `mean(|forecast - seasonal_naive|) / scale`.
- **gap**: `seasonal_dist(líder) - seasonal_dist(melhor candidato)`.
- **score**: razão SMAPE/SMAPE do mean do pool, agregada 0.2/0.3/0.5 nas 3
  janelas; menor é melhor.
