# V10 — funcionamento de ponta a ponta

Documento da versão **v10 sem `diverse_gate`**: a escolha final é `argmin` da
validação.

Observação sobre a pasta histórica: `orchestrator_react_v10_diverse_gate_low`
foi gerada antes da remoção do gate. Os resultados abaixo para o v10 sem gate são
o **top 1 da mesma trajetória** (a decisão final não altera as tentativas do
agente). Para o resultado oficial da versão sem gate, o recomendado é rerodar.

Arquivo de resultados:

```text
timeseries/mestrado/resultados/orchestrator_react_v10_diverse_gate_low/
```

---

## 1. Comandos para rodar o v10x sem gate

Referência em `low`, seed do LLM **11**:

```bash
cd /home/lucas.castro/Statistics_and_Seq2Seq

nohup /home/lucas.castro/.conda/envs/agno/bin/python3 run_tsf_batch.py \
  --datasets ETTH1 ETTH2 ETTM1 ETTM2 NN5_WEEKLY_DATASET ANP_MONTHLY \
  --combinators gpt-oss:20b \
  --gpu 1 \
  --version v10x_low \
  -- --reasoning low --llm-seed 11 --prompt-v2 --catalog-v2 \
  --no-early-stop --scoring-budget \
  --max-iterations 12 --max-total-turns 24 \
  --temperature 0.7 \
  --score-preset smape_only --score-window-mode weighted \
  --no-agent-score-preset --no-seed-attempts --no-best-single \
  --final-strategy argmin \
  > logs/orchestrator_react_v10x_low_batch.log 2>&1 &
```

Depois, com o mesmo seed e o mesmo resto, trocando apenas o reasoning:

```bash
cd /home/lucas.castro/Statistics_and_Seq2Seq

nohup /home/lucas.castro/.conda/envs/agno/bin/python3 run_tsf_batch.py \
  --datasets ETTH1 ETTH2 ETTM1 ETTM2 NN5_WEEKLY_DATASET ANP_MONTHLY \
  --combinators gpt-oss:20b \
  --gpu 1 \
  --version v10x_medium \
  -- --reasoning medium --llm-seed 11 --prompt-v2 --catalog-v2 \
  --no-early-stop --scoring-budget \
  --max-iterations 12 --max-total-turns 24 \
  --temperature 0.7 \
  --score-preset smape_only --score-window-mode weighted \
  --no-agent-score-preset --no-seed-attempts --no-best-single \
  --final-strategy argmin \
  > logs/orchestrator_react_v10x_medium_batch.log 2>&1 &
```

Não há M4 nesta run. A ordem efetiva é: ETTs, NN5, ANP.

O seed `11` é o seed de amostragem do LLM (`options.seed`), não uma tentativa
semente. O `--llm-seed` é aplicado ao combinator, diagnostician e reporter.

O comando exato de cada execução é salvo automaticamente em
`run_command.txt`, e o histórico em `run_commands.log`, com `cwd`,
`CUDA_VISIBLE_DEVICES` e `REACT_OLLAMA_URL`.

---

## 2. Dados e splits

### 2.1 Entrada

- `.tsf` original de cada dataset, em `../forecasting_datasets`.
- Previsões dos modelos base em
  `timeseries/mestrado/resultados/<MODELO>/normal/<DATASET>.csv`.
- 19 modelos base:

```text
ARIMA, ETS, THETA, rf, catboost, CWT_rf, DWT_rf, FT_rf,
CWT_catboost, DWT_catboost, FT_catboost, ONLY_CWT_catboost,
ONLY_CWT_rf, ONLY_DWT_catboost, ONLY_DWT_rf, ONLY_FT_catboost,
ONLY_FT_rf, NaiveSeasonal, NaiveMovingAverage
```

No ETTM, 5 modelos `ONLY_*` são descartados por alvo de teste desalinhado. O ETTM
fica com **14 modelos ativos**. ANP, ETTH e NN5 ficam com 19.

| Dataset | Séries |
|---|---:|
| ANP_MONTHLY | 182 |
| ETTH1 | 7 |
| ETTH2 | 7 |
| ETTM1 | 7 |
| ETTM2 | 7 |
| NN5_WEEKLY_DATASET | 111 |
| **Total** | **321** |

### 2.2 Janelas

- **3 janelas de validação**.
- A janela de teste é a mais recente e fica fora do loop.
- Backtest `expanding`.
- `nested_selection=True`: o pool pode ser reajustado por janela de validação.
- `weights_ols` fica indisponível: exige pelo menos 5 janelas e a run tem 3.
- O `.tsf` é conferido contra o alvo do CSV por `verify_alignment`.
- O teste real nunca entra no prompt, no score de validação ou na decisão.

---

## 3. Fluxo geral

```text
.tsf + previsões dos 19 modelos
        |
        v
Fase 0  ingestão e alinhamento
        |
        v
Fase 1  perfil da série + pool card + diagnóstico determinístico
        |
        v
Fase 2  pool completo; nenhuma tentativa pré-avaliada
        |
        v
Fase 3  agente ReAct com gpt-oss:20b
        |  uma combinação por turno
        |  score apenas na validação
        v
Fase 4  argmin da validação
        |
        v
Fase 5  aplica ao teste, mede, grava CSV e artefatos
```

---

## 4. Fase 0 — ingestão e alinhamento

Para cada série:

1. carrega o `.tsf` e as previsões de cada modelo;
2. identifica o `dataset_index`;
3. verifica se os últimos pontos do `.tsf` batem com o alvo do CSV;
4. mantém apenas séries com 3 janelas válidas;
5. no ANP, `drop_zero_windows_24` deixa 182 séries.

O agente nunca vê os valores brutos da série nem o tensor completo de previsões.

---

## 5. Fase 1 — perfil, pool e diagnóstico

### 5.1 Perfil

O `SERIES PROFILE` contém:

- tendência e sazonalidade;
- estacionariedade;
- outliers;
- entropia espectral;
- tamanho, frequência e horizonte;
- última observação normalizada.

### 5.2 Pool card

O `MODEL POOL` contém:

- tabela de erro por modelo;
- estabilidade do ranking entre janelas (`Kendall tau`);
- grupos de erro correlacionado;
- melhor modelo por janela;
- dispersão dos erros.

### 5.3 Diagnóstico

Sem LLM diagnosticador. O diagnóstico determinístico produz:

- `regime`: trend / seasonal / mixed / noisy;
- `predictability`: high / medium / low;
- `combination_hint`: robust / weighted / selective / full_pool;
- riscos e narrativa curta.

É uma leitura, não uma regra.

---

## 6. Fase 2 — pool e tentativas iniciais

Configuração:

```text
pool_mode = full
no_seed_attempts = True
no_best_single = True
```

Consequências:

- pool de trabalho = pool completo (19 ou 14 modelos);
- **não há tentativas pré-avaliadas**; o histórico começa vazio;
- pools estáveis podem ser registrados como handles, mas não entram como
  tentativas pontuadas;
- o agente precisa construir/testar o que quiser;
- `best_single` não pode ser usado; toda combinação precisa de pelo menos 2
  modelos.

O score de normalização é o **mean do pool completo**, calculado por janela.

---

## 7. Fase 3 — agente ReAct

### 7.1 Modelo e amostragem

```text
combinator       = gpt-oss:20b
reasoning        = low
temperature      = 0.7
seed             = 7 (default da época)
diagnostician    = none
reporter         = none
prompt           = v2
catálogo         = v2
```

O seed vai em `options.seed` para o Ollama. O retry de resposta vazia/erro de API
aumenta o seed em +1 e a temperatura em +0.15, até 0.9. O v10 não salvava
`retry_events`; esse registro foi adicionado depois.

### 7.2 Orçamento

```text
max_iterations      = 12 avaliações
max_total_turns     = 24 turnos
no_early_stop       = True
scoring_budget      = True
accept              = desligado
```

Cada `evaluate_strategy` conta como uma tentativa pontuada. O final é escolhido
depois do orçamento; o agente não pode “aceitar” uma tentativa no meio.

### 7.3 Resposta

```text
Thought: ...
Action: <tool>
Action Input: {...}
```

Erros de parsing ou argumentos inválidos viram observação para o agente corrigir;
não matam a série.

### 7.4 Prompt

A cada turno, o agente recebe:

- `SERIES PROFILE`;
- `MODEL POOL`;
- `DIAGNOSIS`;
- `ATTEMPT HISTORY`, ordenado por score de validação;
- `LEADER DIAGNOSTICS`;
- `HANDLES`;
- `WEIGHT METHODS`;
- `NEXT UNTRIED`;
- `SCRATCHPAD`;
- `LAST OBSERVATION`;
- `FORECAST SHAPE`;
- `SERIES DECISION CARD`;
- orçamento.

O `FORECAST SHAPE` mostra todos os modelos de forma compacta, com detalhamento de
top3 + um representante por cluster; o actual do teste **não** aparece.

O `LEADER DIAGNOSTICS` mostra o top 3 da validação e o spread entre eles. O score
final continua sendo `argmin`.

### 7.5 Revisão do prompt para os parâmetros do v10

Com `scoring_budget=True`, o harness entra em modo direto: só existe
`evaluate_strategy`. A revisão removeu as contradições:

- o prompt não fala mais em `select_*`, `prune_*` ou `weights_*` como ações
  disponíveis;
- não pede para “criar um handle”;
- o bloco `CLASSIC STARTING POINTS` continua válido porque
  `evaluate_strategy` aceita `models`, `pool` e um método de pesos;
- o turno informa que handles de pool vindos da Phase 2 podem ser reutilizados,
  mas novos handles não são criados no modo direto;
- `best_single` aparece apenas como desabilitado;
- `score_preset` aparece como fixo;
- `n_eff` não aparece;
- não há `diverse_gate`.

---

## 8. Como uma tentativa é avaliada

Cada `evaluate_strategy` gera um `Attempt` com:

- `spec`: combinação, pool e weights;
- `per_window`: métricas de cada janela;
- `aggregate`: métricas agregadas;
- `score`: score de validação;
- `n_models`: tamanho do pool;
- `per_window_scores`: score composto por janela;
- `residuals`: para Diebold-Mariano/bootstrap.

### 8.1 Métricas por janela

Com previsão `p` e real `y`:

```text
SMAPE = mean( 2 * |p - y| / (|p| + |y|) )
```

SMAPE é fração em [0, 2], não percentual. Se `|p| + |y| = 0`, o ponto é ignorado.

```text
POCID = 100 * (# passos em que Δp e Δy têm o mesmo sinal) / (n - 1)
```

POCID maior = melhor.

Também são calculados:

```text
MAPE   = mean(|p - y| / |y|), ignorando y = 0
RMSE   = sqrt(mean((p - y)^2))
MAE    = mean(|p - y|)
mSMAPE = mean(2|p-y| / max(0.6, |p| + |y| + 0.1))
```

### 8.2 Score de validação

O v10 usa:

```text
score_preset      = smape_only
score_window_mode = weighted
```

Pesos do preset:

```text
b_smape = 1.0
a_rmse = c_mape = d_pocid = 0.0
```

O score de uma tentativa em uma janela é o SMAPE da tentativa dividido pelo SMAPE
do **mean do pool completo** naquela janela:

```text
score_w = SMAPE_tentativa_w / SMAPE_mean_pool_w
```

Menor é melhor.

O score final da tentativa é:

```text
score = 0.2 * score_0 + 0.3 * score_1 + 0.5 * score_2
```

A janela mais recente pesa mais. O `score` não é o SMAPE bruto; o SMAPE bruto é o
que aparece nas métricas de teste/CSV.

---

## 9. Escolha final

O v10 usa:

```text
final_strategy = argmin
```

Ou seja:

1. ordena todas as tentativas pelo score de validação;
2. aplica a tentativa de menor score;
3. não usa teste em nenhum momento;
4. não há gate, ensemble ou desempate por diversidade.

O `diverse_gate` foi testado em uma execução histórica e removido: no mesmo run,
o top 1 puro teve média melhor. A seção 12 preserva esse resultado apenas como
ablação.

---

## 10. Fase 4 — aplicação e métricas

Depois da escolha:

1. `apply_to_test` aplica a mesma combinação às previsões de teste;
2. o teste real não entra na escolha;
3. calcula SMAPE/POCID/RMSE/MAPE/MAE/mSMAPE contra o alvo;
4. grava CSV e artefatos.

O forecast final é a combinação dos modelos do pool escolhido: mediana, média,
trimmed, DBA ou weighted, conforme o `spec`.

---

## 11. Resultados do v10 sem gate

Como a pasta histórica foi gerada antes da remoção do gate, os números abaixo são
o **top 1 da mesma trajetória** — o que o `argmin` teria aplicado.

| Dataset | SMAPE | POCID |
|---|---:|---:|
| ANP_MONTHLY | 0,219606 | 63,337 |
| ETTH1 | 0,227015 | 62,112 |
| ETTH2 | 0,174639 | 54,037 |
| ETTM1 | 0,523353 | 44,099 |
| ETTM2 | 0,173417 | 40,994 |
| NN5_WEEKLY_DATASET | 0,116596 | 43,372 |
| **Média dos 6** | **0,239104** | **51,325** |

Comparação:

| Dataset | v6 | v10 sem gate | FFORMA |
|---|---:|---:|---:|
| ANP | 0,219481 | 0,219606 | 0,216593 |
| ETTH1 | 0,240168 | 0,227015 | 0,300054 |
| ETTH2 | 0,177242 | 0,174639 | 0,203528 |
| ETTM1 | 0,517196 | 0,523353 | 0,439569 |
| ETTM2 | 0,183850 | 0,173417 | 0,165421 |
| NN5 | 0,117393 | 0,116596 | 0,119654 |
| **Média 6** | **0,242555** | **0,239104** | **0,240803** |

O v10 sem gate fica **0,0017 abaixo do FFORMA** na média dos 6, mas isso é um
counterfactual da mesma trajetória, não uma run nova. O ETTM1 continua longe do
FFORMA; o ETTM2 está próximo.

---

## 13. Composição dos finais sem gate

| Dataset | weighted | mean | median | trimmed | dba |
|---|---:|---:|---:|---:|---:|
| ANP | 67 | 24 | 41 | 21 | 29 |
| ETTH1 | 5 | 0 | 0 | 1 | 1 |
| ETTH2 | 0 | 0 | 3 | 3 | 1 |
| ETTM1 | 2 | 2 | 1 | 1 | 1 |
| ETTM2 | 3 | 1 | 1 | 1 | 1 |
| NN5 | 33 | 16 | 31 | 16 | 15 |

O ETTM1 ainda termina em `median` em uma série e tem `dba` em outra; o ETTM2
termina majoritariamente em `weighted`.

---

## 14. Arquivos gerados

```text
timeseries/mestrado/resultados/orchestrator_react_v10_diverse_gate_low/
├── ANP_MONTHLY.csv
├── ETTH1.csv
├── ETTH2.csv
├── ETTM1.csv
├── ETTM2.csv
├── NN5_WEEKLY_DATASET.csv
├── agent_summary.json
├── run_command.txt
├── run_commands.log
└── llm_artifacts/
    ├── ANP_MONTHLY/
    ├── ETTH1/
    ├── ETTH2/
    ├── ETTM1/
    ├── ETTM2/
    └── NN5_WEEKLY_DATASET/
```

Cada artefato de série contém:

- `series_card`;
- `pool_card`;
- `diagnosis`;
- `decision`;
- `ranking_trace` com validação, teste e `test_forecast` de cada tentativa;
- `react.step_details`;
- `react.summary`;
- `predict_debug`;
- `warnings` (na pasta histórica, as decisões do gate).

O `ranking_trace` permite recalcular top 3, ensemble e regret depois, sem rodar o
agente novamente.

---

## 15. Reprodução e status

O comando sem gate está na seção 1. O código atual usa `argmin` e não aceita
`--final-strategy diverse_gate`.

O v10 histórico usou o seed default 7. As runs novas `v10x_low` e `v10x_medium`
usam seed 11 via `--llm-seed`, e seus artefatos já registram `retry_events`.

---

## 16. Limitações

- M4 não entrou; o agregado é de 6 datasets.
- O ETTM1 continua longe do FFORMA.
- O ETTM2 ficou perto do FFORMA.
- O ANP e o NN5 ficaram na faixa do v6.
- O `diverse_gate` não se sustentou e foi removido.
- Os resultados sem gate acima são um counterfactual da trajetória do v10; para
  o número oficial da versão sem gate, rerodar com `--final-strategy argmin`.
