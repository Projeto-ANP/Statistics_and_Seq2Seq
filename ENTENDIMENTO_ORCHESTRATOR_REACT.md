# Entendimento do orchestrator ReAct — atualizado do código + diagnóstico empírico

> Escrito a partir do **código** (`orchestrator_react/`, `run_tsf_orchestrator.py`,
> `tests/`), não dos .md anteriores (que estavam desatualizados em pontos — ex.
> o docstring de `run_tsf_orchestrator.py` dizia 8 iterações quando o default é 12).
> Todas as medições citadas aqui foram reproduzidas localmente com código
> **determinístico** (nenhuma chamada de LLM) em 2026-09-25, sobre os runs
> publicados em `timeseries/mestrado/resultados/orchestrator_react_*`.
> Última atualização: 2026-09-25.

---

## 1. Arquitetura (o que o código faz)

### 1.1 Fluxo por série — `pipeline.run_series()`

```
Fase 0  ingest.load_series()      CSVs por modelo + .tsf -> ReactState
        (y_true WxH, y_preds WxMxH, test_preds MxH; teste cego nunca entra no state)
Fase 1a tools.series_profile()    card determinístico da série (sempre roda)
Fase 2  pool.run_phase2()         avalia pool + SEMEIA baselines no histórico
Fase 1b phases.run_diagnosis()    interpretação por LLM (opcional; degrada, não falha)
Fase 3  react_loop.run_react_loop()   O AGENTE (Thought -> Action -> Observation)
Fase 4  state.apply_to_test()     melhor tentativa do histórico -> janela de teste
Fase 5  phases.run_report()       prosa de justificativa (opcional)
        csv_writer.ResultWriter   linha de 58+ colunas + artifact JSON por série
```

### 1.2 O contrato central (o que o agente PODE e NÃO PODE)

- **O agente nunca escreve números.** Nem previsões, nem pesos. Ele chama
  ferramentas; elas devolvem *handles* (`pool1`, `w2`) + resumo qualitativo.
- **`evaluate_strategy` é o único caminho** para uma estratégia entrar no
  histórico — e toda estratégia passa por backtest nas janelas de validação
  antes de ser aceita (`state.evaluate` → `state.backtest`).
- **Princípio 5** (`react_loop.run_react_loop`, final): a estratégia aplicada é
  **sempre a melhor de todo o histórico** (sementes incluídas). Se o agente
  aceitar algo pior, `agent_accepted_id` registra e `overridden=True` — nunca
  silencioso.
- **Contenção**: nome de modelo inventado, handle desconhecido, argumento
  inválido, resposta malformada → *observation* estruturada, o loop continua;
  nunca corrompe o resultado (`tests/test_guarantees.py` seção 1).
- **Anti-vazamento** (3 garantias com teste dedicado): `ReactState` não guarda
  os valores reais do teste; baselines externos só são lidos DEPOIS da última
  chamada ao LLM; nenhum prompt contém valor de teste.

### 1.3 O loop — `react_loop.run_react_loop()`

- Orçamento: `max_iterations=12`, early-stop com `patience=4` propostas sem
  melhora. Retries: até 8 para resposta vazia, até 4 para erro de API (não
  consomem iteração). Parses malformados viram observation e o agente se corrige.
- Formato de saída exigido: `Thought:` / `Action:` / `Action Input: {json}`.
- Catálogo fechado de 24 ferramentas (`registry.TOOLS`), em 5 grupos:
  diagnóstico (6), seleção de pool (3), pesos (6), combinação (6), validação (3).
  Ação terminal: `accept(attempt_id, confidence, justification)`.
- Ferramentas inválidas para a rodada são **retiradas do catálogo e do prompt**
  antes do loop (`withheld_tools`): `weights_ols` com <5 janelas;
  `weights_pooled_meta_model` sem meta-modelo LOO; `combine_*` redundantes com
  `--drop-redundant-combine`.

### 1.4 Prompts — `prompts.py`

- Sistema: papel, catálogo exaustivo, formato de 3 linhas, sequência típica,
  diferenças entre métodos de peso, regras (comparações estruturadas; não
  repetir; handle só vale no pool onde foi criado; aceitar semente quando nada
  sugere desvio concreto; etc.). Formato `xml` (default) ou `text`.
- Turno: `SERIES PROFILE` + `MODEL POOL` + `DATASET CARD` (o que funcionou nas
  OUTRAS séries, LOO, só validação) + `DIAGNOSIS` + `ATTEMPT HISTORY` (top-10) +
  `HANDLES` + `SCRATCHPAD` (últimos 6 passos) + última observation.

### 1.5 Onde ficam os resultados

```
entrada  ./timeseries/mestrado/resultados/<MODEL>/normal/<DATASET>.csv   (sep=";")
         ./timeseries/mestrado/resultados/<NAME>/<DATASET>.csv           (mean/median/dba/ADE/FFORMA)
saída    ./timeseries/mestrado/resultados/orchestrator_react_<version>/<DATASET>.csv
         .../orchestrator_react_<version>/llm_artifacts/<DATASET>/dataset_<i>.json
```

O CSV tem 13 colunas core (métricas via `all_functions`, imutáveis) + 23
mantidas + 16 novas de rastreabilidade (`react_trajectory_json`,
`baseline_results_json`, `ablation_config`, `selection_verdict`, ...). O JSON de
artifact tem o audit trail completo: cards, trajectory, step_details,
parse_failures, cross_series, phase2, sanity, warnings.

---

## 2. O problema reportado — medido

> "O resultado do agente está escolhendo piores que o modelo que a semente
> escolhe antes de passar pro agente."

Confirmado, e agora quantificado. Metodologia: **replay determinístico** das
sementes (Fase 2) de cada série e aplicação à janela de teste cega; comparação
com o que cada run realmente reportou. Script: `diagnostics/analyze_run.py`
(que consolida as análises exploratórias desta seção).

### 2.1 NN5 (111 séries) — o agente PIORA o dataset

| braço | sMAPE médio (teste) |
|---|---|
| oráculo: melhor semente escolhida **no teste** (piso real da Fase 2) | **0.1060** |
| determinístico: argmin das sementes **na validação** (`orchestrator_baseline_v1`) | 0.1157 |
| **publicado com agente (`orchestrator_react_v5`)** | **0.1177** ← pior que o braço sem agente |
| ADE / FFORMA (referência externa) | 0.1178 / 0.1197 |

- **95/111 séries**: o vencedor final é pior do que a melhor semente medida no
  teste (delta médio +0.0117).
- **43/50** das finais de origem `agent` são piores que a melhor semente no teste.
- Verdicts do braço v2: `indistinguishable` em **99/111** séries — com 3 janelas
  de validação, o topo do ranking não é separável estatisticamente; o argmin
  escolhe ruído.

### 2.2 ETTM2 (7 séries) — o agente ajuda, mas por 2 séries de sorte

| braço | sMAPE médio |
|---|---|
| determinístico (`orchestrator_baseline_v1`) | 0.2030 |
| oráculo: melhor semente no teste | 0.1960 |
| publicado com agente (`orchestrator_react_iso_control_repeat`) | **0.1608** |
| FFORMA | 0.1654 |

- O agente muda o resultado em **apenas 2/7 séries** (idx0: 0.1306, idx5:
  0.2251). Nas outras 5, o vencedor é uma semente.
- **4/7 séries**: final pior que a melhor semente no teste (deltas pequenos:
  +0.001…+0.010 — argmin de validação escolhendo a semente "errada").
- A série 5 é o caso extremo: a semente vencedora na validação (`mean pool3`)
  dá **45,3%** de sMAPE no teste; o agente escapou dela com uma média de pool
  podado que dá **22,5%** — e a margem na validação que motivou a troca foi
  minúscula (0,0434 vs 0,0447).
- No ETTM2 o agente **bate até o oráculo das sementes** (0.1608 < 0.1960):
  as propostas dele são genuinamente melhores que qualquer semente.

### 2.3 A causa raiz — o sinal do agente é ANTI-preditivo do teste

Replay de **456 propostas** do agente nas 111 séries do NN5
(`diagnostics/analyze_run.py`):

- Spearman(score de validação, sMAPE de teste) das propostas do agente:
  **média +0.005, mediana 0.000, positivo em 49/109 séries** → o score que o
  agente otimiza **não carrega informação** sobre a janela cega.
- **Transferência 9/93 (10%)**: quando uma proposta do agente vence a melhor
  semente na validação, ela só vence também no teste em 10% dos casos. O acaso
  daria ~50%. Ou seja: vencer na validação contra as sementes é, em 90% das
  vezes, **overfit sistemático**, não qualidade real.

**Mas a transferência é dataset-dependente** — e o sinal de qual regime você
está já está computado no CSV (`selection_verdict`):

| dataset | verdicts | transferência das propostas | Spearman(val,teste) |
|---|---|---|---|
| NN5 | `indistinguishable` 99/111 | 9/93 (10%) | +0.005 (49/109 positivos) |
| ETTM2 | `separated` 6/7 | 5/5 (100%) | +0.275 (3/4 positivos) |

Onde o vencedor é estatisticamente **separável** do 2º colocado, a validação
informa e as propostas transferem; onde não é (a regra no NN5 com 3 janelas),
o argmin escolhe ruído. Isso é a base da hipótese H1 (§5): o próprio sistema
já calcula o indicador que distingue os dois regimes — ele só não é usado na
decisão final.

Isso é exatamente a assinatura do **backtest overfitting**: o argmin sobre
dezenas de candidatos avaliados em 3 janelas seleciona o candidato que melhor
absorveu o ruído daquelas janelas (Bailey, Borwein, López de Prado, Zhu 2014 —
"Pseudo-Mathematics and Financial Charlatanism"; PBO — "Probability of Backtest
Overfitting"). O `nested_selection` já corrigiu a seleção de pool (Spearman
−0.468 → +0.547), mas a **escolha final** (argmin do score de 3 janelas, agente
ou semente) continua sobreajustada — o oráculo 0.1060 vs determinístico 0.1157
mostra que até o braço sem agente deixa ~0.01 na mesa só por isso.

---

## 3. Limitações estruturais (ranked)

**L1 — Seleção em 3 janelas é overfit por construção.** Dezenas de candidatos
(as sementes + tudo que o agente propõe) × 3 janelas = comparação múltipla
severa. O vencedor em validação é, em média, pior no teste que a mediana dos
candidatos. Efeito: oráculo 0.1060 vs 0.1157 determinístico.

**L2 — O princípio 5 protege contra "pior que semente NA VALIDAÇÃO", não no
teste.** O loop garante `final = argmin(score de validação)`; ele não tem como
saber que o score é anti-preditivo (2.3). O agente *vencendo* na validação é
precisamente o mecanismo do dano: quanto melhor o agente fica em bater as
sementes nas 3 janelas, pior pode ficar o teste (10% de transferência).

**L3 — O agente só intervém em minoria das séries.** NN5: 50/111 finais de
origem agente; ETTM2: 2/7. Para a maioria das séries, o "resultado do agente"
**é uma semente** — o valor do sistema inteiro está no conjunto de sementes, e
a métrica reportada mede majoritariamente a Fase 2, não o LLM. (É por isso que
`seed_stable_pools` foi o maior lever medido: 0.12036 → 0.11500 no braço
determinístico.)

**L4 — n pequeno por dataset torna a avaliação por série ~loteria.** ETTM2 tem
7 séries; uma série (idx5) move a média do dataset de 0.20 para 0.16. NN5 (111)
é mais estável, mas mesmo lá os deltas por série são grandes (worst: +0.133).

**L5 — Variância de amostragem do LLM.** Seed fixa 7 mascara a variância. Na
série 5 do ETTM2 (config restrita), reasoning=None nunca escapou da semente
(5/5, 45,3%), reasoning=low escapou 3/5 (22,5%). A mesma série, no run de
dataset inteiro, escapa com reasoning=None. O "resultado" de um run depende de
qual rolagem o amostrador deu.

**L6 — Runs parciais (`--indices`) mudam a arquitetura do agente, sem aviso no
resultado.** Com <20 séries o pooled meta-model não é treinado e o DATASET CARD
some → `weights_pooled_meta_model` é retirada do catálogo e o prompt perde o
contexto cross-series. **Isso invalida "tirar níveis" rodando um subconjunto
de séries**: você não está isolando uma variável, está trocando o jogo do
agente. A série 5 é a vítima documentada disso (5/5 = semente no run
restrito; escapa no run completo — ver §4).

**L7 — A métrica composta normaliza contra a média do pool** (`baseline_aggregate`),
o que é sensato, mas o score é uma combinação ad hoc de RMSE/SMAPE/MAPE/POCID
(`score_preset`); o agente otimiza esse composto, não o sMAPE reportado.

---

## 4. Os testes existentes e os runs de isolamento ("níveis")

### 4.1 O que os testes JÁ cobrem (`tests/`, 10 arquivos, todos verdes)

| arquivo | cobre |
|---|---|
| `test_guarantees.py` | contenção (agente alucinando não corrompe nada), vazamento estrutural/comportamental/prompt, alinhamento .tsf↔CSV, "nada do teste chega ao agente" |
| `test_react_loop.py` | parsing do formato de 3 linhas, retries, early-stop, princípio 5 (`test_agent_cannot_pick_something_worse_than_a_baseline`, `test_result_is_never_worse_than_the_best_baseline`), determinismo com script |
| `test_nested_selection.py` | pool re-escolhido por fold, LOO na seleção |
| `test_pooled_meta_model_tool.py` / `test_meta_model.py` | meta-modelo LOO |
| demais | CSV writer, ingest, phases, pipeline |

**O que NÃO é testado — e é exatamente o problema reportado:** nenhum teste
avalia *transferência* validação→teste do agente (2.3). Os testes do princípio 5
garantem "nunca pior que semente na validação", que é uma propriedade **fraca**
diante de um score anti-preditivo. O gap é de especificação, não de implementação.

### 4.2 Os runs de isolamento existentes ("níveis") — leitura correta

Os runs `iso_*` + `v2_gpt_low_nivel` isolaram as 4 flags do run combinado,
uma por braço (ETTM2, 7 séries):

| run | flags | sMAPE | agente interveio |
|---|---|---|---|
| `iso_control_repeat` | nenhuma (reproduz publicado) | **0.1608** | 2/7 |
| `iso_reorder_low` | só `--reorder-weight-tools` | 0.1608 (idêntico) | 2/7 |
| `iso_reasoning_low` | só `--reasoning low` | 0.1933 | 1/7 |
| `iso_reducedseed_low` | só `--reduced-seeding` | 0.1939 | 1/7 |
| `iso_dropcomb_low` | só `--drop-redundant-combine` | 0.2030 | 0/7 |
| `orchestrator_baseline_v1` | determinístico | 0.2030 | — |

Leituras corretas:
1. `--drop-redundant-combine` **removeu as ferramentas que eram o caminho de
   vitória do agente** (as propostas vencedoras usavam exatamente
   `combine_mean`/`combine_weighted`/`best_single` via `evaluate_strategy`
   flat). O run virou o braço determinístico (0.2030 = baseline_v1). **Não é
   evidência de que menos ferramentas = melhor; é o contrário aqui.**
2. `--reasoning low` piorou (0.1608 → 0.1933) — o modelo raciocina menos e não
   encontra a troca da série 5. `reasoning off` **nunca foi testado** e é um
   braço distinto (remove o canal harmony, não só reduz orçamento).
3. `--reorder-weight-tools` é neutro no ETTM2 (idêntico ao controle, mesmo seed).
4. **n=7 é pequeno demais para concluir nada sobre cada flag sozinha.** Os
   deltas são dominados por 1-2 séries (ver L4/L5). Os `iso_s5_*` (5×5 na série
   5) medem variância do agente e mostram o trap do `--indices` (L6).

### 4.3 Faz sentido "ir tirando níveis na hora de rodar"?

Sim, é o desenho correto (uma variável por braço, âncora reproduzindo o
publicado), **com três correções**:
1. **Sempre rodar o dataset inteiro.** Ablação por `--indices` muda a
   arquitetura (L6) e não isola nada. Para séries específicas, usar o braço
   completo e filtrar na análise (é o que `diagnostics/analyze_run.py --series` faz).
2. **Âncora de repetibilidade por braço** (o `iso_control_repeat`), e sementes
   variadas para separar efeito de flag de variância de amostragem (L5).
3. **Reportar por série, não só média do dataset** — com n pequeno a média
   esconde tudo.

---

## 5. Hipóteses de melhoria (com o que medir)

**H1 — Gate de separação estatística na escolha final.** Bloquear o override de
semente quando o vencedor não é estatisticamente separável do melhor baseline
(`selection_confidence`/`selection_verdict` já computam isso). A evidência da
§2.3 torna essa a hipótese mais forte: onde o verdict é `separated`, a
transferência foi 100% (ETTM2); onde é `indistinguishable` (99/111 no NN5),
foi 10%. Predição: NN5 colapsa para ≈ braço determinístico 0.1157
(< 0.1177 do agente) — ganho quase grátis; risco: bloquear trocas
"milagrosas" como a série 5 do ETTM2 (22,5% vs 45,3%), onde a validação não
separava mas o teste sim (no ETTM2 o verdict já é `separated` na maioria das
séries, então o gate quase não agiria lá — o risco é concentrado nas séries
NN5 com semente ruim). Variante: gate assimétrico, só bloquear quando a
semente atual é ela própria boa.

**H2 — Em vez de argmin, encolhimento/ensemble do topo.** `final_strategy="ensemble"`
e `"prior_blend"` já existem; a medição antiga do ensemble foi feita contra o
conjunto de sementes antigo e não sobreviveu às sementes de estabilidade —
**re-medir agora que sabemos que o argmin é overfit (2.3)**. O prior_blend
ganhou no ANP e perdeu no NN5 (dataset-dependente, honesto mas inaceitável como
default sem regra de escolha).

**H3 — `--reasoning off`** (nunca testado; `low` piorou ETTM2). Barato.

**H4 — Mais janelas de validação.** Os CSVs dos modelos têm várias linhas por
série; `n_windows` é escolha do chamador. Mais janelas ⇒ menos overfit, mas
janelas mais antigas valem menos para a janela de teste. Medir o oráculo e o
determinístico com `--windows 4/5` primeiro (determinístico, sem servidor).

**H5 — Peso maior no sinal cross-series e menor no in-series.** O DATASET CARD
e o pooled meta-model já existem; a medição indica que o agente decide quase
só pelo score de validação da própria série (que é ruído). Uma regra
determinística que misture prior do dataset + score da série (tipo FFORMA/ADE)
é o candidato clássico de ML e dispensa o LLM para a decisão final.

**H6 — (rejeitada) menos ferramentas.** `--drop-redundant-combine` piorou
ETTM2 porque removeu o caminho de vitória. Restringir o catálogo NÃO é a
direção; o problema não é o espaço de ação, é a função-objetivo.

**H7 — Teste de transferência no CI.** Adicionar um teste que mede, em
fixture, a correlação validação→teste das propostas (o que 2.3 mede) — o
sistema hoje não tem NENHUM teste sobre a propriedade que está falhando.

---

## 6. Plano de experimentos no servidor

Scripts prontos (não fazem nenhuma chamada de LLM nesta máquina):

- `diagnostics/diag_entry.py` — driver por braço (seed, reasoning, gate, ablações de
  contexto, `--use-llm 0` para o determinístico).
- `diagnostics/run_diagnostics_server.sh` — batelada completa: âncoras, 3 sementes,
  `reasoning off/low`, `calibration_gate`, `no-card`, `no-seed-pooled` em NN5
  (111 séries) + repetições em ETTM2. ~2,5 h no total. Logs de cada braço vão
  para `diagnostics/logs/<braço>_<dataset>.log`.
- `diagnostics/analyze_run.py` — **roda localmente** sobre a pasta trazida de volta:
  final vs melhor-semente-no-teste por série, origem, transferência das
  propostas, Spearman, verdicts, comparação com o braço determinístico.
  (Já validado: reproduz as tabelas da §2.)

O que trazer do servidor: as pastas inteiras
`timeseries/mestrado/resultados/orchestrator_react_diag_*` (CSV +
`llm_artifacts/`).

---

## 7. Referências

- Bailey, Borwein, López de Prado, Zhu (2014). *Pseudo-Mathematics and
  Financial Charlatanism* (AMS Notices) e *The Probability of Backtest
  Overfitting* (2015) — o mecanismo do §2.3/§3-L1.
- Gruver et al. (2023). *Large Language Models Are Zero-Shot Time Series
  Forecasters* (arXiv:2310.07820) — LLMs não batem baselines estatísticos
  simples em previsão pura; contexto para o papel do LLM aqui (decisão, não
  números).
- Montero-Manso & Hyndman (2021). FFORMA — combinação cross-learning; o mesmo
  princípio do pooled meta-model e do DATASET CARD.
- Wang et al. (2022). *Forecast combinations: an over 50-year review*
  (arXiv:2205.04216) — diversidade e médias simples como baselines fortes.
- *Efficient Model Selection for Time Series Forecasting via LLMs*
  (arXiv:2504.02119) e TimeCopilot (arXiv:2509.00616) — agentes para seleção
  de modelos; comparação de desenho (o que eles fazem de diferente: gate,
  verificação, uso de histórico cross-series).
