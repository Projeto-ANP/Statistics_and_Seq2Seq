# V11 — seleção final entre candidatos

Branch de trabalho: `v11_plus`
Base congelada: v10 (`argmin`) em `v3_proposta_final`
Status: ideias, achados empíricos e desenho da primeira versão (v0)

---

## 1. Objetivo

O v10 termina com `argmin` do score de validação: aplica a tentativa de menor
score. A v11 quer manter o loop do agente igual e mudar apenas a **decisão
final**:

1. montar uma shortlist de candidatos (não só o top1);
2. esconder nome de modelo, score, rank, tipo de combinação e origem;
3. pedir ao agente para escolher um candidato — ou manter o líder — com base
   nas características da série e na forma da previsão futura;
4. aplicar a escolha no teste, com fallback determinístico para o líder.

O teste real nunca entra no prompt nem na decisão. O agente só vê as previsões
futuras dos modelos base (que já existem antes do teste) e o passado da série.

---

## 2. O que os dados do v10x mostram

Análise feita sobre `ranking_trace.attempts` dos runs
`orchestrator_react_v10x_low` e `orchestrator_react_v10x_medium`
(321 séries por run nos 6 datasets oficiais, sem M4).

### 2.1 O top3 é estreito

- O melhor de teste está no top3 de validação em **29–30%** das séries.
- Rank médio de validação do melhor de teste: **~6,2**.
- No ETTH2, **0%** das vezes o melhor está no top3.
- No ETTM1, 14–43% (depende do run).

Oráculo por top-k (média dos datasets, `v10x_low`):

| shortlist | SMAPE de teste |
|---|---:|
| top1 (argmin) | 0,23956 |
| top2 | 0,23420 |
| top3 | 0,23337 |
| top5 | 0,23016 |
| top8 | 0,21711 |
| top12 (global) | 0,21585 |

Ou seja: expandir a shortlist captura mais, mas ainda é preciso escolher.

### 2.2 A ordem dentro do top3 é quase inútil

Quem ganha no teste dentro do top3:

| rank de validação | v10x_low | v10x_medium |
|---|---:|---:|
| 1º | 39,3% | 39,6% |
| 2º | 32,1% | 26,5% |
| 3º | 28,7% | 34,0% |

O spread de score entre 1º e 3º tem mediana de apenas **0,017**. A ordem é
quase arbitrária. Isso justifica cegar score/rank/nome.

### 2.3 O padrão mais forte: discordância

Quando os candidatos do top3 produzem previsões parecidas, não há o que
escolher. Quando discordam, existe ganho possível.

| discordância do top3 | ganho médio do oráculo | % séries com ganho > 0,01 |
|---|---:|---:|
| baixa (quartil inferior) | 0,0006–0,0007 | 0% |
| alta (quartil superior) | 0,0115–0,0151 | 30–31% |

O ganho do oráculo do top3 é concentrado:

- ganho mediano: **0,0005**;
- apenas **12,8%** das séries têm ganho > 0,01.

Conclusão: o seletor precisa ser conservador e, de preferência, só entrar
quando os candidatos divergem de verdade.

### 2.4 Forma da previsão ajuda pouco e depende do dataset

- O candidato que ganha no teste é ligeiramente **menos consenso / mais
  extremo** que os outros (distância à mediana do top3: d=+0,17; “é o mais
  extremo”: d=+0,37), mas isso inverte no ETTH1 e no NN5.
- “Seguir a tendência” não prediz nada (`trend_match` d≈-0,02).
- Aderência ao sazonal, slope e volatilidade têm efeitos pequenos.
- Um seletor livre treinado só com features de forma melhora o “acerto” para
  ~45%, mas **piora o SMAPE médio** (0,189 vs 0,188 do argmin). Erros grandes
  dominam; acertar mais vezes não basta.

### 2.5 Tipo de combinação importa mais que a forma

- `weighted` é o tipo mais comum do melhor global (44–45%).
- `dba` que sobrevive ao top3 tem o melhor SMAPE médio de teste
  (0,178–0,181), mas é catastrófico no ETT quando não sobrevive.
- `trimmed_mean` é supervalorizado pela validação e subperform a no teste.
- Tudo isso é dependente do dataset.

### 2.6 Shortlist diversificada cobre mais

Cobertura do melhor global por tipo de shortlist (`v10x_low`):

| shortlist | tamanho médio | cobertura | oráculo (média dos datasets) |
|---|---:|---:|---:|
| top3 puro | 3,0 | 29,6% | 0,2334 |
| melhor por família de combinação | 5,0 | 50,5% | 0,2289 |
| top1 + melhor por família + melhor por pool | 6,5 | 65,7% | 0,2195 |
| top1 + família + pool + weights | 10,5 | 85–100% | ~0,216 |

O ETTH2 é o caso extremo: top3 cobre 0%; top1 + família + pool cobre 57%.

---

## 3. Princípios de desenho

1. **Não mexer no loop do v10.** A escolha final é uma fase separada, depois do
   orçamento de 12 avaliações. Assim a trajetória do agente é idêntica e o
   ganho/perda é atribuível só à decisão final.
2. **Cegar de verdade.** Sem nome de modelo, score, rank, gap, tipo de
   combinação ou origem. A/B/C/... em ordem aleatória. O mapeamento fica só no
   artefato.
3. **Contexto novo.** Não reutilizar a conversa do ReAct: o agente já viu
   nomes e scores no loop e estaria ancorado.
4. **Veto, não escolha livre.** O default é manter o líder. O agente só troca
   se houver uma razão estrutural clara. `KEEP` é uma resposta válida.
5. **Fallback determinístico.** Parsing inválido, confiança baixa ou previsão
   degenerada → mantém o líder.
6. **Gate de discordância.** Só chamar o seletor quando os candidatos diferem
   o suficiente. Se forem quase idênticos, a escolha é cosmética.
7. **Avaliar com baseline determinístico.** O LLM precisa vencer argmin, MCS e
   ensemble. Se não vencer, a contribuição é a shortlist/gate, não o LLM.

---

## 4. v0 — primeira versão simples

Começar pelo mais simples: **melhor de cada família de combinação + escolha
cega, com `KEEP` e fallback**. Sem gate e sem pool na shortlist nesta primeira
versão; isso entra depois.

### 4.1 Shortlist

Famílias de combinação no histórico do v10:

- `weighted`
- `median`
- `dba`
- `trimmed_mean`
- `mean`

Para cada família presente no histórico, pegar a tentativa com **menor score
de validação que seja plausível**. Isso dá no máximo 5 candidatos e inclui
automaticamente o top1 global (o top1 é o melhor da sua família), desde que ele
não esteja fora de escala. Previsões degeneradas (ex.: máximo absoluto > 100×
a escala histórica) são descartadas antes do prompt; se a melhor tentativa de
uma família explode, tenta-se a próxima da mesma família.

Regras:

- deduplicar previsões numericamente idênticas ou com correlação > 0,9999;
- se sobrarem menos de 2 candidatos, pular o LLM e usar o líder;
- ordenar A/B/C/... com embaralhamento determinístico por
  `(dataset, series, seed)`;
- guardar o mapeamento `A -> attempt_id` no artefato.

### 4.2 O que o agente vê

- `SERIES PROFILE`: tendência, sazonalidade, estacionariedade, outliers,
  entropia, horizonte, frequência, tamanho.
- `LAST OBSERVED VALUE`: último valor e os últimos pontos, normalizados pela
  mesma escala dos candidatos (último = 0).
- `POOL CONTEXT`: mediana do pool e banda q10–q90 por horizonte, na mesma
  escala. É contexto, não alvo.
- `CANDIDATES`: para cada A/B/C/...:
  - trajetória futura normalizada (até 12 pontos; mesma escala para todos);
  - `level at h=1` (deslocamento em relação ao último valor);
  - `slope`;
  - `range`;
  - `volatility`;
  - `dist to last` (naive);
  - `dist to seasonal naive`;
  - `dist to pool median`;
  - `outside band` (% de pontos fora de [q10, q90]).

Não mostrar: nome do modelo, tipo de combinação, score de validação, rank,
gap, origem, rationale do agente.

### 4.3 Regras de decisão

- `CHOICE = KEEP` → aplica o líder (top1 da validação).
- `CHOICE = A..E` → aplica o candidato correspondente.
- `CONFIDENCE = low` e `CHOICE != KEEP` → `KEEP` (configurável).
- Parsing inválido → `KEEP`.
- Previsão escolhida não finita ou fora de escala (ex.: máximo absoluto
  > 100 × desvio histórico) → `KEEP`.
- Registrar `fallback`, `gate_skipped`, `chosen_id`, `leader_id`, mapeamento e
  a resposta bruta.

### 4.4 Prompt

System (em inglês, como o restante do harness):

```text
You are a forecast SELECTION agent.

A pool of already-trained forecasting models produced predictions for one time
series. A validation protocol shortlisted a few candidate COMBINATIONS. Each
candidate is a complete forecast for the same blind future window. The test
actual is NOT shown and must not be guessed.

Your job is to choose ONE candidate that is the most structurally plausible for
THIS series, or to KEEP the current leader if no candidate is clearly better.

Model names, validation scores, ranks, and combination types are deliberately
hidden: they are noisy and would bias the choice. Judge only the candidate
trajectories against the series' recent behavior and the pool context.

SELECTION RUBRIC (apply in this order):
1. Continuity: the forecast should start near the last observed value. A large
   unexplained jump is a warning sign.
2. Shape: the trajectory should be consistent with the series' recent trend and
   seasonal pattern, given the SERIES PROFILE. Do not mechanically extrapolate:
   a break in the trend can be legitimate, but it needs support in the profile.
3. Scale: the level, range and volatility must be plausible for this series and
   inside (or near) the pool's forecast band. An off-scale or exploding
   candidate is a strong reason to reject it.
4. Robustness: do not prefer a candidate just because it is the most central or
   the most extreme. Use the profile: strongly seasonal -> check the seasonal
   phase/amplitude; strongly trending -> check direction and magnitude; noisy ->
   prefer a stable trajectory unless the profile justifies a break.
5. If the candidates are nearly indistinguishable, KEEP. Do not override the
   leader just to change something.

Answer exactly three lines, nothing else:
CHOICE: <A|B|C|D|E|KEEP>
CONFIDENCE: <low|medium|high>
REASON: <one or two sentences citing the series characteristics you used>
```

User template:

```text
SERIES PROFILE:
{profile}

LAST OBSERVED VALUE:
last = {last_actual}
recent scaled path (last = 0): {last_path}

POOL CONTEXT (all base models, blind window, same scale):
scaled median: {pool_median_path}
q10-q90 band: {pool_band_path}
dispersion: {pool_dispersion}

CANDIDATES (same scale; each is the best of a different hidden family on
validation; the test actual is NOT shown):

A.
  scaled forecast: {a_path}
  level at h=1: {a_level}
  slope: {a_slope}
  range: {a_range}
  volatility: {a_volatility}
  dist to last: {a_naive_dist}
  dist to seasonal naive: {a_seasonal_dist}
  dist to pool median: {a_pool_dist}
  outside band: {a_outside}

B.
  ...

C.
  ...

REMINDER: choose the candidate that is most plausible for THIS series, or KEEP
if none is clearly better. Do not use hidden names, scores, or ranks.
```

### 4.5 O que NÃO entra na v0

- gate de discordância;
- melhor por pool;
- top-k (5/8);
- MCS;
- pairwise;
- meta-ranker.

Tudo isso fica para as versões seguintes, para isolar o efeito de cada
mudança.

---

## 5. Implementação

### 5.1 Configuração

Adicionar em `ReactConfig`:

```python
final_strategy: Literal[..., "agent_family"] = "argmin"
final_selector: LLMRole = field(default_factory=LLMRole)
final_selector_max_candidates: int = 5
final_selector_allow_keep: bool = True
final_selector_low_confidence_keep: bool = True
final_selector_shuffle_seed: int = 0
final_selector_degenerate_factor: float = 100.0
```

### 5.2 Código

- Novo módulo `orchestrator_react/final_choice.py`:
  - `build_family_shortlist(state, max_candidates=5)`;
  - `blind_candidates(shortlist, dataset, series, seed)`;
  - `build_final_prompt(...)`;
  - `parse_choice(raw, labels, allow_keep)`;
  - `choose_final(...) -> (Attempt, debug)`.
- `state.py`: helper `attempts_by_family()` ou a lógica fica no módulo novo.
- `pipeline.py`: branch na Phase 4 para `final_strategy == "agent_family"`,
  antes de `apply_to_test`.
- `run_tsf_orchestrator.py`: flags
  `--final-strategy agent_family`, `--final-selector-model`,
  `--final-selector-temperature`, `--final-selector-reasoning`,
  `--no-final-selector-allow-keep`, `--final-selector-gate`,
  `--final-selector-gate-corr`, `--final-selector-seed`.
- Artefato: bloco `final_choice` com:
  - `shortlist` (A/B/C -> attempt_id, spec real, score real — só no artefato);
  - `prompt`;
  - `raw_response`;
  - `choice`, `confidence`, `reason`;
  - `fallback`, `gate_skipped`;
  - `leader_id`;
  - métricas de teste de cada candidato (pós-hoc, só para análise).
- Testes em `tests/test_final_choice.py`:
  - shortlist por família;
  - dedup;
  - máscara/embaralhamento;
  - parsing (`A`, `KEEP`, inválido);
  - fallback por confiança baixa;
  - fallback por previsão degenerada.

### 5.3 Compatibilidade

- `argmin` continua default e bit-idêntico ao v10.
- `ensemble`, `mcs`, `seed_gate` e `prior_blend` continuam disponíveis.
- O loop e o score de validação não mudam.

### 5.4 Status da implementação (v11_plus)

- `orchestrator_react/final_choice.py` — shortlist, contexto, prompt cego,
  parsing e decisão.
- `orchestrator_react/config.py` — `final_strategy="agent_family"` e os campos
  `final_selector_*`.
- `orchestrator_react/pipeline.py` — Phase 3.5 (escolha final) e Phase 4
  (`apply_to_test`).
- `orchestrator_react/csv_writer.py` — bloco `final_choice` no artefato.
- `run_tsf_orchestrator.py` — flags `--final-selector-*`.
- `tests/test_final_choice.py` — 16 testes (shortlist, máscara, prompt,
  parsing, fallback, pipeline e CLI).
- `aprovacao_solida/10_final_choice_replay/` — replay offline e README.

A v0 implementada usa a shortlist por família, o filtro de previsão degenerada,
o prompt cego e a política `KEEP`/fallback. O gate de discordância, a shortlist
com pool e o pairwise continuam como próximas versões.

---

## 6. Replay offline

O script `aprovacao_solida/10_final_choice_replay/replay_final_choice.py`
implementa este replay. Ele lê os artefatos, monta o prompt cego e grava
`results.csv`, `summary.csv` e `prompts.jsonl`.

Os artefatos do v10x já têm, por tentativa:

- `validation_score`, `validation_rank`;
- `test_forecast` (a previsão futura do candidato);
- `test_smape`, `test_rank` (só para avaliação).

Isso permite montar a shortlist e o prompt **sem rodar o loop de novo**. O
`test_forecast` é exatamente o que o agente veria na decisão final; o
`test_smape` nunca entra no prompt.

Protocolo sugerido:

1. reconstruir a shortlist por família a partir de `ranking_trace.attempts`;
2. carregar a série e o pool (`state.y_true`, `state.test_preds`,
   `state.train_series`) para montar o contexto;
3. gerar o prompt cego;
4. chamar o LLM (mesmo `gpt-oss:20b`, temperature baixa, seed fixo);
5. avaliar o candidato escolhido contra:
   - argmin;
   - melhor da shortlist por validação (geralmente o líder);
   - oráculo da shortlist;
   - oráculo global;
   - MCS;
   - ensemble.
6. separar um subconjunto de séries para desenvolver o prompt e outro para a
   avaliação final. Não escolher o prompt olhando o teste inteiro.

Métricas:

- SMAPE/POCID de teste por dataset (média dos 6);
- taxa de acerto (escolheu o melhor de teste da shortlist);
- regret vs argmin;
- taxa de override;
- win/loss por série;
- taxa de `KEEP`, `fallback` e `gate_skipped`;
- latência/custo por série.

---

## 7. Baselines e ablações

Ordem de execução:

| versão | o que muda | pergunta |
|---|---|---|
| v0 | top por família + prompt cego + KEEP | o LLM agrega sobre argmin? |
| v0b | v0 sem KEEP | o veto é necessário? |
| v1 | + gate de discordância | o gate evita overrides ruins? |
| v2 | + melhor por pool na shortlist | cobrir o melhor global ajuda? |
| v3 | + pairwise (A vs B, ...) | reduz viés de posição? |
| v4 | shortlist MCS | o conjunto estatístico é melhor que top por família? |
| v5 | meta-ranker/FFORMA-style | features entre séries vencem o LLM? |

Baselines determinísticos em todas as versões:

- argmin (v10);
- melhor da shortlist por validação;
- MCS (média do conjunto de confiança);
- ensemble dos top-k;
- escolha aleatória;
- oráculo da shortlist;
- oráculo global.

---

## 8. Riscos e cuidados

- **Overfitting no teste**: o seletor é avaliado no teste; iterar o prompt
  olhando o resultado completo vira leakage. Usar split de desenvolvimento e
  avaliação, ou pré-registrar o prompt.
- **Ganho concentrado**: o oráculo do top3 melhora pouco na mediana. O seletor
  pode facilmente empatar ou piorar. O desenho conservador (`KEEP`) é
  intencional.
- **Candidatos quase idênticos**: deduplicar e/ou gate. Sem isso, a escolha é
  cosmética e o LLM pode ser prejudicado por ruído.
- **Viés de posição**: embaralhar A/B/C e registrar o mapeamento. Se possível,
  rodar duas ordens e medir consistência.
- **Identidade implícita**: a forma da previsão pode denunciar o modelo (ex.:
  NaiveSeasonal). Isso é aceitável; o que não pode vazar é o rank/score.
- **Custo**: uma chamada extra por série é barato comparado às 12 do loop, mas
  o gate evita chamadas desnecessárias.

---

## 9. Roadmap

1. Criar `final_choice.py` + config + artefato.
2. Rodar replay offline da v0 em um subconjunto (ex.: 50 séries com alta
   discordância e 50 com baixa).
3. Ajustar o prompt só no split de desenvolvimento.
4. Rodar a v0 completa nos 6 datasets.
5. Se o LLM não vencer o argmin, testar v1 (gate) e v2 (pool) antes de
   descartar.
6. Comparar com MCS/ensemble. Se o determinístico vencer, essa passa a ser a
   contribuição principal e o LLM vira ablação.

---

## 10. Referências

**Seleção vs combinação**
- Bates & Granger (1969), *The Combination of Forecasts*.
- Timmermann (2006), *Forecast Combinations*.
- Stock & Watson (2004), *Combination Forecasts of Output Growth in a
  Seven-Country Data Set*.
- Genre, Kenny, Meyler & Timmermann (2013), *Combining Expert Forecasts: Can
  Anything Beat the Simple Average?*
- Smith & Wallis (2009), *A Simple Explanation of the Forecast Combination
  Puzzle*.

**Conjunto de confiança / avaliação**
- Hansen, Lunde & Nason (2011), *The Model Confidence Set*.
- Diebold & Mariano (1995), *Comparing Predictive Accuracy*.
- White (2000), *A Reality Check for Data Snooping*.
- Hansen (2005), *A Test for Superior Predictive Ability*.

**Winner's curse / backtest overfitting**
- Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and
  Financial Charlatanism*.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
  Overfitting*.

**Meta-learning / seleção por features**
- Montero-Manso, Athanasopoulos, Hyndman & Talagala (2020), *FFORMA:
  Feature-based Forecast Model Averaging*.
- Rice (1976), *The Algorithm Selection Problem*.
- Smith-Miles (2009), *Cross-Disciplinary Perspectives on Meta-Learning for
  Algorithm Selection*.

**LLM reranking / juiz**
- Sun et al. (2023), *Is ChatGPT Good at Search? Investigating Large Language
  Models as Re-Ranking Agents*.
- Qin et al. (2023), *Large Language Models are Effective Text Rankers with
  Pairwise Ranking Prompting*.
- Pradeep, Sharifymoghaddam & Lin (2023), *RankVicuna: Zero-Shot Listwise
  Document Reranking with Open-Source LLMs*.
- Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena*.

**Contexto de séries temporais**
- Ansari et al. (2024), *Chronos: Learning the Language of Time Series*.
- Garza & Mergenthaler-Canseco (2023), *TimeGPT-1*.

---

## 11. Resultado da v0 (replay offline)

Run `orchestrator_react_v10x_low`, 321 séries, 6 datasets. O replay usa as
trajetórias do v10x; o teste real só entra na avaliação.

| política | SMAPE médio dos datasets | SMAPE médio por série | ganho vs argmin (datasets) |
|---|---:|---:|---:|
| argmin (v10) | 0,239556 | 0,188715 | — |
| LLM v0 (top por família, cego) | 0,254375 | 0,190183 | -0,014819 |
| `closest_seasonal` (sem gate) | 0,234874 | 0,190863 | +0,004683 |
| `seasonal_guard` (gate) | 0,233789 | 0,187897 | +0,005767 |

Detalhes da v0:

- override em 62,9% das séries, `KEEP` em 37,1%, nenhum fallback;
- confiança: 157 high, 162 medium, 2 low;
- overrides com `high` tiveram ganho -0,0030; `medium` +0,00002 — confiança
  não está calibrada;
- o escolhido foi o melhor da shortlist em 20,9% das séries, contra 24,3% do
  argmin; regret médio vs shortlist +0,01075. O LLM é pior que o líder em
  escolher dentro da própria shortlist;
- ajudou no ANP (+0,00168), mas quebrou no ETTM1 (-0,07046), ETTH1 (-0,00888),
  ETTM2 (-0,00503), ETTH2 (-0,00485) e NN5 (-0,00137);
- o viés do prompt para “começar perto do último valor / ficar dentro da banda
  do pool” coincide com a regra `closest_pool_median`, que é ruim
  (-0,02565 vs argmin).

O achado mais forte é o **seasonal guard** determinístico:

- gate: `seasonal_strength >= 0.5` e `n_points >= 10 * seasonal_period`;
- se o gate é verdadeiro, escolhe o candidato da shortlist com menor
  `seasonal_dist` (mais próximo do seasonal naive); senão, mantém o argmin;
- low: +0,005767 na média dos datasets e +0,000818 na média por série;
- medium: +0,004927 na média dos datasets e +0,000707 na média por série;
- troca em ~15% das séries; ganho concentrado em ETTM1/ETTH1, sem perder no
  ANP/NN5 (o gate exclui esses casos na prática).

Conclusão: **a v0 do LLM não está aprovada**. Não rodar o passo 5 ainda. O
próximo passo é a v0.1:

1. implementar `final_strategy="seasonal_guard"` como baseline determinístico;
2. reduzir o LLM a um veto binário: `KEEP` vs o candidato mais próximo do
   seasonal naive, somente quando o gate é verdadeiro;
3. comparar argmin × seasonal_guard × seasonal_guard + veto do LLM;
4. só rodar o pipeline completo (passo 5) se a v0.1 vencer o seasonal_guard.

Ressalva: o medium replica as trajetórias com outro seed/reasoning, mas usa as
mesmas séries e janelas de teste; não é um teste independente. Os parâmetros do
gate (0,5 e 10 ciclos) são escolhas de desenho e devem ser tratados como tal.

### 11.1 v0.1 implementada

- `final_choice.choose_seasonal_guard` e `final_choice.choose_seasonal_veto`;
- `ReactConfig.final_strategy` aceita `"seasonal_guard"` e `"seasonal_veto"`;
- flags `--seasonal-guard-min-strength` e `--seasonal-guard-min-cycles`;
- o veto binário usa o mesmo role `final_selector`, chama o LLM apenas quando o
  guard quer trocar e cai no guard em erro de LLM/parsing;
- `aprovacao_solida/10_final_choice_replay/compare_three.py` compara
  argmin × guard × guard+veto nas mesmas trajetórias;
- testes em `tests/test_final_choice.py` cobrem gate, guard, veto, baixa
  confiança, erro de LLM, prompt binário e integração no pipeline.

Comandos no README da pasta 10. O dry-run do `compare_three.py` reproduziu os
números do seasonal baseline: low +0,005767 e medium +0,004927 na média dos
datasets. Falta a run real do veto binário (LLM) no servidor para decidir se ele
agrega sobre o guard. Próximo passo: rodar o `compare_three.py` com o LLM no low
e no medium, e só levar para o pipeline completo a política que vencer o guard
determinístico.

---

## 12. Nota de branches

Hoje `v3_proposta_final` e `v11_plus` apontam para o mesmo commit. Antes de
divergir, garantir que a v10 está congelada em `v3_proposta_final` (tag/merge)
e que o desenvolvimento da v11 acontece só em `v11_plus`.
