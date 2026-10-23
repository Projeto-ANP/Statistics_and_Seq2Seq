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

### 11.2 Resultado da comparação três-a-três

O dry-run do `compare_three.py` reproduziu o seasonal baseline. A run real do
veto binário foi feita no low e no medium:

| run | argmin | guard | guard+veto | veto vs guard |
|---|---:|---:|---:|---:|
| low | 0,239556 | 0,233789 | 0,235905 | **-0,002116** |
| medium | 0,238025 | 0,233099 | 0,236058 | **-0,002959** |

- O LLM foi chamado em 48 (low) / 49 (medium) séries, as que o guard queria
  trocar; aceitou a troca em 28/33 e manteve o líder em 20/16.
- `veto_accuracy` (dataset mean): 0,590 low / 0,350 medium. Nos quartis de
  `seasonal_gap` o veto ficou perto de chance.
- O dano está concentrado no ETTH1: na série 0, o guard melhorava
  +0,095 (low) / +0,115 (medium), mas o LLM manteve o líder com confiança
  `high`, dizendo que o líder seguia melhor o ciclo sazonal. A série 2 do ETTH1
  repete o padrão. Esses dois casos anulam o ganho do guard no ETT.
- As razões do LLM são plausíveis, mas não predizem o teste. O veto binário não
  agrega de forma robusta.

**Decisão:** adotar `seasonal_guard` como estratégia final da v11. O veto binário
fica registrado como resultado negativo/ablação: o LLM agrega na busca do v10,
não na escolha final entre líder e candidato sazonal.

Próximo passo: rodar o pipeline completo com `--final-strategy seasonal_guard`
para gerar os artefatos oficiais e comparar com o v10. O comando está no README
da pasta 10.

### 11.3 Mineração de regras nas features do agente

Além do seasonal guard, foi feita uma varredura sistemática de ~50 regras
determinísticas sobre a shortlist e as features que o agente vê: min/max de
`level_1`, `slope`, `range`, `volatility`, `naive_dist`, `seasonal_dist`,
`pool_dist`, `outside_band`, `validation_score`; regras condicionais por perfil
(`seasonal_strength`, ciclos, `trend_strength`, entropia); margens; filtros de
banda; e "não-líder mais próximo do líder". Tudo avaliado por série, comparando
com o argmin e com o oráculo da shortlist, no low e no medium.

| regra | low ds / série | medium ds / série |
|---|---:|---:|
| `seasonal_guard` | +0,005767 / +0,000818 | +0,004927 / +0,000707 |
| `guard + max_outside (baixa sazonalidade)` | +0,006101 / +0,001380 | +0,005187 / +0,000864 |
| `max_outside_band` sozinho | +0,002364 / +0,000362 | +0,002204 / +0,000037 |
| `two_anchor` (sazonal + naive) | -0,000446 / +0,001442 | -0,003163 / +0,000350 |
| `seasonal_margin_0.02` | +0,005446 / -0,002116 | +0,002284 / -0,003323 |
| argmin | 0 / 0 | 0 / 0 |

O achado secundário é `max_outside_band`: escolher o candidato com a maior
fração de pontos fora da banda q10–q90 do pool. Sozinho, ele é positivo nas duas
médias, mas negativo no NN5; combinado com o guard apenas para séries de
sazonalidade fraca e ciclos suficientes, ele adiciona +0,00095 (ANP) no low e
+0,00023 no medium, com ponto estimado positivo mas **IC 95% incluindo zero**.

As regras intuitivas falham: escolher o mais próximo do último valor
(-0,0206 low / -0,0232 medium), o mais próximo da mediana do pool, o maior
slope/range/volatilidade, o que “segue a tendência”, o não-líder mais próximo do
líder, ou o melhor de validação dentro da banda. O viés de continuidade/
centralidade que o LLM mostrava também é ruim nos dados.

Decisão: manter `seasonal_guard` como estratégia final. A adição de
`max_outside_band` para séries de baixa sazonalidade fica como hipótese para uma
run independente; o ganho sobre o guard é pequeno e o IC inclui zero. A varredura
de ~50 regras na mesma base também tem risco de múltiplas comparações; o medium
replica trajetórias, mas não é um teste independente.

Script: `aprovacao_solida/10_final_choice_replay/mine_rules.py`.
Saída: `outputs/resultados/v11_rule_mining/rule_summary.csv`.

### 11.4 Comparação com FFORMA, ADE e v10

SMAPE (teste), 321 séries, mesmas janelas de teste:

| dataset | FFORMA | ADE | v10 low | v10 medium | guard low | guard medium |
|---|---:|---:|---:|---:|---:|---:|
| ANP_MONTHLY | **0,216593** | 0,217722 | 0,219567 | 0,218291 | 0,219451 | 0,218177 |
| ETTH1 | 0,300054 | 0,395094 | 0,240080 | 0,241423 | 0,226407 | **0,224214** |
| ETTH2 | 0,203528 | 0,226148 | **0,174797** | 0,175173 | 0,176668 | 0,177029 |
| ETTM1 | **0,439569** | 0,457223 | 0,509871 | 0,505415 | 0,487956 | 0,491870 |
| ETTM2 | **0,165421** | 0,187240 | 0,176754 | 0,171824 | 0,175984 | 0,171275 |
| NN5_WEEKLY_DATASET | 0,119654 | 0,117800 | 0,116270 | **0,116027** | 0,116270 | **0,116027** |
| **MÉDIA DAS MÉDIAS** | 0,240803 | 0,266871 | 0,239556 | 0,238025 | 0,233789 | **0,233099** |
| **AGREGADO (321 séries)** | 0,188354 | 0,191779 | 0,188715 | 0,187741 | 0,187897 | **0,187034** |

POCID (maior é melhor):

| resumo | FFORMA | ADE | v10 low | v10 medium | guard low | guard medium |
|---|---:|---:|---:|---:|---:|---:|
| MÉDIA DAS MÉDIAS | 52,357 | 51,446 | 53,526 | 53,465 | **54,234** | 53,638 |
| AGREGADO (321 séries) | **56,829** | 55,337 | 55,759 | 56,591 | 55,797 | 56,505 |

Leitura:

- O guard medium é o melhor SMAPE nas duas agregações; o guard low é o melhor
  POCID na média das médias.
- Contra o v10, o guard melhora ANP, ETTH1, ETTM1 e ETTM2 e piora um pouco o
  ETTH2; no NN5 não há troca. O ganho é concentrado em ETTH1/ETTM1.
- Contra o FFORMA, o guard ganha nas duas agregações de SMAPE (0,2331 vs 0,2408
  na média; 0,1870 vs 0,1884 no agregado), mas o FFORMA continua melhor em
  ANP, ETTM1 e ETTM2. O agregado de POCID do FFORMA (56,829) ainda é o melhor.
- O ADE é o mais fraco em SMAPE nas duas agregações, embora tenha um ANP
  competitivo.
- O ganho do guard não é uniforme: é uma aposta em ETTH1/ETTM1, onde o
  seasonal naive é confiável; por isso o gate exclui ANP/NN5.

Os números de v10 são os runs `v10x_low`/`v10x_medium` (mesmas trajetórias do
guard). O doc histórico do v10 traz 0,239104 na média, praticamente igual ao
v10x_low (0,239556). FFORMA/ADE vêm dos CSVs de `timeseries/mestrado/resultados`.
As tabelas completas estão em `comparison_smape.csv` e `comparison_pocid.csv`.

### 11.5 Sem ETTM1/ETTM2

Tirando os dois ETTM (4 datasets, 307 séries), o SMAPE fica:

| resumo | FFORMA | ADE | v10 low | v10 medium | guard low | guard medium |
|---|---:|---:|---:|---:|---:|---:|
| MÉDIA DAS MÉDIAS | 0,209957 | 0,239191 | 0,187678 | 0,187729 | 0,184699 | **0,183862** |
| AGREGADO (307 séries) | 0,183148 | 0,185831 | 0,181665 | 0,180860 | 0,181327 | **0,180443** |

- O guard medium continua o melhor nas duas agregações de SMAPE. O ganho sobre
  o v10 medium cai para +0,003867 na média e +0,000417 no agregado, mas continua
  positivo.
- O POCID muda de figura: sem ETTM, a média das médias é do FFORMA (57,883);
  v10 medium 56,595, guard medium 56,545. No agregado, FFORMA 57,537,
  v10 medium 57,019, guard medium 56,901. O ganho de POCID do guard era
  concentrado no ETTM1.
- O ganho de SMAPE do guard **não é só ETTM**: o ETTH1 é o principal
  contribuinte; por isso o guard ainda vence sem os ETTM. Mas a vantagem de
  POCID desaparece.

Tabelas: `comparison_smape_sem_ETTM.csv` e `comparison_pocid_sem_ETTM.csv`.

### 11.6 Tabela reduzida: guard low × FFORMA/ADE × median/mean/dba

SMAPE (teste), 321 séries:

| dataset | guard low | FFORMA | ADE | median | mean | dba |
|---|---:|---:|---:|---:|---:|---:|
| ANP_MONTHLY | 0,219451 | **0,216593** | 0,217722 | 0,219412 | 0,220649 | 0,225205 |
| ETTH1 | **0,226407** | 0,300054 | 0,395094 | 0,260200 | 2,000000 | 1,999999 |
| ETTH2 | **0,176668** | 0,203528 | 0,226148 | 0,181522 | 1,719699 | 1,725712 |
| ETTM1 | 0,487956 | **0,439569** | 0,457223 | 0,544269 | 2,000000 | 1,999999 |
| ETTM2 | 0,175984 | **0,165421** | 0,187240 | 0,199735 | 1,740484 | 1,743807 |
| NN5_WEEKLY_DATASET | **0,116270** | 0,119654 | 0,117800 | 0,120134 | 0,119939 | 0,122563 |
| **MÉDIA DAS MÉDIAS** | **0,233789** | 0,240803 | 0,266871 | 0,254212 | 1,300128 | 1,302881 |
| **AGREGADO (321 séries)** | **0,187897** | 0,188354 | 0,191779 | 0,191800 | 0,329260 | 0,332955 |

POCID (média das médias / agregado):

| resumo | guard low | FFORMA | ADE | median | mean | dba |
|---|---:|---:|---:|---:|---:|---:|
| MÉDIA DAS MÉDIAS | **54,234** | 52,357 | 51,446 | 54,048 | 49,413 | 49,821 |
| AGREGADO (321 séries) | 55,797 | **56,829** | 55,337 | 55,808 | 54,632 | 54,576 |

Atenção: `mean` e `dba` são a combinação do pool completo e explodem no ETT
(SMAPE ~2,0) porque o pool tem previsões com escala absurda. O `median` é o
único simples robusto no ETT, mas ainda perde do guard e do FFORMA. O guard low
ganha nas duas agregações de SMAPE; no POCID agregado, o FFORMA ainda lidera.

Tabelas: `comparison_smape_guard_low_simple.csv` e
`comparison_pocid_guard_low_simple.csv`.

### 11.7 Sensibilidade dos parâmetros do gate

Grade de `seasonal_strength` (0,0–0,9) × ciclos mínimos (0–50), no low e no
medium. Ganho vs argmin:

| strength | ciclos | low ds / série | medium ds / série | gate low |
|---:|---:|---:|---:|---:|
| 0,3 | 10 | +0,005970 / +0,000682 | +0,005134 / +0,000564 | 115 |
| **0,4** | **10** | **+0,006209 / +0,000850** | **+0,005159 / +0,000717** | 88 |
| 0,5 | 10 | +0,005767 / +0,000818 | +0,004927 / +0,000707 | 64 |
| 0,6 | 10 | +0,005768 / +0,000822 | +0,004920 / +0,000705 | 41 |
| 0,4 | 2 | +0,005878 / +0,000165 | +0,004691 / -0,000253 | 199 |
| 0,4 | 50 | +0,006197 / +0,000811 | +0,005146 / +0,000673 | 14 |
| 0,5 | 50 | +0,005748 / +0,000752 | +0,004908 / +0,000642 | 11 |

- `min_cycles` precisa ser >= 5 para excluir o NN5; com 0/2 o NN5 entra e a
  média por série cai.
- A região 0,4–0,6 de strength é um platô; 0,4/10 é o melhor ponto estimado nos
  dois runs, mas a diferença sobre 0,5/10 é ~0,0002–0,0004 no ds e ~0,00001–
  0,00003 por série — dentro do ruído.
- Não há validação independente para escolher 0,4; o 0,5/10 era o valor
  pré-especificado. Recomendação: manter 0,5/10 como default e reportar a
  sensibilidade; 0,4/10 só se quiser o melhor ponto estimado, declarando que foi
  escolhido na mesma base.

Grade completa: `outputs/resultados/v11_rule_mining/seasonal_gate_sensitivity.csv`.

### 11.8 STL/tendência como âncora

Testamos duas formas de usar STL/tendência na escolha final:

1. os “champions” de tendência/sazonalidade que o `series_profile` já calcula
   via STL (o modelo base cujo componente STL melhor casou na validação);
2. uma âncora de tendência linear ajustada no histórico recente.

Regras e ganho vs argmin:

| regra | low ds / série | medium ds / série |
|---|---:|---:|
| **seasonal_guard** | **+0,005767 / +0,000818** | **+0,004927 / +0,000707** |
| `min_stl_season_dist` | -0,017624 / -0,003452 | -0,011190 / -0,003878 |
| `min_stl_trend_dist` | -0,025634 / -0,003912 | -0,027540 / -0,005483 |
| `champ_by_strength` | -0,022934 / -0,002939 | -0,021489 / -0,004347 |
| `champ_by_score` | -0,026032 / -0,004789 | -0,027833 / -0,006460 |
| `min_trend_anchor_dist` | -0,028054 / -0,002901 | -0,033439 / -0,004348 |
| `trend_if_strong` | -0,019467 / -0,001172 | -0,020256 / -0,002017 |
| `trend_or_seasonal` | -0,017162 / -0,000787 | -0,017366 / -0,001550 |

Todas as âncoras STL/tendência ficaram **piores que o líder**, com taxa de troca
de 69–80% e loss rate maior que win rate. O motivo: o champion é um modelo
único escolhido na validação e a tendência linear não se sustenta na janela de
teste; já o seasonal naive (repetir o último ciclo) é uma âncora robusta e
simples. O guard sazonal continua sendo o único ganho consistente.

A decomposição STL completa de “histórico + previsão do candidato” só é viável
no servidor (statsmodels) e não foi priorizada: o resultado dos champions e da
tendência não é encorajador. Script: `mine_stl_rules.py`; saída:
`outputs/resultados/v11_rule_mining/stl_rule_summary.csv`.

### 11.9 Componentes STL, tendência e ruído

A lógica testada foi:

1. decompor a série em tendência, sazonalidade e resíduo (STL);
2. criar três âncoras: tendência extrapolada, ciclo sazonal repetido e nível de
   ruído/forecastability;
3. comparar cada candidato às âncoras e escolher o de menor distância
   ponderada pelas forças dos componentes;
4. usar o ruído como gate: não trocar quando a série é muito “limpa” e a
   diferença entre candidatos é essencialmente ruído de validação.

Evidência local:

- âncoras de tendência/STL champion: todas negativas (seção 11.8);
- âncora sazonal (seasonal guard): positiva e robusta;
- gate de ruído em cima do guard:

| regra | low: ds / série | medium: ds / série | trocas | loss |
|---|---:|---:|---:|---:|
| guard | +0,005767 / +0,000818 | +0,004927 / +0,000707 | 15,0% / 15,3% | 8,4% / 7,8% |
| `spectral_entropy >= 0,45` | +0,005975 / +0,000926 | +0,005158 / +0,000781 | 4,7% / 4,4% | 1,6% / 1,6% |
| `coef_variation >= 0,5` | +0,005962 / +0,000880 | +0,005148 / +0,000748 | 3,1% / 3,7% | 0,9% / 1,6% |
| ambos | +0,005964 / +0,000887 | +0,005155 / +0,000769 | 1,6% / 1,6% | 0% / 0% |

O gate duplo troca só 5 séries nas 321 e acerta todas nos dois runs; o ganho
sobre o guard é pequeno (~+0,0002 ds, ~+0,00007 por série), mas o perfil de
risco melhora muito (sem losses, muito menos trocas). O ganho continua vindo de
ETTH1/ETTM1; o gate só remove as trocas ruins em ETTH2/ETTM2/ANP.

Literatura que fundamenta a lógica:

- decomposição e previsão por componente: Hyndman & Athanasopoulos, *Forecasting:
  Principles and Practice*, seção “Forecasting with decomposition” (prever a
  série dessazonalizada e o componente sazonal separadamente e recombinar);
- múltiplas agregações temporais / MAPA: Kourentzes, Petropoulos & Trapero
  (2014), *Improving forecasting by estimating time series structural components
  across multiple frequencies*;
- Theta method: Assimakopoulos & Nikolopoulos (2000), decomposição em theta
  lines (tendência/curvatura);
- Prophet: Taylor & Letham (2018), modelo aditivo trend + seasonality + noise;
- seleção por features/meta-learning: FFORMA (Montero-Manso et al., 2020), que
  já é baseline do projeto;
- forecastability/entropy: spectral entropy como medida de previsibilidade
  (Goerg, 2013); é a base do gate de ruído.

Decisão: não adotar a âncora de tendência nem o champion STL. O gate de ruído é
uma hipótese promissora de **v11.1** (guard + `spectral_entropy >= 0.45` e
`coef_variation >= 0.5`), mas o ganho absoluto é pequeno e foi achado na mesma
base; precisa de uma run independente antes de virar default. O guard puro
continua sendo a escolha conservadora.

Saída da grade de ruído: `outputs/resultados/v11_rule_mining/noise_gate_summary.csv`.

### 11.10 Nova combinação para o ETTM: top-k por horizonte

Busca de ~60 receitas no ETTM1/ETTM2. O achado mais forte **não é top1 na
validação**, mas é uma opção nova que o guard escolhe pela forma sazonal:

```text
pool  = top7 por SMAPE na validação
combine = weighted
weights = topk_mean_per_horizon(k=5, metric=smape)
```

O agente não tentou essa combinação. Ela é fraca na validação (não entra no top1
nem no top3 do ETTM1), mas no teste do ETTM1 dá 0,36396 contra 0,50987 do
argmin e 0,43957 do FFORMA.

Adicionando essa receita como candidata extra no shortlist do seasonal guard
(o guard escolhe a menor `seasonal_dist`), respeitando o gate:

| run | dataset | argmin | guard | guard + topk |
|---|---|---:|---:|---:|
| low | ETTM1 | 0,509871 | 0,487956 | **0,337102** |
| low | ETTH1 | 0,240080 | 0,226407 | 0,226070 |
| low | ETTM2 | 0,176754 | 0,175984 | 0,176014 |
| low | ANP | 0,219567 | 0,219451 | 0,219540 |
| low | ETTH2 | 0,174797 | 0,176668 | 0,176668 |
| low | NN5 | 0,116270 | 0,116270 | 0,116270 |
| low | **média dos datasets** | 0,239556 | 0,233789 | **0,208611** |
| medium | **média dos datasets** | 0,238025 | 0,233099 | **0,208035** |

No ETTM1, o topk é escolhido em 2/7 séries, justamente as duas em que o argmin
explode (1,137 e 1,325); o topk cai para 0,645 e 0,762. No ETTM2/NN5 ele não é
usado; no ANP é usado em ~12% das séries com uma perda média de 0,00009; no
ETTH1 melhora 0,0003.

Isso confirma a hipótese: **o gargalo do ETTM não é falta de boa combinação, é a
validação não rankear essa combinação no topo.** A saída é semear a receita e
deixar o guard escolhê-la pela forma sazonal, não pelo score de validação.

Ressalvas: achado em 7 séries do ETTM1, com ganho concentrado em 2 séries; é uma
hipótese forte, mas precisa de validação no pipeline completo/run independente.
Para evitar a pequena perda no ANP, o seed do topk pode ser limitado a séries com
`cycles >= 100` (ETT/ETTH), mantendo ANP/NN5 fora.

Implementação v11.1 (sem seed): em vez de calcular e injetar a receita, o
agente recebe ferramentas e exemplos para chegar nela:

1. `--direct-preparation`: libera `select_*`/`weights_*` no modo
   `--scoring-budget` (sem consumir o orçamento de scores);
2. exemplo `select_top_k(k=7, metric="smape")` →
   `weights_topk_per_horizon(k=5, metric="smape")` → `evaluate_strategy`;
3. dica condicional para séries sazonais com muitos ciclos;
4. `OPTIONAL TOOL CHAIN` no `NEXT UNTRIED`;
5. com `--no-seed-attempts`, esconder os pools estáveis da Phase 2 dos handles e
   do menu, para o agente não começar ancorado;
6. no shortlist final, agrupar `weighted` por método de peso e usar
   `--final-selector-max-candidates 10`, para o guard ver a família topk se o
   agente a criar.

Se o agente chegar na receita, o guard escolhe pela `seasonal_dist`; se não,
testar reasoning/modelo diferente antes de considerar seed.

CSVs: `topk_guard_low.csv`, `topk_guard_medium.csv` em
`outputs/resultados/v11_combo_search/`.

### 11.11 Run completa do v11_seasonal (low)

A run real com `--final-strategy seasonal_guard` foi feita nos 6 datasets. O
resultado **não** confirmou o ganho offline:

| dataset | v11_seasonal | v10x_low | FFORMA |
|---|---:|---:|---:|
| ANP | 0,220603 | 0,219567 | 0,216593 |
| ETTH1 | 0,226110 | 0,240080 | 0,300054 |
| ETTH2 | 0,176888 | 0,174797 | 0,203528 |
| ETTM1 | 0,487849 | 0,509871 | 0,439569 |
| ETTM2 | 0,229306 | 0,176754 | **0,165421** |
| NN5 | 0,116666 | **0,116270** | 0,119654 |
| **MÉDIA** | 0,242904 | **0,239556** | 0,240803 |
| **AGREGADO** | 0,189846 | 0,188715 | **0,188354** |

O guard funcionou: o líder do agente tinha 0,249633 e o guard aplicou 0,242904
(+0,006729). As trocas foram boas no ETTM1 (3/3, +0,0638 na média), ETTH1
(3/3, +0,0336), ETTM2 (1/1, +0,0028); neutras no ANP (35 trocas, +0,0002);
uma perda no ETTH2 (-0,0128); nenhuma no NN5.

O problema é o **líder do ETTM2**: 0,229710 contra 0,176754 no v10x. Na série 5
do ETTM2, o agente só tentou `pool1` (mean/median/trimmed/weighted), todas com
teste ~0,63; não tentou `trimmed_mean(pool_full)`, que no v10x dava 0,288. O
agente ancorou no pool estável da Phase 2 (`pool1`) e o guard não tem como
consertar um histórico ruim.

Conclusão: a decisão final funcionou, mas a run completa ficou pior que o v10
por variância/ancoragem do agente no ETTM2. É exatamente o que o `topkguard`
com `--no-seed-attempts` + handles da Phase 2 escondidos + `--direct-preparation`
tenta corrigir.

### 11.12 Run v11_topk_guard (real, low)

O agente **usou** as ferramentas novas em todos os datasets:

| dataset | select_top_k | weights_topk_per_horizon |
|---|---:|---:|
| ANP | 182 | 153 |
| ETTH1 | 8 | 3 |
| ETTH2 | 8 | 6 |
| ETTM1 | 9 | 4 |
| ETTM2 | 8 | 3 |
| NN5 | 78 | 105 |

SMAPE:

| política | ANP | ETTH1 | ETTH2 | ETTM1 | ETTM2 | NN5 | média | agregado |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v10x_low | 0,219567 | 0,240080 | 0,174797 | 0,509871 | 0,176754 | 0,116270 | 0,239556 | 0,188715 |
| v11_seasonal | 0,220603 | 0,226110 | 0,176888 | 0,487849 | 0,229306 | 0,116666 | 0,242904 | 0,189846 |
| v11_topk (sem gap) | 0,222833 | 0,228466 | 0,176736 | 0,455297 | 0,220161 | 0,120273 | 0,237294 | 0,191497 |
| **v11_topk + gap 0,10** | 0,222630 | 0,227298 | 0,176736 | 0,457370 | **0,179115** | 0,120273 | **0,230570** | 0,190506 |
| FFORMA | 0,216593 | 0,300054 | 0,203528 | 0,439569 | 0,165421 | 0,119654 | 0,240803 | 0,188354 |

O agente chegou na receita com `k=3` (o exemplo dizia `k=5`), mas o guard passou
a escolhê-la. No ETTM2 série 5, o agente finalmente achou o bom
`trimmed_mean(pool_full)` (teste 0,288); o guard trocou para um topk
`weighted(pool_full)` com teste 0,575 porque o gap sazonal era só 0,0507. O
`--seasonal-guard-min-gap 0.10` mantém o líder e leva o ETTM2 de 0,220161 para
0,179115.

O topk+gap é o melhor na **média das médias** (0,230570), mas o **agregado**
(0,190506) ainda fica atrás do v10/FFORMA. O agregado é puxado por ANP
(0,222630 vs 0,219567) e NN5 (0,120273 vs 0,116270): o agente usou topk nesses
datasets e o líder piorou; o guard não troca no NN5 (gate inativo) e no ANP fica
quase neutro. O próximo lever é condicionar o exemplo/dica de topk a séries
longas/sazonais (ETT), em vez de mostrá-lo para todos.

O M4, que não entra na média oficial, ficou 0,09248 no topk vs 0,09096 no
v10x_low.

A flag `--seasonal-guard-min-gap` foi implementada; use 0.10 na próxima run.

### 11.13 Com M4 junto

M4 tem 359 séries. Aqui a média é sobre os 7 datasets (peso igual por dataset) e o
agregado é sobre as 680 séries. O `v11_seasonal` não tem M4 (a run foi só nos 6),
então fica fora desta comparação de 7.

SMAPE:

| dataset | v10_low | v11_topk | v11_topk+gap 0.10 | FFORMA | ADE |
|---|---:|---:|---:|---:|---:|
| ANP | 0,219567 | 0,222833 | 0,222630 | 0,216593 | 0,217722 |
| ETTH1 | 0,240080 | 0,228466 | 0,227298 | 0,300054 | 0,395094 |
| ETTH2 | 0,174797 | 0,176736 | 0,176736 | 0,203528 | 0,226148 |
| ETTM1 | 0,509871 | 0,455297 | 0,457370 | 0,439569 | 0,457223 |
| ETTM2 | 0,176754 | 0,220161 | 0,179115 | 0,165421 | 0,187240 |
| NN5 | 0,116270 | 0,120273 | 0,120273 | 0,119654 | 0,117800 |
| M4 | 0,090964 | 0,092478 | 0,092397 | 0,099552 | 0,106894 |
| **MÉDIA (7)** | 0,218329 | 0,216606 | **0,210831** | 0,220624 | 0,244017 |
| **AGREGADO (680)** | **0,137108** | 0,139221 | 0,138710 | 0,141471 | 0,146965 |

POCID (resumo):

| resumo | v10_low | v11_topk | v11_topk+gap | FFORMA | ADE |
|---|---:|---:|---:|---:|---:|
| MÉDIA (7) | **54,230** | 53,906 | 53,973 | 53,370 | 52,281 |
| AGREGADO (680) | 57,179 | 56,276 | 56,419 | **58,211** | 56,368 |

Leitura: o topk+gap continua o melhor na **média dos 7**, mas o v10 ganha no
**agregado das 680 séries**; o M4 do topk (0,0924) é um pouco pior que o do v10
(0,0910), e o M4 tem 359 séries, então pesa muito no agregado. O FFORMA tem o
melhor POCID agregado.

### 11.14 Run v11_topk_guard_margin_low (real, 6 datasets)

Foi a run real com `--direct-preparation --seasonal-guard-min-gap 0.10`. Sem M4.

| dataset | v11_topk_gap | v10x_low | FFORMA |
|---|---:|---:|---:|
| ANP | 0,219267 | 0,219567 | 0,216593 |
| ETTH1 | 0,223642 | 0,240080 | 0,300054 |
| ETTH2 | 0,173056 | 0,174797 | 0,203528 |
| ETTM1 | **0,430257** | 0,509871 | 0,439569 |
| ETTM2 | **0,164238** | 0,176754 | 0,165421 |
| NN5 | 0,119554 | 0,116270 | 0,119654 |
| **MÉDIA** | **0,221669** | 0,239556 | 0,240803 |
| **AGREGADO** | **0,187275** | 0,188715 | 0,188354 |

POCID: média **53,870** e agregado **56,977**; também ganha do v10
(53,526/55,759) e do FFORMA (52,357/56,829) nos dois resumos.

O guard melhorou o líder do agente: 0,236583 -> 0,221669 na média, e
0,189081 -> 0,187275 no agregado. Trocas: ETTM1 3/3 (+0,0821 na média), ETTH1
1/1 (+0,0053), ETTM2 1/1 (+0,0023), ANP 9 trocas (-0,00027), ETTH2/NN5 sem troca.
O agente usou `select_top_k`/`weights_topk_per_horizon` em todos os datasets.

Caveats: é uma run única; parte do ganho é a trajetória do agente (a run topk
anterior, sem gap, tinha 0,237294). O gap 0.10 foi escolhido nos artefatos
anteriores; precisa de uma confirmação com outro seed ou reasoning. M4 não
entrou; para a média de 7, rodar o M4 com esta config. Com o M4 do topk anterior
(0,0924) como referência, a média7 ficaria ~0,2032 e o agregado7 ~0,1372
(v10: 0,2183/0,1371).

---

## 12. Nota de branches

Hoje `v3_proposta_final` e `v11_plus` apontam para o mesmo commit. Antes de
divergir, garantir que a v10 está congelada em `v3_proposta_final` (tag/merge)
e que o desenvolvimento da v11 acontece só em `v11_plus`.
