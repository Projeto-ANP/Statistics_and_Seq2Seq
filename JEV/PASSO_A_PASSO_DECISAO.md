# Como uma semente chega ao agente e como ele decide trocá-la — passo a passo completo

> Walkthrough com dados **reais**: série 5 do ETTM2, run `laya_v2_text`
> (multilingual 8192, menu v2 com evidência por modelo). Cada número abaixo
> saiu do artifact de telemetria
> `orchestrator_laya_laya_v2_text/llm_artifacts/ETTM2/dataset_5.json`.

---

## 0. O contexto em uma frase

14 modelos previram os próximos 24 passos de uma série temporal. O sistema
precisa entregar UMA previsão, combinando esses modelos. Antes de o agente
(LAYA) abrir a boca, o código já preparou 9 **sementes** — combinações
determinísticas prontas — e o agente decide, turno a turno, se testa algo
diferente ou fica com elas.

---

## 1. Antes do agente: as sementes nascem e são pontuadas

**Fase 2** (`pool.run_phase2`) cria 9 estratégias determinísticas e faz o
backtest de cada uma nas **3 janelas de validação** (protocolo aninhado:
cada janela é pontuada com o resto ajustado sem ela). O resultado é o placar
inicial — na série 5 do ETTM2:

| id | origem | estratégia | score (menor = melhor) |
|---|---|---|---|
| a8 | baseline | `mean pool=pool3` (média dos 9 estáveis) | **0.0447** ← líder |
| a2 | baseline | `median` (mediana dos 14) | 0.0566 |
| a6 | baseline | `mean pool=pool2` (média dos 7 estáveis) | 0.0652 |
| a9 | baseline | `trimmed_mean pool=pool3` | 0.0717 |
| a4 | baseline | `mean pool=pool1` (média dos 5 estáveis) | 0.0749 |
| a5/a7/a3/a1 | baseline | trimmed pool1 / trimmed pool2 / dba / mean | 0.0760–0.7246 |

Essas 9 entram no histórico com `origin: baseline` e IDs `a1..a9`. Os pools
que elas criaram (`pool_full`, `pool1`, `pool2`, `pool3`) ficam registrados e
viram as receitas do menu.

---

## 2. Turno 1 — o que o agente recebe

O classificador recebe **um texto** (o estado) + **uma pergunta com opções**.
Nada mais.

### 2.1 O estado (trecho real)

```json
"series": {"n_points": 34816, "frequency": "half_hourly", "horizon": 24,
           "trend_strength": 0.9936, "seasonal_strength": 0.5147, ...,
           "trend_champion": {"model": "DWT_catboost", ...},
           "seasonality_champion": {"model": "ETS", ...}},
"pool":   {"n_models": 14, "n_windows": 3, "tabela de erros, estabilidade, grupos redundantes...},
"regime": {"summary": "trend=0.9936, seasonal=0.5147, model ranking moderate (tau=0.311)",
           "trend_champion": "DWT_catboost", "seasonality_champion": "ETS"},
"history_best_first": [  ← o placar da seção 1, melhor primeiro
  {"id": "a8", "strategy": "mean pool=pool3", "score": 0.0447, "origin": "baseline"},
  {"id": "a2", "strategy": "median", "score": 0.0566, "origin": "baseline"}, ...]
```

Ou seja: o agente **vê** a forma da série (tendência forte, sazonalidade
moderada), quem são os campeões de tendência/sazonalidade, como o pool se
comportou e o placar das sementes.

### 2.2 A pergunta (instruções reais do run)

> "You are searching for the best forecast combination, scored on 3 validation
> windows (lower score is better). Pick ONE strategy to evaluate, or accept.
> Prefer strategies that are NOT already in the history. If the history leader
> looks strong and no untested option is promising, accept."

(No v2b essas instruções ganham as regras de trabalho + exemplo de sequência.)

### 2.3 O menu — 34 opções com evidência por modelo

Cada opção é uma ação **concreta e completa** — não um conceito. Exemplos reais:

```
mean_full            → average of the 14 models [ARIMA(err=1.70 improving);
                       ETS(err=4.64 unstable rank improving seasonality champion); ...]
median_full          → median of the 14 models [...]
trimmed_mean_full    → trimmed mean (drop 20% from each tail) of the 14 models [...]
weighted_inverse_full→ weighted average with weights = 1/validation error [...]
weighted_softmax_full→ weighted average with weights = softmax(-validation error) [...]
mean_stable5         → average of the 5 models [DWT_rf(err=0.97 lowest error degrading);
                       FT_rf(err=1.02 degrading); CWT_catboost(err=1.06 unstable rank
                       improving); DWT_catboost(err=1.08 ...); ...]
... (receitas: full, stable5, stable7, stable9, top5, prune × 5 métodos)
best_ONLY_CWT_rf     → use only model ONLY_CWT_rf (...) as the forecast
... (best_single dos 5 melhores modelos)
accept               → stop now and keep the current best strategy
```

A evidência por modelo (`err`, tendência do erro, estabilidade de ranking,
campeão) é o material de comparação: a opção diz **o quê** e **por que**.

---

## 3. Como a "escolha" acontece (o pensar dele, em 4 passos)

O LAYA **não raciocina em voz alta**. A decisão é uma única passada de rede:

1. **Tokenização**: estado + pergunta + todas as descrições das opções viram
   tokens.
2. **Encoder bidirecional** (mmBERT, 322M): uma passada só, atenção cruzada
   sobre tudo — cada token do estado influencia cada opção. É aqui que
   "trend=0.99, champion=DWT_catboost, placar a8=0.0447" se mistura com a
   descrição de cada opção.
3. **Marcadores de opção**: cada opção tem um token-marcador próprio; o estado
   oculto desse marcador vira o logit da opção (quanto mais o contexto
   "combina" com a descrição, maior o logit).
4. **Softmax**: logits → probabilidades calibradas sobre as 34 opções.

O "pensar" está congelado nos pesos do treino — é reconhecimento de padrão,
não deliberação. Por isso o fine-tune é o que instala o julgamento do nosso
domínio.

---

## 4. Turno 1 — a resposta real e o que acontece depois

**Resposta do classificador** (42 ms de inferência):

```
choice: best_ONLY_CWT_rf   confidence: 0.6648
probabilities: {median_stable7: 0.0127, trimmed_mean_stable7: 0.0158,
                best_ONLY_CWT_rf: 0.665, ...}
```

Ou seja: ele "achou" que apostar no modelo individual `ONLY_CWT_rf` era a
melhor ideia — ignorando (por ora) o placar em que `a8` lidera.

**Execução determinística** (o classificador não calcula nada — o código faz):

```
spec = {"combine": "best_single", "model": "ONLY_CWT_rf"}
backtest aninhado nas 3 janelas de validação → score = 0.0758
```

**Placar atualizado**: a proposta entra como `a10` com `origin: agent`,
**rank 6** — ou seja, ficou ATRÁS de 5 sementes (o líder `a8` tem 0.0447).
A troca não aconteceu: para trocar, a proposta precisa superar 0.0447.

---

## 5. Turnos 2–4 — a repetição (limite do v2a) e o freio

O run `v2_text` é o braço v2a, **anterior** ao scratchpad. Por isso o estado
dos turnos seguintes é quase idêntico (o placar top-5 nem mudou) e o
classificador, determinístico, repete a escolha:

| turno | escolha | conf | resultado |
|---|---|---|---|
| 2 | `best_ONLY_CWT_rf` | 0.9808 | já testado (rank 6) |
| 3 | `best_ONLY_CWT_rf` | 0.9808 | já testado |
| 4 | `best_ONLY_CWT_rf` | 0.9808 | já testado |

A confiança até **subiu** para 0.98 — ele não "aprendeu" com o resultado,
porque o resultado não mudou a entrada dele. O loop então dispara o freio:
`stop = no new information in 3 consecutive turns`.

**O que o v2b muda exatamente aqui** (já implementado, não neste run):
1. **Scratchpad no estado** (`what_you_tried_so_far`: turno, ação, score,
   rank) — a entrada muda a cada turno e ele vê que já tentou;
2. **Força-diversidade**: opção escolhida 2× sai do menu;
3. **Regras + exemplo** nas instruções ("se não virou a melhor, tente um TIPO
   diferente").

---

## 6. O fim do loop e a aplicação ao teste

O resultado final é a **melhor tentativa de todo o histórico** (princípio 5) —
aqui, nada que o agente propôs superou as sementes, então:

```
final = a8 (mean pool=pool3, score 0.0447, origin baseline)
teste: smape = 0.4531   (o mesmo do piso — o agente não mudou nada nesta série)
```

Curiosidade que os runs anteriores revelaram: `a8` era exatamente a semente
que, no teste, ia **mal** (0.4531), enquanto o gpt-oss achou uma média de pool
podado que deu 0.2251 no teste. O classificador v2a não chegou lá (nem tinha
a poda no menu v0; no v2 tem, mas não a escolheu).

---

## 7. Resumo: como ele "pensa em trocar"

1. **O gatilho da troca é único**: uma proposta dele precisa virar **rank 1**
   (score < 0.0447, o melhor do placar). Enquanto isso não acontece, as
   sementes seguem no comando.
2. **As pistas que ele usa para decidir o que testar**: regime da série
   (tendência/sazonalidade), campeões (DWT_catboost/ETS), evidência por modelo
   nas opções (erro, tendência, estabilidade) e o placar.
3. **O que pode fazê-lo trocar de ideia entre turnos**: só a mudança do
   próprio estado — placar atualizado + (v2b) scratchpad + opções removidas.
   Sem isso, determinístico ⇒ repetição.
4. **O limite honesto**: ele não sabe se a troca vai dar certo no teste — só
   na validação. A decisão de trocar é exatamente o que o **fine-tune do gate**
   treina (com os desfechos das séries passadas, LOO), fechando o ciclo.

---

## Glossário mínimo

| termo | significado |
|---|---|
| seed / semente | combinação determinística pré-avaliada (mean/median/dba + pools estáveis) |
| pool | subconjunto de modelos (ex.: pool3 = os 9 com ranking mais estável) |
| score | nota composta (RMSE+sMAPE+MAPE+POCID) nas 3 janelas de validação; menor = melhor |
| backtest aninhado | pontuar a janela w usando o resto ajustado sem w (sem vazamento) |
| placar | histórico ranqueado de tentativas (sementes + propostas) |
| princípio 5 | o resultado final é sempre a melhor tentativa do histórico inteiro |
