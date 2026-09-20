# Entendendo o agente — o que acontece depois

Continuação de `ENTENDENDO_A_PROPOSTA.md`. Lá, o texto termina no momento em que
todo o trabalho preparatório está pronto. Aqui começa a parte em que a
inteligência artificial entra.

Mesmo estilo: para quem nunca viu o projeto, sempre com exemplos reais. Todos os
textos e números deste documento foram extraídos de uma execução verdadeira sobre
a série 1 do dataset `ANP_MONTHLY` (venda mensal de combustível).

---

## 1. Onde paramos

Antes do agente falar uma única palavra, o sistema já fez tudo isto:

- carregou a série e as previsões dos 19 modelos;
- descreveu a série em 26 características;
- treinou o meta-modelo com as outras 181 séries;
- descobriu quais estratégias costumam funcionar neste dataset;
- **testou 10 estratégias padrão e já sabe a nota de cada uma.**

Esse último ponto é o mais importante para entender o resto: **o agente não
começa do zero.** Ele começa com 10 respostas prontas na mesa e a tarefa de
tentar achar uma décima primeira que seja melhor.

---

## 2. Quem é o agente, e a regra que o prende

O agente é um modelo de linguagem (o mesmo tipo de tecnologia do ChatGPT), mas
rodando localmente. Nas execuções deste projeto foi usado o `gpt-oss:20b`.

Ele opera num padrão chamado **ReAct** — abreviação de *Reasoning + Acting*
(raciocinar + agir). Em vez de responder de uma vez, ele alterna:

```
Pensa  →  Age  →  Observa o resultado  →  Pensa de novo  →  ...
```

### A regra que muda tudo

**O agente nunca escreve um número.**

Ele não escreve previsões. Não escreve pesos. Não escreve porcentagens. A única
coisa que ele produz é **texto dizendo qual ferramenta usar e com quais
argumentos**. Todas as contas são feitas por código comum, sem IA.

Isso resolve o maior medo de usar IA nesse tipo de tarefa: ele não tem como
inventar um número. Se ele escrever algo que não existe no catálogo, o sistema
devolve um erro e ele tenta de novo — mas nada inventado entra no resultado.

Uma analogia: ele é um **gerente**, não um **calculista**. Ele decide "vamos
testar a mediana dos 5 modelos mais estáveis"; quem calcula a mediana é a
calculadora, não ele.

---

## 3. O que o agente lê — parte 1: as instruções fixas

Toda conversa com um modelo de linguagem tem duas partes: as instruções fixas
(que não mudam) e a mensagem do momento. Vamos ver as duas, de verdade.

As instruções fixas dizem quem ele é, o que pode fazer, e como deve responder.
Reproduzindo os trechos principais do texto real:

```
You are a forecast COMBINATION AGENT.

A pool of already-trained forecasting models produced predictions for one time
series. Your job is to decide HOW TO COMBINE them. You never write forecast
numbers and you never write weights: you call tools, and the tools compute.
```

Depois vem a **lista completa de ferramentas** — 24 no total. Algumas delas:

```
  select_stable(k, metric='rmse')
      - Pool with the k most consistent models across windows.
  prune_redundant(pool='pool_full', corr_threshold=0.95)
      - Drops redundant models, keeping the lowest-error one in each group.
  weights_inverse_error(pool='pool_full', metric='rmse', shrinkage=0.0)
      - w proportional to 1/error, optionally shrunk toward uniform weights.
  weights_pooled_meta_model(pool='pool_full', eta=1.0)
      - w from a gradient-boosted model trained across every OTHER series...
  evaluate_strategy(combine=None, pool=None, weights=None, rationale='')
      - Core loop tool: builds the strategy, backtests it, and ranks the result.
  accept(attempt_id, confidence, justification)
      - stop and take an attempt
```

E o **formato exigido da resposta** — exatamente três linhas:

```
OUTPUT FORMAT - exactly three lines, nothing else:
Thought: <one or two sentences on what you want to learn or test next>
Action: <tool name from the list above>
Action Input: {"arg": value}
```

Por fim, um conjunto de regras práticas. Duas que valem destacar, porque foram
escritas em resposta a comportamentos observados em execuções anteriores:

```
- If a weights_* handle comes back with concentration near 0, the weights
  are effectively UNIFORM, and that strategy gives the same forecasts as the
  plain mean of the same pool. Change the POOL, or the kind of combination,
  rather than the weighting method.

- When you accept, justification must explain the choice in terms of
  OBSERVABLE SERIES CHARACTERISTICS, not just 'it had the lowest error'.
```

A segunda é importante para o projeto: ela obriga o agente a **justificar
causalmente**, não apenas apontar o menor número.

---

## 4. O que o agente lê — parte 2: a mensagem do turno

Esta parte é remontada a cada rodada. Vamos ver a mensagem real da **primeira**
rodada da série 1, bloco por bloco.

### Bloco 1 — quantas rodadas restam

```
ITERATION 1 of 12.
```

### Bloco 2 — a ficha da série

```json
{"n_points":407, "frequency":"monthly", "seasonal_period":12, "horizon":12,
 "n_validation_windows":3, "n_models":19,
 "trend_strength":0.9098, "seasonal_strength":0.412,
 "trend_champion":{"model":"THETA","score":0.997},
 "seasonality_champion":{"model":"ARIMA","score":0.927},
 "stationarity":{"verdict":"non_stationary","reliable":true},
 "outliers_pct":0.0,
 "features":{"acf1":0.9444,"spectral_entropy":0.3697,"coef_variation":0.4507},
 "catch22":"computed"}
```

Traduzindo: série mensal, 407 meses, tendência muito forte (0.91), sazonalidade
moderada (0.41), sem outliers, comportamento suave. O `THETA` é quem melhor
captura a tendência, o `ARIMA` quem melhor captura o padrão sazonal.

Repare em `"catch22":"computed"` — as 22 características técnicas **não** são
mostradas ao agente. Elas são 22 números que um modelo de linguagem não
conseguiria usar para raciocinar; servem apenas ao meta-modelo. O agente só é
informado de que foram calculadas.

### Bloco 3 — o boletim dos 19 modelos

```json
{"error_table":{"top":[
    {"model":"THETA","error":2257.1,"rank":1},
    {"model":"ETS","error":2327.5,"rank":2},
    {"model":"FT_rf","error":3156.6,"rank":3}, ...],
  "relative_spread":1.317},
 "best_model_per_window":[
    {"window":0,"best_model":"ARIMA"},
    {"window":1,"best_model":"THETA"},
    {"window":2,"best_model":"DWT_catboost"}],
 "ranking_stability":{"mean_kendall_tau":0.088,"verdict":"unstable"},
 "error_correlation":{"mean_corr":0.804,"n_groups":9,"n_independent":5,
    "redundant_groups":[
      {"models":["ETS","THETA"],"representative":"ETS"},
      {"models":["rf","catboost","CWT_rf","DWT_rf","DWT_catboost",
                 "FT_catboost","ONLY_CWT_rf","ONLY_DWT_rf"],
       "representative":"rf"}, ...]}}
```

Três informações valiosas aqui, e vale entender cada uma:

**"ranking instável" (`tau=0.088`)** — em cada uma das 3 janelas, um modelo
diferente foi o melhor: ARIMA, depois THETA, depois DWT_catboost. Não há um
campeão consistente. Isso é um alerta: apostar tudo num modelo só seria arriscado.

**"erros correlacionados" (`mean_corr=0.804`)** — muitos modelos erram da mesma
forma. Oito deles (rf, catboost, CWT_rf, DWT_rf, ...) formam um único grupo
redundante. Ter 8 modelos que erram igual não é ter 8 opiniões — é ter uma
opinião repetida 8 vezes.

**"apenas 5 independentes"** — dos 19 modelos, só 5 trazem informação
genuinamente distinta.

### Bloco 4 — a dica do dataset

```json
{"best_on_this_dataset":[
   {"strategy":"weighted w=w1","mean_validation_score":0.6589},
   {"strategy":"dba","mean_validation_score":0.6711},
   {"strategy":"mean","mean_validation_score":0.6711},
   {"strategy":"median","mean_validation_score":0.6742}],
 "worst_on_this_dataset":[
   {"strategy":"mean pool=pool2","mean_validation_score":0.6916},
   {"strategy":"mean pool=pool1","mean_validation_score":0.6974}],
 "how_to_use":"validation-only averages over the OTHER series of this dataset.
   A strategy near the top is a good place to START, not a rule: which strategy
   wins is series-specific... Your own attempt history for THIS series
   outranks it."}
```

É o "prior" explicado no documento anterior. Note o texto do `how_to_use`: ele
diz explicitamente ao agente para tratar isso como **sugestão, não regra**.

### Bloco 5 — uma leitura interpretativa

```json
{"regime":"trend_dominated", "predictability":"high",
 "combination_hint":"robust",
 "risks":["model ranking does not hold across windows",
          "several models carry correlated error",
          "only 3 validation windows: weight estimation is high variance"],
 "narrative":"Trend strength 0.9098 and seasonal strength 0.412 put this series
   in the trend dominated regime, with high predictability. Ranking stability
   across windows is unstable (Kendall tau 0.088), which favours a robust
   combination."}
```

É um resumo em linguagem estruturada do que os números significam. Pode ser
gerado por regras fixas (como neste caso) ou por outro modelo de linguagem.

### Bloco 6 — as 10 estratégias já testadas, ranqueadas

Este é o bloco mais importante:

```
ATTEMPT HISTORY (9), best first - lower score is better:
  {"id":"a5","strategy":"trimmed_mean pool=pool1 trim=0.2","n_models":5,
   "score":0.6131,"rmse":3169.2,"smape":0.2156,"origin":"baseline",
   "rationale":"deterministic baseline: trimmed_mean over the 5 models with
                the most consistent ranking across windows"}
  {"id":"a4","strategy":"mean pool=pool1","n_models":5,"score":0.6187, ...}
  {"id":"a7","strategy":"trimmed_mean pool=pool2 trim=0.2","score":0.6522, ...}
  {"id":"a6","strategy":"mean pool=pool2","score":0.6667, ...}
  {"id":"a1","strategy":"mean","n_models":19,"score":0.6848, ...}
  {"id":"a9","strategy":"trimmed_mean pool=pool3 trim=0.2","score":0.6929, ...}
  {"id":"a8","strategy":"mean pool=pool3","score":0.6986, ...}
  {"id":"a3","strategy":"dba","score":0.7013, ...}
  {"id":"a2","strategy":"median","score":0.7115, ...}
```

Cada linha é uma estratégia com nota já calculada. **Menor é melhor.** A líder
atual é `a5` com 0.6131.

> *Nota técnica: esta extração foi feita sem o meta-modelo anexado, por isso
> aparecem 9 estratégias e não 10. Numa execução completa há uma décima entrada,
> `weighted w=w1`, que é a do meta-modelo.*

### Bloco 7 — os identificadores disponíveis

```json
{"pools":{
  "pool1":{"k":5,"models":["ETS","THETA","FT_catboost","ONLY_CWT_rf","ONLY_DWT_rf"]},
  "pool2":{"k":7,"models":["ETS","THETA","rf","FT_catboost","ONLY_CWT_catboost",...]},
  "pool3":{"k":9,"models":["ETS","THETA","rf","DWT_rf","FT_catboost",...]}}}

  no weight handles exist yet - create one with a weights_* tool before any
  combine='weighted' strategy
```

O sistema trabalha com **apelidos**. `pool1` é um apelido para "aquele conjunto
de 5 modelos". Quando o agente cria um conjunto de pesos, recebe um apelido como
`w1`. Ele nunca vê os números por trás — só o apelido.

### Bloco 8 — o lembrete final

```
12 iterations left. Respond with Thought / Action / Action Input.
```

---

## 5. Como o agente responde

Três linhas. Exemplo real, do primeiro turno da série 1:

```
Thought: ranking instability (tau=0.088) and error correlation indicate
         redundancy; pruning first should reduce overfitting
Action: prune_redundant
Action Input: {"pool": "pool_full", "corr_threshold": 0.95, "metric": "rmse"}
```

- **Thought** — o raciocínio. Não faz nada, mas fica gravado e é auditável.
- **Action** — qual ferramenta.
- **Action Input** — os argumentos, em formato JSON.

---

## 6. O que acontece com a resposta

```
resposta do agente
       ↓
[1] separa as três linhas
       ↓
[2] o nome está no catálogo?      não → devolve erro, ele tenta de novo
       ↓ sim
[3] os argumentos são válidos?    não → devolve erro listando os aceitos
       ↓ sim
[4] EXECUTA o código da ferramenta (sem IA nenhuma)
       ↓
[5] devolve um resumo curto do resultado
```

O passo 2 é a "gaiola": se o agente inventar uma ferramenta chamada
`super_combinador_magico`, o sistema responde:

```json
{"error": "unknown_tool",
 "detail": "'super_combinador_magico' is not in the catalog",
 "available": ["series_profile", "stl_summary", ...]}
```

Ele lê isso no turno seguinte e corrige. Nenhuma invenção chega ao resultado.

---

## 7. O que ele recebe de volta — as observações

Aqui está o que torna o loop informativo. Exemplos reais:

**Depois de podar modelos redundantes:**
```
pool pool4  19->13 models, dropped ['ETS', 'rf', 'CWT_rf', 'catboost']
```
Traduzindo: criei um conjunto novo chamado `pool4`, com 13 dos 19 modelos.
Removi os que eram redundantes.

**Depois de calcular pesos:**
```
handle w2 mode=softmax_neg_error active=13/13 conc=0.003
```
Traduzindo: pesos guardados sob o apelido `w2`; 13 dos 13 modelos recebem algum
peso; **concentração 0.003**.

Essa última é a informação mais útil da observação. **Concentração** vai de 0 a
1: perto de 0 significa "todos os pesos são praticamente iguais". Ou seja, esses
pesos são a média disfarçada — não valem uma rodada de teste.

**Outro cálculo de pesos, mesma série:**
```
handle w3 mode=pooled_meta_model active=13/13 conc=0.285
```
Concentração 0.285 — aqui sim há discriminação real entre os modelos. Este vale
testar.

**Depois de testar uma estratégia:**
```
rank 1/12 score=0.5311 rmse=2920.8249 (best so far)
```
Traduzindo: essa estratégia ficou em 1º lugar entre as 12 já testadas, com nota
0.5311. É a nova líder.

**Quando a estratégia perde:**
```
rank 10/13 score=0.6866 rmse=3634.1185, leader is a12 (weighted pool=pool4 w=w3)
```
Perdeu — 10º de 13. E o sistema lembra quem está liderando.

---

## 8. Uma partida completa, turno a turno

Esta é a trajetória real e completa da série 1. Onze turnos.

**Turno 1 — limpar a bagunça**
```
Thought: ranking instability (tau=0.088) and error correlation indicate
         redundancy; pruning first should reduce overfitting
Action: prune_redundant {"pool":"pool_full","corr_threshold":0.95}
→ pool4  19->13 models, dropped ['ETS','rf','CWT_rf','catboost']
```
Ele leu que os erros são correlacionados (0.804) e começou eliminando repetição.

**Turno 2 — testar robustez**
```
Action: evaluate_strategy {"combine":"median","pool":"pool1",
        "rationale":"ranking instability (tau=0.088) and error correlation
                     indicate redundancy; median on stable pool1 for robustness"}
→ rank 8/11  score=0.6865   (líder continua a10)
```
Testou a mediana. Ficou em 8º de 11 — **perdeu**. Faz parte.

**Turnos 3 e 4 — dois jeitos de ponderar**
```
Action: weights_softmax_neg_error {"pool":"pool4"}
→ handle w2  active=13/13  conc=0.003      ← quase uniforme, não vale testar

Action: weights_pooled_meta_model {"pool":"pool4"}
→ handle w3  active=13/13  conc=0.285      ← tem sinal de verdade
```
Aqui a observação fez seu trabalho: o primeiro método deu concentração 0.003
(inútil), o segundo deu 0.285. Ele testa apenas o segundo.

**Turno 5 — e funciona**
```
Action: evaluate_strategy {"combine":"weighted","pool":"pool4","weights":"w3",
        "rationale":"pruned pool4 (13 models) with pooled_meta_model weights
                     to test if reduced redundancy improves performance"}
→ rank 1/12  score=0.5311  (melhor até agora)
```
De 0.6131 (a melhor pré-testada) para **0.5311**. Salto grande.

**Turnos 6 e 7 — testar uma terceira ideia**
```
Action: weights_error_trend {"pool":"pool4"}
→ handle w4  active=11/13  conc=0.04

Action: evaluate_strategy {"combine":"weighted","pool":"pool4","weights":"w4"}
→ rank 10/13  score=0.6866  (líder continua a12)
```
Tentou ponderar por "para onde o erro está indo". Ficou em 10º. Abandonou a
ideia — não insistiu nela.

**Turnos 8, 9 e 10 — refinar o que funcionou**
```
Action: prune_redundant {"pool":"pool4","corr_threshold":0.95}
→ pool5  13->12 models, dropped ['CWT_catboost']

Action: weights_pooled_meta_model {"pool":"pool5"}
→ handle w5  active=12/12  conc=0.298

Action: evaluate_strategy {"combine":"weighted","pool":"pool5","weights":"w5"}
→ rank 1/14  score=0.4765  (melhor até agora)
```
Voltou ao que deu certo e refinou: podou mais um modelo e reaplicou o
meta-modelo. De 0.5311 para **0.4765**.

**Turno 11 — encerrar**
```
Action: accept {"attempt_id":"a14","confidence":0.9,
  "justification":"Trend-dominated series (trend_strength=0.9098) with unstable
   rankings (mean_kendall_tau=0.088) and high error correlation (mean_corr=0.804)
   require robust combination. Pruned pool5 (12 models) with pooled_meta_model
   weights (w5) achieved lowest score (0.4765) by reducing redundancy while
   leveraging cross-series meta-learning."}
```

Repare que a justificativa cita **três características observáveis** da série,
não apenas "teve o menor erro" — é a regra do prompt sendo cumprida.

### O resumo da partida

```
começou em   0.6131  (melhor pré-testada)
terminou em  0.4765  (encontrada pelo agente)
```

E o percurso não foi linear: dois testes falharam (turnos 2 e 7). O valor está
em **testar hipóteses e descartar as ruins rápido**.

---

## 9. Quando o loop para

Três formas:

| forma | o que significa | frequência real |
|---|---|---|
| **o agente aceita** | usou a ação `accept` | 369 de 679 séries |
| **parada por estagnação** | 4 propostas seguidas sem melhorar | 309 de 679 séries |
| **acabaram as rodadas** | chegou às 12 iterações | **0 de 679 séries** |

O terceiro caso nunca aconteceu. O agente sempre parou sozinho antes do limite.

---

## 10. Quando algo dá errado

Modelos de linguagem locais falham às vezes. O sistema trata cada tipo de falha
de forma diferente:

**O modelo responde vazio** (acontece: ele "pensa" e não escreve nada). O sistema
pergunta de novo, até 2 vezes, **sem gastar uma rodada**. Uma falha de geração
não é uma decisão.

**Erro de comunicação com o servidor.** Pergunta de novo, até 4 vezes, também sem
gastar rodada.

**O modelo escreve algo que não dá para interpretar.** Aí sim gasta a rodada — e
recebe de volta:
```json
{"error": "unparsed_response",
 "reminder": "answer with three lines: Thought:, Action:, Action Input: {json}"}
```
Isso ele consegue corrigir, então vira informação, não falha fatal.

---

## 11. Depois do loop — aplicar na janela cega

Terminado o loop, o sistema pega a estratégia vencedora e a aplica às previsões
da **janela de teste** — aquela que ficou escondida esse tempo todo.

### A garantia central

A estratégia aplicada é **sempre a melhor de todo o histórico**, incluindo as 10
pré-testadas.

Isso significa que, se tudo o que o agente propuser for pior que o que já estava
lá, o sistema aplica a melhor pré-testada e ignora o agente. **O agente só pode
melhorar o resultado, nunca piorar.**

E se ele aceitar uma estratégia que não é a melhor? O sistema registra a escolha
dele, aplica a melhor mesmo assim, e marca essa divergência no relatório. Nada
fica escondido.

---

## 12. A nota final e o relatório

Só agora os valores reais da janela de teste são revelados, e as métricas são
calculadas. Para a série 1:

```
sMAPE = 0.1566    RMSE = 2870.8    POCID = 81.8%
```

Junto com o número, o sistema grava uma **planilha de 58 colunas** e um arquivo
de auditoria completo por série. Três informações se destacam:

**Procedência** — prova que cada número veio de uma ferramenta executada de
verdade, não de texto gerado. Registra quantas ferramentas foram chamadas e se
toda estratégia passou por teste real.

**Confiança da escolha** — a vencedora é estatisticamente distinguível da
segunda colocada? Para a série 1:
```
margem 11.5%   |   Diebold-Mariano p=0.0002   |   veredito: "separated"
```
"Separated" significa: sim, a diferença é real, não sorte. Com apenas 3 janelas,
o veredito honesto costuma ser "indistinguível" — e dizer isso é o objetivo.
Marca as decisões que estão dentro do ruído.

> Por que isso existe: pediu-se ao próprio agente que declarasse sua confiança, e
> ele respondia **0.9 em 59 de 61 casos**. Uma constante não sustenta afirmação
> nenhuma. Foi substituída por um teste estatístico.

**Redutibilidade** — a estratégia vencedora é, na prática, a média simples
disfarçada? O sistema compara e informa. Na série 1, o conjunto tinha 12 modelos,
mas apenas **5 recebiam peso relevante** — a ponderação de fato concentrou.

---

## 13. O ciclo completo, de uma vez

```
┌──────────────────────────────────────────────────────────────────┐
│  JÁ PRONTO ANTES DO AGENTE FALAR                                 │
│    ficha da série · boletim dos 19 modelos · dica do dataset     │
│    10 estratégias testadas e ranqueadas                          │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
        ┌──────────── LOOP, até 12 rodadas ────────────┐
        │                                               │
        │   monta a mensagem do turno                   │
        │            ↓                                  │
        │   agente escreve 3 linhas                     │
        │   (Thought / Action / Action Input)           │
        │            ↓                                  │
        │   sistema valida e EXECUTA (sem IA)           │
        │            ↓                                  │
        │   devolve observação curta                    │
        │   ("conc=0.003", "rank 1/12 score=0.5311")    │
        │            ↓                                  │
        │   aceita? estagnou? senão volta ao topo ──────┘
        │
        ↓
   aplica a MELHOR de todo o histórico na janela cega
        ↓
   calcula a nota + grava auditoria completa
```

---

## 14. Resumo em cinco frases

1. O agente recebe tudo pronto — a ficha da série, o boletim dos modelos, a dica
   do dataset e 10 estratégias já pontuadas — e tenta achar algo melhor.
2. Ele nunca escreve números: só escolhe qual ferramenta chamar, e o código
   calcula.
3. Cada rodada é *penso → ajo → observo*, e a observação é curta e informativa o
   bastante para ele decidir se vale insistir ou mudar de ideia.
4. Ele para quando aceita uma resposta ou quando quatro tentativas seguidas não
   melhoram nada — nunca chegou a esgotar as 12 rodadas.
5. A estratégia aplicada é sempre a melhor de todo o histórico, então o agente só
   pode melhorar o resultado, nunca piorar.

---

## 15. Por que usar um agente, afinal

Uma pergunta justa: se o código já faz todas as contas, por que colocar IA no
meio?

Porque a decisão de **qual conta fazer** não é óbvia e muda de série para série.
Na série 1, a sequência vencedora foi: *podar redundância → aplicar o
meta-modelo → podar mais → aplicar de novo*. Nenhuma regra fixa faria isso — e
qual sequência funciona depende do formato da série, da estabilidade do ranking e
de quanta redundância existe no conjunto.

O agente lê essas três informações e decide o caminho. As garantias (catálogo
fechado, tudo testado antes de valer, resultado nunca pior que o piso) fazem com
que essa liberdade não traga risco junto.
