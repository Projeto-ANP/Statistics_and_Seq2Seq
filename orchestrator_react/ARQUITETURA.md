# Arquitetura do agente combinador ReAct

Documento de referência da arquitetura: o que cada peça faz, **por que** ela existe,
e qual medição justificou cada decisão. Nada aqui é preferência de estilo — cada
seção marcada com 📊 registra o número que motivou a escolha, medido nas 111 séries
do `NN5_WEEKLY_DATASET`.

Última atualização: 2026-07-28.

---

## 1. Visão geral

Um único agente ReAct opera sobre um catálogo fechado de 24 ferramentas
determinísticas. O agente **nunca escreve números** — nem previsões, nem pesos. Ele
escolhe qual ferramenta chamar; a ferramenta calcula e devolve um *handle*
(`pool1`, `w2`) mais um resumo qualitativo.

```
Fase 0  ingestão          .tsf + CSVs dos modelos individuais -> ReactState
Fase 1  diagnóstico       series_card + pool_card (LLM opcional)
Fase 2  baselines         mean / median / dba semeadas no histórico
Fase 3  loop ReAct        Thought -> Action -> Observation, até 8 iterações
Fase 4  aplicação         a melhor tentativa é aplicada à janela de teste cega
Fase 5  relato            justificativa final + CSV de 58 colunas
```

O contrato central: **toda estratégia proposta passa por backtest nas janelas de
validação antes de ser aceita.** `evaluate_strategy` é o único caminho para entrar
no histórico.

---

## 2. Protocolo de avaliação (o núcleo científico)

### 2.1 Seleção aninhada de pool — `nested_selection` 📊

**O problema encontrado.** Até esta versão, um pool era escolhido **uma vez**,
olhando as 3 janelas de validação, e depois pontuado **nessas mesmas 3 janelas**. O
número que ranqueava uma estratégia já tinha visto a seleção que estava ranqueando.

**A medição.** Ranqueando 16 regras fixas pelo score de validação e pelo sMAPE de
teste, nas 111 séries:

| protocolo | Spearman(validação, teste) |
|---|---|
| in-sample (antigo) | **-0.468** |
| aninhado (atual) | **+0.547** |

Correlação **negativa**: parecer melhor na validação predizia parecer *pior* no
teste. A inversão era sistemática, não ruído:

| regra | posição na validação | posição no teste |
|---|---|---|
| `top3_mean` | 1º | 17º |
| `stable5_mean` | 10º | 1º |

O mecanismo é direto: `select_top_k` seleciona **pelo erro de validação**, então
domina a validação por construção — está ajustando ao ruído dela. `select_stable`
seleciona por consistência de ranking, ajusta pior à validação e generaliza melhor.

**A correção.** `selection.py` introduz `PoolRecipe`, espelhando o `WeightsRecipe`
que já existia para pesos: guarda-se a **receita**, não os índices. Dentro do
backtest, a janela `i` é pontuada por um pool re-escolhido sem a janela `i`.

- `orchestrator_react/selection.py` — seletores puros + `PoolRecipe`
- `ReactState.pool_for_window(handle, exclude_window)` — membership por fold
- `ReactConfig.nested_selection: bool = True`

**Ganho medido:** escolha por série 0.120430 → 0.119641 sMAPE (p=0.34, não
significativo). O ganho de acurácia é pequeno; o ganho de **correção** é o ponto —
o score que o agente otimiza deixou de ser anticorrelacionado com o objetivo.

### 2.2 Seleção usa LOO, pesos seguem `backtest_mode` 📊

São dois problemas diferentes e recebem protocolos diferentes:

| passo | protocolo | por quê |
|---|---|---|
| ajustar **pesos** | `backtest_mode` (`expanding` por padrão) | é estimativa prospectiva; só o passado pode informar o número aplicado ao futuro |
| escolher **quais modelos comparar** | sempre leave-one-out | é seleção de modelo, onde LOO é o padrão e o uso eficiente de 3 janelas |

Seguir `expanding` na seleção deixaria um **buraco**, não fecharia um: a janela 0
não tem janela anterior, então o fold cairia de volta na membership de todas as
janelas — exatamente o vazamento que o aninhamento existe para remover — em 1/3 dos
folds.

📊 Medido: `expanding` na seleção dá Spearman **+0.047**; LOO dá **+0.547**.

Implementado em `ReactState._selection_windows()`. Nada aqui lê a janela de teste;
todo fold permanece dentro do bloco de validação.

### 2.3 Anti-vazamento da janela de teste

Três garantias, cada uma com teste dedicado:

1. **Estrutural** — `ReactState` nunca guarda os valores reais do teste.
2. **Comportamental** — envenenar a janela cega não altera nada antes da Fase 4
   (`test_selection_never_reads_the_test_window`).
3. **Prompt** — nenhum valor de teste nem métrica de baseline externa aparece em
   qualquer prompt; as externas são lidas só **depois** da última chamada ao LLM.
### 2.4 Sementes de estabilidade — `seed_stable_pools` 📊

**O maior lever medido de todos.** O piso de toda a arquitetura é a melhor baseline
semeada: quando nada que o agente propõe bate esse piso, é ele que é aplicado. Na
rodada v2, isso aconteceu em **68 de 111 séries** — ou seja, para a maioria do
dataset o resultado reportado *era* o conjunto de sementes. E o conjunto era três
combinações de pool completo, que não são boas.

Semear também combinações selecionadas por estabilidade (`SEED_STABLE_POOLS`):

| configuração | sMAPE médio (braço determinístico, sem LLM) |
|---|---|
| v2: só pool completo | 0.120362 |
| **+ sementes de estabilidade** | **0.115361** |

Passa o ADE (0.11780), a média (0.11994), a mediana (0.12013) e o DBA (0.12256).

> ⚠️ **Ressalva medida depois (ver `RELATORIO_TECNICO_COMBINACAO.md` §2):** este
> ganho é **específico do NN5**. Comparando 16 estratégias de seleção nos dois
> datasets, o ranking praticamente não transfere (Spearman +0.121, p=0.66):
> `stable5_mean` é **1º de 16 no NN5 e 11º de 16 no ANP_MONTHLY**. As sementes de
> estabilidade continuam sendo um default defensável (nunca são catastróficas em
> nenhum dos dois), mas o número 0.12036→0.11536 não deve ser reportado como
> ganho geral do método.

**Por que `select_stable` e não `select_top_k`.** `select_top_k` ranqueia modelos
pelo mesmo erro que a estratégia depois é pontuada — ajusta o ruído da validação
duas vezes. `select_stable` ranqueia por consistência entre janelas, uma estatística
diferente. É o mesmo argumento de duplo-mergulho que motivou o `nested_selection`.
Trocar `stable` por `top_k` devolve quase todo o ganho (0.11738).

**Por que três valores de k.** Não há como saber o subconjunto certo com 3 janelas,
então a escolha deliberadamente não é feita: todo k em {3,5,7,9,11} cai dentro de
0.0009 dos outros. É uma varredura de escala, não uma constante calibrada.

### 2.5 Ensemble de estratégias — implementado, medido, **não** é o default 📊

A ideia: em vez de aplicar a estratégia de menor score, fazer média ponderada
`softmax(-eta*score)` das top-M. O raciocínio é sólido — o score de 3 janelas ordena
estratégias contra a janela cega a apenas Spearman +0.33, e 98 de 111 séries não
separam o 1º do 2º.

E funciona, **contra o conjunto de sementes antigo**: 0.12036 → 0.11948.

Mas esse é o *mesmo* ganho que `seed_stable_pools` entrega, e não sobrevive ao lado
dele:

| | argmin | ensemble | p |
|---|---|---|---|
| braço determinístico + sementes novas | 0.115361 | 0.115954 | 0.62 |
| trajetórias reais do agente + sementes novas | 0.116012 | 0.116445 | 0.58 |

Ambas as direções estão dentro do ruído. **O contrato mais simples fica como
default** (`final_strategy="argmin"`) e o ensemble permanece como braço de ablação
implementado e testado (`ReactState.apply_ensemble`).

Isto é um resultado, não uma sobra: mostra que o ganho vem de *ampliar o espaço de
candidatos*, não de mudar a regra de decisão sobre ele.

### 2.6 Convergência do agente com as sementes

Com sementes mais ricas, o agente frequentemente propõe uma estratégia que a Fase 2
já havia semeado — **42 séries** na reprodução das trajetórias v2. Antes isso era
descartado em silêncio: a tentativa já existia, então a justificativa do agente era
perdida e o `origin` continuava `baseline`.

`Attempt.agent_converged` e `Attempt.agent_rationale` registram esses casos. A
distinção importa para o paper: "o agente não acrescentou nada" e "o agente chegou
independentemente à mesma conclusão" são afirmações diferentes.


---

## 3. O catálogo de ferramentas (24) — o que cada uma faz, com exemplo

Convenção para todo exemplo abaixo: `Action Input` é exatamente o que o agente
escreve; `observação` é exatamente o que ele recebe de volta (resumido — os dicts
reais têm mais campos). Nenhum tool devolve um array bruto; sempre um resumo.

### 3.1 Diagnóstico — leitura, nunca decide nada sozinho

**`series_profile()`** — a ficha da série: tendência, sazonalidade, estacionariedade,
outliers, catch22, e os campeões de componente (§ features.py). É a única chamada
sempre feita automaticamente antes do loop abrir (Fase 1a) e reinjetada em todo
turno; o agente pode rechamá-la, mas o conteúdo não muda.
```
Action Input: {}
observação: {"trend_strength": 1.0, "seasonal_strength": 0.9999, "seasonal_period": 52,
             "features": {"spectral_entropy": 0.866, "acf1": 0.015, ...},
             "stationarity": {...}, "outliers": {...}}
```

**`stl_summary()`** — quanto de variância cada componente da STL explica.
```
Action Input: {}
observação: {"trend_pct": 62.3, "seasonal_pct": 31.1, "residual_pct": 6.6,
             "dominant_component": "trend"}
```

**`error_summary(window=None, top_n=8, metric="rmse")`** — tabela de erro por
modelo, ranqueada; `window` restringe a uma janela específica em vez de todas.
```
Action Input: {"top_n": 5}
observação: {"top": [{"model": "ETS", "error": 16.57, "rank": 1}, ...],
             "rest": {"n_models": 14, "median_error": 22.1}, "relative_spread": 0.42}
```

**`ranking_stability(metric="rmse")`** — o ranking de modelos se mantém entre
janelas? Kendall tau médio entre todos os pares de janelas.
```
Action Input: {}
observação: {"mean_kendall_tau": 0.228, "verdict": "unstable"}
```

**`error_correlation(model_ids=None, threshold=0.9)`** — quais modelos erram de
forma parecida (candidatos a redundância, usado por `prune_redundant`).
```
Action Input: {"threshold": 0.95}
observação: {"redundant_groups": [{"models": ["DWT_rf", "DWT_catboost"], "mean_corr": 0.97}]}
```

**`dm_test(model_a, model_b, loss="squared")`** — Diebold-Mariano entre dois
modelos específicos, com a correção Harvey-Leybourne-Newbold de amostra pequena.
```
Action Input: {"model_a": "ETS", "model_b": "ARIMA"}
observação: {"dm_stat": -1.42, "p_value": 0.29, "verdict": "indistinguishable"}
```

### 3.2 Seleção de pool — todas registram uma `PoolRecipe` re-ajustável (§2.1)

**`select_top_k(k, metric="rmse", windows=None)`** — os k modelos de menor erro.
Sob `nested_selection`, a janela sendo pontuada nunca vota na própria seleção.
```
Action Input: {"k": 5}
observação: {"pool": "pool2", "models": ["FT_catboost", "ONLY_FT_rf", "CWT_rf", ...]}
```

**`select_stable(k, metric="rmse")`** — os k modelos mais consistentes entre
janelas (`mean rank + std rank`, não erro puro). É a seleção que sustenta o piso
determinístico (§2.4) — critério diferente de `select_top_k`, de propósito.
```
Action Input: {"k": 5}
observação: {"pool": "pool1", "models": [{"model": "ETS", "mean_rank": 2.3, "rank_std": 0.5}, ...]}
```

**`prune_redundant(pool=FULL_POOL, corr_threshold=0.95, metric="rmse")`** — remove
modelos redundantes, mantendo o de menor erro em cada grupo correlacionado.
```
Action Input: {"pool": "pool_full", "corr_threshold": 0.95}
observação: {"pool": "pool4", "n_before": 19, "n_after": 11, "removed": ["ARIMA", "CWT_rf", ...]}
```

### 3.3 Pesos — sempre devolvem um *handle* (`w1`, `w2`...), nunca números soltos

**`weights_inverse_error(pool, metric="rmse", shrinkage=0.0)`** — `w ∝ 1/erro`.
Com poucos modelos de erro parecido, degenera pra quase-uniforme (§7) —
`conc≈0` é justamente esse sinal.
```
Action Input: {"pool": "pool1", "shrinkage": 0.1}
observação: {"weights": "w1", "summary": {"n_active": 5, "concentration": 0.002}}
```

**`weights_softmax_neg_error(pool, metric="rmse", eta=1.0)`** — `w ∝ softmax(-η·erro)`,
a mesma forma final que o ADE usa. `eta` maior concentra mais peso no melhor modelo.
```
Action Input: {"pool": "pool_full", "eta": 2.0}
observação: {"weights": "w2", "summary": {"n_active": 19, "concentration": 0.031}}
```

**`weights_error_trend(pool, metric="mae", eta=1.0, damping=None)`** — pesa pelo
erro *extrapolado*, não pela média — lê a grade ponto a ponto (janelas×horizonte),
não 3 números agregados. Medido: empata com as outras receitas de peso (§4.2),
mantida como braço de ablação.
```
Action Input: {"pool": "pool_full"}
observação: {"weights": "w3", "effective_mode": "error_trend",
             "n_worsening": 4, "n_improving": 6}
```

**`weights_ols(pool, l2=0.0, nonneg=True)`** — mínimos quadrados projetados no
simplex. **Retirada do catálogo com menos de `min_windows_for_ols` janelas** (§4.1)
— com o protocolo padrão de 3 janelas, essa tool não aparece nem na lista de ações
disponíveis.

**`weights_feature_based(pool, metric="smape", eta=1.0)`** — meta-modelo XGBoost
**por série**, no espírito do FFORMA. Nunca teve amostra suficiente pra treinar de
verdade com 3 janelas — sempre cai no fallback `softmax(-erro)` (§13.1). É o
motivo direto de existir a próxima.

**`weights_pooled_meta_model(pool, eta=1.0)`** — ver §3.6 abaixo, exemplo completo.

### 3.4 Combinação — monta o objeto de estratégia (não pontua nada sozinha)

**`combine_mean()` / `combine_median()`** — sem parâmetros; montam a estratégia.
```
Action Input: {"pool": "pool_full"}
observação: {"strategy": {"combine": "mean", "pool": "pool_full"}, "n_models": 19,
             "next_step": "call evaluate_strategy with exactly this Action Input"}
```

**`combine_trimmed_mean(pool, trim_pct=0.2)`** — média cortando as `trim_pct`
frações mais extremas de cada lado.

**`combine_weighted(pool, weights)`** — exige um handle de peso já calculado por
uma das tools da seção 3.3.
```
Action Input: {"pool": "pool2", "weights": "w1"}
```

**`combine_dba(pool, max_iter=30)`** — DTW Barycenter Averaging. `random_state=7`
fixo internamente (§ correção do bug de determinismo — antes duas séries
idênticas podiam dar previsões diferentes por causa do estado global do numpy).

**`combine_best_single(model)`** — não combina nada; aposta num único modelo.

### 3.5 Validação — o único caminho pra entrar no histórico

**`evaluate_strategy(...)`** — roda o backtest anti-vazamento e ranqueia contra
todo o histórico. Aceita a estratégia em várias formas (flat, aninhada, JSON em
string) porque o modelo varia como escreve isso, e rejeitar por formato custa uma
iteração à toa.
```
Action Input: {"combine": "weighted", "pool": "pool2", "weights": "w1",
               "rationale": "pesos por erro inverso no subconjunto estável"}
observação: {"rank": "2/6", "score": 0.6229, "rmse": 19.40, "leader": "a5 (best_single ETS)"}
```

**`sanity_check(reference)`** — compara a previsão final contra a faixa histórica
da série. Só avisa, nunca bloqueia. `reference` aceita um id de tentativa (`"a3"`)
ou uma estratégia inteira.
```
Action Input: {"reference": "a1"}
observação: {"ok": true, "warnings": [], "n_points": 8}
```

**`list_attempts(top_n=10)`** — histórico ranqueado do melhor pro pior. É o que
prova que `evaluate_strategy` foi mesmo chamada — texto forjado de Observation não
tem como aparecer aqui, porque isso lê `state.attempts` de verdade, não o que o
modelo escreveu antes.
```
Action Input: {}
observação: {"total": 8, "best": "a6", "ranking": [{"id": "a6", "strategy": "median pool=pool1", "score": 0.6466}, ...]}
```

### 3.6 `weights_pooled_meta_model` — exemplo completo, do treino à decisão

Esta é a tool mais recente (§13), e a única cujo cálculo não vem só desta série.

**Passo 0 (antes do loop abrir, uma vez por dataset):** o pré-passo em
`pipeline.run_dataset` roda `series_profile` + erro de validação em toda série do
dataset, treina um XGBoost por modelo pra cada série (excluindo a própria série,
LOSO) e anexa o resultado da série *i* a `state.pooled_meta_model` antes da Fase 3
de *i* abrir. O agente não vê nada disso — só o resultado quando chama a tool.

**A tool em si:**
```
Action: weights_pooled_meta_model
Action Input: {"pool": "pool_full", "eta": 1.0}
```
Por baixo: pega as 4 features de `series_profile()` desta série (tendência,
sazonalidade, entropia, autocorrelação), consulta o modelo já treinado — que
**nunca viu a linha desta série** — e converte erro previsto em peso via
`softmax(-eta·erro)`.

Observação devolvida:
```json
{"weights": "w4", "method": "pooled_meta_model", "n_train_series": 110,
 "n_models_with_a_fit": 19,
 "summary": {"n_active": 19, "concentration": 0.018}}
```
`n_train_series` é o tamanho real da amostra que treinou o modelo desta série —
está aí de propósito, pra distinguir "o número veio de 110 séries" de "o número
veio de 3 janelas desta".

**Exemplo real, duas séries do NN5 com perfis opostos** (mesmo pool completo, mesmo
modelo pooled, cada uma consultando a sua própria versão leave-one-out):

| | série 64 — ruidosa (acf1=0.015, entropia=0.87) | série 25 — persistente (acf1=0.91, entropia=0.33) |
|---|---|---|
| top 3 previsto | ARIMA, THETA, ETS (~18.9 cada) | NaiveMovingAverage, ARIMA (~22.1), ETS (25.6) |
| pior previsto | NaiveSeasonal, ONLY_FT_catboost | DWT_catboost (49.9), DWT_rf (54.2) |

Numa série persistente o modelo pooled penaliza pesado os modelos baseados em
decomposição wavelet (DWT); numa série ruidosa eles nem entram entre os piores.
Isso vem de padrão aprendido nas **outras** séries do dataset, não das 3 janelas
desta.

**Restrição que a tool aplica sozinha:** recusa um `pool` cuja composição muda por
fold sob `nested_selection` (ex. o resultado de `select_top_k`), porque o vetor de
pesos é calculado uma vez só e reaplicado igual em todo fold — se o pool mudasse
de tamanho por fold, o vetor ficaria com tamanho errado. `pool_full` ou um pool
registrado manualmente sempre funcionam.

**Limite honesto (§13.4):** no NN5, 2 das 4 features (`trend_strength`,
`seasonal_strength`) são quase constantes entre as 111 séries — o mesmo problema
de saturação do STL encontrado no diagnosticador determinístico. O modelo pooled
está funcionando com 2 features úteis, não 4, o que ajuda a explicar por que ele
empatou com `softmax_neg_error` em vez de vencer. Ainda não testado no ANP, onde
as séries são mais heterogêneas entre si e essas duas features tendem a variar de
verdade.

---

## 4. Ferramentas com histórico de decisão

### 4.1 `weights_ols` — liberado por número de janelas 📊

Mínimos quadrados de Granger-Ramanathan precisa de mais equações independentes do
que 3 janelas fornecem. Abaixo do limite, a projeção no simplex cai num **vértice**:
a ferramenta vira um *seletor de modelo* disfarçado de ponderador.

📊 Observado na rodada real: chamada 8 vezes em 111 séries, colapsou 1 vez (série
39: peso 1.0 em `FT_catboost`, 1 de 11 ativos).

Pior: o modelo que o OLS escolhe **não é** o de menor erro. Em simulação com 9
modelos em 3 famílias correlacionadas, o OLS deu 83% do peso ao 7º melhor de 9,
enquanto o melhor modelo recebeu peso zero. Ele não pergunta "quem erra menos", e
sim "que combinação cancela o resíduo destas observações" — com poucos pontos, um
cancelamento acidental vence a precisão real.

**Decisão:** `ReactConfig.min_windows_for_ols = 5`. Abaixo disso a ferramenta é
**retirada do catálogo antes do prompt** (`registry.withheld_tools`), não oferecida
e depois recusada — uma ferramenta oferecida e negada custa uma iteração e não
ensina nada. Defesa em profundidade: se o modelo inventar o nome, `call_tool`
recusa e devolve a lista real. O que foi retirado é gravado no artifact da série.

### 4.2 `weights_error_trend` — resultado negativo, mantido como ablação 📊

**Motivação.** O ADE vence usando as **mesmas** 3 janelas, mas em granularidade de
ponto: achata as janelas numa série de erros (24 pontos por modelo) e treina um
meta-regressor que prevê o erro futuro. As demais receitas colapsam cada janela num
escalar — 3 números por modelo, 8× menos informação.

**Implementação.** Lê a grade de erro pontual `(n_fit, n_models, horizon)` e pesa
por erro **extrapolado**. Dois confundidores separados:

- **Rampa de horizonte.** O passo 8 é mais difícil que o passo 1 para todos.
  Concatenar as janelas leria a rampa como degradação. Por isso o slope é ajustado
  **por passo de horizonte**, ao longo das janelas, e o slope do modelo é a mediana
  desses — o que também transforma 8 ajustes ruidosos de 3 pontos em uma estimativa
  utilizável.
- **Ruído do slope.** `damping=None` (padrão) deriva o amortecimento da concordância
  entre os slopes por passo: todos concordam → 1.0, cara-ou-coroa → 0.0.

📊 **Não melhora.** Nas 111 séries, pool completo: `softmax` 0.118576, `trend`
0.119016 (p=0.36). Em `top5` vira o melhor (0.118024) mas p=0.72. Nada
significativo. O sinal de trend é ruído demais com 3 janelas.

**Mantida no catálogo** como braço de ablação — o resultado negativo é reportável e
o mecanismo está correto e testado.

### 4.3 `per_horizon` — usado e sempre perdedor 📊

Observado: o agente chamou com `per_horizon=True` **77 vezes**; **0** estratégias
vencedoras usaram. Esperado — estima `horizonte × modelos` parâmetros das mesmas 3
janelas. A direção correta não é mais parâmetros, é melhor sinal.

---

## 5. Confiança e reprodutibilidade

### 5.1 `accept_confidence` é inútil, e foi substituído 📊

A confiança auto-reportada pelo agente foi **0.9 em 59 de 61 aceites**. Constante
não sustenta afirmação nenhuma.

**Substituição determinística** (`ReactState.selection_confidence()`): margem,
p-valor de bootstrap pareado, p-valor de Diebold-Mariano com correção
Harvey-Leybourne-Newbold, e um veredito. Com menos de 5 janelas o bootstrap é
degenerado (`bootstrap_reliable=False`) e o veredito segue o DM. Gêmeos numéricos
do vencedor são pulados — comparar o vencedor com uma cópia de si mesmo não diz
nada.

📊 **E o veredito é calibrado**, que é o resultado interessante:

| veredito | n | agente | ADE |
|---|---|---|---|
| `separated` | 19 | 0.12359 | 0.12577 → **agente ganha** |
| `indistinguishable` | 92 | 0.11796 | 0.11615 → agente perde |

O agente sabe quando está certo.

### 5.2 Semente de amostragem 📊

O NN5 contém **3 pares de séries idênticas** na fonte (T1≡T47, T11≡T50, T79≡T111).
O agente rodou cada uma independentemente e **escolheu estratégia diferente nas
três** — dispersão de sMAPE entre 3% e 11%. É uma medida gratuita e rigorosa da
variância run-to-run.

**Segunda fonte, encontrada na análise do v2 e corrigida:** `combine_dba` não
passava `random_state` para `tslearn.dtw_barycenter_averaging`, que então sorteia o
centroide inicial do **estado global do numpy**. Duas séries idênticas que ambas
escolheram `dba` no mesmo pool produziram previsões diferentes (diferença máxima
0.79, sMAPE 0.1199 vs 0.1217) — não porque a entrada mudou, mas porque um número
diferente de chamadas aleatórias não relacionadas havia ocorrido no processo até
cada uma chegar naquela linha. Agora `combine_dba(..., random_state=7)`.

`LLMRole.seed = 7` é passado ao Ollama. Entrada igual passa a dar saída igual, sem
tirar liberdade nenhuma do agente. O seed entra no `fingerprint()`.

### 5.3 Não há memória entre séries

Cada série constrói um `ReactState` novo. A série 1 não sabe nada da série 0. Isso é
deliberado: memória entre séries transformaria a avaliação por série em algo
sequencialmente dependente, impossível de comparar com as baselines.

---

## 6. Procedência: o agente chamou mesmo as ferramentas?

`ReactState.verify_provenance()` devolve `agent_called_tools`, `evaluated_via_tool`,
`all_backtested`, `provenance_ok`.

📊 Na rodada real: **`provenance_ok` True em 111/111**. `tool_missing` True em 4
séries (argumentos fora do contrato), sem prejuízo — 3 delas ficaram em 1º lugar.

A coluna `tools_called` guarda a sequência completa com argumentos.

---

## 7. Reducibilidade: a ponderação está fazendo algo? 📊

`SeriesOutcome.reducibility()` compara a previsão vencedora com a **média simples do
mesmo pool**.

📊 `equivalent_to_pool_mean` foi **True em 62/111 séries (56%)**. Em mais da metade,
a estratégia "ponderada" é aritmeticamente a média do próprio pool.

Exemplo real (série 52, `inverse_error`, 5 modelos):

```
pesos    0.2056  0.2080  0.1964  0.1945  0.1955
uniforme 0.2000                              -> desvio máximo 4%
```

A causa: `w ∝ 1/e`, e os 5 modelos erram quase igual (7.6 a 8.2) — porque o agente
já selecionou os melhores, restando um grupo homogêneo. Dividir 1 por números quase
iguais dá números quase iguais.

Isso é o *forecast combination puzzle* (Claeskens et al., 2016) com evidência
numérica direta. **A contribuição do agente é a escolha do subconjunto, não a
ponderação** — e os dados sustentam isso:

| pool efetivo | n | rank médio | % que bate a média |
|---|---|---|---|
| 1 | 19 | 2.82 | 57.9% |
| 2-3 | 7 | 2.86 | 71.4% |
| 4-6 | 33 | 3.18 | 51.5% |
| 7-10 | 23 | 3.74 | 39.1% |
| 11-19 | 29 | 3.91 | 27.6% |

Monotônico. Corrobora o Self-MoA (Li et al., 2025) de forma independente.

---

## 8. Resultado atual e limitações

📊 sMAPE médio, 111 séries NN5:

| | sMAPE |
|---|---|
| ADE | 0.11780 |
| **agente** | **0.11892** |
| FFORMA | 0.11965 |
| mean | 0.11994 |
| median | 0.12013 |
| dba | 0.12256 |

Só vence o DBA com significância (Wilcoxon p=0.018). É **o melhor dos seis em
36/111 séries** (mais que qualquer baseline isolada), mas a distribuição de rank é
bimodal — 32 primeiros lugares e 23 últimos.

### Limitações a declarar no paper

1. **Regra fixa competitiva.** `select_stable(k=5)` + média simples dá **0.114858**,
   melhor que o agente (p=0.0042) e que o ADE. Ressalva: essa regra foi escolhida
   olhando o teste, entre 18 candidatas. O que vale é o achado de **família** — as
   três variantes `stable_k_mean` (k=5,7,9) ocupam os 3 primeiros lugares no teste e
   as posições 10–13 na validação.
2. **Sinal de validação fraco.** Kendall tau validação→teste = **+0.159**; o
   vencedor da validação é o vencedor do teste em 45% das séries (acaso = 33%).
   Iterar mais não ajuda: o agente já acha o ótimo da validação.
3. **83% `indistinguishable`.** Na maioria das séries o vencedor não se separa
   estatisticamente do 2º.
4. **Séries duplicadas.** NN5 tem 3 pares idênticos (§5.2). Afeta todos os métodos
   igualmente, mas reduz o n efetivo.

---

## 9. Ablações disponíveis

Toda `ReactConfig` é serializada em `ablation_config`, então qualquer linha do CSV
diz sob qual configuração foi produzida.

| # | campo | estado |
|---|---|---|
| 1 | `pool_mode`, `pool_k` | disponível |
| 2 | `diagnostic_llm` | **nunca exercitada** — `False` em todas as rodadas |
| 3 | `max_iterations`, `early_stop_patience` | disponível |
| 4 | `show_attempt_history`, `show_attempt_rationales` | disponível |
| 5-6 | modelo por papel | disponível |
| — | `nested_selection` | **novo** (§2.1) |
| — | `min_windows_for_ols` | **novo** (§4.1) |
| — | `backtest_mode` | `expanding` \| `loo` |
| — | `calibration_gate` | **nunca exercitada** |

---

## 10. Convergência com a literatura

Referência cruzada com `insights_trabalhos.md`:

| trabalho | relação com os dados desta arquitetura |
|---|---|
| **Self-MoA** (item 10) | **Confirmado.** Pool menor generaliza melhor, monotonicamente (§7) |
| **DCATS** (item 9) | Já implementado sem intenção: `show_attempt_history` + `show_attempt_rationales` são o formato "tentativas ranqueadas com justificativa" |
| **TimeSeriesScientist** (item 4) | Não testado — `diagnostic_llm` nunca rodou |
| **LLM-Blender** (item 7) | Comparação pareada não resolveria: o problema é a validação não predizer o teste (§8.2), e comparação pareada sobre sinal ruidoso continua ruidosa |
| **Krause et al.** (item 11) | Seguido: RMSE e POCID reportados lado a lado |
| **Claeskens et al. 2016** | Confirmado empiricamente (§7) |

---

## 11. Mapa de arquivos

| arquivo | responsabilidade |
|---|---|
| `config.py` | `ReactConfig`, `LLMRole`, presets de score, fingerprint |
| `data_source.py` | parsing `.tsf`, mapeamento posicional validado |
| `ingest.py` | Fase 0 — monta o `ReactState` de uma série |
| `state.py` | dados, handles, protocolo de backtest, confiança, procedência |
| `selection.py` | **novo** — seletores puros e `PoolRecipe` (§2.1) |
| `weighting.py` | receitas de peso e `WeightsRecipe` |
| `meta_model.py` | **novo** — meta-modelo pooled entre séries, LOSO (§13) |
| `combiners.py` | `apply_combination` — fonte única, backtest e aplicação final |
| `features.py` | STL, sazonalidade declarada, campeões de componente |
| `tools.py` | o catálogo de 24 ferramentas |
| `registry.py` | espaço de ação fechado, despacho, `withheld_tools` |
| `prompts.py` | prompt de sistema e de turno |
| `react_loop.py` | Fase 3 — laço Thought/Action/Observation |
| `pipeline.py` | orquestra as fases, `SeriesOutcome`, `reducibility()` |
| `meta_model.py` | meta-modelo cross-series, LOSO, 2 objetivos (§13-14) |
| `csv_writer.py` | contrato de 58 colunas |
| `llm.py` | cliente Ollama, parser de passo, `ScriptedLLM` |

**471 testes**, todos em CPU, sem servidor.

---

## 12. ANP_MONTHLY — segundo dataset: o que se confirma, o que não, e por quê

Primeira rodada num dataset diferente de NN5 (mensal, escala de volume de
combustível, 182 séries pós-filtro-de-zeros). Serve de teste de generalização para
tudo que foi medido só no NN5 até aqui.

### 12.1 Confirma — mais forte ainda

`weights_concentration` médio: **0.0017** (era 0.0213 no NN5 — ainda mais uniforme).
Por método:

```
mean          100.0% equivalente à média
weighted       88.5% equivalente à média     ← quase sempre
trimmed_mean   45.9%
median          6.2%
best_single     0.0%
```

Exemplo real, série 0, `weighted` sobre os 19 modelos (uniforme seria 1/19 = 0.0526):

```
ARIMA 0.0554  ETS 0.0545  THETA 0.0536  rf 0.0559  catboost 0.0437 ...
DWT_rf 0.0546  FT_rf 0.0547  NaiveSeasonal 0.0465  NaiveMovingAverage 0.0548
```

Toda a faixa cabe entre 0.0437 e 0.0568 — 0.013 de amplitude ao redor de 0.0526.
**Isso deixa de ser peculiaridade do NN5 e vira achado replicado entre dois
datasets de natureza totalmente diferente** (semanal/retail vs mensal/combustível).

### 12.2 Não confirma — e a causa raiz foi isolada

No NN5, dois padrões pareciam sólidos:
- veredito `separated` → agente empata/bate o ADE; `indistinguishable` → perde
- pool efetivo menor → rank médio melhor, monotonicamente

Nenhum dos dois se repete no ANP (§ da mensagem anterior). A hipótese óbvia —
"o mecanismo estatístico está mal calibrado no ANP" — **não se sustenta**: a
distribuição de margem/p-valor por veredito é praticamente idêntica entre os dois
datasets (`separated` tem margem maior e dm_pvalue menor em ambos, na mesma ordem
de grandeza). O teste está funcionando igual nos dois lugares.

A causa real, isolada reproduzindo o mesmo experimento do NN5 (22 regras fixas —
`select_stable`/`select_top_k`, k∈{3,5,7,9,11}, mean/median — comparando ranking de
validação vs ranking de teste):

```
Spearman(validação, teste), NN5  (nested_selection=True): +0.547
Spearman(validação, teste), ANP  (nested_selection=True): -0.290
```

**No ANP o score de validação não carrega o mesmo sinal.** Escolher pela validação
mal empata com não escolher nada:

```
escolher pelo score de validação          : 0.219368 sMAPE
sempre aplicar a média do pool completo   : 0.220649 sMAPE   (a "escolha" quase não ajuda)
ORACLE (melhor das 22 por série)          : 0.199101 sMAPE   (o ganho possível existe, só não é capturado)
```

Isso explica os dois padrões que não replicaram de uma vez: se o score de 3 janelas
não prediz o teste no ANP, nem a confiança estatística sobre esse score (`separated`)
nem a seleção de pool que depende dele (pool menor) têm por que se transferir.

Descartei que fosse o artefato de zero (§12.3): só 5/182 séries têm zero na janela
de teste, e excluí-las não muda o quadro.

**Hipótese em aberto, não testada:** ANP mistura produtos/regiões com regimes bem
diferentes (a série mais estável e a mais intermitente estão no mesmo dataset),
enquanto NN5 é um benchmark homogêneo (todas séries semanais de caixa eletrônico,
mesmo tipo de padrão). Um sinal de validação de 3 janelas pode ser suficiente
quando as séries são parecidas entre si e insuficiente quando cada série tem sua
própria dinâmica. Vale testar segmentando por característica da série (força de
tendência/sazonalidade, escala) antes de aceitar essa hipótese como resposta final.

### 12.3 Dois diagnósticos novos, implementados

**Detalhe do erro de tool no artifact.** Antes, `tool_missing=True` numa linha do
CSV não dizia por quê — o `kind`/`detail` de cada chamada que falhou vivia só em
`state.tool_errors`, nunca gravado em lugar nenhum; para descobrir a causa era
preciso cruzar a coluna `tools_called` na mão. Agora `artifacts_payload()["react"]`
inclui `"tools": tools_called_summary(state)` — mesmo dado que a linha do CSV usa,
com o motivo de cada falha anexado.

No ANP isso revelou algo que só ficaria visível cruzando dados manualmente: dos
35/182 series com `tool_missing=True`, **34 foram "unparsed"** (o modelo não emitiu
Thought/Action/Action Input num turno) — não é o agente pedindo ferramenta ou
argumento inexistente, é formatação de saída. A coluna mistura os dois sinais sob
o mesmo nome.

**`test_has_zero_actual` / `test_min_abs_actual`.** sMAPE tem denominador
`(|previsão| + |real|) / 2`; quando o real é zero ou perto disso, esse denominador
colapsa e o ponto satura perto de 200%, **para qualquer método**. Confirmado nas
duas piores séries do ANP (18 e 85): agente e as cinco baselines externas todas
entre 1.34 e 1.44 de sMAPE — não é nenhum método prevendo mal, é o real bater em
zero em pelo menos um mês.

```
série 85, janela de teste: [20, 45, 15, 20, 5, 30, 0, 0, 10, 0, 15, 5]
                                                    ^^^^   ^^     ^^
```

Isso **não muda a fórmula do sMAPE** — ela continua byte-idêntica via
`all_functions`, como o contrato da Seção 4.1 exige, comparável com toda linha já
escrita por qualquer versão do projeto. Só adiciona visibilidade: um booleano
(`test_has_zero_actual`) e o valor absoluto mínimo do real na janela
(`test_min_abs_actual`, contínuo, para quem quiser aplicar seu próprio limiar em
vez de herdar um cravado no pipeline). Calculado só de `test_values`, no mesmo
ponto do pipeline onde as métricas já são calculadas — não pode enxergar nada que
as métricas não vejam.
## 13. `weights_pooled_meta_model` — a peça clássica que faltava (ADE/FFORMA)

### 13.1 O que já existia, e por que estava morto

`weights_feature_based` já era um meta-modelo XGBoost no espírito do FFORMA —
treinado **por série**, sobre as 3 janelas de validação daquela série. A checagem
de amostra suficiente:

```python
if n_fit < 2 * feats.shape[1]:   # n_fit=3, feats.shape[1]=5  ->  3 < 10, sempre
    return _softmax_fallback(...)
```

Com 3 janelas essa condição é **sempre verdadeira**, para qualquer contagem
razoável de features. Confirmado em log: essa tool nunca apareceu na frequência de
chamadas em nenhuma rodada real (NN5 v1/v2, ANP v3) — o caminho XGBoost nunca
executou uma vez sequer.

### 13.2 O que a pesquisa mostrou (pedido do usuário, ver mensagem anterior)

O FFORMA real não re-treina por série. Ele extrai ~43 características **por
série** (força de tendência, sazonalidade, entropia — a mesma família que
`series_profile` já calcula) e treina **um único meta-modelo usando todas as
séries do dataset como amostras de treino**. A unidade estatística é "quantas
séries tem o dataset" (111 no NN5, 182 no ANP), não "quantas janelas tem uma
série". Isso resolve exatamente o defeito de 13.1.

### 13.3 Implementação

`orchestrator_react/meta_model.py` — novo módulo:

- `extract_meta_features(profile)`: 4 características, deliberadamente restritas
  à forma histórica da série (`trend_strength`, `seasonal_strength`,
  `spectral_entropy`, `acf1`) — todas vêm de `train_series`, que é sempre
  totalmente conhecido antes da Fase 3 abrir e **não muda por fold de backtest**.
  Isso é o que permite reaplicar o mesmo vetor de pesos em todo fold sem recalcular.
- `build_meta_row`: uma linha de treino por série (features + erro de validação de
  cada modelo do pool). Só usa `y_true`/`y_preds` de validação — nunca teste.
- `build_pooled_meta_models`: treina **leave-one-series-out** — um modelo por
  série, cada um excluindo a própria linha daquela série do treino. Mesma
  disciplina do `nested_selection`, pelo mesmo motivo: consultar um modelo sobre a
  série que o treinou mediria memorização, não generalização.
  Retorna `{}` (ferramenta retirada do catálogo, como o `weights_ols`) com menos
  de `pooled_meta_model_min_series` séries (padrão 20) ou sem xgboost instalado.

Nova tool em `tools.py`: `weights_pooled_meta_model(state, pool=FULL_POOL, eta=1.0)`.
Exige um pool cuja composição seja fixa entre folds — recusa `select_top_k`/
`select_stable`/`prune_redundant` sob `nested_selection=True` com um erro que o
agente lê e corrige, porque o vetor de pesos é calculado uma vez e reaplicado
igual em todo fold; se o pool mudasse de tamanho por fold, o vetor ficaria com
tamanho errado.

Encaixe: `pipeline.run_dataset` roda um pré-passo (Fase 0 + `series_profile`, sem
LLM, descartando o `state` pesado logo depois) sobre todas as séries do `todo`
**antes** do loop por série, treina os N modelos LOSO uma vez, e cada série recebe
o seu via `run_series(..., pooled_meta_model=meta_models.get(idx))`.

Config: `pooled_meta_model: bool = True`, `pooled_meta_model_min_series: int = 20`.
CLI: `--no-pooled-meta-model`, `--pooled-meta-model-min-series`.

### 13.4 Validação em dado real (NN5, 111 séries, pool completo)

```
weights_pooled_meta_model : 0.119072 sMAPE
weights_softmax_neg_error : 0.118576 sMAPE   (empate, p=0.73)
mean                      : 0.119939 sMAPE   (pooled bate a média, p=0.0088)
```

**Resultado honesto: empata com o `softmax_neg_error` já existente, bate a média
simples.** Não é a resposta que fecha a lacuna com o ADE sozinha — coerente com o
achado já registrado (§7) de que o valor está na seleção do pool, não na
ponderação. É uma ferramenta nova, legítima e testada no catálogo, não uma vitória
decisiva. Fica disponível para o agente combinar com seleção de pool fixa; não
testei ainda seu comportamento sobre subconjuntos menores (onde `error_trend`
mostrou alguma vantagem antes).



---

## 14. v5 — objetivo FFORMA, prior de dataset, e o leque do agente 📊

Sessão de 2026-07-30. Todos os números do braço **determinístico** (sem LLM),
medidos com `run_dataset` de verdade, nos dois datasets.

### 14.1 O que decidiu: o OBJETIVO, não as features

Investigando por que o FFORMA nos batia no ANP, li `combinations/fforma.py` e
isolei a diferença com features e folds idênticos:

| | nosso desenho anterior | FFORMA |
|---|---|---|
| estrutura | 19 regressores independentes, cada um prevendo o erro do *seu* modelo | 1 booster multi-classe cuja saída softmax **é** o vetor de pesos |
| objetivo | erro individual | gradiente custom que minimiza o **erro da combinação** |
| aprende interação? | não | sim |

```
ANP (182 séries, LOSO, mesmas 26 features):
  regressores por modelo : 0.220466
  objetivo FFORMA        : 0.215938   p=0.030   ← passa o FFORMA real (0.216593)
```

**Isso reconcilia uma contradição aberta do relatório anterior.** A busca gulosa
(que também otimiza erro combinado) era a *pior* estratégia nos dois datasets — mas
com 24 pontos por série. O mesmo objetivo com 182 séries de amostra é o *melhor*.
Otimização direta do erro combinado **falha por série e funciona pooled** — é a
fronteira exata do *forecast combination puzzle* nos nossos dados.

### 14.2 A métrica das contribuições é load-bearing

O gradiente soma contribuições entre séries. Quatro variantes, ANP:

| contribuição | sMAPE | por quê |
|---|---|---|
| RMSE cru | 0.2224 | poluição de escala: séries em milhões dominam o gradiente |
| **sMAPE cru** | **0.2160** | **o que ficou** — já é livre de escala por ponto |
| sMAPE normalizado por linha | 0.2202 | achata séries difíceis e fáceis; a perda é o erro TOTAL, difícil deve pesar mais |
| sMAPE reescalado globalmente | 0.2196 | razões preservadas, mas o divisor age como mudança de passo do boosting |

Efeito colateral documentado: com erros uniformemente minúsculos o gradiente não
move o `base_score` e o booster devolve margens iguais. `PooledMetaModel.degenerate`
reporta isso em vez de aplicar pesos uniformes disfarçados.

catch22 (4→26 features): **efeito ≈ 0** nos dois datasets (p=0.81 / 0.97). Mantido
por fidelidade ao FFORMA, mas **não** é a explicação da vantagem dele.

### 14.3 Piso determinístico final

| | ANP | NN5 |
|---|---|---|
| piso sem a semente pooled | 0.220287 | 0.115361 |
| **piso com a semente pooled (default v5)** | **0.219040** | **0.115394** |
| semente pooled aplicada sozinha | 0.215957 | 0.119702 |
| a semente vence o piso em | 51/182 (28%) | 10/111 (9%) |

ANP melhora (+0.0012, p=0.13), NN5 neutro (−0.00003, p=0.39). Seguro como default.

### 14.4 `prior_blend` — o maior lever do ANP, e por que NÃO é default

Encolher o score de 3 janelas de cada tentativa em direção ao prior LOSO do dataset
(empirical Bayes). Produção reproduz o offline exatamente:

| α | ANP | NN5 |
|---|---|---|
| 0.0 (default) | 0.219040 | **0.115394** |
| 0.6 | 0.216653 | 0.117808 |
| **0.8** | **0.214529** ← passa o FFORMA | 0.118786 |

**Direções opostas e monotônicas.** Tentei três formas honestas de escolher α, todas
falharam:

1. **α fixo** — ajuda um dataset, prejudica o outro.
2. **α por leave-one-window-out dentro da validação** — no ANP escolhe 0.60 (bom,
   0.2167); no NN5 escolhe 0.80 quando 0.00 é o certo. Os scores de validação ficam
   dentro de 0.005 uns dos outros: ruído.
3. **α por estabilidade do ranking de estratégias** — discrimina **ao contrário**:
   ANP tau +0.135 > NN5 +0.091, quando é o ANP que precisa desconfiar da própria
   validação.

Então `final_prior_alpha=0.0` (off) por padrão, opt-in explícito por dataset. **É aqui
que o agente tem um papel que o braço determinístico não pode preencher:** ele recebe
o prior no DATASET CARD e decide por série quanto pesar — um julgamento contextual,
não uma constante que eu tenha que fixar.

### 14.5 O leque do agente (P1–P3)

Três mudanças vindas da forense das 182 trajetórias do v4:

**Dataset card** (`dataset_card=True`): as estratégias semeadas ranqueadas pelo
score de validação nas *outras* N−1 séries, com a ressalva de que a ordem não
sempre vale e que o histórico da própria série manda mais. Recomendação, nunca
restrição — o catálogo continua aberto.

**Anti-aliasing** (`ReactState.numerical_twin`): 591/756 avaliações do v4 pediram
`weighted` e 62% dos vencedores eram aritmeticamente a média do pool. Agora a
observação diz `SAME forecasts as a3 — adds nothing new`, com a instrução de mudar
o pool ou o tipo de combinação em vez do método de peso.

**Prompt desancorado:** o exemplo trabalhado citava `weights_inverse_error`
literalmente e recebeu 462 das chamadas da família (462→223→10→2→1 por posição).
Agora mostra `weights_<method>` mais uma seção explicando que sinal cada família lê,
e uma regra: concentração ≈ 0 significa que aquela estratégia é a média do pool.

### 14.6 P4 — a armadilha do resume, e as opções de servidor

**O mistério das 17 séries, resolvido.** No v4 do ANP, `weights_pooled_meta_model`
apareceu como retirada em 17 séries. Minha hipótese anterior (artefato do resume
81–181) estava errada. Os índices são **165–181, contíguos** — exatamente 17 séries,
e `pooled_meta_model_min_series` é 20. A rodada foi terminada num chunk de 17, o
pré-passo devolveu `{}`, e a tool foi retirada para aquelas 17 linhas.

Causa estrutural: **o meta-modelo pooled treina nas séries DA CHAMADA, não do
dataset.** Um chunk de resume é, para o pré-passo, um dataset menor. Consequência: um
resume pequeno troca a arquitetura de parte do CSV, silenciosamente.

Corrigido em três lugares:
- `run_dataset` distingue "dataset pequeno" de "chunk de resume pequeno" e anexa o
  motivo aos `warnings` de cada série afetada (vai para o CSV e o artifact);
- `exec_dataset_orchestrator` avisa no início do log, antes de gastar horas;
- `EXTRA_DEPENDENCIES.txt` §7b documenta para o servidor.

**Opções de servidor (`LLMRole.reasoning`).** Os 90 vazios e os erros
`error parsing tool call` do v4 vêm do canal de raciocínio do formato harmony do
gpt-oss (ollama/ollama#11781, #11800). `--no-reasoning` passa `reasoning=False`,
verificado como parâmetro real do `langchain-ollama` instalado. Fica **unset por
padrão** — todos os resultados até aqui usaram o default do servidor, então mudar
isso é um braço de A/B deliberado, e entra no `fingerprint()`. `format="json"` está
descartado por documentação: no gpt-oss devolve vazio sempre (#11867).

### 14.7 P5 — preparação, e dois bugs que a preparação encontrou

Rodar o smoke pelo ponto de entrada real (não pelos scripts de medição) expôs dois
problemas que teriam contaminado a rodada v5 inteira.

**1. O CLI nunca era despachado.** `run_tsf_orchestrator.py` tem `build_parser()` e
`main()` completos, mas o bloco `if __name__ == "__main__":` chamava
`exec_dataset_orchestrator` direto com NN5 fixo. Ou seja: `python
run_tsf_orchestrator.py --dataset ANP_MONTHLY ...` **ignorava todos os flags** e
rodava NN5. Todo flag adicionado nas últimas sessões (`--final-prior-alpha`,
`--pooled-objective`, `--no-reasoning`, `--no-dataset-card`) era inalcançável.
Agora, havendo argumentos, o `__main__` despacha para `main()`; sem argumentos, o
bloco editável continua valendo, que é como o servidor sempre foi dirigido.

**2. O objetivo fforma degenera em amostra pequena.** O bloco `cross_series` novo
(§14.7 abaixo) mostrou `degenerate=True` num smoke de 25 séries: com 24 linhas de
treino o gradiente — proporcional às contribuições — não move o `base_score`, e
todo modelo sai com a mesma margem, isto é, pesos uniformes com nome de
meta-modelo. O run completo de 182 séries aprende normalmente (0.2160), então é
piso de amostra, não defeito do objetivo. `build_pooled_meta_models` agora cai para
`per_model` quando **todos** os fits saem degenerados — `per_model` ajusta
regressores independentes e não tem esse piso. Confirmado em produção:
`objective=per_model, degenerate=False` no mesmo smoke.

### 14.8 Observabilidade cross-series (`cross_series` no artifact)

Sem isto, uma rodada v5 não poderia ser analisada como analisei v3 e v4. Cada
artifact passa a trazer:

```json
"cross_series": {
  "strategy_prior":   {"dba": 0.6612, "mean": 0.6698, ...},
  "prior_best": "dba", "prior_worst": "mean pool=pool1",
  "dataset_card_shown": { ... o que o agente de fato leu ... },
  "pooled_meta_model": {"objective": "fforma", "n_train_series": 181,
                        "n_features": 26, "degenerate": false}
}
```

Tudo validação-only e leave-one-series-out por construção — registrar não expõe
nada que a rodada já não tivesse. `degenerate` é o campo que impede uma rodada de
*parecer* ter usado meta-modelo sem ter usado.
