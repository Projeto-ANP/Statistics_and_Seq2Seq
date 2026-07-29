# Arquitetura do agente combinador ReAct

Documento de referência da arquitetura: o que cada peça faz, **por que** ela existe,
e qual medição justificou cada decisão. Nada aqui é preferência de estilo — cada
seção marcada com 📊 registra o número que motivou a escolha, medido nas 111 séries
do `NN5_WEEKLY_DATASET`.

Última atualização: 2026-07-28.

---

## 1. Visão geral

Um único agente ReAct opera sobre um catálogo fechado de 23 ferramentas
determinísticas. O agente **nunca escreve números** — nem previsões, nem pesos. Ele
escolhe qual ferramenta chamar; a ferramenta calcula e devolve um *handle*
(`pool1`, `w2`) mais um resumo qualitativo.

```
Fase 0  ingestão          .tsf + CSVs dos modelos individuais -> ReactState
Fase 1  diagnóstico       series_card + pool_card (LLM opcional)
Fase 2  baselines         mean / median / dba semeadas no histórico
Fase 3  loop ReAct        Thought -> Action -> Observation, até 8 iterações
Fase 4  aplicação         a melhor tentativa é aplicada à janela de teste cega
Fase 5  relato            justificativa final + CSV de 56 colunas
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

## 3. O catálogo de ferramentas (23)

### 3.1 Diagnóstico
`series_profile` · `stl_summary` · `error_summary` · `ranking_stability` ·
`error_correlation` · `dm_test`

### 3.2 Seleção de pool
`select_top_k` · `select_stable` · `prune_redundant`

Todas registram uma `PoolRecipe` re-ajustável (§2.1).

### 3.3 Pesos
`weights_inverse_error` · `weights_softmax_neg_error` · `weights_error_trend` ·
`weights_ols` · `weights_feature_based`

### 3.4 Combinação
`combine_mean` · `combine_median` · `combine_trimmed_mean` · `combine_weighted` ·
`combine_dba` · `combine_best_single`

### 3.5 Validação
`evaluate_strategy` · `sanity_check` · `list_attempts`

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
| `combiners.py` | `apply_combination` — fonte única, backtest e aplicação final |
| `features.py` | STL, sazonalidade declarada, campeões de componente |
| `tools.py` | o catálogo de 23 ferramentas |
| `registry.py` | espaço de ação fechado, despacho, `withheld_tools` |
| `prompts.py` | prompt de sistema e de turno |
| `react_loop.py` | Fase 3 — laço Thought/Action/Observation |
| `pipeline.py` | orquestra as fases, `SeriesOutcome`, `reducibility()` |
| `csv_writer.py` | contrato de 56 colunas |
| `llm.py` | cliente Ollama, parser de passo, `ScriptedLLM` |

**364 testes**, todos em CPU, sem servidor.
