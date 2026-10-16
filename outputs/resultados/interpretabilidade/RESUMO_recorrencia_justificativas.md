# Recorrência de vocabulário nas justificativas do agente (origin=agent)

Só números, medidos exatamente como pedido nos Passos 1-4. Sem conclusão qualitativa.

## Passo 1 — amostra

Total de séries com `origin=agent`: **47** (de 680 séries no total, todos os 7 datasets).

| dataset | n_series |
|---|---|
| ANP_MONTHLY | 11 |
| NN5_WEEKLY_DATASET | 7 |
| M4_WEEKLY_DATASET | 27 |
| ETTH1 | 1 |
| ETTH2 | 0 |
| ETTM1 | 1 |
| ETTM2 | 0 |

## Passo 2 — repetição de frases exatas (n-gramas de 5 a 8 palavras)

Tokenização: minúsculas, hífens/travessões tratados como espaço, mantido só `[a-z0-9]+`. N-gramas contados por presença no documento (um n-grama repetido dentro da mesma justificativa conta uma vez). Limiar de 20% da amostra = **9.4** séries (arredondado para cima na contagem).

N-gramas de 5-8 palavras que aparecem em ≥20% das 47 justificativas: **0**.

Top 15 por frequência:

| n-grama | contagem | percentual |
|---|---|---|

Variação de "mitigat(ing/es) overfitting to the few validation windows" (regex `mitigat\w*\s+overfit\w*\s+(?:to\s+)?the\s+few\s+validation\s+windows`, case-insensitive): aparece em **0/47** justificativas (**0.0%**).

## Passo 3 — similaridade estrutural

Jaccard sobre o conjunto de palavras de conteúdo (stopwords removidas), calculado sobre **todos os 1081 pares** da amostra (não uma subamostra aleatória — com 47 séries o cálculo completo é factível e mais preciso que amostrar).

- Média: **0.1300**
- Mediana: **0.1154**

Separando por `estrategia_params` (chave = `combine`+`pool`+`weights`/`trim_pct`/`model` conforme aplicável, comparado como JSON canônico):

- Pares com a **mesma** estratégia (n=30): média=0.2371, mediana=0.2000
- Pares com estratégia **diferente** (n=1051): média=0.1269, mediana=0.1143

Primeira frase (até a primeira vírgula/ponto/ponto-e-vírgula): **46** moldes distintos por correspondência textual exata (entre 47 séries). Os 5 mais comuns por correspondência exata:

| primeira frase (exata) | n_series |
|---|---|
| "trend‑dominated series" | 2 |
| "median on pool5 is the best robust combination for this trend‑dominated series and the margin is not statistically significant" | 1 |
| "trend‑dominated series with unstable ranking" | 1 |
| "rf alone is the best performer" | 1 |
| "the series shows a strong upward trend and high predictability" | 1 |

Agrupando por quase-duplicata (Jaccard sobre palavras de conteúdo da primeira frase ≥ 0.6, clusterização por componentes conexos): **39** moldes distintos ao todo; **30** moldes cobrem ≥80% da amostra (38/47 séries nesses 30 moldes). Tamanhos dos moldes, do maior ao menor: [4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].

## Passo 4 — variação por dataset e por estratégia

### 4.1 — similaridade Jaccard por dataset (mesma métrica do Passo 3.1, só dentro de cada dataset)

| dataset | n_series | n_pares | média | mediana |
|---|---|---|---|---|
| ANP_MONTHLY | 11 | 55 | 0.1862 | 0.1714 |
| NN5_WEEKLY_DATASET | 7 | 21 | 0.0866 | 0.0909 |
| M4_WEEKLY_DATASET | 27 | 351 | 0.1367 | 0.1176 |
| ETTH1 | 1 | 0 | n/a (0 pares) | n/a |
| ETTH2 | 0 | 0 | n/a (0 pares) | n/a |
| ETTM1 | 1 | 0 | n/a (0 pares) | n/a |
| ETTM2 | 0 | 0 | n/a (0 pares) | n/a |

### 4.2 — similaridade da primeira frase por mecanismo de formação do pool da estratégia final

Mecanismo = ferramenta que criou o `pool` referenciado por `best_strategy_params` daquela série (`prune_redundant`, `select_stable`, `select_top_k`), `pool_full` se a estratégia usa o pool cheio sem nenhuma dessas ferramentas, ou `outro_ou_nao_rastreado` se o pool não foi encontrado na trajetória registrada. Verificado manualmente: os pools em `outro_ou_nao_rastreado` são nomeados `pool1`/`pool2`/`pool3` (ex.: `ANP_MONTHLY` série 1, `pool1`; série 41, `pool3`) -- esses nomes batem com os pools pré-semeados automaticamente por `select_stable` antes do loop ReAct começar (`orchestrator_react/pool.py:SEED_STABLE_POOLS`), não com uma ação dentro da própria trajetória do agente, por isso não aparecem em `react_trajectory_json`.

| mecanismo | n_series | n_pares internos | média Jaccard (1ª frase) dentro do mecanismo |
|---|---|---|---|
| pool_full | 20 | 190 | 0.1385 |
| outro_ou_nao_rastreado | 19 | 171 | 0.0784 |
| prune_redundant | 8 | 28 | 0.1328 |

Média Jaccard da 1ª frase entre pares de mecanismos **diferentes** (n=692 pares): **0.0701**.
