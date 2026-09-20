# Recorrência de vocabulário nas justificativas do agente (origin=agent)

Só números, medidos exatamente como pedido nos Passos 1-4. Sem conclusão qualitativa.

## Passo 1 — amostra

Total de séries com `origin=agent`: **273** (de 680 séries no total, todos os 7 datasets).

| dataset | n_series |
|---|---|
| ANP_MONTHLY | 77 |
| NN5_WEEKLY_DATASET | 50 |
| M4_WEEKLY_DATASET | 135 |
| ETTH1 | 3 |
| ETTH2 | 2 |
| ETTM1 | 3 |
| ETTM2 | 3 |

## Passo 2 — repetição de frases exatas (n-gramas de 5 a 8 palavras)

Tokenização: minúsculas, hífens/travessões tratados como espaço, mantido só `[a-z0-9]+`. N-gramas contados por presença no documento (um n-grama repetido dentro da mesma justificativa conta uma vez). Limiar de 20% da amostra = **54.6** séries (arredondado para cima na contagem).

N-gramas de 5-8 palavras que aparecem em ≥20% das 273 justificativas: **3**.

Top 15 por frequência:

| n-grama | contagem | percentual |
|---|---|---|
| "the series is trend dominated" | 69 | 25.3% |
| "series is trend dominated with" | 68 | 24.9% |
| "the series is trend dominated with" | 66 | 24.2% |

Variação de "mitigat(ing/es) overfitting to the few validation windows" (regex `mitigat\w*\s+overfit\w*\s+(?:to\s+)?the\s+few\s+validation\s+windows`, case-insensitive): aparece em **7/273** justificativas (**2.6%**).

## Passo 3 — similaridade estrutural

Jaccard sobre o conjunto de palavras de conteúdo (stopwords removidas), calculado sobre **todos os 37128 pares** da amostra (não uma subamostra aleatória — com 273 séries o cálculo completo é factível e mais preciso que amostrar).

- Média: **0.1897**
- Mediana: **0.1778**

Separando por `estrategia_params` (chave = `combine`+`pool`+`weights`/`trim_pct`/`model` conforme aplicável, comparado como JSON canônico):

- Pares com a **mesma** estratégia (n=1564): média=0.2303, mediana=0.2129
- Pares com estratégia **diferente** (n=35564): média=0.1879, mediana=0.1765

Primeira frase (até a primeira vírgula/ponto/ponto-e-vírgula): **231** moldes distintos por correspondência textual exata (entre 273 séries). Os 5 mais comuns por correspondência exata:

| primeira frase (exata) | n_series |
|---|---|
| "the series is trend‑dominated with high predictability and unstable model rankings across windows" | 9 |
| "the series is trend‑dominated with high predictability and moderate ranking stability" | 7 |
| "the series has very strong trend and seasonality but unstable model rankings and high error correlation" | 6 |
| "the series is trend‑dominated with unstable model rankings and high error correlation" | 4 |
| "the series is trend‑dominated with unstable model rankings across windows" | 3 |

Agrupando por quase-duplicata (Jaccard sobre palavras de conteúdo da primeira frase ≥ 0.6, clusterização por componentes conexos): **95** moldes distintos ao todo; **41** moldes cobrem ≥80% da amostra (219/273 séries nesses 41 moldes). Tamanhos dos moldes, do maior ao menor: [118, 24, 10, 8, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].

## Passo 4 — variação por dataset e por estratégia

### 4.1 — similaridade Jaccard por dataset (mesma métrica do Passo 3.1, só dentro de cada dataset)

| dataset | n_series | n_pares | média | mediana |
|---|---|---|---|---|
| ANP_MONTHLY | 77 | 2926 | 0.2115 | 0.1967 |
| NN5_WEEKLY_DATASET | 50 | 1225 | 0.2004 | 0.1818 |
| M4_WEEKLY_DATASET | 135 | 9045 | 0.1899 | 0.1778 |
| ETTH1 | 3 | 3 | 0.1518 | 0.1618 |
| ETTH2 | 2 | 1 | 0.0588 | 0.0588 |
| ETTM1 | 3 | 3 | 0.2542 | 0.2321 |
| ETTM2 | 3 | 3 | 0.2175 | 0.2500 |

### 4.2 — similaridade da primeira frase por mecanismo de formação do pool da estratégia final

Mecanismo = ferramenta que criou o `pool` referenciado por `best_strategy_params` daquela série (`prune_redundant`, `select_stable`, `select_top_k`), `pool_full` se a estratégia usa o pool cheio sem nenhuma dessas ferramentas, ou `outro_ou_nao_rastreado` se o pool não foi encontrado na trajetória registrada. Verificado manualmente: os pools em `outro_ou_nao_rastreado` são nomeados `pool1`/`pool2`/`pool3` (ex.: `ANP_MONTHLY` série 1, `pool1`; série 41, `pool3`) -- esses nomes batem com os pools pré-semeados automaticamente por `select_stable` antes do loop ReAct começar (`orchestrator_react/pool.py:SEED_STABLE_POOLS`), não com uma ação dentro da própria trajetória do agente, por isso não aparecem em `react_trajectory_json`.

| mecanismo | n_series | n_pares internos | média Jaccard (1ª frase) dentro do mecanismo |
|---|---|---|---|
| prune_redundant | 140 | 9730 | 0.1426 |
| outro_ou_nao_rastreado | 112 | 6216 | 0.1603 |
| pool_full | 17 | 136 | 0.0919 |
| select_stable | 2 | 1 | 0.0526 |
| select_top_k | 2 | 1 | 0.0714 |

Média Jaccard da 1ª frase entre pares de mecanismos **diferentes** (n=21044 pares): **0.1345**.
