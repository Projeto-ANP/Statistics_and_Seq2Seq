# Alterações do capítulo de resultados — CARTE

Este diretório reúne a reescrita do capítulo `Avaliação Experimental e Discussão de Resultados` de `overleaf/resultados.tex` para o CARTE.

## Configuração de referência

- Resultados do CARTE e da comparação entre modelos de linguagem: CSVs em `timeseries/mestrado/resultados/`.
- Modelo do agente: `gpt-oss:20b`.
- Métricas reportadas: **SMAPE** e **POCID**. O RMSE é usado apenas na construção do contexto e da preparação de pools/pesos, não como resultado.
- O POCID é reportado separadamente e não participa do ranking de estratégias.

## Regras editoriais

- Usar **CARTE**, nunca CREST.
- Não mencionar sementes, aceite, estagnação, paciência, prior ou meta-modelo entre séries.
- Não usar “catálogo de 12 ações”; quando necessário, referir as ações pelos nomes conceituais.
- Tabelas em percentual, com duas casas decimais e vírgula decimal.
- Comparações por série usam o SMAPE; o POCID é analisado em separado.
- Todas as tabelas de comparação — modelos individuais, estratégias de combinação e comparações pareadas — são ordenadas por **SMAPE crescente**. O POCID é coluna complementar e não altera a ordenação, exceto para desempatar valores iguais de SMAPE.
- A agregação principal é a **média dos sete conjuntos de dados**, que dá o mesmo peso a cada conjunto. O **agregado das 680 séries** é reportado como complemento.
- Destacar a constância do CARTE no topo do ranking, especialmente no POCID: quando não for o primeiro, verificar e comentar a posição no top 2. Não forçar a afirmação quando o resultado não sustentar.
- A comparação pareada é apresentada pela **MCM em imagem**, não por uma tabela separada. A tabela de estratégias de combinação concentra as médias de SMAPE e POCID.
- Nos textos por conjunto de dados, manter a sequência: modelos individuais, estratégias de combinação, comparação geral e discussão.

## Estrutura do capítulo

| Seção em `resultados.tex` | Linhas atuais (aprox.) | Arquivo | Status |
|---|---:|---|---|
| `Considerações iniciais` | 3–5 | [`01_consideracoes_iniciais.md`](01_consideracoes_iniciais.md) | pronto para integrar |
| `Design experimental` | 7–13 | [`02_design_experimental.md`](02_design_experimental.md) | pronto para integrar |
| `Escolha do modelo de linguagem` | 15–75 | [`03_modelo_linguagem.md`](03_modelo_linguagem.md) | pronto; Qwen completo e Gemma parcial |
| `ANP` | 77–156 | [`04_anp.md`](04_anp.md) | pronto para integrar |
| `NN5` | 157–233 | [`05_nn5.md`](05_nn5.md) | pronto para integrar |
| `M4` | 234–308 | [`06_m4.md`](06_m4.md) | pronto para integrar |
| `ETTh1` | 309–383 | [`07_etth1.md`](07_etth1.md) | pronto para integrar |
| `ETTh2` | 384–458 | [`08_etth2.md`](08_etth2.md) | pronto para integrar |
| `ETTm1` | 459–533 | [`09_ettm1.md`](09_ettm1.md) | pronto para integrar |
| `ETTm2` | 534–608 | [`10_ettm2.md`](10_ettm2.md) | pronto para integrar |
| `Síntese entre conjuntos de dados` | 609–752 | pendente | — |
| `Análise do comportamento do agente` | 753–947 | pendente | — |
| `Estudo de ablação` | 948–1151 | pendente | decidir escopo |
| `Considerações finais` | 1152–fim | pendente | — |

## Fontes de dados

- Resultados do CARTE e da comparação entre modelos: CSVs em `timeseries/mestrado/resultados/`.
- Baselines `mean`, `median`, `dba`, `ADE` e `FFORMA`: campo `baseline_results_json` dos CSV do CARTE.
- Modelos individuais: tabelas já existentes em `overleaf/resultados.tex`, que descrevem o mesmo pool de preditores.
- MCMs do CARTE: geradas por `outputs/resultados/scripts/mcm_carte.py`; em `overleaf/resultados_final` ficam apenas os PNGs usados no capítulo. PDFs e CSVs auxiliares ficam em `outputs/resultados/mcm_carte`.
- MCMs agregadas: `mcm_agregado_smape.png` e `mcm_agregado_pocid.png` usam as 680 séries; `mcm_media_datasets_smape.png` e `mcm_media_datasets_pocid.png` usam uma linha por conjunto de dados.

## Pendências de decisão

- Definir se o estudo de ablação será mantido ou substituído por uma análise das decisões finais do CARTE.
- Definir quais figuras de MCM serão refeitas para o CARTE.
- Atualizar as referências a `sec:preparacao` e `CREST` que ainda existem em `resultados.tex`.
