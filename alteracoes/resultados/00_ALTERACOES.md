# Alterações do capítulo de resultados — CARTE

Este diretório reúne a reescrita do capítulo `Avaliação Experimental e Discussão de Resultados` de `overleaf/resultados.tex` para o CARTE.

## Configuração de referência

- Resultados do CARTE e da comparação entre modelos de linguagem: CSVs em `timeseries/mestrado/resultados/`.
- Modelo do agente: `gpt-oss:20b`.
- Métricas reportadas: **SMAPE** e **POCID**. O RMSE é usado apenas na construção do contexto e da preparação de pools/pesos, não como resultado.
- O POCID é reportado separadamente e não participa do ranking de estratégias.

## Regras editoriais

- O padrão de escrita está em [`00_ESTILO.md`](00_ESTILO.md).
- Usar **CARTE**, nunca CREST.
- Sementes e execuções são tratadas na Seção~\ref{sec:execucoes-seeds}. Os textos por conjunto de dados usam a média das dez execuções e não repetem a explicação do protocolo. Aceite, estagnação, paciência, prior e meta-modelo entre séries continuam fora do texto.
- Não usar “catálogo de 12 ações”; quando necessário, referir as ações pelos nomes conceituais.
- Tabelas em percentual, com duas casas decimais e vírgula decimal.
- Comparações por série usam o SMAPE; o POCID é analisado em separado.
- Todas as tabelas de comparação — modelos individuais, estratégias de combinação e comparações pareadas — são ordenadas por **SMAPE crescente**. O POCID é coluna complementar e não altera a ordenação, exceto para desempatar valores iguais de SMAPE. A indicação textual desse critério aparece uma única vez, na primeira tabela da ANP; não repetir nas demais seções.
- A comparação entre modelos de linguagem usa **uma execução de cada modelo** e serve apenas para a escolha. Os resultados do CARTE usam a **média de dez execuções independentes**.
- A análise qualitativa do agente usa a execução com **semente 11** da série de dez. Os traces e estudos de caso não representam a execução média.
- Não repetir a definição de SMAPE e POCID nas seções por conjunto de dados. A definição e o protocolo de agregação ficam no design experimental.
- Evitar dois-pontos e ponto e vírgula em excesso. Preferir períodos e vírgulas.
- A agregação principal é a **média dos sete conjuntos de dados**, que dá o mesmo peso a cada conjunto. O **agregado das 680 séries** é reportado como complemento.
- Destacar a constância do CARTE no topo do ranking, especialmente no POCID: quando não for o primeiro, verificar e comentar a posição no top 2. Não forçar a afirmação quando o resultado não sustentar.
- A comparação pareada é apresentada pela **MCM em imagem**, não por uma tabela separada. A tabela de estratégias de combinação concentra as médias de SMAPE e POCID.
- Nos textos por conjunto de dados, manter a sequência: modelos individuais, estratégias de combinação, comparação geral e discussão.

## Estrutura do capítulo

| Seção em `resultados.tex` | Linhas atuais (aprox.) | Arquivo | Status |
|---|---:|---|---|
| `Considerações iniciais` | 3–5 | [`01_consideracoes_iniciais.md`](01_consideracoes_iniciais.md) | pronto para integrar |
| `Design experimental` | 7–13 | [`02_design_experimental.md`](02_design_experimental.md) | pronto para integrar |
| `Escolha do modelo de linguagem` | 48–110 | [`03_modelo_linguagem.md`](03_modelo_linguagem.md) | mantido; texto do .tex preservado |
| `Execuções, sementes e agregação` | nova | [`03_modelo_linguagem.md`](03_modelo_linguagem.md) (segunda seção) | pronto para integrar |
| `ANP` | 116–197 | [`04_anp.md`](04_anp.md) | atualizado para a média das 10; MCM regerada |
| `NN5` | 198–277 | [`05_nn5.md`](05_nn5.md) | atualizado para a média das 10; MCMs regeradas |
| `M4` | 234–308 | [`06_m4.md`](06_m4.md) | pendente; atualizar para a média das 10 |
| `ETTh1` | 309–383 | [`07_etth1.md`](07_etth1.md) | pendente; atualizar para a média das 10 |
| `ETTh2` | 384–458 | [`08_etth2.md`](08_etth2.md) | pendente; atualizar para a média das 10 |
| `ETTm1` | 459–533 | [`09_ettm1.md`](09_ettm1.md) | pendente; atualizar para a média das 10 |
| `ETTm2` | 534–608 | [`10_ettm2.md`](10_ettm2.md) | pendente; atualizar para a média das 10 |
| `Síntese entre conjuntos de dados` | 609–752 | pendente | — |
| `Análise do comportamento do agente` | 753–947 | pendente | — |
| `Estudo de ablação` | 948–1151 | pendente | decidir escopo |
| `Considerações finais` | 1152–fim | pendente | — |

## Fontes de dados

- Resultados do CARTE: `orchestrator_react_consist_seed11` a `orchestrator_react_consist_seed20` (dez execuções). Comparação entre modelos: `orchestrator_react_v12_gptoss_low`, `orchestrator_react_v12_qwen_medium` e `orchestrator_react_v12_gemma_on` (primeira execução de cada modelo).
- Baselines `mean`, `median`, `dba`, `ADE` e `FFORMA`: campo `baseline_results_json` dos CSV do CARTE.
- Modelos individuais: tabelas já existentes em `overleaf/resultados.tex`, que descrevem o mesmo pool de preditores.
- MCMs do CARTE: geradas por `outputs/resultados/scripts/mcm_carte.py`; em `overleaf/resultados_final` ficam apenas os PNGs usados no capítulo. PDFs e CSVs auxiliares ficam em `outputs/resultados/mcm_carte`.
- MCMs agregadas: `mcm_agregado_smape.png` e `mcm_agregado_pocid.png` usam as 680 séries; `mcm_media_datasets_smape.png` e `mcm_media_datasets_pocid.png` usam uma linha por conjunto de dados.

## Pendências de decisão

- Atualizar as tabelas e os textos por conjunto de dados para a média das dez execuções, substituindo os valores da execução canônica antiga.
- Regerar as MCMs com as métricas médias por série das dez execuções.
- Definir se o estudo de ablação será mantido ou substituído por uma análise das decisões finais do CARTE.
- Atualizar as referências a `sec:preparacao` e `CREST` que ainda existem em `resultados.tex`.
