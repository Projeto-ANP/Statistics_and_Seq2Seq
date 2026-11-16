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
- As execuções são tratadas na Seção~\ref{sec:execucoes}. Os textos por conjunto de dados usam a média das dez execuções e não repetem a explicação do protocolo. Aceite, estagnação, paciência, prior e meta-modelo entre séries continuam fora do texto.
- Não usar “catálogo de 12 ações”; quando necessário, referir as ações pelos nomes conceituais.
- Tabelas em percentual, com duas casas decimais e vírgula decimal.
- Comparações por série usam o SMAPE; o POCID é analisado em separado.
- Todas as tabelas de comparação — modelos individuais, estratégias de combinação e comparações pareadas — são ordenadas por **SMAPE crescente**. O POCID é coluna complementar e não altera a ordenação, exceto para desempatar valores iguais de SMAPE. A indicação textual desse critério aparece uma única vez, na primeira tabela da ANP; não repetir nas demais seções.
- A comparação entre modelos de linguagem usa **uma execução de cada modelo** e serve apenas para a escolha. Os resultados do CARTE usam a **média de dez execuções independentes**.
- A análise qualitativa do agente usa a **execução 1** da série de dez. Os traces e estudos de caso não representam a execução média.
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
| `Execuções e agregação` | nova | [`03_modelo_linguagem.md`](03_modelo_linguagem.md) (segunda seção) | pronto para integrar |
| `ANP` | 116–197 | [`04_anp.md`](04_anp.md) | atualizado para a média das 10; MCM regerada |
| `NN5` | 198–277 | [`05_nn5.md`](05_nn5.md) | atualizado para a média das 10; MCMs regeradas |
| `M4` | 234–308 | [`06_m4.md`](06_m4.md) | pronto para integrar; média das 10 |
| `ETTh1` | 309–383 | [`07_etth1.md`](07_etth1.md) | pronto para integrar; média das 10 |
| `ETTh2` | 384–458 | [`08_etth2.md`](08_etth2.md) | pronto para integrar; média das 10 |
| `ETTm1` | 459–533 | [`09_ettm1.md`](09_ettm1.md) | pronto para integrar; média das 10 |
| `ETTm2` | 534–608 | [`10_ettm2.md`](10_ettm2.md) | pronto para integrar; média das 10 |
| `Síntese entre conjuntos de dados` | 683–826 | [`11_sintese_transformacoes.md`](11_sintese_transformacoes.md), [`12_sintese_ranking.md`](12_sintese_ranking.md), [`13_sintese_mcm_agregada.md`](13_sintese_mcm_agregada.md), [`14_sintese_vitorias.md`](14_sintese_vitorias.md), [`15_sintese_horizonte.md`](15_sintese_horizonte.md) | todas as subseções prontas |
| `Análise do comportamento do agente` | 827–1021 | [`16_agente_frequencia.md`](16_agente_frequencia.md), [`17_agente_encerramento.md`](17_agente_encerramento.md), [`18_agente_contexto.md`](18_agente_contexto.md), [`19_agente_trajetoria.md`](19_agente_trajetoria.md), [`20_agente_interpretabilidade.md`](20_agente_interpretabilidade.md), [`21_agente_melhoria.md`](21_agente_melhoria.md), [`22_agente_custo.md`](22_agente_custo.md), [`23_agente_rastreabilidade.md`](23_agente_rastreabilidade.md) | todas as subseções prontas; `Estudo de caso ilustrativo` removido por sobreposição |
| `Estudo de ablação` | 1022–1225 | [`24_ablacao_buscas.md`](24_ablacao_buscas.md), [`25_ablacao_carte_oraculo.md`](25_ablacao_carte_oraculo.md), [`26_ablacao_sintese.md`](26_ablacao_sintese.md) | todas as subseções prontas |
| `Considerações finais` | 1226–fim | [`27_consideracoes_finais.md`](27_consideracoes_finais.md) | pronta |

O capítulo de conclusões foi reescrito separadamente em [`28_conclusoes.md`](28_conclusoes.md). O resumo revisado está em [`resumo.md`](../resumo.md).

## Fontes de dados

- Resultados do CARTE: `orchestrator_react_consist_seed11` a `orchestrator_react_consist_seed20` (dez execuções). Comparação entre modelos: `orchestrator_react_v12_gptoss_low`, `orchestrator_react_v12_qwen_medium` e `orchestrator_react_v12_gemma_on` (primeira execução de cada modelo).
- Baselines `mean`, `median`, `dba`, `ADE` e `FFORMA`: campo `baseline_results_json` dos CSV do CARTE.
- Modelos individuais: tabelas já existentes em `overleaf/resultados.tex`, que descrevem o mesmo pool de preditores.
- MCMs do CARTE: as de SMAPE são geradas por `outputs/resultados/scripts/mcm_carte.py`; as de POCID, por `outputs/resultados/scripts/mcm_carte_pocid_metric.py`, que escolhe o melhor individual e a melhor estática por POCID. Em `overleaf/resultados_final` ficam apenas os PNGs usados no capítulo. PDFs e CSVs auxiliares ficam em `outputs/resultados/mcm_carte`; os PNGs de POCID ficam em `outputs/resultados/mcm_carte_pocid_metric_png`.
- MCMs agregadas: `mcm_agregado_smape.png` e `mcm_agregado_pocid.png` usam as 680 séries; `mcm_media_datasets_smape.png` e `mcm_media_datasets_pocid.png` usam uma linha por conjunto de dados.
- Contrafactual das transformações: `outputs/resultados/scripts/sintese_transformacoes.py` gera as tabelas e a figura `overleaf/resultados_final/sintese_transformacoes_delta.png`; os CSVs intermediários ficam em `outputs/resultados/sintese_transformacoes/`.
- Ranking por conjunto de dados: consolidado a partir das tabelas de SMAPE dos arquivos `04_anp.md` a `10_ettm2.md`; a posição do CARTE usa a média das dez execuções.
- Erro por passo do horizonte: `outputs/resultados/scripts/sintese_horizonte.py` gera as figuras `overleaf/resultados_final/erro_horizonte_smape.png` e `erro_horizonte_pocid.png` com média e intervalo de confiança de 95%; o resumo fica em `outputs/resultados/sintese_horizonte/horizonte_resumo.csv`.
- Análise do comportamento do agente: a execução representativa é a pasta `orchestrator_react_consist_seed11`; a escolha usa a menor distância padronizada à média das dez no SMAPE e no POCID da média dos sete conjuntos. Os traces ficam em `orchestrator_react_consist_seed11/llm_artifacts/`.
- Interpretabilidade das previsões: `outputs/resultados/scripts/interpretabilidade_carte.py` gera `overleaf/resultados_final/caso_sucesso.png` e `caso_fracasso.png` a partir da execução representativa; os JSONs ficam em `outputs/resultados/interpretabilidade_carte/`.
- Rastreabilidade: a similaridade de Jaccard foi calculada sobre as 165 séries com escolha final a11/a12, usando palavras de conteúdo (stopwords removidas) do campo `justificativa_final`. A análise de n-gramas e de primeiras frases usa o mesmo campo.
- Ablação, buscas sem LLM: `outputs/resultados/scripts/ablacao_search_carte.py` roda aleatória com três repetições, gulosa e UCB com 12 avaliações por série, partindo de histórico vazio e aplicando a mesma regra final do CARTE. Os CSVs ficam em `timeseries/mestrado/resultados/orchestrator_search_carte/` e os artefatos com todas as tentativas em `outputs/resultados/ablacao_search_carte/artifacts/`.
- Ablação, oráculos: `ablacao_oraculos_carte.py` calcula melhor tentativa, melhores dez primeiras tentativas, teto de validação, oráculo de teste sobre as tentativas e melhor modelo individual, sempre com a média das dez execuções. `ablacao_oraculo_u_carte.py` enumera o universo fixo de estratégias para o teto de validação e o oráculo de teste amplo. A consolidação fica em `ablacao_oraculos_resumo.py`.

## Pendências de decisão

- Concluído: não há mais `CREST` visível em `overleaf/*.tex`; os rótulos internos `sec:crest`, `alg:crest` e `fig:crest_arquitetura` foram mantidos para não quebrar referências.
