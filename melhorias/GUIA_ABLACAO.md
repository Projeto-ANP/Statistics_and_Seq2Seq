# Guia do estudo de ablação (`sec:ablacao-agente`)

Documento dedicado à seção `\section{Estudo de ablação: contribuição do agente na seleção das estratégias}` de `overleaf/resultados.tex`. Ele substitui os itens E62 a E65 do guia principal (`melhorias/GUIA_EDICOES_DISSERTACAO.md`), que ficam aposentados; a Parte 2.6 daquele guia agora aponta para cá. A numeração aqui é própria, de AB01 a AB07, no mesmo estilo do guia principal: cada item é uma edição pronta para colar, com o texto antigo exato a remover.

## Estrutura final da seção

Nenhuma subseção existente é removida. A seção mantém o texto de abertura e as duas subseções atuais, com os valores trocados pelos da configuração final, e ganha duas subseções novas ao final, antes de `\section{Considerações finais}`.

| Ordem | Subseção | Ação |
|---|---|---|
| 1 | Abertura da seção | Mantém como está, nada a mudar |
| 2 | Desempenho agregado | Mantém; AB01, AB02 e AB03 trocam tabela, parágrafo e p-valores |
| 3 | Quando o agente muda a decisão | Mantém; AB04 troca o parágrafo inteiro |
| 4 | Busca sem LLM: o agente contra buscas simples | NOVA; AB05 (colar ao final da seção) |
| 5 | Vencedor versus melhor semente no teste | NOVA; AB06 (pronta em 16/10) |
| 6 | `\section{Considerações finais}` | Continua no lugar; ajustes de números no guia principal (E65) e o parágrafo de fecho em AB07 |

Índice das edições:

| Código | Onde | O que muda | Status |
|---|---|---|---|
| AB01 | Tabela `tab:ablacao-desempenho` | Sete linhas com os valores finais | Pronto |
| AB02 | Parágrafo após a tabela | Reescrito (sem o destaque do ETTm2) | Pronto |
| AB03 | Parágrafo dos p-valores agregados | p do SMAPE de 0,1403 para 0,3080 | Pronto |
| AB04 | Subseção "Quando o agente muda a decisão" | Texto final (633 e 47 séries; ANP 5 e ANP 137) | Pronto |
| AB05 | Nova subseção ao final da seção | Busca sem LLM com números atuais | Pronto, com decisão pendente (ver abaixo) |
| AB06 | Nova subseção ao final da seção | Vencedor versus melhor semente | Pronta em 16/10 (replay final rodou) |
| AB07 | Considerações finais | Parágrafo de fecho (consequência de AB05) | Depende de AB05 |

---

## AB01. Tabela `tab:ablacao-desempenho`

REMOVER as sete linhas:

```
ANP    & 22,09 & 21,97 & 64,59 & 64,54 \\
NN5    & 11,77 & 11,57 & 43,76 & 46,33 \\
M4     & 9,23  & 9,14  & 59,05 & 59,40 \\
ETTh1  & 22,76 & 22,62 & 62,73 & 61,49 \\
ETTh2  & 17,94 & 17,99 & 55,28 & 55,28 \\
ETTm1  & 45,94 & 46,04 & 55,28 & 55,28 \\
ETTm2  & 16,10 & 20,30 & 45,34 & 45,34 \\
```

SUBSTITUIR POR:

```
ANP    & \textbf{21,97} & \textbf{21,97} & \textbf{64,79} & 64,54 \\
NN5    & \textbf{11,57} & \textbf{11,57} & 45,95 & \textbf{46,33} \\
M4     & 9,15  & \textbf{9,14}  & 59,03 & \textbf{59,40} \\
ETTh1  & 22,72 & \textbf{22,62} & \textbf{63,35} & 61,49 \\
ETTh2  & \textbf{17,99} & \textbf{17,99} & \textbf{55,28} & \textbf{55,28} \\
ETTm1  & \textbf{45,98} & 46,04 & \textbf{55,28} & \textbf{55,28} \\
ETTm2  & \textbf{20,30} & \textbf{20,30} & \textbf{45,34} & \textbf{45,34} \\
```

Convenção: o negrito marca o melhor valor de cada comparação na linha (menor SMAPE e maior POCID); em empates, os dois valores ficam em negrito.

## AB02. Parágrafo após a tabela

REMOVER o parágrafo:

```
Na maioria dos conjuntos de dados, a diferença entre as duas versões é pequena, em ambas as direções. O ETTm2 se destaca como o caso em que a atuação do agente traz o ganho mais expressivo: SMAPE de 16,10\% com o agente, contra 20,30\% sem ele, a maior diferença absoluta da tabela.
```

SUBSTITUIR POR:

```latex
As diferenças entre as duas versões são pequenas em todos os conjuntos de dados, em ambas as direções. Em SMAPE, quatro conjuntos são idênticos e nenhuma diferença passa de 0,10 ponto percentual. Em POCID, o CREST ganha 1,86 ponto percentual no ETTh1 e 0,25 na ANP, perde 0,38 no NN5 e 0,37 no M4, e empata nos três conjuntos restantes. Nenhum conjunto se destaca como caso isolado de ganho da atuação do agente, e a leitura agregada, apresentada a seguir, confirma que as duas versões são estatisticamente indistinguíveis.
```

## AB03. p-valores agregados

REMOVER:

```
$p = 0,1403$ e $p = 0,0785$
```

SUBSTITUIR POR:

```
$p = 0,3080$ e $p = 0,0738$
```

## AB04. Subseção "Quando o agente muda a decisão"

REMOVER o parágrafo inteiro da subseção:

```
Em 373 das 680 séries (54,9\%), a estratégia final é idêntica entre as duas versões, indicando que a fase determinística, por si só, já converge para a decisão correta em mais da metade dos casos. Nas 307 séries restantes (45,1\%), em que o agente de fato propõe algo diferente, o resultado na janela de teste é misto: melhor em 130 séries (42,3\%), pior em 158 (51,5\%), com 19 empates. Essa divisão, mais equilibrada do que desigual, é compatível com a limitação de apenas três janelas de validação disponíveis para fundamentar a decisão (Seção~\ref{sec:formulacao} do Capítulo~\ref{chp:proposta}): uma vantagem observada nas janelas de validação nem sempre se sustenta na janela de teste, nunca vista durante a decisão. Ainda assim, o ganho mais expressivo do experimento inteiro, tanto em direção positiva quanto negativa, veio de uma decisão do agente: a maior melhoria (ETTm2, índice 5, de 45,31\% para 22,51\% de SMAPE) supera em magnitude a maior piora (M4, índice 338, de 15,37\% para 29,91\%), reforçando que a contribuição do agente, quando se manifesta, pode ser substancial, ainda que não uniformemente positiva.
```

SUBSTITUIR POR:

```latex
Em 633 das 680 séries (93,1\%), a estratégia final é idêntica entre as duas versões, indicando que a fase determinística, por si só, já converge para a decisão correta na quase totalidade dos casos. Nas 47 séries restantes (6,9\%), em que o agente de fato propõe algo diferente, o resultado na janela de teste é misto: melhor em 20 séries e pior em 27. Essa divisão, praticamente equilibrada, é compatível com a limitação de apenas três janelas de validação disponíveis para fundamentar a decisão (Seção~\ref{sec:formulacao} do Capítulo~\ref{chp:proposta}): uma vantagem observada nas janelas de validação nem sempre se sustenta na janela de teste, nunca vista durante a decisão. O ganho mais expressivo do experimento inteiro, em ambas as direções, veio de uma decisão do agente: a maior melhoria ocorre na série 5 da ANP (de 20,77\% para 18,68\% de SMAPE, 2,09 pontos percentuais) e a maior piora, na série 137 da ANP (de 6,61\% para 9,74\%, 3,13 pontos percentuais), reforçando que a contribuição do agente, quando se manifesta, pode ser substancial, ainda que não uniformemente positiva.
```

## AB05. Subseção nova: "Busca sem LLM: o agente contra buscas simples"

ADICIONAR ao final da seção de ablação, imediatamente antes de `\section{Considerações finais}`.

DECISÃO PENDENTE: os números das duas tabelas abaixo vêm da execução atual dos braços de busca, cujo espaço de candidatos é um superconjunto do catálogo final do agente (sementes, busca aleatória, busca gulosa e UCB sobre o mesmo orçamento de avaliações). Há duas opções: descrever o protocolo como está (espaço comparável, mesma ordem de grandeza de orçamento) ou restringir o espaço ao catálogo final e reexecutar a busca (determinística, barata, comandos ao final desta seção). Enquanto a decisão não for tomada, este bloco fica pronto para colar com os números atuais.

SUBSTITUIR POR:

```latex
\subsection{Busca sem LLM: o agente contra buscas simples}

Para isolar o efeito do modelo de linguagem, esta subseção compara o CREST a três mecanismos de busca que não utilizam LLM, mas partem das mesmas dez sementes, usam o mesmo orçamento de avaliações por série e aplicam a mesma garantia de que a estratégia final é a de melhor pontuação do histórico: uma busca aleatória, uma busca gulosa e uma seleção por bandit (UCB). A única diferença entre os braços é o mecanismo que propõe novas estratégias.

A Tabela~\ref{tab:busca-nao-llm} apresenta o SMAPE médio de cada braço por conjunto de dados, do melhor braço escolhido pela pontuação de validação e do próprio CREST, e a Tabela~\ref{tab:busca-nao-llm-pocid} apresenta o POCID médio na mesma estrutura. A escolha do melhor braço é feita na validação, e os valores das tabelas são da janela de teste; a última linha de cada tabela traz o agregado das 680 séries, como média ponderada pelo número de séries de cada conjunto de dados.

\begin{table}[H]
\centering
\caption{SMAPE médio (\%) do CREST e de três buscas sem LLM sobre o mesmo orçamento, por conjunto de dados.}
\label{tab:busca-nao-llm}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Sementes} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} & \textbf{Melhor braço} & \textbf{CREST} \\
\midrule
ANP    & 21,97 & 22,26 & \textbf{21,93} & 22,19 & 22,17 & 21,97 \\
ETTh1  & 22,62 & 22,03 & 22,53 & 22,50 & \textbf{22,02} & 22,72 \\
ETTh2  & 17,99 & \textbf{17,53} & 19,26 & 18,85 & 19,19 & 17,99 \\
ETTm1  & 46,04 & 45,41 & \textbf{45,31} & 45,41 & 45,45 & 45,98 \\
ETTm2  & 20,30 & \textbf{19,44} & 20,29 & 20,46 & 20,79 & 20,30 \\
M4     & \textbf{9,14} & 9,45  & 9,32  & 9,40  & 9,56  & 9,15  \\
NN5    & \textbf{11,57} & 12,10 & 11,81 & 12,15 & 12,07 & \textbf{11,57} \\
\midrule
Agregado (680 séries) & \textbf{13,69} & 14,00 & 13,82 & 13,99 & 14,06 & 13,70 \\
\bottomrule

\end{tabular}
\end{table}

Convenção: o negrito marca o menor SMAPE de cada linha; em empates, os dois valores ficam em negrito.

\begin{table}[H]
\centering
\caption{POCID médio (\%) do CREST e de três buscas sem LLM sobre o mesmo orçamento, por conjunto de dados.}
\label{tab:busca-nao-llm-pocid}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Sementes} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} & \textbf{Melhor braço} & \textbf{CREST} \\
\midrule
ANP    & 64,54 & 64,39 & \textbf{64,79} & 63,94 & 64,74 & \textbf{64,79} \\
ETTh1  & 61,49 & 62,11 & 60,87 & 60,25 & 61,49 & \textbf{63,35} \\
ETTh2  & 55,28 & 53,42 & 54,66 & \textbf{55,90} & 53,42 & 55,28 \\
ETTm1  & 55,28 & 53,42 & 55,90 & \textbf{59,01} & 53,42 & 55,28 \\
ETTm2  & \textbf{45,34} & 44,72 & 44,10 & 42,24 & 44,10 & \textbf{45,34} \\
M4     & \textbf{59,40} & 55,59 & 58,61 & 56,22 & 56,13 & 59,03 \\
NN5    & \textbf{46,33} & 44,27 & 43,24 & 40,80 & 41,06 & 45,95 \\
\midrule
Agregado (680 séries) & \textbf{58,43} & 56,01 & 57,56 & 55,69 & 55,85 & 58,26 \\
\bottomrule
\end{tabular}
\end{table}

Convenção: o negrito marca o maior POCID de cada linha; em empates, os dois valores ficam em negrito.

Na linha de agregado, o melhor braço fica em 14,06\% de SMAPE, contra 13,69\% do piso determinístico e 13,70\% do CREST; em POCID, o piso determinístico mantém o maior valor agregado (58,43\%), à frente do CREST (58,26\%). Em nenhum conjunto de dados as diferenças entre os braços e o piso são estatisticamente significativas após correção de Holm no SMAPE, e no POCID a única diferença significativa é uma piora: a busca aleatória no M4 fica 3,81 pontos percentuais abaixo do piso ($p_{Holm} = 0,041$). A leitura conjunta, com agente e buscas simples indistinguíveis entre si e todas próximas do piso determinístico, indica que, neste protocolo, o ganho observável não decorre do mecanismo de busca, e sim da fase determinística de preparação, isto é, das sementes e do meta-modelo entre séries, de forma consistente com a ablação das subseções anteriores.
```

Se a decisão for reexecutar com o espaço restrito (comandos, no servidor):

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
for ds in ANP_MONTHLY NN5_WEEKLY_DATASET M4_WEEKLY_DATASET ETTH1 ETTH2 ETTM1 ETTM2; do
    python3 aprovacao_solida/01_busca_nao_llm/run_search.py --dataset "$ds" --version restrito --methods seeds,random,greedy,ucb
done
python3 aprovacao_solida/01_busca_nao_llm/analyze_search.py --results-dir outputs/resultados/aprovacao_solida > /dev/null
```

Depois de rodar, regenerar a Tabela~\ref{tab:busca-nao-llm} a partir do CSV novo (a análise grava em `outputs/resultados/aprovacao_solida/analise_busca.csv`; os valores por braço e por conjunto estão nas colunas `mean_method` e `p_holm`).

## AB06. Subseção nova: "Vencedor versus melhor semente no teste"

PRONTA (16/10): o replay rodou sobre a rodada final (`--run orchestrator_react_v2_medium_x1_gpt-oss-20b`) e os números abaixo vêm de `outputs/resultados/aprovacao_solida/replay_gate_orchestrator_react_v2_medium_x1_gpt-oss-20b{,_por_dataset}.csv`. ADICIONAR ao final da seção de ablação, depois do bloco do AB05, imediatamente antes de `\section{Considerações finais}`.

SUBSTITUIR POR:

```latex
\subsection{Vencedor versus melhor semente no teste}

A garantia de que a estratégia final nunca é pior que a melhor semente vale para a validação, mas não necessariamente para a janela de teste. Para quantificar essa diferença, as sementes foram reconstruídas com o mesmo procedimento determinístico de preparação e comparadas, na janela de teste, com o vencedor efetivamente aplicado. A Tabela~\ref{tab:replay-vencedor-semente} apresenta o SMAPE médio das duas versões por conjunto de dados.

\begin{table}[H]
\centering
\caption{SMAPE médio (\%) do vencedor aplicado e da melhor semente reconstruída, por conjunto de dados.}
\label{tab:replay-vencedor-semente}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Vencedor} & \textbf{Melhor semente} \\
\midrule
ANP    & \textbf{21,97} & \textbf{21,97} \\
ETTh1  & 22,72 & \textbf{22,62} \\
ETTh2  & \textbf{17,99} & \textbf{17,99} \\
ETTm1  & \textbf{45,98} & 46,04 \\
ETTm2  & \textbf{20,30} & \textbf{20,30} \\
M4     & 9,15  & \textbf{9,14} \\
NN5    & \textbf{11,57} & \textbf{11,57} \\
\midrule
Agregado (680 séries) & 13,70 & \textbf{13,69} \\
\bottomrule
\end{tabular}
\end{table}

Convenção: o negrito marca o menor SMAPE de cada linha; em empates, os dois valores ficam em negrito.

Das 680 séries, em 633 as duas versões produzem previsões numericamente idênticas; nas 47 restantes, o vencedor obtém SMAPE menor em 20 e a semente em 27, sem empates. No agregado, a semente fica ligeiramente à frente do vencedor (13,69\% contra 13,70\% em SMAPE, 58,43\% contra 58,26\% em POCID).

O padrão de sobreajuste se manifesta de forma moderada: nas 27 séries em que o vencedor melhora a pontuação de validação em mais de 5\%, o resultado na janela de teste é, em média, 0,17 ponto percentual pior que o da semente, indício do \textit{winner's curse} descrito na literatura. Regras que só trocam a semente pelo vencedor quando o ganho de validação supera uma margem fixa, entre 0,5\% e 5\%, alteram a decisão em apenas 27 a 37 das 680 séries e obtêm 13,70\% de SMAPE médio nas quatro margens, sem superar a política de manter sempre a melhor semente (13,69\%), que também lidera no POCID (58,43\%), à frente de todas as regras avaliadas.
```

Se precisar reexecutar (determinístico, sem LLM; cerca de 1 a 3 horas):

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
nohup python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \
    --run orchestrator_react_v2_medium_x1_gpt-oss-20b > logs/replay_gate_final.log 2>&1 &
```

## AB07. Parágrafo de fecho em Considerações finais (depende de AB05)

ADICIONAR DEPOIS DO PARÁGRAFO da seção "Considerações finais" que termina com:

```
limitações identificadas que devem orientar os próximos passos deste trabalho.
```

Bloco a inserir:

```latex
Por fim, a ablação complementada pela comparação com buscas sem LLM mostra que a contribuição do agente não é estatisticamente distinguível da de mecanismos de busca simples no mesmo orçamento de avaliações. O desempenho do CREST decorre principalmente da preparação determinística entre séries, e o agente atua como um mecanismo de exploração e de justificativa das decisões, não como fonte de ganho de acurácia mensurável.
```

---

## Imagens da seção

As duas figuras da subseção "Desempenho agregado" usam arquivos já atualizados com a configuração final e presentes em `overleaf/resultados_final/`:

| Figura | Arquivo |
|---|---|
| `fig:mcm-ablacao-smape` | `images/resultados_final/mcm_agente_vs_sem_agente_smape.png` |
| `fig:mcm-ablacao-pocid` | `images/resultados_final/mcm_agente_vs_sem_agente_pocid.png` |

A pasta `overleaf/resultados_final/` agora contém todas as imagens de resultados do capítulo: os dois casos, as quatro de horizonte, a contagem de vitórias, as sete MCM por conjunto, as duas MCM agregadas e as duas MCM da ablação. Nenhuma outra figura é necessária para esta seção.

## Fontes dos números

- `outputs/resultados/ablacao/desempenho_sem_agente_por_dataset.csv`: CREST sem agente por conjunto (AB01 e AB02).
- `outputs/resultados/ablacao/mcm_agente_vs_sem_agente_*.csv`: p-valores agregados (AB03).
- `outputs/resultados/ablacao/pareado_por_serie.csv` (regenerado) e o pareamento por série da configuração final: contagens de 633 e 47 e as séries 5 e 137 da ANP (AB04).
- `outputs/resultados/aprovacao_solida/analise_busca.csv`: busca sem LLM (AB05). O arquivo guarda colunas `mean_method` e `p_holm` por conjunto, método e métrica; a coluna de CREST desse arquivo é da rodada antiga e não deve ser usada (na tabela, o CREST vem dos resultados por conjunto da configuração final).
- `outputs/resultados/aprovacao_solida/replay_gate_orchestrator_react_v5*.csv`: apenas referência da rodada antiga; aguardando reexecução (AB06).
