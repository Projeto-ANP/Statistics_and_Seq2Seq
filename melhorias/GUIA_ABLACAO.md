# Guia do estudo de ablação (`sec:ablacao-agente`)

Documento dedicado à seção `\section{Estudo de ablação: contribuição do agente na seleção das estratégias}` de `overleaf/resultados.tex`. Ele substitui os itens E62 a E65 do guia principal (`melhorias/GUIA_EDICOES_DISSERTACAO.md`), que ficam aposentados; a Parte 2.6 daquele guia agora aponta para cá. A numeração aqui é própria, de AB01 a AB08, no mesmo estilo do guia principal: cada item é uma edição pronta para colar, com o texto antigo exato a remover.

## Estrutura final da seção

Nenhuma subseção existente é removida. A seção mantém o texto de abertura e as duas subseções atuais, com os valores trocados pelos da configuração final, e ganha três subseções novas ao final, antes de `\section{Considerações finais}`.

| Ordem | Subseção | Ação |
|---|---|---|
| 1 | Abertura da seção | Mantém como está, nada a mudar |
| 2 | Desempenho agregado | Mantém; AB01, AB02 e AB03 trocam tabela, parágrafo e p-valores |
| 3 | Quando o agente muda a decisão | Mantém; AB04 troca o parágrafo inteiro |
| 4 | Busca sem LLM: o agente contra buscas simples | NOVA; AB05 (pronta em 18/10) |
| 5 | Vencedor versus melhor semente no teste | NOVA; AB06 (pronta em 16/10) |
| 6 | O teto do espaço de busca: oráculo de validação e oráculo de teste | NOVA; AB08 (pronta em 17/10) |
| 7 | `\section{Considerações finais}` | Continua no lugar; ajustes de números no guia principal (E65) e o parágrafo de fecho em AB07 |

Índice das edições:

| Código | Onde | O que muda | Status |
|---|---|---|---|
| AB01 | Tabela `tab:ablacao-desempenho` | Sete linhas com os valores finais | Pronto |
| AB02 | Parágrafo após a tabela | Reescrito (sem o destaque do ETTm2) | Pronto |
| AB03 | Parágrafo dos p-valores agregados | p do SMAPE de 0,1403 para 0,3080 | Pronto |
| AB04 | Subseção "Quando o agente muda a decisão" | Texto final (633 e 47 séries; ANP 5 e ANP 137) | Pronto |
| AB05 | Nova subseção ao final da seção | Busca sem LLM com o run restrito e tempos | Pronta em 18/10 |
| AB06 | Nova subseção ao final da seção | Vencedor versus melhor semente | Pronta em 16/10 (replay final rodou) |
| AB07 | Considerações finais | Parágrafo de fecho (consequência de AB05) | Depende de AB05 |
| AB08 | Nova subseção ao final da seção | Oráculo de validação e oráculo de teste | Pronta em 17/10 |

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

PRONTA (18/10): o run restrito `catalogo_final` terminou. As tabelas de acurácia usam os braços de `orchestrator_search_catalogo_final/` e o CREST da configuração final (Tabela~\ref{tab:ablacao-desempenho}); a coluna CREST do `analise_busca.csv` é da rodada antiga (v5) e não deve ser usada. Os tempos dos braços vêm de `logs/catalogo_final_busca.log`; o do CREST, de `item_17/custo_computacional.csv`.

ADICIONAR ao final da seção de ablação, depois do AB04 e antes do AB06.

SUBSTITUIR POR:

```latex
\subsection{Busca sem LLM: o agente contra buscas simples}

Para isolar o efeito do modelo de linguagem, esta subseção compara o CREST a três buscas sem LLM que partem das mesmas dez sementes, usam o mesmo orçamento de avaliações por série e aplicam a mesma regra final, a melhor pontuação do histórico: a busca aleatória, a busca gulosa e a seleção por bandit (UCB). Os braços usam as mesmas ferramentas e receitas do agente, com uma grade fixa de parâmetros; o agente pode escolher fora dessa grade, então o espaço dos braços é um subconjunto do espaço do agente. Entre os braços, muda apenas o mecanismo que propõe novas estratégias.

A Tabela~\ref{tab:busca-nao-llm} mostra o SMAPE médio por conjunto de dados para as sementes, para cada braço, para o melhor braço escolhido na validação e para o CREST; a Tabela~\ref{tab:busca-nao-llm-pocid} repete a estrutura para o POCID. O melhor braço é escolhido na validação e as tabelas reportam o teste; a última linha traz o agregado das 680 séries, ponderado pelo número de séries de cada conjunto.

\begin{table}[H]
\centering
\caption{SMAPE médio (\%) do CREST e de três buscas sem LLM sobre o mesmo orçamento, por conjunto de dados.}
\label{tab:busca-nao-llm}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Sementes} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} & \textbf{Melhor braço} & \textbf{CREST} \\
\midrule
ANP    & \textbf{21,97} & 22,24 & 21,99 & 22,20 & 22,18 & \textbf{21,97} \\
ETTh1  & 22,62 & 24,70 & 24,70 & 22,50 & \textbf{22,48} & 22,72 \\
ETTh2  & 17,99 & \textbf{17,64} & 17,70 & 18,85 & 18,82 & 17,99 \\
ETTm1  & 46,04 & 46,37 & \textbf{45,30} & 45,41 & 45,98 & 45,98 \\
ETTm2  & 20,30 & \textbf{19,49} & 20,22 & 20,46 & 20,84 & 20,30 \\
M4     & \textbf{9,14} & 9,45 & 9,38 & 9,37 & 9,56 & 9,15 \\
NN5    & \textbf{11,57} & 12,19 & 11,90 & 12,06 & 12,20 & \textbf{11,57} \\
\midrule
Agregado (680 séries) & \textbf{13,69} & 14,05 & 13,89 & 13,96 & 14,09 & 13,70 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{POCID médio (\%) do CREST e de três buscas sem LLM sobre o mesmo orçamento, por conjunto de dados.}
\label{tab:busca-nao-llm-pocid}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Sementes} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} & \textbf{Melhor braço} & \textbf{CREST} \\
\midrule
ANP    & 64,54 & \textbf{64,84} & 64,79 & 63,79 & 64,54 & 64,79 \\
ETTh1  & 61,49 & 62,11 & 61,49 & 60,25 & 60,87 & \textbf{63,35} \\
ETTh2  & 55,28 & 54,04 & 55,28 & \textbf{55,90} & 54,66 & 55,28 \\
ETTm1  & 55,28 & 54,04 & 57,14 & \textbf{59,01} & 53,42 & 55,28 \\
ETTm2  & \textbf{45,34} & \textbf{45,34} & 44,10 & 42,24 & 44,72 & \textbf{45,34} \\
M4     & \textbf{59,40} & 55,87 & 58,75 & 56,27 & 56,59 & 59,03 \\
NN5    & \textbf{46,33} & 43,89 & 43,24 & 40,80 & 42,21 & 45,95 \\
\midrule
Agregado (680 séries) & \textbf{58,43} & 56,23 & 57,66 & 55,68 & 56,24 & 58,26 \\
\bottomrule
\end{tabular}
\end{table}

No agregado, o melhor braço fica em 14,09\% de SMAPE, contra 13,69\% da semente e 13,70\% do CREST. Em POCID, a semente lidera (58,43\%), à frente do CREST (58,26\%) e do melhor braço (56,24\%). Nenhuma diferença entre os braços e a semente é significativa após correção para as comparações múltiplas, nem no SMAPE nem no POCID; os p-valores brutos abaixo de 0,05 são todos pioras, no NN5 e no M4, e nenhum sobrevive à correção. O agente fica próximo do piso determinístico e as buscas simples não trazem vantagem consistente: o ganho observável não vem do mecanismo de busca, mas da fase determinística de preparação, isto é, das sementes e do meta-modelo entre séries, como na ablação anterior.

\begin{table}[H]
\centering
\caption{Tempo médio por série (s) do CREST e dos braços de busca, por conjunto de dados.}
\label{tab:busca-tempo}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{CREST} & \textbf{Sementes} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} \\
\midrule
ANP    & 10,32 & 0,24 & 0,33 & 0,82 & 0,24 \\
ETTh1  & 14,43 & 0,06 & 0,10 & 0,10 & 0,10 \\
ETTh2  & 14,34 & 0,04 & 0,10 & 0,10 & 0,10 \\
ETTm1  & 6,19  & 0,00 & 0,10 & 0,10 & 0,10 \\
ETTm2  & 7,69  & 0,03 & 0,10 & 0,11 & 0,10 \\
M4     & 8,26  & 0,92 & 1,79 & 3,30 & 1,03 \\
NN5    & 10,04 & 0,10 & 0,20 & 0,27 & 0,13 \\
\midrule
Agregado (680 séries) & 9,20 & 0,57 & 1,07 & 2,01 & 0,63 \\
\bottomrule
\end{tabular}
\end{table}

Os braços custam de 0,6 a 2,0 s por série, contra 9,2 s do laço do CREST; a diferença é o custo das chamadas ao LLM. A soma dos quatro braços nas 680 séries é 0,81 h, contra 1,74 h do laço do CREST. Os braços são mais baratos, mas não melhores em acurácia: o custo do agente não se traduz em ganho.
```

------------------------_

Reprodução: o run restrito foi gerado com `aprovacao_solida/01_busca_nao_llm/run_catalogo_final.py` (version `catalogo_final`) no servidor; o log completo está em `logs/catalogo_final_busca.log`; a análise em `outputs/resultados/aprovacao_solida/analise_busca.csv`. Os tempos dos braços são as linhas `(Xs)` do log; o do CREST é o `elapsed_s` do laço em `item_17`.

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

## AB08. Subseção nova: "O teto do espaço de busca: oráculo de validação e oráculo de teste"

ADICIONAR ao final da seção de ablação, depois do bloco do AB06, imediatamente antes de `\section{Considerações finais}`.

ANTES: o último parágrafo do AB06, que termina com "...que também lidera no POCID (58,43\%), à frente de todas as regras avaliadas."

DEPOIS: `\section{Considerações finais}` (o texto que começa com "Os resultados apresentados neste capítulo..."); o parágrafo de fecho do AB07 entra depois do parágrafo que termina com "...limitações identificadas que devem orientar os próximos passos deste trabalho."

SUBSTITUIR POR:

```latex
\subsection{O teto do espaço de busca: oráculo de validação e oráculo de teste}
\label{subsec:oraculo-condicional}

A subseção anterior comparou o vencedor aplicado com a melhor semente. Esta subseção faz uma pergunta distinta: qual seria a melhor estratégia de validação que o agente teria à disposição e quanto ela valeria no teste? Para isso, define o universo $U$ com o pool completo, os pools referenciados nas tentativas do agente, os pools criados por suas ações e o pool da melhor semente. Sobre esses pools, a enumeração cobre média, mediana, média aparada com \textit{trim} de 0 a 0,4 em passos de 0,1, DBA, melhor modelo individual para cada um dos dezenove modelos e média ponderada com as receitas \textit{inverse} (shrinkage de 0 a 0,5 em passos de 0,1), \textit{error trend}, \textit{best per horizon}, \textit{top-k per horizon} e \textit{forward selection}; em média, $U$ contém 56 estratégias por série. A receita \textit{pooled}/meta, que depende do pré-passo entre séries, e as receitas \textit{softmax}, \textit{feature-based} e OLS ficaram fora da enumeração; a \textit{pooled}/meta aparece em quatro estratégias finais e não altera a comparação entre agente e semente.

Dois oráculos são calculados sobre $U$. O oráculo de validação escolhe a estratégia de menor score de validação, usando o mesmo score composto do agente, e só depois a aplica à janela de teste; ele representa a melhor decisão possível com a informação disponível no momento da escolha. O oráculo de teste escolhe, por série, a estratégia de menor SMAPE e a de maior POCID na própria janela de teste; ele é retrospectivo (um teto): usa informação indisponível na decisão e, por isso, não é um método implantável.

A Tabela~\ref{tab:oraculo-condicional} apresenta os dois oráculos. Os valores do agente e da melhor semente já foram apresentados na Tabela~\ref{tab:ablacao-desempenho} e na subseção anterior; no agregado, eles ficam em 13,70\% e 13,69\% de SMAPE, e 58,26\% e 58,43\% de POCID.

\begin{table}[H]
\centering
\small
\caption{SMAPE e POCID médios (\%) no teste do oráculo de validação e do oráculo de teste, por conjunto de dados.}
\label{tab:oraculo-condicional}
\begin{tabular}{@{}lrr@{}}
\toprule
\multicolumn{3}{c}{\textbf{SMAPE (\%)}} \\
\textbf{Conjunto de dados} & \textbf{Oráculo de validação} & \textbf{Oráculo de teste} \\
\midrule
ANP    & 21,91 & 17,24 \\
NN5    & 12,38 & 9,17  \\
M4     & 10,20 & 5,25  \\
ETTh1  & 25,60 & 19,61 \\
ETTh2  & 18,09 & 15,54 \\
ETTm1  & 48,15 & 35,82 \\
ETTm2  & 23,01 & 13,68 \\
\midrule
Agregado (680 séries) & 14,45 & 9,75 \\
\midrule
\multicolumn{3}{c}{\textbf{POCID (\%)}} \\
\textbf{Conjunto de dados} & \textbf{Oráculo de validação} & \textbf{Oráculo de teste} \\
\midrule
ANP    & 62,84 & 82,02 \\
NN5    & 44,14 & 78,76 \\
M4     & 57,85 & 80,57 \\
ETTh1  & 62,73 & 73,91 \\
ETTh2  & 54,66 & 67,70 \\
ETTm1  & 60,25 & 71,43 \\
ETTm2  & 45,34 & 55,28 \\
\midrule
Agregado (680 séries) & 56,86 & 80,11 \\
\bottomrule
\end{tabular}
\end{table}

Na validação, o oráculo encontra uma estratégia com score menor que o do agente em 93,8\% das séries, com ganho médio de 0,061. No teste, porém, essa vantagem não se traduz em melhor desempenho: o oráculo de validação supera o agente em apenas 39,1\% das séries pelo SMAPE e 29,3\% pelo POCID, e fica, em média, 0,75 ponto percentual pior no SMAPE e 1,4 ponto percentual pior no POCID. No agregado, 14,45\% e 56,86\%, contra 13,70\% e 58,26\% do agente e 13,69\% e 58,43\% da semente. O oráculo de teste fica em 9,75\% e 80,11\%.

Esse é o \textit{winner's curse} em sua forma mais clara. Com apenas três janelas de validação, o máximo de um conjunto que tem, em média, 56 candidatos é uma estatística extrema: parte do ganho de validação é ruído específico dessas janelas e não se repete no teste. Quanto maior o conjunto em que se escolhe o máximo, maior a parcela de ruído capturada. O agente, que avalia um conjunto menor (as dez sementes e no máximo doze propostas), é na prática mais regularizado que o oráculo; sua escolha não é a melhor da validação em $U$, mas generaliza melhor que a do oráculo, embora ainda não supere a semente. O oráculo de teste mostra que existe espaço para melhorar (3,95 pontos percentuais no SMAPE e 21,85 no POCID), mas também que esse espaço não é alcançável por seleção na validação com o protocolo atual.

A leitura conjunta com o replay anterior é coerente com a ablação: o agente não encontra o ótimo global, e uma busca mais ampla no mesmo sinal de validação não melhora o teste. O que o agente oferece não é a escolha do melhor entre todas as estratégias possíveis, e sim uma escolha conservadora, auditável e limitada ao histórico que ele próprio construiu.
```

------------------------_

Fontes: `outputs/resultados/scripts/oraculo_condicional.py` e `outputs/resultados/oraculo_condicional/por_serie.csv`.

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
- `outputs/resultados/aprovacao_solida/analise_busca.csv`: busca sem LLM (AB05). Guarda `mean_method` e `p_holm` por conjunto, método e métrica; a coluna CREST é da rodada antiga e não deve ser usada (o CREST vem da configuração final, Tabela AB01).
- `timeseries/mestrado/resultados/orchestrator_search_catalogo_final/`: CSVs do run restrito dos braços (AB05).
- `logs/catalogo_final_busca.log`: tempos por série dos braços (AB05); `outputs/resultados/item_17/custo_computacional.csv`: `elapsed_s` do CREST (AB05).
- `outputs/resultados/scripts/oraculo_condicional.py` e `outputs/resultados/oraculo_condicional/por_serie.csv`: oráculo de validação e oráculo de teste (AB08).
- `outputs/resultados/aprovacao_solida/replay_gate_orchestrator_react_v2_medium_x1_gpt-oss-20b*.csv`: replay final (AB06).
