# Escolha do modelo de linguagem do agente

Substitui a seção `Escolha do modelo de linguagem do agente` de `overleaf/resultados.tex` (linhas 15–75), mantendo a comparação entre `gpt-oss`, Qwen3.8 e Gemma4.

A execução com Qwen3.8 está completa nos sete conjuntos. A execução com Gemma4 está parcial: não há M4 e o NN5 tem 85 das 111 séries. Por isso, as comparações agregadas com Gemma usam o subconjunto comum de 295 séries (seis conjuntos, sem M4, com a interseção do NN5).

```latex
\section{Escolha do modelo de linguagem do agente} \label{sec:escolha-llm}

Antes de apresentar os resultados por conjunto de dados, é necessário justificar a escolha do modelo de linguagem empregado no papel de agente combinador ao longo do restante deste capítulo. Três modelos de linguagem de peso aberto foram comparados nesse papel, descritos na Seção~\ref{subsec:principais-modelos} do Capítulo~\ref{chp:background}, avaliando tanto o desempenho preditivo quanto o custo computacional de cada um. A execução com Gemma4 está incompleta: não há resultado para o M4 e o NN5 cobre 85 das 111 séries. As comparações que incluem Gemma4 são, portanto, restritas ao subconjunto comum disponível.

\subsection{Desempenho preditivo}

A Tabela~\ref{tab:escolha-llm-desempenho} compara o SMAPE e o POCID médios do CARTE executado com cada um dos três modelos, por conjunto de dados. A Tabela~\ref{tab:escolha-llm-agregado} resume as agregações: a média dos sete conjuntos, a média dos seis conjuntos comuns, o agregado das 680 séries e o agregado das 295 séries comuns aos três modelos.

\begin{table}[H]
\centering
\caption{Desempenho médio do CARTE com os três modelos de linguagem, por conjunto de dados.}
\label{tab:escolha-llm-desempenho}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{\textbf{SMAPE (\%)}} & \multicolumn{3}{c}{\textbf{POCID (\%)}} \\
\textbf{Conjunto de dados} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} \\
\midrule
ANP    & 21,93 & \textbf{21,88} & 22,36 & 63,44 & 64,04 & \textbf{64,09} \\
NN5    & 11,96 & 12,01 & \textbf{11,22}$^{*}$ & \textbf{47,36} & 45,43 & 44,87$^{*}$ \\
M4     & \textbf{9,16}  & 9,24 & -- & \textbf{57,73} & 57,40 & -- \\
ETTh1  & 22,36 & \textbf{22,05} & 22,36 & \textbf{62,73} & 62,11 & 62,11 \\
ETTh2  & \textbf{17,31} & 17,74 & 17,83 & \textbf{54,66} & 50,93 & 49,07 \\
ETTm1  & 43,03 & 39,35 & \textbf{38,05} & \textbf{55,90} & 54,66 & 51,55 \\
ETTm2  & \textbf{16,42} & 17,16 & 17,25 & 39,13 & 37,89 & \textbf{39,75} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Agregados de SMAPE e POCID para os três modelos.}
\label{tab:escolha-llm-agregado}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Agregação} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} \\
\midrule
\multicolumn{4}{l}{\textbf{SMAPE (\%)}} \\
Média dos 7 conjuntos & \textbf{20,31} & 19,92 & -- \\
Média dos 6 conjuntos comuns & 22,05 & 21,59 & \textbf{21,51} \\
Agregado das 680 séries & \textbf{13,67} & 13,69 & -- \\
Agregado das 295 séries comuns & 19,12 & \textbf{19,06} & 19,29 \\
\midrule
\multicolumn{4}{l}{\textbf{POCID (\%)}} \\
Média dos 7 conjuntos & \textbf{54,42} & 53,21 & -- \\
Média dos 6 conjuntos comuns & \textbf{53,79} & 52,75 & 51,91 \\
Agregado das 680 séries & \textbf{57,37} & 56,98 & -- \\
Agregado das 295 séries comuns & 57,69 & \textbf{57,90} & 57,27 \\
\bottomrule
\end{tabular}
\end{table}

$^{*}$ NN5 com 85 das 111 séries.

O Qwen3.8 melhora a média dos sete conjuntos em SMAPE: 19,92\% contra 20,31\% do gpt-oss, uma diferença de 0,39 ponto percentual. O ganho é concentrado no ETTm1, onde o Qwen reduz o SMAPE em 3,68 pontos percentuais, e no ETTh1, com 0,31 ponto. Nos outros quatro conjuntos, o gpt-oss é melhor: NN5 (+0,06), M4 (+0,09), ETTh2 (+0,43) e ETTm2 (+0,74). No agregado das 680 séries, os dois praticamente empatam: 13,67\% para o gpt-oss e 13,69\% para o Qwen3.8. Em POCID, o gpt-oss é superior na média dos sete (54,42\% contra 53,21\%) e no agregado das 680 séries (57,37\% contra 56,98\%); o Qwen só é melhor na ANP.

O Gemma4, na parte disponível, tem a melhor média dos seis conjuntos comuns em SMAPE (21,51\% contra 21,59\% do Qwen e 22,05\% do gpt-oss), mas essa vantagem não se sustenta no agregado das 295 séries comuns: 19,29\% contra 19,06\% do Qwen e 19,12\% do gpt-oss. Em POCID, o Gemma4 fica atrás dos dois na média dos seis (51,91\% contra 52,75\% e 53,79\%) e no agregado das 295 séries (57,27\% contra 57,90\% e 57,69\%). A ausência do M4 e a cobertura parcial do NN5 impedem uma comparação completa.

\subsection{Custo computacional}

A Tabela~\ref{tab:escolha-llm-custo} compara o tempo médio de decisão por série entre os três modelos, por conjunto de dados.

\begin{table}[H]
\centering
\caption{Tempo médio de decisão por série (segundos), por modelo e conjunto de dados.}
\label{tab:escolha-llm-custo}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Conjunto} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} \\
\midrule
ANP    & \textbf{25,52} & 561,92 & 451,06 \\
NN5    & \textbf{24,48} & 523,15 & 438,62$^{*}$ \\
M4     & \textbf{25,35} & 575,29 & -- \\
ETTh1  & \textbf{26,39} & 659,85 & 523,02 \\
ETTh2  & \textbf{25,70} & 644,46 & 532,94 \\
ETTm1  & \textbf{25,17} & 596,21 & 527,21 \\
ETTm2  & \textbf{19,55} & 620,50 & 589,32 \\
\midrule
Média  & \textbf{25,35} & 565,46 & 456,21$^{\dagger}$ \\
\bottomrule
\end{tabular}
\end{table}

$^{\dagger}$ Média sobre as 295 séries disponíveis para o Gemma4.

O Qwen3.8 é aproximadamente 22 vezes mais lento que o gpt-oss, e o Gemma4, cerca de 18 vezes. O custo adicional não é acompanhado por um ganho consistente nas duas métricas: o Qwen melhora a média dos sete em SMAPE, mas perde em POCID, no agregado das 680 séries e em quatro dos sete conjuntos; o Gemma4 tem média de SMAPE competitiva na parte disponível, mas perde em POCID e não completou a grade.

\subsection{Modelo escolhido}

O \texttt{gpt-oss:20b} é mantido como modelo de linguagem do agente nos resultados principais deste capítulo. Ele apresenta o melhor agregado das 680 séries em SMAPE e POCID, a melhor média dos sete em POCID e o menor custo computacional. O Qwen3.8 fica como alternativa completa para uma análise de sensibilidade: ele melhora a média dos sete em SMAPE, mas perde no agregado, no POCID e em custo. O Gemma4 não é elegível para a comparação principal enquanto não completar o M4 e o NN5.
```
