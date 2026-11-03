# ETTh1

Substitui a seção `ETTh1` de `overleaf/resultados.tex` (linhas 309–383), com os resultados do CARTE e as análises atualizadas.

```latex
\section{ETTh1} \label{sec:resultados-etth1}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:etth1-individual} apresenta o desempenho médio, entre as 7 séries do ETTh1, de cada um dos dezenove preditores individuais, ordenados por SMAPE crescente.

\begin{table}[H]
\centering
\caption{Desempenho médio dos modelos individuais no ETTh1.}
\label{tab:etth1-individual}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Modelo} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
Random Forest (CWT)                  & \textbf{20,96} & \textbf{65,84} \\
Random Forest (Original + CWT)       & 22,44 & 65,22 \\
Random Forest (Original + DWT)       & 23,22 & 57,76 \\
Random Forest (DWT)                  & 23,59 & 59,63 \\
CatBoost (Original + DWT)            & 24,15 & 60,25 \\
CatBoost (Original + CWT)            & 24,41 & 60,87 \\
CatBoost (CWT)                       & 24,44 & 58,39 \\
CatBoost (Original + FT)             & 25,15 & 65,22 \\
Random Forest (Original + FT)        & 25,55 & 60,25 \\
CatBoost (DWT)                       & 27,89 & 59,01 \\
Random Forest (Original)             & 43,00 & 55,28 \\
CatBoost (FT)                        & 52,68 & 50,93 \\
Random Forest (FT)                   & 53,34 & 49,69 \\
ARIMA                                & 72,04 & 62,73 \\
Naive Moving Average                 & 87,02 & 48,45 \\
Naive Seasonal                       & 92,62 & 43,48 \\
Theta                                & 115,21 & 52,17 \\
ETS                                  & 119,50 & 50,93 \\
CatBoost (Original)                  & 200,00 & 52,17 \\
\bottomrule
\end{tabular}
\end{table}

No ETTh1, os dez modelos de aprendizagem de máquina com transformação de dados (\textit{wavelet} contínua, discreta e de Fourier, concatenadas ou isoladas) ocupam as dez primeiras posições por SMAPE, com valores entre 20,96\% e 27,89\%. O Random Forest (CWT) obtém o menor SMAPE e o maior POCID do conjunto simultaneamente. A partir da décima primeira posição, o desempenho piora de forma acentuada: o Random Forest e o CatBoost sobre a representação original, sem transformação, têm SMAPE de 43,00\% e 200,00\% respectivamente, este último o pior valor de toda a tabela, e os três modelos estatísticos (ARIMA, Theta e ETS) ficam entre os últimos colocados, com SMAPE acima de 70\%.

\subsection{Desempenho entre estratégias de combinação}

\begin{table}[H]
\centering
\caption{Desempenho médio das estratégias de combinação no ETTh1.}
\label{tab:etth1-estaticas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Combinação} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
CARTE   & \textbf{22,36} & 62,73 \\
Mediana & 26,02 & 62,11 \\
FFORMA  & 30,01 & \textbf{65,22} \\
ADE     & 39,51 & 62,73 \\
Média   & 200,00 & 52,17 \\
DBA     & 200,00 & 54,66 \\
\bottomrule
\end{tabular}
\end{table}

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:etth1-estaticas}, o CARTE obtém o menor SMAPE (22,36\%), à frente da mediana (26,02\%), do FFORMA (30,01\%) e do ADE (39,51\%). A média e o DBA atingem o valor máximo possível de SMAPE (200,00\%). Em POCID, o FFORMA obtém o maior valor entre as seis (65,22\%), seguido do CARTE e do ADE, empatados em 62,73\%, e da mediana (62,11\%); a média e o DBA ficam nas duas últimas posições. O CARTE é, portanto, a melhor combinação em SMAPE e a segunda melhor em POCID.

O resultado do CARTE no ETTh1 é particularmente forte quando comparado aos modelos individuais: seu SMAPE de 22,36\% fica atrás apenas do Random Forest (CWT), com 20,96\%, e supera o segundo melhor modelo individual, o Random Forest (Original + CWT), com 22,44\%. Ou seja, o CARTE supera dezoito dos dezenove preditores individuais em SMAPE e fica a 1,40 ponto percentual do melhor deles. Em POCID, o CARTE empata com o ADE e fica atrás apenas do Random Forest (CWT), com 65,84\%, e do FFORMA, com 65,22\%.

\subsection{Comparação geral e discussão} \label{subsec:etth1-discussao}

A Figura~\ref{fig:mcm-etth1} apresenta a Matriz de Comparação Múltipla entre o Random Forest (CWT), o CARTE, a mediana, o FFORMA e o ADE, ordenados por SMAPE médio.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTH1.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTh1, ordenados por SMAPE médio.}
    \label{fig:mcm-etth1}
\end{figure}

A MCM mostra que o Random Forest (CWT) tem o menor SMAPE médio (20,96\%), seguido do CARTE (22,36\%), da mediana (26,02\%), do FFORMA (30,01\%) e do ADE (39,51\%). O CARTE perde para o Random Forest (CWT) em SMAPE ($p = 0{,}0312$), mas supera a mediana ($p = 0{,}0220$), o FFORMA ($p = 0{,}0156$) e o ADE ($p = 0{,}0156$). Em POCID, o CARTE não se separa do FFORMA ($p = 0{,}4903$), da mediana ($p = 1{,}0000$) nem do ADE ($p = 1{,}0000$); perde para o Random Forest (CWT) ($p = 0{,}0496$).

O ETTh1 é o primeiro conjunto em que o CARTE é a melhor combinação por uma margem ampla: 3,66 pontos percentuais à frente da mediana e 7,65 pontos à frente do FFORMA em SMAPE. O CARTE também é o único método de combinação que supera o segundo melhor modelo individual do conjunto, o Random Forest (Original + CWT), e fica próximo do melhor individual. Em POCID, mantém a constância no top2 entre as combinações, dividindo o segundo lugar com o ADE e ficando atrás apenas do FFORMA. O conjunto reforça, portanto, dois pontos positivos do CARTE: a capacidade de selecionar uma combinação competitiva em um cenário com forte heterogeneidade entre modelos individuais e a constância no topo do ranking de POCID.

A estratégia final escolhida pelo CARTE foi a média em 3 das 7 séries, a mediana em 2 e uma combinação ponderada em 2. O pool completo foi usado em apenas 1 série, o que indica que o agente recorreu com frequência a subconjuntos menores para este conjunto.
```
