# ETTh1

Atualiza a seção `ETTh1` de `overleaf/resultados.tex` (linhas 361–442) com a média das dez execuções do CARTE. A MCM de SMAPE usa os comparadores escolhidos por SMAPE; a de POCID usa os comparadores escolhidos por POCID.

```latex
\section{ETTh1} \label{sec:resultados-etth1}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:etth1-individual} apresenta o desempenho médio, entre as 7 séries do ETTh1, de cada um dos dezenove preditores individuais.

\begin{table}[H]
\small
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

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:etth1-estaticas}, o CARTE obtém o menor SMAPE (22,72\%), à frente da mediana (26,02\%), do FFORMA (30,01\%) e do ADE (39,51\%). A média e o DBA atingem o valor máximo possível de SMAPE (200,00\%). Em POCID, o FFORMA obtém o maior valor entre as seis (65,22\%), seguido do ADE (62,73\%), do CARTE (62,11\%) e da mediana (62,11\%); a média e o DBA ficam nas duas últimas posições. O CARTE é a melhor combinação em SMAPE e a terceira em POCID, empatada com a mediana.

\begin{table}[H]
\small
\centering
\caption{Desempenho médio das estratégias de combinação no ETTh1.}
\label{tab:etth1-estaticas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Combinação} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
CARTE   & \textbf{22,72} & 62,11 \\
Mediana & 26,02 & 62,11 \\
FFORMA  & 30,01 & \textbf{65,22} \\
ADE     & 39,51 & 62,73 \\
Média   & 200,00 & 52,17 \\
DBA     & 200,00 & 54,66 \\
\bottomrule
\end{tabular}
\end{table}

O resultado do CARTE no ETTh1 é particularmente forte em SMAPE: 22,72\% ficam atrás apenas do Random Forest (CWT), com 20,96\%, e do Random Forest (Original + CWT), com 22,44\%, e à frente do terceiro melhor individual, o Random Forest (Original + DWT), com 23,22\%. O CARTE supera dezessete dos dezenove preditores individuais em SMAPE e fica a 1,76 ponto percentual do melhor deles. Em POCID, o CARTE tem 62,11\% e fica atrás do Random Forest (CWT) (65,84\%), do Random Forest (Original + CWT) (65,22\%), do CatBoost (Original + FT) (65,22\%), do FFORMA (65,22\%), do ADE (62,73\%) e do ARIMA (62,73\%); empata com a mediana em 62,11\%. Não é top2 em POCID no conjunto completo.

\subsection{Comparação geral e discussão} \label{subsec:etth1-discussao}

A Figura~\ref{fig:mcm-etth1} apresenta a Matriz de Comparação Múltipla entre o Random Forest (CWT), o CARTE, a mediana, o FFORMA e o ADE. A MCM de SMAPE usa o melhor individual e a melhor estática por SMAPE.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTH1.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTh1.}
    \label{fig:mcm-etth1}
\end{figure}

A MCM de SMAPE mostra que o Random Forest (CWT) tem o menor SMAPE médio (20,96\%), seguido do CARTE (22,72\%), da mediana (26,02\%), do FFORMA (30,01\%) e do ADE (39,51\%). O CARTE perde para o Random Forest (CWT) em SMAPE ($p = 0{,}0312$), mas supera a mediana ($p = 0{,}0156$), o FFORMA ($p = 0{,}0156$) e o ADE ($p = 0{,}0156$).

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTH1_pocid.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTh1, para o POCID.}
    \label{fig:mcm-etth1-pocid}
\end{figure}

A MCM de POCID usa o Random Forest (CWT) como melhor individual e a mediana como melhor estática. O Random Forest (CWT) lidera (65,84\%), seguido do FFORMA (65,22\%), do ADE (62,73\%), do CARTE (62,11\%) e da mediana (62,11\%). O CARTE não se separa do Random Forest (CWT) ($p = 0{,}0781$), do FFORMA ($p = 0{,}4688$), do ADE ($p = 0{,}9375$) nem da mediana ($p = 1{,}0000$).

O ETTh1 é o conjunto em que o CARTE é a melhor combinação em SMAPE por uma margem ampla: 3,30 pontos percentuais à frente da mediana e 7,29 pontos à frente do FFORMA. É a única combinação que supera o terceiro melhor modelo individual, o Random Forest (Original + DWT), com 23,22\%, mas fica atrás dos dois melhores. Em POCID, o CARTE fica no grupo intermediário, empatado com a mediana e atrás do FFORMA e do ADE, sem separação estatística de nenhum deles. O ETTh1 reforça o ponto forte do CARTE em SMAPE, mas não sustenta a constância top2 em POCID.

Ao longo das dez execuções, a estratégia final foi uma combinação ponderada em cerca de 3,2 das 7 séries, em média; a mediana em cerca de 2,1, a média em cerca de 1,1 e a média aparada em cerca de 0,6. O pool completo foi usado em cerca de uma série, em média, o que indica que o agente recorreu com frequência a subconjuntos menores para este conjunto.
```
