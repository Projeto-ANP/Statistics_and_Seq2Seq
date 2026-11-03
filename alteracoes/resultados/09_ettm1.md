# ETTm1

Substitui a seção `ETTm1` de `overleaf/resultados.tex` (linhas 459–533), com os resultados do CARTE e as análises atualizadas.

```latex
\section{ETTm1} \label{sec:resultados-ettm1}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:ettm1-individual} apresenta o desempenho médio, entre as 7 séries do ETTm1, de cada um dos dezenove preditores individuais, ordenados por SMAPE crescente.

\begin{table}[H]
\centering
\caption{Desempenho médio dos modelos individuais no ETTm1.}
\label{tab:ettm1-individual}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Modelo} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
Random Forest (DWT)                  & \textbf{40,52} & 52,17 \\
CatBoost (DWT)                       & 50,13 & 53,42 \\
CatBoost (FT)                        & 50,52 & 44,72 \\
CatBoost (CWT)                       & 50,72 & 47,83 \\
Random Forest (Original + DWT)       & 51,45 & \textbf{58,39} \\
Random Forest (CWT)                  & 52,98 & 54,66 \\
Random Forest (FT)                   & 52,99 & 43,48 \\
Random Forest (Original + FT)        & 53,06 & 55,90 \\
ETS                                  & 54,99 & 48,45 \\
Theta                                & 55,42 & 52,17 \\
CatBoost (Original + CWT)            & 60,43 & 54,66 \\
Random Forest (Original + CWT)       & 60,51 & 54,66 \\
CatBoost (Original + FT)             & 61,65 & 56,52 \\
CatBoost (Original + DWT)            & 61,84 & 54,04 \\
Random Forest (Original)             & 64,22 & 45,34 \\
Naive Moving Average                 & 65,11 & 41,61 \\
ARIMA                                & 77,76 & 52,80 \\
Naive Seasonal                       & 78,40 & 46,58 \\
CatBoost (Original)                  & 200,00 & 44,72 \\
\bottomrule
\end{tabular}
\end{table}

O ETTm1 se distingue dos demais ETT pela posição dos modelos estatísticos: ETS e Theta ficam entre os dez primeiros colocados por SMAPE (54,99\% e 55,42\%), à frente de seis dos catorze modelos de aprendizagem de máquina, algo que não ocorreu no ETTh1 nem no ETTh2. O Random Forest (DWT) obtém o menor SMAPE do conjunto, com uma margem considerável sobre o segundo colocado (40,52\% contra 50,13\%). O CatBoost sobre a representação original volta a atingir o valor máximo de SMAPE, 200,00\%.

\subsection{Desempenho entre estratégias de combinação}

\begin{table}[H]
\centering
\caption{Desempenho médio das estratégias de combinação no ETTm1.}
\label{tab:ettm1-estaticas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Combinação} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
CARTE   & \textbf{43,03} & 55,90 \\
FFORMA  & 43,96 & 51,55 \\
ADE     & 45,72 & 50,93 \\
Mediana & 54,43 & \textbf{58,39} \\
Média   & 200,00 & 44,72 \\
DBA     & 200,00 & 47,20 \\
\bottomrule
\end{tabular}
\end{table}

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:ettm1-estaticas}, o CARTE obtém o menor SMAPE (43,03\%), à frente do FFORMA (43,96\%), do ADE (45,72\%) e da mediana (54,43\%). A média e o DBA atingem o valor máximo possível de SMAPE (200,00\%). Em POCID, a mediana obtém o melhor resultado entre as seis (58,39\%), seguida do CARTE (55,90\%), do FFORMA (51,55\%), do ADE (50,93\%), do DBA (47,20\%) e da média (44,72\%). O CARTE é, portanto, a melhor combinação em SMAPE e a segunda melhor em POCID.

O resultado em SMAPE é particularmente forte: o CARTE fica atrás apenas do Random Forest (DWT), com 40,52\%, e supera os outros dezoito modelos individuais. A vantagem sobre o segundo melhor modelo individual, o CatBoost (DWT), é de 7,10 pontos percentuais. Em POCID, o CARTE fica atrás apenas da mediana entre as combinações e abaixo de dois modelos individuais, o Random Forest (Original + DWT) e o CatBoost (Original + FT).

\subsection{Comparação geral e discussão} \label{subsec:ettm1-discussao}

A Figura~\ref{fig:mcm-ettm1} apresenta a Matriz de Comparação Múltipla entre o Random Forest (DWT), o CARTE, o FFORMA, o ADE e a mediana, ordenados por SMAPE médio.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTM1.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTm1, ordenados por SMAPE médio.}
    \label{fig:mcm-ettm1}
\end{figure}

A MCM mostra que o Random Forest (DWT) tem o menor SMAPE médio (40,52\%), seguido do CARTE (43,03\%), do FFORMA (43,96\%), do ADE (45,72\%) e da mediana (54,43\%). O CARTE não se separa do Random Forest (DWT) ($p = 0{,}2969$), do FFORMA ($p = 0{,}9375$), do ADE ($p = 0{,}5781$) nem da mediana ($p = 0{,}9375$). A única comparação significativa da MCM é entre o Random Forest (DWT) e a mediana ($p = 0{,}0156$), com vantagem do modelo individual. Em POCID, o CARTE também não se separa da mediana ($p = 0{,}2256$), do Random Forest (DWT) ($p = 0{,}3508$), do FFORMA ($p = 0{,}2685$) nem do ADE ($p = 0{,}3750$).

O ETTm1 reforça dois pontos positivos do CARTE: a melhor combinação em SMAPE, com vantagem de 0,93 ponto percentual sobre o FFORMA, e a constância no top2 em POCID, atrás apenas da mediana. O CARTE fica a 2,51 pontos percentuais do melhor modelo individual em SMAPE, mas supera todos os outros dezoito preditores e todas as demais combinações. A mediana, que lidera o POCID, tem SMAPE 11,40 pontos percentuais pior, o que evidencia o trade-off entre as duas métricas e a posição equilibrada do CARTE.

A estratégia final escolhida pelo CARTE foi a mediana em 3 das 7 séries, uma combinação ponderada em 3 e a média aparada em 1. O pool completo foi usado em 3 séries, e o tamanho mediano do pool final foi de 7 modelos.
```
