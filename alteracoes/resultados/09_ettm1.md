# ETTm1

Atualiza a seção `ETTm1` de `overleaf/resultados.tex` (linhas 522–600) com a média das dez execuções do CARTE. A MCM de SMAPE usa os comparadores escolhidos por SMAPE; a de POCID usa os comparadores escolhidos por POCID.

```latex
\section{ETTm1} \label{sec:resultados-ettm1}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:ettm1-individual} apresenta o desempenho médio, entre as 7 séries do ETTm1, de cada um dos dezenove preditores individuais.

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

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:ettm1-estaticas}, o CARTE obtém o menor SMAPE (41,03\%), à frente do FFORMA (43,96\%), do ADE (45,72\%) e da mediana (54,43\%). A média e o DBA atingem o valor máximo possível de SMAPE (200,00\%). Em POCID, a mediana obtém o melhor resultado entre as seis (58,39\%), seguida do CARTE (54,47\%), do FFORMA (51,55\%), do ADE (50,93\%), do DBA (47,20\%) e da média (44,72\%). O CARTE é, portanto, a melhor combinação em SMAPE e a segunda melhor em POCID.

\begin{table}[H]
\centering
\caption{Desempenho médio das estratégias de combinação no ETTm1.}
\label{tab:ettm1-estaticas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Combinação} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
CARTE   & \textbf{41,03} & 54,47 \\
FFORMA  & 43,96 & 51,55 \\
ADE     & 45,72 & 50,93 \\
Mediana & 54,43 & \textbf{58,39} \\
Média   & 200,00 & 44,72 \\
DBA     & 200,00 & 47,20 \\
\bottomrule
\end{tabular}
\end{table}

O resultado em SMAPE é particularmente forte: o CARTE fica atrás apenas do Random Forest (DWT), com 40,52\%, e supera os outros dezoito modelos individuais. A vantagem sobre o segundo melhor modelo individual, o CatBoost (DWT), é de 9,10 pontos percentuais. Em POCID, o CARTE é segundo entre as combinações, atrás da mediana, mas no conjunto completo fica atrás de seis modelos individuais: Random Forest (Original + DWT) 58,39\%, CatBoost (Original + FT) 56,52\%, Random Forest (Original + FT) 55,90\%, CatBoost (Original + CWT) 54,66\%, Random Forest (CWT) 54,66\% e Random Forest (Original + CWT) 54,66\%. O valor de 41,03\% é a média de dez execuções, com desvio-padrão de 4,35 pontos percentuais; a variabilidade do ETTm1 é discutida na Seção~\ref{sec:execucoes-seeds}.

\subsection{Comparação geral e discussão} \label{subsec:ettm1-discussao}

A Figura~\ref{fig:mcm-ettm1} apresenta a Matriz de Comparação Múltipla entre o Random Forest (DWT), o CARTE, o FFORMA, o ADE e a mediana. A MCM de SMAPE usa o melhor individual e a melhor estática por SMAPE.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTM1.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTm1.}
    \label{fig:mcm-ettm1}
\end{figure}

A MCM de SMAPE mostra que o Random Forest (DWT) tem o menor SMAPE médio (40,52\%), seguido do CARTE (41,03\%), do FFORMA (43,96\%), do ADE (45,72\%) e da mediana (54,43\%). O CARTE não se separa do Random Forest (DWT) ($p = 0{,}5781$), do FFORMA ($p = 0{,}9375$), do ADE ($p = 0{,}5781$) nem da mediana ($p = 0{,}9375$). A única comparação significativa da MCM é entre o Random Forest (DWT) e a mediana ($p = 0{,}0156$), com vantagem do modelo individual.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTM1_pocid.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTm1, para o POCID.}
    \label{fig:mcm-ettm1-pocid}
\end{figure}

A MCM de POCID usa o Random Forest (Original + DWT) como melhor individual e a mediana como melhor estática. A mediana e o Random Forest (Original + DWT) empatam em 58,39\%, seguidos do CARTE (54,47\%), do FFORMA (51,55\%) e do ADE (50,93\%). O CARTE não se separa da mediana ($p = 0{,}2188$), do Random Forest (Original + DWT) ($p = 0{,}2969$), do FFORMA ($p = 0{,}6875$) nem do ADE ($p = 0{,}6875$).

O ETTm1 reforça dois pontos positivos do CARTE: a melhor combinação em SMAPE, com vantagem de 2,93 pontos percentuais sobre o FFORMA, e a posição de segundo melhor em POCID entre as combinações, atrás apenas da mediana. O CARTE fica a 0,51 ponto percentual do melhor modelo individual em SMAPE, mas supera todos os outros dezoito preditores e todas as demais combinações. A mediana, que lidera o POCID, tem SMAPE 13,40 pontos percentuais pior, o que evidencia o trade-off entre as duas métricas e a posição equilibrada do CARTE.

Ao longo das dez execuções, a estratégia final foi uma combinação ponderada em cerca de 3,0 das 7 séries, em média; a média aparada em cerca de 1,7, a mediana em cerca de 1,5, o DBA em cerca de 0,5 e a média em cerca de 0,3. O pool completo não foi usado em nenhuma execução; o tamanho mediano do pool final ficou entre 7 e 14 modelos, com mediana de 8,0.
```
