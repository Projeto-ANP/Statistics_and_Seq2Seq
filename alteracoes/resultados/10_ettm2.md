# ETTm2

Substitui a seção `ETTm2` de `overleaf/resultados.tex` (linhas 534–608), com os resultados do CARTE e as análises atualizadas.

```latex
\section{ETTm2} \label{sec:resultados-ettm2}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:ettm2-individual} apresenta o desempenho médio, entre as 7 séries do ETTm2, de cada um dos dezenove preditores individuais, ordenados por SMAPE crescente.

\begin{table}[H]
\centering
\caption{Desempenho médio dos modelos individuais no ETTm2.}
\label{tab:ettm2-individual}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Modelo} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
Random Forest (DWT)                  & \textbf{7,48} & 32,30 \\
CatBoost (DWT)                       & 8,54 & 36,65 \\
CatBoost (CWT)                       & 9,02 & 31,68 \\
Random Forest (FT)                   & 9,55 & 39,13 \\
CatBoost (FT)                        & 9,68 & 31,06 \\
Random Forest (CWT)                  & 20,79 & 44,10 \\
Random Forest (Original + FT)        & 22,71 & 40,37 \\
CatBoost (Original + CWT)            & 22,84 & 40,99 \\
Random Forest (Original)             & 23,22 & 47,20 \\
CatBoost (Original + FT)             & 23,84 & 40,37 \\
CatBoost (Original + DWT)            & 24,23 & 41,61 \\
Random Forest (Original + DWT)       & 25,45 & 47,83 \\
Random Forest (Original + CWT)       & 27,39 & 42,86 \\
ARIMA                                & 28,14 & \textbf{48,45} \\
ETS                                  & 46,24 & 41,61 \\
Theta                                & 55,43 & 34,78 \\
Naive Moving Average                 & 91,67 & 35,40 \\
Naive Seasonal                       & 100,33 & 29,19 \\
CatBoost (Original)                  & 171,64 & 42,24 \\
\bottomrule
\end{tabular}
\end{table}

O ETTm2 apresenta a maior separação observada até aqui entre um grupo de destaque e o restante da tabela: as cinco variantes de CatBoost e Random Forest treinadas exclusivamente sobre uma única representação transformada (CWT, DWT ou FT, sem concatenação com a representação original) ocupam as cinco primeiras posições, todas com SMAPE abaixo de 10\%, enquanto a sexta colocada, o Random Forest (CWT), já salta para 20,79\%. O Random Forest (DWT) obtém o menor SMAPE do conjunto (7,48\%). O modelo de melhor POCID, o ARIMA (48,45\%), está longe de figurar entre os melhores por SMAPE, reforçando novamente a distinção entre as duas métricas.

\subsection{Desempenho entre estratégias de combinação}

\begin{table}[H]
\centering
\caption{Desempenho médio das estratégias de combinação no ETTm2.}
\label{tab:ettm2-estaticas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Combinação} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
CARTE   & \textbf{16,42} & 39,13 \\
FFORMA  & 16,54 & 31,06 \\
ADE     & 18,72 & 31,06 \\
Mediana & 19,97 & 39,13 \\
Média   & 174,05 & \textbf{43,48} \\
DBA     & 174,38 & 37,27 \\
\bottomrule
\end{tabular}
\end{table}

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:ettm2-estaticas}, o CARTE obtém o menor SMAPE (16,42\%), à frente do FFORMA (16,54\%), do ADE (18,72\%) e da mediana (19,97\%). A média e o DBA atingem valores muito acima das demais estratégias (174,05\% e 174,38\%). Em POCID, a média obtém o maior valor entre as seis (43,48\%), seguida do CARTE e da mediana, empatados em 39,13\%, do DBA (37,27\%) e do FFORMA e do ADE (31,06\%). O CARTE é, portanto, a melhor combinação em SMAPE e a segunda melhor em POCID, empatado com a mediana.

O ETTm2 é um caso em que os modelos individuais transformados dominam: o CARTE é a melhor combinação, mas fica atrás dos cinco primeiros modelos individuais em SMAPE. Ainda assim, supera os outros catorze preditores individuais e todas as demais combinações. A vantagem sobre o FFORMA é pequena (0,12 ponto percentual), mas suficiente para colocá-lo no topo das combinações; a vantagem sobre o ADE e a mediana é de 2,30 e 3,55 pontos percentuais, respectivamente.

\subsection{Comparação geral e discussão} \label{subsec:ettm2-discussao}

A Figura~\ref{fig:mcm-ettm2} apresenta a Matriz de Comparação Múltipla entre o Random Forest (DWT), o CARTE, o FFORMA, o ADE e a mediana, ordenados por SMAPE médio.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTM2.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTm2, ordenados por SMAPE médio.}
    \label{fig:mcm-ettm2}
\end{figure}

A MCM mostra que o Random Forest (DWT) tem o menor SMAPE médio (7,48\%), seguido do CARTE (16,42\%), do FFORMA (16,54\%), do ADE (18,72\%) e da mediana (19,97\%). O CARTE perde para o Random Forest (DWT) ($p = 0{,}0469$) e não se separa do FFORMA ($p = 1{,}0000$), do ADE ($p = 0{,}2969$) nem da mediana ($p = 0{,}1562$). Em POCID, o CARTE não se separa da mediana ($p = 1{,}0000$), do Random Forest (DWT) ($p = 0{,}2969$), do ADE ($p = 0{,}0749$) nem do FFORMA ($p = 0{,}0781$).

O ETTm2 confirma a capacidade do CARTE de ser a melhor combinação em SMAPE mesmo quando o conjunto é dominado por um pequeno grupo de modelos individuais muito fortes. O CARTE não alcança os cinco primeiros individuais, mas supera os catorze restantes e todas as outras combinações. Em POCID, mantém a constância no top2 entre as combinações, empatado com a mediana e atrás apenas da média, cujo SMAPE de 174,05\% inviabiliza seu uso como estratégia principal. O caso ilustra o compromisso do CARTE: ele não é o melhor em todos os cenários, mas permanece competitivo e no topo das combinações.

A estratégia final escolhida pelo CARTE foi uma combinação ponderada em 3 das 7 séries, a média aparada em 2, a mediana em 1 e a média em 1. O pool completo foi usado em 2 séries, e o tamanho mediano do pool final foi de 7 modelos.
```
