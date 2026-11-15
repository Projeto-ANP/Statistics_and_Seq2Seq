# ETTm2

Atualiza a seção `ETTm2` de `overleaf/resultados.tex` (linhas 601–682) com a média das dez execuções do CARTE. A MCM de SMAPE usa os comparadores escolhidos por SMAPE; a de POCID usa os comparadores escolhidos por POCID.

```latex
\section{ETTm2} \label{sec:resultados-ettm2}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:ettm2-individual} apresenta o desempenho médio, entre as 7 séries do ETTm2, de cada um dos dezenove preditores individuais.

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
FFORMA  & \textbf{16,54} & 31,06 \\
CARTE   & 17,02 & 40,37 \\
ADE     & 18,72 & 31,06 \\
Mediana & 19,97 & 39,13 \\
Média   & 174,05 & \textbf{43,48} \\
DBA     & 174,38 & 37,27 \\
\bottomrule
\end{tabular}
\end{table}

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:ettm2-estaticas}, o FFORMA obtém o menor SMAPE (16,54\%), seguido do CARTE (17,02\%), do ADE (18,72\%) e da mediana (19,97\%). A média e o DBA atingem valores muito acima das demais estratégias (174,05\% e 174,38\%). Em POCID, a média obtém o maior valor entre as seis (43,48\%), seguida do CARTE (40,37\%), da mediana (39,13\%), do DBA (37,27\%) e do FFORMA e do ADE (31,06\%). O CARTE é, portanto, a segunda melhor combinação em SMAPE e a segunda melhor em POCID, atrás da média e à frente da mediana.

O ETTm2 é um caso em que os modelos individuais transformados dominam: o CARTE é a segunda melhor combinação, mas fica atrás dos cinco primeiros modelos individuais em SMAPE. Ainda assim, supera os outros catorze preditores individuais e todas as demais combinações, exceto o FFORMA. A desvantagem para o FFORMA é de 0,48 ponto percentual; a vantagem sobre o ADE e a mediana é de 1,70 e 2,95 pontos percentuais, respectivamente.

\subsection{Comparação geral e discussão} \label{subsec:ettm2-discussao}

A Figura~\ref{fig:mcm-ettm2} apresenta a Matriz de Comparação Múltipla entre o Random Forest (DWT), o FFORMA, o CARTE, o ADE e a mediana. A MCM de SMAPE usa o melhor individual e a melhor estática por SMAPE.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTM2.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTm2.}
    \label{fig:mcm-ettm2}
\end{figure}

A MCM de SMAPE mostra que o Random Forest (DWT) tem o menor SMAPE médio (7,48\%), seguido do FFORMA (16,54\%), do CARTE (17,02\%), do ADE (18,72\%) e da mediana (19,97\%). O CARTE perde para o Random Forest (DWT) ($p = 0{,}0312$) e não se separa do FFORMA ($p = 0{,}9375$), do ADE ($p = 0{,}5781$) nem da mediana ($p = 0{,}2188$).

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTM2_pocid.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTm2, para o POCID.}
    \label{fig:mcm-ettm2-pocid}
\end{figure}

A MCM de POCID usa o ARIMA como melhor individual e a média como melhor estática. O ARIMA lidera (48,45\%), seguido da média (43,48\%), do CARTE (40,37\%), do ADE (31,06\%) e do FFORMA (31,06\%). O CARTE não se separa do ARIMA ($p = 0{,}1094$), da média ($p = 0{,}6875$), do ADE ($p = 0{,}0781$) nem do FFORMA ($p = 0{,}0781$).

O ETTm2 mostra duas posições distintas do CARTE. Em SMAPE, é a segunda melhor combinação, atrás do FFORMA por 0,48 ponto percentual, e supera os catorze modelos individuais que não estão no grupo das cinco variantes transformadas. Em POCID, é o segundo entre as seis combinações, atrás apenas da média e à frente da mediana, mas no conjunto completo fica atrás de nove modelos individuais e empata com dois: ARIMA 48,45\%, Random Forest (Original + DWT) 47,83\%, Random Forest (Original) 47,20\%, Random Forest (CWT) 44,10\%, Random Forest (Original + CWT) 42,86\%, CatBoost (Original) 42,24\%, CatBoost (Original + DWT) 41,61\%, ETS 41,61\% e CatBoost (Original + CWT) 40,99\%; empata com Random Forest (Original + FT) e CatBoost (Original + FT), ambos com 40,37\%. O caso ilustra o compromisso do CARTE: ele permanece no topo das combinações, mas não domina os modelos individuais transformados.

Ao longo das dez execuções, a estratégia final foi uma combinação ponderada em cerca de 3,1 das 7 séries, em média; a mediana em cerca de 1,3, a média aparada em cerca de 1,1, o DBA em cerca de 0,9 e a média em cerca de 0,6. O pool completo não foi usado em nenhuma execução, e o tamanho mediano do pool final foi de 7 modelos.
```
