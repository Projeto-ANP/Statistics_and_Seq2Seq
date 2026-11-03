# ETTh2

Substitui a seção `ETTh2` de `overleaf/resultados.tex` (linhas 384–458), com os resultados do CARTE e as análises atualizadas.

```latex
\section{ETTh2} \label{sec:resultados-etth2}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:etth2-individual} apresenta o desempenho médio, entre as 7 séries do ETTh2, de cada um dos dezenove preditores individuais, ordenados por SMAPE crescente.

\begin{table}[H]
\centering
\caption{Desempenho médio dos modelos individuais no ETTh2.}
\label{tab:etth2-individual}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Modelo} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
CatBoost (Original + CWT)            & \textbf{17,94} & 55,90 \\
CatBoost (FT)                        & 18,30 & 46,58 \\
Random Forest (FT)                   & 18,44 & 50,93 \\
CatBoost (CWT)                       & 18,44 & 52,17 \\
Random Forest (Original)             & 18,58 & 49,69 \\
Random Forest (CWT)                  & 18,75 & 49,07 \\
CatBoost (DWT)                       & 19,01 & 54,66 \\
Random Forest (DWT)                  & 19,50 & \textbf{56,52} \\
CatBoost (Original + DWT)            & 19,50 & 52,80 \\
CatBoost (Original + FT)             & 20,02 & 53,42 \\
Random Forest (Original + DWT)       & 20,50 & 49,69 \\
Random Forest (Original + CWT)       & 20,86 & 52,17 \\
Random Forest (Original + FT)        & 21,98 & 51,55 \\
Naive Moving Average                 & 44,42 & 47,83 \\
ARIMA                                & 46,34 & 54,66 \\
Naive Seasonal                       & 61,96 & 51,55 \\
ETS                                  & 64,35 & 50,93 \\
Theta                                & 65,55 & 48,45 \\
CatBoost (Original)                  & 172,70 & 49,07 \\
\bottomrule
\end{tabular}
\end{table}

O padrão observado no ETTh2 é semelhante ao do ETTh1: os treze modelos de aprendizagem de máquina, com ou sem transformação, ocupam as treze primeiras posições por SMAPE, com valores entre 17,94\% e 21,98\%, enquanto os três modelos estatísticos e os dois métodos ingênuos ficam nas últimas posições, com SMAPE acima de 44\%. O CatBoost (Original + CWT) obtém o menor SMAPE do conjunto, e o Random Forest (DWT) o maior POCID. O CatBoost sobre a representação original, sem transformação, volta a apresentar o pior SMAPE entre os dezenove modelos (172,70\%), embora, diferentemente do ETTh1, não atinja o valor máximo de 200\%.

\subsection{Desempenho entre estratégias de combinação}

\begin{table}[H]
\centering
\caption{Desempenho médio das estratégias de combinação no ETTh2.}
\label{tab:etth2-estaticas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Combinação} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
CARTE   & \textbf{17,31} & 54,66 \\
Mediana & 18,15 & \textbf{56,52} \\
FFORMA  & 20,35 & \textbf{56,52} \\
ADE     & 22,61 & \textbf{56,52} \\
Média   & 171,97 & 49,69 \\
DBA     & 172,57 & 52,80 \\
\bottomrule
\end{tabular}
\end{table}

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:etth2-estaticas}, o CARTE obtém o menor SMAPE (17,31\%), à frente da mediana (18,15\%), do FFORMA (20,35\%) e do ADE (22,61\%). O valor é também inferior ao do melhor modelo individual do conjunto, o CatBoost (Original + CWT), com 17,94\% (Tabela~\ref{tab:etth2-individual}): o ETTh2 é o primeiro conjunto em que o CARTE é o melhor método geral em SMAPE, superando todos os dezenove preditores individuais e todas as combinações. A média e o DBA ficam muito acima das demais estratégias (171,97\% e 172,57\%, respectivamente). Em POCID, a mediana, o FFORMA e o ADE empatam no maior valor entre as seis estratégias (56,52\%), com o CARTE em 54,66\%, à frente do DBA (52,80\%) e da média (49,69\%).

\subsection{Comparação geral e discussão} \label{subsec:etth2-discussao}

A Figura~\ref{fig:mcm-etth2} apresenta a Matriz de Comparação Múltipla entre o CARTE, o CatBoost (Original + CWT), a mediana, o FFORMA e o ADE, ordenados por SMAPE médio.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ETTH2.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos no ETTh2, ordenados por SMAPE médio.}
    \label{fig:mcm-etth2}
\end{figure}

A MCM mostra que o CARTE tem o menor SMAPE médio (17,31\%), seguido do CatBoost (Original + CWT) (17,94\%), da mediana (18,15\%), do FFORMA (20,35\%) e do ADE (22,61\%). O CARTE supera a mediana de forma estatisticamente significativa ($p = 0{,}0314$) e não se separa do CatBoost (Original + CWT) ($p = 0{,}3750$), do FFORMA ($p = 0{,}1562$) nem do ADE ($p = 0{,}2188$). Em POCID, o CARTE não se separa da mediana ($p = 0{,}4446$), do FFORMA ($p = 0{,}5781$), do CatBoost (Original + CWT) ($p = 0{,}7889$) nem do ADE ($p = 0{,}6875$).

O ETTh2 é o resultado mais favorável ao CARTE em SMAPE entre os conjuntos analisados até aqui: ele é o melhor método geral, com vantagem de 0,63 ponto percentual sobre o melhor modelo individual, 0,84 ponto sobre a mediana, 3,04 pontos sobre o FFORMA e 5,30 pontos sobre o ADE. A única ressalva é o POCID: o CARTE fica em quarto lugar entre as combinações, atrás dos três métodos empatados em 56,52\%, e abaixo do CatBoost (Original + CWT), com 55,90\%. O ETTh2 é, portanto, a exceção mais clara à constância top2 no POCID, mas compensa essa posição com a melhor colocação geral em SMAPE.

A estratégia final escolhida pelo CARTE foi a mediana em 2 das 7 séries, a média aparada em 2, a combinação ponderada em 2 e o DBA em 1. O pool completo foi usado em 4 séries, e o tamanho mediano do pool final foi de 19 modelos.
```
