# ANP

Substitui a seção `ANP` de `overleaf/resultados.tex` (linhas 77–156), com os resultados do CARTE e as análises atualizadas.

```latex
\section{ANP} \label{sec:resultados-anp}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:anp-individual} apresenta o desempenho médio, entre as 182 séries do conjunto de dados de demanda de combustíveis da ANP, de cada um dos dezenove preditores individuais que compõem $\mathcal{M}$, ordenados por SMAPE crescente.

\begin{table}[H]
\centering
\caption{Desempenho médio dos modelos individuais na ANP.}
\label{tab:anp-individual}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Modelo} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
Theta                                & \textbf{21,54} & 63,14 \\
ETS                                  & 21,66 & \textbf{64,69} \\
Random Forest (Original + CWT)       & 21,76 & 58,39 \\
Random Forest (DWT)                  & 21,88 & 58,39 \\
Random Forest (Original + FT)        & 22,08 & 59,84 \\
Random Forest (Original + DWT)       & 22,29 & 58,79 \\
CatBoost (Original + DWT)            & 22,41 & 58,14 \\
CatBoost (Original + FT)             & 22,49 & 58,04 \\
Random Forest (CWT)                  & 22,87 & 58,14 \\
CatBoost (DWT)                       & 23,09 & 57,99 \\
ARIMA                                & 23,19 & 62,39 \\
CatBoost (Original + CWT)            & 23,33 & 56,64 \\
CatBoost (CWT)                       & 23,41 & 56,89 \\
Random Forest (Original)             & 23,82 & 57,44 \\
Naive Moving Average                 & 24,61 & 58,09 \\
Random Forest (FT)                   & 24,98 & 52,10 \\
CatBoost (FT)                        & 25,27 & 50,45 \\
CatBoost (Original)                  & 26,18 & 57,34 \\
Naive Seasonal                       & 28,15 & 61,29 \\
\bottomrule
\end{tabular}
\end{table}

Os modelos estatísticos Theta e ETS ocupam as duas primeiras posições por SMAPE, mas não coincidem: o Theta tem o menor SMAPE do conjunto (21,54\%), enquanto o ETS tem o maior POCID (64,69\%), evidenciando que o ranqueamento entre preditores pode variar conforme a métrica considerada, um comportamento esperado dada a natureza distinta de cada uma delas, discutida na Seção~\ref{sec:metricas}. Os dois métodos ingênuos, como esperado, ocupam posições inferiores por SMAPE, mas o Naive Seasonal ainda supera catorze dos dezessete modelos mais sofisticados em POCID (61,29\%), perdendo apenas para o Theta, o ETS e o ARIMA, indicando que a repetição do padrão sazonal do ano anterior continua sendo um indicador razoável da direção de variação da demanda, ainda que não da sua magnitude.

\subsection{Desempenho entre estratégias de combinação}

\begin{table}[H]
\centering
\caption{Desempenho médio das estratégias de combinação na ANP.}
\label{tab:anp-estaticas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Combinação} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
FFORMA & \textbf{21,66} & \textbf{65,13} \\
ADE    & 21,77 & 62,39 \\
CARTE  & 21,93 & 63,44 \\
Mediana & 21,94 & 61,94 \\
Média   & 22,06 & 61,89 \\
DBA     & 22,52 & 60,54 \\
\bottomrule
\end{tabular}
\end{table}

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:anp-estaticas}, o FFORMA obtém o melhor resultado em ambas as métricas, com SMAPE de 21,66\% e POCID de 65,13\%, à frente do ADE (21,77\% e 62,39\%), do CARTE (21,93\% e 63,44\%), da mediana (21,94\% e 61,94\%), da média (22,06\% e 61,89\%) e do DBA (22,52\% e 60,54\%). Nenhuma das seis supera o melhor modelo individual do conjunto em SMAPE, o Theta, cujo valor é 21,54\% (Tabela~\ref{tab:anp-individual}): combinar os dezenove preditores, por qualquer uma das seis estratégias aqui comparadas, não é, neste conjunto de dados específico, melhor do que simplesmente escolher o modelo estatístico mais forte do conjunto. Em POCID, o quadro se altera parcialmente: o FFORMA supera o melhor modelo individual nessa métrica, o ETS (64,69\%), e o CARTE fica próximo, mas abaixo dele (63,44\%), enquanto a mediana, a média, o DBA e o ADE ficam atrás do ETS. Esse descolamento entre o desempenho em SMAPE e em POCID já havia sido observado por \citet{krause2026foundation} no mesmo domínio de aplicação, e reforça que a acurácia de magnitude e a acurácia de direção capturam aspectos distintos do problema, conforme discutido na Seção~\ref{sec:metricas}.

\subsection{Comparação geral e discussão} \label{subsec:anp-discussao}

A Figura~\ref{fig:mcm-anp} apresenta a Matriz de Comparação Múltipla entre o melhor modelo individual (Theta), a melhor combinação estática (mediana), o FFORMA, o ADE e o CARTE, ordenados por SMAPE médio. A matriz mostra, para cada par, a diferença média, a contagem de vitórias, empates e derrotas e o valor $p$ do teste de Wilcoxon; ela concentra as comparações pareadas.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ANP_MONTHLY.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos na ANP, ordenados por SMAPE médio.}
    \label{fig:mcm-anp}
\end{figure}

O SMAPE médio dos métodos fica em um intervalo estreito, entre 21,54\% (Theta) e 22,52\% (DBA). A MCM mostra que o CARTE não se separa do Theta ($p = 0{,}9334$), do ADE ($p = 0{,}5832$) nem da mediana ($p = 0{,}8466$); o FFORMA é o único método que se separa do CARTE em SMAPE, com vantagem ($p = 0{,}0375$). Em POCID, o FFORMA lidera entre as combinações (65,13\%), seguido do CARTE (63,44\%); o CARTE supera o ADE (62,39\%), a mediana (61,94\%), a média (61,89\%) e o DBA (60,54\%), mas fica abaixo do ETS individual (64,69\%).

A comparação com o FFORMA é a mais desfavorável ao CARTE e a única, entre as comparações com estratégias de combinação, que se aproxima de uma separação consistente: o FFORMA tem SMAPE médio menor (21,66\% contra 21,93\%) e POCID maior (65,13\% contra 63,44\%), com diferença significativa em SMAPE na MCM ($p = 0{,}0375$). Na contagem por série, o FFORMA vence 107 das 182 séries em SMAPE, contra 75 vitórias do CARTE. A vantagem é pequena em magnitude, mas se sustenta na maioria das séries.

O CARTE supera o DBA nas duas métricas: SMAPE de 21,93\% contra 22,52\% e POCID de 63,44\% contra 60,54\%. Contra a mediana, a média e o ADE, as diferenças são pequenas. O CARTE fica praticamente empatado com a mediana em SMAPE (21,93\% contra 21,94\%) e a supera em POCID (63,44\% contra 61,94\%); contra a média, a diferença em SMAPE também é desprezível (21,93\% contra 22,06\%). Contra o ADE, o CARTE tem SMAPE ligeiramente pior (21,93\% contra 21,77\%) e POCID melhor (63,44\% contra 62,39\%).

Na ANP, a estratégia final escolhida pelo CARTE foi uma combinação ponderada em 79 das 182 séries, média em 34, mediana em 34, DBA em 23 e média aparada em 12. O pool completo foi usado em 104 séries, e o tamanho mediano do pool final foi de 19 modelos. A variedade de estratégias não se traduziu, porém, em uma vantagem agregada sobre as combinações estáticas: o desempenho do CARTE fica próximo da mediana e do ADE em SMAPE e abaixo do FFORMA nas duas métricas. A ANP é, portanto, um caso em que a combinação não supera o melhor modelo individual em SMAPE e em que o ganho do agente sobre as combinações simples é limitado, o que ajuda a delimitar em que condições o CARTE agrega valor. Entre as combinações, o CARTE ocupa a terceira posição em SMAPE e a segunda em POCID; a ANP é a única exceção à constância top2 em SMAPE observada nos demais conjuntos, enquanto no POCID o padrão top2 se mantém.
```
