# ANP

Atualiza a seção `ANP` de `overleaf/resultados.tex` (linhas 116–197) com a média das dez execuções do CARTE. A MCM da ANP foi regerada com a média por série das dez execuções; os testes pareados abaixo usam essa média.

```latex
\section{ANP} \label{sec:resultados-anp}

\subsection{Desempenho dos modelos individuais}

A Tabela~\ref{tab:anp-individual} apresenta o desempenho médio, entre as 182 séries do conjunto de dados de demanda de combustíveis da ANP, de cada um dos dezenove preditores individuais que compõem $\mathcal{M}$. Em todas as tabelas apresentadas a seguir com o desempenho médio, os modelos estão ordenados por SMAPE de maneira crescente.

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
Mediana & 21,94 & 61,94 \\
Média   & 22,06 & 61,89 \\
CARTE  & 22,24 & 63,28 \\
DBA     & 22,52 & 60,54 \\
\bottomrule
\end{tabular}
\end{table}

Entre as seis estratégias de combinação comparadas na Tabela~\ref{tab:anp-estaticas}, o FFORMA obtém o melhor resultado em ambas as métricas, com SMAPE de 21,66\% e POCID de 65,13\%, à frente do ADE (21,77\% e 62,39\%), da mediana (21,94\% e 61,94\%), da média (22,06\% e 61,89\%), do CARTE (22,24\% e 63,28\%) e do DBA (22,52\% e 60,54\%). Apesar da quinta posição em SMAPE, o CARTE tem o segundo maior POCID entre as seis combinações. Nenhuma das seis supera o melhor modelo individual do conjunto em SMAPE, o Theta, cujo valor é 21,54\% (Tabela~\ref{tab:anp-individual}): combinar os dezenove preditores, por qualquer uma das seis estratégias aqui comparadas, não é, neste conjunto de dados específico, melhor do que simplesmente escolher o modelo estatístico mais forte do conjunto. Em POCID, o quadro se altera parcialmente: o FFORMA supera o melhor modelo individual nessa métrica, o ETS (64,69\%), e o CARTE fica próximo, mas abaixo dele (63,28\%), enquanto a mediana, a média, o DBA e o ADE ficam atrás do ETS. O CARTE supera o Theta em POCID (63,28\% contra 63,14\%), mas não o ETS. Esse descolamento entre o desempenho em SMAPE e em POCID já havia sido observado por \citet{krause2026foundation} no mesmo domínio de aplicação, e reforça que a acurácia de magnitude e a acurácia de direção capturam aspectos distintos do problema, conforme discutido na Seção~\ref{sec:metricas}.

\subsection{Comparação geral e discussão} \label{subsec:anp-discussao}

A Figura~\ref{fig:mcm-anp} apresenta a Matriz de Comparação Múltipla entre o melhor modelo individual (Theta), o FFORMA, o ADE, a mediana e o CARTE. A matriz mostra, para cada par, a diferença média, a contagem de vitórias, empates e derrotas e o valor $p$ do teste de Wilcoxon. Ela concentra as comparações pareadas.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ANP_MONTHLY.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos na ANP.}
    \label{fig:mcm-anp}
\end{figure}

O SMAPE médio dos cinco métodos da MCM fica em um intervalo estreito, entre 21,54\% (Theta) e 22,24\% (CARTE). A MCM mostra que o CARTE não se separa do Theta ($p = 0{,}3556$) nem da mediana ($p = 0{,}1033$); separa-se do ADE ($p = 0{,}0116$) e do FFORMA ($p \leq 0{,}0001$), ambos com SMAPE menor.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_ANP_MONTHLY_pocid.png}
    \caption{Matriz de Comparação Múltipla entre os cinco métodos na ANP, para o POCID.}
    \label{fig:mcm-anp-pocid}
\end{figure}

Em POCID, o FFORMA lidera entre os cinco métodos (65,13\%), seguido do CARTE (63,28\%), do Theta (63,14\%), do ADE (62,39\%) e da mediana (61,94\%). O CARTE é o segundo entre as combinações e o terceiro no conjunto completo, atrás do FFORMA e do ETS individual (64,69\%). As diferenças do CARTE para Theta, ADE e mediana não são significativas ($p = 0{,}1058$, $p = 0{,}2827$ e $p = 0{,}1397$); o FFORMA se separa no topo ($p = 0{,}0014$).

A comparação com o FFORMA é a mais desfavorável ao CARTE: o FFORMA tem SMAPE médio menor (21,66\% contra 22,24\%) e POCID maior (65,13\% contra 63,28\%), com diferença significativa nas duas métricas ($p \leq 0{,}0001$ em SMAPE e $p = 0{,}0014$ em POCID). Na contagem por série, o FFORMA vence 117 das 182 séries em SMAPE, contra 65 vitórias do CARTE. A vantagem é de 0,58 ponto percentual em SMAPE e 1,85 ponto em POCID, e se sustenta na maioria das séries.

O CARTE supera o DBA nas duas métricas: SMAPE de 22,24\% contra 22,52\% e POCID de 63,28\% contra 60,54\%. Contra a mediana, o CARTE fica 0,30 ponto percentual atrás em SMAPE (22,24\% contra 21,94\%), diferença que não atinge significância ($p = 0{,}1033$), e à frente em POCID (63,28\% contra 61,94\%), também sem significância ($p = 0{,}1397$). Contra a média, o CARTE tem SMAPE ligeiramente maior (22,24\% contra 22,06\%) e POCID maior (63,28\% contra 61,89\%). Contra o ADE, o CARTE tem SMAPE pior (22,24\% contra 21,77\%), diferença significativa ($p = 0{,}0116$), e POCID melhor (63,28\% contra 62,39\%), sem significância ($p = 0{,}2827$).

Na ANP, ao longo das dez execuções, a estratégia final foi uma combinação ponderada em cerca de 82 das 182 séries, em média; a mediana em cerca de 31, a média em cerca de 28, o DBA em cerca de 27 e a média aparada em cerca de 14. O pool completo foi usado em cerca de 99 séries, e o tamanho mediano do pool final foi de 19 modelos. A variedade de estratégias não se traduziu em vantagem agregada sobre as combinações estáticas: em SMAPE, o CARTE fica atrás da mediana, da média e do ADE; em POCID, é o segundo entre as combinações, à frente de ADE, mediana, média e DBA. A ANP é, portanto, um caso em que a combinação não supera o melhor modelo individual em SMAPE e em que o ganho do agente sobre as combinações simples depende da métrica. Entre as combinações, o CARTE ocupa a quinta posição em SMAPE e a segunda em POCID. O padrão top2 não se mantém em SMAPE, mas se mantém em POCID.
```
