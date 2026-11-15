# Síntese entre conjuntos de dados - Ranking por conjunto de dados

Atualiza a subseção `Ranking por conjunto de dados` de `overleaf/resultados.tex` (linhas 711–744) com a média das dez execuções do CARTE. O texto revisado foi preservado. A tabela e os números foram atualizados.

```latex
\subsection{Ranking por conjunto de dados}

A Tabela~\ref{tab:sintese-ranking} consolida a posição de cada método em cada um dos sete conjuntos de dados, por SMAPE médio, com 1 indicando a primeira posição no ranking. As colunas \emph{melhor individual} e \emph{melhor estática} referem-se, em cada conjunto de dados, ao modelo individual e à combinação estática de melhor desempenho naquele conjunto especificamente, não a um método fixo. Conforme discutido anteriormente, o modelo que ocupa essa posição varia conforme o conjunto de dados.

\begin{table}[H]
\centering
\caption{Ranking dos métodos por conjunto de dados, segundo o SMAPE médio.}
\label{tab:sintese-ranking}
\footnotesize
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Melhor individual} & \textbf{Melhor estática} & \textbf{FFORMA} & \textbf{ADE} & \textbf{CARTE} \\
\midrule
ANP    & 1 (Theta)                       & 4 (Mediana) & 2 & 3 & 5 \\
NN5    & 5 (CatBoost Original + FT)      & 3 (Média)   & 2 & 1 & 4 \\
M4     & 3 (Random Forest CWT)           & 1 (Mediana) & 4 & 5 & 2 \\
ETTh1  & 1 (Random Forest CWT)           & 3 (Mediana) & 4 & 5 & 2 \\
ETTh2  & 2 (CatBoost Original + CWT)     & 3 (Mediana) & 4 & 5 & 1 \\
ETTm1  & 1 (Random Forest DWT)           & 5 (Mediana) & 3 & 4 & 2 \\
ETTm2  & 1 (Random Forest DWT)           & 5 (Mediana) & 2 & 4 & 3 \\
\midrule
\textbf{Rank médio} & \textbf{2,00} & 3,43 & 3,00 & 3,86 & 2,71 \\
\bottomrule
\end{tabular}
\end{table}

Nenhum método ocupa a primeira posição em todos os conjuntos de dados. Descontado o melhor modelo individual, cuja posição de destaque no rank médio (2,00) reflete uma escolha retrospectiva, feita conhecendo o resultado de cada um dos dezenove modelos naquele conjunto de dados específico, e não uma decisão que poderia ser tomada de antemão, o CARTE tem o melhor rank médio entre os métodos comparáveis (2,71), à frente do FFORMA (3,00), da melhor combinação estática (3,43) e do ADE (3,86).

Essa posição reflete um padrão de consistência, não de dominância: o CARTE ocupa a primeira ou a segunda posição em quatro dos sete conjuntos de dados (M4, ETTh1, ETTh2 e ETTm1), fica em terceiro no ETTm2, em quarto no NN5 e em quinto na ANP. O FFORMA nunca ocupa a primeira posição em nenhum dos sete conjuntos, atingindo no máximo a segunda colocação (ANP, NN5 e ETTm2), mas também nunca ocupa a última. O ADE lidera no NN5, mas termina em último em três conjuntos (M4, ETTh1 e ETTh2). Em resumo, o CARTE concentra suas melhores posições no M4, no ETTh1, no ETTh2 e no ETTm1, o FFORMA permanece entre a segunda e a quarta posição e o ADE é penalizado pelas perdas de grande magnitude, ainda que vença com frequência em contagem direta.

Vale notar que o melhor modelo individual de cada conjunto de dados, apesar de ser por definição o mais forte entre os dezenove preditores daquele conjunto, nem sempre se sustenta como o melhor entre os cinco métodos comparados nesta tabela: no NN5, o CatBoost (Original + FT) ocupa a última posição, superado pela média, pelo FFORMA, pelo ADE e pelo CARTE, todos métodos de combinação.

As duas subseções seguintes adotam uma definição diferente de melhor individual e melhor estática, por uma exigência estatística, não por inconsistência, pois em vez de recalculados por conjunto de dados, como nesta subseção, passam a ser um único modelo e um único método, fixados uma vez a partir do desempenho médio sobre a totalidade das 680 séries. Essa mudança é necessária porque as comparações pareadas empregadas a seguir, a Matriz de Comparação Múltipla e a contagem de vitórias, exigem que cada método comparado mantenha uma identidade única e constante ao longo de toda a população avaliada, o que uma definição que muda por conjunto de dados não permitiria.
```
