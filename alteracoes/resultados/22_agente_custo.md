# Análise do comportamento do agente - Custo computacional

Atualiza a subseção `Custo computacional` de `overleaf/resultados.tex` (linhas 986–1009) com a execução representativa. Os tempos foram recalculados a partir de `agent_wall_clock_s`.

```latex
\subsection{Custo computacional}

A Tabela~\ref{tab:agente-custo} apresenta o tempo médio de decisão por série, por conjunto de dados.

\begin{table}[H]
\centering
\caption{Tempo médio de decisão por série, por conjunto de dados.}
\label{tab:agente-custo}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Tempo médio (s)} & \textbf{Desvio-padrão (s)} \\
\midrule
ETTm1  & 17,63 & 2,36 \\
ETTh2  & 17,73 & 1,89 \\
ANP    & 19,10 & 2,83 \\
ETTm2  & 19,72 & 2,66 \\
ETTh1  & 20,45 & 2,66 \\
M4     & 21,98 & 3,14 \\
NN5    & 22,95 & 5,97 \\
\bottomrule
\end{tabular}
\end{table}

O tempo médio de decisão varia de aproximadamente 18 a 23 segundos por série, dependendo do conjunto de dados, um custo baixo quando comparado ao tempo de execução dos modelos individuais, das combinações estáticas e dos métodos de meta-aprendizagem da literatura. O desvio-padrão fica entre 1,9 e 6,0 segundos, o que indica variação moderada entre séries de um mesmo conjunto. A maior variação aparece no NN5, com 6,0 segundos. A variação reflete tanto o número de turnos quanto a latência variável do modelo de linguagem.
```
