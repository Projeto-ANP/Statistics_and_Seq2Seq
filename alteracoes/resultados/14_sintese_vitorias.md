# Síntese entre conjuntos de dados - Contagem de vitórias

Atualiza a subseção `Contagem de vitórias` de `overleaf/resultados.tex` (linhas 773–798) com a média das dez execuções do CARTE. A população e os cinco métodos são os mesmos da MCM agregada.

```latex
\subsection{Contagem de vitórias}

A Tabela~\ref{tab:sintese-vitorias} apresenta o número de séries, entre as 680 séries dos sete conjuntos de dados, em que cada método obteve o menor SMAPE entre os cinco comparados. Diferentemente da Tabela~\ref{tab:sintese-ranking}, em que "melhor individual" e "melhor estática" variam por conjunto de dados, aqui os dois termos referem-se a um único modelo e um único método fixos, escolhidos uma única vez pela menor média de SMAPE entre as 680 séries: o Random Forest (DWT) e a mediana, respectivamente, os mesmos identificados na MCM agregada da Seção anterior.

\begin{table}[H]
\centering
\caption{Contagem de vitórias por método, entre as 680 séries dos sete conjuntos de dados.}
\label{tab:sintese-vitorias}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Método} & \textbf{Vitórias} & \textbf{\%} \\
\midrule
Random Forest (DWT) & \textbf{205} & \textbf{30,15} \\
CARTE               & 142 & 20,88 \\
ADE                 & 127 & 18,68 \\
FFORMA              & 114 & 16,76 \\
Mediana             & 92  & 13,53 \\
\bottomrule
\end{tabular}
\end{table}

O Random Forest (DWT), apesar de ser um único modelo fixo aplicado indistintamente às 680 séries de domínios tão distintos quanto demanda de combustíveis, séries semanais do M4 e temperatura de transformadores elétricos, vence em 205 delas, a maior contagem entre os cinco métodos. Isso não contradiz o resultado da MCM agregada da Seção anterior, em que o CARTE obtém o menor SMAPE médio: um único modelo pode vencer com mais frequência, série a série, sem que isso implique o menor erro médio, caso suas derrotas sejam, em média, maiores em magnitude do que suas vitórias. A ordem obtida pela contagem de vitórias também não coincide inteiramente com a ordem do rank médio da Tabela~\ref{tab:sintese-ranking}: o ADE vence em mais séries que o FFORMA (127 contra 114), mesmo tendo o pior rank médio entre os quatro métodos comparáveis (3,86 contra 3,00) e o pior SMAPE médio agregado da Seção anterior. Esse padrão, já observado repetidamente ao longo deste capítulo em conjuntos de dados individuais, volta a se confirmar na agregação: o desempenho médio do ADE é penalizado por perdas de grande magnitude, ainda que vença com mais frequência do que o FFORMA em termos absolutos.
```
