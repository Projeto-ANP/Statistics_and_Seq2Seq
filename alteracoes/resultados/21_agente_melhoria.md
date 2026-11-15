# Análise do comportamento do agente - Distribuição da melhoria relativa

Atualiza a subseção `Distribuição da melhoria relativa` de `overleaf/resultados.tex` (linhas 964–985) com a execução representativa. Como a execução é agent-only, a referência não é a melhor semente. A comparação é com a melhor das primeiras dez tentativas do próprio agente.

```latex
\subsection{Distribuição da melhoria relativa}

Para cada uma das 680 séries, comparou-se a pontuação de validação da estratégia final com a menor pontuação entre as primeiras dez tentativas, segundo o mecanismo descrito na Seção~\ref{subsec:pontuacao} do Capítulo~\ref{chp:proposta}. A estratégia final é, por construção, a melhor tentativa do histórico. Como as primeiras dez tentativas fazem parte desse histórico, ela nunca é pior que a melhor delas. Em nenhuma das 680 séries a estratégia final apresentou pontuação pior que a melhor das primeiras dez tentativas (0\%). Em 509 séries (74,9\%), a estratégia final iguala essa melhor tentativa, dentro da precisão de registro dos escores. Nas 171 restantes (25,1\%), a estratégia final apresenta pontuação estritamente melhor. A maior melhoria observada ocorre na série 9 do M4, na qual a pontuação final é 0,232 menor que a da melhor das primeiras dez tentativas.

\begin{table}[H]
\centering
\caption{Distribuição das 171 séries com melhoria estrita, por faixa de ganho na pontuação.}
\label{tab:melhoria-faixas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Faixa de ganho} & \textbf{Séries} & \textbf{\% das 680} \\
\midrule
até 0,001        & 29  & 4,3  \\
0,001 a 0,01     & 53  & 7,8  \\
0,01 a 0,05      & 67  & 9,9  \\
0,05 a 0,10      & 16  & 2,4  \\
acima de 0,10    & 6   & 0,9  \\
\midrule
\textbf{Total}   & \textbf{171} & \textbf{25,1} \\
\bottomrule
\end{tabular}
\end{table}

A Tabela~\ref{tab:melhoria-faixas} detalha esses ganhos. A maioria das 171 séries com melhoria estrita apresenta ganho de até 0,05 ponto de pontuação (149 séries, 21,9\% das 680), e apenas 6 séries (0,9\%) superam 0,10. Entre as 171, a mediana do ganho é de 0,011 e as dez maiores melhorias concentram 34,2\% do ganho total. O padrão é de ganhos pequenos na maioria dos casos e de um grupo menor com ganhos maiores, o que é consistente com a exploração adicional do agente depois das primeiras dez tentativas.
```
