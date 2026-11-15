# Análise do comportamento do agente - Motivo de encerramento do ciclo

Atualiza a subseção `Motivo de encerramento do ciclo` de `overleaf/resultados.tex` (linhas 861–883) com a execução representativa. A configuração dessa execução usa orçamento de avaliação e limite total de turnos, sem parada antecipada.

```latex
\subsection{Motivo de encerramento do ciclo} \label{subsec:motivo-encerramento}

A Tabela~\ref{tab:agente-encerramento} apresenta a distribuição dos motivos de encerramento do ciclo na execução representativa. O ciclo não termina por aceite explícito nem por estagnação. A configuração usada não tem parada antecipada, e as duas únicas formas de encerramento são o orçamento de avaliação e o orçamento total de turnos.

\begin{table}[H]
\centering
\caption{Motivo de encerramento do ciclo, na execução representativa.}
\label{tab:agente-encerramento}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Motivo} & \textbf{Séries} & \textbf{\%} \\
\midrule
Orçamento de avaliação esgotado & 552 & 81,18 \\
Orçamento de turnos esgotado    & 128 & 18,82 \\
\bottomrule
\end{tabular}
\end{table}

Em 552 séries (81,2\%), o ciclo termina quando o orçamento de avaliação se esgota. Em 128 (18,8\%), termina quando o total de 24 turnos se esgota antes. A diferença entre os dois grupos reflete a composição do turno. As ações de seleção e de ponderação não consomem o orçamento de avaliação, mas consomem turnos. Quando o agente prepara muito, o limite de 24 turnos chega primeiro. Quando avalia muito, o orçamento de avaliação chega primeiro. A mediana de 21 turnos e a média de 20,86 mostram que o agente usa quase todo o orçamento disponível.

O encerramento por orçamento não é falha. A escolha final recai sobre a melhor tentativa do histórico, que inclui todas as avaliações feitas. O orçamento limita o custo e garante que o ciclo termine. A distribuição 552/128 mostra que o orçamento de avaliação é o limite efetivo na maioria das séries, e o limite de turnos atua como rede de segurança nas séries em que a preparação consome mais turnos.
```
