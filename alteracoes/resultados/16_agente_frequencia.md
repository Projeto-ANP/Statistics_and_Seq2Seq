# Análise do comportamento do agente - Abertura e frequência de uso do catálogo

Atualiza a abertura da seção `Análise do comportamento do agente` e a subseção `Frequência de uso do catálogo de ações` de `overleaf/resultados.tex` (linhas 827–860) com a execução representativa da série de dez. As contagens vêm dos registros de auditoria dessa execução.

```latex
\section{Análise do comportamento do agente} \label{sec:resultados-agente}

Esta seção analisa o comportamento do agente combinador ao longo das 680 séries dos sete conjuntos de dados, usando os registros de auditoria descritos na Seção~\ref{sec:aplicacao} do Capítulo~\ref{chp:proposta}. Para a análise qualitativa, usamos a execução mais representativa do conjunto de dez execuções. A escolha não foi manual. Para cada execução, calculamos a média dos sete conjuntos em SMAPE e POCID. A referência é a média das dez execuções, 20,27\% de SMAPE e 54,05\% de POCID. A execução escolhida é a que fica mais próxima dessa referência nas duas métricas, com 20,33\% de SMAPE e 53,99\% de POCID. A distância é padronizada pelas duas médias, para que SMAPE e POCID tenham o mesmo peso. Os traces, contagens e estudos de caso desta seção vêm dessa execução, não da média das dez.

\subsection{Frequência de uso do catálogo de ações}

A Tabela~\ref{tab:agente-acoes} apresenta o número de séries em que cada ação foi usada ao menos uma vez e o número total de chamadas bem-sucedidas, na execução representativa.

\begin{table}[H]
\centering
\caption{Ações usadas na execução representativa, por número de séries e de chamadas bem-sucedidas.}
\label{tab:agente-acoes}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Ação} & \textbf{Séries} & \textbf{Chamadas} \\
\midrule
Avaliação da estratégia & 680 & 12.675 \\
Seleção pelos $k$ melhores & 462 & 550 \\
Peso pela média dos melhores por passo & 337 & 480 \\
Peso pelo melhor preditor por passo & 101 & 109 \\
Peso pela seleção gulosa & 92 & 99 \\
Peso pelo erro inverso & 82 & 110 \\
Seleção pelos mais estáveis & 41 & 46 \\
Seleção gulosa de subconjunto & 8 & 8 \\
Peso pela tendência do erro & 8 & 8 \\
Peso pelo meta-modelo & 1 & 1 \\
\bottomrule
\end{tabular}
\end{table}

A avaliação da estratégia é a ação central. Ela aparece em todas as 680 séries e responde por 12.675 chamadas, 18,6 por série, em média. O agente testa muitas combinações antes de escolher. A seleção pelos $k$ melhores é a ação de seleção mais usada, em 462 séries e 550 chamadas. Em seguida vem a média dos melhores por passo, em 337 séries e 480 chamadas. O padrão é reduzir o pool e testar uma ponderação por passo.

As receitas de peso pelo melhor preditor por passo (101 séries), pela seleção gulosa (92), pelo erro inverso (82), pela seleção mais estável (41), pela seleção gulosa de subconjunto (8), pela tendência do erro (8) e pela ação de peso pelo meta-modelo (1) completam o repertório. As ações menos frequentes aparecem em situações específicas, não como ruído. Não há poda de redundância na execução representativa. A redução do pool é feita pelas ações de seleção, sobretudo a seleção pelos $k$ melhores e a seleção pelos mais estáveis.

O perfil é de exploração intensa. A mediana é de 21 turnos por série, e a média é de 20,86 turnos por série. O agente usa o orçamento para avaliar candidatos e só então define o líder do histórico. O uso do catálogo, portanto, não é o de um agente que aceita a primeira proposta. É o de um agente que testa alternativas de seleção e ponderação antes de decidir.
```
