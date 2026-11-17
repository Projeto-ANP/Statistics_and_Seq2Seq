# Síntese entre conjuntos de dados - Erro por passo do horizonte

Atualiza a subseção `Erro por passo do horizonte` de `overleaf/resultados.tex` (linhas 799–826) com a média das dez execuções do CARTE. As duas figuras foram regeradas com a média por passo e intervalo de confiança de 95%, sem painel separado de desvio-padrão.

```latex
\subsection{Erro por passo do horizonte}

A Figura~\ref{fig:erro-horizonte-smape} ilustra a evolução do SMAPE ao longo do horizonte de previsão, e a Figura~\ref{fig:erro-horizonte-pocid} a evolução do POCID, entre os cinco métodos já discutidos: o Random Forest (DWT) e a mediana como modelo e método fixos globais, ao lado do FFORMA, do ADE e do CARTE. Cada figura apresenta a média de cada método por passo do horizonte e uma faixa de confiança de 95\% para essa média, calculada a partir da variabilidade entre as séries. Como os sete conjuntos de dados têm horizontes de teste diferentes entre si (12 na ANP, 8 no NN5, 13 no M4, e 24 nos quatro ETT), ambas as figuras são restritas aos oito primeiros passos do horizonte, os únicos em que a população de 680 séries dos sete conjuntos de dados permanece constante e integralmente comparável.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{images/resultados_final/erro_horizonte_smape.png}
    \caption[SMAPE por passo do horizonte (oito primeiros passos).]{SMAPE médio por passo do horizonte, com intervalo de confiança de 95\% para a média, restrito aos oito primeiros passos, em que as 680 séries dos sete conjuntos de dados contribuem igualmente.}
    \label{fig:erro-horizonte-smape}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{images/resultados_final/erro_horizonte_pocid.png}
    \caption[POCID por passo do horizonte (oito primeiros passos).]{POCID médio por passo do horizonte, com intervalo de confiança de 95\% para a média, restrito aos oito primeiros passos, em que as 680 séries dos sete conjuntos de dados contribuem igualmente.}
    \label{fig:erro-horizonte-pocid}
\end{figure}

Na Figura~\ref{fig:erro-horizonte-smape}, o SMAPE médio dos cinco métodos cresce de forma moderada entre o primeiro e o oitavo passo, com oscilações. No primeiro passo, a mediana tem o menor valor (9,17\%), seguida do Random Forest (DWT) (9,46\%), do FFORMA (9,68\%), do CARTE (9,86\%) e do ADE (10,43\%). No oitavo, o CARTE assume a liderança (14,64\%), à frente do Random Forest (DWT) (14,97\%), da mediana (15,12\%), do FFORMA (15,49\%) e do ADE (15,72\%). Ao longo dos oito passos, a mediana lidera nos passos 1, 2, 3, 5 e 6, e o CARTE lidera nos passos 4, 7 e 8. As faixas de confiança de 95\% se sobrepõem entre os métodos em todos os passos, o que indica que as diferenças entre as médias são pequenas em relação à variabilidade entre as séries.

Na Figura~\ref{fig:erro-horizonte-pocid}, o POCID médio não apresenta a mesma tendência de crescimento do SMAPE. Ele oscila ao longo dos oito passos, de valores entre 59,85\% e 66,18\% no primeiro passo para valores entre 55,29\% e 63,38\% no oitavo. No primeiro passo, a mediana lidera (66,18\%), à frente do CARTE (64,72\%), do FFORMA (64,26\%), do Random Forest (DWT) (63,24\%) e do ADE (59,85\%). No oitavo, o FFORMA assume a maior taxa (63,38\%), à frente da mediana (60,59\%), do CARTE (57,24\%), do Random Forest (DWT) (56,91\%) e do ADE (55,29\%). O ADE mantém o menor POCID nos dois extremos. As faixas de confiança de 95\% também se sobrepõem entre os métodos, especialmente nos passos intermediários, o que reflete a variabilidade da métrica binária por série.

A partir do nono passo, a população de séries que alimentaria uma média equivalente deixa de ser constante: o NN5 (horizonte 8) para de contribuir a partir daí, a ANP (horizonte 12) a partir do décimo terceiro passo, e o M4 (horizonte 13) a partir do décimo quarto, restando apenas as 28 séries dos quatro conjuntos ETT do décimo quarto ao vigésimo quarto passo, cerca de 4\% da população original. Por essa razão, o comportamento do erro nessa parte estendida do horizonte não é apresentado nesta seção: qualquer diferença observada ali refletiria majoritariamente a composição dos conjuntos de dados, discutida individualmente nas Seções~\ref{sec:resultados-etth1} a~\ref{sec:resultados-ettm2}, e não um efeito genuíno de acúmulo de erro ao longo da distância de previsão.
```
