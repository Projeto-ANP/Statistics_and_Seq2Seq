# Síntese entre conjuntos de dados - Impacto das transformações

Atualiza a subseção `Impacto das transformações` de `overleaf/resultados.tex` (linhas 684–712) com a média das dez execuções do CARTE. O texto e a tabela já revisados foram preservados. O restante é novo.

A comparação principal usa as 670 séries em que o pool final tem ao menos um modelo não transformado. As 10 séries em que o pool fica vazio (4 do ETTh1 e 6 do M4) ficam fora, porque não há modelo não transformado para substituir as transformações. A figura `sintese_transformacoes_delta.png` já está em `overleaf/resultados_final/`.

```latex
\subsection{Impacto das transformações}

Antes de comparar os cinco métodos de combinação, vale destacar um padrão que se repete nas sete tabelas de desempenho dos modelos individuais apresentadas neste capítulo: as variantes de Random Forest e CatBoost treinadas sobre representações obtidas por transformação \textit{wavelet} ou de Fourier, concatenadas à representação original ou isoladas, ocupam a grande maioria das dez primeiras posições por SMAPE em todos os sete conjuntos de dados, conforme a Tabela~\ref{tab:sintese-transformacoes}.

\begin{table}[H]
\centering
\caption{Número de variantes com transformação entre os dez melhores modelos individuais, por conjunto de dados.}
\label{tab:sintese-transformacoes}
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Variantes com transformação no top 10} \\
\midrule
ANP    & 8/10  \\
NN5    & 9/10  \\
M4     & 9/10  \\
ETTh1  & 10/10 \\
ETTh2  & 9/10  \\
ETTm1  & 8/10  \\
ETTm2  & 9/10  \\
\midrule
\textbf{Total} & \textbf{62/70 (88,6\%)} \\
\bottomrule
\end{tabular}
\end{table}

Em nenhum dos sete conjuntos de dados menos de oito das dez melhores posições são ocupadas por essas variantes, e no ETTh1 a totalidade das dez melhores posições é ocupada por modelos com alguma forma de transformação. Esse resultado é consistente com a decisão de projeto de ampliar a diversidade do conjunto de preditores por meio de diferentes representações dos dados, discutida na Seção~\ref{sec:formulacao} do Capítulo~\ref{chp:proposta}: a transformação dos dados não apenas contribui com preditores adicionais ao conjunto $\mathcal{M}$, mas contribui, de forma consistente entre domínios distintos, com os preditores de melhor desempenho individual entre os dezenove disponíveis.

As transformações também dominam as escolhas finais do CARTE. Nas dez execuções, elas aparecem em 6.744 dos 6.800 pools (99,2\%), proporção que fica entre 98,2\% e 100\% em todos os conjuntos. Elas ocupam 58.179 dos 90.581 lugares (64,2\%) e recebem 61,7\% do peso nas estratégias ponderadas. A Tabela~\ref{tab:sintese-transformacoes-escolhas} mostra a distribuição por conjunto de dados.

\begin{table}[H]
\centering
\caption{Composição dos pools finais do CARTE, por conjunto de dados. A última coluna é o peso médio das variantes transformadas nas estratégias ponderadas.}
\label{tab:sintese-transformacoes-escolhas}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Conjunto} & \textbf{Lugares transf. (\%)} & \textbf{Pools sem base ML (\%)} & \textbf{Peso transf. (\%)} \\
\midrule
ANP    & 61,3 & 22,3 & 49,8 \\
NN5    & 59,3 & 22,0 & 45,5 \\
M4     & 66,9 & 20,9 & 74,8 \\
ETTh1  & 88,3 & 80,0 & 99,2 \\
ETTh2  & 67,7 & 32,9 & 84,4 \\
ETTm1  & 58,1 & 20,0 & 63,7 \\
ETTm2  & 74,7 & 34,3 & 69,8 \\
\midrule
\textbf{Total} & \textbf{64,2} & \textbf{22,3} & \textbf{61,7} \\
\bottomrule
\end{tabular}
\end{table}

O ETTh1 é o caso extremo. Nele, 88,3\% dos lugares são transformados, 80,0\% dos pools não têm nenhum modelo base de aprendizado de máquina e as transformações recebem 99,2\% do peso nas estratégias ponderadas. O ETTm1 tem 58,1\% dos lugares transformados. A presença quase universal das transformações não diz, por si só, se elas ajudam a combinação. Para isso, foi realizado o exercício a seguir.

Para medir a contribuição das transformações, removi do pool final as doze variantes transformadas (CWT, DWT e FT de Random Forest e CatBoost), tanto as concatenadas com a representação original quanto as isoladas, e recalculei a previsão com a mesma regra de combinação, os mesmos pesos renormalizados e o mesmo percentual de aparamento. A comparação usa as 670 séries em que o pool final ainda tem ao menos um modelo não transformado. As 10 séries em que o pool fica vazio (4 do ETTh1 e 6 do M4) ficam fora, porque não há modelo não transformado para substituir as transformações. Essas séries são as mais dependentes das transformações e não entram na comparação. A reconstrução da previsão original reproduz exatamente o CARTE. A Tabela~\ref{tab:sintese-transformacoes-remocao-smape} apresenta o resultado.

\begin{table}[H]
\centering
\caption{SMAPE médio do CARTE e da versão sem transformações, nas 670 séries em que o pool final tem ao menos um modelo não transformado.}
\label{tab:sintese-transformacoes-remocao-smape}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Conjunto} & \textbf{SMAPE CARTE (\%)} & \textbf{SMAPE sem transf. (\%)} & \textbf{$\Delta$ SMAPE} \\
\midrule
ANP    & 22,24 & 22,99 & +0,75 \\
NN5    & 12,03 & 13,46 & +1,42 \\
M4     & 9,17  & 23,70 & +14,53 \\
ETTh1  & 14,85 & 35,04 & +20,19 \\
ETTh2  & 17,83 & 50,68 & +32,84 \\
ETTm1  & 41,04 & 48,12 & +7,08 \\
ETTm2  & 17,03 & 25,75 & +8,72 \\
\midrule
\textbf{Total 670} & \textbf{13,73} & \textbf{22,42} & \textbf{+8,69} \\
\bottomrule
\end{tabular}
\end{table}

Sem as transformações, o SMAPE médio sobe de 13,73\% para 22,42\%, um aumento de 8,69 pontos. O aumento ocorre nos sete conjuntos. Os maiores ganhos estão no ETTh2 (+32,84), no ETTh1 (+20,19) e no M4 (+14,53). No ETTm2 (+8,72), no ETTm1 (+7,08), no NN5 (+1,42) e na ANP (+0,75), o ganho também é positivo.

\begin{table}[H]
\centering
\caption{POCID médio do CARTE e da versão sem transformações, nas 670 séries em que o pool final tem ao menos um modelo não transformado.}
\label{tab:sintese-transformacoes-remocao-pocid}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Conjunto} & \textbf{POCID CARTE (\%)} & \textbf{POCID sem transf. (\%)} & \textbf{$\Delta$ POCID} \\
\midrule
ANP    & 63,42 & 65,24 & +1,82 \\
NN5    & 46,02 & 42,41 & -3,61 \\
M4     & 57,58 & 50,46 & -7,12 \\
ETTh1  & 57,65 & 50,08 & -7,57 \\
ETTh2  & 54,14 & 50,42 & -3,72 \\
ETTm1  & 53,90 & 46,86 & -7,05 \\
ETTm2  & 41,43 & 46,69 & +5,26 \\
\midrule
\textbf{Total 670} & \textbf{57,01} & \textbf{53,06} & \textbf{-3,95} \\
\bottomrule
\end{tabular}
\end{table}

$\Delta$ = sem transformações $-$ original. Para o SMAPE, valores positivos indicam piora da versão sem transformações. Para o POCID, valores positivos indicam melhora da versão sem transformações.

A Tabela~\ref{tab:sintese-transformacoes-remocao-pocid} repete a comparação para o POCID. A versão com transformações tem média 57,01\% contra 53,06\% da versão sem transformações, uma diferença de 3,95 pontos. A versão sem transformações só é melhor na ANP (+1,82) e no ETTm2 (+5,26). Nos outros cinco conjuntos, a versão com transformações tem POCID maior.

A Figura~\ref{fig:sintese-transformacoes-delta} resume os deltas médios por conjunto de dados.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{images/resultados_final/sintese_transformacoes_delta.png}
    \caption{$\Delta$ SMAPE médio e $\Delta$ POCID médio entre a versão sem transformações e o CARTE, por conjunto de dados, nas 670 séries em que o pool final tem ao menos um modelo não transformado. Para o SMAPE, valores positivos indicam piora da versão sem transformações. Para o POCID, valores negativos indicam que a versão com transformações tem POCID maior.}
    \label{fig:sintese-transformacoes-delta}
\end{figure}

A leitura é direta. As transformações ajudam a combinação. Sem elas, o SMAPE médio sobe 8,69 pontos e o POCID médio cai 3,95 pontos. O ganho no SMAPE é positivo nos sete conjuntos. É maior no ETTh1, no ETTh2 e no M4, e também positivo na ANP, no NN5, no ETTm1 e no ETTm2. A implicação de não ter as transformações é uma perda de 8,69 pontos de SMAPE na média. A perda é maior no ETTh1, no ETTh2 e no M4, e também aparece na ANP, no NN5, no ETTm1 e no ETTm2.

A ressalva metodológica é que o contrafactual mantém a decisão final do agente fixa. Ele mede a sensibilidade da previsão à composição do pool, não o que o CARTE faria se nunca tivesse tido acesso às transformações.
```
