# Síntese entre conjuntos de dados - MCM agregada

Atualiza a subseção `MCM agregada entre todos os conjuntos de dados` de `overleaf/resultados.tex` (linhas 745–772) com a média das dez execuções do CARTE. As MCMs já foram regeradas com a pasta `orchestrator_react_consist_seed11_20_mean`.

```latex
\subsection{MCM agregada entre todos os conjuntos de dados}

A Figura~\ref{fig:mcm-agregado-smape} apresenta a Matriz de Comparação Múltipla por SMAPE, e a Figura~\ref{fig:mcm-agregado-pocid} por POCID, tratando as 680 séries dos sete conjuntos de dados como uma única população de comparação, entre CARTE, a melhor combinação estática global (a mediana), o FFORMA, o ADE, e o melhor modelo individual global (o Random Forest treinado sobre a transformação \textit{wavelet} discreta).

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_agregado_smape.png}
    \caption{Matriz de Comparação Múltipla agregada por SMAPE, entre as 680 séries dos sete conjuntos de dados.}
    \label{fig:mcm-agregado-smape}
\end{figure}

Por SMAPE, o CARTE obtém o menor valor médio entre os cinco métodos (13,78\%), à frente da mediana (13,79\%), do FFORMA (14,15\%), do Random Forest (DWT) (14,28\%) e do ADE (14,70\%). A diferença para a mediana é de 0,01 ponto percentual. Nenhuma comparação entre o CARTE e os demais métodos atinge significância estatística. Os valores de $p$ são 0,2342 contra a mediana, 0,2397 contra o FFORMA, 0,2539 contra o Random Forest (DWT) e 0,2249 contra o ADE. As comparações significativas da matriz são a mediana contra o Random Forest (DWT) ($p = 0{,}0048$) e contra o ADE ($p = 0{,}0002$), e o FFORMA contra o ADE ($p \leq 0{,}0001$). A comparação entre o CARTE e a mediana registra 17 empates. A vantagem do CARTE em SMAPE é numérica, não significativa.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/resultados_final/mcm_agregado_pocid.png}
    \caption{Matriz de Comparação Múltipla agregada por POCID, entre as 680 séries dos sete conjuntos de dados.}
    \label{fig:mcm-agregado-pocid}
\end{figure}

Por POCID, o FFORMA obtém o maior valor médio (58,21\%), seguido da mediana (57,20\%), do CARTE (57,18\%), do ADE (56,37\%) e do Random Forest (DWT) (56,16\%). O FFORMA supera o CARTE ($p = 0{,}0090$), a mediana ($p = 0{,}0392$), o ADE ($p = 0{,}0001$) e o Random Forest (DWT) ($p = 0{,}0003$), com significância estatística. O CARTE fica em terceiro, praticamente empatado com a mediana ($p = 0{,}6635$), à frente do ADE ($p = 0{,}0777$) e do Random Forest (DWT) ($p = 0{,}1358$), sem diferença significativa. A mediana supera o Random Forest (DWT) ($p = 0{,}0369$). As demais comparações não são significativas.

Em conjunto, as duas matrizes agregadas mostram que o CARTE tem o menor SMAPE médio (13,78\%), mas sem separação estatística dos demais métodos. No POCID, fica em terceiro (57,18\%), atrás do FFORMA (58,21\%) e da mediana (57,20\%), sem diferença significativa para a mediana. O FFORMA lidera o POCID com significância. A vantagem do CARTE não se repete de forma uniforme conjunto a conjunto. Ele é primeiro no ETTh2, segundo no M4, no ETTh1 e no ETTm1, terceiro no ETTm2, quarto no NN5 e quinto na ANP. O ganho médio decorre de um desempenho relativamente estável na maioria dos conjuntos, não de uma superioridade absoluta em cada um.
```
