# Contexto por série

Substitui a seção `Caracterização e diagnóstico por série` de `overleaf/proposta.tex` (linhas 367–408). A seção passa a reunir o perfil, a avaliação do pool, o diagnóstico e a representação da forma das previsões, todos específicos da série.

```latex
\section{Contexto por série} \label{sec:diagnostico}

Antes de o agente propor qualquer estratégia de combinação para a série $i$, o CARTE constrói um contexto específico dessa série, sem participação de modelo de linguagem. Esse contexto reúne o perfil $\mathbf{z}_i$ do histórico $\mathcal{S}_i$; a avaliação $\mathcal{E}_i$ do desempenho dos preditores nas janelas de validação; o diagnóstico interpretativo $\mathcal{C}_i$; e a representação $\mathcal{F}_i$ da forma das previsões entregue ao agente. O perfil, a avaliação e o diagnóstico são calculados antes do ciclo; a representação $\mathcal{F}_i$ é derivada das mesmas previsões e entregue ao agente a cada turno. Após o ciclo, uma segunda visão da forma das previsões, $\mathcal{G}_i$, é construída para a decisão final, com escala comum, referência sazonal e características dos candidatos. No contexto do turno, as quatro representações correspondem aos blocos \texttt{series\_profile}, \texttt{model\_pool}, \texttt{diagnosis} e \texttt{forecast\_shape}, registrados na Tabela~\ref{tab:blocos_contexto}. Nenhuma delas utiliza os valores reais da janela de teste.

\subsection{Perfil da série}

O perfil $\mathbf{z}_i$ é calculado apenas sobre o trecho do histórico $\mathcal{S}_i$ anterior à janela de teste. Ele descreve a escala, a estrutura temporal e o comportamento estatístico da série, e serve de insumo para a leitura do agente, para o diagnóstico interpretativo e para a condição sazonal da decisão final. O perfil reúne o número de observações $n_i$, a frequência de amostragem, o horizonte $h$, o período sazonal e sua origem, o número de janelas de validação e o número de preditores disponíveis.

O período sazonal $p_i$ é resolvido em três níveis: um valor explicitamente definido para o conjunto de dados; a frequência declarada da série; e, na ausência de ambos, o horizonte de previsão, limitado a $12$ observações. O perfil registra o período adotado, sua origem e o método de decomposição empregado, que indica se o STL foi aplicado ou se houve recurso à tendência linear.

A decomposição da série em tendência, sazonalidade e resíduo é feita por STL robusto quando o histórico contém pelo menos dois ciclos completos do período adotado; caso contrário, o perfil recorre a uma tendência linear e registra força sazonal nula. O período usado na decomposição é limitado para que o STL disponha de ao menos dois ciclos, e o período declarado é preservado à parte quando não cabe no histórico. A partir dessa decomposição são calculadas a força de tendência e a força de sazonalidade, ambas normalizadas no intervalo $[0,1]$, além da direção da tendência e da mudança relativa entre o início e o fim do histórico. Também compõem o perfil o veredito de estacionariedade obtido por testes ADF e KPSS, o percentual de observações atípicas identificadas por escore $z$, a autocorrelação de primeira ordem, a autocorrelação no período sazonal, a entropia espectral, o expoente de Hurst, o coeficiente de variação e um indicador de cruzamentos por zero.

O perfil completo, incluindo as características descritivas calculadas para a série, é registrado para auditoria. O agente recebe uma versão resumida, restrita aos campos que informam uma decisão de combinação: escala, período, forças de tendência e sazonalidade, método de decomposição, campeões de tendência e sazonalidade, estacionariedade, outliers, autocorrelações, entropia, coeficiente de variação, cruzamentos por zero e um marcador de disponibilidade de \textit{catch22}. A versão resumida evita que o contexto do turno seja dominado por dezenas de números que não se traduzem em uma escolha de pool ou de ponderação.

\subsection{Avaliação do conjunto de preditores}

A avaliação $\mathcal{E}_i$ mede o desempenho de cada preditor $M_j \in \mathcal{M}$ isoladamente, e não o de uma combinação. Para cada preditor e cada janela de validação $w$, calcula-se o RMSE na escala original da série,
\begin{equation}
    \text{RMSE}_{i,j}^{(w)}
    =
    \sqrt{\frac{1}{h} \sum_{k=1}^{h}
        \left(y_{i,k}^{(w)} - \hat{y}_{i,j,k}^{(w)}\right)^2},
\end{equation}
em que $y_{i,k}^{(w)}$ e $\hat{y}_{i,j,k}^{(w)}$ são, respectivamente, a componente $k$ do vetor de valores reais $\mathbf{y}_i^{(w)}$ e do vetor de previsões $\hat{\mathbf{y}}_{i,j}^{(w)}$, definidos na Seção~\ref{sec:formulacao}. A tabela de erro principal agrega o RMSE das três janelas; o melhor preditor por janela e a estabilidade do \textit{ranking} usam o RMSE calculado separadamente em cada janela. O RMSE é a métrica padrão desses três resumos e também é o padrão das ações de seleção e de parte das ações de pesos. A pontuação que ordena as tentativas de combinação utiliza o SMAPE conforme o procedimento da Seção~\ref{sec:pontuacao}; o RMSE não entra nessa pontuação nem nas métricas finais de teste, que são SMAPE e POCID.

A partir do RMSE são construídos três resumos. O primeiro é uma tabela de erro que ranqueia os preditores e mostra os primeiros colocados, acompanhados de uma agregação dos demais e da dispersão relativa entre o melhor e o pior preditor. O segundo identifica o melhor preditor em cada janela de validação, o que evidencia se o vencedor muda ao longo do tempo. O terceiro mede a estabilidade do \textit{ranking} entre janelas pelo coeficiente de correlação de postos de Kendall, $\tau$, calculado para cada par de janelas e resumido pela média. Um $\tau$ alto indica que a ordem dos preditores se mantém; um $\tau$ baixo indica que o melhor preditor de uma janela não é garantia de bom desempenho na janela seguinte.

A avaliação também preserva os resíduos
\[
    r_{i,j,k}^{(w)}
    =
    \hat{y}_{i,j,k}^{(w)} - y_{i,k}^{(w)},
\]
com o sinal da diferença. A correlação entre os resíduos de cada par de preditores permite identificar grupos que erram de forma parecida e na mesma direção. Preditores com erro fortemente correlacionado tendem a ser redundantes: combiná-los acrescenta menos diversidade do que combinar preditores cujos erros não seguem o mesmo padrão. O contexto do agente recebe apenas os grupos redundantes e um resumo da correlação média, não a matriz completa.

\subsection{Diagnóstico interpretativo}

O diagnóstico $\mathcal{C}_i$ traduz o perfil e a avaliação do pool em quatro categorias textuais e uma narrativa curta. A tradução é feita por regras determinísticas, sem participação de modelo de linguagem, e possui vocabulário fechado.

O regime da série é determinado pelas forças de tendência e de sazonalidade, com limiar de $0{,}6$. Quando as duas são altas, o regime é misto; quando apenas a sazonalidade é alta, o regime é dominado por sazonalidade; quando apenas a tendência é alta, o regime é dominado por tendência; e quando nenhuma das duas atinge o limiar, o regime é ruidoso.

A previsibilidade é determinada pela entropia espectral. Entropia alta ($\geq 0{,}85$) indica previsibilidade baixa, entropia baixa ($\leq 0{,}5$) indica previsibilidade alta e valores intermediários, assim como a ausência da característica, indicam previsibilidade média.

A indicação de combinação nasce do cruzamento entre a estabilidade do \textit{ranking}, medida por $\tau$, e a dispersão relativa de erro entre os preditores. Quando $\tau<0{,}3$, o \textit{ranking} é pouco estável e a indicação é de combinação robusta, menos sensível à escolha exata de pesos; quando a dispersão é pelo menos $0{,}5$ e $\tau\geq0{,}5$, a indicação é de combinação seletiva, restrita a um subconjunto; quando a dispersão é pelo menos $0{,}3$, a indicação é de combinação ponderada; fora desses casos, a indicação é de uso do conjunto completo, sem seleção ou ponderação adicional.

Por fim, o diagnóstico reúne até três riscos, construídos a partir de condições independentes: instabilidade do \textit{ranking} entre janelas; existência de algum grupo de preditores com erro correlacionado; previsibilidade baixa, sob o risco de que pesos ajustados a uma série próxima do ruído resultem em sobreajuste; e a limitação de haver apenas três janelas de validação para fundamentar qualquer estimativa. Como a lista é truncada nos três primeiros, o risco das três janelas entra quando os demais não o deslocam. As categorias e os riscos são sintetizados em uma narrativa textual curta, entregue ao agente como leitura qualitativa e não como regra.

\subsection{Forma das previsões}

A forma das previsões é usada em dois momentos, com recortes distintos. Durante o ciclo, o agente recebe uma representação $\mathcal{F}_i$ construída para comparar trajetórias sem revelar o alvo. Após o ciclo, a decisão final usa uma representação $\mathcal{G}_i$, com escala comum e referência sazonal, para calcular as distâncias e as características de plausibilidade dos candidatos. Ambas derivam das previsões de teste e da última janela de validação; nenhuma acessa $\mathbf{y}_i^{(\mathrm{teste})}$.

No ciclo, $\mathcal{F}_i$ normaliza o vetor de valores reais da última janela de validação por escore $z$ e resume cada previsão de teste por inclinação, volatilidade e posição do último passo, além do SMAPE das janelas de validação e do grupo de correlação formado a partir das previsões de teste. Para os preditores selecionados, inclui os vetores de previsão de teste normalizados por escore $z$ e a previsão correspondente na última janela de validação. O agente recebe os vetores detalhados dos preditores de melhor desempenho na última janela e de até um representante de cada grupo correlacionado, as correlações par a par entre esses preditores e a dispersão média entre os modelos. A normalização por vetor preserva a forma e remove a escala; a comparação de nível é feita dentro de cada trajetória.

Na decisão final, $\mathcal{G}_i$ usa uma escala comum, o desvio-padrão do trecho recente do histórico, com fallback para o desvio-padrão de todo o histórico e, se ainda inválido, para o maior valor absoluto entre a última observação e a unidade. Para o conjunto de preditores, $\mathcal{G}_i$ inclui a mediana das previsões de teste em cada passo do horizonte, uma banda entre os quantis inferior e superior e uma medida de dispersão entre os modelos, além do último trecho do histórico na mesma escala. A partir de $\mathcal{G}_i$, calculam-se para cada previsão candidata o nível no primeiro passo do horizonte, a inclinação média, a amplitude, a volatilidade, a distância ao último valor observado, a distância à referência sazonal ingênua, a distância à mediana do pool e a fração de pontos fora da banda de previsões do pool.

A referência sazonal ingênua é construída pela repetição do último ciclo do histórico até o horizonte $h$, quando o período sazonal é válido e o histórico contém ao menos um ciclo completo; caso contrário, a distância sazonal não é definida. As características de $\mathcal{G}_i$ permitem comparar a plausibilidade das trajetórias sem consultar o valor real de teste. A representação $\mathcal{F}_i$ é usada durante o ciclo como evidência de forma para o agente escolher pools e ponderações; $\mathcal{G}_i$ é usada após o ciclo pela condição determinística que decide entre o líder de validação e um candidato da shortlist, descrita na Seção~\ref{subsec:decisao-final}. Em nenhum dos dois usos o vetor $\mathbf{y}_i^{(\mathrm{teste})}$ é acessado.
```
