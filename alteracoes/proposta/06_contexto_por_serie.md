# Contexto por série

Substitui a seção `Caracterização e diagnóstico por série` de `overleaf/proposta.tex` (linhas 367–408). A seção passa a reunir o perfil, a avaliação do pool, o diagnóstico e a representação da forma das previsões, todos específicos da série.

```latex
\section{Contexto por série} \label{sec:diagnostico}

Antes de o agente propor qualquer estratégia de combinação para a série $i$, o CARTE constrói um contexto específico dessa série, sem participação de modelo de linguagem. Esse contexto reúne quatro representações complementares: o perfil $\mathbf{z}_i$ do histórico $\mathcal{S}_i$; a avaliação $\mathcal{E}_i$ do desempenho dos preditores nas janelas de validação; o diagnóstico interpretativo $\mathcal{C}_i$; e a representação $\mathcal{F}_i$ da forma das previsões. As quatro são calculadas uma única vez por série, antes do ciclo de decisão, e entregues ao agente a cada turno. No contexto do turno, elas correspondem ao perfil da série, ao pool de modelos, ao diagnóstico e à forma das previsões, cujos rótulos literais no prompt são registrados na Tabela~\ref{tab:blocos_contexto}. Nenhuma delas utiliza os valores reais da janela de teste.

\subsection{Perfil da série}

O perfil $\mathbf{z}_i$ é calculado apenas sobre o trecho do histórico $\mathcal{S}_i$ anterior à janela de teste. Ele descreve a escala, a estrutura temporal e o comportamento estatístico da série, e serve de insumo para a leitura do agente, para o diagnóstico interpretativo e para a condição sazonal da decisão final. O perfil reúne o número de observações $n_i$, a frequência de amostragem, o horizonte $h$, o período sazonal e sua origem, o número de janelas de validação e o número de preditores disponíveis.

O período sazonal $p_i$ é resolvido em três níveis: um valor explicitamente definido para o conjunto de dados; a frequência declarada da série; e, na ausência de ambos, o horizonte de previsão, limitado a $12$ observações. O perfil registra o período adotado, sua origem e se o histórico contém pelo menos dois ciclos completos.

A decomposição da série em tendência, sazonalidade e resíduo é feita por STL robusto quando o histórico contém pelo menos dois ciclos completos do período sazonal; caso contrário, o perfil recorre a uma tendência linear e registra força sazonal nula. A partir dessa decomposição são calculadas a força de tendência e a força de sazonalidade, ambas normalizadas no intervalo $[0,1]$, além da direção da tendência e da mudança relativa entre o início e o fim do histórico. Também compõem o perfil o veredito de estacionariedade obtido por testes ADF e KPSS, o percentual de observações atípicas identificadas por escore $z$, a autocorrelação de primeira ordem, a autocorrelação no período sazonal, a entropia espectral, o expoente de Hurst, o coeficiente de variação e um indicador de cruzamentos por zero.

O perfil completo, incluindo as características descritivas calculadas para a série, é registrado para auditoria. O agente recebe uma versão resumida, restrita aos campos que informam uma decisão de combinação: escala, período, forças de tendência e sazonalidade, estacionariedade, outliers, autocorrelações, entropia, coeficiente de variação e cruzamentos por zero. A versão resumida evita que o contexto do turno seja dominado por dezenas de números que não se traduzem em uma escolha de pool ou de ponderação.

\subsection{Avaliação do conjunto de preditores}

A avaliação $\mathcal{E}_i$ mede o desempenho de cada preditor $M_j \in \mathcal{M}$ isoladamente, e não o de uma combinação. Para cada preditor e cada janela de validação $w$, calcula-se o RMSE na escala original da série,
\begin{equation}
    \text{RMSE}_{i,j}^{(w)}
    =
    \sqrt{\frac{1}{h} \sum_{k=1}^{h}
        \left(y_{i,k}^{(w)} - \hat{y}_{i,j,k}^{(w)}\right)^2},
\end{equation}
em que $y_{i,k}^{(w)}$ e $\hat{y}_{i,j,k}^{(w)}$ são, respectivamente, a componente $k$ do vetor de valores reais $\mathbf{y}_i^{(w)}$ e do vetor de previsões $\hat{\mathbf{y}}_{i,j}^{(w)}$, definidos na Seção~\ref{sec:formulacao}. O RMSE é a métrica padrão da tabela de erro do pool, do melhor preditor por janela e da estabilidade do ranking; ele também é o padrão das ações de seleção e de parte das ações de pesos. A pontuação que ordena as tentativas de combinação é o SMAPE, definido na Seção~\ref{sec:pontuacao}; o RMSE não entra nessa pontuação nem nas métricas finais de teste, que são SMAPE e POCID.

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

O regime da série é determinado pelas forças de tendência e de sazonalidade. Quando as duas são altas, o regime é misto; quando apenas a sazonalidade é alta, o regime é dominado por sazonalidade; quando apenas a tendência é alta, o regime é dominado por tendência; e quando nenhuma das duas é alta, o regime é ruidoso.

A previsibilidade é determinada pela entropia espectral. Entropia alta indica previsibilidade baixa, entropia baixa indica previsibilidade alta e valores intermediários, assim como a ausência da característica, indicam previsibilidade média.

A indicação de combinação nasce do cruzamento entre a estabilidade do \textit{ranking}, medida por $\tau$, e a dispersão relativa de erro entre os preditores. Quando o \textit{ranking} é pouco estável, a indicação é de combinação robusta, menos sensível à escolha exata de pesos; quando a dispersão é alta e o \textit{ranking} é razoavelmente estável, a indicação é de combinação seletiva, restrita a um subconjunto; quando apenas a dispersão é alta, a indicação é de combinação ponderada; fora desses casos, a indicação é de uso do conjunto completo, sem seleção ou ponderação adicional.

Por fim, o diagnóstico reúne uma lista de riscos, construída a partir de condições independentes: instabilidade do \textit{ranking} entre janelas; existência de algum grupo de preditores com erro correlacionado; previsibilidade baixa, sob o risco de que pesos ajustados a uma série próxima do ruído resultem em sobreajuste; e a limitação, presente em toda série avaliada por este método, de haver apenas três janelas de validação para fundamentar qualquer estimativa. As categorias e os riscos são sintetizados em uma narrativa textual curta, entregue ao agente como leitura qualitativa e não como regra.

\subsection{Forma das previsões}

A representação $\mathcal{F}_i$ descreve a forma das previsões sem revelar o alvo. Ela reúne os $h$ valores reais da janela de validação mais recente e as previsões dos $M$ preditores para a janela de teste, ambos normalizados por uma escala comum. A escala é o desvio-padrão do trecho recente do histórico, com fallback para o desvio-padrão de todo o histórico e, se ainda inválido, para o maior valor absoluto entre a última observação e a unidade.

Para o conjunto de preditores, $\mathcal{F}_i$ inclui a mediana das previsões de teste em cada passo do horizonte, uma banda entre os quantis inferior e superior, uma medida de dispersão entre os modelos e agrupamentos de preditores cujas previsões de teste são fortemente correlacionadas. O agente recebe os vetores detalhados dos preditores de melhor desempenho na última janela de validação e de um representante de cada grupo correlacionado, além das correlações par a par entre esses preditores.

Para cada previsão candidata, $\mathcal{F}_i$ também resume características de forma: nível no primeiro passo do horizonte, inclinação média, amplitude, volatilidade, distância ao último valor observado, distância à referência sazonal ingênua, distância à mediana do pool e fração de pontos fora da banda de previsões do pool. Essas características permitem comparar a plausibilidade das trajetórias sem consultar o valor real de teste. A referência sazonal ingênua é construída pela repetição do último ciclo do histórico até o horizonte $h$, quando o período sazonal é válido e o histórico contém ao menos um ciclo completo; caso contrário, a distância sazonal não é definida.

A representação $\mathcal{F}_i$ é usada de duas formas. Durante o ciclo, ela é evidência de forma para o agente escolher pools e ponderações. Após o ciclo, ela fornece as distâncias sazonais empregadas pela condição determinística que decide entre o líder de validação e um candidato da shortlist, descrita na Seção~\ref{subsec:decisao-final}. Em nenhum dos dois usos o vetor $\mathbf{y}_i^{(\mathrm{teste})}$ é acessado.
```
