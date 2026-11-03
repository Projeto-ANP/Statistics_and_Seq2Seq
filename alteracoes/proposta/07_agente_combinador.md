# Agente combinador

Substitui a seção `Agente combinador` de `overleaf/proposta.tex` (linhas 409–480). Remove sementes, aceite, estagnação, paciência e o catálogo com contagem fixa; descreve as ações de preparação e avaliação, o ciclo ReAct e o encerramento por orçamento fixo.

```latex
\section{Agente combinador} \label{sec:agente}

Com o contexto da série já construído, o agente entra em ação para decidir a estratégia de combinação a ser empregada. O histórico de tentativas $\mathcal{H}_i$ começa vazio: não há estratégias previamente avaliadas nem piso determinístico inicial. A primeira avaliação realizada pelo agente é a primeira tentativa da série e, portanto, o primeiro líder. Essa escolha impede que o agente ancore em um conjunto fixo de combinações e desloca o ponto de partida para as evidências da própria série.

\subsection{Ações de preparação e de avaliação}

O agente opera sobre um conjunto fechado de ações determinísticas, implementadas como funções convencionais, sem participação de modelo de linguagem em sua execução. O agente interage com elas exclusivamente pelo nome da ação e pelos argumentos fornecidos em sua \textit{Action Input}. Ele nunca escreve valores numéricos de previsão, pesos ou métricas: as ações calculam esses valores, e o agente recebe identificadores e resumos.

As ações de preparação dividem-se em dois grupos. As ações de seleção de pool produzem um subconjunto de $\mathcal{M}$, referenciado por um identificador que pode ser usado como argumento de ações posteriores. As ações de seleção são a seleção dos $k$ preditores de menor erro segundo uma métrica e um conjunto de janelas indicados pelo agente; a seleção dos $k$ preditores mais estáveis, que combina a posição média no \textit{ranking} com sua variação entre janelas; a poda de redundância, que remove preditores com erro fortemente correlacionado e mantém o de menor erro em cada grupo; e a seleção gulosa, que adiciona iterativamente o preditor que mais reduz o erro da média corrente. O valor de $k$, a métrica e as janelas são escolhas do agente, não constantes do método.

As ações de cálculo de pesos produzem um vetor de pesos sobre um pool, também referenciado por um identificador. As ações de pesos são o peso pelo erro inverso, o peso pela tendência do erro ao longo das janelas, o peso pelo melhor preditor em cada passo do horizonte, o peso pela média dos melhores preditores em cada passo e o peso pela contagem de seleções da busca gulosa. A ação de peso por top-k em cada passo do horizonte atribui peso igual aos $k$ preditores de menor erro naquele passo e zero aos demais, produzindo uma matriz de pesos de ordem $|\mathcal{M}_s| \times h$. O agente escolhe $k$ e a métrica. Essa ação permite que preditores diferentes sejam usados em passos diferentes do horizonte, quando o melhor preditor muda ao longo dele.

Uma rota possível, e não obrigatória, combina a seleção dos $k$ melhores preditores, a ponderação pela média dos $k$ melhores em cada passo do horizonte e a avaliação da combinação ponderada. A rota é apresentada ao agente como exemplo de composição de ações, não como receita a ser seguida. O agente pode avaliar diretamente uma média, uma mediana, uma média aparada, uma combinação ponderada por um método de pesos ou um DBA sobre um pool fixo, sem passar por ações de preparação.

A ação de avaliação é a única por meio da qual uma estratégia passa a integrar o histórico $\mathcal{H}_i$. Ela monta a estratégia especificada, executa o procedimento de \textit{backtest} descrito na Seção~\ref{sec:pontuacao} e devolve ao agente a posição da tentativa no histórico, a pontuação obtida e o erro por janela. Os métodos de combinação disponibilizados por essa ação são a média, a mediana, a média aparada, a combinação ponderada e o DBA; toda combinação usa pelo menos dois preditores. O subconjunto e, quando aplicável, os pesos, são informados como argumentos, referenciados por identificador. Se a estratégia já tiver sido avaliada, a ação devolve a tentativa existente e sinaliza que não houve nova tentativa, evitando duplicatas no histórico.

A Tabela~\ref{tab:acoes_carte} reúne as ações efetivas do CARTE, o identificador correspondente na implementação e uma descrição. O conjunto é fechado, mas não é definido por um número fixo de entradas: a Tabela organiza as ações pela função que desempenham.

\begin{table}[H]
\small
\centering
\caption{Ações determinísticas disponíveis no CARTE.}
\label{tab:acoes_carte}
\begin{tabular}{@{}p{3.8cm}p{5.6cm}p{4.6cm}@{}}
\toprule
\textbf{Ação} & \textbf{Identificador no código} & \textbf{Descrição} \\
\midrule
Seleção dos $k$ preditores de menor erro & \texttt{select\_top\_k} & Menor erro segundo a métrica e as janelas indicadas. \\
Seleção dos preditores mais estáveis & \texttt{select\_stable} & Melhor posição média no \textit{ranking}, penalizada pela variação. \\
Poda de redundância & \texttt{prune\_redundant} & Remove preditores de erro correlacionado. \\
Seleção gulosa & \texttt{select\_forward} & Adiciona o preditor que mais reduz o erro da média corrente. \\
Peso pelo erro inverso & \texttt{weights\_inverse\_error} & Pesos proporcionais ao inverso do erro. \\
Peso pela tendência do erro & \texttt{weights\_error\_trend} & Pesos pela direção do erro entre janelas. \\
Peso pelo melhor preditor por passo & \texttt{weights\_best\_per\_horizon} & Peso unitário ao melhor preditor de cada passo. \\
Peso pela média dos $k$ melhores por passo & \texttt{weights\_topk\_per\_horizon} & Peso igual aos $k$ melhores de cada passo. \\
Peso pela busca gulosa & \texttt{weights\_forward} & Pesos pela frequência de seleção na busca gulosa. \\
Avaliação de estratégia & \texttt{evaluate\_strategy} & Monta, executa o \textit{backtest} e registra a tentativa. \\
\bottomrule
\end{tabular}
\end{table}

Além das ações, a avaliação aceita métodos de pesos nomeados. A Tabela~\ref{tab:metodos_pesos} registra a correspondência entre o nome usado no texto e o identificador empregado na implementação. As ações de seleção usam RMSE por padrão e aceitam SMAPE como alternativa. As ações de pesos recebem uma métrica de erro; quando o agente especifica SMAPE, esse valor é usado no ajuste dos pesos. Ações que exigem mais janelas de validação do que as três disponíveis não integram o conjunto oferecido ao agente.

\begin{table}[H]
\small
\centering
\caption{Métodos de pesos aceitos pela avaliação de estratégia.}
\label{tab:metodos_pesos}
\begin{tabular}{@{}p{7.0cm}p{7.0cm}@{}}
\toprule
\textbf{Método} & \textbf{Identificador} \\
\midrule
Erro inverso & \texttt{inverse\_error} \\
Softmax do erro negativo & \texttt{softmax\_neg\_error} \\
Tendência do erro & \texttt{error\_trend} \\
Melhor preditor por passo & \texttt{best\_per\_horizon} \\
Média dos $k$ melhores por passo & \texttt{topk\_per\_horizon} \\
Busca gulosa & \texttt{forward} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Ciclo de raciocínio e ação}

O agente é implementado sobre um modelo de linguagem de peso aberto, executado localmente, operando segundo o paradigma ReAct discutido na Seção~\ref{subsec:react}. A cada iteração, ele recebe um conjunto de informações textuais sobre o estado atual da decisão e deve produzir, em resposta, exatamente três linhas: um \textit{Thought}, em que verbaliza seu raciocínio sobre o que fazer a seguir; uma \textit{Action}, que identifica qual ação deseja invocar; e uma \textit{Action Input}, que especifica os argumentos dessa ação. Nenhum conteúdo além dessas três linhas é aceito como parte da decisão do agente.

O texto fornecido ao agente combina uma parte fixa e uma parte que se atualiza a cada turno. A parte fixa estabelece o papel do agente, o conjunto de ações disponível, o formato de saída exigido, um procedimento de raciocínio em quatro passos, exemplos de tipos distintos de estratégia e diretrizes práticas. Entre as diretrizes estão a orientação de que pesos quase uniformes sobre o pool podem indicar que o problema está na composição do subconjunto, e não no cálculo dos pesos, e a exigência de que cada hipótese seja motivada por uma característica observável da série, como tendência, sazonalidade, estacionariedade, dispersão ou estabilidade do \textit{ranking}.

O prompt organiza o contexto em blocos. A Tabela~\ref{tab:blocos_contexto} registra o nome usado no texto, o rótulo literal do prompt e o conteúdo de cada bloco, indicando também se ele é fixo para a série ou atualizado a cada turno. O bloco de diagnóstico do líder informa a pontuação do líder, a distância para o vice-líder e um veredito de separabilidade estatística entre os dois. O veredito é calculado a partir das pontuações por janela e dos resíduos de validação e é apresentado apenas como informação; ele é evidência para o agente, não uma regra de parada e não participa da decisão final.

\begin{table}[H]
\small
\centering
\caption{Blocos do contexto entregue ao agente a cada turno.}
\label{tab:blocos_contexto}
\begin{tabular}{@{}p{3.4cm}p{3.8cm}p{6.8cm}@{}}
\toprule
\textbf{Bloco} & \textbf{Rótulo no prompt} & \textbf{Conteúdo e atualização} \\
\midrule
Perfil da série & \textsc{Series Profile} & Escala, período, forças de tendência e sazonalidade, estacionariedade, outliers, autocorrelações e entropia. Fixa. \\
Pool de modelos & \textsc{Model Pool} & Tabela de erro, melhor preditor por janela, estabilidade do \textit{ranking} e grupos de erro correlacionado. Fixa. \\
Diagnóstico & \textsc{Diagnosis} & Regime, previsibilidade, indicação de combinação, riscos e narrativa. Fixa. \\
Forma das previsões & \textsc{Forecast Shape} & Forma das previsões de validação e de teste, sem o valor real de teste. Fixa. \\
Histórico de tentativas & \textsc{Attempt History} & Tentativas avaliadas, ordenadas por pontuação, com a justificativa registrada. Variável. \\
Diagnóstico do líder & \textsc{Leader Diagnostics} & Líder, margem para o vice-líder e veredito de separabilidade estatística. Variável. \\
Identificadores & \textsc{Handles} & Pools e pesos criados durante a execução. Variável. \\
Métodos de pesos & \textsc{Weight Methods} & Métodos de pesos disponíveis para a avaliação. Fixa. \\
Combinações não tentadas & \textsc{Next Untried} & Combinações ainda não avaliadas e eventual rota de preparação. Variável. \\
Resumo das ações & \textsc{Scratchpad} & Resumo das últimas ações e observações. Variável. \\
Última observação & \textsc{Last Observation} & Última observação completa devolvida por uma ação. Variável. \\
Cartão de decisão & \textsc{Series Decision Card} & Características da série que devem motivar o próximo teste. Fixa. \\
Orçamento & \textsc{Budget} & Avaliações e turnos restantes. Variável. \\
\bottomrule
\end{tabular}
\end{table}

A resposta do agente passa por uma etapa de validação antes de qualquer ação ser executada. Caso o nome da ação não exista no conjunto disponível, ou caso os argumentos fornecidos não sejam válidos para a ação solicitada, uma mensagem de erro estruturada é devolvida ao agente, que pode corrigir sua próxima resposta a partir dela. Uma resposta que viola o formato de três linhas é tratada como decisão mal formada, consome um turno e devolve um lembrete do formato exigido. Falhas de comunicação com o modelo, como resposta vazia ou erro transitório de conexão, não representam uma decisão do agente e são reexecutadas sem consumir o orçamento de avaliações.

O ciclo utiliza um orçamento fixo de avaliações e de turnos. As ações de preparação não consomem o orçamento de avaliações, mas consomem turnos; a avaliação de uma estratégia consome uma avaliação. O agente utiliza todo o orçamento disponível para testar hipóteses, e o ciclo termina quando ele se esgota.

\subsection{Encerramento do ciclo e líder de validação}

Ao final do orçamento, o histórico $\mathcal{H}_i$ contém todas as estratégias avaliadas na série, ordenadas pela pontuação da Equação~\eqref{eq:score}. O líder de validação $\ell_i$ é a tentativa de menor pontuação,
\[
    \ell_i = \arg\min_{s \in \mathcal{H}_i} \text{score}_i(s).
\]
Como o histórico começa vazio, o líder é sempre uma estratégia construída e avaliada pelo agente. Todas as tentativas permanecem registradas, com sua especificação, pontuação, erro por janela e justificativa, para fins de auditoria.

O líder $\ell_i$ é o ponto de partida da decisão final, mas não é necessariamente a estratégia aplicada. A Seção~\ref{subsec:decisao-final} descreve a construção da shortlist por família e a condição sazonal determinística que pode substituir o líder por um candidato mais próximo da referência sazonal. Essa etapa é externa ao ciclo, não utiliza modelo de linguagem e não acessa os valores reais da janela de teste.
```
