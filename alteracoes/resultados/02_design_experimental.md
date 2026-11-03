# Design experimental

Substitui a seção `Design experimental` de `overleaf/resultados.tex` (linhas 7–13).

```latex
\section{Design experimental} \label{sec:design-experimental}

O conjunto de preditores individuais utilizado em todos os experimentos é composto por dezenove modelos: três estatísticos (ARIMA, ETS e Theta), dois regressores de aprendizagem de máquina treinados sobre a representação original dos dados (Random Forest e CatBoost), doze variantes desses mesmos dois regressores treinadas sobre representações obtidas por transformação \textit{wavelet} contínua, \textit{wavelet} discreta e de Fourier, tanto de forma concatenada à representação original quanto isoladamente, e dois métodos ingênuos (Naive Seasonal e Naive Moving Average). Cada um desses métodos foi apresentado individualmente no Capítulo~\ref{chp:background}, e sua combinação no conjunto $\mathcal{M}$ foi formalizada na Seção~\ref{sec:formulacao} do capítulo de método.

Os experimentos foram conduzidos sobre sete conjuntos de dados de domínios e frequências de amostragem distintos, resumidos na Tabela~\ref{tab:resultados-datasets}: a demanda de combustíveis fornecida pela ANP, obtida por meio da biblioteca StreamFuels, desenvolvida no âmbito desta dissertação \citep{streamfuels}; os conjuntos NN5 e M4 do repositório \textit{Monash Forecasting}, ambos em frequência semanal; e os quatro subconjuntos do \textit{Electricity Transformer Temperature} (ETTh1, ETTh2, ETTm1 e ETTm2).

\begin{table}[H]
\centering
\caption{Conjuntos de dados utilizados nos experimentos.}
\label{tab:resultados-datasets}
\begin{tabular}{@{}l l r r@{}}
\toprule
\textbf{Conjunto} & \textbf{Domínio e frequência} & \textbf{Séries} & \textbf{Horizonte} \\
\midrule
ANP  & Combustíveis; mensal & 182 & 12 \\
NN5  & Demanda; semanal & 111 & 8 \\
M4   & Séries diversas; semanal & 359 & 13 \\
ETTh1 & Eletricidade; horária & 7 & 24 \\
ETTh2 & Eletricidade; horária & 7 & 24 \\
ETTm1 & Eletricidade; 15 minutos & 7 & 24 \\
ETTm2 & Eletricidade; 15 minutos & 7 & 24 \\
\midrule
Total & & 680 & \\
\bottomrule
\end{tabular}
\end{table}

Em todos os casos, o protocolo de avaliação segue o descrito na Seção~\ref{sec:formulacao}: cada série é dividida em um conjunto de treino, três janelas de validação e uma janela de teste posterior. As três janelas de validação são usadas tanto para a escolha de hiperparâmetros dos modelos individuais quanto para a decisão de combinação do CARTE. A janela de teste é reservada exclusivamente para a avaliação final e não é observada em nenhuma etapa de decisão: os valores reais de teste só são lidos depois que a previsão combinada já foi produzida.

As métricas reportadas são o SMAPE e o POCID, definidos na Seção~\ref{sec:metricas}. O SMAPE é calculado entre a previsão combinada e os valores reais da janela de teste; o POCID é calculado dentro dessa mesma janela, comparando a direção das variações previstas e observadas. O SMAPE é a métrica de erro usada para ordenar as estratégias na validação e para comparar os métodos neste capítulo; o POCID é analisado em separado e não participa desse ranking. As comparações entre métodos são feitas por série, com testes pareados quando indicado. Além das comparações por série, os resultados são agregados de duas formas: a média dos sete conjuntos de dados, que atribui o mesmo peso a cada conjunto e é a referência principal deste capítulo, e o agregado das 680 séries, que atribui o mesmo peso a cada série e é reportado como complemento. O agregado é influenciado pelo tamanho dos conjuntos, especialmente por M4 e ANP, e por isso não substitui a média dos sete.

Os experimentos foram realizados em uma máquina equipada com duas GPUs NVIDIA GeForce RTX 5090, com 32 GB de memória VRAM cada, e um processador AMD Ryzen 9 7950X, com 16 núcleos e 32 threads. Os tempos de decisão do agente incluem as chamadas ao modelo de linguagem e a execução das ações determinísticas, e são reportados apenas como medida de custo computacional.
```
