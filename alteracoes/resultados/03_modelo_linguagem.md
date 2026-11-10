# Escolha do modelo de linguagem e seção de execuções

Mantém a seção `Escolha do modelo de linguagem do agente` exatamente como está em `overleaf/resultados.tex` (linhas 48–110), sem a subseção `Modelo escolhido`, que foi removida na revisão. Acrescenta apenas a frase sobre a execução única por modelo e a nova seção `Execuções, sementes e agregação dos resultados`, que deve entrar entre a escolha do modelo e a seção da ANP.

```latex
\section{Escolha do modelo de linguagem do agente} \label{sec:escolha-llm}

Antes de apresentar os resultados por conjunto de dados, é necessário justificar a escolha do modelo de linguagem empregado no papel de agente combinador ao longo do restante deste capítulo. Três modelos de linguagem de peso aberto foram comparados, sendo avaliado tanto o desempenho preditivo quanto o custo computacional. Todos foram executados no mesmo protocolo experimental, com as configurações de raciocínio adotadas para cada modelo. A comparação usa uma execução de cada modelo.

\subsection{Desempenho preditivo}

A Tabela~\ref{tab:escolha-llm-desempenho} compara o SMAPE e o POCID médios do CARTE executado com cada um dos três modelos, por conjunto de dados, e resume a média dos sete conjuntos e o agregado das 680 séries.

\begin{table}[H]
\small
\centering
\caption{Desempenho médio do CARTE com os três modelos de linguagem, por conjunto de dados e agregados.}
\label{tab:escolha-llm-desempenho}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{\textbf{SMAPE (\%)}} & \multicolumn{3}{c}{\textbf{POCID (\%)}} \\
\textbf{Conjunto de dados} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} \\
\midrule
ANP    & 21,93 & 21,88 & 22,36 & 63,44 & 64,04 & 64,09 \\
NN5    & 11,96 & 12,01 & 11,85 & 47,36 & 45,43 & 44,27 \\
M4     & 9,16  & 9,24  & 9,23  & 57,73 & 57,40 & 58,17 \\
ETTh1  & 22,36 & 22,05 & 22,36 & 62,73 & 62,11 & 62,11 \\
ETTh2  & 17,31 & 17,74 & 17,83 & 54,66 & 50,93 & 49,07 \\
ETTm1  & 43,03 & 39,35 & 38,05 & 55,90 & 54,66 & 51,55 \\
ETTm2  & 16,42 & 17,16 & 17,25 & 39,13 & 37,89 & 39,75 \\
\midrule
Média dos 7 conjuntos & 20,31 & 19,92 & 19,85 & 54,42 & 53,21 & 52,72 \\
Agregado (680 séries) & 13,67 & 13,69 & 13,77 & 57,37 & 56,98 & 57,17 \\
\bottomrule
\end{tabular}
\end{table}

Nota-se a partir dos resultados, que não há um modelo vencedor em todos os conjuntos. O Qwen3.8 alcançou os menores valores de SMAPE nos conjuntos ANP e ETTh1; o Gemma4 é melhor no NN5 e no ETTm1; o gpt-oss é melhor no M4, no ETTh2 e no ETTm2. Na média dos sete conjuntos, o Gemma4 obtém o menor SMAPE (19,85\%), seguido do Qwen3.8 (19,92\%) e do gpt-oss (20,31\%). A vantagem do Gemma4 sobre o Qwen3.8 é de apenas 0,07 ponto percentual, e sobre o gpt-oss, de 0,46 ponto.

Quando a avaliação considera o agregados de todas as séries, o quadro se altera: o gpt-oss obtém o melhor SMAPE  (13,67\%), contra 13,69\% do Qwen3.8 e 13,77\% do Gemma4. Em POCID, o gpt-oss é o melhor nas duas agregações: 54,42\% na média dos sete conjuntos e 57,37\% no agregado das séries, contra 53,21\% e 56,98\% do Qwen3.8 e 52,72\% e 57,17\% do Gemma4. Os resultados mostram que Qwen3.8 e Gemma4 podem superar o gpt-oss em conjuntos específicos e até na média dos sete em SMAPE, mas as diferenças são pequenas, heterogêneas entre conjuntos e não se sustentam nas duas agregações nem no POCID.

\subsection{Custo computacional}

A Tabela~\ref{tab:escolha-llm-custo} compara o tempo médio de decisão por série entre os três modelos, por conjunto de dados.

\begin{table}[H]
\small
\centering
\caption{Tempo médio de decisão por série (segundos), por modelo e conjunto de dados.}
\label{tab:escolha-llm-custo}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Conjunto} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} \\
\midrule
ANP    & 25,52 $\pm$ 3,90   & 561,92 $\pm$ 89,62   & 451,06 $\pm$ 112,81 \\
NN5    & 24,48 $\pm$ 3,57   & 523,15 $\pm$ 79,76   & 436,65 $\pm$ 117,43 \\
M4     & 25,35 $\pm$ 3,57   & 575,29 $\pm$ 100,06  & 502,56 $\pm$ 115,90 \\
ETTh1  & 26,39 $\pm$ 2,83   & 659,85 $\pm$ 126,82  & 523,02 $\pm$ 176,45 \\
ETTh2  & 25,70 $\pm$ 2,92   & 644,46 $\pm$ 115,49  & 532,94 $\pm$ 106,70 \\
ETTm1  & 25,17 $\pm$ 3,52   & 596,21 $\pm$ 67,71   & 527,21 $\pm$ 88,68 \\
ETTm2  & 19,55 $\pm$ 2,69   & 620,50 $\pm$ 81,50   & 589,32 $\pm$ 147,43 \\
\midrule
Média por série (680 séries) & \textbf{25,20} & 565,46 & 479,69 \\
\bottomrule
\end{tabular}
\end{table}

Para a escolha do modelo de linguagem do agente, o custo computacional é um fator decisivo devido às diferenças significativas entre os modelos avaliados. O gpt-oss leva, em média, 25,20 segundos por série; o Qwen3.8 leva 565,46 segundos, cerca de 22,4 vezes mais, e o Gemma4 leva 479,69 segundos, cerca de 19,0 vezes mais. Em tempo serial, os 680 séries custariam aproximadamente 4,8 horas com o gpt-oss, 106,8 horas com o Qwen3.8 e 90,6 horas com o Gemma4. Desse modo, o gpt-oss foi selecionado, visto que os valores de SMAPE obtidos com Gemma4 e Qwen3.8 são inferiores a 1 ponto percentual e não justificam um custo computacional aproximadamente 20 vezes maior.

\section{Execuções, sementes e agregação dos resultados} \label{sec:execucoes-seeds}

Os resultados principais do CARTE vêm de dez execuções independentes do agente, com as sementes 11 a 20, realizadas depois da escolha do modelo. Todas usam o mesmo modelo, o mesmo protocolo de preparação, o mesmo conjunto de preditores e a mesma configuração de prompts. A semente altera a sequência de amostragem do modelo de linguagem e, com ela, as decisões do agente. As execuções foram realizadas no mesmo servidor e no mesmo período, o que mantém as mudanças de ambiente sob controle e permite atribuir a variação observada principalmente à semente. A execução usada na Seção~\ref{sec:escolha-llm} serviu apenas para escolher o modelo e não faz parte das dez execuções. Os resultados por conjunto de dados usam a média dessas dez execuções. Por isso, os valores da Tabela~\ref{tab:escolha-llm-desempenho} diferem ligeiramente dos agregados apresentados a seguir.

A Tabela~\ref{tab:execucoes-agregados} apresenta os agregados de cada execução. Para cada uma, o SMAPE e o POCID foram calculados por série e agregados da mesma forma que no restante do capítulo. A linha final mostra a média e o desvio-padrão entre as dez execuções.

\begin{table}[H]
\small
\centering
\caption{Agregados de SMAPE e POCID por execução do CARTE. Valores em percentual.}
\label{tab:execucoes-agregados}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Execução} & \textbf{Média 7 SMAPE} & \textbf{Agregado 680 SMAPE} & \textbf{Média 7 POCID} & \textbf{Agregado 680 POCID} \\
\midrule
11 & 20,33 & 13,87 & 53,99 & 57,22 \\
12 & 19,47 & 13,69 & 54,30 & 57,07 \\
13 & 20,94 & 13,79 & 52,95 & 56,86 \\
14 & 20,48 & 13,72 & 53,03 & 57,01 \\
15 & 20,23 & 13,82 & 54,28 & 57,42 \\
16 & 20,28 & 13,58 & 54,62 & 57,67 \\
17 & 20,20 & 13,88 & 55,05 & 57,48 \\
18 & 19,36 & 13,63 & 53,54 & 56,85 \\
19 & 21,59 & 13,85 & 54,43 & 57,11 \\
20 & 19,84 & 13,94 & 54,34 & 57,16 \\
\midrule
\textbf{Média} & \textbf{20,27 $\pm$ 0,66} & \textbf{13,78 $\pm$ 0,12} & \textbf{54,05 $\pm$ 0,68} & \textbf{57,19 $\pm$ 0,27} \\
\bottomrule
\end{tabular}
\end{table}

Na média dos sete conjuntos, o SMAPE médio das dez execuções é 20,27\%, com desvio-padrão de 0,66 ponto percentual. O agregado das 680 séries é 13,78\%, com desvio-padrão de 0,12 ponto. Em POCID, a média dos sete é 54,05\%, com desvio-padrão de 0,68 ponto, e o agregado é 57,19\%, com desvio-padrão de 0,27 ponto. As execuções individuais variam entre 19,36\% e 21,59\% na média dos sete em SMAPE e entre 13,58\% e 13,94\% no agregado. Em POCID, a média dos sete varia entre 52,95\% e 55,05\%, e o agregado entre 56,85\% e 57,67\%. O agregado é mais estável que a média dos sete porque é dominado pelos conjuntos com muitas séries, enquanto a média dos sete dá o mesmo peso a cada conjunto.

A dispersão entre execuções é desigual entre os conjuntos. Nos seis conjuntos que não o ETTm1, o desvio-padrão do SMAPE fica entre 0,09 e 0,63 ponto percentual. No ETTm1, sobe para 4,35 pontos, com valores entre 33,81\% e 48,91\%. A variação do ETTm1 se concentra em poucas séries de erro alto, que dominam a média do conjunto. Como o ETTm1 tem apenas sete das 680 séries, seu efeito no agregado é pequeno. O efeito na média dos sete é maior, porque o conjunto recebe o mesmo peso que os demais.

A análise do comportamento do agente, apresentada na Seção~\ref{sec:resultados-agente}, usa a execução com semente 11 dessa série. Essa execução foi escolhida por estar próxima da média das dez nas quatro agregações. Ela apresenta 20,33\% na média dos sete, 13,87\% no agregado, 53,99\% na média dos sete em POCID e 57,22\% no agregado em POCID, valores próximos das médias de 20,27\%, 13,78\%, 54,05\% e 57,19\%. A execução foi escolhida por proximidade com a média das dez, não por ser a melhor. Os traces, as previsões e os estudos de caso dessa seção vêm dessa execução e ilustram uma trajetória representativa. Eles não correspondem à execução média.
```
