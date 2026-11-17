# Estudo de ablação - CARTE-oráculo

Continua a seção `Estudo de ablação: a seleção pelo agente` com a subseção `CARTE-oráculo`.

```latex
\subsection{CARTE-oráculo} \label{subsec:carte-oraculo}

O CARTE-oráculo é o melhor cenário possível para o agente com acesso à janela de teste. Ele não é um método, porque usa informação que o CARTE não tem no momento da escolha. Para separar o que o agente poderia alcançar sem novas propostas do que o catálogo oferece, a análise distingue três versões. O CARTE-oráculo restrito escolhe a melhor entre as tentativas do agente em cada execução. O CARTE-oráculo escolhe a melhor entre as estratégias de $U$. O CARTE-oráculo de validação escolhe a melhor pela pontuação de validação e não usa a janela de teste. O melhor individual é o menor SMAPE entre os dezenove preditores em cada série. Todos, exceto o de validação, usam a janela de teste.

O CARTE-oráculo restrito é calculado execução a execução, e o valor reportado é a média das dez execuções. A coluna 10 primeiras restringe o conjunto às tentativas a1 a a10 e reporta o melhor resultado entre elas. O CARTE-oráculo e o CARTE-oráculo de validação são calculados sobre $U$. O melhor individual é escolhido em cada série.

$U$ é uma grade fixa de estratégias. Ela contém o pool completo, as seleções por estabilidade, as seleções pelos melhores, a seleção forward, a média, a mediana, a média aparada, o DBA e as médias ponderadas, com as grades de \textit{trim} e \textit{shrinkage}. No agregado, $U$ tem em média 226 estratégias por série. Na média dos sete, são 215. Em ANP, NN5 e M4, chega a 228. Nos ETT, fica em 216 ou menos porque a ação de peso pelo meta-modelo não é treinada com sete séries. O CARTE-oráculo é um limite dentro de $U$. Como a grade não esgota todos os parâmetros, o limite verdadeiro pode ser menor.

A Tabela~\ref{tab:ablacao-carte-oraculo-smape} apresenta o SMAPE médio do CARTE e dos oráculos. Em negrito, o menor valor da linha.

\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption[SMAPE do CARTE e dos oráculos.]{SMAPE médio (\%) do CARTE e dos oráculos, por conjunto de dados. Em negrito, o menor valor da linha. Restrito, 10 primeiras, Validação, Oráculo e Individual são abreviações de CARTE-oráculo restrito, melhor das 10 primeiras, CARTE-oráculo de validação, CARTE-oráculo e melhor individual.}
\label{tab:ablacao-carte-oraculo-smape}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{CARTE} & \textbf{Restrito} & \textbf{10 primeiras} & \textbf{Validação} & \textbf{Oráculo} & \textbf{Individual} \\
\midrule
ANP    & 22,24 & 20,42 & 20,59 & 22,27 & 18,99 & \textbf{17,36} \\
NN5    & 12,04 & 10,54 & 10,64 & 12,03 & 9,66 & \textbf{9,33} \\
M4     & 9,18 & 6,57 & 6,67 & 9,57 & 5,66 & \textbf{5,57} \\
ETTh1  & 22,72 & 21,80 & 21,91 & 25,06 & 20,60 & \textbf{19,61} \\
ETTh2  & 17,67 & 16,78 & 16,86 & 17,66 & 16,22 & \textbf{15,54} \\
ETTm1  & 41,03 & 38,55 & 38,92 & 49,47 & \textbf{37,80} & 38,60 \\
ETTm2  & 17,02 & 15,82 & 16,52 & 17,29 & \textbf{12,83} & 16,85 \\
\midrule
Média dos 7 & 20,27 & 18,64 & 18,87 & 21,91 & \textbf{17,39} & 17,55 \\
Agregado (680 séries) & 13,78 & 11,61 & 11,74 & 14,11 & 10,55 & \textbf{10,04} \\
\bottomrule
\end{tabular}
\end{table}

O CARTE-oráculo restrito fica em 18,64\% na média dos sete e 11,61\% no agregado. O melhor das 10 primeiras fica em 18,87\% e 11,74\%. O CARTE-oráculo fica em 17,39\% e 10,55\%. O melhor individual fica em 17,55\% e 10,04\%. O CARTE fica 1,63 e 2,17 pontos percentuais acima do CARTE-oráculo restrito, 1,40 e 2,04 acima do melhor das 10 primeiras, 2,88 e 3,23 acima do CARTE-oráculo, e 2,72 e 3,74 acima do melhor individual. A distância para o CARTE-oráculo restrito e para o melhor das 10 primeiras é menor que a distância para o CARTE-oráculo e para o melhor individual. A seleção do agente é eficiente porque aproveita bem o histórico que ele próprio construiu.

A diferença entre o CARTE-oráculo restrito e o melhor das 10 primeiras é de 0,23 e 0,13 ponto percentual. Ela é menor que a distância entre o CARTE e o melhor das 10 primeiras, de 1,40 e 2,04. O gargalo está na seleção, não na quantidade de propostas.

O CARTE-oráculo de validação fica em 21,91\% e 14,11\%, pior que o CARTE em 1,64 e 0,33 ponto percentual. Em ETTh1 e ETTm1, a deterioração é grande, de 22,72\% para 25,06\% e de 41,03\% para 49,47\%. Com cerca de 226 estratégias e três janelas, o máximo da validação é uma estatística extrema. Parte do ganho é ruído das janelas e não se repete no teste. Esse é o \textit{winner's curse}. O agente avalia até doze propostas, enquanto o CARTE-oráculo de validação escolhe entre 226. A seleção do agente é mais conservadora, e é mais eficiente que a maximização da validação em $U$.

O CARTE-oráculo fica em 17,39\% e 10,55\%, 2,88 e 3,23 pontos percentuais abaixo do CARTE. É o espaço que resta, mas é um limite que usa a janela de teste. O CARTE-oráculo de validação é pior que o CARTE, o que mostra que o agente não se limita a escolher o máximo da validação em um conjunto grande. Ele usa o diagnóstico da série e o histórico de tentativas para propor e selecionar.

A Tabela~\ref{tab:ablacao-carte-oraculo-pocid} repete a estrutura para o POCID. Nessa tabela, o melhor individual é o modelo de maior POCID entre os dezenove preditores em cada série.

\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption[POCID do CARTE e dos oráculos.]{POCID médio (\%) do CARTE e dos oráculos, por conjunto de dados. Em negrito, o maior valor da linha. Restrito, 10 primeiras, Validação, Oráculo e Individual são abreviações de CARTE-oráculo restrito, melhor das 10 primeiras, CARTE-oráculo de validação, CARTE-oráculo e melhor individual. O melhor individual é o modelo de maior POCID entre os dezenove preditores em cada série.}
\label{tab:ablacao-carte-oraculo-pocid}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{CARTE} & \textbf{Restrito} & \textbf{10 primeiras} & \textbf{Validação} & \textbf{Oráculo} & \textbf{Individual} \\
\midrule
ANP    & 63,28 & 74,52 & 73,63 & 64,64 & \textbf{82,07} & 80,82 \\
NN5    & 46,04 & 65,82 & 64,29 & 45,30 & \textbf{77,22} & 76,32 \\
M4     & 57,88 & 72,79 & 72,14 & 56,15 & \textbf{79,34} & 78,69 \\
ETTh1  & 62,11 & 70,43 & 70,43 & 61,49 & \textbf{73,91} & 71,43 \\
ETTh2  & 54,22 & 64,22 & 63,79 & 58,39 & \textbf{68,94} & 64,60 \\
ETTm1  & 54,47 & 67,33 & 66,71 & 54,04 & \textbf{73,29} & 70,81 \\
ETTm2  & 40,37 & 50,06 & 49,69 & 41,61 & \textbf{54,66} & \textbf{54,66} \\
\midrule
Média dos 7 & 54,05 & 66,45 & 65,81 & 54,52 & \textbf{72,78} & 71,05 \\
Agregado (680 séries) & 57,18 & 71,71 & 70,87 & 56,56 & \textbf{79,25} & 78,32 \\
\bottomrule
\end{tabular}
\end{table}

No POCID, o CARTE-oráculo fica em 72,78\% e 79,25\%, o CARTE-oráculo restrito em 66,45\% e 71,71\%, e o melhor das 10 primeiras em 65,81\% e 70,87\%. O melhor individual por POCID máximo fica em 71,05\% e 78,32\%. O CARTE-oráculo de validação fica em 54,52\% e 56,56\%, praticamente o CARTE, que fica em 54,05\% e 57,18\%. O CARTE está 18,72 e 22,06 pontos percentuais abaixo do CARTE-oráculo, 12,40 e 14,53 abaixo do CARTE-oráculo restrito, e 11,76 e 13,68 abaixo do melhor das 10 primeiras. No POCID, o CARTE-oráculo de validação não agrega. Ele fica 0,46 ponto acima do CARTE na média dos sete e 0,63 abaixo no agregado.

No POCID, o CARTE-oráculo supera o melhor individual em seis dos sete conjuntos e empata no ETTm2. O POCID tem muitos empates e não participa da seleção. A leitura principal é o SMAPE. O POCID confirma que o espaço para melhorar é grande, mas não indica um caminho de seleção.

As Tabelas~\ref{tab:ablacao-carte-oraculo-ganho-smape} e~\ref{tab:ablacao-carte-oraculo-ganho-pocid} resumem o ganho dos oráculos sobre o CARTE e as contagens pareadas. Valores positivos indicam que o oráculo é melhor.

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption[Ganho dos oráculos sobre o CARTE em SMAPE.]{Ganho médio dos oráculos sobre o CARTE em SMAPE, em pontos percentuais, na média dos sete conjuntos e no agregado das 680 séries, e número de séries em que o oráculo vence, empata ou perde (O/E/C). Restrito, 10 primeiras, Validação, Oráculo e Individual seguem a Tabela~\ref{tab:ablacao-carte-oraculo-smape}. Valores positivos indicam que o oráculo é melhor.}
\label{tab:ablacao-carte-oraculo-ganho-smape}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Oráculo} & \textbf{Média 7} & \textbf{680} & \textbf{O/E/C} \\
\midrule
Restrito      & +1,63 & +2,17 & 672/8/0 \\
10 primeiras  & +1,40 & +2,04 & 666/10/4 \\
Validação     & $-$1,64 & $-$0,33 & 308/27/345 \\
Oráculo       & +2,88 & +3,23 & 679/1/0 \\
Individual    & +2,72 & +3,74 & 659/0/21 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption[Ganho dos oráculos sobre o CARTE em POCID.]{Ganho médio dos oráculos sobre o CARTE em POCID, em pontos percentuais, na média dos sete conjuntos e no agregado das 680 séries, e número de séries em que o oráculo vence, empata ou perde (O/E/C). Restrito, 10 primeiras, Validação, Oráculo e Individual seguem a Tabela~\ref{tab:ablacao-carte-oraculo-smape}. Valores positivos indicam que o oráculo é melhor.}
\label{tab:ablacao-carte-oraculo-ganho-pocid}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Oráculo} & \textbf{Média 7} & \textbf{680} & \textbf{O/E/C} \\
\midrule
Restrito      & +12,40 & +14,53 & 648/32/0 \\
10 primeiras  & +11,76 & +13,68 & 639/36/5 \\
Validação     & +0,46 & $-$0,63 & 281/106/293 \\
Oráculo       & +18,72 & +22,06 & 662/17/1 \\
Individual    & +16,99 & +21,14 & 653/22/5 \\
\bottomrule
\end{tabular}
\end{table}

O CARTE-oráculo supera o CARTE em 679 das 680 séries no SMAPE e em 662 no POCID. O CARTE-oráculo restrito supera em 672 e 648. O melhor das 10 primeiras supera em 666 e 639. O melhor individual supera em 659 e 653. O CARTE-oráculo de validação supera o CARTE em 308 séries no SMAPE e 281 no POCID, empata em 27 e 106, e perde em 345 e 293. A contagem confirma que o CARTE-oráculo de validação não é uma alternativa. Ele vence menos da metade das séries e perde mais do que vence.

Os oráculos mostram que o CARTE não é o ótimo global. O espaço restante até o melhor das 10 primeiras é de 1,40 e 2,04 pontos percentuais, na média dos sete e no agregado. Até o CARTE-oráculo restrito, é de 1,63 e 2,17. Até o CARTE-oráculo, é de 2,88 e 3,23. A maior parte do espaço perdido está na seleção, não na geração. O melhor das 10 primeiras já é melhor que o CARTE em 1,40 e 2,04 pontos percentuais, e as duas últimas tentativas acrescentam 0,23 e 0,13 ponto percentual ao valor do melhor das 10 primeiras. O CARTE-oráculo de validação em $U$ é pior que o CARTE, o que mostra que a seleção do agente é mais eficiente do que maximizar a validação em um conjunto grande. O que o CARTE oferece é uma escolha conservadora, auditável e próxima do melhor que ele próprio gerou, não o ótimo global.
```
