# Estudo de ablação - Buscas sem LLM

Atualiza a seção `Estudo de ablação: a seleção pelo agente` e a subseção `Buscas sem LLM`.

```latex
\section{Estudo de ablação: a seleção pelo agente} \label{sec:ablacao-agente}

Esta seção isola o mecanismo de seleção do CARTE. A comparação é feita em dois níveis. No primeiro, o agente é substituído por três buscas sem LLM que usam o mesmo catálogo, o mesmo orçamento de doze avaliações e a mesma regra final. No segundo, os oráculos mostram o que uma escolha perfeita teria alcançado, com e sem acesso à janela de teste. A coluna do CARTE usa a média das dez execuções, e todas as comparações usam a mesma janela de teste.

\subsection{Buscas sem LLM} \label{subsec:buscas-sem-llm}

Para isolar o efeito do modelo de linguagem, o agente foi substituído por três políticas que não usam LLM. A busca aleatória foi repetida três vezes, e o valor reportado é a média. A busca gulosa e a seleção por bandit (UCB) são determinísticas. As três partem de um histórico vazio, usam as mesmas ferramentas e receitas do agente, com uma grade fixa de parâmetros, e aplicam a mesma regra final. O espaço das buscas é um subconjunto do espaço do agente, que pode propor estratégias fora da grade. Entre as políticas de busca, muda apenas o mecanismo que propõe novas estratégias.

A Tabela~\ref{tab:ablacao-buscas-smape} apresenta o SMAPE médio por conjunto de dados. A coluna \textit{política escolhida} indica, em cada série, a política de busca de menor pontuação de validação, e reporta o teste dessa política. As duas últimas linhas trazem a média dos sete conjuntos e o agregado das 680 séries.

\begin{table}[H]
\centering
\small
\caption{SMAPE médio (\%) do CARTE e de três buscas sem LLM sobre o mesmo orçamento, por conjunto de dados. Em negrito, o menor valor da linha. A coluna \textit{política escolhida} reporta o teste da política de busca de menor pontuação de validação em cada série, não um método fixo.}
\label{tab:ablacao-buscas-smape}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{CARTE} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} & \textbf{Política escolhida} \\
\midrule
ANP    & 22,24 & \textbf{22,05} & 22,30 & 22,24 & 22,18 \\
NN5    & 12,04 & 12,04 & 12,29 & \textbf{11,89} & 12,12 \\
M4     & \textbf{9,18} & 9,31 & 10,50 & 9,39 & 9,49 \\
ETTh1  & 22,72 & 22,68 & 24,91 & \textbf{22,24} & 24,91 \\
ETTh2  & \textbf{17,67} & 18,27 & 18,44 & 18,85 & 18,50 \\
ETTm1  & \textbf{41,03} & 44,60 & 49,09 & 43,84 & 49,26 \\
ETTm2  & \textbf{17,02} & 18,62 & 18,78 & 18,10 & 18,73 \\
\midrule
Média dos 7 & \textbf{20,27} & 21,08 & 22,33 & 20,94 & 22,17 \\
Agregado (680 séries) & \textbf{13,78} & 13,85 & 14,66 & 13,91 & 14,07 \\
\bottomrule
\end{tabular}
\end{table}

O CARTE obtém o menor SMAPE na média dos sete conjuntos, com 20,27\%, e no agregado das 680 séries, com 13,78\%. A busca aleatória fica em 21,08\% e 13,85\%, a UCB em 20,94\% e 13,91\%, e a busca gulosa em 22,33\% e 14,66\%. A vantagem do CARTE é de 0,81 ponto percentual na média dos sete e 0,08 no agregado sobre a aleatória, 0,66 e 0,13 sobre a UCB, e 2,06 e 0,89 sobre a gulosa. A diferença contra a gulosa é a única que sobrevive à correção de Holm ($p = 0{,}0003$). As demais têm p corrigido acima de 0,12. A ausência de significância nas outras comparações não significa que os métodos sejam equivalentes. A diferença observada aparece nas duas agregações.

O CARTE é o melhor SMAPE em quatro dos sete conjuntos, ETTh2, ETTm1, ETTm2 e M4. A busca aleatória vence na ANP, e a UCB vence no ETTh1 e no NN5. As maiores diferenças a favor do CARTE contra a aleatória estão no ETTm1 e no ETTm2. Contra a UCB, estão no ETTm1 e no ETTh2. Contra a gulosa, a maior diferença está no ETTm1, seguida do ETTh1. Em ANP, ETTh1 e NN5, a melhor busca vence por margens de 0,15 a 0,48 ponto percentual. Uma diferença de 0,1 ponto percentual no SMAPE médio é uma diferença no erro médio de previsão, não uma posição no ranking. Ela se repete nas duas agregações.

A Tabela~\ref{tab:ablacao-buscas-pocid} repete a estrutura para o POCID. O POCID é reportado em separado e não participa da escolha da estratégia.

\begin{table}[H]
\centering
\small
\caption{POCID médio (\%) do CARTE e de três buscas sem LLM sobre o mesmo orçamento, por conjunto de dados. Em negrito, o maior valor da linha. A coluna \textit{política escolhida} reporta o teste da política de busca de menor pontuação de validação em cada série, não um método fixo.}
\label{tab:ablacao-buscas-pocid}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{CARTE} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} & \textbf{Política escolhida} \\
\midrule
ANP    & 63,28 & 63,34 & \textbf{64,14} & 62,14 & 63,00 \\
NN5    & \textbf{46,04} & 43,63 & 40,28 & 41,96 & 43,03 \\
M4     & \textbf{57,88} & 56,79 & 56,10 & 56,75 & 56,86 \\
ETTh1  & \textbf{62,11} & 61,49 & \textbf{62,11} & 57,76 & \textbf{62,11} \\
ETTh2  & 54,22 & 56,73 & 57,76 & 58,39 & \textbf{59,63} \\
ETTm1  & 54,47 & 54,04 & 52,80 & \textbf{54,66} & 52,59 \\
ETTm2  & 40,37 & 41,20 & \textbf{46,58} & 44,72 & 45,55 \\
\midrule
Média dos 7 & 54,05 & 53,89 & 54,25 & 53,77 & \textbf{54,68} \\
Agregado (680 séries) & \textbf{57,18} & 56,25 & 55,62 & 55,66 & 56,17 \\
\bottomrule
\end{tabular}
\end{table}

No POCID, o CARTE é o melhor no agregado das 680 séries, com 57,18\%, à frente da aleatória (56,25\%), da gulosa (55,62\%) e da UCB (55,66\%). Na média dos sete, o maior valor é o da política escolhida (54,68\%), seguido da gulosa (54,25\%) e do CARTE (54,05\%). A aleatória fica em 53,89\% e a UCB em 53,77\%. O CARTE fica em terceiro quando a política escolhida entra na comparação e em segundo entre os métodos fixos. Nenhuma diferença de POCID é significativa após a correção de Holm, e há muitos empates.

O CARTE é melhor no M4 e no NN5. A gulosa vence na ANP e no ETTm2, a UCB vence no ETTm1, e a política escolhida vence no ETTh2. No ETTh1, o CARTE empata com a gulosa e com a política escolhida em 62,11\%. No ETTh2, a diferença para o melhor valor da linha é de 5,41 pontos percentuais, e no ETTm2, de 6,21. Como o POCID não participa da seleção, ele é uma leitura complementar. A ausência de significância não deve ser lida como equivalência, porque o teste tem pouca potência com tantos empates.

A Tabela~\ref{tab:ablacao-buscas-vantagem} resume a vantagem do CARTE e as contagens pareadas. Valores positivos indicam vantagem do CARTE, e negativos, vantagem da política.

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Vantagem média do CARTE sobre as buscas sem LLM, em pontos percentuais, e número de séries em que o CARTE vence, empata ou perde (C/E/P). As colunas 7 e 680 indicam a média dos sete conjuntos e o agregado das 680 séries. Valores positivos indicam vantagem do CARTE.}
\label{tab:ablacao-buscas-vantagem}
\begin{tabular}{@{}lrrrrcc@{}}
\toprule
\textbf{Política} & \textbf{SMAPE 7} & \textbf{SMAPE 680} & \textbf{POCID 7} & \textbf{POCID 680} & \textbf{C/E/P SMAPE} & \textbf{C/E/P POCID} \\
\midrule
Aleatória    & +0,81 & +0,08 & +0,17 & +0,93 & 358/0/322 & 335/44/301 \\
Gulosa       & +2,06 & +0,89 & $-$0,20 & +1,57 & 373/0/307 & 312/61/307 \\
UCB          & +0,66 & +0,13 & +0,29 & +1,52 & 320/52/308 & 276/127/277 \\
Política escolhida & +1,90 & +0,29 & $-$0,63 & +1,01 & 329/46/305 & 290/116/274 \\
\bottomrule
\end{tabular}
\end{table}

A vantagem do CARTE aparece em 14 das 16 células da tabela. As exceções são o POCID na média dos sete contra a gulosa e contra a política escolhida. O CARTE vence a aleatória em 358 séries e perde em 322 no SMAPE. Contra a UCB, vence em 320 e perde em 308. Contra a gulosa, vence em 373 e perde em 307. As contagens não são unânimes, mas as vitórias do CARTE contra a gulosa são, em média, maiores que as derrotas.

A Tabela~\ref{tab:ablacao-buscas-vitorias} complementa a leitura com a contagem exclusiva do melhor método em cada série.

\begin{table}[H]
\centering
\small
\caption{Número de séries em que cada método obtém o melhor valor, por métrica. Empates são contados à parte e não entram na coluna de nenhum método.}
\label{tab:ablacao-buscas-vitorias}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Métrica} & \textbf{CARTE} & \textbf{Aleatória} & \textbf{Gulosa} & \textbf{UCB} & \textbf{Empates} \\
\midrule
SMAPE & 150 & 119 & 196 & 158 & 57 \\
POCID & 124 & 95 & 145 & 107 & 209 \\
\bottomrule
\end{tabular}
\end{table}

A gulosa é o melhor SMAPE em 196 séries, contra 150 do CARTE. A diferença entre as duas contagens não se traduz em melhor média para a gulosa porque suas perdas são maiores.

A política escolhida por série na validação tem 22,17\% de SMAPE na média dos sete, atrás do CARTE, da aleatória e da UCB, e à frente apenas da gulosa. No POCID, tem o maior valor na média dos sete, 54,68\%, mas fica em 56,17\% no agregado, abaixo do CARTE. A escolha da política na validação não reproduz a vantagem do agente. O p bruto dessa comparação no SMAPE é 0,042, mas o valor corrigido é 0,127. A diferença entre o CARTE e a política escolhida é compatível com o uso do diagnóstico da série e do histórico de tentativas na escolha das estratégias, e não somente da pontuação de validação. A seleção do agente é, portanto, mais eficiente que a escolha da política por validação no SMAPE e competitiva no POCID.

As buscas custam entre 0,13 e 5,76 s por série, contra 17,63 a 22,95 s do CARTE na execução representativa. A diferença é o custo das chamadas ao LLM. As buscas são mais baratas, mas não superam o CARTE de forma consistente. O custo adicional do CARTE corresponde à seleção feita pelo agente.

As três buscas mostram que a vantagem do CARTE não depende de um único conjunto. A busca gulosa é a mais competitiva em número de vitórias por série, mas perde por margens maiores. A aleatória e a UCB ficam próximas no agregado, sem superar o CARTE nas duas agregações. A vantagem do CARTE é de décimos de ponto percentual em algumas comparações e de pontos inteiros em outras, mas a direção é predominantemente consistente. O que distingue o CARTE é a seleção das estratégias pelo modelo de linguagem, guiada pelo diagnóstico da série e pelo histórico de tentativas, e não o orçamento de avaliações.
```
