# Estudo de ablação - Síntese

Fecha a seção `Estudo de ablação: a seleção pelo agente` com a subseção `Síntese da ablação`.

```latex
\subsection{Síntese da ablação} \label{subsec:sintese-ablacao}

A ablação compara o CARTE em dois eixos. No primeiro, o agente é substituído por buscas sem LLM com o mesmo catálogo e o mesmo orçamento. No segundo, os oráculos mostram o melhor resultado que uma escolha perfeita teria alcançado. A leitura conjunta é que a seleção do agente é eficiente e que o espaço restante está na escolha, não na geração de candidatos.

No SMAPE, o CARTE lidera as duas agregações (Tabela~\ref{tab:ablacao-buscas-smape}), com 20,27\% na média dos sete e 13,78\% no agregado. A busca aleatória fica em 21,08\% e 13,85\%, a UCB em 20,94\% e 13,91\%, e a gulosa em 22,33\% e 14,66\%. A diferença contra a gulosa é a única que sobrevive à correção de Holm. As diferenças contra a aleatória e a UCB são de 0,81 e 0,66 ponto percentual na média dos sete, e de 0,08 e 0,13 no agregado. Essas diferenças são menores que a diferença contra a gulosa, mas são diferenças no erro médio e aparecem nas duas agregações. O CARTE vence quatro dos sete conjuntos. A aleatória vence na ANP, e a UCB vence no ETTh1 e no NN5. A gulosa vence mais séries, 196 contra 150, mas perde por margens maiores, o que explica a média pior.

No POCID, o CARTE é o melhor no agregado (Tabela~\ref{tab:ablacao-buscas-pocid}), com 57,18\%, à frente da aleatória (56,25\%), da gulosa (55,62\%) e da UCB (55,66\%). Na média dos sete, o CARTE fica em 54,05\%, atrás da política escolhida (54,68\%) e da gulosa (54,25\%), e à frente da aleatória (53,89\%) e da UCB (53,77\%). Nenhuma diferença de POCID é significativa, e há muitos empates. O POCID não participa da seleção, então ele é uma leitura complementar.

Os oráculos mostram que o CARTE não é o ótimo global, mas que a seleção do agente é eficiente (Tabela~\ref{tab:ablacao-carte-oraculo-smape}). O CARTE fica 1,63 e 2,17 pontos percentuais acima do CARTE-oráculo restrito, 1,40 e 2,04 acima do melhor das 10 primeiras, 2,88 e 3,23 acima do CARTE-oráculo e 2,72 e 3,74 acima do melhor individual. A distância para o CARTE-oráculo restrito e para o melhor das 10 primeiras é menor que a distância para o CARTE-oráculo e para o melhor individual. O CARTE-oráculo de validação fica em 21,91\% e 14,11\%, pior que o CARTE em 1,64 e 0,33 ponto percentual. Com cerca de 226 estratégias e três janelas, o máximo da validação captura ruído. A seleção do agente, mais conservadora, generaliza melhor. O mesmo padrão aparece na política escolhida entre as buscas, que fica em 22,17\% de SMAPE, atrás do CARTE.

A diferença entre o CARTE-oráculo restrito e o melhor das 10 primeiras é de 0,23 e 0,13 ponto percentual. O CARTE ainda está 1,40 e 2,04 acima do melhor das 10 primeiras. O gargalo está na seleção, não na quantidade de propostas. O limite das dez primeiras tentativas já está próximo do limite de todas as tentativas. A dificuldade é escolher o melhor com três janelas de validação.

A ablação não sustenta a ideia de que o CARTE é o ótimo global. Ela sustenta que a seleção feita pelo modelo de linguagem, com o diagnóstico da série e o histórico de tentativas, é mais eficiente que uma busca fixa e que a maximização da validação em $U$. As buscas custam entre 0,13 e 5,76 s por série, contra 17,63 a 22,95 s do CARTE na execução representativa. A diferença é o custo das chamadas ao LLM. O que o CARTE oferece é uma escolha conservadora, auditável e próxima do melhor que ele próprio gerou.
```
