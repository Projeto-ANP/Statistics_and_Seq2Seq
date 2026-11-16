# Análise do comportamento do agente - Rastreabilidade da decisão

Atualiza a subseção `Rastreabilidade da decisão do agente` de `overleaf/resultados.tex` (linhas 1010–1021) com a execução representativa. A análise de similaridade foi refeita para as 165 séries com estratégia proposta pelo agente.

```latex
\subsection{Rastreabilidade da decisão do agente} \label{subsec:rastreabilidade}

Além da acurácia agregada, o CARTE oferece um diferencial qualitativo que o FFORMA e o ADE não replicam. Cada série tem um histórico com os pensamentos, as ações, os argumentos e os resultados de validação de cada tentativa, como ilustrado na Subseção~\ref{subsec:transcricao-decisao} e nos casos da Subseção~\ref{subsec:interpretabilidade-previsoes}. Esse histórico permite reconstruir a decisão e verificar por que uma estratégia foi escolhida, o que não está disponível nos métodos de meta-aprendizagem tradicionais.

A justificativa final, porém, é um resumo padronizado. Entre as 632 séries em que a decisão final manteve o líder proposto pelo agente, a similaridade média de Jaccard entre pares de justificativas, com palavras de conteúdo e stopwords removidas, é 0,64, com mediana 0,64. Entre pares que compartilham a mesma estratégia, a média é 0,69. Entre pares com estratégias diferentes, 0,64. A diferença é pequena, o que mostra que o texto é dominado pelo molde, não pelas particularidades da série. Todas as 632 justificativas têm a mesma primeira frase depois de normalizados os números e o nome da estratégia, e o trecho \textit{"an agent proposal as the best of backtested"} aparece em todas. As 48 séries com troca sazonal usam outro texto e ficam fora dessa contagem.

Esse resultado não elimina o valor da rastreabilidade, mas desloca o foco. O que permite auditar a decisão é o histórico de tentativas, não a frase final. O histórico registra as hipóteses testadas, as ações de seleção e ponderação, os escores de validação e a evolução do líder. A justificativa final identifica a estratégia escolhida e o escore, mas não articula uma justificativa causal para a escolha. A leitura da trajetória, como no passo a passo da série 87 do M4, continua sendo a via principal para entender a escolha.
```
