# Considerações iniciais

Substitui a seção `Considerações iniciais` de `overleaf/resultados.tex` (linhas 3–5).

```latex
\section{Considerações iniciais} \label{sec:resultados-iniciais}

Este capítulo apresenta os resultados obtidos pelo CARTE, em comparação com os modelos individuais que compõem o conjunto de preditores, com as combinações estáticas média, mediana e DBA, e com os métodos de combinação por meta-aprendizagem revisados no Capítulo~\ref{chp:revisao}, FFORMA e ADE. O agente utiliza o modelo \texttt{gpt-oss:20b}; a comparação com Qwen3.8 e Gemma4 é apresentada na Seção~\ref{sec:escolha-llm}.

Os resultados são reportados em SMAPE e POCID. O SMAPE mede o erro relativo de magnitude e é a métrica que ordena as estratégias durante a validação; o POCID mede o acerto de direção e é reportado separadamente, sem participar do ranking. As comparações são feitas em duas agregações: a média dos sete conjuntos de dados, que dá o mesmo peso a cada conjunto e é a referência principal deste capítulo, e o agregado das 680 séries, que dá o mesmo peso a cada série e é reportado como complemento. Na média dos sete conjuntos, o CARTE ocupa a primeira posição em SMAPE, com vantagem sobre o FFORMA, e a segunda em POCID; no agregado das 680 séries, repete a primeira posição em SMAPE e a segunda em POCID. Ao longo do capítulo, além da posição média, observa-se a constância do CARTE no topo do ranking, com atenção especial ao POCID. O capítulo está organizado por conjunto de dados, com uma seção para cada um dos sete conjuntos, seguida de uma síntese entre eles e de uma análise do comportamento do agente. Em cada conjunto, apresenta-se inicialmente o desempenho dos modelos individuais, depois o das estratégias de combinação e, por fim, a comparação geral e a discussão.
```
