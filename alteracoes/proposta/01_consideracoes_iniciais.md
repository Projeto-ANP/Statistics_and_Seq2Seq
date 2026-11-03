# Considerações iniciais

Substitui a seção `Considerações iniciais` de `overleaf/proposta.tex`
(linhas 3–8).

## O que mudou

- `CREST` → `CARTE`.
- O método não “seleciona entre estratégias previamente definidas”; ele constrói
  e seleciona a estratégia com o agente.
- A decisão final é descrita como líder de validação + condição sazonal
  determinística.
- O roadmap remove a seção de preparação por conjunto de dados, que não existe no
  CARTE, e aponta para contexto, agente e aplicação.

```latex
\section{Considerações iniciais}

Este capítulo descreve o método desenvolvido para a combinação dinâmica de previsões de séries temporais utilizando um agente baseado em grandes modelos de linguagem. O objetivo deste trabalho foi investigar e desenvolver um método de combinação de modelos para previsão de valores em séries temporais (\textit{forecasting}) que integrasse abordagens lineares e não-lineares a partir de um conjunto diverso de modelos, incluindo métodos estatísticos e de aprendizagem de máquina, e que explorasse distintas representações dos dados durante a etapa de treinamento, obtidas a partir das transformações \textit{wavelet} e de Fourier, de modo a aumentar a diversidade do conjunto de preditores combinados. Para a etapa de combinação, foi desenvolvido o CARTE (\textit{Context-Aware Reasoning for Time-series Ensembles}), um método que delega a um agente baseado em modelo de linguagem a construção e a seleção de uma estratégia de combinação para cada série, a partir do contexto da própria série e das evidências disponíveis nas janelas de validação, em substituição a uma regra fixa de ponderação aplicada indistintamente a todas as séries de um conjunto de dados. A decisão final do CARTE combina o líder de validação com uma condição sazonal determinística, sem utilizar os valores reais da janela de teste, com o objetivo de investigar em que medida essa decisão, mediada por um agente, supera os modelos individuais e as estratégias de combinação estáticas e de meta-aprendizagem, e em que condições essa vantagem se manifesta.

O restante deste capítulo está organizado da seguinte forma. A Seção~\ref{sec:formulacao} formaliza o problema de combinação de predições resolvido pelo método. A Seção~\ref{sec:carte} apresenta a visão geral do CARTE e o fluxo completo de execução. A Seção~\ref{sec:pontuacao} define o mecanismo de pontuação que ordena as estratégias de cada série. A Seção~\ref{sec:diagnostico} descreve como o contexto de cada série é construído antes da decisão de combinação. A Seção~\ref{sec:agente} detalha o agente combinador, suas ações e seu ciclo de decisão. A Seção~\ref{sec:aplicacao} descreve a decisão final, a aplicação da estratégia escolhida e os mecanismos de rastreabilidade do método. Por fim, a Seção~\ref{sec:consideracoes-finais-proposta} encerra o capítulo com as considerações finais sobre a proposta.
```
