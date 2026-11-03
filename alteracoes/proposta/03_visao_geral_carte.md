# Visão geral do CARTE

Substitui a seção `Visão geral do método` de `overleaf/proposta.tex`
(linhas 104–125), incluindo a figura e a legenda.

```latex
\section{Visão geral do CARTE} \label{sec:carte}

O CARTE (\textit{Context-Aware Reasoning for Time-series Ensembles}) é o método desenvolvido para resolver o problema formulado na Seção~\ref{sec:formulacao}. Para cada série $i$, o CARTE utiliza um agente baseado em modelo de linguagem para construir e selecionar uma estratégia de combinação que define a função $f_i$ aplicada às previsões de teste. A decisão não parte de um conjunto fixo de estratégias previamente definidas: o agente dispõe de ações que constroem pools, pesos e combinações, e o histórico de tentativas é formado pelas estratégias que ele efetivamente avalia.

Essa escolha decorre da possibilidade de séries distintas se beneficiarem de estratégias distintas e da disponibilidade de apenas três janelas de validação por série, cada uma com $h$ observações. O agente não estima pesos por otimização direta sobre essas janelas. Sua decisão se apoia em um contexto específico da série, formado pelo perfil do histórico $\mathcal{S}_i$, pelo desempenho dos preditores nas janelas de validação, pelo diagnóstico interpretativo e pela forma das previsões dos modelos para a janela de teste. Os valores reais da janela de teste não são disponibilizados ao agente; as previsões de teste são usadas apenas como evidência de forma na decisão, nunca como alvo.

A execução do CARTE é dividida em duas macrofases, ilustradas na Figura~\ref{fig:carte_arquitetura}.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{images/proposta/carte_arquitetura.pdf}
    \caption[Visão geral da arquitetura proposta.]{Visão geral do CARTE. O contexto é construído por série; o agente itera sobre ações de preparação e avaliação; a decisão final combina o líder de validação com uma condição sazonal determinística; os valores reais de teste não participam da decisão.}
    \label{fig:carte_arquitetura}
\end{figure}

Na primeira macrofase, sem participação de modelo de linguagem, são construídos o perfil da série $\mathbf{z}_i$, a avaliação do pool de preditores nas janelas de validação $\mathcal{E}_i$, o diagnóstico interpretativo $\mathcal{C}_i$ e a representação da forma das previsões $\mathcal{F}_i$. Essa representação reúne o último trecho real de validação e as previsões dos modelos para a janela de teste, sem os valores reais dessa janela. O contexto é específico da série: não há meta-modelo entre séries, prior de estratégias do conjunto de dados nem estratégias iniciais fornecidas ao agente.

Na segunda macrofase, executada de forma independente para cada série, o agente entra em um ciclo de raciocínio e ação. A cada turno, ele recebe o contexto da série, o histórico das tentativas já avaliadas, o líder de validação e as referências de pools e pesos criadas durante a execução. O agente escolhe uma ação determinística: ações de preparação, como selecionar um pool ou construir pesos, não produzem pontuação; ações de avaliação aplicam a estratégia proposta ao backtest das três janelas de validação e atualizam o histórico e o líder.

Entre as ações disponíveis, o agente pode, por exemplo, selecionar um pool com os melhores preditores por SMAPE e atribuir pesos iguais aos melhores preditores em cada passo do horizonte, avaliando a combinação resultante. O ciclo utiliza um orçamento fixo de avaliações e turnos; não há aceite explícito nem critério de estagnação. Ao final do orçamento, o líder é a tentativa de menor pontuação de validação.

Sobre o líder, o CARTE constrói uma shortlist com a melhor tentativa de cada família de estratégia. Tentativas ponderadas são separadas pelo método de pesos, de modo que a família top-k por horizonte não seja absorvida por outra ponderação. A decisão final aplica uma condição sazonal determinística: quando a série apresenta sazonalidade estimável, o método compara as previsões dos candidatos da shortlist com uma previsão sazonal ingênua e adota o candidato somente se ele estiver suficientemente mais próximo dessa referência do que o líder; caso contrário, mantém o líder. A condição não utiliza os valores reais de teste.

A estratégia escolhida determina a função $f_i$, aplicada às previsões de teste $\hat{\mathbf{y}}_{i,j}^{(\mathrm{teste})}$, para $j \in \{1,\dots,M\}$, produzindo a previsão final $\hat{\mathbf{y}}_i^{(\mathrm{comb})}$. Os valores reais da janela de teste são usados apenas depois da aplicação, para o cálculo das métricas de desempenho.
```

## Observações de integração

- Exportar o novo diagrama do `.excalidraw` para
  `overleaf/images/proposta/carte_arquitetura.pdf`; se o projeto usar PNG, trocar
  a extensão no `\includegraphics`.
- As seções `Caracterização`, `Agente`, `Aplicação` e `Considerações finais` já
  foram reescritas nos arquivos 06–09; a aplicação agora inclui líder, shortlist
  e condição sazonal, e o capítulo está completo.
- O texto não menciona flags, seed, temperatura, número de iterações ou valores
  numéricos de orçamento.
