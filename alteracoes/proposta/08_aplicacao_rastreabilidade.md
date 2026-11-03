# Decisão final, aplicação e rastreabilidade

Substitui a seção `Aplicação da estratégia e rastreabilidade` de `overleaf/proposta.tex` (linhas 481–497). A seção passa a incluir a decisão final, com shortlist por família e condição sazonal, antes da aplicação no teste e do registro de auditoria.

```latex
\section{Decisão final, aplicação e rastreabilidade} \label{sec:aplicacao}

Ao término do ciclo de decisão, o histórico $\mathcal{H}_i$ contém todas as estratégias avaliadas na série, e o líder de validação $\ell_i$ é a tentativa de menor pontuação. O líder é o ponto de partida da decisão final, mas não é necessariamente a estratégia aplicada: o CARTE verifica se existe, entre as melhores tentativas de cada família, um candidato cuja forma seja mais coerente com a sazonalidade da série. A decisão final é determinística, não utiliza modelo de linguagem e não acessa os valores reais da janela de teste.

\subsection{Decisão final} \label{subsec:decisao-final}

\subsubsection{Shortlist por família}

A shortlist $\mathcal{K}_i$ é construída a partir do histórico $\mathcal{H}_i$. As tentativas são ordenadas pela pontuação da Equação~\eqref{eq:score} e agrupadas por família de estratégia. Estratégias ponderadas são subdivididas pelo método de pesos, de modo que a família top-k por horizonte seja preservada como uma família própria e não seja absorvida por outra ponderação.

Para cada família, seleciona-se a melhor tentativa que satisfaça três condições de plausibilidade. A primeira é que a previsão de teste seja finita e não vazia. A segunda é que sua magnitude não ultrapasse um limite fixo, calculado a partir do último valor observado e da escala recente da série, o que descarta previsões degeneradas ou explosivas. A terceira é que a trajetória não seja duplicata de outra já aceita na shortlist: duas previsões são consideradas a mesma quando são quase idênticas ou quando têm correlação muito alta e a mesma forma padronizada. A shortlist é limitada a um número fixo de famílias.

O líder global é normalmente o melhor membro de sua própria família e, portanto, integra a shortlist. Se o líder for degenerado e ficar de fora, ele é substituído pelo primeiro candidato plausível da shortlist, e a substituição é registrada. As previsões usadas para construir a shortlist são as previsões de teste produzidas pela aplicação de cada tentativa às previsões dos preditores; nenhum valor real de teste é lido nessa etapa.

\subsubsection{Condição sazonal}

A condição sazonal só é avaliada quando a referência sazonal ingênua é estimável. Sejam $p_i$ o período sazonal e $n_i$ o número de observações do histórico. A condição é ativada quando a força sazonal é maior ou igual a $0{,}5$, o período é maior ou igual a $2$ e o histórico contém pelo menos $10$ ciclos completos, isto é, $n_i \geq 10\,p_i$. Se qualquer uma dessas condições falhar, a decisão mantém o líder e o motivo é registrado.

Quando a condição é ativada, a referência sazonal ingênua $\mathbf{q}_i \in \mathbb{R}^{h}$ é construída pela repetição do último ciclo completo do histórico:
\[
    q_{i,k}
    =
    x_{i,\,n_i - p_i + ((k-1) \bmod p_i) + 1},
    \qquad k = 1,\dots,h.
\]
A escala $\sigma_i$ é o desvio-padrão do trecho recente do histórico. O trecho tem comprimento $\max(2,\min(n_i,\max(2p_i,2h,2)))$; se a escala for inválida ou nula, usa-se o desvio-padrão de todo o histórico e, se ainda inválido, o maior valor absoluto entre a última observação e a unidade.

Para cada candidato $s \in \mathcal{K}_i$, a distância à referência sazonal é
\[
    d_i(s)
    =
    \frac{1}{h}
    \sum_{k=1}^{h}
    \frac{
        \bigl|\hat{y}_{i,s,k}^{(\mathrm{teste})} - q_{i,k}\bigr|
    }{
        \sigma_i
    },
\]
em que $\hat{\mathbf{y}}_{i,s}^{(\mathrm{teste})}$ é a previsão de teste produzida pela estratégia $s$. Essa é a mesma distância resumida para cada candidato na representação $\mathcal{F}_i$, agora formalizada para a decisão final. A distância é uma medida L1 normalizada pela escala recente; valores menores indicam maior proximidade da referência sazonal. Ela utiliza apenas o histórico e as previsões de teste dos candidatos, nunca o valor real de teste.

\subsubsection{Regra de decisão}

Seja $s_i^{\star}$ o candidato da shortlist com menor distância sazonal,
\[
    s_i^{\star}
    =
    \arg\min_{s \in \mathcal{K}_i} d_i(s).
\]
A decisão segue uma ordem fixa de verificações. Se a shortlist for vazia, a decisão recai sobre o líder e o caso é registrado como fallback. Se a shortlist tiver menos de dois candidatos, a decisão recai sobre o líder efetivo e o caso também é registrado como fallback. Se a condição sazonal estiver inativa, o líder é mantido. Se nenhum candidato tiver distância sazonal definida, o líder também é mantido. Se o próprio líder for o candidato de menor distância, o líder é mantido. Caso contrário, calcula-se a diferença
\[
    \text{gap}_i
    =
    d_i(\ell_i) - d_i(s_i^{\star}).
\]
O líder é substituído por $s_i^{\star}$ somente quando $\text{gap}_i > 0{,}10$; caso contrário, o líder é mantido. Como a distância é dividida por $\sigma_i$, o limiar é expresso em unidades de desvio-padrão recente: a troca exige que o candidato esteja pelo menos $10\%$ de um desvio-padrão mais próximo da referência sazonal do que o líder.

A estratégia final é denotada $c_i$ e pode ter pontuação de validação pior que a do líder. Isso é intencional: a decisão final não otimiza apenas a pontuação de validação, mas combina essa evidência com a plausibilidade sazonal da trajetória. A escolha de $c_i$, o motivo da manutenção ou da troca, o valor do gate e as distâncias envolvidas são registrados para auditoria.

\subsection{Aplicação na janela de teste}

A estratégia $c_i$ define a função de combinação $f_i$ aplicada às previsões de teste dos preditores de $\mathcal{M}$:
\[
    \hat{\mathbf{y}}_i^{(\mathrm{comb})}
    =
    f_i\!\left(
        \hat{\mathbf{y}}_{i,1}^{(\mathrm{teste})},
        \ldots,
        \hat{\mathbf{y}}_{i,M}^{(\mathrm{teste})}
    \right).
\]
O mesmo pool, o mesmo método de agregação e, quando aplicável, a mesma regra de pesos da estratégia escolhida são recalculados sobre as previsões de teste. Para estratégias com pool selecionado a partir dos dados, a aplicação utiliza o pool completo escolhido nas três janelas de validação. Para estratégias com pesos ajustados, os pesos são ajustados nas três janelas de validação e aplicados à janela de teste. A aplicação não consulta os valores reais de teste e não calcula pontuação de validação nessa etapa.

\subsection{Métricas finais}

As métricas de desempenho são calculadas somente depois que a previsão combinada $\hat{\mathbf{y}}_i^{(\mathrm{comb})}$ já foi produzida. O CARTE reporta duas métricas: o SMAPE e o POCID.

O SMAPE é calculado diretamente entre a previsão combinada e os valores reais da janela de teste,
\[
    \text{SMAPE}_i
    =
    \frac{1}{h}
    \sum_{k=1}^{h}
    \frac{
        2\,\bigl|\hat{y}_{i,k}^{(\mathrm{comb})} - y_{i,k}^{(\mathrm{teste})}\bigr|
    }{
        \bigl|\hat{y}_{i,k}^{(\mathrm{comb})}\bigr|
        +
        \bigl|y_{i,k}^{(\mathrm{teste})}\bigr|
    },
\]
com pontos de denominador nulo ignorados. O POCID é calculado dentro da janela de teste,
\[
    \text{POCID}_i
    =
    \frac{100}{h-1}
    \sum_{k=2}^{h}
    \mathbf{1}\!\left[
        \bigl(\hat{y}_{i,k}^{(\mathrm{comb})} - \hat{y}_{i,k-1}^{(\mathrm{comb})}\bigr)
        \bigl(y_{i,k}^{(\mathrm{teste})} - y_{i,k-1}^{(\mathrm{teste})}\bigr)
        > 0
    \right],
\]
em que $\mathbf{1}[\cdot]$ é a função indicadora. Valores maiores de POCID indicam maior acerto de direção. SMAPE e POCID são as únicas métricas reportadas pelo método; a pontuação interna da Equação~\eqref{eq:score} não é uma métrica de teste e não é reportada como resultado.

\subsection{Registro e auditoria} \label{subsec:auditoria}

Para cada série, o método registra a evidência que sustenta a decisão final. O registro contém, para cada turno do ciclo, a ação solicitada, os argumentos, a observação devolvida, o líder do histórico antes e depois da chamada e o orçamento utilizado. Para a decisão final, o registro contém o líder, sua pontuação de validação e sua distância sazonal; a shortlist com as tentativas, suas especificações, pontuações e distâncias; o estado do gate sazonal e o motivo de sua ativação ou inativação; o candidato $s_i^{\star}$, o valor de $\text{gap}_i$ e a decisão tomada, seja de manutenção do líder, de troca ou de fallback. Também são registrados o identificador e a especificação da estratégia final $c_i$.

Essa estrutura garante que toda previsão, peso ou distância que compõe o resultado final se origine da execução real de uma ação determinística ou do cálculo determinístico da decisão final. O agente emite apenas nomes de ações e argumentos, nunca valores numéricos, e a única ação que insere uma tentativa no histórico é a avaliação de estratégia. Os valores reais da janela de teste são lidos apenas após a produção de $\hat{\mathbf{y}}_i^{(\mathrm{comb})}$, para o cálculo de SMAPE e POCID. Caso alguma inconsistência seja detectada no registro, a série não é interrompida; a decisão é preservada e a inconsistência é registrada para análise posterior.
```
