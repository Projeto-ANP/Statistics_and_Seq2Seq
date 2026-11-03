# Mecanismo de pontuação de estratégias

Substitui a seção `Preparação em nível de conjunto de dados` de `overleaf/proposta.tex` (linhas 290–365). A subseção `Mecanismo de pontuação de estratégias` é preservada e reescrita para o CARTE; as subseções `Meta-modelo entre séries` e `Prior de estratégias do conjunto de dados` são removidas.

```latex
\section{Mecanismo de pontuação de estratégias} \label{sec:pontuacao}\label{subsec:pontuacao}

Toda estratégia de combinação avaliada pelo CARTE é pontuada pelo mesmo mecanismo, aplicado separadamente a cada série. A pontuação é usada durante o ciclo de decisão para ordenar as tentativas do agente e, ao final, para identificar o líder de validação. Ela não agrega resultados entre séries e não depende de conhecimento construído a partir das demais séries do conjunto de dados.

Uma estratégia de combinação $s$ especifica uma regra de seleção de preditores, um método de agregação e, quando aplicável, uma regra de cálculo de pesos. O método de agregação pode ser a média, a mediana, a média aparada, a combinação ponderada ou o DBA. Para a série $i$ e a janela de validação $w$, essas regras determinam um subconjunto não vazio $\mathcal{M}_{i,s}^{(w)} \subseteq \mathcal{M}$ e, quando aplicável, um vetor de pesos $\mathbf{p}_{i,s}^{(w)}$. A aplicação da estratégia produz o vetor de previsões combinadas
\[
    \hat{\mathbf{y}}_{i,s}^{(w)}
    =
    \bigl(\hat{y}_{i,s,k}^{(w)}\bigr)_{k=1}^{h}
    \in \mathbb{R}^{h}.
\]

A pontuação compara o erro da estratégia com o erro de uma referência $s_0$, definida como a média simples sobre o conjunto completo $\mathcal{M}$, sem seleção de preditores e sem ponderação. Para cada janela de validação $w$, a pontuação é a razão entre o SMAPE da estratégia e o SMAPE da referência,
\[
    \text{score}_{i,s}^{(w)}
    =
    \frac{
        \text{SMAPE}\bigl(\hat{\mathbf{y}}_{i,s}^{(w)}, \mathbf{y}_i^{(w)}\bigr)
    }{
        \text{SMAPE}\bigl(\hat{\mathbf{y}}_{i,s_0}^{(w)}, \mathbf{y}_i^{(w)}\bigr)
    },
\]
em que $\text{SMAPE}(\cdot,\cdot)$ é calculado separadamente em cada janela, com os $h$ passos do horizonte. A pontuação de $s$ para a série $i$ é a média ponderada das três pontuações por janela,
\begin{equation}
    \text{score}_i(s)
    =
    \sum_{w=1}^{3} \alpha_w \, \text{score}_{i,s}^{(w)},
    \qquad
    (\alpha_1, \alpha_2, \alpha_3) = (0{,}2,\ 0{,}3,\ 0{,}5),
    \label{eq:score}
\end{equation}
em que $\alpha_w$ é fixo e cresce com a proximidade da janela em relação ao teste. A janela mais recente recebe o maior peso porque é a que melhor representa o regime imediatamente anterior à janela de teste. Como a razão é adimensional, $\text{score}_i(s)$ é comparável entre estratégias da mesma série, e a referência $s_0$ obtém pontuação igual a $1$; valores menores que $1$ indicam estratégia melhor que a média simples do pool.

O POCID não participa da pontuação. Por medir acerto de direção dentro de uma sequência temporal contínua, ele é calculado separadamente em cada janela e reportado como a média simples dos três valores. A pontuação de ranking usa o SMAPE calculado separadamente em cada janela, pois precisa atribuir pesos distintos a elas. Métricas usadas na construção do contexto do pool, como o RMSE, também não participam da pontuação.

Duas salvaguardas evitam que a pontuação seja artificialmente otimista. A primeira diz respeito a estratégias cujos pesos $\mathbf{p}_{i,s}^{(w)}$ são ajustados a partir dos dados: o ajuste segue um esquema de janela expansiva, de modo que, ao pontuar a janela $w$, os pesos utilizados só podem ter sido calculados a partir das janelas anteriores a $w$. A segunda diz respeito a estratégias cujo subconjunto de preditores é selecionado a partir dos dados: a seleção segue um esquema \textit{leave-one-out} entre janelas, de modo que o subconjunto usado para pontuar a janela $w$ é determinado a partir apenas das demais janelas, excluindo $w$. Estratégias com pool e pesos fixos não possuem canal de vazamento e são pontuadas diretamente nas três janelas.

O mecanismo é específico da série: cada série possui seu próprio histórico, seu próprio líder e sua própria pontuação. A comparabilidade entre séries proporcionada pela razão de SMAPE é uma propriedade descritiva, usada apenas para relatar os resultados, e não uma evidência empregada na decisão.
```
