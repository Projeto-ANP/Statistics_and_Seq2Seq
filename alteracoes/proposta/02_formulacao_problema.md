# Formulação do problema

Substitui a seção `Formulação do problema` de `overleaf/proposta.tex`
(linhas 20–102). A formulação matemática é preservada; as alterações são:

- `CREST` → `CARTE`;
- remoção da dependência de “todas as séries do conjunto de dados”;
- explicitação de que as previsões de teste podem ser usadas como evidência de
  forma, mas os valores reais de teste não são acessados.

```latex
\section{Formulação do problema}
\label{sec:formulacao}

Seja $\mathcal{D} = \{\mathcal{S}_i\}_{i=1}^{N}$ um conjunto de históricos de $N$ séries temporais univariadas, em que $\mathcal{S}_i = (x_{i,t})_{t=1}^{n_i}$ contém as $n_i$ observações disponíveis da série $i$ antes da janela de teste que terá os seus valores preditos. O índice $i$ identifica a série, enquanto $t$ indica a posição temporal de uma observação nessa série. Seja ainda $\mathcal{M} = \{M_j\}_{j=1}^{M}$ um conjunto fixo de $M$ métodos preditivos (ou \textit{pool} de preditores), identificados pelo índice $j$. Todos os métodos produzem previsões para um horizonte comum de $h$ observações.

O conjunto de preditores $\mathcal{M}$ pode incluir métodos estatísticos, de aprendizagem de máquina e ingênuos, bem como variantes aplicadas a representações transformadas dos dados. A composição do \textit{pool} de preditores é uma escolha de projeto, de modo que o CARTE opera sobre as previsões fornecidas, sem exigir uma arquitetura ou seleção específica dos preditores. Para permitir a combinação, as previsões devem corresponder aos mesmos instantes e escala da variável-alvo.

Para cada série $i$, consideram-se três janelas de validação $w \in \{1,2,3\}$, e uma janela de teste posterior a elas. Seja $\tau_i^{(w)}$ a origem de previsão da janela de validação $w$, definida como o índice da última observação anterior a essa janela, com
\[
    1 \leq \tau_i^{(1)} < \tau_i^{(2)} < \tau_i^{(3)}
    \quad \text{e} \quad
    \tau_i^{(w)} + h \leq n_i,
    \qquad w \in \{1,2,3\}.
\]

O vetor de valores reais da janela de validação $w$ é
\[
    \mathbf{y}_i^{(w)}
    =
    \bigl(y_{i,k}^{(w)}\bigr)_{k=1}^{h}
    \in \mathbb{R}^{h},
    \qquad
    y_{i,k}^{(w)}
    =
    x_{i,\tau_i^{(w)}+k},
\]
em que $k \in \{1,\dots,h\}$ identifica o passo no horizonte de previsão. Denota-se por
\[
    \hat{\mathbf{y}}_{i,j}^{(w)}
    =
    \bigl(\hat{y}_{i,j,k}^{(w)}\bigr)_{k=1}^{h}
    \in \mathbb{R}^{h}
\]
o vetor de previsões produzido pelo método $M_j$ para essa janela, cuja componente $\hat{y}_{i,j,k}^{(w)}$ corresponde à previsão de $x_{i,\tau_i^{(w)}+k}$. Cada vetor de previsões é produzido utilizando apenas informações disponíveis até a respectiva origem $\tau_i^{(w)}$.

A janela de teste compreende os $h$ instantes imediatamente posteriores a $n_i$. Denotam-se por $x_{i,n_i+k}$, para $k \in \{1,\dots,h\}$, os valores futuros da série $i$, que não integram o histórico disponível $\mathcal{S}_i$ e são desconhecidos no momento da decisão de combinação. Esses valores compõem o vetor
\[
    \mathbf{y}_i^{(\mathrm{teste})}
    =
    \bigl(x_{i,n_i+k}\bigr)_{k=1}^{h}
    \in \mathbb{R}^{h}.
\]

O vetor de previsões do método $M_j$ para essa janela é
\[
    \hat{\mathbf{y}}_{i,j}^{(\mathrm{teste})}
    =
    \bigl(\hat{y}_{i,j,k}^{(\mathrm{teste})}\bigr)_{k=1}^{h}
    \in \mathbb{R}^{h},
\]
em que $\hat{y}_{i,j,k}^{(\mathrm{teste})}$ corresponde à previsão de $x_{i,n_i+k}$. Essas previsões utilizam apenas informações disponíveis até $n_i$.

O problema de combinação consiste em definir, para cada série $i$, uma função
\[
    f_i : (\mathbb{R}^{h})^{M} \longrightarrow \mathbb{R}^{h}
\]
que combine os $M$ vetores de previsões de teste em uma única previsão final:
\[
    \hat{\mathbf{y}}_i^{(\mathrm{comb})}
    =
    f_i\!\left(
        \hat{\mathbf{y}}_{i,1}^{(\mathrm{teste})},
        \ldots,
        \hat{\mathbf{y}}_{i,M}^{(\mathrm{teste})}
    \right).
\]

Busca-se reduzir o erro dessa previsão em relação a $\mathbf{y}_i^{(\mathrm{teste})}$. Como os valores reais de teste são desconhecidos no momento da decisão, a escolha de $f_i$ é fundamentada nas previsões e nos valores reais das três janelas de validação,
\[
    \left\{
        \left(
            \mathbf{y}_i^{(w)},
            \hat{\mathbf{y}}_{i,1}^{(w)},
            \ldots,
            \hat{\mathbf{y}}_{i,M}^{(w)}
        \right)
    \right\}_{w=1}^{3},
\]
nas características observáveis do histórico $\mathcal{S}_i$ da própria série e na forma das previsões de teste dos preditores. Os valores reais da janela de teste, $\mathbf{y}_i^{(\mathrm{teste})}$, não são acessados durante a decisão. Nessa etapa, as previsões de teste são utilizadas apenas como evidência de forma; a aplicação final também as utiliza como entradas da combinação, mas nunca como alvo.

A formulação permite selecionar estratégias de combinação distintas para diferentes séries, sem impor uma única função a todo o conjunto $\mathcal{D}$. No protocolo considerado, as três janelas de validação fornecem evidências sobre o desempenho dos preditores em um número limitado de períodos.
```
