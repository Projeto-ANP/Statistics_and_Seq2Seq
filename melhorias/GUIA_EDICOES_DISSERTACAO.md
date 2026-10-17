# Guia de edições da dissertação

Guia único de edição, em ordem: primeiro `overleaf/proposta.tex`, depois
`overleaf/resultados.tex`, depois `background.tex` e `revisao.tex`. Cada mudança
traz o rótulo REMOVER e, logo abaixo, SUBSTITUIR POR, com o texto exato do
arquivo. Não use nomes de execução, versão ou artefato interno no texto final;
as ações aparecem pelos nomes em português, como já é o padrão do capítulo.

Os valores abaixo são os da configuração final do CREST, aquela efetivamente
usada nos resultados: catálogo enxuto, prompt estruturado, decisão final pela
melhor pontuação do histórico e esforço de raciocínio médio.

## Numeração e controle de aplicação

Cada alteração tem um código estável, de E01 a E75. A coluna Revisão do índice
abaixo indica os itens que mudaram depois da primeira versão do guia: se você
colou o guia antes da data indicada, reaplique o item (o bloco REMOVER de cada
um mostra o texto antigo exato que está no seu documento). A coluna Colado em é
para você anotar a data em que aplicou cada item no Overleaf; nas próximas
revisões do guia eu aponto apenas os códigos afetados.

### Índice de edições

| Código | Onde | O que muda | Revisão | Colado em |
|---|---|---|---|---|
| E01 | 1.1 | Objetivo nas considerações iniciais (proposta) |  |  |
| E02 | 1.2 | Notação: ações de 24 para 12 |  |  |
| E03 | 1.3 | Entradas do Algoritmo CReST: 24 para 12 |  |  |
| E04 | 1.4 | Catálogo completo (três categorias, 12 ações) |  |  |
| E05 | 1.5 | Prompt estruturado (adicionar) |  |  |
| E06 | 1.6 | Paciência e estagnação (adicionar) |  |  |
| E07 | 1.7 | Regras alternativas de decisão final (adicionar) |  |  |
| E08 | 1.8 | Registro e auditoria (bloco comentado) |  |  |
| E09 | 1.9 | "procedência" para "registro de auditoria" |  |  |
| E10 | 2.1 | Parágrafo de abertura da escolha do modelo |  |  |
| E11 | 2.1 | Tabela de desempenho dos três modelos | corrigido antes de 16/10 |  |
| E12 | 2.1 | Parágrafo de desempenho (gpt-oss contra Qwen3.8) | revisado em 16/10 (p-valores) |  |
| E13 | 2.1 | Tabela de custo dos três modelos | corrigido antes de 16/10 |  |
| E14 | 2.1 | Parágrafo de custo |  |  |
| E15 | 2.1 | Parágrafo "Modelo escolhido" |  |  |
| E16 a E22 | 2.2 | Linha do CREST nas sete tabelas por conjunto |  |  |
| E23 a E25 | 2.3 | ANP: ordem de mérito, frase do ETS, p-valores da MCM | E25 corrigido; parágrafo completo em E67 |  |
| E26 a E28 | 2.3 | NN5: SMAPE, POCID, nota da MCM | parágrafos completos em E68 |  |
| E29 e E30 | 2.3 | M4: SMAPE e POCID |  |  |
| E31 e E32 | 2.3 | ETTh1: SMAPE e POCID |  |  |
| E33 | 2.3 | ETTh2: SMAPE |  |  |
| E34 e E35 | 2.3 | ETTm1: SMAPE e contagem 5 contra 2 |  |  |
| E36 a E38 | 2.3 | ETTm2: mudança de argumento (FFORMA à frente) |  |  |
| E39 | 2.4 | Linha do ETTm2 na tabela de ranking | 16/10: negrito no rank 1 |  |
| E40 | 2.4 | Linha do rank médio |  |  |
| E41 | 2.4 | Parágrafo "Nenhum método ocupa..." |  |  |
| E42 | 2.4 | Parágrafo "Essa posição..." | revisado em 16/10 (ADE em três conjuntos) |  |
| E43 | 2.4 | Parágrafo da MCM agregada (SMAPE) | corrigido antes de 16/10 |  |
| E44 | 2.4 | Parágrafo da MCM agregada (POCID) | corrigido antes de 16/10 |  |
| E45 | 2.4 | Parágrafo "Em conjunto, as duas matrizes..." |  |  |
| E46 a E50 | 2.4 | Cinco linhas da contagem de vitórias | corrigido antes de 16/10 |  |
| E51 | 2.4 | Parágrafo do Random Forest (DWT) | corrigido antes de 16/10 |  |
| E52 | 2.4 | Parágrafo do horizonte (SMAPE) |  |  |
| E53 | 2.4 | Parágrafo do horizonte (POCID) | revisado em 16/10 (segundo lugar, não terceiro) |  |
| E54 | 2.5.1 | Frequência de uso do catálogo |  |  |
| E55 | 2.5.2 | Motivo de encerramento do ciclo |  |  |
| E56 | 2.5.3 | Estudo de caso (ANP 8 e ANP 5) | texto final em 16/10 |  |
| E57 | 2.5.4 | Interpretabilidade (M4 206 e ANP 13) | texto final em 16/10 |  |
| E58 | 2.5.5 | Distribuição da melhoria relativa |  |  |
| E59 | 2.5.6 | Custo computacional |  |  |
| E60 | 2.5.7 | Rastreabilidade da decisão |  |  |
| E61 | 2.5 | Menções a 273 séries para 47 |  |  |
| E62 | 2.6 | Tabela da ablação por conjunto | substituído por AB01 (GUIA_ABLACAO.md) |  |
| E63 | 2.6 | p-valores agregados da ablação | substituído por AB03 (GUIA_ABLACAO.md) |  |
| E64 | 2.6 | Subseção "Quando o agente muda a decisão" | substituído por AB04 (GUIA_ABLACAO.md) |  |
| E65 | 2.6 | Considerações finais | segue valendo (rank e custo) |  |
| E66 | 2.7 | Substituição dos arquivos de figura | atualizado em 16/10 |  |
| E67 | 2.3b | Discussão da MCM da ANP (parágrafos) | novo em 16/10 |  |
| E68 | 2.3b | Discussão da MCM do NN5 (parágrafos) | novo em 16/10 |  |
| E69 | 2.3b | Discussão da MCM do M4 (frases) | novo em 16/10 |  |
| E70 | 2.3b | Discussão da MCM do ETTh1 (frases) | novo em 16/10 |  |
| E71 | 2.3b | Discussão da MCM do ETTh2 (frases) | novo em 16/10 |  |
| E72 | 2.3b | Discussão da MCM do ETTm1 (frase do p) | novo em 16/10 |  |
| E73 | 2.3b | Discussão da MCM do ETTm2 (ordem, contagens e p) | novo em 16/10 |  |
| E74 | 2.4 | Linhas da tabela de ranking (negrito no rank 1) | novo em 16/10 |  |
| E75 | 2.5.2b | Nova subseção: passo a passo de uma decisão (M4 206) | novo em 16/10 |  |

### Histórico de revisões do guia

- 16/10 (esta revisão): apenas E56 e E57 mudaram de conteúdo; antes eles traziam somente a instrução de que os casos precisavam ser trocados, então não existe versão antiga desses blocos para desfazer, basta colar os textos finais agora. As figuras dos casos agora exibem também ADE, FFORMA, mediana e DBA, e as fontes das figuras de linha (casos, horizonte e vitórias) foram ampliadas, com as legendas fora da área de plotagem (os textos de E57 citam os SMAPE desses métodos nas duas séries; se você colou E57 na primeira versão de hoje, reaplique). A Seção 2.7 e a Parte 4 também foram atualizadas (figuras de horizonte e de caso prontas, bloco opcional da figura de vitórias). Os itens de colagem alterados depois desta revisão estão listados na entrada seguinte (auditoria de números).
- 16/10 (auditoria de números): correções em E12 (p-valores do gpt-oss contra o Qwen3.8, agora no mesmo método de Wilcoxon do restante do capítulo, com os empates tratados como nas MCM), E25 (instrução errada sobre o DBA), E42 (ADE em três conjuntos) e E53 (CREST e mediana empatados em segundo, não terceiro); e nova Seção 2.3b com os blocos completos das sete discussões das MCM (E67 a E73), porque os parágrafos do texto antigo usavam números da rodada anterior.
- 16/10 (figuras dos casos): as figuras de sucesso e fracasso passaram a ter dois painéis: escala completa (com os modelos individuais em cinza, como antes) e ampliação na faixa das estratégias de combinação (sem eles, para o detalhe entre os métodos ficar visível); o texto cita as previsões individuais como antes, e a figura da trajetória (séries 5 e 8) foi descartada.
- 16/10 (nova subseção): criado o passo a passo completo de uma decisão (E75), com a trajetória da série 206 do M4 desde as sementes até o aceite, pensamentos em cada iteração e citações literais; a 2.5.3 volta ao resumo dos dois casos.
- 16/10 (casos da 2.5.4): as séries foram trocadas para M4 206 (sucesso) e ANP 13 (fracasso), buscando séries visualmente mais limpas e maior contraste entre o CREST e as demais combinações; legendas e parágrafos atualizados.
- 16/10 (negrito em tabelas): convenção do melhor resultado aplicada às tabelas que faltavam: contagem de vitórias (E46, negrito em 199/29,26), tabela da ablação (AB01) e busca sem LLM (AB05) no guia de ablação, e tabela de ranking (E39 e E74, negrito no rank 1 de cada linha). Em empates, os dois valores ficam em negrito.
- 16/10 (guia de ablação): a Seção 2.6 virou o arquivo dedicado `melhorias/GUIA_ABLACAO.md` (AB01 a AB07), com a estrutura final da seção de ablação e as duas subseções novas.
- Correções anteriores (já incorporadas nesta versão): contagem de vitórias (E46 a E51), colunas de gpt-oss e Qwen3.8 nas tabelas da Subseção 2.1 (E11 e E13), linha do Random Forest (DWT) na MCM agregada (E43 e E44) e p-valores agregados da ablação (E63). Se você colou o guia antes dessas correções, reaplique os itens marcados na coluna Revisão.

---

# Parte 1. overleaf/proposta.tex

## 1.1. Considerações iniciais — E01

REMOVER (fim do primeiro parágrafo):

```
com o objetivo de superar os resultados obtidos pelos modelos individuais e por outras estratégias de combinação, como as baseadas em ponderação
```

SUBSTITUIR POR:

```latex
com o objetivo de investigar em que medida essa decisão, mediada por um agente, supera os modelos individuais e as estratégias de combinação estáticas e de meta-aprendizagem, e em que condições essa vantagem se manifesta
```

## 1.2. Pseudocódigo, notação — E02

REMOVER:

```
$i$, formado por pares $(s, \text{score}_i(s))$; $\mathcal{A}$, o catálogo de 24 ações descritas na Seção~\ref{sec:agente}, ao qual se soma o aceite, que não
```

SUBSTITUIR POR:

```latex
$i$, formado por pares $(s, \text{score}_i(s))$; $\mathcal{A}$, o catálogo de 12 ações descritas na Seção~\ref{sec:agente}, ao qual se soma o aceite, que não
```

## 1.3. Pseudocódigo, entradas do Algoritmo~\ref{alg:crest} — E03

REMOVER:

```
$\hat{\mathbf{Y}}_i^{(\mathrm{teste})}$; catálogo $\mathcal{A}$ com $|\mathcal{A}|=24$; orçamento $K$; paciência $P$; limite $L$}
```

SUBSTITUIR POR:

```latex
$\hat{\mathbf{Y}}_i^{(\mathrm{teste})}$; catálogo $\mathcal{A}$ com $|\mathcal{A}|=12$; orçamento $K$; paciência $P$; limite $L$}
```

## 1.4. Catálogo de ações: substituir o bloco inteiro — E04

REMOVER todo o bloco que começa em:

```
O catálogo é composto por 24 ações determinísticas, organizadas em cinco categorias
```

e termina em:

```
ambas registradas junto ao histórico da série para fins de análise posterior.
```

O bloco removido é este (cinco tabelas e o parágrafo de aceite). O valor 24
aparece porque é o texto que está hoje no arquivo, é exatamente o que se remove;
na versão nova o catálogo passa a ter 12 ações:

```latex
O catálogo é composto por 24 ações determinísticas, organizadas em cinco categorias que correspondem às etapas naturais da tarefa de combinação: diagnóstico, seleção de pool, cálculo de pesos, combinação e validação. Cada ação é implementada como uma função convencional, sem qualquer participação de modelo de linguagem em sua execução, e o agente interage com elas exclusivamente pelo nome da ação e pelos argumentos fornecidos em sua \textit{Action Input}, descrita na Seção~\ref{sec:agente}.

\textbf{Ações de diagnóstico.} Produzem as evidências sobre a série e sobre o conjunto de preditores $\mathcal{M}$, parte das quais já compõe o contexto inicial fornecido ao agente, descrito na Seção~\ref{sec:diagnostico}, mas que também podem ser recalculadas pelo agente ao longo do ciclo de decisão, com parâmetros distintos dos usados nesse contexto inicial. As ações disponíveis nessa categoria e suas respectivas descrições são apresentadas na Tabela~\ref{tab:acoes_descricao}.

\begin{table}[H]
\centering
\caption{Ações de diagnóstico e suas respectivas descrições.}
\label{tab:acoes_descricao}
\begin{tabular}{@{}p{5cm}p{9cm}@{}}
\toprule
\textbf{Ação} & \textbf{Descrição} \\
\midrule
Perfil da série & Retorna comprimento, sazonalidade, estacionariedade, presença de \textit{outliers} e características do conjunto \textit{catch22}. \\
Resumo da decomposição & Retorna a proporção da variância da série explicada por tendência, sazonalidade e resíduo. \\
Resumo de erro dos preditores & Retorna a tabela de RMSE por preditor, ranqueada, limitada aos de melhor desempenho e a um resumo dos demais. \\
Estabilidade do \textit{ranking} & Retorna o coeficiente de estabilidade do \textit{ranking} entre janelas e identifica quais preditores mais mudam de posição. \\
Correlação de erro & Retorna grupos de preditores cujos resíduos são fortemente correlacionados entre si, indicando redundância. \\
Teste de Diebold-Mariano & Compara dois preditores nomeados, com a correção para amostras pequenas de \citet{harvey1997testing}. \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Ações de seleção de pool.} Produzem um subconjunto de $\mathcal{M}$, referenciado por um identificador que pode ser usado como argumento de ações posteriores, sem que o agente precise enumerar os preditores individualmente. A Tabela~\ref{tab:acoes_selecao_pool} apresenta as ações disponíveis para selecionar e refinar esses subconjuntos.

\begin{table}[H]
\centering
\caption{Ações de seleção de pool de preditores.}
\label{tab:acoes_selecao_pool}
\begin{tabular}{@{}p{5cm}p{9cm}@{}}
\toprule
\textbf{Ação} & \textbf{Descrição} \\
\midrule
Seleção pelos $\kappa$ melhores & Seleciona os $\kappa$ preditores de menor erro nas janelas indicadas. \\
Seleção pelos mais estáveis & Seleciona os $\kappa$ preditores mais consistentes entre janelas, combinando a posição média no \textit{ranking} com sua variação. \\
Poda de redundância & Remove de um subconjunto os preditores redundantes identificados pela ação de correlação de erro, mantendo apenas o de menor erro em cada grupo. \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Ações de cálculo de pesos.} Produzem um vetor de pesos sobre um subconjunto de preditores, também referenciado por um identificador, nunca pelos valores numéricos em si. Conforme apresentado na Tabela~\ref{tab:acoes_pesos}, as ações dessa categoria utilizam diferentes critérios para determinar a contribuição de cada preditor na combinação.

\begin{table}[H]
\centering
\caption{Ações de cálculo de pesos e suas respectivas descrições.}
\label{tab:acoes_pesos}
\begin{tabular}{@{}p{5cm}p{9cm}@{}}
\toprule
\textbf{Ação} & \textbf{Descrição} \\
\midrule
Peso pelo erro inverso & Pesos proporcionais ao inverso do erro de cada preditor, com um parâmetro que permite aproximar o resultado do peso uniforme. \\
Peso por \textit{softmax} do erro & Pesos proporcionais ao \textit{softmax} do erro negativo, a formulação empregada pelo ADE \citep{ade}. \\
Peso pela tendência do erro & Pesos que consideram para onde o erro de cada preditor está evoluindo ao longo das janelas, não apenas sua média; calculados, por padrão, a partir do erro absoluto médio (MAE) de cada preditor, para que um único ponto de erro elevado não domine a direção da tendência estimada. \\
Peso por mínimos quadrados & Pesos obtidos por mínimos quadrados, na formulação de \citet{granger1984}, projetados sobre o simplex de probabilidade por padrão. \\
Peso por características da série & Pesos de um meta-modelo ajustado especificamente para a série, a partir de suas características, no espírito do FFORMA \citep{fforma2019}; recorre à ação de \textit{softmax} do erro quando há poucas janelas disponíveis. \\
Peso pelo meta-modelo entre séries & Pesos do meta-modelo descrito na Seção~\ref{sec:preparacao}, treinado com as demais séries do conjunto de dados. \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Ações de combinação.} Produzem a previsão combinada propriamente dita, a partir de um subconjunto de preditores e, quando aplicável, de um vetor de pesos. As diferentes estratégias de combinação disponibilizadas ao agente estão descritas na Tabela~\ref{tab:acoes_combinacao}.

\begin{table}[H]
\centering
\caption{Ações de combinação de previsões.}
\label{tab:acoes_combinacao}
\begin{tabular}{@{}p{5cm}p{9cm}@{}}
\toprule
\textbf{Ação} & \textbf{Descrição} \\
\midrule
Combinação por média & Média simples entre os preditores do subconjunto. \\
Combinação por mediana & Mediana entre os preditores do subconjunto, mais robusta a um preditor com previsão discrepante. \\
Combinação por média aparada & Descarta uma proporção das previsões mais extremas em cada extremidade antes de calcular a média. \\
Combinação ponderada & Média ponderada, a partir de um identificador de pesos já calculado por uma ação da categoria anterior. \\
Combinação por DBA & Alinha temporalmente as previsões, por meio de \textit{Dynamic Time Warping Barycenter Averaging}, antes de combiná-las. \\
Seleção de um único preditor & Adota um único preditor nomeado, sem qualquer combinação. \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Ações de validação.} Não produzem uma nova previsão, mas avaliam e organizam o que já foi produzido pelas ações anteriores. A Tabela~\ref{tab:acoes_validacao} apresenta as ações utilizadas para avaliar estratégias, verificar a plausibilidade das previsões e consultar os resultados de tentativas anteriores.

\begin{table}[H]
\centering
\caption{Ações de validação e acompanhamento das estratégias.}
\label{tab:acoes_validacao}
\begin{tabular}{@{}p{5cm}p{9cm}@{}}
\toprule
\textbf{Ação} & \textbf{Descrição} \\
\midrule
Avaliação da estratégia & Monta a estratégia especificada, executa o procedimento de \textit{backtest} descrito na Seção~\ref{subsec:pontuacao} e insere o resultado no histórico de tentativas da série; é a única ação por meio da qual uma estratégia passa a integrar esse histórico. \\
Verificação de sanidade & Compara a previsão de teste de uma estratégia contra os limites históricos da série, emitindo um alerta sem impedir a continuidade do ciclo. \\
Listagem do histórico & Retorna o histórico de tentativas já registradas para a série, ranqueado do melhor para o pior resultado. \\
\bottomrule
\end{tabular}
\end{table}

Além dessas 24 ações, existe uma operação adicional, o aceite, que não integra o catálogo no mesmo sentido das demais: em vez de ser executada e devolver uma observação para uma nova iteração, ela encerra o ciclo de decisão da série, formalizando o critério de aceite descrito na Seção~\ref{sec:agente}. O aceite é acompanhado de uma confiança autodeclarada pelo agente e de uma justificativa para a estratégia escolhida, ambas registradas junto ao histórico da série para fins de análise posterior.
```

SUBSTITUIR POR:

```latex
As informações de diagnóstico (perfil da série, decomposição, tabela de erro dos preditores, estabilidade de ranking e correlação de erros) não constituem ações do catálogo: elas são calculadas uma única vez e entregues no contexto inicial do agente, descrito na Seção~\ref{sec:diagnostico}. O catálogo final reúne 12 ações determinísticas, organizadas em três categorias que correspondem às etapas da decisão de combinação: seleção de subconjunto, cálculo de pesos e avaliação de estratégias. Cada ação é implementada como uma função convencional, sem qualquer participação de modelo de linguagem em sua execução, e o agente interage com elas exclusivamente pelo nome da ação e pelos argumentos fornecidos em sua \textit{Action Input}, descrita na Seção~\ref{sec:agente}.

\textbf{Ações de seleção de subconjunto.} Produzem um subconjunto de $\mathcal{M}$, referenciado por um identificador que pode ser usado como argumento de ações posteriores, sem que o agente precise enumerar os preditores individualmente. A Tabela~\ref{tab:acoes_selecao_pool} apresenta as ações dessa categoria.

\begin{table}[H]
\centering
\caption{Ações de seleção de subconjunto de preditores.}
\label{tab:acoes_selecao_pool}
\begin{tabular}{@{}p{5cm}p{9cm}@{}}
\toprule
\textbf{Ação} & \textbf{Descrição} \\
\midrule
Seleção pelos $\kappa$ melhores & Seleciona os $\kappa$ preditores de menor erro nas janelas indicadas. \\
Seleção pelos mais estáveis & Seleciona os $\kappa$ preditores mais consistentes entre janelas, combinando a posição média no \textit{ranking} com sua variação. \\
Poda de redundância & Remove de um subconjunto os preditores redundantes identificados pela correlação de erro, mantendo apenas o de menor erro em cada grupo. \\
Seleção gulosa de subconjunto & Adiciona iterativamente o preditor que mais reduz o erro da média corrente, mantendo os preditores escolhidos ao menos uma vez. \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Ações de cálculo de pesos.} Produzem um vetor de pesos sobre um subconjunto de preditores, também referenciado por um identificador, nunca pelos valores numéricos em si. A Tabela~\ref{tab:acoes_pesos} apresenta as ações dessa categoria.

\begin{table}[H]
\centering
\caption{Ações de cálculo de pesos e suas respectivas descrições.}
\label{tab:acoes_pesos}
\begin{tabular}{@{}p{5cm}p{9cm}@{}}
\toprule
\textbf{Ação} & \textbf{Descrição} \\
\midrule
Peso pelo erro inverso & Pesos proporcionais ao inverso do erro de cada preditor, com um parâmetro que permite aproximar o resultado do peso uniforme. \\
Peso pela tendência do erro & Pesos que consideram para onde o erro de cada preditor está evoluindo ao longo das janelas, não apenas sua média; calculados, por padrão, a partir do erro absoluto médio (MAE) de cada preditor, para que um único ponto de erro elevado não domine a direção da tendência estimada. \\
Peso pelo meta-modelo & Pesos de um meta-modelo ajustado a partir das características da própria série ou treinado com as demais séries do conjunto de dados, no espírito do FFORMA \citep{fforma2019}; recorre ao peso pelo erro inverso quando há poucas janelas disponíveis. \\
Peso por mínimos quadrados & Pesos obtidos por mínimos quadrados, na formulação de \citet{granger1984}, projetados sobre o simplex de probabilidade por padrão; exige mais janelas de validação do que as três disponíveis no protocolo adotado e, por isso, não é oferecida. \\
Peso pelo melhor preditor por passo & Pesos que se concentram, em cada passo do horizonte, no preditor de menor erro de validação naquele passo. \\
Peso pela média dos melhores por passo & Pesos uniformes sobre os $\kappa$ preditores de menor erro em cada passo do horizonte. \\
Peso pela seleção gulosa & Pesos proporcionais ao número de vezes que cada preditor é escolhido pela seleção gulosa com reposição. \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Ação de avaliação.} A avaliação da estratégia é a única ação por meio da qual uma estratégia passa a integrar o histórico de tentativas da série. Ela monta a estratégia especificada, executa o procedimento de \textit{backtest} descrito na Seção~\ref{subsec:pontuacao} e devolve ao agente a posição da tentativa no histórico, a pontuação obtida e o erro por janela. Os métodos de combinação disponibilizados por essa ação são a média, a mediana, a média aparada, a combinação ponderada, o DBA e a adoção de um único preditor nomeado; o subconjunto e, quando aplicável, os pesos, são informados como argumentos, referenciados por identificador.

Além dessas 12 ações, existe uma operação adicional, o aceite, que não integra o catálogo no mesmo sentido das demais: em vez de ser executada e devolver uma observação para uma nova iteração, ela encerra o ciclo de decisão da série, formalizando o critério de aceite descrito na Seção~\ref{sec:agente}. O aceite é acompanhado de uma confiança autodeclarada pelo agente e de uma justificativa para a estratégia escolhida, ambas registradas junto ao histórico da série para fins de análise posterior.
```

Nota: no protocolo de três janelas, a ação de mínimos quadrados não é oferecida
ao agente. Se preferir evitar a ressalva, descreva o catálogo como "onze ações
efetivamente oferecidas" e omita a linha de mínimos quadrados.

## 1.5. Ciclo de raciocínio: acrescentar o prompt estruturado — E05

ADICIONAR DEPOIS do parágrafo que termina com:

```
justificada em termos de características observáveis da série, e não apenas pelo valor numérico do erro obtido.
```

Bloco a inserir:

```latex
Além dessas diretrizes, o texto entregue ao agente organiza o catálogo por etapa, apresenta um procedimento de quatro passos, dois exemplos de tipos distintos de estratégia e, a cada turno, um bloco de diagnóstico do líder da validação, com a margem de pontuação para o segundo colocado e um veredito de separabilidade estatística entre os dois. Esse bloco sustenta a regra de parada entregue ao agente: aceitar a semente quando o veredito for indistinguível, aceitar a própria proposta quando ela se separa da semente e, quando as tentativas retornam previsões idênticas ou pesos quase uniformes, alterar o subconjunto ou o tipo de combinação em vez de insistir no esquema de pesos. O agente é executado sobre um modelo de linguagem de peso aberto, local, com esforço de raciocínio médio.
```

## 1.6. Critérios de encerramento: precisar a paciência — E06

ADICIONAR DEPOIS do parágrafo que termina com:

```
incluindo as estratégias fixas testadas antes de o agente atuar.
```

Bloco a inserir:

```latex
A estagnação é contabilizada apenas sobre propostas de avaliação de estratégia, de modo que turnos dedicados à seleção de subconjunto ou ao cálculo de pesos não consomem a paciência; além disso, tanto o aceite quanto a estagnação são verificados contra a melhor pontuação de todo o histórico, e não apenas contra a última proposta.
```

## 1.7. Aplicação: registrar as regras alternativas avaliadas — E07

ADICIONAR DEPOIS do parágrafo que termina com:

```
essa divergência é registrada para fins de rastreabilidade, discutidos na Seção~\ref{subsec:auditoria}.
```

Bloco a inserir:

```latex
Regras alternativas de decisão final foram implementadas e comparadas no Capítulo~\ref{chp:resultados}, entre elas a média das melhores tentativas do histórico, o retorno à melhor semente na ausência de separação estatística e a média de um conjunto de confiança construído por testes pareados entre as tentativas. A decisão adotada neste trabalho permanece a de melhor pontuação do histórico, por ser a de melhor desempenho médio e a de interpretação mais direta.
```

## 1.8. Rastreabilidade: substituir o bloco comentado — E08

REMOVER o bloco comentado que começa em:

```
% \subsection{Procedência} \label{subsec:auditoria}
```

e termina em:

```
discutido na Seção~\ref{sec:diagnostico}.
```

SUBSTITUIR POR:

```latex
\subsection{Registro e auditoria} \label{subsec:auditoria}

Além da aplicação da estratégia vencedora, o método registra a evidência que sustenta cada decisão, permitindo a análise crítica dos resultados e a verificação de que toda previsão, peso ou métrica que compõe o resultado final de uma série se origina da execução real de uma ação do catálogo. Essa garantia decorre de uma restrição estrutural: uma tentativa só pode ser registrada pelas estratégias fixas avaliadas antes da atuação do agente ou pela ação de avaliação de estratégia. Como a saída do agente é sempre texto, o nome de uma ação e seus argumentos, nunca um valor numérico, não existe caminho pelo qual um número possa ser inserido no resultado final sem ter sido calculado a partir das previsões originais dos preditores.

Para tornar essa garantia auditável, dois tipos de registro são mantidos ao final do processamento de cada série. O primeiro é o registro dos turnos, que contém, para cada iteração, a ação solicitada, os argumentos, o resultado da execução, o líder do histórico antes e depois da chamada, os tempos de cada etapa e o motivo de encerramento do ciclo, acompanhado das métricas derivadas do processo de decisão. O segundo é a verificação de forma sobre a evidência associada a cada tentativa: o número de resíduos registrados deve corresponder ao produto entre o número de janelas de validação e o horizonte de previsão, e o número de pontuações por janela deve corresponder ao número de janelas, confirmando que essa evidência tem a forma esperada de um resultado calculado sobre as três janelas. Essa verificação é executada automaticamente e seu resultado é registrado; caso indique alguma inconsistência, a série não deixa de ser processada, seguindo o mesmo princípio de degradação sem interrupção adotado em outras partes do método.
```

## 1.9. Considerações finais — E09

REMOVER:

```
o mecanismo de procedência torna essa decisão passível de verificação posterior
```

SUBSTITUIR POR:

```latex
o registro de auditoria torna essa decisão passível de verificação posterior
```

---

# Parte 2. overleaf/resultados.tex

Regra geral: todos os números do CREST passam a ser os da configuração final.
Os valores atuais do texto vieram da execução anterior; a substituição é
obrigatória em cada tabela e em cada frase citada abaixo. As figuras de Matriz
de Comparação Múltipla e as figuras de erro por passo do horizonte precisam ser
regeradas com a configuração final antes de fechar as seções correspondentes.

## 2.1. Escolha do modelo de linguagem

(E10) REMOVER o parágrafo:

```
Antes de apresentar os resultados por conjunto de dados, é necessário justificar a escolha do modelo de linguagem empregado no papel de agente combinador ao longo do restante deste capítulo. Três modelos de linguagem de peso aberto foram comparados nesse papel, gpt-oss:20b, qwen3:30b-a3b e gemma4:26b, descritos na Seção~\ref{subsec:principais-modelos} do Capítulo~\ref{chp:background}, avaliando tanto o desempenho preditivo quanto o custo computacional de cada um.
```

SUBSTITUIR POR:

```latex
Antes de apresentar os resultados por conjunto de dados, é necessário justificar a escolha do modelo de linguagem empregado no papel de agente combinador ao longo do restante deste capítulo. Três modelos de linguagem de peso aberto foram comparados nesse papel, descritos na Seção~\ref{subsec:principais-modelos} do Capítulo~\ref{chp:background}, avaliando tanto o desempenho preditivo quanto o custo computacional de cada um.
```

(E11) REMOVER a Tabela~\ref{tab:escolha-llm-desempenho} inteira (título, tabela e linhas) e SUBSTITUIR POR:

```latex
\begin{table}[H]
\centering
\caption{Desempenho médio do CREST com os três modelos de linguagem, por conjunto de dados.}
\label{tab:escolha-llm-desempenho}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{\textbf{SMAPE (\%)}} & \multicolumn{3}{c}{\textbf{POCID (\%)}} \\
\textbf{Conjunto de dados} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} \\
\midrule
ANP    & 21,97 & \textbf{21,88} & -- & \textbf{64,79} & 64,39 & -- \\
NN5    & \textbf{11,57} & 11,84 & -- & \textbf{45,95} & 45,30 & -- \\
M4     & \textbf{9,15}  & 9,20  & -- & 59,03 & \textbf{59,40} & -- \\
ETTh1  & \textbf{22,72} & 24,75 & -- & \textbf{63,35} & 62,11 & -- \\
ETTh2  & 17,99 & \textbf{17,62} & -- & 55,28 & \textbf{56,52} & -- \\
ETTm1  & \textbf{45,98} & 46,64 & -- & 55,28 & \textbf{57,14} & -- \\
ETTm2  & 20,30 & \textbf{18,95} & -- & \textbf{45,34} & 43,48 & -- \\
\bottomrule
\end{tabular}
\end{table}
```

Convenção: o negrito marca o melhor valor de cada comparação na linha (menor SMAPE e maior POCID). Nota: as colunas de gpt-oss e Qwen3.8 são da configuração final e foram conferidas com os scripts de geração das tabelas. As células do Gemma4 ficam com -- até a nova execução; não reaproveite os valores antigos, que são de outra configuração.

(E12) REMOVER o parágrafo que começa em:

```
Na maior parte dos sete conjuntos de dados, as diferenças entre os três modelos de linguagem permanecem pequenas
```

SUBSTITUIR POR:

```latex
As diferenças entre o gpt-oss e o Qwen3.8 permanecem pequenas na maior parte dos conjuntos de dados, abaixo de um ponto percentual em cinco dos sete, e os dois modelos trocam vitórias entre conjuntos: o Qwen3.8 é melhor na ANP, no ETTh2 e no ETTm2, enquanto o gpt-oss é melhor na NN5, no M4, no ETTh1 e no ETTm1. No agregado das 680 séries, as duas versões são estatisticamente indistinguíveis (SMAPE $p = 0{,}7595$, POCID $p = 0{,}5264$), com 67 séries em que o gpt-oss vence, 64 em que o Qwen3.8 vence e 549 empates em SMAPE. A maior diferença por conjunto ocorre na NN5 em SMAPE, a favor do gpt-oss (média de 0,27 ponto percentual, $p = 0{,}0781$), e nenhuma das quatorze comparações por conjunto atinge significância estatística. Em nenhum caso a diferença altera a ordem de mérito relativa entre os métodos de combinação discutida nas seções anteriores.
```

(E13) REMOVER a Tabela~\ref{tab:escolha-llm-custo} inteira e SUBSTITUIR POR:

```latex
\begin{table}[H]
\centering
\caption{Tempo médio de decisão por série (segundos), por modelo e conjunto de dados.}
\label{tab:escolha-llm-custo}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{\textbf{Média}} & \multicolumn{3}{c}{\textbf{Desvio-padrão}} \\
\textbf{Conjunto de dados} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} & \textbf{gpt-oss} & \textbf{Qwen3.8} & \textbf{Gemma4} \\
\midrule
ANP    & 10,3 & \textbf{9,5}  & -- & 13,9 & 9,4  & -- \\
NN5    & \textbf{10,0} & 11,6 & -- & 12,5 & 11,7 & -- \\
M4     & \textbf{8,3}  & 18,5 & -- & 8,7  & 16,5 & -- \\
ETTh1  & \textbf{14,4} & 27,1 & -- & 22,0 & 25,4 & -- \\
ETTh2  & \textbf{14,3} & 29,7 & -- & 7,5  & 34,9 & -- \\
ETTm1  & \textbf{6,2}  & 29,3 & -- & 7,7  & 11,5 & -- \\
ETTm2  & \textbf{7,7}  & 47,9 & -- & 9,4  & 27,2 & -- \\
\bottomrule
\end{tabular}
\end{table}
```

Convenção: o negrito marca o menor tempo médio de cada linha; o desvio-padrão não entra na comparação.

(E14) REMOVER o parágrafo que começa em:

```
O gpt-oss:20b é o mais rápido dos três em todos os sete conjuntos de dados
```

SUBSTITUIR POR:

```latex
O gpt-oss é o mais rápido dos modelos avaliados em todos os conjuntos de dados, com médias entre 6 e 15 segundos por série. O Qwen3.8 é de duas a seis vezes mais lento, com médias entre 9 e 48 segundos, e responde por tempos mais altos nos conjuntos ETT. O Gemma4 completa a comparação após a execução pendente.
```

(E15) REMOVER o parágrafo da subseção "Modelo escolhido" e SUBSTITUIR POR:

```latex
Como o desempenho preditivo dos modelos é equivalente (as duas versões avaliadas são estatisticamente indistinguíveis no agregado) e o custo computacional do gpt-oss é o menor entre os avaliados, sem contrapartida de ganho de acurácia por parte do Qwen3.8, o gpt-oss foi adotado como modelo de linguagem do agente em todos os demais resultados apresentados neste capítulo.
```

## 2.2. Tabelas por conjunto de dados

Substituir a linha do CREST nas tabelas de estratégias de combinação:

ANP (E16). REMOVER `CREST & 22,09 & 64,59 \\` e SUBSTITUIR POR `CREST & 21,97 & 64,79 \\`.

NN5 (E17). REMOVER `CREST & \textbf{11,77} & 43,76 \\` e SUBSTITUIR POR `CREST & \textbf{11,57} & 45,95 \\`.

M4 (E18). REMOVER `CREST & 9,23 & 59,05 \\` e SUBSTITUIR POR `CREST & 9,15 & 59,03 \\`.

ETTh1 (E19). REMOVER `CREST & \textbf{22,76} & 62,73 \\` e SUBSTITUIR POR `CREST & \textbf{22,72} & 63,35 \\`.

ETTh2 (E20). REMOVER `CREST & \textbf{17,94} & 55,28 \\` e SUBSTITUIR POR `CREST & \textbf{17,99} & 55,28 \\`.

ETTm1 (E21). REMOVER `CREST & 45,94 & 55,28 \\` e SUBSTITUIR POR `CREST & 45,98 & 55,28 \\`.

ETTm2 (E22). REMOVER `CREST & \textbf{16,10} & \textbf{45,34} \\` e SUBSTITUIR POR `CREST & 20,30 & \textbf{45,34} \\`.

## 2.3. Frases por conjunto de dados

ANP:

- (E23) REMOVER `da mediana (21,94\% e 61,94\%), da média (22,06\% e 61,89\%), do CREST (22,09\% e 64,59\%)` e SUBSTITUIR POR `da mediana (21,94\% e 61,94\%), do CREST (21,97\% e 64,79\%), da média (22,06\% e 61,89\%)`.
- (E24) REMOVER `também superando o ETS por uma margem menor (64,59\%)` e SUBSTITUIR POR `também superando o ETS por uma margem menor (64,79\%)`.
- (E25) O parágrafo de discussão da MCM da ANP tem números antigos (intervalo de SMAPE, três pares significativos, contagens de vitória e empates); use o bloco completo do item E67, na Seção 2.3b. A instrução anterior sobre uma comparação contra o DBA estava errada: o DBA não faz parte da matriz da ANP.

NN5:

- (E26) REMOVER `o CREST obtém o menor SMAPE (11,77\%), à frente do ADE (11,78\%)` e SUBSTITUIR POR `o CREST obtém o menor SMAPE (11,57\%), à frente do ADE (11,78\%)`.
- (E27) REMOVER `enquanto o CREST apresenta o menor valor (43,76\%)` e SUBSTITUIR POR `enquanto o CREST fica atrás apenas da mediana e do DBA nessa métrica, com 45,95\%`.
- (E28) Os parágrafos de discussão da MCM do NN5 têm números antigos, e uma das conclusões muda com os dados finais (agora há duas comparações do CREST estatisticamente significativas); use o bloco completo do item E68, na Seção 2.3b.

M4:

- (E29) REMOVER `seguida de perto pelo CREST (9,23\%)` e SUBSTITUIR POR `seguida de perto pelo CREST (9,15\%)`.
- (E30) REMOVER `à frente do CREST (59,05\%)` e SUBSTITUIR POR `à frente do CREST (59,03\%)`.

ETTh1:

- (E31) REMOVER `o CREST obtém o menor SMAPE (22,76\%)` e SUBSTITUIR POR `o CREST obtém o menor SMAPE (22,72\%)`.
- (E32) REMOVER `à frente do ADE e do CREST, empatados em segundo lugar (62,73\%)` e SUBSTITUIR POR `à frente do ADE (62,73\%), com o CREST em segundo lugar (63,35\%)`.

ETTh2:

- (E33) REMOVER `o CREST obtém o menor SMAPE (17,94\%)` e SUBSTITUIR POR `o CREST obtém o menor SMAPE (17,99\%)`.
- (E33, continuação) No mesmo parágrafo, REMOVER `com diferença apenas na quarta casa decimal` e SUBSTITUIR POR `com diferença de apenas 0,05 ponto percentual`.

ETTm1:

- (E34) REMOVER `e pelo CREST (45,94\%)` e SUBSTITUIR POR `e pelo CREST (45,98\%)`.
- (E34, continuação) No parágrafo dos modelos individuais, REMOVER `à frente de sete dos dezoito modelos de aprendizagem de máquina` e SUBSTITUIR POR `à frente de seis dos catorze modelos de aprendizagem de máquina`.
- (E35) REMOVER `a mediana obtém SMAPE menor que o CREST em 4 das 7 séries, contra 3 em que o CREST é melhor` e SUBSTITUIR POR `a mediana obtém SMAPE menor que o CREST em 5 das 7 séries, contra 2 em que o CREST é melhor`.

ETTm2 (mudança de argumento):

- (E36) REMOVER o trecho `o CREST obtém o menor SMAPE (16,10\%), à frente do FFORMA (16,54\%) e do ADE (18,72\%), e também o maior POCID entre as seis (45,34\%)` e SUBSTITUIR POR `o FFORMA obtém o menor SMAPE (16,54\%), seguido do ADE (18,72\%) e da mediana (19,97\%), com o CREST em quarto lugar (20,30\%) mas mantendo o maior POCID entre as seis (45,34\%)`.
- (E36, continuação) Na tabela, a linha do FFORMA passa a ter o negrito no menor SMAPE: `FFORMA & \textbf{16,54} & 31,06 \\` (o CREST deixa de ser o de menor SMAPE).
- (E37) REMOVER `entre 16,10\% e 19,97\%` e SUBSTITUIR POR `entre 16,54\% e 20,30\%`.
- (E38) No parágrafo final do conjunto, três frases precisam de ajuste:
  - REMOVER `ainda fica atrás do FFORMA, do ADE e do CREST em SMAPE (19,97\%)` e SUBSTITUIR POR `ainda fica atrás do FFORMA e do ADE em SMAPE (19,97\%), mas supera o CREST (20,30\%)` (o CREST deixa de estar à frente da mediana em SMAPE).
  - REMOVER `o único caso entre os sete conjuntos de dados em que a mediana não obtém o maior POCID entre as três combinações estáticas` e SUBSTITUIR POR `um dos dois casos entre os sete conjuntos de dados em que a mediana não obtém o maior POCID entre as três combinações estáticas, junto com o NN5`.
  - REMOVER `superando as demais quinze linhas da tabela` e SUBSTITUIR POR `superando as demais dezesseis linhas da tabela`.

Valores da MCM regenerada para a linha do CREST, por dataset (p da matriz em
`outputs/resultados/item_05/mcm_<DATASET>.csv`; o diff e a contagem V/T/D
estão nas mesmas células):

| dataset | comparação | p da MCM |
|---|---|---|
| ANP | vs FFORMA | 0,0414 |
| ANP | vs ADE | 0,8899 |
| ANP | vs Mediana | 0,4277 |
| ANP | vs Theta (melhor individual) | 0,9725 |
| NN5 | vs CatBoost (Original + FT) | 0,0375 |
| NN5 | vs Média | 0,0347 |
| NN5 | vs FFORMA | 0,0785 |
| NN5 | vs ADE | 0,2598 |
| M4 | vs ADE | 0,0002 |
| M4 | vs FFORMA | 0,3474 |
| M4 | vs Random Forest (CWT) | 0,1582 |
| M4 | vs Mediana | 0,9841 |
| ETTh1 | vs Random Forest (CWT) | 0,0312 |
| ETTh1 | vs Mediana | 0,0469 |
| ETTh1 | vs FFORMA | 0,0156 |
| ETTh1 | vs ADE | 0,0156 |
| ETTh2 | vs CatBoost (Original + CWT) | 0,4688 |
| ETTh2 | vs Mediana | 0,9136 |
| ETTh2 | vs FFORMA | 0,2188 |
| ETTh2 | vs ADE | 0,2188 |
| ETTm1 | vs Random Forest (DWT) | 0,2969 |
| ETTm1 | vs FFORMA | 0,5781 |
| ETTm1 | vs ADE | 1,0000 |
| ETTm1 | vs Mediana | 0,9375 |
| ETTm2 | vs Random Forest (DWT) | 0,0312 |
| ETTm2 | vs FFORMA | 0,4688 |
| ETTm2 | vs ADE | 0,8125 |
| ETTm2 | vs Mediana | 0,5781 |

### 2.3b. Discussões das MCM por conjunto (blocos de correção)

Os parágrafos de discussão das MCM por conjunto (a última subseção de cada um dos sete conjuntos) ainda usam números da rodada anterior e, em alguns casos, a conclusão muda com os dados finais. As figuras já são as atualizadas em `overleaf/resultados_final/`; o que falta é o texto.

#### E67. ANP

REMOVER a frase de abertura:

```
A Figura~\ref{fig:mcm-anp} apresenta a Matriz de Comparação Múltipla entre cinco métodos na ANP: Theta, o melhor modelo individual; a mediana, a melhor combinação estática, conforme a Tabela~\ref{tab:anp-estaticas}; FFORMA; ADE; e CREST, ordenados pelo SMAPE médio entre as 182 séries.
```

SUBSTITUIR POR:

```latex
A Figura~\ref{fig:mcm-anp} apresenta a Matriz de Comparação Múltipla entre cinco métodos na ANP: Theta, o melhor modelo individual; FFORMA; ADE; a mediana, a melhor combinação estática, conforme a Tabela~\ref{tab:anp-estaticas}; e CREST, ordenados pelo SMAPE médio entre as 182 séries.
```

No primeiro parágrafo, REMOVER `entre 21,54\% (Theta) e 22,09\% (CREST)` e SUBSTITUIR POR `entre 21,54\% (Theta) e 21,97\% (CREST)`.

Ainda no primeiro parágrafo, REMOVER `supera o CREST ($p = 0,0060$)` e SUBSTITUIR POR `supera o CREST ($p = 0,0414$)`.

Ainda no primeiro parágrafo, REMOVER `contra o CREST ($p = 0,7599$)` e SUBSTITUIR POR `contra o CREST ($p = 0,9725$)`.

REMOVER o segundo parágrafo inteiro:

```
A comparação entre o FFORMA e o CREST merece um comentário à parte, porque a contagem de vitórias por série conta uma história diferente da média isolada. Embora o FFORMA tenha SMAPE médio menor, a MCM mostra que o CREST obtém SMAPE menor que o FFORMA em 114 das 182 séries (62,6\%), contra 68 séries em que o FFORMA é melhor. Ou seja, a vantagem média do FFORMA é sustentada por um subconjunto menor de séries nas quais a diferença é grande, não por uma superioridade consistente ao longo da maioria das séries individuais.
```

SUBSTITUIR POR:

```latex
A comparação entre o FFORMA e o CREST merece um comentário à parte, porque é a única, entre as significativas, que coloca frente a frente duas estratégias de combinação. O FFORMA tem SMAPE médio menor (21,66\% contra 21,97\%) e vence também na contagem direta por série, com 107 das 182 séries a seu favor (58,8\%), contra 75 em que o CREST obtém SMAPE menor: aqui, ao contrário do padrão observado em outros conjuntos de dados, a vantagem média não depende de poucas vitórias de grande magnitude, mas se sustenta ao longo da maioria das séries.
```

No terceiro parágrafo, REMOVER `em 23 das 182 séries` e SUBSTITUIR POR `em 22 das 182 séries`.

#### E68. NN5

No primeiro parágrafo, REMOVER `com SMAPE médio de 11,77\%, à frente do ADE por uma margem pequena (11,78\%)` e SUBSTITUIR POR `com SMAPE médio de 11,57\%, à frente do ADE por uma margem pequena (11,78\%)`.

No primeiro parágrafo, REMOVER:

```
Nenhuma das quatro comparações envolvendo o CREST atinge significância estatística pelo teste de Wilcoxon, com valores de $p$ entre 0,10 e 0,89: a posição de destaque do CREST no ranqueamento pela média não pode ser considerada estatisticamente confiável frente a nenhum dos quatro métodos comparados neste conjunto de dados.
```

SUBSTITUIR POR:

```latex
Duas das quatro comparações envolvendo o CREST atingem significância estatística pelo teste de Wilcoxon: contra a média ($p = 0,0347$) e contra o CatBoost (Original + FT) ($p = 0,0375$), o melhor modelo individual; contra o ADE e o FFORMA, os valores de $p$ são 0,2598 e 0,0785. Nas duas comparações significativas, a vantagem do CREST na média se sustenta também na contagem de vitórias por série: o CREST obtém SMAPE menor que a média em 59 das 111 séries (53,2\%), contra 42 em que a média é melhor, e vence o CatBoost em 68 séries, contra 43.
```

REMOVER o segundo parágrafo inteiro:

```
Entre as dez comparações par a par possíveis, apenas uma atinge significância estatística: o FFORMA supera a média ($p \leq 0,0001$). Essa diferença, no entanto, é a menor diferença média observada em toda a matriz (0,0003 ponto percentual de SMAPE), e a contagem de vitórias por série mostra o padrão oposto ao que a significância sugere: a média obtém SMAPE menor que o FFORMA em 84 das 111 séries (75,7\%), contra apenas 27 séries em que o FFORMA é melhor. Assim como já observado na comparação entre o FFORMA e o CREST na ANP (Seção~\ref{subsec:anp-discussao}), uma diferença estatisticamente significativa na média não implica superioridade na maioria das séries individuais: aqui, a significância decorre da magnitude das vitórias do FFORMA nas poucas séries em que ele vence, não da frequência com que vence.
```

SUBSTITUIR POR:

```latex
Além dessas duas, apenas uma outra comparação atinge significância estatística: o FFORMA supera a média ($p \leq 0,0001$). É a menor diferença média observada em toda a matriz (0,03 ponto percentual de SMAPE), mas ela se sustenta ao longo das séries, e não apenas em poucas vitórias de grande magnitude: o FFORMA obtém SMAPE menor que a média em 84 das 111 séries (75,7\%), contra apenas 27 séries em que a média é melhor.
```

No terceiro parágrafo, REMOVER `em 9 das 111 séries` e SUBSTITUIR POR `em 10 das 111 séries`.

#### E69. M4

REMOVER `($p = 0,0005$ contra o CREST)` e SUBSTITUIR POR `($p = 0,0002$ contra o CREST)`.

REMOVER `(8,97\% contra 9,23\%), não é estatisticamente significativa ($p = 0,6794$)` e SUBSTITUIR POR `(8,97\% contra 9,15\%), não é estatisticamente significativa ($p = 0,9841$)`.

REMOVER `57 das 359 séries produzem` e SUBSTITUIR POR `63 das 359 séries produzem`.

#### E70. ETTh1

REMOVER:

```
De fato, oito das dez comparações são unânimes, com um dos dois métodos superando o outro nas sete séries simultaneamente; as duas exceções são o Random Forest (CWT) contra o CREST (6 vitórias a 1 para o Random Forest) e a mediana contra o FFORMA (6 vitórias a 1 para a mediana), ambas ainda assim estatisticamente significativas.
```

SUBSTITUIR POR:

```latex
De fato, sete das dez comparações são unânimes, com um dos dois métodos superando o outro nas sete séries simultaneamente; as três exceções são o Random Forest (CWT) contra o CREST (6 vitórias a 1 para o Random Forest), o CREST contra a mediana (6 a 1 para o CREST) e a mediana contra o FFORMA (6 a 1 para a mediana), todas ainda assim estatisticamente significativas.
```

REMOVER:

```
O CREST perde apenas para o Random Forest (CWT), tanto na média quanto na contagem direta (uma vitória em sete), e vence unanimemente a mediana, o FFORMA e o ADE nas sete séries.
```

SUBSTITUIR POR:

```latex
O CREST perde apenas para o Random Forest (CWT), tanto na média quanto na contagem direta (uma vitória em sete); contra a mediana, vence em seis das sete séries, e vence unanimemente o FFORMA e o ADE.
```

#### E71. ETTh2

REMOVER `com valores de $p$ entre 0,1562 e 0,5781` e SUBSTITUIR POR `com valores de $p$ entre 0,1562 e 0,9136`.

REMOVER:

```
a diferença média de SMAPE entre os dois é praticamente nula, e a contagem de vitórias por série é de quatro a três a favor do CatBoost, um resultado próximo do empate ($p = 0,5781$).
```

SUBSTITUIR POR:

```latex
a diferença média de SMAPE entre os dois é de apenas 0,05 ponto percentual, e a contagem de vitórias por série é de cinco a dois a favor do CatBoost ($p = 0,4688$).
```

REMOVER `em 4 das 7 séries` e SUBSTITUIR POR `em 5 das 7 séries`.

REMOVER `amostra reduzida a apenas 3 séries não empatadas entre os dois métodos, a ausência de significância estatística nesta comparação ($p = 0,4456$) já era esperada.` e SUBSTITUIR POR `amostra reduzida a apenas 2 séries não empatadas entre os dois métodos, a ausência de significância estatística nesta comparação ($p = 0,9136$) já era esperada.`

#### E72. ETTm1

REMOVER `um resultado próximo do empate em contagem direta ($p = 1,0000$)` e SUBSTITUIR POR `um resultado próximo do empate em contagem direta ($p = 0,9375$)`.

No mesmo parágrafo, REMOVER `incluindo as quatro que envolvem a mediana contra o FFORMA, o ADE e o CREST` e SUBSTITUIR POR `incluindo as três que envolvem a mediana contra o FFORMA, o ADE e o CREST`.

#### E73. ETTm2

REMOVER `o Random Forest (DWT), o melhor modelo individual; o CREST; o FFORMA; o ADE; e a mediana, ordenados pelo SMAPE médio entre as 7 séries.` e SUBSTITUIR POR `o Random Forest (DWT), o melhor modelo individual; o FFORMA; o ADE; a mediana; e o CREST, ordenados pelo SMAPE médio entre as 7 séries.`.

REMOVER `três delas com vitória nas sete séries simultaneamente` e SUBSTITUIR POR `duas delas com vitória nas sete séries simultaneamente`.

REMOVER `com valores de $p$ entre 0,1094 e 0,9375` e SUBSTITUIR POR `com valores de $p$ entre 0,1094 e 0,8125`.

## 2.4. Síntese entre conjuntos de dados

Tabela de ranking (E39), REMOVER a linha:

```
ETTm2  & 1 (Random Forest DWT)   & 5 (Mediana) & 3 & 4 & 2 \\
```

SUBSTITUIR POR:

```
ETTm2  & \textbf{1 (Random Forest DWT)}   & 4 (Mediana) & 2 & 3 & 5 \\
```

(E40) REMOVER a linha do rank médio:

```
\textbf{Rank médio} & \textbf{1,86} & 3,57 & 3,14 & 3,86 & 2,57 \\
```

SUBSTITUIR POR:

```
\textbf{Rank médio} & \textbf{1,86} & 3,43 & 3,00 & 3,71 & 3,00 \\
```

(E74) Tabela de ranking: negrito no melhor rank de cada linha. Nas seis linhas restantes da Tabela~\ref{tab:sintese-ranking}, o melhor valor de cada linha (o rank 1) passa a ficar em negrito. Substituir cada linha por:

```latex
ANP    & \textbf{1 (Theta)}                       & 4 (Mediana) & 2 & 3 & 5 \\
NN5    & 5 (CatBoost Original + FT)      & 4 (Média)   & 3 & 2 & \textbf{1} \\
M4     & 3 (Random Forest CWT)           & \textbf{1 (Mediana)} & 4 & 5 & 2 \\
ETTh1  & \textbf{1 (Random Forest CWT)}           & 3 (Mediana) & 4 & 5 & 2 \\
ETTh2  & \textbf{1 (CatBoost Original + CWT)}     & 3 (Mediana) & 4 & 5 & 2 \\
ETTm1  & \textbf{1 (Random Forest DWT)}           & 5 (Mediana) & 2 & 3 & 4 \\
```

Convenção: o negrito marca o melhor rank de cada linha (a posição 1); a linha do rank médio já usa a mesma convenção no E40.

(E41) REMOVER o parágrafo que começa em:

```
Nenhum método ocupa a primeira posição em todos os conjuntos de dados.
```

SUBSTITUIR POR:

```latex
Nenhum método ocupa a primeira posição em todos os conjuntos de dados. Descontado o melhor modelo individual, cuja posição de destaque no rank médio (1,86) reflete uma escolha retrospectiva, feita conhecendo o resultado de cada um dos dezenove modelos naquele conjunto de dados específico, e não uma decisão que poderia ser tomada de antemão, o CREST divide com o FFORMA o melhor rank médio entre os métodos comparáveis (3,00 cada), à frente da melhor combinação estática (3,43) e do ADE (3,71).
```

(E42) REMOVER o parágrafo que começa em:

```
Essa posição de destaque no rank médio reflete um padrão de consistência
```

SUBSTITUIR POR:

```latex
Essa posição reflete um padrão de consistência, não de dominância: o CREST ocupa a primeira ou a segunda posição em quatro dos sete conjuntos de dados (NN5, M4, ETTh1 e ETTh2), com três exceções, a quarta posição no ETTm1 e a última posição na ANP e no ETTm2. O FFORMA, com o mesmo rank médio (3,00), nunca ocupa a primeira posição em nenhum dos sete conjuntos, atingindo no máximo a segunda colocação (ANP, ETTm1 e ETTm2), mas também nunca ocupa a última; o ADE termina em último em três conjuntos (M4, ETTh1 e ETTh2). Em resumo, o CREST alterna entre o topo e as últimas posições, o FFORMA permanece no meio do ranqueamento e o ADE é penalizado pelas perdas de grande magnitude, ainda que vença com frequência em contagem direta.
```

Parágrafos da MCM agregada por SMAPE (E43):

REMOVER:

```
Por SMAPE, o CREST obtém o menor valor médio entre os cinco métodos (13,76\%), seguido de perto pela mediana (13,79\%) e pelo FFORMA (14,15\%), com o Random Forest (DWT) (14,28\%) e o ADE (14,70\%) nas duas últimas posições. Das dez comparações par a par, seis atingem significância estatística pelo teste de Wilcoxon. O CREST supera o Random Forest (DWT) ($p = 0,0174$) e o ADE ($p = 0,0017$) de forma estatisticamente significativa, mas não há diferença significativa entre o CREST, a mediana ($p = 0,8850$) e o FFORMA ($p = 0,9639$): os três formam um grupo de desempenho equivalente no topo do ranqueamento, seguido pelo Random Forest (DWT) e pelo ADE, que também não diferem significativamente entre si ($p = 0,2175$). A comparação entre o CREST e a mediana é a única desta matriz que registra empates: 103 das 680 séries produzem exatamente o mesmo SMAPE para os dois métodos.
```

SUBSTITUIR POR:

```latex
Por SMAPE, o CREST obtém o menor valor médio entre os cinco métodos (13,70\%), à frente da mediana (13,79\%) e do FFORMA (14,15\%), com o Random Forest (DWT) (14,28\%) e o ADE (14,70\%) nas duas últimas posições. Das dez comparações par a par, cinco atingem significância estatística pela Matriz de Comparação Múltipla. O CREST supera o Random Forest (DWT) ($p = 0,0023$) e o ADE ($p = 0,0002$), a mediana também supera esses dois ($p = 0,0048$ e $p = 0,0002$) e o FFORMA supera o ADE ($p \leq 0,0001$). Entre os três primeiros colocados não há diferença significativa (CREST contra a mediana, $p = 0,3780$; CREST contra o FFORMA, $p = 0,5371$; mediana contra o FFORMA, $p = 0,7840$), e a diferença entre o FFORMA e o Random Forest (DWT) fica no limite da significância ($p = 0,0644$). O Random Forest (DWT) e o ADE não diferem entre si ($p = 0,2175$). A comparação entre o CREST e a mediana é a única desta matriz que registra empates: 108 das 680 séries produzem exatamente o mesmo SMAPE para os dois métodos.
```

Parágrafos da MCM agregada por POCID (E44):

REMOVER:

```
Por POCID, o FFORMA obtém o maior valor médio (58,21\%), seguido pelo CREST (57,86\%), sem diferença estatisticamente significativa entre os dois ($p = 0,4045$). Os dois formam um grupo de desempenho equivalente à frente da mediana (57,20\%), do ADE (56,37\%) e do Random Forest (DWT) (56,16\%): tanto o FFORMA quanto o CREST superam o ADE e o Random Forest (DWT) com significância estatística, enquanto a diferença entre a mediana e o CREST, assim como entre o ADE e o Random Forest (DWT), não é significativa.
```

SUBSTITUIR POR:

```latex
Por POCID, o CREST obtém o maior valor médio (58,26\%), seguido de perto pelo FFORMA (58,21\%), sem diferença estatisticamente significativa entre os dois ($p = 0,9609$). Os dois formam um grupo de desempenho equivalente no topo do ranqueamento, à frente da mediana (57,20\%), do ADE (56,37\%) e do Random Forest (DWT) (56,16\%). O CREST supera o ADE ($p = 0,0024$) e o Random Forest (DWT) ($p = 0,0014$) com significância estatística; o FFORMA supera a mediana ($p = 0,0392$), o ADE ($p = 0,0001$) e o Random Forest (DWT) ($p = 0,0003$); a mediana supera o Random Forest (DWT) ($p = 0,0369$); e as diferenças da mediana contra o CREST ($p = 0,1131$) e do ADE contra o Random Forest (DWT) ($p = 0,4926$) não são significativas.
```

(E45) REMOVER o parágrafo que começa em:

```
Em conjunto, as duas matrizes agregadas indicam que o CREST integra
```

SUBSTITUIR POR:

```latex
Em conjunto, as duas matrizes agregadas indicam que o CREST lidera as duas métricas na comparação com a totalidade das 680 séries (13,70\% de SMAPE e 58,26\% de POCID), integrando o grupo de melhor desempenho estatisticamente indistinguível da mediana e do FFORMA por SMAPE e do FFORMA por POCID, e superando o ADE e o melhor modelo individual isolado nas duas métricas. É importante ressalvar, no entanto, que essa vantagem não se repete de forma uniforme conjunto a conjunto: o CREST ocupa a última posição na ANP e no ETTm2 e fica atrás de modelos individuais isolados no ETTh1 e no ETTm1, o que indica que seu ganho médio decorre de um desempenho relativamente estável entre a maioria dos conjuntos de dados, e não de uma superioridade absoluta em cada um deles.
```

Tabela de contagem de vitórias: substituir as cinco linhas (valores regenerados com a configuração final):

(E46) REMOVER `Random Forest (DWT) & 201 & 29,56 \\` e SUBSTITUIR POR `Random Forest (DWT) & \textbf{199} & \textbf{29,26} \\`.

(E47) REMOVER `CREST                & 152 & 22,35 \\` e SUBSTITUIR POR `CREST                & 147 & 21,62 \\`.

(E48) REMOVER `ADE                  & 130 & 19,12 \\` e SUBSTITUIR POR `ADE                  & 126 & 18,53 \\`.

(E49) REMOVER `FFORMA               & 103 & 15,15 \\` e SUBSTITUIR POR `FFORMA               & 108 & 15,88 \\`.

(E50) REMOVER `Mediana              & 94  & 13,82 \\` e SUBSTITUIR POR `Mediana              & 100 & 14,71 \\`.

Convenção desta tabela: o negrito marca a maior contagem, o melhor resultado da tabela (Random Forest (DWT), 199 e 29,26\%).

(E51) REMOVER o parágrafo que começa em:

```
O Random Forest (DWT), apesar de ser um único modelo fixo
```

SUBSTITUIR POR:

```latex
O Random Forest (DWT), apesar de ser um único modelo fixo aplicado indistintamente às 680 séries de domínios tão distintos quanto demanda de combustíveis, séries semanais do M4 e temperatura de transformadores elétricos, vence em 199 delas, a maior contagem entre os cinco métodos. Isso não contradiz o resultado da MCM agregada da Seção anterior, em que o CREST obtém o menor SMAPE médio: um único modelo pode vencer com mais frequência, série a série, sem que isso implique o menor erro médio, caso suas derrotas sejam, em média, maiores em magnitude do que suas vitórias. A ordem obtida pela contagem de vitórias também não coincide inteiramente com a ordem do rank médio da Tabela~\ref{tab:sintese-ranking}: o ADE vence em mais séries que o FFORMA (126 contra 108), mesmo tendo o pior rank médio entre os quatro métodos comparáveis (3,71 contra 3,00) e o pior SMAPE médio agregado da Seção anterior. Esse padrão, já observado repetidamente ao longo deste capítulo em conjuntos de dados individuais, volta a se confirmar na agregação: o desempenho médio do ADE é penalizado por perdas de grande magnitude, ainda que vença com mais frequência do que o FFORMA em termos absolutos.
```

Parágrafos do erro por passo do horizonte:

(E52) REMOVER:

```
No painel superior da Figura~\ref{fig:erro-horizonte-smape}, o SMAPE médio dos cinco métodos cresce de forma moderada e aproximadamente paralela entre o primeiro e o oitavo passo, sem inversões de ordem relevantes: o CREST mantém a segunda posição entre os cinco métodos em ambas as extremidades da faixa (9,26\% no primeiro passo, 14,69\% no oitavo), atrás apenas da mediana no primeiro passo (9,17\%) e do Random Forest (DWT) no oitavo (14,97\%). O painel inferior mostra que o desvio-padrão cresce de forma acompanhada à média, e de maneira muito semelhante entre os cinco métodos, sem que nenhum se distinga como consistentemente mais ou menos variável que os demais ao longo do horizonte.
```

SUBSTITUIR POR:

```latex
No painel superior da Figura~\ref{fig:erro-horizonte-smape}, o SMAPE médio dos cinco métodos cresce de forma moderada e aproximadamente paralela entre o primeiro e o oitavo passo, e o CREST mantém a primeira posição entre os cinco métodos em todos os oito passos (9,04\% no primeiro passo, 14,54\% no oitavo). A mediana ocupa a segunda posição no primeiro passo (9,17\%) e o Random Forest (DWT), a segunda no oitavo (14,97\%). O painel inferior mostra que o desvio-padrão cresce de forma acompanhada à média, e de maneira muito semelhante entre os cinco métodos, sem que nenhum se distinga como consistentemente mais ou menos variável que os demais ao longo do horizonte.
```

(E53) REMOVER:

```
No painel superior da Figura~\ref{fig:erro-horizonte-pocid}, o POCID médio não apresenta a mesma tendência de crescimento monótono observada no SMAPE: decresce de forma leve ao longo dos oito passos para a maioria dos métodos, de valores entre 59,85\% e 66,18\% no primeiro passo para valores entre 55,29\% e 63,38\% no oitavo. O CREST acompanha essa tendência (65,88\% no primeiro passo, 59,85\% no oitavo), posicionado entre a mediana, que mantém o maior POCID em ambas as extremidades, e o ADE, que mantém o menor. O painel inferior mostra um desvio-padrão elevado e praticamente constante ao longo de todo o horizonte, em torno de 48 pontos percentuais para os cinco métodos, reflexo da natureza binária dessa métrica por observação (acerto ou erro de direção).
```

SUBSTITUIR POR:

```latex
No painel superior da Figura~\ref{fig:erro-horizonte-pocid}, o POCID médio não apresenta a mesma tendência de crescimento monótono observada no SMAPE: decresce de forma leve ao longo dos oito passos para a maioria dos métodos, de valores entre 59,85\% e 66,32\% no primeiro passo para valores entre 55,29\% e 63,38\% no oitavo. O CREST lidera no primeiro passo (66,32\%) e recua para 60,59\% no oitavo, segundo lugar empatado com a mediana, atrás do FFORMA, que assume a maior taxa no fim do horizonte (63,38\%); o ADE mantém o menor POCID nos dois extremos. O painel inferior mostra um desvio-padrão elevado e praticamente constante ao longo de todo o horizonte, em torno de 49 pontos percentuais para os cinco métodos, reflexo da natureza binária dessa métrica por observação (acerto ou erro de direção).
```

## 2.5. Análise do comportamento do agente

Regra desta seção: substituir a **subseção inteira** do `resultados.tex` (do `\subsection{...}` até o `\subsection{...}` seguinte) pelo bloco indicado.

### 2.5.1. Frequência de uso do catálogo de ações — E54

SUBSTITUIR a subseção inteira por:

```latex
\subsection{Frequência de uso do catálogo de ações}

As onze ações oferecidas ao agente no protocolo adotado foram todas chamadas ao menos uma vez ao longo das 680 séries, além do aceite; a única ação registrada e não oferecida é o peso por mínimos quadrados, que exige mais janelas de validação do que as três disponíveis. A Tabela~\ref{tab:agente-acoes} apresenta a contagem total de chamadas de cada ação.

\begin{table}[H]
\centering
\caption{Contagem total de chamadas por ação, entre as 680 séries.}
\label{tab:agente-acoes}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Ação} & \textbf{Chamadas} \\
\midrule
Peso pelo erro inverso                 & 366 \\
Avaliação da estratégia                & 338 \\
Poda de redundância                    & 324 \\
Peso pela seleção gulosa               & 85 \\
Seleção pelos $k$ melhores             & 55 \\
Peso pela média dos melhores por passo & 33 \\
Peso pelo meta-modelo                  & 30 \\
Seleção pelos mais estáveis            & 17 \\
Peso pelo melhor preditor por passo    & 17 \\
Seleção gulosa de subconjunto          & 12 \\
Peso pela tendência do erro            & 10 \\
\bottomrule
\end{tabular}
\end{table}

O peso pelo erro inverso é o critério de ponderação mais utilizado, seguido da avaliação da estratégia, o único caminho pelo qual uma tentativa entra no histórico, e da poda de redundância. Entre as ações de seleção de subconjunto, a seleção pelos $k$ melhores lidera, à frente da seleção pelos mais estáveis e da seleção gulosa; entre as receitas adicionais de ponderação, a seleção gulosa e a média dos melhores por passo aparecem mais que o melhor preditor por passo. O equilíbrio entre exploração e preparação acompanha o comportamento conservador do agente neste protocolo: em média, pouco mais de meia avaliação de estratégia por série.
```

(E54, adição opcional) Para explicar a concentração das três ações mais usadas com base nos pensamentos do agente, ADICIONAR ao final da subseção:

```latex
Uma leitura dos pensamentos que antecedem cada chamada, combinada à distribuição das ações entre as séries, ajuda a explicar essa concentração. Das 680 séries, 225 (33\%) terminam no aceite de uma estratégia já semeada sem nenhuma chamada adicional, e a mediana de passos por série é de 2: a tabela reflete, portanto, o que o agente faz quando decide ir além das sementes, e nesse subconjunto o roteiro típico de intervenção é podar, ponderar e avaliar. A poda de redundância é a ação de seleção mais usada (304 séries, 45\%) e tende a abrir o ciclo: em 74\% das séries que a usam, ela é a primeira chamada. Em 88\% das suas chamadas, os pensamentos citam erros correlacionados, 86\% citam redundância e 42\% descrevem o ranqueamento como instável, como neste exemplo real, em uma série da ANP: \textit{"The series has unstable ranking and many correlated errors; pruning redundant models should reduce noise, and then inverse-error weighting on the cleaned pool may improve performance."} O peso pelo erro inverso é a receita de ponderação padrão do roteiro (269 séries, 40\%): em 85\% dos pensamentos que o antecedem a própria expressão aparece, 92\% citam o erro dos preditores e 57\% mencionam a média vigente, quase sempre como a referência a ser batida. A avaliação da estratégia, única via de entrada no histórico de tentativas, ocorre em 194 séries (29\%), justamente aquelas em que o agente testa uma proposta: entre elas, 70\% passam também pela poda e 77\% pelo erro inverso, e 64\% dos pensamentos que antecedem a avaliação citam pesos, refletindo o padrão de testar uma receita de ponderação e registrá-la. Nas ações menos frequentes, o motivo é mais específico: a seleção pelos mais estáveis aparece com o ranqueamento instável citado em 76\% dos casos, o peso pela tendência do erro cita a tendência em 100\% deles, o peso pelo meta-modelo cita o conhecimento entre séries em 90\%, e as duas receitas por passo do horizonte citam o horizonte em 100\% deles. A concentração do catálogo, portanto, não indica preferência do agente por ferramentas específicas: ela reflete o roteiro de poda, ponderação e avaliação das séries em que o agente decide ir além das sementes.
```

Fonte: `outputs/resultados/item_10/motivacao_acoes.md` (script `outputs/resultados/scripts/motivacao_acoes.py`). As contagens por ação das trajetórias batem exatamente com a Tabela~\ref{tab:agente-acoes} quando se excluem as 9 chamadas com erro de argumento registradas (contagens por série verificadas à parte).

### 2.5.2. Motivo de encerramento do ciclo — E55

SUBSTITUIR a subseção inteira por:

```latex
\subsection{Motivo de encerramento do ciclo} \label{subsec:motivo-encerramento}

A Tabela~\ref{tab:agente-encerramento} apresenta a distribuição dos três critérios de encerramento do ciclo de decisão, descritos na Seção~\ref{sec:agente}, entre as 680 séries.

\begin{table}[H]
\centering
\caption{Motivo de encerramento do ciclo, entre as 680 séries.}
\label{tab:agente-encerramento}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Motivo} & \textbf{Séries} & \textbf{\%} \\
\midrule
Aceite explícito       & 671 & 98,68 \\
Estagnação            & 8   & 1,18  \\
Orçamento esgotado    & 1   & 0,15  \\
\bottomrule
\end{tabular}
\end{table}

O aceite explícito encerra a quase totalidade das séries (98,68\%), e a estagnação responde pelas oito restantes, distribuídas entre ANP, NN5, M4 e ETTh1. O esgotamento do orçamento de iterações, a rede de segurança descrita na Seção~\ref{sec:agente}, ocorre em uma única série (NN5), o que indica que o orçamento estabelecido é, na prática, mais que suficiente: o agente encerra o ciclo por decisão própria bem antes do limite.
```

(E55, adição opcional) Para comentar o caso da única série que esgota o orçamento, ADICIONAR ao final da subseção:

```latex
Vale registrar o comportamento da única série que esgota o orçamento, no NN5. O agente encontra uma liderança logo na segunda iteração, ao testar o peso pelo erro inverso sobre um subconjunto estável (pontuação 0,6586), e em seguida testa, sem superá-la, uma média dos melhores por passo, uma mediana e um peso pelo erro inverso sobre o conjunto podado. Como o líder não se separa com margem confiável, o ciclo continua e, na décima iteração, o agente encontra uma nova melhor pontuação, com o peso pela tendência do erro (0,6582); a melhora marginal reinicia a janela de estagnação, a tentativa seguinte não supera o novo líder e o ciclo atinge o limite de doze iterações antes de acumular quatro avaliações consecutivas sem melhora. A decisão final recai sobre a melhor tentativa do histórico, como previsto pelo mecanismo: o teto de iterações funciona como a rede de segurança que encerra o ciclo mesmo quando o agente preferiria continuar explorando.
```

Fonte: trajetória da série 27 do NN5 no run final (`react_trajectory_json`), 12 iterações, motivo `iteration_budget_exhausted`; conferido contra `outputs/resultados/item_11/motivo_encerramento.csv`.

### 2.5.2b. Nova subseção: passo a passo de uma decisão — E75

ADICIONAR como nova subseção logo depois de "Motivo de encerramento do ciclo" (E55) e antes de "Estudo de caso ilustrativo" (E56). Ela percorre a trajetória completa da série 206 do M4, a mesma do caso de sucesso da Subseção 2.5.4, desde as sementes até o aceite, com o pensamento de cada iteração. A numeração das subseções seguintes se ajusta automaticamente no LaTeX.

SUBSTITUIR POR:

```latex
\subsection{Passo a passo de uma decisão: da semente ao aceite} \label{subsec:transcricao-decisao}

Esta subseção acompanha, iteração a iteração, a decisão do agente para a série 206 do M4, a mesma do caso de sucesso da Subseção~\ref{subsec:interpretabilidade-previsoes}, desde o ponto de partida determinístico até o aceite final. Os pensamentos são apresentados em tradução livre na Tabela~\ref{tab:transcricao-decisao}, e os trechos decisivos são citados literalmente no texto. Todos os escores e posições citados até o aceite são de validação, calculados sobre as três janelas usadas na decisão; as métricas de teste aparecem apenas no fecho desta subseção. Para evitar ambiguidade, os identificadores são os registrados pelo próprio método: pool1, pool2 e pool3 são os subconjuntos dos 5, 7 e 9 modelos mais estáveis, preparados antes do ciclo, e pool4 em diante são os subconjuntos criados pelas ações de seleção durante o ciclo (na série, o pool3 reúne os 9 modelos mais estáveis e a primeira poda dá origem ao pool4, com 7); w$N$ denota um vetor de pesos calculado na série, e a$N$ denota a $N$-ésima tentativa do histórico, com as dez sementes ocupando a1 a a10 e a estratégia final sendo sempre a de melhor pontuação entre todas.

Antes de o agente atuar, dez estratégias fixas são avaliadas e registradas no histórico da série. A melhor delas é a média aparada sobre o pool3 (a9), com pontuação de validação 0,4099, à frente da média (0,4177), da mediana (0,4385) e da combinação ponderada semeada (0,6516); o DBA fica bem atrás (0,8106). É esse piso de 0,4099 que o agente precisa superar.

\begin{table}[H]
\centering
\caption{Transcrição da trajetória de decisão do agente na série 206 do M4.}
\label{tab:transcricao-decisao}
\footnotesize
\begin{tabular}{@{}cp{6.2cm}p{3.8cm}p{3.2cm}@{}}
\toprule
\textbf{Iter.} & \textbf{Pensamento (tradução livre)} & \textbf{Ação} & \textbf{Resultado (validação)} \\
\midrule
1 & O líder é uma média aparada no pool3, mas o diagnóstico sugere que uma combinação ponderada pode melhorar; testar o peso pelo erro inverso no pool3 com pequeno encolhimento para reduzir a variância das três janelas. & Peso pelo erro inverso (pool3, encolhimento 0,1) & w2 registrado (9 de 9 modelos) \\
2 & O líder é uma média aparada no pool3, mas a correlação de erros mostra grupos redundantes; podar deve reduzir o ruído e permitir que a ponderação supere a média aparada. & Poda de redundância (pool3) & pool4: 9 para 7 modelos \\
3 & Testar a combinação ponderada com o w2 no pool3 para ver se a média ponderada regularizada supera a média aparada. & Avaliação do w2 (pool3) & 2º de 11 (0,4142); líder: a9 (média aparada no pool3) \\
4 & Reduzir os erros correlacionados podando e reponderando o conjunto reduzido; o erro inverso no pool4 pode superar o líder. & Peso pelo erro inverso (pool4, encolhimento 0) & w3 registrado (7 de 7 modelos) \\
5 & O líder é a média aparada no pool3; uma média ponderada com os pesos do w3 no pool4 pode reduzir a variância, já que o pool4 contém os modelos mais estáveis e não redundantes. & Avaliação do w3 (pool4) & 1º de 12 (0,378; melhor até aqui) \\
6 & O melhor até aqui (ponderação no pool4 com w3) é estatisticamente indistinguível do segundo colocado e nenhuma outra estratégia deve superá-lo. & Aceite & a12 aceito \\
\bottomrule
\end{tabular}
\end{table}

O primeiro pensamento já antecipa o desfecho: \textit{"The current leader is a trimmed-mean on pool3, but the diagnostics suggest a weighted combination could improve performance. I'll try inverse-error weights on pool3 with a small shrinkage to reduce variance from only three windows."} A decisão que muda a liderança vem na quinta iteração: \textit{"The current leader is a trimmed-mean on pool3. A weighted average using the inverse-error weights on the pruned pool4 might reduce variance and improve over the trimmed-mean, especially since pool4 contains the most stable, non-redundant models."} E o aceite é acompanhado de \textit{"The current best weighted pool4 w3 is statistically indistinguishable from the runner-up, and no further strategy is likely to beat it."}, com a justificativa registrada: \textit{"Weighted pool4 w3 achieves the lowest score and is not statistically worse than the runner-up; accepting it is the prudent choice."}

Em seis iterações e duas avaliações, o agente substitui a média aparada semeada por uma combinação ponderada pelo erro inverso sobre o subconjunto podado. A Tabela~\ref{tab:transcricao-validacao-teste} compara as duas estratégias nas duas janelas.

\begin{table}[H]
\centering
\caption{Pontuação de validação e métricas de teste da melhor semente e da estratégia aceita, na série 206 do M4.}
\label{tab:transcricao-validacao-teste}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Estratégia} & \textbf{Pontuação (validação)} & \textbf{SMAPE (teste, \%)} & \textbf{POCID (teste, \%)} \\
\midrule
Melhor semente (a9)     & 0,4099 & 10,93 & \textbf{50,00} \\
Estratégia aceita (a12) & \textbf{0,3780} & \textbf{10,37} & \textbf{50,00} \\
\bottomrule
\end{tabular}
\end{table}

A melhora de 0,0319 na validação se confirmou na direção no teste: o SMAPE cai de 10,93\% na melhor semente para 10,37\% na estratégia aceita, uma diferença de 0,56 ponto percentual, enquanto o POCID permanece idêntico (50,00\%). A trajetória ilustra o padrão descrito nas subseções anteriores: o agente parte do diagnóstico, testa a receita de ponderação sobre o conjunto disponível, compara cada proposta com o líder vigente e aceita a melhor pontuação do histórico quando a margem é indistinguível.
```

Fonte: trajetória da série 206 do M4 no run final (`react_trajectory_json`) e `baseline_results_json` (escores das sementes); 6 iterações, 2 avaliações, veredito `indistinguishable`.

(E75, adição opcional) Se quiser deixar explícito o conteúdo dos pools desta série, acrescente ao fim da frase da notação, depois de "melhor pontuação entre todas.", a nota de rodapé:

```latex
\footnote{Para a série 206, o pool1 reúne DWT\_rf, CWT\_catboost, FT\_catboost, ONLY\_DWT\_catboost e ONLY\_DWT\_rf; o pool2 acrescenta rf e FT\_rf; o pool3 acrescenta DWT\_catboost e ONLY\_CWT\_catboost; e a poda da segunda iteração remove rf e ONLY\_CWT\_catboost, deixando o pool4 com os sete modelos restantes.}
```

### 2.5.3. Estudo de caso ilustrativo — E56 (texto final nesta revisão)

ATENÇÃO: com a configuração final, as séries 2 e 1 da ANP usadas no texto atual terminaram com estratégia de origem semeadas, não do agente. Casos escolhidos: ANP 8 (origem semeadas, aceitou a mediana a2) e ANP 5 (origem no agente, mediana sobre o subconjunto podado que ele mesmo construiu; maior melhoria da ablação, −2,09pp). Trajetórias completas em `outputs/resultados/item_15/candidatos_estudo_caso.json`.

REMOVER a subseção inteira, do `\subsection{Estudo de caso ilustrativo}` até a linha anterior ao `\subsection{Interpretabilidade das previsões...}`.

SUBSTITUIR POR:

```latex
\subsection{Estudo de caso ilustrativo} \label{subsec:estudo-caso-trajetoria}

Esta subseção ilustra o funcionamento do ciclo de decisão do agente a partir de duas séries reais da ANP: uma em que a estratégia final aceita já constava entre as sementes pré-testadas (série de índice 8), e outra em que a estratégia final resultou de um subconjunto construído pelo próprio agente (série de índice 5).

Na série de índice 8, o agente explorou, ao longo de onze iterações, pesos pelo erro inverso sobre o conjunto podado, pesos pelo meta-modelo entre séries, pesos pela seleção gulosa e pesos pela tendência do erro, além de uma segunda poda do conjunto completo; as duas alternativas avaliadas nas janelas de validação ficaram atrás da mediana, e o agente aceitou a estratégia a2, a mediana sobre o conjunto completo de preditores, que já constava entre as sementes pré-testadas. Nenhuma das alternativas exploradas nas iterações seguintes a superou.

Na série de índice 5, o agente também iniciou pela poda de redundância, reduzindo o conjunto de dezenove para doze modelos; avaliou de imediato a mediana sobre o subconjunto podado, que assumiu a liderança, e calculou em seguida pesos pelo meta-modelo, pesos pelo erro inverso com dois níveis de encolhimento e pesos pela seleção gulosa, que reteve um único modelo em doze. Aplicou então uma segunda poda, de doze para dez modelos; a mediana sobre esse subconjunto final voltou a liderar e foi aceita como estratégia final (a12), de origem no próprio agente, e não entre as sementes. Na avaliação final, essa decisão superou a versão do método sem o agente em 2,09 pontos percentuais de SMAPE, a maior melhoria entre as 680 séries deste estudo.

Em ambos os casos, a justificativa textual do agente aponta a robustez da mediana como motivo da escolha final e descreve a margem do líder como estatisticamente indistinguível da do segundo colocado. Essa coincidência, à primeira vista, poderia sugerir um padrão de vocabulário fixo entre justificativas; a Seção~\ref{subsec:rastreabilidade} apresenta uma análise quantitativa dessa questão sobre a totalidade das séries com estratégia final de origem no agente, não apenas estes dois casos.

Os dois casos ilustram o mesmo padrão observado de forma agregada nas seções seguintes: o ciclo de raciocínio do agente frequentemente envolve testar múltiplas hipóteses de ponderação sobre subconjuntos podados do conjunto de preditores antes de convergir para uma decisão final, que pode ou não superar o que já estava disponível antes de o agente começar a atuar.
```

### 2.5.4. Interpretabilidade das previsões — E57 (texto final nesta revisão)

Casos escolhidos: M4 206 (sucesso; maior vantagem do CREST sobre as demais combinações entre as 47 séries de origem no agente: SMAPE 10,37\% contra 13,90\% do segundo melhor, com série real sem oscilações extremas) e ANP 13 (fracasso; queda e recuperação que a combinação não acompanha, SMAPE 21,55\% com as cinco estratégias na mesma faixa; escolhida por ter oscilações bem menos extremas do que a antiga ANP 64). O script `interpretabilidade_casos.py` foi atualizado (séries novas, curvas de ADE, FFORMA, mediana e DBA, fontes ampliadas, modelos individuais no painel superior, como antes, e painel inferior de ampliação na faixa das combinações, sem os individuais) e gerou os JSONs e as figuras em `outputs/resultados/interpretabilidade/figuras/`, já copiadas para `overleaf/resultados_final/`.

REMOVER a subseção inteira, do `\subsection{Interpretabilidade das previsões...}` até a linha anterior ao `\subsection{Distribuição da melhoria relativa}`, incluindo o parágrafo comentado (%) que antecede a próxima subseção.

SUBSTITUIR POR:

```latex
\subsection{Interpretabilidade das previsões: um caso de sucesso e um de fracasso} \label{subsec:interpretabilidade-previsoes}

Para ilustrar concretamente como a estratégia final se traduz em previsões, esta subseção apresenta duas outras séries em extremos opostos de desempenho, uma do M4 semanal (série 206) e uma da ANP (série 13), ambas com estratégia final de origem no próprio agente, e ambas combinando múltiplos preditores por peso, não um único modelo isolado.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{images/resultados_final/caso_sucesso.png}
    \caption{Previsões individuais, valor real, combinação final do CREST e demais métodos de combinação na série 206 do M4, a série com a maior vantagem do CREST sobre os demais métodos de combinação entre as séries com estratégia de origem no agente; o painel inferior amplia a faixa das estratégias de combinação.}
    \label{fig:caso-sucesso}
\end{figure}

Na série 206 do M4 (Figura~\ref{fig:caso-sucesso}), o CREST obtém SMAPE de 10,37\%, combinando sete modelos (Random Forest e CatBoost sobre as transformações wavelet contínua e discreta e de Fourier) por média ponderada, com pesos médios entre 0,11 e 0,18 e sem concentração em um único modelo. As sete previsões individuais acompanham o nível geral da série, mas divergem entre si nos picos e vales ao longo do horizonte; a combinação final segue a tendência de alta de forma mais fiel que as demais estratégias de combinação, e é o único método cujo intervalo de previsão se estende acima de 14.000, aproximando-se do patamar atingido pelo valor real, enquanto todas as demais combinações permanecem abaixo de 13.500. No pico isolado do sexto passo, nenhuma previsão de combinação acompanha o salto da série, um dos poucos momentos em que o CREST se afasta do valor observado. As demais estratégias de combinação ficam bem atrás nesta série: contra 10,37\% do CREST, o ADE obtém 13,90\%, o FFORMA, 14,15\%, a mediana, 15,39\%, e o DBA, 21,87\%. A justificativa real do agente para este caso foi: \textit{"Weighted pool4 w3 achieves the lowest score and is not statistically worse than the runner-up; accepting it is the prudent choice."}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{images/resultados_final/caso_fracasso.png}
    \caption{Previsões individuais, valor real, combinação final do CREST e demais métodos de combinação na série 13 da ANP, um dos maiores SMAPE entre as séries com estratégia de origem no agente; o painel inferior amplia a faixa das estratégias de combinação.}
    \label{fig:caso-fracasso}
\end{figure}

Na série 13 da ANP (Figura~\ref{fig:caso-fracasso}), o CREST obtém SMAPE de 21,55\%, combinando nove modelos (ARIMA e ETS, Random Forest e CatBoost em representações originais e transformadas, e o Naive Seasonal) por média ponderada, com pesos médios entre 0,09 e 0,14 e sem concentração em um único modelo. A figura revela por que a combinação não teve sucesso: o valor real recua de aproximadamente 1.540 para pouco mais de 900 nos primeiros passos, recupera-se parcialmente e volta a subir até 1.884 no fim do horizonte, enquanto a maioria das previsões individuais permanece numa faixa estreita e a combinação final fica confinada entre 1.370 e 1.536, sem reproduzir nem o vale nem a recuperação. A ponderação pelo erro inverso, com pesos espalhados por nove modelos, dilui os poucos preditores que acompanham a mudança de nível e resulta em uma previsão suavizada que não segue nenhum dos extremos; o melhor modelo individual isolado, o ETS, teria obtido 16,31\% de SMAPE nesta série. As demais estratégias de combinação erram na mesma faixa do CREST: FFORMA (21,17\%), DBA (21,82\%), mediana (21,98\%) e ADE (22,27\%), e nenhuma delas reproduz os movimentos da série. Nesse caso, a justificativa do agente foi: \textit{"trend-dominated series with unstable ranking; the weighted pool4 w2 strategy gives the lowest RMSE and is statistically indistinguishable from the runner-up, so it is the optimal choice."}
```

### 2.5.5. Distribuição da melhoria relativa — E58

SUBSTITUIR o parágrafo inteiro por:

```latex
Para cada uma das 680 séries, comparou-se a pontuação da estratégia final com a menor pontuação entre as estratégias semeadas, segundo o mecanismo descrito na Seção~\ref{subsec:pontuacao} do Capítulo~\ref{chp:proposta}. Em nenhuma das 680 séries a estratégia final apresentou pontuação pior que a melhor semente (0\%), confirmando empiricamente, sobre a totalidade das séries deste estudo, a garantia de que a estratégia final nunca é pior que o piso estabelecido pelas sementes, descrita na Seção~\ref{sec:aplicacao}. Em 351 séries (51,6\%), a estratégia final iguala essa melhor semente, dentro da precisão de registro dos escores. Nas demais 329 séries (48,4\%), a estratégia final apresenta pontuação estritamente melhor, o que reflete tanto propostas do agente quanto a escolha entre as estratégias de estabilidade semeadas. A maior melhoria observada ocorre em uma série do M4, na qual a pontuação final é 0,345 menor que a da melhor semente.
```

(E58, adição opcional) Para detalhar a distribuição dos ganhos em tabela, ADICIONAR depois do parágrafo:

```latex
\begin{table}[H]
\centering
\caption{Distribuição das 329 séries com melhoria estrita, por faixa de ganho na pontuação.}
\label{tab:melhoria-faixas}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Faixa de ganho} & \textbf{Séries} & \textbf{\% das 680} \\
\midrule
até 0,001        & 14  & 2,1  \\
0,001 a 0,01     & 108 & 15,9 \\
0,01 a 0,05      & 150 & 22,1 \\
0,05 a 0,10      & 38  & 5,6  \\
acima de 0,10    & 19  & 2,8  \\
\bottomrule
\end{tabular}
\end{table}

A Tabela~\ref{tab:melhoria-faixas} detalha esses ganhos: a maioria das 329 séries com melhoria estrita apresenta ganho de até 0,05 ponto de pontuação (272 séries, 40\% das 680), e apenas 19 séries (2,8\%) superam 0,10; entre as 329, a mediana do ganho é de 0,017 e as dez maiores melhorias concentram 20\% do ganho total.
```

Fonte: `outputs/resultados/item_16/melhoria_relativa.csv` (versão de 16/10; não usar `item_16_melhoria_relativa.csv`, que é da rodada antiga). Tolerância de 1e-4 para os empates, compatível com a precisão de registro dos escores.

### 2.5.6. Custo computacional — E59

SUBSTITUIR a tabela e o parágrafo pelos valores finais do primeiro modelo:

```latex
\begin{table}[H]
\centering
\caption{Tempo médio de decisão por série, por conjunto de dados.}
\label{tab:agente-custo}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Conjunto de dados} & \textbf{Tempo médio (s)} & \textbf{Desvio-padrão (s)} \\
\midrule
ETTm1  & 6,19  & 7,70  \\
ETTm2  & 7,69  & 9,42  \\
M4     & 8,26  & 8,68  \\
NN5    & 10,04 & 12,46 \\
ANP    & 10,32 & 13,94 \\
ETTh2  & 14,34 & 7,48  \\
ETTh1  & 14,43 & 21,97 \\
\bottomrule
\end{tabular}
\end{table}

O tempo médio de decisão varia de aproximadamente 6 a 14 segundos por série, dependendo do conjunto de dados, um custo baixo quando comparado ao tempo de execução dos modelos individuais, das combinações estáticas e dos métodos de meta-aprendizagem da literatura. O desvio-padrão é, em vários conjuntos, comparável à própria média, o que indica grande variação no tempo de decisão entre séries de um mesmo conjunto de dados, compatível com o número variável de iterações consumidas antes do encerramento do ciclo, descrito na Seção~\ref{subsec:motivo-encerramento}.
```

### 2.5.7. Rastreabilidade da decisão do agente — E60

SUBSTITUIR os dois parágrafos por:

```latex
Além da comparação de acurácia agregada discutida nas seções anteriores, o CREST oferece uma vantagem qualitativa que os métodos de meta-aprendizagem revisados no Capítulo~\ref{chp:revisao} não replicam: cada decisão de combinação é acompanhada de um histórico de tentativas e de uma justificativa textual explícita, auditável a posteriori, como ilustrado nas Seções~\ref{subsec:estudo-caso-trajetoria} e~\ref{subsec:interpretabilidade-previsoes} desta seção. Nem o FFORMA nem o ADE produzem um artefato equivalente: suas decisões de combinação resultam de um modelo estatístico ou de pesos aprendidos, sem uma explicação interpretável do motivo pelo qual uma determinada combinação foi escolhida para uma série específica.

Essa auditabilidade é particularmente relevante em contextos de aplicação em que a decisão de combinação precisa ser justificada a um usuário humano, ou em que a checagem manual das séries de maior impacto é viável, dois cenários em que os métodos de meta-aprendizagem tradicionais oferecem pouco além do valor previsto pelo modelo. Essa vantagem, no entanto, deve ser lida com uma ressalva, quantificada sobre as 47 séries cuja estratégia final tem origem no próprio agente. A similaridade textual entre pares dessas justificativas, medida pelo coeficiente de Jaccard \citep{jaccard1912} (a razão entre a interseção e a união dos conjuntos de palavras de conteúdo das duas justificativas), é de 0,13 em média entre todos os pares de séries, e sobe para 0,24 entre pares que compartilham exatamente a mesma estratégia, contra 0,13 entre pares com estratégias diferentes: uma diferença pequena, mas na direção esperada, indicando que o texto acompanha, ainda que fracamente, a decisão real, e não é puro ruído desconectado dela.

Não há repetição de frases longas: nenhum trecho de cinco a oito palavras aparece em mais de 20\% das justificativas. A abertura, no entanto, é mais repetitiva que o corpo: por correspondência exata, as primeiras frases se distribuem em 46 moldes distintos entre as 47 séries, e, agrupando por quase-duplicata, 30 moldes cobrem 38 das 47 séries. Esse padrão é compatível com o vocabulário fechado do diagnóstico interpretativo descrito na Seção~\ref{sec:diagnostico} do Capítulo~\ref{chp:proposta}, que entrega ao agente quatro categorias de vocabulário fechado (regime, previsibilidade, indicação de combinação, riscos), preenchidas no diagnóstico a partir das evidências: parte da semelhança estrutural entre justificativas é, portanto, esperada por construção, e não indica necessariamente que o agente ignore as particularidades de cada série ao redigir sua justificativa.
```

Se usar a citação, acrescente ao `.bib` do Overleaf:

```bibtex
@article{jaccard1912,
  author  = {Jaccard, Paul},
  title   = {The Distribution of the Flora in the Alpine Zone},
  journal = {New Phytologist},
  volume  = {11},
  number  = {2},
  pages   = {37--50},
  year    = {1912},
  doi     = {10.1111/j.1469-8137.1912.tb05611.x}
}
```

Notas de consistência para 2.5 (E61):

- O texto atual cita 273 séries de origem no agente; a configuração final tem 47. Substituir todas as menções (inclusive no parágrafo comentado do estudo de caso e no fecho do capítulo).
- As subseções "Confiança estatística da escolha" e "Redutibilidade" estão comentadas; se forem reativadas, os números novos estão em `outputs/resultados/item_12/` e `outputs/resultados/item_13/`.

## 2.6. Ablação e considerações finais

NOTA (16/10): esta seção foi detalhada e substituída pelo guia dedicado `melhorias/GUIA_ABLACAO.md`, com numeração própria AB01 a AB07 e todos os blocos prontos (tabela, parágrafos, subseções novas). Os itens E62 a E65 abaixo ficam aposentados; use o guia de ablação. A atualização das Considerações finais (E65) continua valendo para os números de rank e custo, e o fecho do capítulo está no novo guia (AB07).

(E62) Tabela da ablação (CREST contra sem agente): substituir pelos valores finais por
conjunto de dados (SMAPE, pontos percentuais):

ANP 21,97 contra 21,97; NN5 11,57 contra 11,57; M4 9,15 contra 9,14;
ETTh1 22,72 contra 22,62; ETTh2 17,99 contra 17,99; ETTm1 45,98 contra 46,04;
ETTm2 20,30 contra 20,30.

(E63) No texto agregado: CREST contra sem agente, SMAPE $p = 0{,}3080$ e POCID $p = 0{,}0738$ (matriz regenerada em `outputs/resultados/ablacao/`); a diferença entre os dois é indistinguível, reforçando que a contribuição do agente é a exploração e a auditoria, não a acurácia.

(E64) Na subseção "Quando o agente muda a decisão", substituir os números pelo resultado final: em 633 das 680 séries (93,1\%) a estratégia final é idêntica entre as duas versões; nas 47 séries restantes (6,9\%), o resultado no teste é melhor em 20 séries e pior em 27. A maior melhoria isolada é na série 5 da ANP (2,09 pontos percentuais de SMAPE) e a maior piora é na série 137 da ANP (3,13 pontos percentuais).

(E65) Considerações finais: atualizar o rank médio para 3,00 (empate com o FFORMA), a
menção à posição em cada conjunto (quatro entre os dois primeiros, último na ANP
e quinto no ETTm2) e a frase sobre a vantagem da exploração do agente, que passa
a ser 6,9 por cento das séries. O custo citado deve ser o da configuração final
(por exemplo, entre 6 e 48 segundos por série no modelo adotado, conforme a
tabela de custo).

---

## 2.7. Figuras: o que substituir no documento — E66

Nenhuma figura precisa ser deletada: todas mantêm o mesmo nome de arquivo no `overleaf`. Substitua o conteúdo dos arquivos abaixo pelos que foram regenerados com a configuração final.

| Figura no texto | Arquivo no overleaf | Novo arquivo gerado |
|---|---|---|
| `fig:mcm-anp` | `images/resultados_final/mcm_ANP_MONTHLY.png` | `outputs/resultados/item_05/mcm_ANP_MONTHLY.png` |
| `fig:mcm-nn5` | `images/resultados_final/mcm_NN5_WEEKLY_DATASET.png` | `outputs/resultados/item_05/mcm_NN5_WEEKLY_DATASET.png` |
| `fig:mcm-m4` | `images/resultados_final/mcm_M4_WEEKLY_DATASET.png` | `outputs/resultados/item_05/mcm_M4_WEEKLY_DATASET.png` |
| `fig:mcm-etth1` | `images/resultados_final/mcm_ETTH1.png` | `outputs/resultados/item_05/mcm_ETTH1.png` |
| `fig:mcm-etth2` | `images/resultados_final/mcm_ETTH2.png` | `outputs/resultados/item_05/mcm_ETTH2.png` |
| `fig:mcm-ettm1` | `images/resultados_final/mcm_ETTM1.png` | `outputs/resultados/item_05/mcm_ETTM1.png` |
| `fig:mcm-ettm2` | `images/resultados_final/mcm_ETTM2.png` | `outputs/resultados/item_05/mcm_ETTM2.png` |
| `fig:mcm-agregado-smape` | `images/resultados_final/mcm_agregado_smape.png` | `outputs/resultados/item_07/mcm_agregado_smape.png` |
| `fig:mcm-agregado-pocid` | `images/resultados_final/mcm_agregado_pocid.png` | `outputs/resultados/item_07/mcm_agregado_pocid.png` |
| `fig:mcm-ablacao-smape` | `images/resultados_final/mcm_agente_vs_sem_agente_smape.png` | `outputs/resultados/ablacao/mcm_agente_vs_sem_agente_smape.png` (gerado e já copiado) |
| `fig:mcm-ablacao-pocid` | `images/resultados_final/mcm_agente_vs_sem_agente_pocid.png` | `outputs/resultados/ablacao/mcm_agente_vs_sem_agente_pocid.png` (gerado e já copiado) |
| `fig:erro-horizonte-smape` | `images/resultados_final/erro_horizonte_smape.png` | `outputs/resultados/item_09/figuras/erro_horizonte_smape.png` (gerado e já copiado) |
| `fig:erro-horizonte-pocid` | `images/resultados_final/erro_horizonte_pocid.png` | `outputs/resultados/item_09/figuras/erro_horizonte_pocid.png` (gerado e já copiado) |
| `fig:caso-sucesso` | `images/resultados_final/caso_sucesso.png` | `outputs/resultados/interpretabilidade/figuras/caso_sucesso.png` (gerado e já copiado) |
| `fig:caso-fracasso` | `images/resultados_final/caso_fracasso.png` | `outputs/resultados/interpretabilidade/figuras/caso_fracasso.png` (gerado e já copiado) |

As figuras de horizonte são geradas pelo script `outputs/resultados/scripts/figuras_horizonte.py`, que lê os CSVs de `outputs/resultados/item_09/` e grava em `outputs/resultados/item_09/figuras/` (os quatro arquivos de horizonte e um de contagem de vitórias, descrito abaixo). Os arquivos atualizados já foram copiados para `overleaf/resultados_final/`. As figuras de linha (casos, horizonte e vitórias) foram regeradas em 16/10 com fontes maiores e legendas fora da área de plotagem, para ficarem legíveis quando reduzidas no documento; as legendas das figuras de horizonte agora ficam abaixo do gráfico. As figuras de MCM permanecem como estavam: acima da fonte 18 o texto das células estoura o layout da matriz, então foram mantidas na configuração atual.

Figura opcional para a subseção de contagem de vitórias (o texto atual usa apenas a tabela): `outputs/resultados/item_09/figuras/vitorias_por_metodo.png`, gerada a partir de `item_08/contagem_vitorias.csv`. Se quiser incluí-la, o bloco é:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{images/resultados_final/vitorias_por_metodo.png}
    \caption{Contagem de vitórias por método, entre as 680 séries dos sete conjuntos de dados.}
    \label{fig:sintese-vitorias}
\end{figure}
```

Verificação feita: a subseção "Recorrência das transformações entre conjuntos de dados" (tabela `tab:sintese-transformacoes`) não depende da execução do agente, apenas dos resultados dos modelos individuais. Os valores foram reconferidos com os CSVs e batem com o texto atual (8/10, 9/10, 9/10, 10/10, 9/10, 8/10, 9/10 e 62/70, 88,6\%). Nada a alterar ali.


# Parte 3. background.tex e revisao.tex

- Conferir se a revisão cobre Model Confidence Set, winner's curse e seleção com
  validação curta, usados nas discussões.
- Conferir a lista de modelos de linguagem citados e alinhar com os três modelos
  efetivamente comparados no capítulo de resultados.
- A descrição do protocolo de janelas e das métricas continua válida.

---

# Parte 4. Pendências executáveis antes de fechar a escrita

1. Gemma4: rodar com a configuração final (mesma do gpt-oss e do Qwen3.8) e preencher a coluna nas duas tabelas da Subseção 2.1. Atenção: o Gemma4 não aceita níveis de raciocínio, apenas on/off, então o comando usa --reasoning on ou omite o parâmetro; registre essa diferença no texto. Os casos e as figuras de sucesso e fracasso (Subseção 2.5.4) foram atualizados para as séries M4 206 e ANP 13.
2. Figuras do erro por passo do horizonte: geradas pelo novo script `outputs/resultados/scripts/figuras_horizonte.py` e já copiadas para `overleaf/resultados_final/` (inclui as duas de dois painéis, a de média com 8 passos e a de horizonte completo).
3. Figuras de MCM: as por conjunto e as agregadas já foram regeneradas com a configuração final (`outputs/resultados/item_05/` e `outputs/resultados/item_07/`); basta substituir os arquivos no overleaf conforme a Seção 2.7.
4. Busca sem LLM: o espaço de busca atual é um superconjunto do catálogo final;
   ou o texto descreve o orçamento igual e o espaço comparável, ou a busca é
   restrita ao catálogo final e reexecutada (determinística, barata).
5. Replay vencedor contra melhor semente: reexecutar sobre a configuração final
   para que os números citados sejam os do método descrito.
6. O conteúdo de ablação do `aprovacao_solida/DOCUMENTO_ADICOES.md` foi absorvido
   pelo novo `melhorias/GUIA_ABLACAO.md`; o restante daquele documento (proposta e
   resumo) segue pendente de consolidação.
