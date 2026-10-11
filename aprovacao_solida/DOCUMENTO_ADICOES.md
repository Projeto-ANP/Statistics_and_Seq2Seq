# O que adicionar no documento

Guia de edição para `overleaf/proposta.tex` e `overleaf/resultados.tex`, feito a
partir dos experimentos do pacote `aprovacao_solida/`.

Como usar: cada mudança traz o rótulo SUBSTITUIR, ADICIONAR ANTES DE ou
ADICIONAR DEPOIS DE, seguido do trecho âncora do seu texto e do bloco pronto
para colar. Não usei números de linha, porque eles mudam a cada edição. Os
blocos de LaTeX já estão no estilo do documento, sem negrito na prosa e sem
travessão.

Observação sobre o escopo: os resultados novos são da ANP, em execução
determinística local, e estão em `outputs/resultados/aprovacao_solida/`. Os
outros seis conjuntos de dados ainda não foram rodados; os comandos estão na
Seção 4. Até rodá-los, mantenha as frases restritas à ANP; depois, generalize as
tabelas e o texto.

---

## 1. overleaf/resultados.tex

### Mudança 1.1. Duas subseções novas no fim do estudo de ablação

ADICIONAR ANTES DE:

```
\section{Considerações finais}
```

(a última seção do arquivo, logo depois da subseção "Quando o agente muda a
decisão")

Bloco a inserir:

```latex
\subsection{Busca sem LLM: o agente contra buscas simples}

Para isolar o efeito do modelo de linguagem, esta subseção compara o CREST a
três mecanismos de busca que não utilizam LLM, mas operam sobre o mesmo
catálogo de ações e com o mesmo orçamento de avaliações por série: uma busca
aleatória, uma busca gulosa e uma seleção por bandit (UCB). Cada braço parte
das mesmas dez sementes e aplica a mesma garantia de que a estratégia final é a
de melhor pontuação do histórico, de modo que a única diferença entre os braços
é o mecanismo que propõe novas estratégias.

Na ANP, nenhum dos três braços supera o CREST de forma estatisticamente
confiável, e a busca gulosa, com a menor média (21,89\%), empata em contagem
direta, com 66 séries em que é melhor, 74 em que é pior e 42 empates
($p = 0{,}942$, teste de Wilcoxon pareado). A leitura conjunta, com agente e
buscas simples indistinguíveis entre si, indica que, neste protocolo, o ganho
observável não decorre do mecanismo de busca, e sim da fase determinística de
preparação, isto é, das sementes e do meta-modelo entre séries, de forma
consistente com a ablação da subseção anterior.

\begin{table}[H]
\centering
\caption{SMAPE e POCID médios do CREST e de três buscas sem LLM sobre o mesmo
catálogo e orçamento, na ANP.}
\label{tab:busca-nao-llm}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Método} & \textbf{SMAPE (\%)} & \textbf{POCID (\%)} \\
\midrule
Busca gulosa      & 21,89 & 64,9 \\
Busca aleatória   & 22,26 & 64,3 \\
Busca UCB         & 22,15 & 63,9 \\
Melhor dos braços & 22,01 & 64,7 \\
Sementes (piso)   & 21,94 & 64,5 \\
CREST             & 22,09 & 64,6 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Vencedor versus melhor semente no teste}

A garantia de que a estratégia final nunca é pior que a melhor semente vale
para a validação, mas não necessariamente para o teste. Para quantificar essa
diferença, as sementes foram reconstruídas com o mesmo procedimento
determinístico de preparação e comparadas, na janela de teste, com o vencedor
efetivamente aplicado.

Na ANP, em 88 das 182 séries as duas previsões são numericamente idênticas; nas
94 restantes, a semente obtém SMAPE menor em 52, contra 42 em que o vencedor
vence. O SMAPE médio da semente (21,94\%) é menor que o do vencedor (22,09\%).
O padrão se acentua quando se observa o ganho de validação: nas séries em que a
proposta do agente melhora a validação em mais de 5\%, o resultado no teste é,
em média, 1,14 ponto percentual pior que o da semente, indício de sobreajuste
às três janelas de validação, conhecido na literatura como \textit{winner's
curse}. Regras que só trocam a semente pelo agente quando o ganho de validação
supera uma margem fixa, entre 0,5\% e 5\%, não melhoram o resultado (22,10\% a
22,12\%), e a política de manter sempre a melhor semente obtém 21,94\%, o menor
SMAPE médio entre as políticas comparadas neste conjunto de dados.
```

Parágrafo opcional, para colar no fim da subseção acima (apague se não quiser):

```latex
Um teste determinístico adicional avaliou incluir a mediana entre as
combinações sobre os $\kappa$ preditores mais estáveis, para
$\kappa \in \{5,7,9\}$. Na ANP, o piso passou de 21,94\% para 22,00\%, e nas 39
séries em que a mudança altera a decisão o piso com medianas é 0,30 ponto
percentual pior; a hipótese de que a mediana deveria integrar as sementes não
se confirma neste conjunto de dados.
```

### Mudança 1.2. Fecho do capítulo

ADICIONAR DEPOIS DO PARÁGRAFO que termina com:

```
limitações identificadas que devem orientar os próximos passos deste trabalho.
```

Bloco a inserir:

```latex
Por fim, a ablação complementada pela comparação com buscas sem LLM mostra que
a contribuição do agente não é estatisticamente distinguível da de mecanismos
de busca simples no mesmo espaço de ações. O desempenho do CREST decorre
principalmente da preparação determinística entre séries, e o agente atua como
um mecanismo de exploração e de justificativa das decisões, não como fonte de
ganho de acurácia mensurável.
```

---

## 2. overleaf/proposta.tex

### Mudança 2.1. Reenquadramento do objetivo

SUBSTITUIR o trecho (está no primeiro parágrafo das considerações iniciais):

```
com o objetivo de superar os resultados obtidos pelos modelos individuais e por outras estratégias de combinação, como as baseadas em ponderação
```

POR:

```latex
com o objetivo de investigar se a decisão de combinação por série, mediada por
um agente, supera, e em que condições supera, os modelos individuais e as
estratégias de combinação estáticas e de meta-aprendizagem
```

### Mudança 2.2. Fecho do capítulo de método

SUBSTITUIR o último parágrafo do arquivo:

```
O capítulo seguinte apresenta os resultados obtidos pela aplicação deste método aos conjuntos de dados descritos, comparando seu desempenho ao dos modelos individuais, das estratégias de combinação estática e dos métodos de combinação por meta-aprendizagem revisados no Capítulo~\ref{chp:revisao}.
```

POR:

```latex
O capítulo seguinte apresenta os resultados obtidos pela aplicação deste método
aos conjuntos de dados descritos, comparando seu desempenho ao dos modelos
individuais, das estratégias de combinação estática e dos métodos de combinação
por meta-aprendizagem revisados no Capítulo~\ref{chp:revisao}, e inclui um
estudo de ablação que confronta a atuação do agente com os componentes
determinísticos do método e com mecanismos de busca que não utilizam modelo de
linguagem.
```

### Mudança 2.3. Justificativa dos pesos da pontuação

ADICIONAR DEPOIS DO PARÁGRAFO que termina com:

```
qualquer estratégia com $\text{score}_i(s) < \text{score}_i(s_0)$ superou a média simples naquela série.
```

Bloco a inserir:

```latex
A escolha dos pesos da Equação~\eqref{eq:score} é uma decisão de projeto,
fixada a priori e mantida constante em todos os experimentos deste trabalho,
sem ajuste em função dos resultados de teste. A atribuição segue três
critérios. Primeiro, as três métricas de erro recebem, juntas, peso $0{,}8$,
e o POCID recebe $0{,}2$, o que reflete a prioridade dada à acurácia de
magnitude da previsão e mantém a acurácia de direção como critério secundário,
ainda que relevante para as aplicações visadas. Segundo, entre as métricas de
erro, RMSE e SMAPE recebem peso $0{,}3$ cada, superior ao $0{,}2$ atribuído ao
MAPE, porque o MAPE é a mais sensível a valores iguais ou próximos de zero,
condição presente em parte das séries analisadas, e não deve dominar a decisão.
Terceiro, como cada um dos três primeiros termos é uma razão contra a
estratégia de referência $s_0$, os pesos comparam grandezas adimensionais e
expressam apenas a prioridade relativa de cada métrica, não a escala em que ela
é medida. Vale destacar que a pontuação assim definida é um critério interno de
seleção de estratégias, e não a métrica de avaliação final: o desempenho do
método é reportado, no Capítulo~\ref{chp:resultados}, por MAPE, SMAPE, RMSE e
POCID calculados diretamente entre a previsão combinada e os valores reais da
janela de teste, sem normalização nem ponderação.
```

Se quiser fechar a resposta com evidência empírica, rode a análise de
sensibilidade descrita na Seção 6 deste arquivo e acrescente, ao fim do
parágrafo acima, uma frase como:

```latex
Uma análise de sensibilidade a esses pesos, apresentada na
Seção~\ref{sec:...}, mostra que a estratégia selecionada e o desempenho final
variam pouco entre presets alternativos.
```

---

## 3. Resumo e abstract

A pasta `overleaf/` não contém arquivo de resumo ou abstract. Se existir em
outro arquivo, revisar qualquer frase do tipo "supera os baselines" para algo
como "integra o grupo de melhor desempenho" ou equivalente.

---

## 4. O que ainda falta rodar

No servidor, a partir da raiz do repositório, sem arquivos `.sh`:

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
mkdir -p logs

# busca não-LLM nos outros seis datasets, em sequência
for ds in NN5_WEEKLY_DATASET M4_WEEKLY_DATASET ETTH1 ETTH2 ETTM1 ETTM2; do
    python3 aprovacao_solida/01_busca_nao_llm/run_search.py \
        --dataset "$ds" --version v1 --methods seeds,random,greedy,ucb
done
python3 aprovacao_solida/01_busca_nao_llm/analyze_search.py --version v1

# replay vencedor versus semente
nohup python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \
    --run orchestrator_react_v5 > logs/replay_gate_v5.log 2>&1 &

# mediana nas sementes
nohup python3 aprovacao_solida/05_sementes_medianas/seed_medians.py \
    > logs/seed_medians.log 2>&1 &
```

Depois de rodar, atualize as tabelas das Mudanças 1.1 com os valores dos novos
conjuntos de dados e remova a restrição "na ANP" das frases.

---

## 5. Fontes dos números citados

- `outputs/resultados/aprovacao_solida/analise_busca_anp.csv` e
  `outputs/resultados/aprovacao_solida/busca_csv/ANP_MONTHLY.csv`:
  busca sem LLM.
- `outputs/resultados/aprovacao_solida/replay_gate_v5_anp_fixed.csv` e
  `..._por_dataset.csv`: vencedor versus semente.
- `outputs/resultados/aprovacao_solida/sementes_medianas_anp.csv`:
  mediana nas sementes.
- `outputs/resultados/ablacao/RESUMO_contribuicao_agente.md`: ablação oficial
  e pareamento por série.

---

## 6. Sensibilidade dos pesos da pontuação (opcional, ainda não implementado)

Para transformar a justificativa escrita em evidência, o experimento é
recomputar, para as mesmas estratégias candidatas já avaliadas por série
(sementes, braços de busca e o vencedor do run), a pontuação sob cada preset de
`orchestrator_react/config.py` (`balanced`, `rmse_focus`, `direction_focus`,
`robust_smape`, `scale_free_safe`), tomar o argmin de cada preset e comparar
duas coisas: em quantas séries a estratégia selecionada muda, e qual é o SMAPE
e o POCID de teste da estratégia selecionada em cada preset. Como a pontuação é
calculada a partir de `Attempt.aggregate`, a análise é determinística e roda em
minutos, sem LLM.

Quando pedido, a implementação vai em
`aprovacao_solida/06_sensibilidade_score/run_score_sensitivity.py`, com o
comando:

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
python3 aprovacao_solida/06_sensibilidade_score/run_score_sensitivity.py \
    --datasets ANP_MONTHLY NN5_WEEKLY_DATASET
```

Leitura esperada: se a seleção muda em poucas séries e o SMAPE de teste varia
pouco entre presets, a frase de robustez da Mudança 2.3 se sustenta; se mudar
muito, a resposta ao orientador passa a ser que os pesos são um parâmetro de
projeto com efeito mensurável, e não uma escolha neutra.
