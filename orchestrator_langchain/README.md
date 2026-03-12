# Estrutura de Combinação e Previsão de Séries Temporais (Statistics & Seq2Seq)

Este repositório implementa uma arquitetura baseada em múltiplos agentes para a combinação inteligente de múltiplos modelos de Time Series Forecasting. O principal objetivo é superar as previsões de modelos únicos, decidindo qual o melhor método de ensembling ou selecionando modelos a cada horizonte de tempo valendo-se apenas de comportamento passado (validação cruzada expandindo/rolando por janelas).

## 🚀 Como Funciona a Pipeline?

1. **Extração e Formatação**: Os dados originam-se em previsões preexistentes (ARIMA, ETS, SVR, modelos deep learning) armazenadas nos arquivos estáticos de saída de resultados (`results.csv`).
2. **Isolamento de Teste**: Os resultados são separados nas últimas observações. A última linha é retida para teste **cego** (`final`), e as linhas imediatamente anteriores (ex: `val1, val2, val3`) tornam-se conjuntos temporais de validação.
3. **Análise de Agentes**: LLMs analisam estatísticas sobre o aprendizado de cada modelo ao longo das janelas passadas.
4. **Debate e Validação**: Diversos agentes constroem um pipeline candidato de combinações (ensembles). Há a proposição, o ceticismo contra estratégias inseguras, e métodos baseados em Regularização e estatística que filtram a melhor tática que não cometa *leakage*.
5. **Combinação/Agrupamento determinístico**: A `Orchestrator` avalia determinísticamente o sucesso histórico de toda essa lista e escolhe a combinação vencedora, processando a projeção com a amostra real retida.

## 🤖 Arquitetura Multi-Agente

A orquestração do pipeline de decisão pode ser feita usando ferramentas de Agentes LLM (`orchestrator/` via biblioteca Agno ou `orchestrator_langchain/` via módulo LangChain). O ecossistema contém diferentes *personas*:

*   **`PatternAnalyst`**: Avalia o sumário das métricas das janelas de validação das previsões históricas de cada modelo visando apontar especializações por horizonte e tendências. (Entrada: Métricas históricas de folds. Saída: Dicas e *champions* por série).
*   **`Proposer`**: Recebe o raciocínio sintético do Analista de Padrões e sugere estratégias candidatas de combinação, indicando possíveis overrides de parâmetros.
*   **`Skeptic` / `Auditor`**: Age como um gatekeeper de Segurança contra overfittings. Se o Proposer criar estratégias vazadas (*data leakage*) ou irrealistas com pouca evidência, o Skeptic impõe revisões robustas ou altera hiperparâmetros candidos.
*   **`Statistician`**: Especialista quantitativo que injeta rigidez aos ensembles — por exemplo, sugerindo restrições `top_k` robustas ou `shrinkage` para suavizar médias móveis onde a variância é muito dispersa.
*   **`Orchestrator` determinístico**: Executa todas as receitas dadas pelossistema multi-agentes de maneira segura por janela (rolling/expanding originado pela classe `RollingConfig`) e pontua um ranking para extrair a melhor de todo o debate.

## 🧮 Funções Padrões de Combinação

Estas funções estão no diretório `combinations/` e `orchestrator/strategies.py`:

*   **`mean` / `median`**: Baselines simples sem peso.
*   **`trimmed_mean`**: Tira os "piores" e "melhores" absolutos $\alpha$ do leque por horizonte e calcula a média para remover modelos extremos que alucinem demais.
*   **`best_single` / `best_per_horizon`**: Seleção pura do modelo com o menor RMSE nas subjanelas de validação acumuladas rigorosamente no passado para evitar *leakage*.
*   **`topk_mean_per_horizon`**: Computa uma média somente dos modelos "Top K" com base em ordenação de erro passado.
*   **`inverse_rmse_weights`**: Ensemble onde os modelos com erros baixos recebem linearmente (ou suavizado por shrinkage) mais pesos no somatório da probabilidade simplex.
*   **`ridge_stacking`**: Solucionador dos Mínimos Quadrados Penalizado L2, previne o uso de modelos fortemente correlacionados aprendendo em cima dos resultados passados para regularizar e escolher uma combinação unicamente testada.
*   **`exp_weighted_average`**: Distribuição Exponencial onde a importância afunda exponencialmente de acordo com a predição errática do modelo histórico nas rodadas passadas.
*   **`dba`** *(DTW-Barycenter Averaging)*: Encontra médias com alinhamento flexível da série temporal lidando bem com delays nos picos.

## 🛡️ Rigor Científico e *Data Leakage*

O processo de treinamento do pipeline é matematicamente blindado nas classes base do avaliador (_train_slice). O aprendizado possui os seguintes guard rails:
*   Os agentes **jamais** têm acesso ao vetor de resposta/real que servirá como teste final no output do sistema. O JSON global com que são alimentados limita-se ao objeto `all_validations`.
*   O uso contínuo de janelas validatórias (`rolling` ou `expanding`) restringe indexação de treino. Ao simular o processo da época $i$, o fatiamento da janela é feito na chave `0` até $i-1$.
*   Mesmo se as parametrizações dos multi-agentes enviarem por alucinação um tamanho de janela de treinamento astronômico (Ex: train_window = 100 num conjunto com apenas `3` observações), o código faz *clipping* restritivo com o max(0, ...), cravando que o ponteiro temporal jamais busque índices negativos ou futuros da previsão, não sofrendo com exceções *Out of Bounds* nem com uso de dado isolado da última janela.