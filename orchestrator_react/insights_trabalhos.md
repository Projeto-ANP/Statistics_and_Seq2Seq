# Notas de Design: Insights de Trabalhos sobre Combinação via LLM/Agentes

**Propósito deste documento:** reunir, de forma organizada, o que cada trabalho pesquisado até agora relata como o que efetivamente levou a melhores resultados (não apenas "o que o trabalho faz", mas "por que funcionou melhor"), junto de ideias concretas que podem inspirar decisões de arquitetura do seu agente combinador mais adiante. Isso é uma nota de trabalho, não texto para o TCC — serve como memória de pesquisa.

---

## Síntese: os pontos mais importantes a considerar, juntando todos os trabalhos

Antes de ir para as entradas individuais, aqui está o que, olhando tudo junto, parece mais decisivo para você conseguir resultados satisfatórios. Organizado por tema, não por trabalho.

**1. Composição do pool de modelos (o ponto com evidência mais forte e mais contraditória ao mesmo tempo)**
- O Self-MoA (item 10) mostra que misturar modelos de qualidades muito diferentes pode *piorar* o resultado frente a concentrar-se nos melhores. O MoA original (item 1), por outro lado, mostra que até respostas de qualidade inferior parecem ajudar o raciocínio final. Essas duas conclusões parecem conflitantes, e é exatamente por isso que **testar empiricamente as duas hipóteses no seu próprio pool de modelos (ETS, ARIMA, CatBoost, SVR, RF, etc.) é provavelmente o experimento de maior valor que você pode rodar antes de fechar a arquitetura**, em vez de assumir uma resposta por analogia.

**2. Estruturar o raciocínio do agente em comparações, não em uma decisão única e ampla**
- O LLM-Blender (item 7) mostra que comparar candidatos aos pares é mais confiável do que pontuar cada um isoladamente. O TSOrchestr (item 2) estrutura sua decisão como um ciclo iterativo de hipóteses testadas uma a uma, não uma decisão única. Isso sugere que pedir ao agente para "atribua um peso para cada um dos N modelos de uma vez" é provavelmente pior do que estruturar o raciocínio em comparações sucessivas ou rodadas iterativas.

**3. Forçar o agente a justificar a decisão em termos causais/observáveis, não apenas emitir o número**
- Tanto o TSOrchestr (pontuação de fidelidade SHAP) quanto o Nexus (explicações alinhadas a tendência/sazonalidade) melhoram quando o agente é forçado a expressar sua decisão em termos de características observáveis da série (tendência, sazonalidade, erro recente), não apenas como um peso numérico solto. Isso é replicável no seu \textit{prompt} do ReAct sem nenhum ajuste fino, só pela forma como a instrução é escrita.

**4. Separar diagnóstico de decisão**
- TimeSeriesScientist (item 4) e Nexus (item 5) mostram, de formas diferentes, que separar a etapa de "entender a série" (diagnóstico, características, força de sazonalidade) da etapa de "decidir a combinação" traz ganho real e mensurável (a divisão de papéis do TimeSeriesScientist reduziu erro em até 38,2% frente a um LLM sem essa divisão). Mesmo num agente único, valeria a pena estruturar isso como dois passos sequenciais de raciocínio/ferramentas.

**5. Não subestimar o custo de ajuste fino, nem a engenharia de prompt**
- O ganho do TSOrchestr depende de ajuste fino caro (SHAP + RL). O ganho do Ren \& Wang depende quase inteiramente de engenharia de prompt bem feita, sem ajuste fino algum. Como sua proposta é *zero-shot*/\textit{few-shot} local, o caminho mais realista de ganho de qualidade para vocês é investir pesado na engenharia do prompt e nos exemplos de demonstração (o que o próprio Meta-Tool também confirma), não assumir que vai faltar o "empurrão extra" que só o ajuste fino traria.

**6. Cuidado com a suposição de que ajuste fino (ou mais complexidade) sempre ajuda**
- O Krause et al. (item 11, no seu próprio domínio de aplicação) mostra que ajuste fino pode **piorar drasticamente** o desempenho de um modelo (o TimesFM colapsou após ajuste fino individual). Isso é um lembrete direto de testar sempre a alternativa mais simples como baseline antes de assumir que mais sofisticação (mais ajuste fino, mais camadas, mais agentes) vai ajudar.

**7. Usar mais de uma métrica de avaliação, elas capturam coisas diferentes**
- O Krause et al. usa RRMSE (magnitude do erro) e POCID (acerto de direção) lado a lado, mostrando que um modelo pode ganhar em uma métrica e perder em outra. Vale considerar reportar mais de uma métrica de natureza diferente para o seu agente combinador, não só uma métrica de erro agregado.

---

## 1. Mixture-of-Agents (Wang et al., 2024/2025)

**Mecanismo:** camadas de LLMs "Propositores" geram respostas; "Agregadores" recebem as respostas da camada anterior como contexto adicional e sintetizam uma nova resposta, via *prompting* puro, sem nenhum treinamento adicional.

**O que levou a melhores resultados:**
- O desempenho do MoA se correlaciona tanto com a **qualidade quanto com a diversidade** dos propositores, mas análises posteriores ao artigo original (citadas em trabalhos que o revisam) mostram que a **qualidade tem correlação mais forte que a diversidade**. Ou seja: colocar modelos ruins no pool não ajuda tanto quanto se pensava; qualidade do "pool" ainda importa mais.
- Mais camadas de agregação melhoram o resultado, mas com retorno decrescente (testaram 2 vs. 3 camadas no MATH, e 3 já trouxe ganho sobre 2).
- Nem todo modelo forte como "propositor" é bom como "agregador". Os papéis pedem habilidades diferentes: gerar uma resposta boa isoladamente é diferente de sintetizar bem várias respostas de outros.
- O efeito de "colaboratividade" (melhora ao ver respostas de outros modelos) se mantém mesmo quando essas respostas auxiliares são de qualidade **inferior** à que o próprio modelo geraria sozinho — sugere que exposição a alternativas, mesmo piores, ainda ajuda o raciocínio final.

**Ideias para a sua arquitetura:**
- Ao decidir quais modelos (ETS, ARIMA, CatBoost, SVR, RF, etc.) entram no "pool" de entrada do agente combinador, talvez valha testar explicitamente se dar acesso a modelos historicamente mais fracos ainda ajuda o agente a calibrar melhor sua decisão (paralelo ao achado de colaboratividade), em vez de assumir que só os melhores modelos devem ser mostrados a ele.
- Se a arquitetura final tiver mais de um agente (ex: planner + combinator), vale testar explicitamente se o mesmo LLM funciona bem nos dois papéis, ou se um LLM diferente por papel (ex: um mais barato como planner, outro como combinator) dá resultado melhor — o MoA sugere que isso não é garantido.

**Limitação relatada pelos autores:** maior latência (*Time to First Token*) por causa da agregação iterativa em múltiplas camadas — relevante se o custo de inferência for uma preocupação da sua avaliação experimental.

---

## 2. TSOrchestr / Conversational Time Series Foundation Models (Cao et al., 2025)

**Mecanismo:** LLM reposicionado como "juiz" que pondera um conjunto de quatro TSFMs (Toto, Sundial, Moirai-2, TabPFN-TS), combinando otimização numérica (SLSQP) com raciocínio do agente. O agente passa por um ajuste fino adicional estilo R1 (DeepSeek-R1/GRPO), guiado por uma pontuação de fidelidade baseada em SHAP, e depois opera por meio de um ciclo de coordenação em duas etapas.

**O que levou a melhores resultados, detalhado:**
- **Fundamentação teórica prévia**: os autores não apenas constatam empiricamente que nenhum modelo domina, eles formalizam isso com um "Índice de Incompatibilidade Temporal" e um teorema ("Superioridade do Ensemble"), o que dá uma justificativa matemática, não só empírica, para investir em combinação em vez de buscar um único modelo perfeito.
- **SHAP como sinal de treinamento, não só de explicação**: o ponto central não é usar SHAP para *explicar depois*, é usar a pontuação de fidelidade SHAP como **recompensa durante o treinamento por reforço (GRPO)**, ensinando o modelo a tratar os pesos como afirmações causais verificáveis, e não como números soltos.
- **O ciclo de coordenação tem duas etapas bem definidas**: (1) uma "avaliação prospectiva" que decide *se* vale a pena recalibrar os pesos (detectando mudança de regime), evitando recalcular tudo a cada previsão; (2) só quando a recalibração é acionada, um ciclo iterativo onde o agente escolhe uma métrica de erro para testar (MAE, SMAPE, MSE ou CRPS), avalia os resultados de validação cruzada, e decide entre continuar testando outra métrica ou aceitar a configuração atual. Ou seja, o agente não otimiza "o erro" de forma genérica, ele testa hipóteses específicas, uma de cada vez.
- **Calibração amortizada como resposta ao problema de latência**: os autores reconhecem o mesmo problema de latência do MoA (raciocínio em múltiplos turnos é lento), mas argumentam que isso não importa tanto em previsão de séries temporais porque o processo caro só roda na fase de calibração (quando uma mudança de regime é detectada), não a cada nova previsão gerada.
- **Modelo de raciocínio usado é pequeno e local**: os autores usam o Qwen-2.5-3B-Instruct como agente de raciocínio (também testam GPT-4o e Claude-3.5-Sonnet, mas o resultado com o modelo pequeno já é forte), o que é evidência direta de que um modelo pequeno rodado localmente, como vocês pretendem usar, é viável para esse tipo de tarefa.

**Ideias para a sua arquitetura:**
- Mesmo sem fazer o ajuste fino por reforço (que é caro e foge do escopo de vocês, que querem *zero-shot*/*few-shot* local), é possível emular parte desse benefício apenas **pedindo ao agente, no próprio prompt do ReAct, que justifique cada peso ou escolha de combinação em termos de características observáveis da série** (tendência, sazonalidade, erro recente de cada modelo), em vez de só emitir o número.
- A separação entre "decidir se recalibrar" e "como recalibrar" é uma ideia concreta e barata de replicar: o agente pode primeiro checar, com base na janela de validação de 3 passos, se o desempenho relativo dos modelos mudou o suficiente para justificar uma nova combinação, em vez de sempre recalcular do zero a cada série.
- A ideia de testar métricas de erro diferentes uma a uma, em vez de otimizar tudo de uma vez com um critério único, também é replicável: o agente poderia, por exemplo, checar se a combinação escolhida é boa tanto em termos de erro absoluto quanto de acerto de direção (ver também o item 11, Krause et al., sobre usar métricas complementares).

**Limitação:** depende de ajuste fino específico (RL + SHAP) para atingir os resultados completos relatados; a parte de aprendizado por reforço não é diretamente replicável em um cenário puramente *zero-shot*, o que é uma diferença importante a mencionar ao comparar com a proposta de vocês. Já a estrutura do ciclo de coordenação (avaliação prospectiva + exploração iterativa por hipóteses) não depende do ajuste fino e é inteiramente replicável via *prompting*.

---

## 3. Ren & Wang (2025) — Combinação de previsões do SPF do BCE via LLM

**Mecanismo:** LLM em modo *zero-shot* gera diretamente os pesos de combinação a partir do histórico de previsões de especialistas humanos.

**O que levou a melhores resultados:**
- Os autores relatam que a eficácia da abordagem depende fortemente da **engenharia de prompt** (como o histórico de previsões é apresentado ao modelo).
- Um teste estatístico (regressão de efeitos mistos) mostrou que a combinação via LLM é mais **resiliente a flutuações na atenção dos previsores** do que a acurácia das previsões individuais — ou seja, o LLM parece compensar bem quando a qualidade dos insumos varia ao longo do tempo.

**Ideia para a sua arquitetura:**
- Reforça que vale a pena investir tempo real testando diferentes formulações de prompt para apresentar o histórico de erro dos modelos ao agente (não é um detalhe menor), e que pode valer a pena medir explicitamente se o agente mantém boa performance mesmo quando um ou mais modelos do pool degradam temporariamente (não só o cenário "todo mundo performando bem").

---

## 4. TimeSeriesScientist (Zhao et al., 2025)

**Mecanismo:** quatro agentes especializados (Curador, Planejador, Previsor, Relator), cada um com um papel único no pipeline.

**O que levou a melhores resultados:**
- A divisão de papéis por si só parece ser uma fonte real de ganho: os autores relatam redução de erro de **10,4% frente a baselines estatísticos/DL** e de **38,2% frente a abordagens de LLM sem essa divisão de papéis**. Isso é evidência empírica forte de que dividir a tarefa em sub-agentes especializados ajuda mais do que simplesmente usar um LLM maior ou mais capaz em um único papel genérico.

**Ideia para a sua arquitetura:**
- Mesmo que a primeira versão do seu sistema seja um único agente combinador, esse resultado sugere que, se o desempenho não for satisfatório, **separar diagnóstico (ex: extração de características, força de sazonalidade) da decisão de combinação em si** (mesmo que como duas chamadas de ferramenta sequenciais dentro do mesmo agente, antes de evoluir para múltiplos agentes de fato) pode ser um caminho de melhoria concreto e testável.

---

## 5. Nexus (Das et al., 2026)

**Mecanismo:** três estágios — Contextualização (Agente de Contexto Histórico), Geração de Perspectiva em Dupla Resolução (Agente Macro + Agente Micro), Síntese e Calibração (Agente Sintetizador + Agente de Calibração que aprende com erros passados via *backtesting*).

**O que levou a melhores resultados:**
- Separar raciocínio de **longo prazo (macro)** e **curto prazo (micro)** em agentes distintos, em vez de pedir a um único agente que raciocine em todas as escalas temporais ao mesmo tempo, parece evitar sobrecarga cognitiva do modelo.
- O **Agente de Calibração**, que ajusta a estratégia continuamente a partir de erros passados via *backtesting*, é tratado como uma etapa/agente separado, não misturado com a decisão principal de combinação.

**Ideia para a sua arquitetura:**
- Isso é muito próximo da sua própria ideia original (agente lê uma janela de validação de 3 passos + série real completa para decidir a combinação). O Nexus sugere que vale a pena tratar essa etapa de "aprender com o erro recente" como um **passo/ferramenta explícito e separado** da decisão final de combinação, em vez de fundir os dois em uma única chamada do agente.

---

## 6. Buratto et al. (2026)

**Mecanismo:** agente LLM modular, 100% local, sem depender de APIs proprietárias, com sub-agente especializado em pré-processamento, detecção de anomalias e previsão.

**Relevância:** não traz um achado de "o que melhora resultado" tão específico quanto os outros, mas **valida a viabilidade prática** de um sistema totalmente local como o que vocês pretendem construir, aplicado a um domínio industrial real (previsão de geração de energia térmica).

---

## 7. LLM-Blender (Jiang et al., 2023) — não será o segundo trabalho do texto, mas o insight vale a pena guardar

**Mecanismo:** PairRanker (compara candidatos aos pares) + GenFuser (funde os $K$ melhores).

**O que levou a melhores resultados:**
- Os autores mostram que **comparar candidatos aos pares** (perguntar "A é melhor que B?") é mais confiável do que pontuar cada candidato isoladamente, porque diferenças sutis de qualidade são mais fáceis de perceber em comparação direta do que em avaliação isolada.
- A combinação **ranquear primeiro, depois fundir** (em vez de só selecionar um vencedor) supera tanto a seleção pura quanto a fusão sem ranqueamento prévio.

**Ideia para a sua arquitetura:**
- Em vez de pedir ao agente para atribuir pesos a todos os modelos do pool de uma vez só, pode valer a pena estruturar o raciocínio do agente em **comparações pareadas** entre modelos (parecido com o espírito do teste de Diebold-Mariano que já sugeri antes), antes de consolidar isso em uma combinação final. Isso pode reduzir a carga cognitiva sobre o LLM e produzir julgamentos relativos mais confiáveis, no mesmo espírito do que o LLM-Blender encontrou para texto.

---

## 8. Gorur, Rago & Toni (2025) — Combinação por argumentação (Imperial College London)

**Mecanismo (baseado apenas no abstract, ainda não li o artigo completo):** trata previsão de julgamento (probabilidade de eventos futuros) como um problema de verificação de alegações. Múltiplos agentes de LLM podem discordar sobre a veracidade de uma alegação e trazem evidências a favor e contra, representadas como estruturas de argumentação bipolares quantitativas (QBAFs). A combinação final emerge da agregação dessas estruturas argumentativas, não de uma média ponderada ou de votação.

**Por que vale registrar:** é um mecanismo de combinação genuinamente diferente dos outros sete (nem votação, nem meta-learner estatístico, nem *prompting* direto de peso). Mesmo aplicado a previsão probabilística de eventos binários, e não a regressão numérica, ele sugere uma alternativa conceitual interessante: em vez de o agente combinador emitir diretamente um peso ou uma combinação, ele poderia ser estruturado para "argumentar" a favor e contra cada modelo do pool antes de decidir, tornando o processo de decisão mais explícito e auditável. Vale explorar essa ideia como direção futura, mesmo que não seja adotada na primeira versão da arquitetura. Como não li o texto completo, não tenho detalhes de implementação para registrar ainda.

## 9. Yeh et al. (2025) — DCATS: Data-Centric Agent for Time Series (Visa Research)

**Mecanismo:** diferente de todos os outros trabalhos revisados, o DCATS não combina previsões já geradas por múltiplos modelos. Em vez disso, um agente de LLM decide quais séries temporais "vizinhas" (outras localizações, com padrões ou proximidade geográfica semelhantes) devem ser incluídas no conjunto de treinamento de um único modelo de previsão, para enriquecer os dados de treino. A combinação aqui é de **dados de entrada**, não de previsões de saída.

**O que levou a melhores resultados:** o agente opera em um ciclo iterativo bem definido: (1) gera propostas iniciais de subconjuntos de dados com uma justificativa textual para cada uma; (2) cada proposta é avaliada por um módulo de previsão separado, que treina o modelo no subconjunto proposto e reporta o erro de validação; (3) o agente recebe os resultados ranqueados do melhor para o pior, com as justificativas de cada proposta anterior, e gera uma nova rodada de propostas refinando a estratégia vencedora; (4) o processo para quando nenhuma proposta da rodada atual supera o melhor resultado obtido até então. Os autores relatam uma redução média de erro de 6% de forma consistente em quatro modelos de previsão diferentes (Linear, MLP, SparseTSF, UltraSTF), o que indica que o ganho é agnóstico ao modelo de previsão usado.

**Ideia para a sua arquitetura, e essa é bem concreta:** a estrutura de *prompt* que os autores usam é praticamente um modelo pronto para o ciclo ReAct que vocês pretendem implementar. O *prompt* de proposta inicial tem cinco seções (contexto/background, tarefa, diretrizes, conjunto de opções disponíveis, formato de saída esperado), e o *prompt* de refinamento tem seis seções (objetivo, contexto, resultados de experimentos anteriores ranqueados do melhor para o pior com explicação de cada um, tarefa, considerações adicionais, formato de saída). Essa estrutura de "mostrar ao agente os resultados anteriores ranqueados, com a justificativa de cada tentativa, antes de pedir a próxima decisão" é diretamente adaptável para o seu agente decidir qual técnica de combinação usar a partir do erro de cada modelo na janela de validação, e vale a pena usar esse formato de *prompt* como ponto de partida concreto em vez de construir um do zero.

## 10. Li, Lin, Xia & Jin (2025) — Self-MoA, o contraponto crítico ao MoA

**Mecanismo:** em vez de agregar respostas de múltiplos LLMs diferentes (como no MoA original), o Self-MoA agrega múltiplas respostas amostradas repetidamente de um único modelo, o de melhor desempenho individual disponível no conjunto.

**O que levou a melhores resultados:** os autores mostram que o Self-MoA supera o MoA convencional em uma ampla gama de cenários (+6,6 pontos no AlpacaEval 2.0, +3,8% em média no MMLU/CRUX/MATH), evidenciando um compromisso entre qualidade e diversidade: misturar modelos de qualidades distintas pode degradar a síntese final, porque o agregador acaba incorporando também o conteúdo dos modelos mais fracos do conjunto. A diversidade obtida apenas pela variação natural entre amostras de um único modelo já forte parece ser suficiente, sem o custo de qualidade de incluir modelos fracos.

**Ideia/cautela para a sua arquitetura, é a mais importante desta entrada:** isso é uma contraindicação direta à suposição de que "quanto mais modelos no pool, melhor". Vale testar explicitamente, nos seus experimentos, se um subconjunto menor e mais criterioso dos modelos historicamente mais precisos (ex: só ETS, Theta e o melhor CatBoost) produz uma combinação tão boa ou melhor do que o pool completo incluindo os modelos historicamente mais fracos (Naive, Random Forest sem tuning, etc.), em vez de assumir por analogia com a literatura clássica de combinação que "mais diversidade sempre ajuda". Essa é uma hipótese barata de testar e que pode virar um resultado interessante por si só na sua dissertação.

## 11. Krause et al. (2026) — Modelos de Fundação para Previsão, Evidências do Setor de Combustíveis (PUCPR)

**Mecanismo:** não é um trabalho de agentes/LLM, é um *benchmark* comparando seis modelos de fundação para séries temporais (Chronos, LagLlama, Moirai-MoE, Time-MoE, TimeGPT, TimesFM) contra dez métodos tradicionais (estatísticos, ML, DL), em modo *zero-shot* e com três estratégias de ajuste fino, usando 34 anos de dados de demanda de combustíveis no Brasil.

**O que é útil considerar, mesmo não sendo um trabalho de agentes:**
- **Nenhuma família de modelo venceu em todos os cenários**: em 6 dos 7 tipos de combustível um modelo de fundação venceu, mas na gasolina o ETS (estatístico, simples) superou todos os FMs testados. Isso é evidência direta, no domínio de aplicação mais próximo do seu, de que não existe "o melhor modelo" fixo, reforçando por que uma combinação dinâmica (a proposta de vocês) faz sentido.
- **Ajuste fino pode piorar drasticamente o resultado, não só deixar de ajudar**: o TimesFM teve RRMSE quase dobrado e POCID caindo de 64,31 para 45,70 depois de ajuste fino individual, o que os autores atribuem a uma possível ruptura das representações pré-treinadas do modelo. Isso é uma cautela importante: nunca assumir que "mais adaptação/ajuste sempre ajuda" sem testar.
- **Métricas diferentes capturam coisas diferentes**: os autores usam RRMSE (erro de magnitude) e POCID (acerto da direção da mudança) lado a lado deliberadamente, porque um modelo pode ganhar em uma e perder na outra, e cada uma importa para decisões operacionais diferentes.
- **Chamado explícito por trabalho futuro**: a conclusão do artigo pede explicitamente por "\textit{hybrid or adaptive pipelines that combine the predictive capacity of FMs with the inductive biases of statistical models}" e por "\textit{dynamic ensembles or adaptive model selection}", o que é quase uma descrição direta da proposta de vocês.

**Ideia para a sua arquitetura:**
- Ao montar os experimentos, vale reportar RRMSE (ou métrica equivalente de magnitude) e POCID (ou métrica equivalente de direção) lado a lado para o seu agente combinador, em vez de uma métrica só, seguindo esse padrão. E se for considerar incluir algum TSFM (Chronos, TimesFM, etc.) no seu próprio pool de modelos, vale testar zero-shot antes de investir em ajuste fino, já que o ajuste fino não é garantia de ganho e pode, inclusive, piorar bastante o resultado.

## Como usar este documento

Isso é uma lista viva. Conforme formos aprofundando o TSOrchestr (próximo trabalho a detalhar no texto do `revisao.tex`) e, mais adiante, quando você me contar a arquitetura real do sistema (planner, combinator, etc.), posso voltar aqui e adicionar mais entradas, ou marcar quais dessas ideias vocês decidiram efetivamente testar.