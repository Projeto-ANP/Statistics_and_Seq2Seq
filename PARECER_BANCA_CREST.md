# Parecer de banca (simulado) — CREST

**Objeto**: método proposto no capítulo "Método Proposto" (`proposta.tex`) e
implementado em `orchestrator_react/`, avaliado a partir do código e dos
resultados em `outputs/resultados/` e `timeseries/mestrado/resultados/`.

**Veredito curto**: trabalho de engenharia sólido e com honestidade científica
rara, mas a hipótese central — *"o agente LLM melhora a combinação"* — não é
sustentada pelos dados. O ganho observável vem do componente determinístico
(semeadura + meta-modelo entre séries). É defensável com reenquadramento do
claim e dois ou três experimentos adicionais; com o texto atual, **não passa sem
ressalvas**, porque o capítulo promete ("superar") o que os experimentos do
próprio autor negam.

---

## 1. O que o método faz (síntese para contexto)

Duas macro-fases.

1. **Preparação por conjunto de dados (sem LLM)**: 26 características por série
   (4 interpretáveis + 22 catch22); um meta-modelo único ajustado
   *leave-one-series-out* (saída softmax, treinado no erro da combinação,
   à la FFORMA); e um *prior* sobre 10 estratégias fixas (média/mediana/DBA +
   média/média aparada sobre os κ ∈ {5,7,9} preditores mais estáveis +
   ponderação do meta-modelo), com prior_i(s) = média de score_i'(s) nas demais
   séries.
2. **Decisão por série**: perfil, avaliação do pool (RMSE, Kendall τ,
   correlação de erros) e diagnóstico interpretativo; semeadura do histórico
   com as 10 estratégias fixas; ciclo ReAct de um agente LLM sobre um catálogo
   fechado de 24 ações determinísticas + o aceite, com pontuação anti-vazamento
   (janela expansiva p/ pesos, *leave-one-out* p/ seleção de pool) e parada por
   aceite, estagnação ou orçamento. A estratégia aplicada é sempre a de melhor
   pontuação de todo o histórico — inclusive sobrepondo um aceite pior.

---

## 2. Pontos fortes

- **Protocolo anti-vazamento bem pensado**: janela expansiva para pesos,
  *leave-one-out* entre janelas para seleção de pool e seleção aninhada por
  fold (*nested_selection*). Poucos trabalhos de combinação explicitam isso.
- **Reprodutibilidade e auditoria**: seeds fixas, artefatos por série,
  verificação de procedência (`provenance_ok`), resolução de nomes alucinados,
  *clamps* de parâmetros, fingerprints de configuração por linha do CSV.
- **Desenho limpo da ablação**: o braço sem agente usa exatamente o mesmo pool,
  janelas, protocolo e conjunto de sementes. A comparação é justa.
- **Resultado determinístico com mérito próprio**: o piso (sementes de
  estabilidade + meta-modelo pooled) empata com o FFORMA e supera o ADE de
  forma significativa no agregado (MCM: ΔSMAPE −0,0093 vs ADE, p=0,0017;
  empate com FFORMA p=0,96 e com a mediana p=0,885).
- **Instrumento metodológico**: o teste de "indistinguibilidade" da seleção
  (bootstrap + Diebold-Mariano, com correção HLN) mostra que a escolha do
  agente quase sempre está dentro do ruído — algo que a literatura raramente
  reporta.
- **Reporte do bug do DBA** (o v5 calculava média onde dizia DBA) e das
  limitações dos baselines no M4/ETT é prática científica exemplar.

---

## 3. Fragilidades e perguntas de arguição

1. **A contribuição central não se sustenta.** A ablação mostra empate
   estatístico e média levemente pior com agente (ver §4). *Pergunta*: se o
   agente não melhora, qual é a hipótese falsificável da tese e qual é a
   contribuição do LLM? O texto precisa parar de prometer e passar a medir.

2. **Falta o experimento decisivo: busca não-LLM sobre o mesmo catálogo, com o
   mesmo orçamento.** A ablação compara agente vs *nenhuma busca* (melhor
   semente). Isso não isola o efeito do LLM: pode ser que busca alguma ajude e o
   LLM seja apenas um *search* ruim. Enquanto não houver braços de busca gulosa,
   aleatória, hill-climbing ou bandit nas mesmas 12 iterações e 24 ações, a
   conclusão "LLM não agrega" não está separada de "busca não agrega". **É o
   furo metodológico mais grave.**

3. **Validade estatística.** 680 séries de 7 datasets com dinâmicas distintas
   são agregadas num único Wilcoxon; efeitos são minúsculos (ΔSMAPE 0,0007
   entre agente e sem agente; 0,0038 vs FFORMA); dezenas de comparações sem
   correção de multiplicidade (Holm/BH); e o vencedor é o **mínimo de ~20
   tentativas sobre 3 janelas** — *winner's curse* clássico. Exigir: testes por
   dataset, intervalos de confiança e correção de multiplicidade.

4. **Vazamento entre séries duplicadas.** O NN5 tem séries idênticas duas a duas
   (T1≡T47, T11≡T50, T79≡T111). O meta-modelo e o prior são LOSO: treinar em
   T47 para prever T1 é vazamento direto. Deduplicar (ou justificar) e re-rodar
   o meta-modelo — hoje o número pooled em NN5 tende a ser otimista.

5. **Métricas inadequadas em séries que cruzam zero (ETT).** O preset default
   inclui MAPE/SMAPE e a âncora (média) explode nesses datasets (ETTH1:
   MAPE ≈ 2,7×10¹¹, SMAPE ≈ 2,0). A pontuação normalizada por razão fica
   distorcida exatamente onde as séries são problemáticas. Usar RMSSE/MASE ou
   preset *scale-free* e/ou tratar ETT como anedota explícita.

6. **Comparações não são todas apples-to-apples.** Em ETTM1/2 o CREST descarta
   5 modelos desalinhados enquanto FFORMA/ADE/mean combinam janelas
   incompatíveis; no M4 os campos RMSE/MAPE do FFORMA/ADE estão corrompidos
   (~10¹³¹). E o "melhor individual" à frente do CREST no ranking do item_06 é
   escolhido com a janela de teste — é um **oráculo** e precisa ser rotulado
   como tal, senão a leitura fica injusta contra o CREST.

7. **Custo-benefício.** Se o agente custa minutos de GPU por série e não agrega
   acurácia, o trade-off precisa estar na mesa (item_17 já instrumenta o custo;
   trazer para o capítulo e discutir quando, se algum dia, o LLM se paga).

8. **Novidade e posicionamento.** Agentes LLM para seleção de modelos/hiperparâmetros
   já existem em AutoML (2023–2025). O capítulo precisa de *related work* recente
   e de explicitar o que é novo: o framework anti-vazamento com catálogo fechado
   + validação por backtest, e/ou o resultado negativo rigoroso.

9. **Generalização.** O componente central (meta-modelo/prior entre séries) só
   existe com ≥20 séries; nos ETT (7 séries) é omitido — ou seja, o "Cross-series"
   não é testado onde o LLM teria mais espaço (poucos dados por série). Nada
   conclusivo pode sair dos ETT.

10. **Detalhes menores**: pesos da pontuação (0,3/0,3/0,2/0,2) fixos sem
    análise de sensibilidade; uma única execução do LLM por série (seed=7) sem
    variância entre seeds; versão do modelo Ollama sem hash fixado; POCID
    precisa de definição inequívoca; 62% das estratégias ponderadas vencedoras
    são numericamente a média do próprio pool (o método deveria dizer isso em
    voz alta nos resultados).

---

## 4. Números-chave (com fontes)

**Agente vs. sem agente (680 séries, `outputs/resultados/ablacao/`)**

| Métrica | CREST (agente) | CREST sem agente | Wilcoxon p |
|---|---|---|---|
| SMAPE médio | 0,1376 | **0,1369** | 0,1403 |
| POCID médio | 57,86 | **58,43** | 0,0785 |

- Estratégia final **idêntica em 54,9%** (373/680); quando muda: agente melhor
  em 130, pior em 158 (ΔSMAPE médio +0,0016; excluindo as 66 séries em que o
  baseline vence com DBA real, +0,0017).
- No v5 o `combine_dba` tinha bug e devolvia a média: a semente `a3` do v5 é
  idêntica à média em 666/680 séries; no baseline (corrigido) o DBA real venceu
  em **66 séries** — vantagem estrutural do braço sem agente que precisa ser
  explicitada.

**CREST vs. métodos (MCM agregado, `outputs/resultados/item_07/`)**

| Comparação | ΔSMAPE | p (Wilcoxon) | Leitura |
|---|---|---|---|
| CREST vs ADE | −0,0093 | 0,0017 | CREST melhor |
| CREST vs FFORMA | −0,0038 | 0,9639 | empate |
| CREST vs Mediana | −0,0002 | 0,8850 | empate |
| CREST vs RF(DWT) | −0,0051 | 0,0174 | CREST melhor |

POCID: CREST vs ADE +1,49 pontos (p=0,0186); vs FFORMA +0,36 (p=0,40);
vs Mediana +0,65 (p=0,33).

**SMAPE por dataset (menor é melhor)**

| Dataset | CREST | sem agente | FFORMA | ADE | mediana |
|---|---|---|---|---|---|
| ANP (182) | 0,2209 | 0,2197 | **0,2166** | 0,2177 | 0,2194 |
| NN5 (111) | 0,1177 | **0,1157** | 0,1197 | 0,1178 | 0,1201 |
| M4 (359) | 0,0923 | 0,0914 | 0,0996 | 0,1069 | **0,0897** |
| ETTM2 (7) | **0,1610** | 0,2030 | 0,1654 | 0,1872 | 0,1997 |

Outros: ETTH1 CREST 0,2276 vs FFORMA 0,3001; ETTM1 FFORMA 0,4396 vs CREST
0,4594. ETT tem 7 séries cada — ruído alto.

**Vitórias por série (`item_08`)**: melhor_individual 201 (29,6%, **oráculo**),
CREST 152 (22,4%), ADE 130, FFORMA 103, melhor estática 94.

**Outras evidências**

- Seleção quase sempre dentro do ruído: "98 of 111" séries NN5 classificadas
  como `indistinguishable` (`orchestrator_react/config.py`).
- 62% das estratégias ponderadas vencedoras são numericamente a média do próprio
  pool (`orchestrator_react/state.py`).
- `pooled_meta_model` é retirado quando há <20 séries
  (`orchestrator_react/config.py`).
- M4: RMSE/MAPE do FFORMA/ADE corrompidos (~10¹³¹) em `item_03`.
- MAPE da âncora no ETTH1 ≈ 2,7×10¹¹ em `item_02`.

---

## 5. Experimentos exigidos antes do aceite

1. **Braço de busca não-LLM** (gulosa + aleatória + bandit/hill-climbing) sobre
   o mesmo catálogo, com o mesmo orçamento. Sem isso a tese não responde à
   própria pergunta.
2. **Deduplicação do NN5** (T1≡T47 etc.) e re-treino do meta-modelo/prior,
   verificando o impacto no pooled.
3. **Estatística por dataset**, com correção de multiplicidade (Holm/BH) e
   intervalos — e o efeito do agente reportado como nulo, não omitido.
4. **Ressalvas explícitas** para ETT/ETTM (métricas e alinhamento) e rótulo de
   **oráculo** nos baselines "melhor individual/estática".
5. **Custo-benefício do LLM** (tempo/energia por série vs ganho nulo),
   incorporado à discussão.

---

## 6. Reenquadramento sugerido do claim

Em vez de "método que supera baselines via agente LLM":

> Propomos um framework de combinação por série com um agente LLM embutido,
> protocolo anti-vazamento explícito e catálogo fechado de ações verificáveis.
> A ablação mostra que o ganho observável vem da semeadura determinística com
> pools estáveis e do meta-learning entre séries — que empatam com o FFORMA e
> superam o ADE — e que o agente **não** adiciona acurácia mensurável, embora
> agregue rastreabilidade, justificativa textual e um mecanismo de seleção
> auditável.

Resultado negativo bem instrumentado é ciência; promessa não cumprida é
problema de defesa.

---

## 7. Veredito final

- **Com o texto atual**: não aprovado sem ressalvas — o capítulo vende uma
  contribuição que o próprio experimento refuta.
- **Com o reframing (§6) e os experimentos 1–3 (§5)**: aprovado com ressalvas.
  A engenharia, a reprodutibilidade, a caracterização honesta da
  indistinguibilidade e o reporte do bug do DBA são mérito suficiente para um
  mestrado que forma pesquisador, e o achado negativo tem valor para a
  comunidade.

---

## 8. Fontes consultadas

**Resultados**

- `outputs/resultados/ablacao/RESUMO_contribuicao_agente.md`
- `outputs/resultados/ablacao/desempenho_sem_agente_por_dataset.csv`
- `outputs/resultados/ablacao/mcm_agente_vs_sem_agente_{smape,pocid}.csv`
- `outputs/resultados/ablacao/pareado_por_serie.csv`
- `outputs/resultados/item_02/item_02_estaticas_por_dataset.csv`
- `outputs/resultados/item_03/item_03_fforma_ade_por_dataset.csv`
- `outputs/resultados/item_04/item_04_crest_por_dataset.csv`
- `outputs/resultados/item_06/item_06_ranking_consolidado.csv`
- `outputs/resultados/item_07/mcm_agregado_{smape,pocid}.csv`
- `outputs/resultados/item_08/item_08_contagem_vitorias.csv`
- `outputs/resultados/item_18/`
- `outputs/resultados/scripts/common.py` (definição de `melhor_individual` e
  `melhor_estatica`: seleção pelo SMAPE da janela de teste — oráculo)

**Código**

- `orchestrator_react/pool.py` (sementes e piso determinístico)
- `orchestrator_react/state.py` (score, seleção, confiança, gêmeos numéricos)
- `orchestrator_react/combiners.py` (bug do DBA; duplicatas do NN5)
- `orchestrator_react/config.py` (presets, `pooled_meta_model_min_series=20`,
  "98 of 111")
- `orchestrator_react/registry.py` (24 ações + aceite)
- `run_tsf_orchestrator.py` / `run_tsf_baselines.py` (braços com e sem agente)

**Texto**

- `proposta.tex` (caps. de formulação, preparação, agente, aplicação)
