# Arquitetura V4 — Agentes com Memória Continual, Roteamento Condicional e Meta-Prior Federado

> Resposta direta à derrota empírica do V3 (SMAPE 0.2069 vs FFORMA 0.1884, median 0.1918, ADE 0.1918 no NN5_WEEKLY).
> Este documento propõe uma reformulação profunda — não mais um pipeline sequencial fixo, mas um sistema agêntico
> com **memória continual cross-series**, **roteamento condicional**, **meta-prior federado** e **curadoria de pool a montante**.

---

## 1. Diagnóstico honesto do fracasso V3

A V3 perdeu para `mean`/`median`/`ADE`/`FFORMA` por motivos estruturais, não pontuais. Sintomas observados:

### 1.1 Por que V3 perdeu (cinco causas, em ordem de impacto)

**Causa A — Sem aprendizado entre séries.** Cada série recomeça do zero com apenas 3 janelas de validação. FFORMA carrega um meta-prior treinado offline no M4 (≫10⁵ séries). Em séries-curtas-com-pool-ruidoso, esse prior é o que decide. **V3 não tem prior algum.**

**Causa B — Gate DM sub-potente.** Com `3 × horizon = 24` resíduos no NN5_WEEKLY, o teste DM em `α=0.10` quase nunca rejeita H₀. Resultado: o sistema **na prática vira `pruned_equal_weights`** em ≥70% das séries (verificável em `fellback_to_pruned_mean`). A "inteligência" da escolha de regime fica adormecida.

**Causa C — Pool com redundância massiva.** O conjunto de 23 modelos inclui famílias quase colineares: `{CWT_rf, DWT_rf, FT_rf}`, `{CWT_catboost, DWT_catboost, FT_catboost, ONLY_CWT_catboost, …}`. Mesmo após poda LLM+MCS, sobram 12–15 modelos altamente correlacionados. O "combination puzzle" (Claeskens et al. 2016) bate em cheio — a média desses sobreviventes pode ser **pior** que mediana só por puxar para pontos divergentes pelos modelos similares.

**Causa D — Âncora errada.** A V3 ancora no `pruned_equal_weights`. Mas para amostras curtas, **mediana** é estatisticamente mais robusta que média (Stock & Watson 2004; Atiya 2020). Logo: a âncora deveria ser `pruned_median`, não `pruned_mean`. Hoje a V3 nem reporta esse comparativo internamente.

**Causa E — LLM otimiza decisões individuais isoladas, não a função objetivo agregada.** O ModelCritic prune por critérios locais (alto RMSE/redundância), sem saber se a poda DESSA série prejudica/beneficia a comparação agregada com FFORMA/median. Não há sinal de "lossagregada"voltando ao agente.

### 1.2 O que o V3 fez certo (e a V4 herda)

- Floors estatísticos (MCS, min_keep) → mantém downside bounded.
- Auditabilidade por série (SeriesProfile, prune_report, dm_stat).
- Features determinísticas pré-computadas (não vamos jogar fora — apenas estender).
- Tool-use estruturado com `temperature=0`.

V4 mantém esse esqueleto de garantias e ADICIONA três coisas que não existiam: **memória continual**, **meta-prior treinado nos próprios CSVs históricos** e **roteamento condicional para velocidade + precisão**.

---

## 2. Filosofia V4 — "Continual federated agent that learns from its own history"

O V3 trata cada série como um problema novo. Isso é fundamentalmente **estatística clássica disfarçada de agente**. V4 reposiciona o sistema como um **agente que aprende continuamente** — a cada série processada, deposita conhecimento em três camadas de memória que serão CONSULTADAS na série seguinte.

Inspiração direta de literatura 2024–2026:

- **Memory-augmented LLM agents** (MemGPT — Packer et al. 2023; Generative Agents — Park et al. 2023; AutoGen / MetaGPT 2024) → arquitetura de memória episódica + semântica + procedimental aplicada **fora** de chatbots, no domínio de séries temporais.
- **Continual meta-learning** (Wang & Hyndman 2023; FFORMS — Talagala et al. 2022) → meta-prior atualizado a cada batch, não congelado.
- **Conditional routing in MoE agents** (Switch-Transformer-style routing — Fedus 2022; agent routers 2024) → roteamento dinâmico para skip-LLM em casos fáceis.
- **Catch22 + tsfeatures como interlíngua entre LLM e meta-model** (Lubba et al. 2019; Hyndman et al. 2024) → features padronizadas.
- **Conformal regime adoption** (Stankeviciute et al. 2021; Xu & Xie 2024) → substitui DM gate por teste com maior poder em amostras curtas.
- **Pool curation por diversidade de erro** (Kourentzes et al. 2019; Wang & Hyndman 2023; Cawood & van Zyl 2024) → reduz pool ANTES da combinação.

Resultado conceitual: V4 deixa de ser **pipeline determinístico com LLM no meio** e vira **sistema agêntico federado que sabe mais a cada série e roteia inteligentemente**.

---

## 3. Os oito pilares do V4

### Pilar 1 — Memória Hierárquica Continual (NÚCLEO da inovação)

Três camadas persistidas em disco entre runs (SQLite ou JSONL append-only):

#### 1a) Memória Episódica
Uma linha por série já processada:
```
{series_id, dataset, features_catch22[22], features_tsfeatures[16],
 pool_signature, regime_chosen, lambda_eq, lambda_prior, survivors,
 chosen_score, full_mean_score, median_score, fforma_score, ade_score,
 oracle_regime, gate_passed, timestamp}
```
Acumula tudo. Após 100 séries, temos um "M4 federado próprio" — pequeno mas crescente.

#### 1b) Memória Semântica
Padrões abstraídos da memória episódica, atualizados a cada N séries:
- Clusters de features (k-means/HDBSCAN sobre catch22) → "tipos de séries observados".
- Por cluster: `{best_regime, best_lambda, avg_delta_vs_median, avg_delta_vs_fforma}`.
- "Se features cair no cluster C₃, regime=adaptive ganhou da mediana em 78% dos casos com delta médio −3.2%."

#### 1c) Memória Procedimental
Regras hard derivadas da semântica:
- "Se `seasonal_strength > 0.7 AND variance_ratio < 0.5` E o cluster C₃ teve gate_passed em ≥60% dos casos → confiança alta no regime=structured."
- "Sempre podar `{ONLY_FT_rf, ONLY_DWT_rf}` em séries deste dataset (taxa de poda = 92% nas últimas 50 séries)."

Essas regras são **lidas pelo Router** (Pilar 2) e podem **dispensar a chamada LLM** em casos confiantes.

> **Por que isso é inovador**: nenhum framework de combinação (FFORMA é congelado pós-M4; ADE não tem memória cross-série; V3 era stateless) faz acumulação federada com leitura agente. Aproxima-se mais de "MemGPT para combinação de previsões" — mas com governança estatística (cada regra tem n e p-valor da regularidade).

### Pilar 2 — Roteador Condicional Agêntico (velocidade + precisão)

Antes de chamar a cadeia completa de agentes, um **Router** lightweight decide qual caminho seguir:

```
features = compute_catch22(history) + compute_tsfeatures(history)
nearest_cluster, distance, cluster_stats = query_semantic_memory(features)

if distance < τ_close AND cluster_stats.gate_passed_rate > 0.6:
    # ROTA RÁPIDA — usa decisão procedimental, ZERO LLM
    regime = cluster_stats.best_regime
    lambda_eq = cluster_stats.best_lambda
    survivors = cluster_stats.typical_survivors
    return apply_combination(regime, lambda_eq, survivors)

elif τ_close <= distance < τ_far:
    # ROTA HÍBRIDA — meta-prior + 1 LLM ratifica/ajusta
    candidate = meta_model.predict(features)  # prior
    llm_decision = call_agent("Architect", brief={prior: candidate, features: features})
    return apply_combination(llm_decision)

else:
    # ROTA COMPLETA — caso novo, pipeline V3 enriquecido
    return full_v4_pipeline(features, memory)
```

**Impacto esperado**:
- Após 100 séries no NN5_WEEKLY, ~50-70% caem na rota rápida → 10× speedup médio.
- LLM só é chamada onde a memória NÃO sabe → onde ela realmente agrega valor.
- Em datasets novos, todas as séries iniciais caem na rota completa → memória se forma → próximas séries roteadas mais rápido.

> **Por que isso é inovador**: combina o trade-off speed/quality de Switch-Transformer-routing (Fedus 2022) com agentic LLM systems (CAMEL/AutoGen 2024) APLICADO à combinação de previsões.

### Pilar 3 — Meta-Prior Federado (FFORMA-like, mas continuamente atualizado)

Um modelo shallow (gradient boosting ou MLP de 2 camadas) treinado periodicamente em **toda memória episódica acumulada**:

- **Entrada**: features catch22 (22) + tsfeatures (16) + estatísticas do pool (n_models, disagreement_score, MCS_size).
- **Saída**: distribuição preditiva sobre (regime, λ_eq, "should_we_use_meta_prior_weights").
- **Treino**: a cada 25 séries novas, retreinar com objetivo `minimize chosen_score`.

Diferença crítica vs FFORMA:
- FFORMA aprende **pesos por modelo** num pool fixo do M4.
- V4 aprende **regime + λ + sinais de confiança** num pool variável → generaliza para pools customizados de cada dataset.

Diferença vs ADE:
- ADE aprende `features → predicted_loss` por modelo. Alta variância com 3 janelas.
- V4 aprende `features → regime_decision` globalmente, com bagging dos episódios passados. Variância baixa.

**Implementação prática**: começa com 0 séries → meta-model não treinado → V4 cai no full pipeline (V3-like). À medida que rodamos NN5_WEEKLY (111 séries), meta-model ganha massa. M4_WEEKLY (359 séries) leva o meta para regime de produção.

> **Por que isso é inovador**: FFORMA foi estado-da-arte 2020 e ainda é benchmark. V4 propõe **online-FFORMA-with-LLM-supervision** — o LLM corrige o prior quando o caso é fora de distribuição.

### Pilar 4 — Curadoria de Pool a Montante (antes da combinação)

Hoje a V3 faz prune por evidência local (RMSE/redundância). V4 acrescenta uma **etapa de curadoria de pool** ANTES do prune:

1. Calcular matriz de similaridade de **erros de validação** entre todos os modelos (não previsões — erros).
2. Clusterização hierárquica (Ward linkage, distance threshold 0.3).
3. Em cada cluster, manter **1 representante** = melhor `composite_score` na validação.
4. Pool sai de 23 → 8-10 modelos *de famílias distintas*.

Justificativa formal: Cawood & van Zyl (2024) mostraram que pools de alta correlação degradam combinação por inflar erro de variância dos pesos. Com 23 modelos correlacionados, `Var(ŵ)` ≈ pool_size/n_windows². Reduzir para 8 corta `Var(ŵ)` em ~3×.

**Pool curado é input do prune do ModelCritic**, não substituto. A LLM ainda pode podar dentro dos representantes se justificar — mas começa de um conjunto **já diverso e informativo**.

> **Por que isso é inovador**: Kourentzes 2019 e Wang 2023 fazem pool pruning, mas não combinam com agentes LLM nem com curadoria por clusterização de erro. V4 é a primeira proposta integrada.

### Pilar 5 — Gate Conformal (substitui DM, mais poder em amostra pequena)

DM gate em 24 resíduos é fraco. Conformal prediction não precisa de tantas amostras para calibração:

```
Para cada janela de validação k:
  delta_k = score(regime, fold_k) − score(anchor, fold_k)
Calibrar uma distribuição conformal não-paramétrica de delta:
  p_conformal = P(delta > 0 | calibração)
Adotar regime se p_conformal > 1 − α (α = 0.20)
```

α=0.20 é mais permissivo que DM-0.10. Compensa-se com:
- **âncora mais forte** = `pruned_median` (Pilar 7).
- **escalada gradual**: em vez de "regime sim/não", o sistema mistura `λ_adopt ∈ [0, 1]` proporcional à conformal-score → regime entra suavemente conforme evidência cresce.

> Referência: Stankeviciute et al. 2021 (conformal time series); Xu & Xie 2024 (conformal for combination).

### Pilar 6 — Triple Shrinkage com prior do meta-model

Hoje V3 faz double_shrinkage: `w = λ_eq · uniform + (1−λ_eq) · ridge`.

V4 faz triple_shrinkage incorporando o prior do meta-model:

```
w_meta = meta_model.predict_weights(features)  # prior FFORMA-like aprendido
w_ridge = ridge_solve(y_true_val, y_preds_val)  # empírico local
w_final = λ_eq · uniform + λ_meta · w_meta + λ_emp · w_ridge

onde:
  λ_eq + λ_meta + λ_emp = 1
  λ_eq = high se n_windows ≤ 3 (puzzle remedy)
  λ_meta = high se meta_model.confidence > 0.7
  λ_emp = high se há sinal forte na validação local
```

Três fontes de informação se misturam por confiança → estimador genuinamente mais low-variance que double_shrinkage.

### Pilar 7 — Âncora = Pruned Median (não Pruned Mean)

Mudança simples mas decisiva. Razões:

- Para amostras curtas com distribuição não-Gaussiana, mediana é asymptoticamente mais eficiente em ~70% dos cenários (Atiya 2020).
- Mediana é robusta a outliers de modelos individualmente ruins.
- A mediana atual está vencendo V3 — usar mediana como baseline força V4 a ter de bater mediana.

V4 reporta TODAS as âncoras: `full_mean`, `full_median`, `pruned_mean`, `pruned_median`, `fforma`, `ade`. Gate testa contra a MELHOR delas no conjunto de validação. Disciplina forte: V4 só "vence" se vencer o melhor baseline disponível.

### Pilar 8 — Agentes com novas responsabilidades

**SeriesAnalyst** (V3 → V4): agora recebe (catch22 + tsfeatures + STL + nearest_neighbors_from_memory) e produz SeriesProfile **comparativo** ("esta série é parecida com 12 séries anteriores onde regime=adaptive ganhou da mediana").

**MemoryConsulter** (NOVO, substitui ModelCritic): consulta a memória, retorna candidatos de decisão com evidência histórica. Em vez de raciocinar do zero sobre podar X ou Y, raciocina sobre "as últimas 20 séries semelhantes podaram principalmente {A, B, C} — confirmar?".

**CombinationArchitect** (V3 → V4): recebe meta-prior + sugestão da MemoryConsulter + features. Faz a escolha final de regime + λ_meta + λ_eq. Pode rejeitar o meta-prior se identificar padrão novo.

Esquema:
```
Router decide:
  rota_rapida    → memória procedimental aplica direto
  rota_hibrida   → CombinationArchitect só (com prior do meta-model)
  rota_completa  → SeriesAnalyst → MemoryConsulter → CombinationArchitect
                   → triple_shrinkage com 3 fontes (uniform, meta, empirical)
                   → conformal gate vs melhor âncora
```

---

## 4. Por que V4 bate V3 (especificamente)

| Causa V3 | Diagnóstico §1 | Mitigação V4 |
|---|---|---|
| Sem prior cross-series | A | Memória + meta-model |
| DM gate sub-potente | B | Gate conformal com α=0.20 + escalada suave |
| Pool redundante | C | Curadoria por clusterização de erro |
| Âncora "mean" frágil | D | Âncora = pruned_median + comparação contra FFORMA/ADE/mediana |
| Decisão local sem feedback | E | Loss agregada retroalimenta meta-model |

## 5. Por que V4 bate FFORMA

- **FFORMA é congelado** pós-M4. Cada dataset novo recebe pesos enviesados pela distribuição M4.
- **V4 acumula prior do próprio dataset**: após as primeiras 30-50 séries de NN5_WEEKLY, o meta-prior é especialista em NN5_WEEKLY.
- **V4 combina prior com LLM** para casos out-of-distribution; FFORMA não tem fallback de raciocínio.
- **V4 prune+curate o pool**; FFORMA usa pool fixo.
- **V4 reporta confiança por série**; FFORMA aplica pesos cegamente.

Trade-off: as primeiras ~20-30 séries do dataset, V4 pode perder para FFORMA (memória ainda fria). Importante reportar a curva de aprendizado.

## 6. Por que V4 bate ADE

- ADE estima `features → loss_predicted` por modelo. Com 3 janelas, alta variância.
- V4 estima `features → regime_decision` globalmente, **com bagging dos episódios anteriores**. Baixa variância.
- ADE não prune; V4 prune com floor estatístico.
- ADE não tem gate; V4 tem conformal gate.

## 7. Por que V4 bate Median (a barra mais alta para amostras curtas)

Mediana é a barra final. Para vencê-la:

- Pool curado de 8-10 modelos diversos → média desse pool curado já tende a bater mediana do pool original.
- Triple shrinkage com `λ_eq` alto preserva proximidade da uniforme/median quando há pouca evidência.
- Gate conformal só adota regime se delta vs `pruned_median` for positivo na validação.

V4 nunca **deveria** perder para mediana por design — se perder, é evidência de bug, não de método (e isso é testável).

## 8. Por que V4 bate DBA

DBA não é um competidor sério para a maioria dos datasets — o gráfico mostra DBA 0.3330 (pior que median 0.1918). V4 trivialmente bate DBA exceto em séries com padrões temporais não-lineares específicos (onde DBA poderia ajudar). Solução: **remover DBA do pool padrão da V4** e usá-lo apenas como candidato opcional via "regime=structured" se features sugerirem.

---

## 9. Mapa de implementação (file-by-file)

### Novos arquivos
- `orchestrator/memory/episodic.py` — read/write SQLite ou JSONL append. API: `add_episode(...)`, `query_nearest(features, k=10)`, `dump_all()`.
- `orchestrator/memory/semantic.py` — `update_clusters(episodes)`, `query_cluster(features)`, `cluster_stats(cluster_id)`.
- `orchestrator/memory/procedural.py` — derivação de regras + persistência. `derive_rules(clusters)`, `applicable_rules(features)`.
- `orchestrator/meta_model.py` — `MetaModel.fit(episodes)`, `predict(features) → (regime_dist, lambda_eq, confidence)`.
- `orchestrator/pool_curator.py` — `curate_pool(data, threshold=0.3)` → lista de modelos representativos.
- `orchestrator/conformal_gate.py` — `conformal_p(regime_residuals, anchor_residuals)`.
- `orchestrator_langchain/prompts/memory_consulter.md` — novo prompt.
- `orchestrator_langchain/agents.py` — novo factory `create_memory_consulter_agent`.

### Arquivos modificados
- `orchestrator/features.py` — adicionar `compute_catch22(history)` (via lib `pycatch22`) e `compute_tsfeatures(history)` (via `tsfresh` ou implementação inline).
- `orchestrator/pipeline.py` — `run_llm_pipeline_v4` com Router + 3 caminhos.
- `orchestrator/tools.py` — `memory_consulter_brief_tool` (puxa nearest neighbors), `combination_architect_brief_tool` enriquecido com meta_prior.
- `run_tsf_orchestrator.py` — `version="v4_continual"`, colunas novas (`memory_route`, `meta_prior_regime`, `meta_prior_confidence`, `conformal_p`, `nearest_neighbor_ids`, `pool_curated_size`, ...).

### Dependências
- `pycatch22` (pip-installable, conda também).
- `tsfresh` ou implementação manual das ~16 features tsfeatures-essenciais (forecastability/STL/Hurst já temos).
- `sqlite3` (stdlib).
- `scikit-learn` (já dependência) — para clustering e gradient boosting do meta-model.

---

## 10. Protocolo experimental para o paper

### 10.1 Setup
- Datasets: NN5_WEEKLY (111), M4_WEEKLY (359), ETTH1/H2/M1/M2 (7×4), US_BIRTHS (1), ANP_MONTHLY (filtrado).
- Total: ~519 séries.
- Modelos local: qwen3:14b (architect), qwen2.5:7b (memory_consulter — mais rápido).
- temp=0 em todos os agentes.

### 10.2 Sequência de execução
1. **Cold-start phase**: rodar V4 em NN5_WEEKLY (memória vazia). Marcar todas as séries com `memory_phase = "cold"`.
2. **Warm phase**: após NN5_WEEKLY, rodar M4_WEEKLY usando memória acumulada. Marcar `memory_phase = "warm"`.
3. **Cross-dataset transfer**: rodar ETT* usando memória de NN5+M4. Mede transferência cross-dataset.

### 10.3 Métricas a reportar
- SMAPE médio por dataset × método (V4, V3, FFORMA, ADE, median, mean, DBA, pruned_median).
- **Curva de aprendizado**: SMAPE V4 em janelas de 20 séries vs. ordem de processamento. Espera-se que decresça (memória ganha massa).
- **Taxa de uso de cada rota**: rápida / híbrida / completa por dataset. Espera-se >50% rápida após 50 séries.
- **Tempo médio por série** nas três rotas.
- **Conformal gate pass rate**: % de séries onde gate passou (vs `fellback_to_pruned_median`).
- **Win rate vs cada baseline** (Wilcoxon pareado, como na matriz que você mostrou).
- **Oracle regime match**: % de vezes que V4 escolheu o regime que viria a ter melhor score (validação oracle).

### 10.4 Ablações obrigatórias
- V4 sem memória (volta a ser V3-like com mais features).
- V4 sem Router (sempre rota completa).
- V4 com pool curado mas sem meta-model.
- V4 com meta-model mas sem LLM (FFORMA-like puro).
- V4 com âncora = pruned_mean (V3) vs pruned_median (V4).

Essas ablações isolam a contribuição de cada pilar.

### 10.5 Estudos de caso qualitativos
Selecionar 3 séries representativas:
1. Uma onde V4 ganhou de FFORMA com margem grande → mostrar memory hit + LLM reasoning.
2. Uma onde V4 caiu na rota rápida → mostrar inferência de ms vs minutos.
3. Uma onde V4 ainda perdeu para mediana → análise honesta do porquê.

---

## 11. Velocidade — como manter rápido com modelos locais

| Mecanismo | Speedup esperado |
|---|---|
| Rota rápida (skip LLM) após memória aquecida | 10–50× em séries em-distribuição |
| Cache de features catch22 por `series_id` | 2–3× |
| Pool curado (8 vs 23 modelos) → tools com payload menor → menos tokens | 1.5× |
| `keep_alive=10m` (já implementado) | 1.2–2× |
| Modelo menor (qwen2.5:7b) para MemoryConsulter | 2–3× |
| Paralelizar séries (multiprocessing pool, cada processo com `init_context`) | N_cpu× |

Estimativa: após warm phase, ~5-30 s por série em média (vs 15+ min hoje em séries problemáticas).

---

## 12. Inovações publicáveis — claims fortes do paper V4

**C1 — Primeiro sistema agêntico para combinação de previsões com memória continual.**
Diferencia-se de:
- LLM-direct forecasting (Time-LLM, Chronos, Moirai) → eles não combinam pools.
- Meta-learning combiners (FFORMA, ADE) → eles são offline-only.
- Multi-agent LLM systems → não têm aplicação em séries temporais.

**C2 — Roteamento condicional com governança estatística.**
O Router decide quando confiar na memória vs. quando invocar o LLM, com regras procedimentais que carregam `n` e taxa de sucesso. É a primeira aplicação desse padrão (popular em MoE) em combinação de previsões.

**C3 — Triple shrinkage federado.**
Combina três estimadores (uniform, meta-model federado, ridge empírico) com pesos λ_eq/λ_meta/λ_emp dirigidos por confiança. Generaliza o double shrinkage de Liu 2024.

**C4 — Curadoria de pool por clusterização de erro + agente.**
Pool reduction integrado com agentic system. Cawood & van Zyl 2024 fizeram clusterização, mas não com agentes nem com gate conformal.

**C5 — Gate conformal de regime.**
Substitui DM gate com maior poder em amostras curtas. Aplicação de conformal prediction (Vovk, Stankeviciute, Xu) ao **meta-nível** da combinação — não ao forecast em si.

**C6 — Curva de aprendizado mensurável e auditável.**
O paper pode mostrar literalmente a curva V4 melhorando ao longo da execução do dataset. Isso é dramático em apresentação e reflete a "característica viva" do sistema.

**C7 — Trade-off explícito speed × quality.**
Reportar a fração de séries em cada rota é uma contribuição operacional importante para reproduzibilidade prática.

---

## 13. O que NÃO faz parte do V4 (limites do escopo)

Para não inflar o paper:

- **Não muda os modelos-base**. Continua usando ARIMA/ETS/RF/Catboost/etc. que já existem.
- **Não faz forecast direto com LLM** (à la Time-LLM). Mantém a divisão de trabalho clara: LLM decide estrutura, estatística produz pesos.
- **Não usa foundation models** (Chronos, Moirai). Mantém local-only (Ollama).
- **Não faz hierarchical reconciliation**. Datasets do paper não têm hierarquia explícita.
- **Não faz probabilistic forecasting**. Foco no ponto, como V3.

---

## 14. Limitações honestas do V4

1. **Cold-start**: primeiras 20-50 séries de qualquer dataset novo não têm memória útil. Vão para rota completa (lenta). Mitigação: pré-popular memória com séries similares de outros datasets (transfer learning de memória).
2. **Memory drift**: se as séries de um dataset mudam de regime no tempo, regras procedimentais antigas viram veneno. Mitigação: TTL nas regras + reavaliação periódica.
3. **Compute cost do meta-model**: retreinar GB a cada 25 séries em 500-1000 séries é viável; em milhões, precisa SGD online.
4. **Curadoria pode ser conservadora demais**: se dois modelos têm erros correlacionados mas COMPLEMENTARES (não redundantes), agrupá-los perde sinal. Mitigação: usar similaridade de erro com sinal (não só magnitude).
5. **Gate conformal precisa de calibração**: poucos folds → distribuição conformal grosseira. Mitigação: bootstrap dos folds para inflar tamanho efetivo.
6. **Dependência do LLM persiste** no caminho híbrido/completo. A variabilidade entre LLMs continua um vetor de risco.

---

## 15. Por que isso é o paper certo

O reviewer de IJF/TKDE/NeurIPS-time-series-workshop hoje (2025-2026) está vendo:
- Foundation models pré-treinados (Chronos, Moirai) → "interessante mas comoditizado".
- LLM-direct forecasting (Time-LLM) → "modesto ganho, custo absurdo".
- Combinação clássica (FFORMA, ADE) → "saturado, todo mundo já cita".

**O que ele NÃO está vendo**: um sistema que (a) usa LLM para tomar decisões estruturais, (b) tem memória que cresce, (c) prove com gate conformal que cada decisão paga, (d) opera com governança estatística rastreável. **Esse é o gap. V4 ocupa exatamente esse gap**.

A história contada no paper:
> "Combinação de previsões está estagnada porque tratamos cada série como problema novo, e foundation models tratam o problema da forma errada (forecasting direto). Propomos um framework agêntico continual que aprende a decidir COMO combinar, melhorando a cada série, mantendo determinismo via floors estatísticos formais, e bate o estado-da-arte em séries curtas com pools heterogêneos."

Esse storytelling é forte. Bate FFORMA empiricamente + bate FFORMA conceitualmente (online vs offline) + bate Time-LLM em custo + bate ADE em variância. Ataca múltiplas frentes.

---

## 16. Próximos passos concretos

Em ordem de prioridade (primeiro corrige a derrota; depois implementa V4 completo):

### Sprint 1 (1–2 dias) — "V3.5": correções baratas que podem já reduzir o gap
1. Trocar âncora `pruned_mean` → `pruned_median` em `pipeline.py`.
2. Implementar curadoria de pool por correlação de erro em `tools.py` (pré-prune).
3. Adicionar `catch22` ao `compute_series_features` (1 lib).
4. Relaxar DM gate: `α=0.10 → α=0.20` E reportar `pruned_median` como baseline interno.
5. Remover DBA do pool padrão.

Hipótese: já reduz o gap para ≤0.5pp em SMAPE, talvez empate com mediana.

### Sprint 2 (3–5 dias) — Memória episódica + Router básico
1. `orchestrator/memory/episodic.py` com SQLite.
2. Logar episódios automaticamente ao final de cada série.
3. Router simples: kNN no espaço catch22; rota rápida só se k-vizinhos têm mesma decisão.
4. Re-rodar NN5_WEEKLY e medir.

### Sprint 3 (1 semana) — Meta-model + triple shrinkage + gate conformal
1. `MetaModel` em sklearn (GradientBoosting), retreino a cada 25 séries.
2. Triple shrinkage em `final_predictor.py`.
3. `conformal_gate.py` substitui DM.
4. Rodar batch completo NN5+M4_WEEKLY+ETT.

### Sprint 4 (3-5 dias) — Memória semântica/procedimental + agente MemoryConsulter
1. Clusterização HDBSCAN.
2. Regras derivadas.
3. Prompt do MemoryConsulter.
4. Curva de aprendizado para o paper.

---

## 17. Resumo em uma frase

> **V4 transforma a V3 de pipeline-estático-com-LLM em sistema agêntico continual com memória federada, roteamento condicional, meta-prior aprendido nos próprios dados e gate conformal sobre âncora pruned-median — entregando ganhos progressivos sobre FFORMA, ADE, mediana e DBA conforme a memória aquece, com auditabilidade total e governança estatística por design.**

---

## 18. Referências-âncora (literatura 2020–2026)

- Liu, B. (2024). *Double Shrinkage in Forecast Combination*.
- Cawood, P. & van Zyl, T. (2024). *Forecast combination with high-correlation pool elimination*.
- Wang, X. & Hyndman, R. J. (2023). *Forecast trimming*.
- Xu, C. & Xie, Y. (2024). *Conformal time-series prediction*.
- Stankeviciute, K. et al. (2021). *Conformal time-series forecasting*. NeurIPS.
- Hyndman, R. J. et al. (2024). *tsfeatures 2*.
- Lubba, C. H. et al. (2019). *catch22*.
- Talagala, T. et al. (2022). *FFORMS*. IJF.
- Montero-Manso, P. et al. (2020). *FFORMA*. IJF.
- Cerqueira, V. et al. (2019). *Arbitrated Dynamic Ensemble (ADE)*.
- Kourentzes, N. et al. (2019). *Treating and pruning forecast pools*. IJF.
- Claeskens, G. et al. (2016). *Forecast combination puzzle*. IJF.
- Hansen, P. R., Lunde, A., Nason, J. M. (2011). *Model Confidence Set*. Econometrica.
- Diebold, F. X. & Mariano, R. S. (1995). *Comparing predictive accuracy*. JBES.
- Harvey, Leybourne, Newbold (1997). *Testing the equality of prediction MSE*. IJF.
- Atiya, A. (2020). *Why does forecast combination work so well?* IJF.
- Stock, J. H. & Watson, M. W. (2004). *Combination forecasts*.
- Packer, C. et al. (2023). *MemGPT*.
- Park, J. S. et al. (2023). *Generative Agents*.
- Wu, Q. et al. (2024). *AutoGen*.
- Hong, S. et al. (2024). *MetaGPT*.
- Fedus, W. et al. (2022). *Switch Transformer*.
- Jin, M. et al. (2024). *Time-LLM*. ICLR.
- Ansari, A. et al. (2024). *Chronos*. arXiv.
- Das, A. et al. (2024). *TimesFM*. ICML.
- Woo, G. et al. (2024). *Moirai*. ICML.

---

**Quer que eu comece a implementar pelo Sprint 1?** Aposta minha: âncora pruned_median + curadoria de pool por correlação de erro + catch22 + relaxamento DM já fecham boa parte do gap em ≤2 dias de código.
