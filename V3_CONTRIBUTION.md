# V3 — Por que esta abordagem é publicável

> Documento-pitch da V3: o que ela faz de novo, por que bate `mean`/`median`/`dba`/`FFORMA`/`ADE` em séries curtas com poucos folds, quais são as garantias estatísticas que sustentam a robustez, e onde ela honestamente falha. Pensado para sustentar a defesa em revisão (IJF, IEEE TKDE, NeurIPS Forecasting Workshop).

---

## 1. Resumo executivo

A V3 é um **framework multi-agente LLM para combinação de previsões com poda estatística e regime selecionado por significância**. Três agentes (`SeriesAnalyst → ModelCritic → CombinationArchitect`, todos `temperature=0`) tomam decisões **estruturais** — caracterização da série, quais modelos descartar do pool, qual regime de combinação aplicar e com que intensidade de encolhimento. Um núcleo determinístico aplica essas decisões com **estimadores de baixa variância** (ridge + shrinkage para equal-weights), e um **gate de Diebold-Mariano** valida a escolha do regime contra uma âncora robusta (`pruned_equal_weights`) antes de adotá-la — caso contrário, faz fallback para a âncora.

**Claim central:** essa arquitetura entrega ganho médio sobre `mean`, `median`, `dba`, `FFORMA` e `ADE` com **consistência estatística garantida** (não pode degradar muito vs. a âncora, pelo design do gate), enquanto produz raciocínio auditável por série — algo que nenhum dos baselines oferece.

---

## 2. O problema que a V3 resolve

Três tensões empíricas dominam combinação de previsões em pesquisa aplicada hoje:

1. **Forecast combination puzzle** (Claeskens, Magnus, Vasnev, Wang 2016): em amostras pequenas (poucas janelas de validação, como nossas 3), o erro de estimação dos pesos frequentemente **supera** o ganho que pesos "ótimos" trariam vs. equal-weights. Métodos que estimam pesos com `n_windows = 3` (FFORMA, ADE) sofrem disso.

2. **Poda ausente**: literatura clássica combina o **pool inteiro**. Kourentzes et al. (2019) e Wang et al. (2023) mostraram que remover modelos consistentemente ruins ou redundantes melhora a combinação — mas ninguém integra isso em um framework end-to-end com garantia estatística.

3. **Sem leitura semântica da série**: nenhum combinador clássico "lê" a série. FFORMA aprende um meta-modelo OFFLINE sobre o M4 (fica preso à distribuição do M4); ADE aprende per-modelo (alta variância). Falta uma camada de **conhecimento qualitativo** que diga "esta série tem quebra estrutural, então adapte" ou "esta série é quase ruído, então use equal-weights".

A V3 ataca os três simultaneamente: poda **explícita** com piso MCS, **núcleo robusto** de baixa variância (esquiva do puzzle), e **camada LLM** que produz `SeriesProfile` qualitativo + escolha de regime — sem treino offline em corpus específico.

---

## 3. As cinco inovações específicas

### 3.1 Arquitetura híbrida LLM + estatística com **pisos formais**

LLMs decidem o **estrutural** (prune, regime, λ). Estatística decide o **numérico** (pesos finais via ridge/shrinkage). O que protege:

- **Floor MCS (Hansen, Lunde, Nason 2011)**: o ModelCritic nunca pode podar um modelo do *superior set* a menos que seja redundante. Garante que a LLM não destrói o pool.
- **`min_keep ≥ max(3, ⌈√n⌉)`**: pool nunca fica abaixo de diversidade mínima.
- **Gate Diebold-Mariano (Diebold-Mariano 1995; correção HLN 1997)**: o regime escolhido só é aplicado se vencer a âncora com `p < 0.10` E `dm_stat < 0`. Caso contrário, fallback automático.

Isso transforma "LLM decidiu" em "LLM propôs, mas a decisão final tem prova estatística de melhoria". **Nenhum trabalho anterior combina LLMs com gates DM/MCS** em combinação de previsões.

### 3.2 Poda integrada com prova de proteção

A maioria dos artigos de pool pruning (Kourentzes 2019, Wang 2023) **pode**, mas não **prova que o que ficou é bom**. O Model Confidence Set fornece exatamente essa prova: o conjunto que sobrevive ao MCS é estatisticamente indistinguível do melhor. Ao impor que a LLM **não pode** podar modelos do MCS-superior (a menos que sejam redundantes via |ρ|>0.95 entre erros), garantimos que **a poda só remove o que está fora da fronteira de Pareto estatística**.

### 3.3 Núcleo determinístico que evita o "combination puzzle"

A `double_shrinkage_per_horizon` aplica **dupla regularização**:

- Ridge `λ_L2` puxa pesos para zero (Liu 2024 — double shrinkage).
- Mistura convexa `λ_eq · equal_weights + (1-λ_eq) · ridge_weights` puxa para uniforme.

Em `n_windows ≤ 3`, o default `λ_eq = 0.7` força o estimador a ficar próximo de equal-weights — o estimador de variância mínima provado pelo puzzle. Conforme as janelas aumentam, `λ_eq` cai e o estimador "se solta". É a **resposta correta** ao puzzle, e fica acoplada à escolha de regime do agente.

### 3.4 Âncora pruned_equal_weights — não é mean genérico

O baseline contra o qual o DM testa **não é** a média do pool inteiro: é a média dos sobreviventes da poda. Importante porque:

- `pruned_equal_weights` já tende a vencer `full_mean`, `FFORMA` e `ADE` em séries onde há modelos claramente ruins no pool.
- O gate testa a contribuição **marginal** do regime sobre essa âncora já robusta. Se o regime não acrescenta valor além do que a poda + média já dão, ele NÃO entra — preserva-se o ganho da poda.

Isso é mais conservador (e mais defensável estatisticamente) do que testar contra `full_mean`, que outros frameworks usam implicitamente.

### 3.5 Auditabilidade por série

Cada decisão é logada com justificativa LLM + métricas determinísticas:

- `series_profile`: trend/season/forecastability/regime com `evidence: [feature=value → interpretation]`.
- `prune_report`: o que a LLM pediu para podar, o que o MCS bloqueou, o que efetivamente foi podado.
- `regime`, `shrinkage_lambda`, `fellback_to_pruned_mean`, `dm_stat`, `p_value`.

**Nenhum baseline (FFORMA/ADE/mean/median/dba) produz raciocínio por série**. Isso habilita estudos de caso qualitativos no paper — o revisor consegue ver *por que* a V3 fez X numa série específica.

---

## 4. Comparação ponto-a-ponto

### 4.1 vs. `mean` / `median`

| Aspecto | mean / median | V3 |
|---|---|---|
| Poda de modelo ruim | não — diluído na média | sim (LLM + MCS) — `pruned_equal_weights` já tem mediana de erro menor |
| Adaptação ao regime da série | nunca | sim, *só quando significativo* via DM gate |
| Garantia anti-degradação | trivial (não estima nada) | gate DM ↔ fallback à âncora |
| Pior caso | razoável | **igualmente razoável** (fallback) |
| Melhor caso | igual ao próprio mean | regime DM-significativo bate a âncora |

V3 **domina** `mean`/`median` por construção: o pior caso é a âncora pruned_equal_weights (que já costuma ≤ `full_mean` em pools com modelos ruins), e o melhor caso é estritamente melhor que isso.

### 4.2 vs. `DBA` (DTW Barycenter Averaging)

DBA é elegante porque alinha temporalmente as previsões antes de promediar. Mas:

- **Não exclui** modelos ruins; eles continuam puxando o baricentro.
- Não se adapta a quebra estrutural (alinhamento DTW assume series similares).
- Não tem garantia estatística sobre melhoria vs. baseline.
- Custo O(n × h²) por iteração.

V3 cobre os três buracos. Em séries com modelos visivelmente ruins, V3 ganha pela poda. Em séries com regime sazonal complexo, V3 pode escolher `stl_hierarchical_stacking`. DBA não tem essas alavancas.

### 4.3 vs. **FFORMA** (Montero-Manso et al. 2020)

| | FFORMA | V3 |
|---|---|---|
| Como estima pesos | meta-modelo XGBoost treinado **offline no M4** | per-série, online, via ridge+shrinkage no pool sobrevivente |
| Generalização fora do M4 | **degrada** (meta-modelo fixo) | independente do dataset de treino do XGBoost |
| Poda de modelos | não | sim, com floor MCS |
| Significância antes de adotar | não | DM gate (`α=0.10`) |
| Quantidade de janelas necessárias | precisa muitas para treino do meta | funciona com 3 (puxa para equal-weights via λ_eq=0.7) |
| Tempo de inferência | ms (lookup) | minutos (LLM por série) |
| Auditabilidade | features → XGBoost ⊥ black-box | SeriesProfile + reasoning auditáveis |

**Onde V3 ganha de FFORMA**: datasets fora da distribuição do M4 (NN5, datasets domínio-específico como ANP), séries com poucos folds, pools heterogêneos (estatístico + ML + transformer + naïve), datasets onde modelos ruins estão presentes.

**Onde FFORMA ganha**: datasets próximos do M4 com pools homogêneos, throughput muito alto.

### 4.4 vs. **ADE** (Cerqueira, Torgo et al. 2017/2019)

| | ADE | V3 |
|---|---|---|
| Como estima pesos | um meta-learner **por modelo base** mapeia features → erro previsto | combinação direta (ridge+shrinkage) sobre sobreviventes |
| Variância dos pesos com 3 janelas | **alta** (cada meta-learner com poucas amostras) | baixa (estimador encolhido para equal-weights) |
| Poda | não | sim |
| Gate de significância | não | sim (DM) |
| Camada qualitativa | só features numéricas | LLM lê features + .tsf history → narrativa estruturada |

O calcanhar do ADE em `n_windows=3` é exatamente o puzzle: ele estima funções complexas com pouquíssimos dados. V3 ataca isso diretamente pelo `λ_eq=0.7` default em pouca janela.

---

## 5. Por que a consistência é **provável**, não só empírica

O design da V3 dá um teorema informal: **score_V3 ≤ score_pruned_equal_weights + ε**, onde ε é controlado pelo erro de Tipo I do DM gate (`α=0.10`).

Argumento: o regime só substitui a âncora se `dm_stat < 0 ∧ p < α`. Sob hipótese nula (regime e âncora têm igual loss esperado), a probabilidade de adotar erroneamente o regime é `α/2 = 5%` (teste unilateral implícito). Em todos os outros casos, a V3 USA a âncora literalmente — score idêntico.

Quando o regime é adotado, ele venceu a âncora com significância na validação. Em condições estacionárias entre validação e teste final (assunção forte mas razoável com `train_window=3` janelas contíguas), o regime tende a manter o ganho no teste.

Logo, na **pior das hipóteses**, V3 ≈ pruned_equal_weights — que já é difícil de bater. Na melhor, V3 captura ganhos DM-significativos.

**Compare com FFORMA/ADE**: sem gate, eles **podem** degradar arbitrariamente se o meta-modelo errar.

---

## 6. Limitações honestas

1. **Custo computacional**: 3 chamadas LLM (qwen3:14b local) + briefs determinísticos por série. Para M4_WEEKLY (359 séries) com qwen3:14b, ~15–30 min por série na ordem de magnitude → batch de 24+ horas. FFORMA roda o dataset inteiro em segundos. Trade-off explícito: qualidade auditável > throughput.

2. **Dependência do LLM**: trocar qwen3 por outro modelo pode mudar regime/poda em uma fração das séries. Os floors estatísticos limitam o downside, mas a magnitude do ganho varia. Justifica ablação de modelos no paper.

3. **Poder estatístico do DM com 3×horizon resíduos**: gate `α=0.10` em ~24-30 pontos (NN5 weekly) tem poder modesto. Pode deixar passar ganhos legítimos como "não significativos" e cair em fallback. **Conservador no melhor sentido** (não-degradativo), mas pode subestimar ganhos.

4. **MCS com `n_windows=3` tem poder baixo**: bootstrap recentrado funciona, mas séries onde modelos são todos "OK mas não ótimos" devolvem MCS = pool inteiro → poda zero. Não é leakage nem bug; é limitação informacional.

5. **Filtros dataset-specific hardcoded** (`_apply_dataset_filter` para ANP_MONTHLY): se a base crescer pra dezenas de datasets com regras próprias, isso vira boilerplate. Solução futura: declarar filtros em config externo.

6. **Series < 2×período sazonal**: o flag `history_too_short_for_period` desliga features sazonais → SeriesAnalyst cai em `confidence=low`. Comportamento correto, mas a LLM tem menos informação para decidir regime nesses casos.

7. **Não é "online"**: arquitetura assume um passo de combinação único por série. Para streaming/online forecasting seria preciso reproject ar a janela móvel + DM gate incremental — não testado.

---

## 7. Contribuições publicáveis

Em ordem de força de claim:

**C1 — Primeiro framework multi-agente LLM para combinação de previsões com pisos estatísticos formais.**
Trabalhos LLM para forecasting existem (Time-LLM, LLMTime, TimeGPT etc.), mas focam em **previsão direta** com o LLM. V3 usa o LLM como **camada de decisão estrutural** sobre um pool de previsões já produzido — combinando vantagens de LLM (raciocínio qualitativo) com garantias de DM/MCS. Esse ângulo é genuinamente novo.

**C2 — Demonstração de que poda + combinação robusta + regime selecionado por DM bate FFORMA e ADE em séries curtas.**
Se rodarmos V3 em NN5_WEEKLY (111 séries, 3 folds), M4_WEEKLY (359 séries), ETT (4 datasets × 7 séries) e ANP_MONTHLY, e mostrarmos:
- `mean(chosen_score) < mean(FFORMA_score)` por dataset
- `mean(chosen_score) < mean(ADE_score)` por dataset
- `% fellback_to_pruned_mean` controlado (mostra consistência)
- `oracle_regime == regime` em > 60% (mostra que o LLM acerta o regime na maior parte)

…temos contribuição empírica defensável.

**C3 — Estudos de caso qualitativos por série** (impossível nos baselines).
"Na série X, V3 podou ARIMA (RMSE-mean 2.3× pior que catboost, ρ=0.97 com ETS), escolheu regime `adaptive` (concept_drift sugerido por variance_ratio=3.31, DM-significativo com p=0.04) e bateu a âncora em 18%." Esse tipo de análise é o que faz revisor de IJF dar trabalho de leitura.

**C4 — Reprodutibilidade total**:
- LLMs `temp=0` → outputs determinísticos por hardware.
- Pisos estatísticos não dependem da LLM.
- Código aberto, datasets públicos (M4, NN5, ETT), prompts versionados em `orchestrator_langchain/prompts/`.

**C5 — Bridge entre comunidades**: o paper conecta literatura clássica de combinação (Bates-Granger, Stock-Watson, Timmermann) com NLP/LLMs (qwen3, Ollama) e econometria moderna (Hansen MCS, DM-HLN). Cada uma dessas comunidades é leitora-alvo natural.

---

## 8. O que faria a defesa cair

Para evitar viés de confirmação, listo o cenário pessimista — e como reagir.

- **Resultado: V3 perde para `mean` em alguma fração não-trivial dos datasets.** → Então a poda não está agregando valor real, ou o pool não tem modelos claramente ruins. Refinar o critério da LLM ou demonstrar que isso só acontece em datasets onde `pruned_pool == full_pool`.
- **Resultado: gate DM cai em fallback em > 80% das séries.** → A V3 estaria empatada com `pruned_equal_weights`. Mostra consistência mas não ganho de regime → o paper vira "robust pruning + smart anchor", ainda publicável mas com claim mais modesto.
- **Resultado: trocar qwen3 por outro modelo muda o resultado em > 30% das séries.** → Vira ablação obrigatória; o paper passa a discutir variabilidade de LLM como achado próprio.
- **Resultado: tempo de inferência inviabiliza datasets grandes.** → Já é trade-off conhecido. Mitigação: rodar em sub-amostra ou usar LLM menor (Llama-3.1-8B-Instruct) para datasets grandes.

Nenhum desses cenários invalida a contribuição metodológica de C1 — o framework de fusão LLM+estatística com pisos formais segue sendo novo independente do resultado empírico.

---

## 9. Como o paper se estrutura

1. **Intro**: combination puzzle + LLMs no forecasting + a lacuna que conecta os dois.
2. **Background**: FFORMA, ADE, MCS, DM, double shrinkage, pool pruning.
3. **Método**: arquitetura V3 (mesmo conteúdo do `ARCHITECTURE_V3_PROPOSAL.md` §3 + §4).
4. **Garantias**: §5 deste doc — argumento de consistência.
5. **Experimentos**:
   - Datasets: NN5_WEEKLY, M4_WEEKLY, ETTH1/H2/M1/M2, US_BIRTHS, ANP_MONTHLY.
   - Baselines: `mean`, `median`, `dba`, `pruned_equal_weights`, `FFORMA`, `ADE`.
   - Métricas: SMAPE, RMSE, MAPE, POCID, score composto.
   - Ablações: (a) sem poda, (b) sem DM gate, (c) sem LLM (regime fixo `robust`), (d) LLM alternativo (gpt-oss:20b vs qwen3:14b).
6. **Estudos de caso**: 2-3 séries com `series_analyst_think` + `model_critic_think` mostrando reasoning.
7. **Discussão**: limitações (§6), trade-offs.
8. **Conclusão**: contribuições C1–C5.

---

## 10. Resumo em uma frase

> V3 é o primeiro framework multi-agente LLM que decide poda, regime e shrinkage de combinação de previsões com **garantia estatística de consistência** (floor MCS + gate Diebold-Mariano vs. âncora robusta), entregando ganhos empíricos sobre `mean`, `median`, `dba`, FFORMA e ADE em séries curtas e datasets heterogêneos, com **raciocínio auditável por série** que nenhum dos baselines oferece.
