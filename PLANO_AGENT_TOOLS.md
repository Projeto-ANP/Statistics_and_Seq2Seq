# Plano — por que o agente usa 6 de 24 tools, e o que fazer a respeito

Investigação em três frentes: (1) forense das 182 trajetórias reais do v4 no
ANP_MONTHLY, (2) um experimento novo que operacionaliza a hipótese "e se o agente
visse todas as séries antes, como ADE/FFORMA?", (3) estado da arte em seleção de
tools por agentes LLM (2025–2026). Números marcados 📊 foram medidos aqui.

Data: 2026-07-30.

---

## 0. Respostas diretas às perguntas feitas

| pergunta | resposta curta | evidência |
|---|---|---|
| "Tem tool demais e ele satura em 6?" | **Não é excesso — é âncora.** O espaço útil real é ~10 tools (6 de diagnóstico são pré-injetadas no prompt, 6 `combine_*` são atalhos redundantes documentados como desnecessários no próprio prompt). Dentro da família de pesos, o uso decai por posição/familiaridade: 462→223→10→2→1 | §1.2 |
| "São generalistas demais?" | Parcialmente: 591 das 756 avaliações são `weighted`, e a maioria dessas é **numericamente a média do pool disfarçada** (113/182 vencedores ≈ média). O agente re-testa o mesmo ponto com rótulos diferentes | §1.3 |
| "É problema de prompt?" | **Em grande parte, sim.** O exemplo "A TYPICAL SEQUENCE" cita `weights_inverse_error` textualmente → 462 usos. A tool nova (`pooled_meta_model`) foi chamada 1 vez e **falhou pela nossa própria guarda** | §1.2, §1.4 |
| "Precisamos de mais análises/leque maior?" | Não de mais tools de análise (as 6 de diagnóstico têm uso **zero** porque o resultado já está no prompt — comportamento racional). Precisamos de **conhecimento de dataset**, não de série | §2 |
| "Separar em mais agents?" | **Não em peer multi-agent.** Evidência 2025-26: com orçamento de tokens igual, single-agent ≥ multi-agent; o padrão vencedor é orquestrador + subagentes efêmeros — que é o que nosso `diagnostician` já é (e está por testar no ANP) | §3.2 |
| "Ver todas as séries antes, como ADE/FFORMA, ajudaria?" | **Sim, ajuda — confirmado por experimento** (📊 ANP 0.2223→0.2191; NN5 0.1207→0.1166, batendo ADE e FFORMA no NN5). Mas prior sobre *estratégias fixas* não fecha o gap no ANP; fechar exige prior sobre *modelos* com features ricas — que é o nosso meta-modelo pooled, hoje bloqueado por design | §2 |

---

## 1. Forense das trajetórias do v4 (ANP, 182 séries) 📊

### 1.1 O espaço de ação efetivo não tem 24 tools

| grupo | tools | uso | por quê |
|---|---|---|---|
| diagnóstico | 6 | **0 chamadas** | o resultado já é injetado no prompt (series card, pool card) — chamar seria redundante. **Racional, não falha** |
| `combine_*` | 6 | 2 | o próprio prompt diz "You do NOT need combine_* first" — obedecido |
| `weights_ols` | 1 | retirada (gate de 3 janelas) | — |
| **espaço real** | **~10** | concentrado em 4–5 | é aqui que mora o problema |

Distintas tools por série: média **3.86** (moda 4). Não é um agente perdido — é um
agente com um *hábito*.

### 1.2 O hábito é ancorado no exemplo do prompt

Primeira ação do loop (182 séries): `weights_inverse_error` 90×, `prune_redundant`
75×, `weights_softmax_neg_error` 10×, resto ≈ 0.

Uso dentro da família de pesos, na ordem em que aparecem no catálogo:

```
weights_inverse_error      462   ← citada VERBATIM no exemplo "A TYPICAL SEQUENCE"
weights_softmax_neg_error  223   ← adjacente/similar à anterior
weights_error_trend         10
weights_feature_based        2
weights_pooled_meta_model    1   ← e essa 1 chamada FALHOU (ver 1.4)
```

Decaimento monotônico por posição — exatamente o padrão que a literatura de 2025
documenta ([BiasBusters: position bias em seleção de tools](https://arxiv.org/pdf/2605.18857);
[degradação com tamanho de catálogo](https://arxiv.org/html/2605.24660v1)). Mas com
um agravante nosso: o **exemplo trabalhado** no system prompt usa
`select_stable → weights_inverse_error → evaluate`. O agente reproduz o template.
`select_stable` só tem 4 usos porque a Fase 2 já semeia pools estáveis (de novo:
racional) — sobra `prune_redundant` como único movimento de pool não-redundante, e
`weights_inverse_error` como o peso "oficial" do exemplo.

### 1.3 O desperdício real: aliasing de estratégias

- 591/756 avaliações são `combine=weighted`;
- 113/182 estratégias vencedoras são **numericamente equivalentes à média do pool**
  (`equivalent_to_pool_mean=True`, concentração média 0.0028);
- bigrama `evaluate→evaluate` 125×, `[already tested]` 30×.

Ou seja: boa parte das 10.4 iterações médias é gasta re-derivando a média simples
sob nomes diferentes. O agente não explora pouco por falta de opções — explora
pouco porque **muitas das suas ações são apelidos do mesmo ponto no espaço de
previsões**, e nada avisa isso a ele.

### 1.4 A tool nova era inutilizável no fluxo natural do agente

A única chamada de `weights_pooled_meta_model` (série 151):

```
ERROR invalid_argument: pool 'pool4' is re-selected per backtest fold under
nested_selection, but weights_pooled_meta_model computes its weights once...
```

O agente fez tudo "certo": podou o pool (seu hábito dominante) e pediu os pesos
pooled sobre o pool podado. **Nossa guarda de invariância por fold rejeitou.** Como
o fluxo dominante é `prune → weights`, a tool só funcionaria em `pool_full` — que o
agente raramente pondera. Design meu que não sobrevive ao contato com o
comportamento real. Corrigível sem abrir mão da correção estatística (P0 abaixo).

(Nota: withheld em 17 séries por ausência do meta-modelo — provável artefato do
resume 0–80/81–181; verificar no P0.)

### 1.5 Os 90 "unparsed" são respostas VAZIAS, não confusão de formato

Todos os `parse_failures` amostrados têm `raw=''` — são gerações vazias que
persistiram após os 2 retries de vazio (gpt-oss gasta o orçamento no canal de
raciocínio harmony e não emite nada no canal final). Problema conhecido do
ecossistema gpt-oss+Ollama ([#11781](https://github.com/ollama/ollama/issues/11781),
[#11800](https://github.com/ollama/ollama/issues/11800)); e **não** se resolve com
`format=json` — isso deixa o gpt-oss 100% vazio
([#11867](https://github.com/ollama/ollama/issues/11867)). Nosso protocolo de texto
de 3 linhas está certo; o ataque é operacional (P4).

---

## 2. O experimento decisivo: prior de dataset (a hipótese ADE/FFORMA) 📊

**Desenho.** 14 estratégias (full/top5/stable5/stable7/islands/divpen5/greedy ×
mean/median), validação aninhada honesta (seletor nunca vê a janela que pontua).
*Prior de dataset* = sMAPE médio de validação de cada estratégia sobre **todas as
outras séries** (leave-one-series-out — só validação, teste jamais). *Shortlist* =
top-M do prior; escolha local = argmin da validação da própria série dentro da
shortlist.

| M | ANP (teste) | NN5 (teste) |
|---|---|---|
| 1 (só o prior global) | 0.2206 | 0.1199 |
| 2 | 0.2192 | 0.1201 |
| **3** | **0.2191** | 0.1174 |
| **4** | 0.2210 | **0.1166** |
| 14 (sem shortlist) | 0.2197 | 0.1198 |
| **agente atual** | **0.2223** | **0.1207** |
| melhor fixa | 0.2181 | 0.1149 |
| FFORMA | 0.2166 | 0.1197 |
| ADE | 0.2177 | 0.1178 |

Leituras honestas:

1. **A hipótese do usuário está certa na direção**: conhecimento de dataset +
   escolha local bate o agente atual nos dois datasets, e no NN5 **bate ADE e
   FFORMA** (0.1166 < 0.1178 < 0.1197).
2. **No ANP não fecha o gap para o FFORMA** (0.2191 vs 0.2166). O prior sobre 14
   estratégias fixas é um aprendiz mais fraco que o FFORMA (42 features × LightGBM
   sobre *modelos*). O caminho para fechar é o nosso meta-modelo pooled — mesma
   família do FFORMA — que hoje usa **4 features (das quais catch22, já computado
   por série, está fora)** e é bloqueado pela guarda (§1.4).
3. A shortlist M≈3–4 é melhor que M=14 nos dois datasets: **restringir o espaço
   com evidência de dataset é melhor do que deixar tudo aberto** — coerente com
   RAG-de-tools ([RAG-MCP: 13.6%→43.1% de acerto ao restringir](https://arxiv.org/abs/2505.03275),
   [tool retrieval 2025](https://webscraft.org/blog/tool-rag-scho-robiti-koli-u-agenta-zabagato-instrumentiv?lang=en)).
4. Spearman(prior de validação → teste, nível dataset): **+0.398 (ANP), +0.218
   (NN5)** — fraco no nível série, mas *utilizável* no nível dataset. É o mesmo
   princípio da "calibração amortizada" do TSOrchestr (item 2 do
   `insights_trabalhos.md`): o que não dá para decidir por série, decide-se uma vez
   por dataset.

**Sobre "fazer ele testar todas as tools por série":** a evidência vai contra.
`greedy_combo` — que otimiza exaustivamente a validação por série — é a pior
estratégia nos dois datasets (relatório anterior), e mais avaliações por série não
melhoraram teste (v4 vs v3: +15 vitórias de validação, resultado igual/pior).
**Teste-tudo uma vez por dataset (pré-pass determinístico, barato), não por série.**
A cobertura vira observável: o pré-pass garante que toda estratégia foi exercitada
em toda série — sem depender do LLM escolher.

---

## 3. Estado da arte relevante (para o paper)

### 3.1 Seleção de tools
- Degradação com catálogo grande e **viés de posição** (início/fim > meio):
  [BiasBusters 2025](https://arxiv.org/pdf/2605.18857), [How Many Tools Should an
  LLM Agent See?](https://arxiv.org/html/2605.24660v1), [TRAJECT-Bench](https://arxiv.org/pdf/2510.04550).
  Nosso decaimento 462→223→10→2→1 dentro da família de pesos é uma instância
  disso *mais* ancoragem por exemplo trabalhado — mensurada em produção, não em
  benchmark sintético. **Isso é material de paper.**
- Mitigação padrão: reduzir/rotear o conjunto exposto (Tool-RAG). Nossa versão:
  shortlist por prior de dataset (§2), que além de reduzir exposição, é
  estatisticamente fundamentada.

### 3.2 Single vs multi-agent
- Com orçamento de tokens pareado, single-agent iguala ou supera multi-agent; as
  vantagens reportadas frequentemente se explicam por computação extra
  ([evidência 2025-26](https://medium.com/@mjgmario/single-agent-vs-multi-agent-systems-when-coordination-helps-hurts-and-pays-off-57735ee7916d)).
  Taxonomia de 14 modos de falha específicos de multi-agent (MAST).
- O padrão que sobreviveu: **orquestrador + subagentes efêmeros especializados**
  (Claude Code, agents-as-tools). Nosso `diagnostician` (Fase 1) já é exatamente
  isso — papel isolado, contexto próprio, não conversa com o combinador.
  **Recomendação: rodar a ablação já preparada (no ANP), não construir mais
  agentes.**

### 3.3 Reuso de experiência entre tarefas
- [ExpeL](https://arxiv.org/abs/2308.10144), Agent Workflow Memory, Dynamic
  Cheatsheet: agentes melhoram reutilizando insights de tarefas anteriores no
  contexto. Nossa versão *auditável e sem vazamento*: o **dataset card** (P1) —
  experiência agregada de validação das outras séries, injetada no prompt. Difere
  de memória livre: é determinística, order-independent (pré-pass), e só validação.

### 3.4 gpt-oss + Ollama
- Falhas de parse do canal de tool-call harmony e vazios com `format=json` são
  issues conhecidas ([#11781](https://github.com/ollama/ollama/issues/11781),
  [#11800](https://github.com/ollama/ollama/issues/11800),
  [#11867](https://github.com/ollama/ollama/issues/11867)). Valida nosso protocolo
  de texto + retries; aponta upgrade de servidor e reasoning-effort como mitigação.

---

## 4. O plano, ranqueado por valor/esforço

### P0 — Destravar e semear o meta-modelo pooled (o fix do §1.4) · ~meio dia
1. **Guardar erros preditos por modelo** (dict nome→erro) na recipe, e compor os
   pesos **por fold** a partir da membership do fold (softmax sobre os membros).
   Fold-safe por construção → **remove a guarda** que rejeitou o agente. A tool
   passa a funcionar sobre qualquer pool, inclusive podado (o hábito do agente).
2. **Semear `weighted(pooled_meta_model, pool_full)` na Fase 2** — vira baseline
   automática avaliada em 100% das séries. Resposta estrutural ao "tool nunca
   chamada": promover de opção a piso. (Mesma lógica que já validamos com
   `seed_stable_pools`.)
3. **Features: 4 → 4 + catch22** (22 features já computadas por série, hoje
   descartadas pelo meta-modelo). No ANP as features têm variância real
   (`seasonal_strength` σ=0.226 vs 0.0001 no NN5) — é onde o upgrade paga.
4. Verificar os 17 withheld do resume.
- **Métrica de sucesso:** braço determinístico ANP ≤ 0.218 (hoje 0.2223); a seed
  pooled vence ≥ 15% das séries.

### P1 — Dataset card + shortlist recomendada (o §2 em produção) · ~1 dia
No pré-pass que já existe (ele já carrega todas as séries): computar o prior LOSO
de estratégias e injetar no prompt um bloco novo:

```
DATASET CARD (validation-only, computed across the other 181 series):
  strategy families ranked by mean validation sMAPE: full_median 0.2251, ...
  recommended shortlist for this dataset: [full_median, stable7_median, stable5_median]
  caution: 'weighted' proposals were numerically the pool MEAN in 62% of series here
```

Recomendação, **não** restrição — o agente mantém a liberdade (exigência sua), mas
deixa de decidir no vácuo. Complemento: linha "weight methods not yet tried this
series: [...]" (espelho da nota de unscored que já temos).
- **Métricas:** entropia de uso da família de pesos ↑; `evaluate→evaluate`
  redundante ↓; ANP ≥ 0.219 no braço com agente.

### P2 — Anti-aliasing nas observações · ~2h
Quando `evaluate_strategy` produz previsões numericamente idênticas a uma tentativa
existente (já temos `_numerically_identical` para os twins da confiança), dizer na
observação: `"numerically identical to a3 (the pool mean) — this strategy adds
nothing new"`. Ataca diretamente as 591 avaliações `weighted`≈média e os bigramas
`evaluate→evaluate`.
- **Métrica:** nº de tentativas numericamente distintas por série ↑ (hoje ~3–4).

### P3 — Prompt: desancorar o exemplo · ~1h
Trocar `weights_inverse_error` no "A TYPICAL SEQUENCE" por um placeholder
(`weights_<method>` com a lista de métodos) e mencionar que métodos diferentes
exploram sinais diferentes (erro médio vs tendência do erro vs cross-series).
Barato; mede-se pela distribuição de primeira-ação.
- **Métrica:** primeira ação `weights_inverse_error` de 90/182 → <50; família de
  pesos com entropia ≥ 1.5 bits.

### P4 — Ops do LLM · ~1h + rodada
Upgrade do Ollama no servidor; testar reasoning-effort "low" para gpt-oss;
**nunca** `format=json` (issue conhecida); manter retries atuais.
- **Métrica:** unparsed 90 → <20; `llm_error` 2 → 0.

### P5 — Ablação do diagnosticador **no ANP** (já preparada) · custo = 1 rodada
Já apontamos antes: era para rodar no ANP (features com variância), não no NN5.

### P6 — O que NÃO fazer
- **Peer multi-agent / debate**: evidência contrária sob orçamento pareado (§3.2);
  já abandonamos o debate uma vez neste projeto, com razão.
- **Teste exaustivo por série**: greedy é a pior estratégia nos 2 datasets; mais
  avaliações de validação por série não viram teste (§2).
- **Mais tools de peso**: com 3 janelas, todas colapsam para ≈uniforme
  (conc. média 0.0028); a família já tem aliasing demais.

---

## 5. Como isso alimenta o paper

1. **Medição de ancoragem/viés de posição em agente de tool-use em produção**
   (462→223→10→2→1 dentro de uma família semântica; primeira ação 90/182 = o
   exemplo do prompt) — eco aplicado do BiasBusters, com trajetórias auditáveis.
2. **"Amortized dataset calibration" para combinação por série**: o experimento §2
   como ponte entre FFORMA (global) e agente (local) — com o resultado honesto de
   que bate ADE/FFORMA no NN5 e não fecha no ANP, e o porquê.
3. **Aliasing de estratégias como mecanismo de falsa exploração**: 78% das
   avaliações em `weighted`, 62% dos vencedores ≈ média — a "liberdade" do agente
   colapsa num único ponto do espaço de previsões sem feedback de identidade.
4. As correções P0–P3 são cada uma um braço de ablação limpo (antes/depois nas
   mesmas 182+111 séries).

---

## Fontes

- [BiasBusters — position bias in tool selection](https://arxiv.org/pdf/2605.18857)
- [How Many Tools Should an LLM Agent See?](https://arxiv.org/html/2605.24660v1)
- [TRAJECT-Bench — trajectory-aware tool-use benchmark](https://arxiv.org/pdf/2510.04550)
- [Tool-RAG / RAG-MCP — retrieval-based tool exposure](https://webscraft.org/blog/tool-rag-scho-robiti-koli-u-agenta-zabagato-instrumentiv?lang=en)
- [Single vs multi-agent under matched budgets (2026 synthesis)](https://medium.com/@mjgmario/single-agent-vs-multi-agent-systems-when-coordination-helps-hurts-and-pays-off-57735ee7916d)
- [Multi-agent LLMs 2026 overview](https://www.superannotate.com/blog/multi-agent-llms)
- [ExpeL — experience reuse across tasks](https://arxiv.org/abs/2308.10144); Agent Workflow Memory; Dynamic Cheatsheet (via survey [Efficient Agents](https://efficient-agents.github.io/))
- Ollama × gpt-oss issues: [#11781](https://github.com/ollama/ollama/issues/11781), [#11800](https://github.com/ollama/ollama/issues/11800), [#11867](https://github.com/ollama/ollama/issues/11867)
- Internos: `RELATORIO_TECNICO_COMBINACAO.md`, `orchestrator_react/ARQUITETURA.md`, `orchestrator_react/insights_trabalhos.md` (TSOrchestr §2 — calibração amortizada)
