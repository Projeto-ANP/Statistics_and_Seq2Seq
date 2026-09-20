# Plano de execução v5 — arquitetura para o topo de ANP e NN5

Documento de trabalho: tudo o que foi entendido até aqui, o que pretendo fazer,
em que ordem, com critério de sucesso e **plano B/C para cada passo**. Números
marcados 📊 foram medidos nos nossos dados (nunca citados de literatura sem teste).

Data: 2026-07-30. Estado do código: **460 testes passando**.

> **STATUS: P0.5a, P0.5b, P1, P2, P3 implementados e medidos.** Resultados reais
> em `ARQUITETURA.md` §14. Resumo: objetivo FFORMA + sMAPE reproduz 0.2160 no ANP
> (passa o FFORMA real); piso ANP 0.2203 -> 0.2190, NN5 neutro; `prior_blend`
> chega a 0.2145 no ANP mas alpha nao pode ser escolhido honestamente, logo e
> opt-in. Falta apenas P4 (ops/verificacoes) e P5 (rodadas no servidor).

---

## 1. O que sabemos (fatos medidos que orientam tudo)

### 1.1 Placar atual por dataset 📊

| método | ANP (182 séries) | NN5 (111 séries) |
|---|---|---|
| FFORMA (baseline externa) | **0.2166** | 0.1197 |
| ADE | 0.2177 | 0.1178 |
| melhor regra fixa | 0.2181 (islands_median) | **0.1149** (stable5_mean) |
| median / mean | 0.2194 / 0.2206 | 0.1201 / 0.1199 |
| **nosso agente (última rodada)** | 0.2223 | 0.1207 |
| dba | 0.2252 | 0.1226 |
| **objetivo-FFORMA pooled (exp. isolado, LOSO)** | **0.2159** ← bate tudo | 0.1197 |
| prior de dataset + shortlist M=3–4 (exp.) | 0.2191 | **0.1166** (bate ADE e FFORMA) |

### 1.2 Os cinco fatos estruturais 📊

1. **Nada transfere entre datasets.** Ranking de 16 estratégias: Spearman +0.12
   entre ANP e NN5. `stable5_mean` é 1º no NN5 e 11º no ANP. Já nos queimamos uma
   vez calibrando sementes no NN5 — **nenhuma decisão nova pode ser calibrada num
   dataset só**.
2. **3 janelas de validação não sustentam otimização livre por série.** Três
   evidências independentes: `weights_ols` colapsa; score de validação era
   anticorrelacionado com teste antes do `nested_selection`; busca gulosa direta
   é a pior estratégia nos DOIS datasets.
3. **Otimizar o erro combinado funciona pooled entre séries e falha por série.**
   O objetivo do FFORMA (mesma coisa que a busca gulosa, mas com 182 séries de
   amostra em vez de 24 pontos) é o melhor mecanismo no ANP. Essa é a fronteira
   exata do *forecast combination puzzle* nos nossos dados.
4. **O agente usa 4–5 tools de ~10 úteis, ancorado no exemplo do prompt.**
   Família de pesos: 462→223→10→2→1 usos por posição/familiaridade. 591/756
   avaliações são `weighted`, e 62% dos vencedores "weighted" são numericamente a
   média do pool (aliasing — falsa exploração).
5. **Mais vitórias do agente na validação ≠ melhor teste.** v4 venceu o piso em
   mais séries que v3 (99 vs 84) e ficou igual/pior no teste. O valor do agente
   está em arbitrar entre campeões fortes, não em "explorar mais".

### 1.3 A causa-raiz do resultado do passo anterior (importante) 📊

Acabei de medir o piso com a semente do objetivo-FFORMA e deu **0.2213 (pior)**,
enquanto o experimento isolado dava **0.2159**. Causa identificada, ainda **não
corrigida** (a edição foi interrompida): em produção, `build_meta_row` alimenta o
objetivo com **RMSE bruto**; no experimento eu usei **sMAPE**. O gradiente do
objetivo-FFORMA **soma contribuições entre séries** — com RMSE bruto, as séries de
escala grande (ANP varia ordens de magnitude) dominam o gradiente e o meta-modelo
aprende escala, não competência. O FFORMA original usa OWA (livre de escala) pelo
mesmíssimo motivo.

---

## 2. Estado exato do código (o que esta sessão já implementou)

| item | estado |
|---|---|
| Objetivo FFORMA no `meta_model.py` (booster multi-classe, gradiente custom, LOSO) | ✅ implementado + testado |
| Score por nome com dois tipos (`error`/`margin`), subset-softmax fold-safe | ✅ implementado + testado |
| Guarda que inutilizava a tool (rejeitava pool podado) | ✅ removida, com testes da propriedade que a substitui |
| Semente `weighted(pooled)` na Fase 2 (`seed_pooled_meta_model=True`) | ✅ implementada + testada |
| catch22 nas features (4→26) | ✅ (efeito ≈ 0 medido; mantido por fidelidade ao FFORMA) |
| `pooled_meta_model_objective: "fforma"` default no config | ✅ |
| **Correção do métrica de contribuição (RMSE→sMAPE)** | ⛔ **PENDENTE — próxima ação, edição foi interrompida** |
| Dataset card no prompt (P1) | ❌ não começado |
| Anti-aliasing nas observações (P2) | ❌ não começado |
| Desancorar exemplo do prompt (P3) | ❌ não começado |
| Mistério: pooled withheld em 17 séries do v4 | ❌ a verificar (suspeita: artefato do resume 0–80/81–181) |

---

## 3. Alvo global e critérios de sucesso

**Alvo do braço determinístico (testável aqui, sem LLM):**

| dataset | piso hoje | meta | referência a bater |
|---|---|---|---|
| ANP | 0.2203 | **≤ 0.216** | FFORMA 0.2166 |
| NN5 | 0.1154 | **manter ≤ 0.1155** (não regredir) | melhor fixa 0.1149 |

**Alvo do braço com agente (só no servidor, mas o "leque" fica pronto):** agente ≥
piso nos dois datasets; diversidade de tools ↑ (entropia da família de pesos ≥ 1.5
bits, primeira ação `weights_inverse_error` < 50/182); unparsed < 20.

Princípio de decisão para TODO passo: **medir nos dois datasets antes de aceitar
qualquer default.** Um ganho num dataset com regressão no outro exige mecanismo
adaptativo (prior LOSO) ou vira ablação, nunca default cego.

---

## 4. Passo a passo

### P0.5a — Corrigir a métrica de contribuição do meta-modelo (RMSE → sMAPE)

**O quê:** `build_meta_row(metric="smape")` como default, com docstring explicando
o porquê (gradiente soma entre séries ⇒ precisa ser livre de escala).
**Por quê:** é a diferença medida entre 0.2224 (produção, RMSE) e 0.2159
(experimento, sMAPE) no ANP.
**Medição:** rerodar o `p0_check` nos dois datasets.
**Sucesso:** pooled-alone ANP ≤ 0.217; piso ANP com semente < piso sem; NN5 sem
regressão (a semente só vence o piso quando a validação dela ganha — o argmin
protege).

- **Plano B (se sMAPE não reproduzir 0.2159 em produção):** normalizar a
  contribuição por linha (erro da série ÷ mediana da linha) em vez de trocar a
  métrica — mesmo efeito de escala, sem a instabilidade do sMAPE perto de zero
  (lembrar: 5 séries do ANP têm zeros na janela).
- **Plano C:** implementar OWA (como o FFORMA original): erro relativo ao naïve
  seasonal por série. Mais fiel, um pouco mais de código.
- **Plano D (se nada disso mover o ANP):** `seed_pooled_meta_model=False` como
  default, tool continua no catálogo, e o piso do ANP passa a depender do P1.

### P0.5b — Regressão NN5 do piso com a nova semente

**O quê:** mesma medição no NN5.
**Risco específico:** no NN5 o objetivo-FFORMA é levemente pior que o per_model
(0.1197 vs 0.1188) e ambos perdem do piso stable (0.1149). A semente nova só entra
no argmin — mas ela pode ganhar a validação por ruído e perder no teste.
**Sucesso:** piso NN5 ∈ [0.1149, 0.1160].

- **Plano B (se a semente poluir o NN5):** gate adaptativo por dataset — o
  pré-pass já computa tudo; semear pooled **somente se** o prior LOSO de validação
  da estratégia pooled estiver no top-3 do dataset. Determinístico, honesto
  (validação-only), e resolve o não-transfere sem calibrar em dataset nenhum.
- **Plano C:** semear as DUAS variantes (fforma e per_model) e deixar o argmin da
  validação escolher — custo: +1 avaliação por série.

### P1 — Dataset card no prompt (o conhecimento cross-series para o agente)

**O quê:** o pré-pass (que já carrega todas as séries) passa a computar também o
**prior LOSO por estratégia-semente** (sMAPE médio de validação nas *outras*
séries) e injeta no turn prompt:

```
DATASET CARD (validation-only, computed across the other N-1 series):
  seed families ranked: fforma_pooled 0.226, stable7_median 0.230, full_median 0.231...
  models that most often rank top-3 on validation: ETS, ARIMA, THETA
  caution: weighted proposals were numerically the pool MEAN in 62% of series here
```

Recomendação, não restrição (mantém a liberdade do agente — requisito seu).
Também gravar o card no artifact JSON e uma coluna resumo no CSV.
**Por quê:** experimento do prior mostrou que shortlist M=3–4 + escolha local ≥
escolha sem prior nos dois datasets; ExpeL/AWM (literatura) fazem isso com memória
livre — a nossa é determinística e auditável.
**Medição local (sem LLM):** teste unitário do card (LOSO exclui a própria série;
zero acesso a teste). Medição real: rodada no servidor (entropia de tools, sMAPE).
**Sucesso (servidor):** ANP com agente ≤ 0.219 e diversidade ↑.

- **Plano B (agente ignora o card):** transformar recomendação em *shortlist
  branda* — as estratégias fora do top-M do prior continuam disponíveis mas o
  prompt diz explicitamente "estas N foram fracas neste dataset nas outras
  séries". Se ainda ignorar: shortlist dura via `withheld_tools` como **ablação**
  (não default), medindo o custo da liberdade.
- **Plano C (card não ajuda nem atrapalha):** mover a decisão para o
  determinístico — piso adaptativo por prior LOSO (Plano B do P0.5b) já entrega o
  ganho sem depender do LLM ler o card.

### P2 — Anti-aliasing nas observações do `evaluate_strategy`

**O quê:** quando a proposta produz previsões numericamente idênticas a uma
tentativa existente (já temos `_numerically_identical` para os twins), a
observação diz: `"numerically identical to a3 (the pool mean) — adds nothing new"`.
**Por quê:** 591/756 avaliações são `weighted` e 62% dos vencedores ≈ média — o
agente re-testa o mesmo ponto sem saber.
**Medição local:** testes (proposta ≈ média → aviso presente; proposta distinta →
ausente). Servidor: nº de tentativas numericamente distintas por série ↑.

- **Plano B (agente continua repetindo mesmo avisado):** o aviso vira contagem no
  early-stop — proposta numericamente idêntica conta como "sem melhora" para a
  paciência (hoje conta como avaliação normal). Barato e coerente.
- **Plano C:** dedup duro — tentativa idêntica nem entra no histórico, observação
  devolve a existente (`already_tested` estendido a igualdade numérica).

### P3 — Desancorar o exemplo do system prompt

**O quê:** "A TYPICAL SEQUENCE" deixa de citar `weights_inverse_error` e passa a
mostrar `weights_<method>` com a lista dos métodos e uma frase sobre o que cada
família lê (erro médio / tendência do erro / padrões cross-series). Regra nova:
"se um weights_* devolve conc≈0, outro weights_* no MESMO pool dará a mesma
previsão — mude o pool ou a família".
**Medição local:** asserts de texto. Servidor: primeira ação e entropia.

- **Plano B:** exemplos rotacionados por série (seed do RNG da série escolhe qual
  método aparece no exemplo) — elimina a âncora por construção.
- **Plano C:** reduzir a família de pesos exposta a UMA tool `make_weights(method=...)`
  com enum na assinatura (menos entradas no catálogo, alternativas visíveis no
  ponto de escolha). Mais invasivo; só se B falhar.

### P4 — Verificações e ops (rápidos)

1. **Mistério dos 17 withheld** no v4: reproduzir o resume e confirmar que é
   artefato de `todo=81..181` (pré-pass menor). Se for: documentar; se não:
   investigar de verdade.
2. **Ollama no servidor:** upgrade + testar `reasoning: low` para gpt-oss (issues
   conhecidas #11781/#11800/#11867). NUNCA `format=json`.
3. Rodada do **diagnosticador no ANP** (já pronta, só rodar).

- **Plano B (unparsed continuar alto):** reasoning low; se persistir, trocar o
  combinator para `qwen3:14b` numa rodada A/B — formato de 3 linhas é
  model-agnostic de propósito.

### P5 — Rodadas finais no servidor (você roda; eu deixo tudo pronto)

Ordem: (1) baseline determinístico v5 nos dois datasets — confirma os pisos desta
máquina; (2) agente v5 ANP; (3) agente v5 NN5; (4) diagnosticador ANP; (5) MCM
com `orchestrator_react_v5` nas comparações.

**Sucesso final:** MCM mostrando v5 ≥ FFORMA no ANP (mean-difference, Wilcoxon) e
≥ ADE no NN5, com `provenance_ok` 100% e diversidade de tools documentada.

- **Plano B (agente < piso em algum dataset):** reportar o braço determinístico
  como resultado principal (ele já é o nosso melhor número) e o agente como
  camada de arbitragem + instrumento de auditoria — narrativa honesta que os
  dados desta conversa já sustentam.
- **Plano C (ANP continuar < FFORMA mesmo no piso):** o P0.5a isolado já bateu
  FFORMA em LOSO honesto (0.2159 < 0.2166). Se produção não reproduzir, a
  diferença é bug nosso, não ciência — cavar até a paridade experimento/produção
  (mesma disciplina do check de alinhamento .tsf).

---

## 5. Riscos transversais

| risco | mitigação |
|---|---|
| Calibrar de novo num dataset só | toda mudança medida nos DOIS; ganhos assimétricos → mecanismo adaptativo (prior LOSO) ou ablação |
| sMAPE instável com zeros (5 séries ANP) | `test_has_zero_actual` já existe; plano B do P0.5a (normalização por linha) evita sMAPE no treino se necessário |
| Dependência entre séries via pooled/card | já documentado: validação-only, LOSO, mesma classe de dependência do ADE/FFORMA — comparação continua justa |
| Semente nova bagunça ids de attempts nos testes | lição das sessões passadas: tests referenciam por spec/lookup, não por id literal |
| Context/tempo de rodada no servidor | pré-pass é 1× por dataset (barato); card adiciona ~0 custo — reusa o que o pré-pass já computa |

## 6. O que NÃO fazer (descartado com evidência) 📊

- Peer multi-agent/debate (single ≥ multi sob orçamento pareado; já abandonamos o
  debate uma vez).
- Otimização livre da validação por série (greedy: pior nos 2 datasets).
- Mais tools de peso por série (aliasing: quase tudo ≈ média com 3 janelas).
- Teste exaustivo por série (o lugar do teste-tudo é o pré-pass por dataset).
- `format=json` no gpt-oss (issue conhecida: resposta vazia).

## 7. Sequência imediata (quando você aprovar)

1. P0.5a (edição pendente do `metric="smape"` + docstring) → suite → medição ANP.
2. P0.5b medição NN5 → decisão default vs plano B (gate adaptativo).
3. P2 (anti-alias) — pequeno, independente.
4. P3 (prompt) — pequeno, independente.
5. P1 (dataset card) — maior; por último porque depende dos priors do pré-pass já
   estabilizados pelos passos 1–2.
6. Atualizar `ARQUITETURA.md` (§13 objetivo FFORMA + métrica; §14 card) e este
   plano com os números reais de cada medição.
