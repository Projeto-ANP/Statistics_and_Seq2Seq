# JEV/ — ideias e implementações "System One" para o combinador

Pasta de exploração das alternativas ao LLM gerador (gpt-oss:20b) na Fase 3 do
orquestrador, inspiradas na família **JEV** (TypeSafe) — modelos de *decisão*
(classificadores calibrados) em vez de geradores de texto. Contexto e medições
em `ENTENDIMENTO_ORCHESTRATOR_REACT.md` §2-§5 (resumo: transferência das
propostas do agente 10% no NN5; decisão final = gargalo).

## Resultados medidos — 2ª onda: 7 datasets × 3 braços + comparação

sMAPE médio por dataset (2026-09-25, servidor):

| dataset | english compacto | ml8192 rico | sem sementes | piso sementes | gpt-oss v5 |
|---|---|---|---|---|---|
| ETTM1 | 0.4592 | — | 0.4980 | 0.4604 | 0.4594 |
| ETTM2 | 0.2417 | **0.2043** | 0.2072 | 0.2030 | **0.1610** |
| ETTH1 | 0.2472 | — | 0.2485 | 0.2262 | 0.2276 |
| ETTH2 | 0.1800 | — | 0.1910 | 0.1799 | 0.1794 |
| ANP_MONTHLY | 0.2208 | — | 0.2222 | 0.2212 | 0.2209 |
| NN5_WEEKLY | 0.1188 | **0.1172** | 0.1296 | 0.1156 | 0.1177 |
| M4_WEEKLY | 0.0979 | — | 0.1036 | 0.0910 | 0.0923 |

Leituras:

1. **A compressão do estado MACHUCAVA — pergunta respondida com medição.**
   Estado rico (multilingual 8192) recupera quase tudo: ETTM2 0.2417 → 0.2043
   (≈ piso 0.2030); NN5 0.1188 → 0.1172 (≈ gpt-oss 0.1177). Mecanismo visível
   por série: o estado comprimido levou o classificador a apostas `best_single`
   (ETTM2 série 5: 0.7310); com estado rico ele ficou na semente (0.4531).
2. **Mesmo com estado rico, o LAYA zero-shot não bate o piso em nenhum
   dataset** (empata no ETTM2, perde por ~0.001 no NN5). Confirma: o gargalo é
   a função-objetivo (argmin de 3 janelas), não o tomador de decisão.
3. **Sem sementes é sempre pior** (7/7 datasets): o piso é proteção real.
4. **gpt-oss é o único que bate o piso**, e só no ETTM2 (0.1610, via série 5).
   Nos demais, ≈ piso ± ruído. Em ETTM1/ETTH1/ANP **ninguém** agrega nada.
5. **Consequência para o fine-tune**: usar estado RICO (multilingual 8192) como
   input do treino; o rótulo continua sendo o desfecho no teste (§fine-tune).

## Resultados medidos — v0 (zero-shot, english, menu de 16 opções)
Executado no servidor (GPU), 2026-09-25, `orchestrator_laya_laya_v0/`.

| dataset | LAYA v0 | piso sementes (determinístico) | agente gpt-oss (v5) | ADE | FFORMA |
|---|---|---|---|---|---|
| NN5 (111 séries) | 0.1188 | 0.1156 | 0.1177 | 0.1178 | 0.1197 |
| ETTM2 (7 séries) | 0.2417 | 0.2030 | 0.1608 | 0.1872 | 0.1654 |

Custo: NN5 em **17,3 s** (vs ~24 min do LLM) — ~1000× mais barato/rápido.

Leituras (verificadas com `diagnostics/analyze_run.py`):

1. **O classificador zero-shot tem o MESMO comportamento anti-preditivo do
   LLM**: intervém em 45/111 séries (LLM: 50/111), final pior que a melhor
   semente no teste em 94/111 (LLM: 95/111), delta médio vs oráculo +0.0128
   (LLM: +0.0117). **Conclusão forte: o gargalo é a função-objetivo (argmin
   sobre 3 janelas de validação), não o tomador de decisão.** Trocar o LLM
   por um classificador não conserta; consertar o objetivo sim.
2. **As probabilidades do LAYA não separam escolha boa de ruim zero-shot**:
   fração que bate o piso no teste por faixa de `probability_of_chosen` —
   0-0.50: 9,2% | 0.50-0.70: 31,9% | 0.70-0.85: 16,1% | 0.85-1.00: 19,4%
   (não-monotônico). No geral só 14,5% das estratégias avaliadas batem o piso
   no teste (LLM: 10%). Não usar a confiança como gate sem fine-tune.
3. **O menu importa**: o pior erro do LAYA (ETTM2 série 5: 0.7310) veio de
   `best_single` sobre um modelo que venceu as 3 janelas por sorte; o gpt-oss
   escapou da mesma série (0.2251) com média de pool PODADO — opção que o menu
   v0 não tinha. Adicionar pools de `prune_redundant` ao menu é o próximo fix
   barato.
4. **Caminho confirmado**: fine-tune do LAYA/BERT sobre os desfechos rotulados
   (456 propostas do LLM + 304 do LAYA, todas com validação E teste) — é a
   Alternativa 1/5. O zero-shot serviu para medir a régua e provar que o
   problema não é o modelo.

## Fine-tune do LAYA — receita (o caminho que sobrou)

O zero-shot mediu perto do acaso na nossa tarefa (14,5% de acerto contra o piso
no teste). O próprio repo diz: *"treat Laya as a fast base to specialise, not as
a zero-shot decision engine"* — fine-tuned 0.766 vs 0.362 zero-shot no benchmark
deles; no caso browser-agent: top-1 entre ~45 candidatos 0.10 → 0.66, sucesso
real 0% → 62%.

**Como funciona o treino** (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
no repo do laya): dataset de decisões (state + questions + resposta correta) →
treino com **RLCD** (recompensa = regra de pontuação própria, policy gradient
estilo GRPO) → ajuste de temperaturas por tipo de pergunta → avaliação → push
pro Hub. O pacote pip já expõe as peças (`laya.proper_reward`, `laya.ece_score`,
`laya.LayaEvaluator`, `laya.td_lambda_targets`).

**Tempo**: referência deles = **4-5 h para 30k perguntas × 4 épocas em 2×T4**.
Nosso dataset (~760 decisões: 456 do LLM + 304 do LAYA) é ~40× menor → **minutos
a ~1 h em T4** (Kaggle 2×T4 é gratuito); CPU funciona mas multiplica por ~5-10×.

**O rótulo é a decisão mais importante** — três opções:
1. **Imitação**: rotular com a ação que o agente tomou. Aprende o comportamento
   atual (ruim: 10-14% de transferência). Não fazer.
2. **Desfecho (o certo)**: rotular com a ação cuja estratégia VENCEU na janela
   de teste — ou `accept` quando nenhuma proposta bate a semente. Ensina
   exatamente o que queremos prever; é a Alternativa 1 do mapa (gate aprendido).
3. **Híbrido**: imitação como pré-treino, desfecho como fine-tune.

**Pipeline no nosso repo**: (1) `JEV/build_finetune_dataset.py` exporta
`(state, questions, label)` dos runs existentes via replay determinístico
(o `analyze_run.py` já faz o replay; falta o export); (2) rodar o notebook
(Kaggle 2×T4 ou a GPU do servidor); (3) plugar o checkpoint local direto no
loop — o `laya_loop.LayaAgent` já aceita caminho local:
`run_laya.py --checkpoint ./JEV/models/laya_nn5_finetuned` (o `laya.load` aceita
diretório local com `model.safetensors` + `rl_agent_config.json`).

## Auditoria: o que difere entre o run LAYA e o run padrão (gpt-oss)

A troca LLM→classificador NÃO é 100% limpa. O contrato de avaliação é o mesmo;
4 adaptações foram necessárias (e precisam ser reportadas no paper):

**IDÊNTICO (sem adaptação):**
- Fase 0 (ingestão), Fase 2 (sementes mean/median/dba + pools estáveis — MESMOS
  valores), Fase 4 (aplicação à janela de teste);
- Contrato de avaliação: `state.evaluate` com backtest ANINHADO nas 3 janelas +
  princípio 5 (aplicada = melhor do histórico);
- Orçamento: 12 iterações; métricas; baseline dos logs;
- Cartões mostrados ao tomador de decisão: `_slim_series_card` /
  `_slim_pool_card` / `attempt.brief` — as MESMAS funções do prompt do LLM.

**ADAPTADO (porque o classificador não gera texto):**

| aspecto | run padrão (gpt-oss) | run LAYA |
|---|---|---|
| espaço de ação | 24 ferramentas, argumentos livres gerados | MENU fixo de ~13-16 ações concretas (mean/median/trimmed × 4 pools + best_single top-3 + accept); argumentos definidos pelo menu |
| ferramentas de pesos | inverse_error, softmax, trend, ols, feature_based, pooled | NENHUMA no menu v0 |
| semente pooled meta-model | semeada quando ≥20 séries | NÃO anexada → semente ausente (piso difere levemente em NN5: 0.1156 vs 0.1157) |
| contexto cross-series | DATASET CARD (prior LOO) + diagnosis + lista de handles + scratchpad no prompt | ausentes do estado do classificador (só cards + histórico top-6) |
| formato do prompt | system prompt (papel + catálogo + regras) + turn prompt | estado compacto/rico + pergunta com critérios por opção |
| saída | CSV de 58 colunas + artifacts JSON | CSV enxuto (13 core + poucas colunas) |
| early-stop | patience de propostas sem melhora | stale-stop de escolhas repetidas |

**Consequência para o paper**: dizer "só trocamos o tomador de decisão" seria
impreciso. O honesto é: "mesmo protocolo de avaliação; o classificador decide
sobre um menu enumerado de estratégias, sem as ferramentas de pesos e sem o
contexto cross-series do agente LLM". Se quisermos a troca LIMPA, falta: adicionar
o DATASET CARD + handles + semente pooled ao run LAYA (paridade total de contexto).

## O que está implementado

| arquivo | o que faz | onde roda |
|---|---|---|
| `laya_loop.py` | loop de decisão por classificador: estado compacto + pergunta `choice` com ações concretas (menu) → escolha → `state.evaluate` (mesmo contrato de backtest do ReAct) | servidor |
| `run_laya.py` | runner por dataset (CLI estilo `run_tsf_batch.py`); CSV em `resultados/orchestrator_laya_<version>/`; log com `this run` + **`baseline`** (piso das sementes) + baselines externos | servidor |
| `run_laya_batch.sh` | batelada: `nohup bash JEV/run_laya_batch.sh > logs/laya_v0_batch.log 2>&1 &` | servidor |

Pré-requisito no servidor: `pip install laya` (uma vez; baixa ~800 MB de
checkpoint no primeiro uso).

## O que é JEV e por que não dá para rodar

JEV (`jev-1.13.0`) é o primeiro "System One model" da TypeSafe: modelo de
decisão não-autoregressivo, treinado com RLCD (probabilidades calibradas por
construção), que responde perguntas tipadas (`choice`/`score`/`noul`) sobre um
`state` — **nunca gera texto**. É privado: só API (typesafe.ai / OpenRouter),
sem pesos abertos. Por isso usamos o **LAYA** (`convaiinnovations/laya`,
Apache-2.0), que é a réplica open do mesmo paradigma e até expõe a mesma API
`POST /v1/systemone`.

| | LLM gerador (gpt-oss:20b) | System One (JEV/LAYA) |
|---|---|---|
| paradigma | autoregressivo, token a token | encoder + head de decisão, 1 forward pass |
| saída | texto livre (parse, alucinação) | escolha tipada + probabilidade calibrada |
| latência | segundos | ~33 ms GPU / 200-460 ms CPU |
| janela | 8k+ | LAYA english 512 / multilingual 1024→8192; JEV 64k |

## Limites do LAYA relevantes para nós

- **Checkpoints**: english (ModernBERT-large 421M, **512 tokens**) é o mais
  acurado em inglês; multilingual (mmBERT-base 322M, 1024 default, `max_len=8192`)
  lê mais mas é mais fraco em inglês. Nossos prompts medidos (system ~2k tokens,
  turno ~1.3-2.5k) **não cabem no english** — por isso o `laya_loop.py` monta um
  estado próprio compacto (cards + histórico ≈ 1.4k chars ≈ 350 tokens) e não
  envia o system prompt (as opções levam critérios na própria pergunta).
- **Zero-shot é fraco no domínio**: no benchmark de typed-decisions do próprio
  repo, o base faz 0.362 e o fine-tuned 0.766. Zero-shot serve para o primeiro
  número; **fine-tuning é o caminho para resultado** (notebook oficial em 2×T4
  Kaggle; nossos dados cabem folgado).
- **Não gera argumentos**: o menu tem que enumerar ações concretas. O loop v0
  oferece {mean, median, trimmed_mean} × 4 pools + best_single × 3 + accept.

## Mapa das ideias (o que ainda não está implementado)

### Alternativa 1 — GATE aprendido treinado nos nossos logs (maior alavancagem)

**Ideia.** Em vez de o agente decidir se a proposta dele substitui a semente,
um classificador treinado decide: features já computadas (verdict, margem,
prior do dataset, perfil da série, score de validação) → alvo = "a proposta
transferiu para o teste?" (temos **456 propostas rotuladas** nos artifacts do
NN5 v5). Substitui/complementa o gate estatístico H1 de
`ENTENDIMENTO_ORCHESTRATOR_REACT.md` §5.

**Como.** (1) Extrair dataset: `diagnostics/analyze_run.py` já replaya as
propostas; adicionar a ele um export de features+labels por proposta (série,
spec, score, verdict, margem, prior, transferiu_s/n). (2) Baseline: regressão
logística nas features (treino instantâneo) — se separar transferidas de
não-transferidas, subir para (3) SetFit ou fine-tune BERT/ModernBERT (456
exemplos: minutos em CPU / ~20 min em GPU). (4) Integrar como gate na Fase 4:
só aplica proposta `agent` se P(transferir) > limiar.

**Custo:** dados ~1 h (determinístico), treino minutos~1 h, GPU opcional.
**Risco:** 456 exemplos é pouco; features importam mais que arquitetura.

### Alternativa 2 — LAYA zero-shot como agente (implementada: ver acima)

**Medição que importa:** além do sMAPE por dataset, avaliar se as
probabilidades do LAYA separam boas de más escolhas (o gpt-oss dava 0.9
constante). O `trace` do `laya_loop.py` grava `confidence` por turno para isso.

### Alternativa 3 — Seletor de menu (o V5 antigo, com classificador)

**Ideia.** O V5 anterior do repo (`orchestrator_v5.log`) já era um *selector
sobre menu* (`chosen_method: single_best_val / trimmed_mean_20`) com LLM. Trocar
o LLM desse desenho por LAYA/BERT treinado: dado o card da série + pool,
escolher 1 de N estratégias canônicas. É o `laya_loop.py` com `max_iterations=1`
e menu ampliado (inclui pesos), possivelmente fine-tuned com os desfechos dos
nossos runs.

### Alternativa 4 — Routing explorar-vs-aceitar (RouteLLM style)

**Ideia.** Classificador barato decide POR SÉRIE: "vale abrir o loop do LLM ou
fica na semente?" — análogo ao `calibration_gate`, mas aprendido
(cross-series), não por limiar de Kendall tau. Precedente open: RouteLLM
(lmsys). Encaixa como pré-fase: LAYA/bert → explore com gpt-oss só onde o
verdict diz que há o que descobrir.

### Alternativa 5 — Fine-tune do LAYA no nosso domínio

**Ideia.** Usar os desfechos dos runs (ações + scores) para fine-tunar o LAYA
(mesmo pipeline do notebook oficial: construir dataset de decisões, treinar,
calibrar temperaturas, avaliar). Pode ser o classificador das alternativas 1/2/3.

## Ordem sugerida de execução

1. Rodar `run_laya_batch.sh` (v0, zero-shot) → número base do classificador
   por dataset + calibração do `trace`.
2. Export do dataset de 456 propostas rotuladas (Alt 1) + regressão logística
   baseline do gate.
3. Se (2) mostrar sinal: fine-tune BERT/SetFit ou LAYA (Alt 5) e integrar o
   gate na Fase 4 do orquestrador principal.
