# JEV/ — ideias e implementações "System One" para o combinador

Pasta de exploração das alternativas ao LLM gerador (gpt-oss:20b) na Fase 3 do
orquestrador, inspiradas na família **JEV** (TypeSafe) — modelos de *decisão*
(classificadores calibrados) em vez de geradores de texto. Contexto e medições
em `ENTENDIMENTO_ORCHESTRATOR_REACT.md` §2-§5 (resumo: transferência das
propostas do agente 10% no NN5; decisão final = gargalo).

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
