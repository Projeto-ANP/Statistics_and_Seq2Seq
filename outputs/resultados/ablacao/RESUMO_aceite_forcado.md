# O agente é forçado a propor algo antes de aceitar uma semente já boa?

Fatos (código e dados) separados de estimativas. Números: `outputs/resultados/scripts/ablacao_aceite_forcado.py`, sobre os CSVs de `orchestrator_react_v5` (gpt-oss:20b, 680 séries) e `orchestrator_react_v5_qwen` (qwen3:30b-a3b, 680 séries). O Gemma não está nesta branch e não entrou.

## Passo 1 — existe aceite imediato?

**Fato: sim, existe, e o código não o restringe.**

- A resposta do agente é sempre `Thought / Action / Action Input`, mas o catálogo inclui a ação terminal `accept` (`registry.py:55`, `TERMINAL_ACTION = "accept"`). Formato: `Action: accept` e `Action Input: {"attempt_id": "aN", "confidence": 0.7, "justification": "..."}` (exemplo em `prompts.py:79`).
- O loop trata `accept` antes de qualquer ferramenta (`react_loop.py:280-301`): registra `agent_accepted_id`, `accept_confidence`, marca `stop_reason = "agent_accepted"` e encerra o ciclo.
- `_read_accept` (`react_loop.py:394-424`) só valida que o `attempt_id` exista no histórico, e o histórico já contém as sementes; sem `attempt_id`, aceita a melhor tentativa. **Não há exigência de iteração mínima, de ter chamado `evaluate_strategy` antes, nem de a tentativa ser do agente.** Ou seja, um `accept` na iteração 1 sobre a melhor semente é válido.
- O que o prompt diz (fato sobre o texto): o prompt de sistema lista `accept` entre as ferramentas (`prompts.py:49`) e descreve o passo 4 como "test another hypothesis, or accept the best attempt" (l.62-63). O prompt de cada turno diz "N iterations left. Respond with Thought / Action / Action Input." e só na última iteração menciona `accept` ("This is your LAST iteration. Use accept to take the best attempt.", `prompts.py:257-260`). Há também regras que orientam a propor: "If a SEEDED BASELINE is leading, the most direct improvement is usually the SAME method on a better pool" (l.104-107) e "Repeating a strategy already in the history wastes an iteration" (l.98).

Conclusão de código: **não é estrutural** — o aceite direto existe. Se o comportamento observado é de não aceitar cedo, ele vem do modelo/prompt, não de uma trava no loop.

## Passo 2 — estagnação e aceite explícito

**Fatos (`react_loop.py`, `config.py`):**

- **Estagnação:** `early_stop_patience = 4` propostas **consecutivas** sem melhora (`config.py:87`, `react_loop.py:320-334`). Só contam chamadas bem-sucedidas de `evaluate_strategy`; outras ferramentas (`weights_*`, `select_*`, `prune_redundant`) e chamadas com erro não contam nem zeram o contador. "Melhora" = ganho relativo do melhor score > `min_improvement = 1e-4` (`_improved`, l.380-386). Ao atingir 4, `stop_reason = "no improvement in 4 consecutive proposals"`.
- **Orçamento:** `max_iterations = 12`; se o `for` termina sem `break`, `stop_reason = "iteration_budget_exhausted"` (l.335-336).
- **Aceite explícito:** é uma ação distinta e ativa do agente (`Action: accept`); **não** é inferido da estagnação nem do orçamento.
- **Independente disso, a estratégia aplicada é sempre a melhor do histórico** (`react_loop.py:338-346`): se o agente aceita uma tentativa que não é a de menor score, o código aplica a melhor mesmo assim e marca `overridden`. Sem aceite (estagnação/orçamento), a justificativa vem do texto determinístico `_fallback_justification`.

## Passo 3 — iterações quando a estratégia final é uma semente

`iterações` = `loop.iterations_used`; `chamadas ao LLM` = passos da trajetória + respostas vazias + repetições por erro de LLM; `avaliações` = `n_evaluate_calls`. Formato: mín / mediana / média / máx.

**gpt-oss:20b**

| grupo | n | iterações | chamadas ao LLM | avaliações | tempo médio (s) |
|---|---|---|---|---|---|
| `origin=baseline` | 407 | 1 / 10 / 9,8 / 12 | 5 / 12 / 12,2 / 20 | 0 / 4 / 3,7 / 4 | 58,0 |
| `origin=agent` | 273 | 8 / 12 / 11,8 / 12 | 9 / 15 / 14,6 / 19 | 1 / 4 / 4,2 / 7 | 73,6 |
| todas | 680 | 1 / 11 / 10,6 / 12 | 5 / 13 / 13,2 / 20 | 0 / 4 / 3,9 / 7 | 64,2 |

- Séries `origin=baseline`: **406 de 407 (99,8%) usam mais de 1 iteração** (a exceção é uma série que terminou por erro de LLM). Motivo de encerramento: **317 estagnação (77,9%)**, **88 `accept`** (21,6%), 2 erro de LLM. Em 317 delas o agente fez ≥4 avaliações e não superou a semente.
- Aceite imediato: **0 de 680** séries do gpt-oss aceitaram na iteração 1. Distribuição do `accept` por iteração (336 aceites no total): iteração 9: 1, 10: 3, 11: 7, **12: 325**. Ou seja, 96,7% dos aceites do gpt-oss ocorrem na última iteração do orçamento.

**qwen3:30b-a3b (para comparação)**

| grupo | n | iterações | chamadas ao LLM | avaliações | tempo médio (s) |
|---|---|---|---|---|---|
| `origin=baseline` | 337 | 1 / 7 / 7,2 / 12 | 1 / 7 / 7,3 / 12 | 0 / 4 / 2,9 / 4 | 134,2 |
| `origin=agent` | 343 | 2 / 10 / 9,3 / 12 | 2 / 10 / 9,3 / 14 | 1 / 4 / 3,9 / 7 | 169,7 |

- `origin=baseline`: 335/337 (99,4%) usam >1 iteração; 173 estagnação, 164 `accept`. O Qwen3 aceita em várias iterações (461 aceites: iteração 1: 2, 2: 7, 3: 35, … 12: 111), então o caminho de aceite antecipado é usado na prática, mas pouco: 2 aceites na iteração 1.

## Passo 4 — custo evitável (**estimativa**, não fato)

**Medido (gpt-oss):** as séries `origin=baseline` consomem **6,56 h de 12,13 h (54,0%)** do tempo total do ciclo e **4.962 de 8.943 chamadas ao LLM (55,5%)**. Média de 12,2 chamadas e 58,0 s por série nesse grupo (contra 64,2 s na média geral). Por dataset (relacionado à tabela de custo do `item_17`, que dá o tempo médio por dataset):

| dataset | séries baseline | tempo médio nelas (s) | tempo médio do dataset (s) | % do tempo do dataset gasto nelas |
|---|---|---|---|---|
| ANP_MONTHLY | 105/182 | 62,5 | 74,8 | 48,2% |
| NN5_WEEKLY_DATASET | 61/111 | 61,0 | 67,2 | 49,9% |
| M4_WEEKLY_DATASET | 224/359 | 54,5 | 58,6 | 58,0% |
| ETTH1 | 4/7 | 45,0 | 41,3 | 62,3% |
| ETTH2 | 5/7 | 40,3 | 41,6 | 69,4% |
| ETTM1 | 4/7 | 40,1 | 39,1 | 58,6% |
| ETTM2 | 4/7 | 145,0 | 102,5 | 80,8% |

Para o Qwen3: 12,56 h de 28,73 h (43,7%) e 2.447 de 5.628 chamadas (43,5%).

**Estimativa de teto:** se cada série `origin=baseline` tivesse encerrado após **1** chamada (4,88 s por chamada em média no gpt-oss), o tempo evitável seria ≈ **6,0 h (≈ 49,5% do total)** e ≈ **4.555 chamadas**; para o Qwen3, ≈ 10,8 h (≈ 37,7%) e ≈ 2.110 chamadas.

**Ressalvas:**
1. É um **teto**: pressupõe saber de antemão que a semente é a resposta. Em 273 séries (40,1%) a proposta do agente venceu; encerrar todas cedo perderia esses casos. Só seria evitável de fato o custo em séries identificáveis como `origin=baseline` sem rodar o ciclo.
2. `elapsed_s` cobre só o ciclo ReAct (`react_loop.py:157`); ingestão, diagnóstico e semeadura não estão nesse tempo (nem no do `item_17`), então os percentuais são sobre o tempo do ciclo, a mesma base da tabela de custo.
3. Não há dado de tokens; "chamadas" contam respostas do LLM, não seu tamanho.
