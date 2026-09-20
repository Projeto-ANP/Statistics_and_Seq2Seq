# Frequência do truncamento em `_compact` e aplicação da correção

Script: `outputs/resultados/scripts/truncamento_compact.py` (dados brutos: `truncamento_chamadas_antes.csv`, `truncamento_chamadas_depois.csv`, `truncamento_replay_*.csv`, nesta pasta).

## Método

- Amostra: **103 séries** de `orchestrator_react_v5` (25 ANP_MONTHLY, 25 NN5_WEEKLY_DATASET, 25 M4_WEEKLY_DATASET sorteadas com seed 0; as 7 séries de cada ETTH1/ETTH2/ETTM1/ETTM2).
- Cada série: estado real reconstruído (`ingest.load_series` + `pool.run_phase2`), `series_card`/`pool_card`/`diagnosis`/`strategy_prior` lidos do artefato gravado, e `run_react_loop` reexecutado com `ScriptedLLM` repetindo a trajetória gravada (Thought/Action/Action Input). Assim o `build_turn_prompt` real roda com scratchpad, histórico e observações reais. **Nenhum LLM é chamado.**
- Um wrapper em `prompts._compact` registra toda chamada (tipo pelo ponto de chamada, limite, tamanho do JSON completo) e, quando o limite é excedido, avalia se o texto cortado é JSON válido (`json.loads`).
- Total: **20581 chamadas** a `_compact` (uma por turno de prompt reconstruído).

## Ressalvas da reconstrução

- O meta-modelo pooled não é treinado (exige o dataset inteiro), então a semente `a10` não existe no estado reconstruído; `combine_dba` usa o código atual (o v5 gravado usava a versão com o bug de fallback para a média).
- Em 38 de 103 séries o replay tem menos passos que o gravado (diferença máxima 3); 54 das 103 terminam em `llm_error` (o roteiro acabou) (o roteiro acabou); o histórico/handles dessas séries é um subconjunto do real. As chamadas de `series_card`, `pool_card`, `dataset_card` e `diagnosis` não dependem disso.
- O cartão de dataset (`dataset_card`) reflete `strategy_prior` gravado.
- `summarize_observation` (limite 160, fallback) **nunca foi chamado** na amostra, então não há dado sobre ele.

## Passo 1 — comportamento antigo (`text[:limit] + " ...[truncated]"`)

Por tipo de chamada:

| tipo | limite | chamadas | maior JSON (chars) | excedem o limite | corte com JSON inválido | após correção: JSON válido |
|---|---|---|---|---|---|---|
| series_card | 1800 | 1061 | 720 | 0 (0.0%) | 0 | 0 |
| pool_card | 1800 | 1061 | 1398 | 0 (0.0%) | 0 | 0 |
| dataset_card | 900 | 1061 | 816 | 0 (0.0%) | 0 | 0 |
| diagnosis | 700 | 1061 | 484 | 0 (0.0%) | 0 | 0 |
| historico | 320 | 10105 | 423 | 252 (2.5%) | 252 | 252 |
| handles | 600 | 1061 | 1112 | 403 (38.0%) | 403 | 403 |
| scratchpad_args | 120 | 4213 | 412 | 787 (18.7%) | 787 | 787 |
| ultima_observacao | 1400 | 958 | 640 | 0 (0.0%) | 0 | 0 |
| **total** | | 20581 | | 1442 (7.0%) | 1442 | 1442 |

- `series_card`, `pool_card`, `dataset_card`, `diagnosis` e `ultima_observacao`: **0 disparos** na amostra (maiores tamanhos: 720/1398/816/484/640 contra limites 1800/1800/900/700/1400).
- Todos os cortes antigos (**1442**) produzem texto que **não é JSON válido** (0 de 1442 passam em `json.loads`), o que é esperado por construção: o corte é por posição de caractere. Onde o corte caiu: {'dentro_de_string': 1201, 'estrutural': 200, 'dentro_de_numero': 29, 'dentro_de_literal': 12}. (`dentro_de_string` = no meio de uma string; `dentro_de_numero`/`dentro_de_literal` = no meio de número/`true`/`null`; `estrutural` = entre tokens, mas com chaves/colchetes abertos.) Ou seja, 1242 cortes (86.1%) caem no meio de um valor.

Por dataset:

| dataset | séries | chamadas | excedem o limite | corte inválido | séries com ≥1 truncamento |
|---|---|---|---|---|---|
| ANP_MONTHLY | 25 | 4539 | 655 (14.4%) | 655 | 25 (100% das séries) |
| NN5_WEEKLY_DATASET | 25 | 5299 | 236 (4.5%) | 236 | 24 (96% das séries) |
| M4_WEEKLY_DATASET | 25 | 5032 | 264 (5.2%) | 264 | 23 (92% das séries) |
| ETTH1 | 7 | 1435 | 56 (3.9%) | 56 | 7 (100% das séries) |
| ETTH2 | 7 | 1299 | 80 (6.2%) | 80 | 7 (100% das séries) |
| ETTM1 | 7 | 1498 | 56 (3.7%) | 56 | 7 (100% das séries) |
| ETTM2 | 7 | 1479 | 95 (6.4%) | 95 | 7 (100% das séries) |

## Passo 2 — correção aplicada

`_compact` em `orchestrator_react/prompts.py` substituído pela versão fornecida (encolhe o objeto até caber e acrescenta `" [truncated: some fields omitted]"`). Mesma amostra reexecutada (mesmas 20581 chamadas, mesmos 1442 disparos).

- **(a) JSON inválido:** nos 1442 casos que excedem o limite, **1442 produzem JSON válido** (antes: 0). A validação é sobre o texto antes do sufixo `" [truncated: ...]"`; a string completa com o sufixo não é JSON estrito (0 de 1442 passam em `json.loads`), o que é esperado.
- **(b) Cortes no meio de um valor:** antes 1242; depois **0**. Todos os 1442 cortes inválidos antigos passam a ser JSON válido.
- Todos cabem no limite (`1442/1442`); o fallback `{"note":"payload too large to include"}` foi acionado **0** vezes; o marcador antigo `...[truncated]` não aparece mais.
- O que é omitido: em `historico`, o campo removido foi apenas `rationale` (verificado em 50 briefs de M4 acima de 320 chars: campo removido = `rationale` em 50/50); em `handles`, chaves finais (handles) ou elementos finais de listas; em `scratchpad_args`, parâmetros finais. Como o encolhimento remove chaves do fim primeiro, a ordem das chaves define o que se perde (ex.: `{"strategy":..,"params":..,"note":..}` com limite pequeno pode perder `params` antes de `strategy`).

## Fora do escopo, mas encontrado

Os dois ajustes anteriores de `prompts.py` (staged) haviam removido o `return "\n".join(parts)` no fim de `build_turn_prompt` (a função retornava `None`). Restaurei essa linha; o replay acima já roda com ela.
