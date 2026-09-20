# Instrumentação de tempo por ação e registro por passo

Script de validação: `outputs/resultados/scripts/instrumentacao_validacao.py` (saída: `instrumentacao_validacao.csv`).

## O que foi implementado (aditivo)

- `orchestrator_react/react_loop.py`
  - Cada entrada de `react.trajectory` ganha `llm_call_s` (tempo de `client.complete`, somando as re-perguntas por resposta vazia/erro transitório do mesmo turno) e `tool_exec_s` (só `call_tool`; 0.0 em `accept` e em resposta não parseada). `thought` (clipado a 600) e `elapsed_s` ficam como estavam. Como a trajetória é a mesma lista do CSV, essas duas chaves numéricas também aparecem em `react_trajectory_json` (duas floats por passo).
  - Novo `ReactResult.step_details`, uma entrada por passo, **só no artefato JSON** (`react.step_details`, via `csv_writer.artifacts_payload`; não entra no CSV): `thought_full` (sem clip), `raw_response` (resposta bruta completa), `has_think_block`, `think_chars` (bloco `<think>…</think>` não vazio, pelo mesmo `split_think` do parser), `llm_call_s`, `llm_calls_in_turn` e os contadores do Ollama com prefixo `ollama_`.
  - `summary()` ganha `llm_call_s_total` e `tool_exec_s_total`. Turno cuja chamada falha e encerra o loop (`llm_error`) não gera entrada na trajetória; seu tempo vai para `ReactResult.llm_failed_turn_s` e entra em `llm_call_s_total`.
- `orchestrator_react/llm.py`: `OllamaClient.last_meta` guarda, após cada `complete()`, apenas as chaves que o Ollama devolveu (`prompt_eval_count`, `eval_count`, `eval_duration`, `prompt_eval_duration`, `total_duration`, `load_duration`, `done_reason`) e `reasoning_content_chars` (tamanho do raciocínio que o langchain separa em `additional_kwargs["reasoning_content"]`). Chave ausente = não registrada. Com falha na chamada, `last_meta` fica vazio.
- Nenhum caminho de decisão foi tocado (só medição e registro).

## Validação (sem GPU, sem LLM real)

15 séries: NN5_WEEKLY_DATASET 0–7 e ETTM2 0–6. O loop real (`run_react_loop`) foi executado com o replay `ScriptedLLM` da trajetória gravada, com um cliente que dorme 50 ms por chamada e expõe `last_meta` no formato do Ollama. **Portanto `llm_call_s` aqui é o sleep fixo, não tempo de modelo; o que se valida é a contabilidade.** `tool_exec_s` é real (as ferramentas rodam de verdade).

**(a) Soma vs `elapsed_s`.** `elapsed_s` cobre só o ciclo ReAct (não inclui ingestão/Fase 2, que ficam fora, como já era).

| | soma nas 15 séries (s) |
|---|---|
| `elapsed_s` | 10,80 |
| `llm_call_s` (incl. turno que falhou) | 9,78 |
| `tool_exec_s` | 0,69 |
| resíduo (`elapsed_s − llm − tool`) | 0,33 |

O resíduo é o que fica fora das duas medidas (montagem do prompt, `summarize_observation`, parse): 0,014–0,028 s por série, ≤ 4% de `elapsed_s` em todas as séries (média 0,022 s). Antes de contar o turno que falhou, o resíduo era 2,32 s (0,27 s a mais nas 7 séries que terminaram em `llm_error`); isso foi corrigido com `llm_failed_turn_s`.

**(b) JSON e tamanho.** As 15 saídas parseiam com `json.loads` e todas as entradas da trajetória têm `llm_call_s` e `tool_exec_s`. Tamanho da parte de trajetória (+ `step_details`) por série: 4,8 KB → 12,8 KB em média (72.580 → 191.316 bytes no total, ×2,6). Para referência, os artefatos atuais de NN5 pesam em média 27,6 KB (máx. 30,9 KB), então o acréscimo medido é de ≈ +8 KB por série (~ +29%).

Ressalva: esse acréscimo está **subestimado**. No replay a `raw_response` é o roteiro reconstruído a partir da trajetória gravada (com `thought` já clipado em 600 caracteres e sem o bloco `<think>` do modelo), enquanto na execução real ela traz o texto completo do modelo, incluindo o raciocínio. O tamanho real só se conhece rodando o agente; conferir em algumas séries do primeiro run com GPU.

## Não validado aqui

- Os contadores `ollama_*` e `reasoning_content_chars` só foram exercitados com o cliente simulado; o formato real de `response_metadata` do `ChatOllama` deve ser conferido no primeiro run (o código copia apenas as chaves que existirem).
- Se o modelo gerar o raciocínio pelo canal `reasoning` do Ollama (parâmetro `reasoning` do papel), ele não vem dentro de `content`, então `has_think_block` fica `false` mesmo com raciocínio; nesse caso o tamanho aparece em `ollama_reasoning_content_chars`.
- Não foi rodado o pipeline completo (`run_tsf_orchestrator`) nem comparados SMAPE/POCID: nenhuma linha de código de decisão mudou, e nas 7 séries de ETTM2 que também estavam na rodada do truncamento (anterior às mudanças) o replay deu o mesmo número de passos e o mesmo motivo de parada.
