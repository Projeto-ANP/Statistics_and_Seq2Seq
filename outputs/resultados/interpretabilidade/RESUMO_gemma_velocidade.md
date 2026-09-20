# O modo de raciocínio do Gemma estava ativo? (`gemma4:26b`)

## 1. O token `<|think|>` estava no prompt de sistema?

**Não.** A string `<|think|>` não existe em nenhum `.py`, `.md`, `.txt` ou `.json` do repositório (busca em todo o projeto, excluindo `outputs/` e os artefatos). O prompt de sistema é montado em `orchestrator_react/prompts.py:27` (`build_system_prompt`) e começa por:

```python
rules = [
    "You are a forecast COMBINATION AGENT.",
    "",
    ...
```

**O código é compartilhado pelos três modelos, sem lógica condicional por modelo.** `OllamaClient` (`orchestrator_react/llm.py:61`) monta o mesmo payload `/api/chat` para qualquer modelo (`_build_chat_payload`, l.117-143): mensagem `system` = texto de `build_system_prompt`, mensagem `user`, `stream=False`, opções `num_ctx`/`temperature`/`seed`. Nenhum `if` compara o nome do modelo.

**Único caminho existente para ligar/desligar o raciocínio** (`llm.py:135-142`): só se `role.reasoning is not None` o payload inclui `"think": <valor>`. Nesta execução `reasoning` era `None` nos 680 artefatos do Gemma (`config.combinator.reasoning = null`), e o comando registrado em `logs/v5_batch.log` não passa `--reasoning`. Logo **o campo `think` não foi enviado ao Ollama**; o comportamento passou a ser o padrão do Ollama para esse modelo.

**Limite desta verificação:** o repositório não permite saber qual é esse padrão para `gemma4:26b` (se o Ollama insere o gatilho do modelo quando `think` é omitido depende da versão do Ollama e do template do modelo, que não estão no repositório). Também não há `ollama show`/log do servidor salvo. Portanto o Passo 1 mostra que **o código não injeta o token nem pede `think`**, mas não prova que o modo esteve desligado.

## 2. Comprimento das respostas, por modelo

**A resposta bruta do LLM não é armazenada em lugar nenhum** (nem no CSV, nem em `llm_artifacts/*.json`, nem nos logs). Cada passo do `react.trajectory` guarda só `iteration`, `thought`, `action`, `action_args` e `observation_summary`, e o `thought` é cortado em 600 caracteres (`react_loop.py:253`, `_clip(step.thought, 600)`). `thought` é o conteúdo do bloco `<think>...</think>` se o modelo emitiu um, senão a linha `Thought:` (`llm.py:464-500`). O `thinking` separado do Ollama é fundido em `<think>` por `combine_ollama_message` (`llm.py:173-183`) antes disso, mas nada disso é persistido. Não há contagem de tokens salva (`eval_count` só aparece no log de respostas vazias, e esse log tem 0 ocorrências nas execuções do Gemma).

Por isso o comprimento abaixo é o do **`thought` parseado**, não o da resposta inteira. Fonte: `react_trajectory_json` e `description.loop` dos CSVs, todas as séries (n ≥ 30 por modelo em todos os casos). Script: `outputs/resultados/scripts/interpretabilidade_gemma_velocidade.py`.

| modelo | séries | passos | `thought` (chars) média | desvio | mediana | % no limite de 600 | chamadas/série | respostas vazias/série | s por chamada (média) | desvio |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss:20b | 679 | 7217 | 141,3 | 56,7 | 138 | 0,0% | 13,2 | 2,52 | 4,71 | 7,32 |
| qwen3:30b-a3b | 680 | 5616 | 253,9 | 89,2 | 233 | 0,9% | 8,3 | 0,02 | 18,11 | 12,70 |
| gemma4:26b | 680 | 7841 | 240,4 | 112,1 | 266 | 0,0% | 16,0 | 4,47 | 13,18 | 2,03 |

"Chamadas/série" = passos da trajetória + respostas vazias + repetições por erro de LLM; "s por chamada" = `elapsed_s` da série dividido por esse número (inclui o tempo das ferramentas determinísticas, que é pequeno). O gpt-oss tem 679 séries porque uma série não tem trajetória.

## 3. Leitura dos números

- O `thought` do Gemma (média 240 chars) **não é visivelmente maior** que o do Qwen3 (254) e é maior que o do gpt-oss (141). Nenhuma resposta do Gemma chega ao corte de 600 caracteres. Como o `thought` só preserva um bloco `<think>` inteiro se ele existir, um raciocínio estendido longo tenderia a aparecer como `thought` longo/cortado; isso não aparece para o Gemma. É um indício indireto, mas **não confirma nem refuta** o modo de raciocínio, porque não vemos o comprimento total da resposta.
- O que os dados mostram sobre a diferença de tempo é outra coisa: o Gemma faz **mais chamadas por série** (16,0 contra 13,2 do gpt-oss e 8,3 do Qwen3) e tem **mais respostas vazias** (4,47 por série, contra 2,52 e 0,02), cada uma exigindo repetir a chamada. Por chamada, o Gemma (13,2 s) é ~2,8× mais lento que o gpt-oss (4,7 s) e mais rápido que o Qwen3 (18,1 s). O tempo total maior do Gemma vem de combinar as duas coisas.
- Como pedido, isso **não** deve ser apresentado como confirmação do modo de raciocínio: o Passo 1 não encontrou gatilho no código e não dá para saber o padrão do Ollama; o Passo 2 não tem a resposta bruta para medir. Para decidir a hipótese seria preciso registrar `message.thinking`/`eval_count` (ou rodar uma chamada de teste ao Ollama com `think` explícito `true` e `false`) e comparar o tempo.

## Observação fora do escopo, encontrada no caminho

Em `orchestrator_react_v5/llm_artifacts/ANP_MONTHLY/` os 182 JSONs têm `combinator.model = qwen3:30b-a3b`, e `orchestrator_react_v5_qwen/llm_artifacts/ANP_MONTHLY/` não existe. Ou seja, os artefatos de ANP do gpt-oss foram sobrescritos pelos do Qwen3 na pasta do gpt-oss. Os **CSVs estão corretos** (`llm_model` bate com a pasta nos 21 pares modelo×dataset), então nenhum levantamento baseado em CSV é afetado; só quem ler os JSONs de ANP dessa pasta receberá dados do Qwen3.
