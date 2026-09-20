# Granularidade de tempo nos logs do CREST (orchestrator_react_v5)

## Resposta direta

**Não existe granularidade por etapa/ferramenta.** Só o tempo total do ciclo, por série, foi registrado. Não há tempo de início/fim por chamada de ferramenta, por iteração, nem separação entre tempo de geração do LLM e tempo de execução de código determinístico. Como pedido, não tentei estimar ou inferir essa decomposição a partir dos dados existentes.

## Como cheguei a essa conclusão

Verifiquei em três fontes independentes, todas concordando:

**1. Código-fonte** (`orchestrator_react/react_loop.py`): busquei toda ocorrência de medição de tempo (`time.perf_counter`, `time.time()`, `elapsed`) em todo o pacote `orchestrator_react/`. Só há 5 linhas no projeto inteiro, todas parte da mesma medição única:
- `react_loop.py:157`: `started = time.perf_counter()` — marcado uma vez, no início do loop ReAct.
- `react_loop.py:170` e `react_loop.py:387`: `result.elapsed_s = time.perf_counter() - started` — calculado no fim (dois pontos de saída possíveis: early-return e o caminho normal), sempre contra o mesmo `started`.
- `react_loop.py:89`, `:128`: só a declaração do campo `elapsed_s` e sua serialização.

Não há nenhuma segunda chamada a `time.perf_counter()`/`time.time()` em nenhum outro ponto do pacote (`pipeline.py`, `tools.py`, `pool.py`, `llm.py`, `state.py`, `features.py`, `ingest.py`) — busquei `import time` e toda ocorrência de `time.` nesses arquivos; as únicas outras aparições da palavra "time" são em comentários/docstrings, não chamadas de medição. Em particular, `llm.py` (o wrapper que chama o LLM) não mede tempo internamente.

**2. Artefato JSON rico por série** (`timeseries/mestrado/resultados/orchestrator_react_v5/llm_artifacts/{dataset}/dataset_{idx}.json`, um arquivo por série, bem mais detalhado que o CSV — inclui `series_card`, `pool_card`, `diagnosis`, `react.trajectory`, `react.tools.tools_called`, `sanity`, etc.): varri recursivamente todo o JSON de uma série (`ETTM1`, `dataset_index=0`) procurando qualquer chave com "time", "elapsed", "duration", "latency" ou "ts" no nome. O único campo relacionado a tempo em todo o arquivo é `decision.loop.elapsed_s` / `react.summary.elapsed_s` — o mesmo valor total (17.54s), duplicado em dois lugares do mesmo JSON, mas não decomposto por fase. Os passos de `react.trajectory[]` e `react.tools.tools_called[]` têm ação, argumentos e resumo da observação, mas nenhum timestamp ou duração individual.

**3. Colunas do CSV** (`timeseries/mestrado/resultados/orchestrator_react_v5/*.csv`): a única coluna com granularidade de tempo é `react_iterations_used`/`description.loop.elapsed_s`, ambas relativas ao ciclo inteiro, não a uma iteração ou ferramenta específica.

## Consequência

Os itens 1, 2 e 3 pedidos pelo prompt (tempo por fase numa amostra de séries, checagem se a soma bate com o total, e a fração LLM-vs-determinístico) não são respondíveis com os dados atuais — não por limitação de amostra, mas porque a informação nunca foi registrada.
