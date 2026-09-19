# Resumo -- por que o CREST não herda a divergência do CatBoost nos ETT

Fatos apurados a partir de `outputs/diagnostico/passo6_protecao_crest.csv` (26 séries: as mesmas onde `catboost` diverge nos 4 datasets ETT, de `outputs/resultados/diagnostico/passo5_contexto_divergencia.csv`) e do código-fonte de `orchestrator_react/pool.py`, `tools.py`, `pipeline.py`, `state.py`. Sem recomendação de tratamento.

## 1. Em que percentual das séries afetadas o CREST excluiu catboost do pool efetivo?

**100% (26 de 26).** `catboost_no_pool_efetivo = False` em toda linha do Passo 6 — nenhuma série divergente teve `catboost` entre os modelos que efetivamente pesaram no resultado final do CREST.

## 2. Essa exclusão veio predominantemente de uma ação do agente, ou de uma estratégia semeada que já excluía catboost por construção?

**Predominantemente de uma estratégia semeada por construção — mas por dois mecanismos distintos, não um só.** Todo run do CREST semeia 9 candidatos determinísticos *antes* do loop ReAct começar (`orchestrator_react/pool.py:90-160`, `seed_baselines`), todos com `origin="baseline"`: 3 sobre o pool cheio (19 modelos: `mean`, `median`, `dba`) e 6 sobre pools "estáveis" pré-selecionados por `select_stable` para k ∈ {5,7,9} × {`mean`,`trimmed_mean`} (`SEED_STABLE_POOLS`, `pool.py:54-58`). `select_stable` (`tools.py:369-400`) ranqueia cada modelo por (rank médio + desvio-padrão do rank) do RMSE entre as janelas de validação (`selection.py:49-69`) — um modelo com previsões que explodem em ordens de grandeza absurdas fica sistematicamente nas últimas posições desse ranking em qualquer janela, garantindo exclusão de todo pool `top-k` estável, sem qualquer ação do agente.

Das 26 séries:
- **13/26 (50%)**: vitória de uma dessas baselines semeadas de pool reduzido (`origin=baseline`, tamanho do pool nominal 5, 7 ou 9) — catboost já estava fora do pool nominal por construção, via `select_stable`, antes do agente agir.
- **10/26 (38,5%)**: vitória de uma proposta do próprio agente (`origin=agent`) — nesses 10 casos, o pool nominal da estratégia vencedora também já excluía catboost (`catboost_no_pool_nominal=False` nos 10), ou seja, o agente construiu/usou um pool sem catboost.
- **3/26 (11,5%)**: um terceiro mecanismo, fora da dicotomia pedida — ver pergunta 3.

Nota metodológica: o prompt também pediu checar `tools_called`/`react_trajectory_json` por uma chamada explícita de `prune_redundant`/`select_stable`/`select_top_k`. Esse sinal bruto ("alguma chamada de poda em algum ponto da trajetória") aparece em 10/10 dos casos `origin=agent` mas também em 14/16 dos casos `origin=baseline` — ou seja, o agente frequentemente chama essas ferramentas mesmo em séries onde acaba perdendo para uma baseline semeada. Esse sinal bruto não distingue de forma confiável qual pool efetivamente venceu; `origin` é o sinal preciso (confirmado em código: `origin="baseline"` só é atribuído dentro de `seed_baselines`, nunca pela trajetória do agente — `pool.py:115,126,148`).

## 3. Há algum caso em que catboost permaneceu no pool efetivo de uma série divergente sem quebrar o resultado?

**Não, no sentido literal — 0 de 26.** Mas há um caso limítrofe que se encaixa no espírito da pergunta: em **3/26 séries (11,5%, todas em `ETTH2`, `dataset_index` 1, 4 e 5)**, catboost **permaneceu no pool nominal** (`catboost_no_pool_nominal=True`, pool cheio de 19 modelos — a baseline semeada `median` sobre `pool_full`, `origin=baseline`), mas ainda assim `catboost_no_pool_efetivo=False`.

A razão está na aritmética da própria estratégia vencedora, não em um filtro: `best_strategy_method="median"` nas 3. `effective_models` é calculado a partir do peso implícito de cada modelo no horizonte 0 (`orchestrator_react/pipeline.py:223-238`, limiar `abs(peso) > 0.01`), e a combinação por mediana usa apenas 1–2 modelos por ponto do horizonte, os de rank central (`pipeline.py:102-113`) — uma previsão consideravelmente maior ou menor que as demais é empurrada para a borda do ranking e nunca reflete no peso central, independente de quão grande seja o valor absurdo. Não existe, em nenhum ponto do código de combinação (`orchestrator_react/state.py:562-584`, `combiners.py:29-167`), um filtro por magnitude/valor-fora-de-escala que remova modelos antes de `mean`/`dba` serem calculados — a robustez da mediana é a única razão desses 3 casos ficarem limpos apesar de catboost nominalmente presente. (Confirma-se também que, se a estratégia vencedora tivesse sido a `mean`/`dba` semeada sobre o pool cheio nessas mesmas séries, catboost teria peso uniforme e contaminaria o resultado — `pipeline.py:88-100` — o que não ocorreu porque, nessas 3 séries, a baseline `median` pontuou melhor que `mean`/`dba` no histórico de tentativas.)

`peso_catboost` é vazio nas 3 (estratégia não é `weighted`); nas 7 séries onde a estratégia vencedora é `weighted` e catboost aparece em `weights_by_horizon`, seu peso no horizonte 0 é `0.0` em todas — consistente com catboost já estar fora do pool nominal dessas 7 estratégias.

---

**Distribuição completa de `best_strategy_method` nas 26 séries**: `trimmed_mean` 10, `weighted` 7, `median` 4, `mean` 4, `best_single` 1.

Arquivo: `outputs/diagnostico/passo6_protecao_crest.csv`.
