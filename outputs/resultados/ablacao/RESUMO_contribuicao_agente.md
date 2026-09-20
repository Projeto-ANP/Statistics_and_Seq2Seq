# Ablação: CREST sem o agente (`orchestrator_baseline_v1`) contra o CREST completo (`orchestrator_react_v5`)

Scripts: `outputs/resultados/scripts/ablacao_verifica_desenho.py` (Passo 1), `ablacao_desempenho.py` (Passo 2), `ablacao_mcm.py` (Passo 3), `ablacao_contribuicao.py` (Passo 4).

## Passo 1 — o desenho do experimento está correto?

**Nunca invoca o agente/LLM: confirmado.** Nos 680 artefatos de `orchestrator_baseline_v1`: `combinator.model = null` (e diagnostician/reporter também `null`), `llm_model = none`, `iterations_used = 0`, `stop_reason = no_llm_client`, `n_agent_attempts = 0`. A execução vem de `run_tsf_baselines.py`, que força os três papéis para `LLMRole(model=None)`; sem cliente, `run_react_loop` devolve a melhor semente (`react_loop.py:165-172`). A estratégia final de cada série é a semente de menor `score`.

**Elementos idênticos ao `orchestrator_react_v5`** (config salva nos artefatos, comparada chave a chave nos 680 pares): conjunto de preditores, janelas de validação (`n_validation_windows`, `backtest_mode`, `nested_selection`), `pool_mode`, `score_preset`, `final_strategy=argmin`, `seed_stable_pools`, `pooled_meta_model` (+ objetivo e mínimo de séries), `seed_pooled_meta_model`, `max_iterations`, etc. Os conjuntos de sementes (id, estratégia, tamanho do pool) são idênticos nas 680 séries.

**Diferenças encontradas (a reportar antes de qualquer número):**

1. **Configuração:** `combinator` (LLM vs nenhum) e `name`, como esperado, e `dataset_card` (`false` no baseline, `true` no v5). O `dataset_card` só alimenta o prompt do agente e o `prior_blend` (`pipeline.py:647`), não o score das sementes. Além disso, `drop_misaligned_models` não está gravado em 484 dos artefatos do v5 (chave ausente); onde está gravado, vale `true` como no baseline.
2. **Número de sementes:** **10** em 652 séries e **9** em 28 (as 4 bases ETT × 7 séries). A décima (`a10`, `weighted` com o meta-modelo pooled) só existe quando o dataset tem ≥ 20 séries; isso vale igual nas duas execuções, então "dez sementes" descreve ANP/NN5/M4, não os ETT.
3. **A semente `a3` (DBA) não é a mesma.** É a **única** semente cujo score difere entre as execuções (660 das 680 séries; as outras 9 sementes têm score idêntico em todas as séries). Causa: o `combine_dba` usado no v5 tinha um bug documentado no próprio código (`orchestrator_react/combiners.py`, docstring de `combine_dba`): o `TypeError` de `random_state` caía no `except` e devolvia a **média simples**. Resultado nos artefatos: no v5, `a3` é numericamente idêntica a `a1` (média) em **666/680** séries; no baseline (código corrigido), em apenas 5/680. Ou seja, o baseline calcula DBA de verdade e o v5 não. Consequência: em **66 séries o vencedor do baseline é o DBA**, uma estratégia que o v5 nunca teve efetivamente. Por isso o Passo 4 é reportado com e sem essas 66 séries.

Observação: os JSONs de ANP em `orchestrator_react_v5/llm_artifacts/` são do Qwen3 (sobrescritos; os CSVs estão corretos). A comparação de sementes usou esses JSONs no lado v5 para ANP; as sementes não dependem do LLM, então não afeta a conclusão.

## Passo 2 — desempenho

`outputs/resultados/ablacao/desempenho_sem_agente_por_dataset.csv` (`regressor = CREST_sem_agente`, mesmo formato do `item_04`).

## Passo 3 — MCM pareada (680 séries como uma população)

Mesma chamada de `MCM.compare` do `item_07` (Wilcoxon + ProbaWinTieLoss), dois comparates: `CREST` e `CREST sem agente`.

| métrica | CREST sem agente (média) | CREST (média) | diferença de médias | r>c / r=c / r<c (linha CREST × coluna sem agente) | p-valor (Wilcoxon) |
|---|---|---|---|---|---|
| SMAPE | 0,1369 | 0,1376 | 0,0007 | 165 / 373 / 142 | 0,1403 |
| POCID | 58,4339 | 57,8561 | −0,5778 | 63 / 533 / 84 | 0,0785 |

Nenhuma das duas comparações é significativa a 0,05. Os 373 empates exatos em SMAPE coincidem com as 373 séries de estratégia idêntica do Passo 4. Como no `item_07`, `order_WinTieLoss` ficou no padrão do pacote ("higher"), então em SMAPE `r>c` conta séries em que o SMAPE da linha é maior. Arquivos em `outputs/resultados/ablacao/`: `mcm_agente_vs_sem_agente_smape.{csv,png,pdf}`, `mcm_agente_vs_sem_agente_pocid.{csv,png,pdf}` e as entradas `mcm_input_agente_vs_sem_agente_{smape,pocid}.csv`.

## Passo 4 — quantas séries o agente altera

Estratégia = `combine` + `pool` + `weights`/`trim_pct`/`model` (`best_strategy_params`). Empate: |ΔSMAPE de teste| ≤ 1e-4. Nota: o `item_16` **não** tem limiar de empate no código (calcula só a diferença de scores de validação); apliquei o 1e-4 indicado ao SMAPE de teste.

**Todas as 680 séries:**

| | séries | % |
|---|---|---|
| estratégia final idêntica | 373 | 54,9% |
| estratégia final diferente | 307 | 45,1% |
| — agente melhor (SMAPE menor) | 130 | |
| — agente pior (SMAPE maior) | 158 | |
| — empate | 19 | |

Nas 373 séries de estratégia idêntica, as previsões também são idênticas (dif. máx. ≤ 1e-6 em 373/373). Diferença média de SMAPE (com agente − sem agente) nas 307 diferentes: +0,0016.

**Excluindo as 66 séries em que o vencedor sem agente é o DBA (n = 614):** 373 idênticas (60,7%), 241 diferentes (39,3%): 100 agente melhor, 124 pior, 17 empates; diferença média +0,0017.

**Por dataset (680 séries):**

| dataset | n | idêntica | diferente | agente melhor | agente pior | empate |
|---|---|---|---|---|---|---|
| ANP_MONTHLY | 182 | 96 | 86 | 34 | 48 | 4 |
| NN5_WEEKLY_DATASET | 111 | 51 | 60 | 26 | 31 | 3 |
| M4_WEEKLY_DATASET | 359 | 209 | 150 | 61 | 77 | 12 |
| ETTH1 | 7 | 4 | 3 | 2 | 1 | 0 |
| ETTH2 | 7 | 5 | 2 | 2 | 0 | 0 |
| ETTM1 | 7 | 4 | 3 | 3 | 0 | 0 |
| ETTM2 | 7 | 4 | 3 | 2 | 1 | 0 |

Origem da estratégia vencedora no v5: 273 séries `agent`, 407 `baseline`. Em 34 séries a estratégia difere mesmo com origem `baseline` no v5; todas ocorrem porque o vencedor do baseline é o DBA (ver Passo 1, item 3).

Detalhe por série: `outputs/resultados/ablacao/pareado_por_serie.csv`.
