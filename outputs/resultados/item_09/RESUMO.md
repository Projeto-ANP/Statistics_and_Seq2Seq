# Item 09 v2 -- erro por passo do horizonte (SMAPE + POCID)

## Fórmulas

- **SMAPE ponto a ponto**: `2*|pred-test| / (|test|+|pred|)` (0 se ambos forem 0), idêntico ao v1.
- **POCID ponto a ponto**: `1` se `sign(pred[t]-pred[t-1]) == sign(test[t]-test[t-1])` (produto `> 0`), `0` caso contrário — mesma fórmula usada nas métricas agregadas por série do projeto (`all_functions.py:134-141`, `orchestrator_react/metrics.py:62-71`), só que aplicada a cada ponto do horizonte em vez de uma vez para a série inteira.

## Passo 0 do POCID

O passo 1 do horizonte não tem um passo anterior *dentro* da janela de teste, então precisa de um valor de referência de fora dela (o último ponto observado do treino).

**Não há coluna de treino em nenhum CSV de resultado.** A primeira tentativa foi reconstruir esse valor a partir do `.tsf` original (`forecasting_datasets/`): `valor_completo_da_série[-(horizonte+1)]`. Isso bateu exatamente com o `test` salvo nos resultados para 5 dos 7 datasets (ANP_MONTHLY, NN5_WEEKLY_DATASET, M4_WEEKLY_DATASET, ETTH1, ETTH2) — mas **não bateu para ETTM1 nem ETTM2**, em nenhuma das variantes de arquivo disponíveis em `forecasting_datasets` (`ETTM1.tsf`, `ETTm1.tsf`, `ETTm1.csv`, e equivalentes de ETTM2): nenhuma das 7 séries de nenhum desses arquivos reproduz o `test` de nenhum `dataset_index` desses dois datasets, em nenhuma posição da série (busca exaustiva, não só na cauda). O arquivo `.tsf` atualmente em disco para ETTM1/ETTM2 não é a fonte que gerou os resultados que estão em `timeseries/mestrado/resultados/`.

**Solução usada, sem depender de nenhum arquivo externo**: os CSVs de origem de modelos individuais (usei `rf` como sonda) guardam múltiplas janelas de validação por série antes do filtro de janela de teste (`load_filtered` mantém só a mais recente) — 31 janelas em ANP_MONTHLY, 4 em todos os outros datasets, sempre ≥2 em toda série de todo dataset (confirmado, sem exceção). Como a divisão treino/teste caminha para trás em blocos de tamanho `horizonte` (`aux_series[:-horizonte]` / `aux_series[-horizonte:]`, repetido), o último valor do `test` da segunda janela mais recente é exatamente o ponto que precede o primeiro ponto da janela mais recente — o "passo 0" procurado. Validado batendo exatamente (6 casas decimais) com a reconstrução via `.tsf` nos 5 datasets onde essa reconstrução é confiável (ex.: M4_WEEKLY_DATASET, `dataset_index=208`: as duas dão `4185.45`). Por consistência, esse método (não o `.tsf`) foi usado para os 7 datasets, não só para ETTM1/ETTM2.

Implementação: `common.pre_test_reference_values(dataset)`, em `outputs/resultados/scripts/common.py`.

## Nota

Essa divergência entre o `.tsf` atual e os resultados salvos é específica de ETTM1/ETTM2 e não foi investigada além do necessário para resolver o passo 0 do POCID -- não sei se afeta algo além disso.
