# Mudança 4 — `reduced_seeding`

## Implementado
- Flag `ReactConfig.reduced_seeding` (default `False`; CLI `--reduced-seeding` em `run_tsf_orchestrator.py` e em `run_tsf_baselines.py`).
- `pool.run_phase2` passa `methods=("dba",)` (`REDUCED_SEED_BASELINES`) em vez de `("mean","median","dba")`. As seis sementes de pool estável (k=5/7/9 × mean/trimmed_mean) e a do meta-modelo pooled ficam como estão. O total é **8 sementes: DBA + 6 estáveis + a do meta-modelo pooled** (9→7 em datasets sem meta-modelo). O enunciado fala em "DBA e as seis de estabilidade" (7); o oitavo é o meta-modelo pooled, que a flag mantém.
- O pré-passe do cartão do dataset (`pipeline.py`, `strategy_prior`) usa o mesmo `run_phase2`, então com a flag o cartão também deixa de listar mean/median do pool completo.
- Efeito colateral a ter em mente: os ids das tentativas se deslocam (o DBA passa a ser `a1`).

## Verificação
- Com a flag `False` o comportamento é o de antes (mesma lista de sementes); teste `test_reduced_seeding_drops_full_pool_mean_and_median_only` confere que a flag remove exatamente as duas sementes e preserva o resto e a ordem.

## Validação isolada — parte SEM agente (executada, NN5, 111 séries)
`run_tsf_baselines.py` (só a parte determinística: melhor semente por score de validação), mesmo código, uma execução com 10 sementes e outra com `--reduced-seeding` (`semeadura_reduzida_nn5.py`). É o **piso** do CREST, não o CREST com agente.

| | controle (10 sementes) | reduzida (8) | diferença |
|---|---|---|---|
| SMAPE médio | 0,11597 | 0,11679 | +0,00082 |
| POCID médio | 46,203 | 46,203 | 0,000 |
| RMSE médio | 18,229 | 18,421 | +0,192 |

- SMAPE por série: reduzida melhor em 18, pior em 9, empate em 84 (|dif| ≤ 1e-4). Wilcoxon nas 30 séries com diferença: p = 0,57 (não significativo). POCID por série: maior em 7, menor em 5, igual em 99.
- Estratégia final idêntica em **81 de 111** séries. Nas 30 que mudam, o controle vencia justamente com mean/median do pool completo (o controle vence com essas duas em exatamente 30 séries); sem elas, a melhor semente passa a ser outra.

## Validação isolada — parte COM agente: NÃO executada
O teste de redescoberta (o agente propõe mean/median do pool completo sozinho? recupera o piso?) exige o gpt-oss; sem GPU aqui. Comandos (NN5, 111 séries; controle e teste com os mesmos `--reasoning`/`prompt_format`):
```
python3 run_tsf_batch.py --datasets NN5_WEEKLY_DATASET --combinators gpt-oss:20b --version t_ctrl   -- --reasoning low
python3 run_tsf_batch.py --datasets NN5_WEEKLY_DATASET --combinators gpt-oss:20b --version t_reduced -- --reasoning low --reduced-seeding
```
Para medir a redescoberta, procure em `react_trajectory_json` do run reduzido as chamadas `evaluate_strategy` com `combine` mean/median e `pool_full` (ou sem `pool`), e compare o SMAPE/POCID final contra o controle série a série (o piso determinístico acima é a referência do que se perde sem redescoberta).
