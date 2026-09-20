# Mudança 3 — `drop_redundant_combine_actions`

## Confirmação de que as quatro ações são redundantes (`tools.py`, `registry.py`, `state.py`)
- `combine_mean/median/weighted/best_single` chamam só `_strategy(state, spec)` (`tools.py:618`): `state.normalize_spec(spec)` (valida e canonicaliza) e devolvem `{strategy, n_models, next_step, next_action_input}`. **Não registram nada no estado** e não pontuam nada; `get_pool`/`model_index` só leem/validam.
- `evaluate_strategy` recebe os mesmos campos planos (`combine`, `pool`, `weights`, `model`) e chama `state.evaluate` → `state.backtest` → `normalize_spec`, a mesma função. Portanto a spec resultante é idêntica (o teste `test_dropped_actions_are_reproducible_through_evaluate_strategy` compara as quatro).
- Diferenças sem efeito: `n_models` retornado (deduzível do pool) e o dica `next_step` (só descrevia chamar `evaluate_strategy`). `combine_best_single` usa o argumento `model_id`, `evaluate_strategy` usa `model`; `normalize_spec` aceita os dois nomes, mas `evaluate_strategy` descartaria `model_id` (é permissiva), então o prompt já orienta `model`.
- `combine_trimmed_mean` e `combine_dba` **ficam** no catálogo, como pedido, mesmo sendo igualmente redundantes.

## Implementado
- Flag `drop_redundant_combine_actions` (default `False`; CLI `--drop-redundant-combine`). Reusa o mecanismo `withheld_tools` que já esconde `weights_ols`/`weights_pooled_meta_model`: as quatro somem do catálogo do prompt e chamá-las devolve `unknown_tool` ("unavailable for this run"). A flag `False` mantém as 24 entradas.
- Texto do prompt (só com a flag ligada): a linha "You do NOT need combine_* first" passa a citar `combine_trimmed_mean / combine_dba`, e a regra "Pass the handle to combine_weighted" vira `Pass the handle to evaluate_strategy as "weights"`. Sem a flag, o texto é o de antes.

## Números pedidos (680 séries do `orchestrator_react_v5`, `catalogo_uso.py`)
| | ações no catálogo | nunca chamadas |
|---|---|---|
| hoje | **24** | **14 de 24** |
| com a flag | **20** | **10 de 20** |

- As quatro removidas nunca foram chamadas (0 séries cada), então o numerador cai de 14 para 10 e o denominador de 24 para 20. As 10 restantes nunca chamadas: `series_profile`, `stl_summary`, `error_summary`, `ranking_stability`, `error_correlation`, `dm_test`, `weights_ols`, `weights_feature_based`, `sanity_check`, `list_attempts`.
- Ações chamadas (10, iguais antes e depois): `evaluate_strategy` 678 séries, `weights_inverse_error` 666, `prune_redundant` 591, `weights_softmax_neg_error` 504, `weights_error_trend` 238, `weights_pooled_meta_model` 225, `select_top_k` 28, `select_stable` 19, `combine_trimmed_mean` 1, `combine_dba` 1. O critério é "aparece em `tools_called` da série" (chamada com ou sem sucesso); contando só chamadas com sucesso o resultado é o mesmo, 14/24 e 10/20.
- **Ressalva importante para o texto da dissertação:** os 24/20 são do registro (`TOOLS`), não do que o agente enxergou. Com 3 janelas de validação, `weights_ols` (`min_windows_for_ols=5`) é sempre retida, então o catálogo **anunciado** era de **23 ações** em todas as séries (22 nas séries dos datasets sem meta-modelo pooled, por exemplo os ETT, que retêm também `weights_pooled_meta_model`). Sobre o catálogo anunciado, o "nunca chamadas" é 13 de 23 hoje e **9 de 19** com a flag. Decida qual denominador vai no texto; os dois números acima (14/24 → 10/20) são os exatos sobre o registro.

## Validação isolada com o agente — NÃO executada
Sem GPU aqui. O efeito só existe via modelo. Comando (ETTm2, 7 séries; compare com o controle rodado com os mesmos `--reasoning`/`prompt_format`):
```
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version t_ctrl -- --reasoning low
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version t_dropcomb -- --reasoning low --drop-redundant-combine
```
`catalogo_uso.py` recalcula o uso do catálogo para um novo run (passe a pasta como argumento: `python3 outputs/resultados/scripts/catalogo_uso.py orchestrator_react_<versao>`). Em ETTM2 o catálogo anunciado seria 18 com a flag (22 − 4).
Verificações sem LLM: `unknown_tool` ao chamar as quatro com a flag; catálogo do prompt com 20 (19 com 3 janelas: sem `weights_ols`) entradas; texto do prompt sem `combine_weighted`/`combine_best_single`.
