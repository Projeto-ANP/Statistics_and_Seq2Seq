# Mudança 1 — `reorder_weight_tools`

## Implementado
- Flag `ReactConfig.reorder_weight_tools` (default `False`; CLI `--reorder-weight-tools`).
- `registry.py`: o dicionário `TOOLS` **não** é alterado (a flag é de execução, não de import). `describe_tools(withheld, reorder_weight_tools)` gera o catálogo do prompt com `weights_softmax_neg_error` e `weights_inverse_error` trocadas de lugar; as demais 22 entradas ficam na mesma ordem.
- `prompts.py`: `build_system_prompt(..., reorder_weight_tools)` inverte o par no parágrafo que descreve os dois métodos como parecidos entre si (`weights_softmax_neg_error / weights_inverse_error - each model's AVERAGE ...`). O restante do parágrafo é idêntico. O loop passa `config.reorder_weight_tools`.

## Verificação (sem LLM)
- Com a flag `False`, os 31 prompts (system + turno) reconstruídos em 2 séries (ANP e NN5) são **byte a byte iguais** aos de antes da mudança (`prompts_before` vs `prompts_flagsoff`, `prompt_format=text`).
- Com `True`, o diff do system prompt é só: a linha de `weights_softmax_neg_error` sobe uma posição acima da de `weights_inverse_error`, e o par no parágrafo troca de ordem. Mesmo tamanho (7.934 caracteres).
- Testes: `test_reorder_weight_tools_swaps_only_the_two_entries_and_the_paragraph`, `test_catalog_flags_default_off_and_do_not_move_the_fingerprint`. Suíte: 484 passam.
- O `ablation_config` (fingerprint) de um run com a flag `False` não muda; com `True`, muda.

## Validação isolada com o agente — NÃO executada
Esta máquina não tem GPU/Ollama com o gpt-oss, então não há SMAPE/POCID nem contagem de ações chamadas com o agente. O efeito da flag existe só através do modelo (a ordem no prompt). Para rodar na máquina com GPU (controle e teste, mesmo `--reasoning`, mesmo `prompt_format`):

```
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version t_ctrl    -- --reasoning low
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version t_reorder -- --reasoning low --reorder-weight-tools
```
Com 7 séries o teste só serve para ver mudança na contagem de chamadas de `weights_softmax_neg_error` (hoje 504 de 680 séries no v5, contra 666 de `weights_inverse_error`), não para inferência sobre SMAPE.
