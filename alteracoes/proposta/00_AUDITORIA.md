# Auditoria do capítulo de método

Data: 16/11/2026.

## Escopo

Confronto do `overleaf/proposta.tex` com `orchestrator_react/`, com o prompt efetivamente usado nas 10 execuções e com os artefatos de `timeseries/mestrado/resultados/orchestrator_react_consist_seed*`.

## Correções aplicadas

1. **Duas visões da forma das previsões.** O texto tratava `F_i` como uma única representação usada pelo agente e pela decisão final. O código usa duas construções distintas: `_forecast_shape_card` no prompt (escores z por vetor, resumos, grupos de correlação) e `build_series_context`/`_candidate_features` na decisão final (escala comum, mediana e quantis do pool, referência sazonal, distâncias). O texto passou a distinguir `F_i` (agente) e `G_i` (decisão final), com a notação, o Algoritmo 1 e a tabela de blocos atualizados.

2. **Limiares do diagnóstico.** Acrescentados ao texto e ao Algoritmo 3: regime 0,6; entropia 0,85/0,5; tau 0,3; dispersão 0,5 com tau 0,5 para seletiva; dispersão 0,3 para ponderada. Os rótulos qualitativos do algoritmo foram substituídos pelas condições numéricas.

3. **Riscos truncados.** O código mantém apenas os três primeiros riscos (`risks[:3]`); a limitação das três janelas pode ficar de fora quando os outros três riscos ocorrem. O texto agora diz “até três riscos” e explica a truncagem.

4. **Catálogo de ações.** `prune_redundant` existe no código, mas foi removida do conjunto oferecido pela configuração (`scoring_budget` + `direct_preparation`). A tabela de ações passou a listar apenas as ações oferecidas; o texto menciona a poda como parte do catálogo, não do conjunto avaliado.

5. **Métodos de pesos.** A tabela ganhou `feature_based`, `meta_model` e `pooled_meta_model`, com a ressalva de que o último depende de um meta-modelo entre séries treinado. O texto explica que `weights_meta_model` reúne os escopos de série e de conjunto.

6. **Blocos do prompt.** A coluna “rótulo literal” foi trocada por tags XML (`series_profile`, `model_pool`, `diagnosis`, `forecast_shape`, etc.), que são o que o prompt realmente contém. O conteúdo de `forecast_shape` foi corrigido; o histórico lista até dez tentativas; o scratchpad, as últimas quatro.

7. **Detalhes factuais.** Corrigidos: o agente não escreve pesos; a seleção por erro aceita janelas; o limite de admissibilidade é 100x o maior valor entre último valor, desvio recente e 1; a duplicata exige correlação acima de 0,9999 e mesma forma padronizada; a média da distância sazonal ignora passos não finitos; a substituição do líder usa o primeiro candidato admissível da shortlist, não necessariamente o de menor pontuação.

8. **Condição sazonal sem distância do líder.** `final_choice.py` agora mantém o líder quando a distância sazonal do líder não está definida, como o texto já afirmava. Nenhuma das 680 séries da execução representativa tinha troca sem `gap` definido; a mudança não altera os resultados relatados.

## DATASET CARD

Verificação nos 23.994 artefatos das 10 execuções:

- todas as configurações têm `no_seed_attempts=True`;
- `phase2.baselines` está vazio em todos os artefatos;
- `_build_strategy_priors` roda `run_phase2` sem semear tentativas, portanto `strategy_prior` fica vazio;
- `build_dataset_card` retorna `None` quando o prior é vazio.

Conclusão: o DATASET CARD não foi emitido nas execuções relatadas, embora `dataset_card=True` no código. O texto da dissertação não menciona o cartão; a descrição fica restrita ao contexto específico da série.

## Prompt e decisão final

O prompt em `orchestrator_react/prompts.py` afirma que “the strategy applied at the end is the best-scoring one of the whole history” e que testar uma hipótese “cannot make the final validation score worse”. O bloco `leader_diagnostics`, em `orchestrator_react/react_loop.py`, repete a ideia no campo `note`: “the final history will take the best validation score”. Com `final_strategy="seasonal_guard"`, a condição sazonal pode substituir o líder por um candidato com pontuação de validação pior. Na execução representativa, houve 48 trocas; em todas, o `gap` sazonal estava definido.

Decisão: o prompt das 10 execuções foi mantido para preservar a reprodutibilidade. O texto do método descreve a decisão real; as considerações finais registram que o agente foi instruído de que a decisão aplicaria a melhor validação e que a condição sazonal, aplicada depois do ciclo, pode substituí-la. Se o prompt for corrigido no futuro, as 10 execuções precisam ser repetidas para que o repositório corresponda ao prompt executado.

## Pendências de compilação

- Não há `pdflatex` local. A compilação e a quebra das tabelas novas precisam ser conferidas no Overleaf.
- A figura usa `images/proposta/carte_arquitetura.png`; o arquivo local equivalente é `crest_architeture_final.png`. Conferir o caminho e a extensão no projeto do Overleaf.
