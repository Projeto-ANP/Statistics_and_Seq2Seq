# Mudança 2 — `sanity_check_trigger`: NÃO implementada (sem gatilho sensato)

Conforme o pedido, li `tools.py` antes e concluo que não há gatilho objetivo que faça sentido; por isso a flag **não foi criada**.

## O que `sanity_check` faz de fato (`tools.py:775`)
- Assinatura: `sanity_check(state, reference)`; `reference` é um id de tentativa (`"a3"`) ou uma spec de estratégia.
- Calcula o **forecast de teste** da estratégia (`state.apply_to_test(spec)`) e o compara com a série de treino: mínimo/máximo históricos e uma faixa `mediana ± tol·desvio` (`config.sanity_check_tolerance = 3.0`).
- Retorna `n_points`, `forecast_range`, `historical_range`, `extrapolates_history`, `points_outside_history`, `points_outside_band`, `warnings` e `ok`. Avisa para pontos não finitos ou fora da faixa. "Extrapolar o histórico" é só informação (normal em série com tendência). **Não bloqueia nada.**

## Por que não dá um gatilho
1. **O resultado nunca altera a estratégia aplicada.** O loop aplica sempre a melhor tentativa por score de validação (`react_loop.py`, "principle 5": `best = state.best_attempt()`, com `overridden` se o agente aceitou outra; `final_strategy=argmin`). Mesmo que `sanity_check` acuse um problema na líder, qualquer alternativa que o agente proponha só vira a aplicada se pontuar melhor na validação. O resultado da checagem não entra nessa escolha.
2. **Já roda sempre, depois do loop.** `pipeline.py:538` chama `sanity_check` na estratégia final de toda série (Fase 4) e grava em `warnings`/artefato. Pedir ao agente que o faça antes do `accept` duplica isso e gasta uma iteração (o `accept` do gpt-oss cai quase sempre na iteração 12, então não sobra iteração para checar e depois aceitar).
3. **O agente não vê o que a checagem examina.** O gatilho teria de sair do que ele já vê (cards, histórico). O forecast de teste e a faixa histórica só aparecem dentro do próprio `sanity_check`.
4. **Na prática, quase nunca há o que acusar.** Nas execuções gravadas: `sanity_check` da Fase 4 deu `ok=False` em 48 de 498 séries do gpt-oss (sem ANP, cujos artefatos são do qwen) — 47 delas em M4 e 1 em ETTH1; qwen: 49 de 498, 48 em M4. Nenhuma ação do agente foi possível sobre isso. O agente nunca chamou `sanity_check` em nenhuma das 680 séries (`catalogo_uso.py`).

## O que teria efeito (fora do escopo pedido)
Um gatilho com poder de decisão seria determinístico e fora do prompt: por exemplo, se a Fase 4 acusa `ok=False` na estratégia final, cair para a melhor semente com `ok=True`. Isso muda o critério de seleção (deixa de ser argmin puro da validação) e precisa ser desenhado e medido como mudança separada. Se quiser, implemento como flag própria.
