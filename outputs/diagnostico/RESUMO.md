# Resumo -- previsões divergentes (catboost/rf)

Fatos apurados nos passos 1–5 (`outputs/diagnostico/passo{1,2,3,4}_*`, `outputs/resultados/diagnostico/passo5_*`). Sem recomendação de tratamento.

## 1. É a previsão em si, ou artefato do cálculo da métrica?

É a previsão em si. Nos 8 casos investigados no Passo 1 (união entre o top-5 real por RMSE médio — que é 100% `M4_WEEKLY_DATASET × {ONLY_DWT_catboost, FT_catboost, CWT_catboost, ONLY_CWT_catboost, DWT_catboost}`, RMSE de 1e118 a 1e136 — e os 5 pares citados no prompt), o `test` de cada série permanece em ordem de grandeza normal (ex.: `M4_WEEKLY_DATASET` idx 208, `test` entre 3944 e 4185; `ETTH1` idx 3, `test` entre 1.5 e 4.3) enquanto `predictions` diverge em 11 a 133 ordens de grandeza (`log10(max|pred|/max|test|)` de 11.3 a 132.5 nos 8 casos, todos > 11).

No caso mais extremo (`M4_WEEKLY_DATASET`, `ONLY_DWT_catboost`, idx 208), os 13 valores de `predictions` alternam de sinal e crescem geometricamente passo a passo: `3.3e12, -1.4e23, 8.7e33, -3.9e43, 2.4e54, -1.1e64, 6.6e74, -3.0e84, 1.8e95, -8.2e104, 5.0e115, -2.3e125, 1.4e136` — cada passo ~10 ordens de grandeza maior que o anterior, com sinal invertido. Em `ETTH1 × catboost` (idx 3, horizonte 24), o padrão é parecido mas mais gradual: os primeiros ~17 passos ficam em 1e8–1e10, e a partir do passo 18 saltam para 1e13–1e14. Ambos os padrões (crescimento geométrico ao longo do horizonte, por vezes com sinal alternado) são a assinatura de um mecanismo de prev-multi-step que realimenta sua própria saída a cada passo — não de um erro de fórmula de métrica (que produziria previsões na mesma ordem de grandeza do `test`, não 10¹¹–10¹³³ vezes maior).

Arquivo: `outputs/diagnostico/passo1_exemplos_brutos.json` (8 casos, `test`/`predictions` completos).

## 2. Quantas séries são afetadas, no total e por dataset?

Usando o critério `max(abs(predictions)) > 100 × max(abs(test))` (Passo 2, 98 combinações `dataset × regressor` entre os 14 regressores das duas famílias):

| dataset | catboost family (7 regressors) | rf family (7 regressors) |
|---|---|---|
| ANP_MONTHLY | 0% em todas | 0% em todas |
| NN5_WEEKLY_DATASET | 0% em todas | 0% em todas |
| M4_WEEKLY_DATASET | 0.28%–0.84% (1–3 de 359 séries) | 0.28%–1.11% (1–4 de 359 séries) |
| ETTH1 | 100% (`catboost`), 0% nas 6 variantes com transformação | 0% em todas |
| ETTH2 | 85.7% (`catboost`), 0% nas 6 variantes com transformação | 0% em todas |
| ETTM1 | 100% (`catboost`), 0% nas 6 variantes com transformação | 0% em todas |
| ETTM2 | 85.7% (`catboost`), 0% nas 6 variantes com transformação | 0% em todas |

Em número de séries únicas (Passo 5, `dataset_index` distintos afetados por pelo menos um regressor): ETTH1 7/7, ETTM1 7/7, ETTH2 6/7, ETTM2 6/7, `M4_WEEKLY_DATASET` apenas 4/359 (`dataset_index` 208, 9, 217, 75). `ANP_MONTHLY` e `NN5_WEEKLY_DATASET`: 0 séries afetadas, em qualquer regressor das duas famílias.

Em `M4_WEEKLY_DATASET`, a série `dataset_index=208` diverge em 14 dos 14 regressores testados (todas as variantes catboost e rf); `dataset_index=9` diverge em 8 dos 14; `dataset_index=75` e `217` só divergem nas variantes `DWT_rf`/`ONLY_DWT_rf`/`ONLY_DWT_catboost` (2 cada).

Arquivos: `outputs/diagnostico/passo2_contagem_divergencia.csv` (98 linhas), `outputs/resultados/diagnostico/passo5_contexto_divergencia.csv` (52 linhas, uma por `dataset, regressor, dataset_index` divergente).

## 3. É específico do CatBoost, ou também afeta o RF?

Depende do dataset — não é sistemático (Passo 3):

- `ANP_MONTHLY`, `NN5_WEEKLY_DATASET`: 0% para ambas as famílias (sem sinal).
- `M4_WEEKLY_DATASET`: catboost 0.557% médio, rf 0.478% médio — mesma ordem de grandeza, RF não fica atrás.
- `ETTH1`/`ETTM1`: catboost 14.29% médio (média das 7 variantes, puxada pelo `catboost` puro a 100%) vs. rf 0.00% médio.
- `ETTH2`/`ETTM2`: catboost 12.24% médio vs. rf 0.00% médio.

Ou seja: nos 4 datasets ETT, a divergência é exclusiva do CatBoost (RF: 0% em toda variante, em toda série). Em `M4_WEEKLY_DATASET`, as duas famílias divergem em taxa comparável — ali não é um problema específico do CatBoost.

Arquivo: `outputs/diagnostico/passo3_catboost_vs_rf.csv`.

## 4. A transformação (wavelet/Fourier) piora, cria, ou é indiferente ao problema?

O efeito tem sinal oposto dependendo do dataset (Passo 4):

- Nos 4 datasets ETT: a transformação **elimina** o problema do CatBoost. `catboost` (original, sem transformação) diverge em 85.7–100% das séries; todas as 6 variantes com transformação (`CWT_/DWT_/FT_catboost` e as 3 `ONLY_*`) divergem em 0%. Para RF, indiferente (0% nas 3 representações, em todo dataset ETT).
- Em `M4_WEEKLY_DATASET`: a transformação **aumenta ligeiramente** a taxa, para as duas famílias — catboost sobe de 0.279% (original) para 0.557% (concatenada) e 0.650% (only); rf sobe de 0.279% (original) para 0.557% (concatenada) e 0.464% (only). Em valor absoluto o aumento é pequeno (< 0.4 ponto percentual).
- Em `ANP_MONTHLY`/`NN5_WEEKLY_DATASET`: indiferente, 0% nas 3 representações, nas 2 famílias.

Arquivo: `outputs/diagnostico/passo4_efeito_transformacao.csv`.

## 5. Há relação aparente com séries curtas ou com valores pequenos?

Nenhuma das duas explica o padrão observado (Passo 5):

- **Comprimento**: as séries onde `catboost` diverge (as 4 ETT) têm comprimento total 17420 (ETTH1/ETTH2) ou 69680 pontos (ETTM1/ETTM2) — as mais LONGAS de todo o estudo, não as mais curtas. `ANP_MONTHLY` (419 pontos) e `NN5_WEEKLY_DATASET` (113 pontos), que são bem mais curtas, têm 0% de divergência. Em `M4_WEEKLY_DATASET` (comprimento variável, 93–2610), as 4 séries divergentes têm comprimento 947, 947, 470 e 260 — perto da mediana (947) ou abaixo dela, não nos extremos curtos (mínimo do dataset é 93).
- **Valores pequenos**: das 52 linhas divergentes do Passo 5, `tem_valor_pequeno` é `True` em apenas 5 (9.6%) e `False` em 47 (90.4%). Em `M4_WEEKLY_DATASET`, nenhuma das 4 séries divergentes tem `tem_valor_pequeno = True`.

Arquivo: `outputs/resultados/diagnostico/passo5_contexto_divergencia.csv`.

---

**Nota metodológica sobre o Passo 1**: o prompt pede "os cinco pares com maior RMSE médio" e também lista 5 pares nominais como piso mínimo ("pelo menos"). Os dois critérios divergem — o top-5 real por RMSE médio é inteiramente `M4_WEEKLY_DATASET × variantes catboost` (RMSE 1e118–1e136); os pares `ETTH1/ETTM1/ETTM2 × catboost` citados no prompt nem entram no top-10 por RMSE médio (ficam na casa de 1e12–1e14). Os 8 casos investigados são a união dos dois critérios (ver `top5_rmse_medio_real` e as flags `in_top5_rmse_medio_real`/`in_lista_nomeada_no_prompt` em `passo1_exemplos_brutos.json`).
