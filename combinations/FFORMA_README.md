# Combinação FFORMA — `combinations/fforma.py`

Implementação do algoritmo **FFORMA (Feature-based Forecast Model Averaging)**
adaptada para funcionar com as previsões já geradas pelos modelos base.

Referência: Montero-Manso et al. (2020) — <https://robjhyndman.com/publications/fforma/>

---

## O que é FFORMA

FFORMA combina previsões de vários modelos usando **pesos aprendidos** a partir de
**características da série temporal** (tsfeatures). Em vez de uma média fixa, o método:

1. Extrai ~42 features estatísticas de cada série (autocorrelação, sazonalidade,
   hurst, unitroot, etc.) via a biblioteca `tsfeatures`.
2. Calcula o erro de cada modelo base numa janela de **validação** (nunca teste).
3. Treina um **meta-learner LightGBM** que aprende a prever qual modelo terá
   menor erro dado as features da série.
4. Usa `softmax(raw_score_LGB)` como pesos → aplica sobre as previsões de teste.

Com poucas séries (<10), o LightGBM não tem dados suficientes para generalizar;
o script detecta isso e usa **softmax direto dos erros de validação** como fallback
(modelos com menor erro histórico recebem maior peso — simples e robusto).

---

## Separação validação / teste (sem leakage)

Cada modelo base tem N janelas por série (`dataset_index`) em seu CSV:

```
janela 0 (mais antiga)  → validação
janela 1                → validação
janela 2                → validação
...
janela N-1 (mais recente) → TESTE  ← nunca usado no treinamento
```

A janela de teste entra **somente** para calcular as métricas finais
(SMAPE, RMSE, MAE…). Os valores reais (`test` col) do teste não são
vistos pelo meta-learner.

---

## Como rodar

```bash
cd Statistics_and_Seq2Seq
conda run -n agno python -m combinations.fforma
```

O bloco `if __name__ == "__main__":` no fim do arquivo define os modelos,
dataset, sazonalidade e horizonte padrão (ETTH1, seasonality=24, horizon=24).

### API programática

```python
from combinations.fforma import fforma_combination

fforma_combination(
    models=["ARIMA", "ETS", "rf", "catboost", "median"],
    dataset_name="ETTH1",
    seasonality=24,      # período sazonal para tsfeatures (24h → diário)
    horizon=24,
    exp_name="FFORMA",   # subpasta de saída em resultados/
)
```

---

## Parâmetros da função `fforma_combination`

| Parâmetro        | Tipo            | Default    | Descrição |
|------------------|-----------------|------------|-----------|
| `models`         | `list[str]`     | —          | Modelos base. Cada um precisa ter `resultados/<M>/normal/<D>.csv` |
| `dataset_name`   | `str`           | —          | Nome do dataset (ex: `'ETTH1'`) |
| `seasonality`    | `int`           | —          | Período sazonal para tsfeatures (24 = horário, 12 = mensal, 7 = diário com sazonalidade semanal) |
| `horizon`        | `int`           | —          | Horizonte de previsão |
| `exp_name`       | `str`           | `"FFORMA"` | Nome da pasta de saída em `resultados/` |
| `lgb_params`     | `dict\|None`   | `None`     | Parâmetros extras para o LightGBM (só usado se ≥10 séries) |
| `n_estimators`   | `int`           | `100`      | Número de árvores LightGBM |
| `force_softmax`  | `bool`          | `False`    | Se `True`, ignora o LightGBM e usa softmax dos erros mesmo com muitas séries |
| `smape_threshold`| `float`         | `1.5`      | Modelos com SMAPE de validação ≥ este valor recebem peso zero. Evita que previsões numericamente absurdas contaminem o ensemble. |

---

## Testando outras configurações

### Outro dataset (mensal)

```python
fforma_combination(
    models=["ARIMA", "ETS", "rf", "catboost"],
    dataset_name="ANP_MONTHLY",
    seasonality=12,
    horizon=12,
    exp_name="FFORMA_anp",
)
```

### Forçar o meta-learner LightGBM (mesmo com poucas séries)

```python
fforma_combination(
    models=["ARIMA", "ETS", "rf"],
    dataset_name="ETTH1",
    seasonality=24,
    horizon=24,
    exp_name="FFORMA_lgb",
    force_softmax=False,        # padrão; mas com 7 séries vai usar softmax anyway
    smape_threshold=1.5,
)
```

### Ajustar threshold de exclusão de modelos ruins

```python
# Modelos com SMAPE ≥ 1.0 (em vez de 1.5) recebem peso zero
fforma_combination(
    models=[...],
    dataset_name="ETTH1",
    seasonality=24,
    horizon=24,
    smape_threshold=1.0,
)
```

### Usar apenas softmax de erros (sem tsfeatures, sem LightGBM)

```python
fforma_combination(
    models=[...],
    dataset_name="ETTH1",
    seasonality=24,
    horizon=24,
    force_softmax=True,
)
```

### LightGBM com parâmetros customizados (quando tiver ≥10 séries)

```python
fforma_combination(
    models=[...],
    dataset_name="GRANDE_DATASET",
    seasonality=12,
    horizon=12,
    lgb_params={"eta": 0.05, "max_depth": 6, "subsample": 0.8},
    n_estimators=200,
)
```

---

## Estrutura esperada dos CSVs de entrada

```
resultados/<MODEL>/normal/<DATASET>.csv
```

Separado por `;`, colunas mínimas:

```
dataset_index ; horizon ; start_test ; final_test ; test ; predictions ; ...
```

- `test`: valores reais da janela (lista como string `[1.2, 3.4, ...]`)
- `predictions`: previsões do modelo (mesma estrutura)
- Precisa de pelo menos **2 janelas** por `dataset_index` (1 validação + 1 teste)

**Validações automáticas:**

- `FileNotFoundError` se algum modelo não tem o CSV do dataset.
- `ValueError` se modelos têm número diferente de janelas por série.

---

## Saída

`resultados/<exp_name>/<DATASET>.csv` com uma linha por `dataset_index`:

```
dataset_index ; horizon ; regressor ; mape ; pocid ; smape ; rmse ; msmape ; mae ; test ; predictions ; start_test ; final_test
```

---

## Fluxo interno detalhado

```
1. validate_models_have_dataset()   → FileNotFoundError se CSV faltando
2. _read_model_df()                 → carrega e ordena por (ds_idx, start_test)
3. _check_windows_alignment()       → mesmo nº de janelas em todos os modelos
4. _split_val_test()                → separa val (posições 0..N-2) e teste (N-1)
5. _compute_errors()                → SMAPE médio por (série, modelo) na validação
6. _build_series_df()               → valores reais de val em formato long (uid,ds,y)
7. tsfeatures(series_df, freq=...)  → ~42 features por série
8. meta-learner:
     se n_series >= 10 e não force_softmax:
       _train_lgb_fforma()          → LightGBM com objetivo FFORMA
       _compute_weights_lgb()       → softmax(raw_scores) por série
     else:
       _compute_weights_softmax()   → softmax(-SMAPE), threshold zera modelos ruins
9. para cada série:
     combined = Σ w_m × pred_m      → soma ponderada das previsões de teste
     aux.save_to_csv()              → métricas + salva em CSV
```

---

## Por que o meta-learner usa o objetivo FFORMA (não multiclasse simples)?

O objetivo FFORMA original não otimiza "acertar o modelo vencedor" (cross-entropy),
mas sim **minimizar o erro ponderado esperado**:

```
L = E[softmax(score) · erro_por_modelo]
```

Isso penaliza mais dar peso alto a modelos ruins, em vez de apenas prever o rótulo
da classe vencedora. O gradiente/hessiana personalizado (`_fforma_objective`) é
passado ao LightGBM via `fobj`.

Com poucas séries, esse objetivo mais complexo não estabiliza — daí o fallback
para softmax direto dos erros, que é matematicamente equivalente ao "1 rodada
de FFORMA" sem aprendizado de features.

---

## Instalação das dependências

```bash
conda run -n agno pip install tsfeatures lightgbm
```

`lightgbm` já deve estar instalado (`metaforecast` puxa ele). O `tsfeatures`
instala também `antropy` e `supersmoother`.
