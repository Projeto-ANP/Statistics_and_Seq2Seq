# Combinação FFORMA — `combinations/fforma.py`

Implementação do algoritmo **FFORMA (Feature-based Forecast Model Averaging)**
adaptada para funcionar com as previsões já geradas pelos modelos base.

Referência: Montero-Manso et al. (2020) — <https://robjhyndman.com/publications/fforma/>

---

## O que é FFORMA

FFORMA combina previsões de vários modelos usando **pesos aprendidos** a partir de
**características da série temporal** (tsfeatures). Em vez de uma média fixa, o método:

1. Extrai as features estatísticas de cada série (autocorrelação, sazonalidade,
   hurst, unitroot, etc.) via a biblioteca `tsfeatures`.
2. Calcula o erro de cada modelo base numa janela de **validação** (nunca teste).
3. Treina um **meta-learner XGBoost** que aprende a prever qual modelo terá
   menor erro dado as features da série.
4. Usa `softmax(raw_score)` como pesos → aplica sobre as previsões de teste.

Com poucas séries (<10), o meta-learner não tem dados suficientes para generalizar;
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

Primeira vez na máquina:

```bash
conda env create -f fforma_environment.yml
```

Depois, a partir da pasta `Statistics_and_Seq2Seq/`:

```bash
conda activate fforma-combinations
python -m combinations.fforma --dataset ANP_MONTHLY
```

Não há mais bloco `__main__` com dataset hardcoded: frequência, sazonalidade e
horizonte vêm de `combinations/dataset_specs.py` (o horizonte é lido direto da
coluna `horizon` do CSV). Rodar um dataset novo é só trocar `--dataset`.

### Flags

| flag | default | para quê |
|------|---------|----------|
| `--dataset` | (obrigatória) | `ANP_MONTHLY`, `M4_WEEKLY_DATASET`, `US_BIRTHS_DATASET`, … |
| `--models` | os 19 modelos base | lista de modelos base |
| `--exp-name` | `FFORMA` | subpasta de saída em `resultados/` |
| `--seasonality` | de `dataset_specs.py` | período sazonal do `tsfeatures` |
| `--horizon` | lido do CSV | sobrescreve o horizonte |
| `--n-estimators` | `100` | rodadas de boosting |
| `--force-softmax` | desligado | pula o meta-learner |
| `--smape-threshold` | `1.5` | no fallback, zera modelos acima disso |
| `--resume` | desligado | continua de onde parou em vez de apagar a saída |

**Sobre `--resume`:** `aux.save_to_csv` faz *append*. Sem `--resume`, o script
apaga o CSV de saída antes de começar — de propósito, porque rerodar sem apagar
duplica as linhas e o `drop_duplicates(keep="first")` do MCM passa a ler as
linhas **antigas**, ou seja, o resultado novo nunca apareceria na comparação.

### API programática

```python
from combinations.fforma import fforma_combination

fforma_combination(
    dataset_name="ETTH1",
    models=["ARIMA", "ETS", "rf", "catboost"],  # default: os 19
    exp_name="FFORMA",   # subpasta de saída em resultados/
)
```

---

## Parâmetros da função `fforma_combination`

| Parâmetro        | Tipo            | Default    | Descrição |
|------------------|-----------------|------------|-----------|
| `dataset_name`   | `str`           | —          | Nome do dataset (ex: `'ETTH1'`) |
| `models`         | `list[str]\|None` | `None`   | Modelos base; `None` = os 19 de `DEFAULT_MODELS`. Cada um precisa ter `resultados/<M>/normal/<D>.csv` |
| `exp_name`       | `str`           | `"FFORMA"` | Nome da pasta de saída em `resultados/` |
| `seasonality`    | `int\|None`     | `None`     | `None` = usa `dataset_specs.py` (24 = horário, 12 = mensal, 7 = diário com sazonalidade semanal, 1 = não-sazonal) |
| `horizon`        | `int\|None`     | `None`     | `None` = lê da coluna `horizon` do CSV |
| `lgb_params`     | `dict\|None`    | `None`     | Parâmetros extras para o XGBoost (só usado se ≥10 séries) |
| `n_estimators`   | `int`           | `100`      | Rodadas de boosting |
| `force_softmax`  | `bool`          | `False`    | Se `True`, ignora o meta-learner e usa softmax dos erros |
| `smape_threshold`| `float`         | `1.5`      | Modelos com SMAPE de validação ≥ este valor recebem peso zero. Evita que previsões numericamente absurdas contaminem o ensemble. |
| `resume`         | `bool`          | `False`    | Continua de onde parou em vez de apagar a saída |

---

## Testando outras configurações

```bash
# Outro dataset — sazonalidade e horizonte saem de dataset_specs.py / do CSV
python -m combinations.fforma --dataset ANP_MONTHLY --exp-name FFORMA_anp

# Subconjunto de modelos base
python -m combinations.fforma --dataset ETTH1 --models ARIMA ETS rf catboost

# Só softmax dos erros (sem tsfeatures, sem meta-learner)
python -m combinations.fforma --dataset ETTH1 --force-softmax

# Threshold mais agressivo para excluir modelos ruins no fallback
python -m combinations.fforma --dataset ETTH1 --force-softmax --smape-threshold 1.0

# Mais rodadas de boosting
python -m combinations.fforma --dataset ANP_MONTHLY --n-estimators 200

# Sobrescrever a sazonalidade do registro
python -m combinations.fforma --dataset NN5_WEEKLY_DATASET --seasonality 52
```

Parâmetros extras do XGBoost não têm flag (são um dict); use a API:

```python
fforma_combination(
    dataset_name="ANP_MONTHLY",
    lgb_params={"eta": 0.05, "max_depth": 6, "subsample": 0.8},
    n_estimators=200,
)
```

### Datasets registrados

`combinations/dataset_specs.py` guarda a sazonalidade de cada dataset — a única
coisa que é escolha do pesquisador. Frequência é conferida contra os timestamps
reais e o horizonte é lido da coluna `horizon` do CSV.

| dataset | freq | seasonality | séries |
|---|---|---|---|
| `ETTH1` / `ETTH2` | `h` | 24 | 7 |
| `ETTM1` / `ETTM2` | `30min` | 48 | 7 |
| `ANP_MONTHLY` | `ME` | 12 | 182 |
| `NN5_WEEKLY_DATASET` | `W` | 7 † | 111 |
| `M4_WEEKLY_DATASET` | `7D` | 1 ‡ | 359 |
| `US_BIRTHS_DATASET` | `D` | 7 | 1 |

† Para dado semanal o período natural seria 52 (anual) ou 1 (não-sazonal); o 7
vem de uma confusão com a versão **diária** do NN5. Está mantido porque é o
valor com que os resultados já publicados (`mcm_output_v5`) foram gerados —
trocar exige regerar o FFORMA e refazer as comparações.

‡ A competição M4 trata as séries semanais como não-sazonais (Makridakis et al.
2020).

Um dataset não registrado **já roda**: `resolve_spec` mede o passo entre os
timestamps, infere a frequência e imprime o que inferiu.

> **Atenção com `US_BIRTHS_DATASET`:** tem **1 série só**. Abaixo de 10 séries o
> meta-learner não tem amostra para aprender nada entre séries, então o script
> cai automaticamente no softmax dos erros de validação. O resultado é válido,
> mas **não é FFORMA de verdade** — é ponderação inversa ao erro. Vale dizer isso
> explicitamente se esse dataset entrar na comparação do paper.

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
7. tsfeatures(series_df, freq=...)  → features por série (42 com sazonalidade,
                                       37 quando seasonality=1)
8. meta-learner:
     se n_series >= 10 e não force_softmax:
       _train_fforma_booster()      → XGBoost com objetivo FFORMA
       _compute_weights_booster()   → softmax(raw_scores) por série
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
passado ao XGBoost via `obj`.

Com poucas séries, esse objetivo mais complexo não estabiliza — daí o fallback
para softmax direto dos erros, que é matematicamente equivalente ao "1 rodada
de FFORMA" sem aprendizado de features.

---

## Instalação das dependências

```bash
conda env create -f fforma_environment.yml
conda activate fforma-combinations
```

Env separado do `ade-combinations`: o FFORMA não precisa de torch/lightning, e
o `tsfeatures` exige statsmodels/numba, que o ADE não usa.

O yml pina `xgboost==2.1.4`: a partir da 2.1 o XGBoost entrega `predt` com
shape `(n_amostras, n_classes)` para objetivos customizados, que é o formato que
`_fforma_objective` espera. Versões anteriores entregam achatado e o gradiente
sai errado **sem levantar erro**.

### Reprodutibilidade

Rodar duas vezes no mesmo env dá resultado bit-a-bit idêntico (verificado no
ANP: `max |diff| = 0.0`).

Os resultados publicados em `resultados/FFORMA/` foram gerados num env `fforma`
antigo que não existe mais, com outra versão de XGBoost. Regerando com o env
pinado, o ANP dá SMAPE médio **0.216600** contra **0.216593** publicado
(diferença de 0.003%, máximo de 0.00196 numa série). A lógica é a mesma — só a
versão do booster mudou. Para ter todos os números vindos de um ambiente
reprodutível, regere a baseline:

```bash
python -m combinations.fforma --dataset ANP_MONTHLY
```
