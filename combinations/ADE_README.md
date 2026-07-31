# Combinação ADE — `combinations/ade.py`

Script para gerar combinações de previsões usando **ADE (Arbitrated Dynamic
Ensemble)** da biblioteca [`metaforecast`](https://pypi.org/project/metaforecast/).

Mantém a mesma estrutura de chamada de `combinations/mean.py` e `combinations/dba.py`:
você configura a lista de **modelos base** e o **dataset**, e o script gera
arquivos `resultados/ADE/<DATASET>.csv` com as previsões combinadas para cada
série (`dataset_index`).

---

## O que é ADE

ADE combina previsões de vários modelos especialistas usando pesos **dinâmicos**:
em vez de tirar média (`mean`) ou usar centroide DTW (`dba`), ele aprende um
**meta-modelo** que prevê qual especialista tende a errar mais em cada ponto, e
penaliza os ruins. Em alto nível:

1. Para cada modelo base, monta-se uma série de erros sobre janelas históricas
   (validação) — esse é o "histórico de desempenho" do modelo.
2. Um meta-regressor (`LightGBM` por padrão) aprende a prever o erro futuro de
   cada modelo a partir de **lags** dessa série de erros (`meta_lags`).
3. No momento da previsão, os pesos são `softmax(-erro_previsto)`: modelos cujo
   erro esperado é menor ganham mais peso.

Parâmetro útil: `trim_ratio < 1` corta os piores especialistas antes do
ponderamento (default 1.0 = mantém todos).

---

## Como rodar

Primeira vez na máquina:

```bash
conda env create -f ade_environment.yml
```

Depois, a partir da pasta `Statistics_and_Seq2Seq/`:

```bash
conda activate ade-combinations
python -m combinations.ade --dataset ANP_MONTHLY
```

Não há mais bloco `__main__` com dataset hardcoded: frequência, sazonalidade e
horizonte vêm de `combinations/dataset_specs.py` (e o horizonte é lido direto da
coluna `horizon` do CSV). Rodar um dataset novo é só trocar `--dataset`.

### Flags

| flag | default | para quê |
|------|---------|----------|
| `--dataset` | (obrigatória) | `ANP_MONTHLY`, `M4_WEEKLY_DATASET`, `US_BIRTHS_DATASET`, … |
| `--models` | os 19 modelos base | lista de modelos base |
| `--exp-name` | `ADE` | subpasta de saída em `resultados/` |
| `--trim-ratio` | `1.0` | fração dos melhores especialistas a manter |
| `--horizon` | lido do CSV | sobrescreve o horizonte |
| `--resume` | desligado | continua de onde parou em vez de apagar a saída |
| `--on-error` | `raise` | `skip` para pular séries que falharem |

Rodadas longas (M4 semanal tem 359 séries, ~50 s no total):

```bash
nohup python -m combinations.ade --dataset M4_WEEKLY_DATASET > logs/ade_m4.log 2>&1 &
```

**Sobre `--resume`:** `aux.save_to_csv` faz *append*. Sem `--resume`, o script
apaga o CSV de saída antes de começar — de propósito, porque rerodar sem apagar
duplica as linhas e o `drop_duplicates(keep="first")` do MCM passa a ler as
linhas **antigas**, ou seja, o resultado novo nunca apareceria na comparação.

### API programática

```python
from combinations.ade import ade_combination

ade_combination(
    dataset_name="ETTH1",
    models=["ARIMA", "ETS", "THETA", "rf", "catboost"],  # default: os 19
    exp_name="ADE",     # subpasta de saída em resultados/
    trim_ratio=1.0,     # 1.0 = usa todos; 0.5 = só metade superior
)
```

---

## Frequências aceitas

`dataset_specs.py` guarda dois campos de frequência por dataset porque são
vocabulários diferentes: `freq` é o alias pandas usado para reconstruir o eixo
temporal (aliases modernos — `h`, `ME`; `H` e `M` foram removidos no pandas 3),
e `freq_ade` é a chave que o ADE aceita. Meia-hora não existe na tabela do ADE,
então `ETTM1/ETTM2` mapeiam para `H` e o script avisa no console.

ADE só aceita estas chaves (definidas em `metaforecast.ensembles.ADE.WINDOW_SIZE_BY_FREQ`):

| freq | Significado    | Window default ADE |
|------|----------------|--------------------|
| H    | horária        | 48                 |
| D    | diária         | 14                 |
| W    | semanal        | 16                 |
| M    | mensal         | 12                 |
| ME   | mês final      | 12                 |
| MS   | mês início     | 12                 |
| Q    | trimestral     | 4                  |
| QS   | trimestre início | 4                |
| Y    | anual          | 6                  |

Letras minúsculas (`h`, `d` etc.) são aceitas e convertidas internamente.

---

## Requisitos sobre os CSVs de entrada

Cada modelo precisa ter um arquivo em:

```
./timeseries/mestrado/resultados/<MODEL>/normal/<DATASET>.csv
```

Separado por `;` e contendo no mínimo as colunas:

```
dataset_index ; horizon ; regressor ; smape ; ... ; test ; predictions ; start_test ; final_test
```

E precisa de pelo menos **2 janelas** por `dataset_index` (uma de treino/validação
e uma de teste). Recomenda-se ter mais janelas de validação (3+) para que o
meta-modelo tenha histórico para aprender.

**Validações automáticas que o script faz:**

1. Se algum modelo não tem o CSV do dataset, levanta `FileNotFoundError` com
   o caminho exato que está faltando.
2. Todos os modelos precisam ter o **mesmo conjunto de `dataset_index`** e o
   **mesmo número de janelas** por série. Se um modelo tem 4 janelas e outro
   tem 3, o script aborta.
3. **Não** exige que as datas (`start_test`) sejam idênticas entre modelos —
   por exemplo, modelos baseados em wavelet às vezes têm splits temporais
   diferentes. As janelas são alinhadas **posicionalmente** (mais antiga → mais
   recente) usando as datas do primeiro modelo da lista como referência canônica.

---

## O que sai

Um arquivo `resultados/<exp_name>/<DATASET>.csv` (ex: `resultados/ADE/ETTH1.csv`)
no mesmo formato dos outros métodos (`mean`, `dba`, `median`), com uma linha
por `dataset_index`. Colunas: `dataset_index, horizon, regressor, mape, pocid,
smape, rmse, msmape, mae, test, predictions, start_test, final_test`.

As métricas (`smape`, `mape`, `rmse`, …) são calculadas comparando as previsões
do ADE contra os valores reais (coluna `test` do **modelo de referência**, que
é o primeiro da lista `models`).

---

## Fluxo interno (resumo)

1. **`validate_models_have_dataset`** — checa que todo modelo tem o CSV. Erro
   se faltar.
2. **`_read_model_df`** — carrega cada CSV, converte datas, ordena por
   `(dataset_index, start_test)`.
3. **`_check_windows_alignment`** — confere consistência (#janelas e
   #dataset_index iguais entre modelos).
4. **`_build_long_dataframes`** — transforma os dados em dois DataFrames no
   formato esperado pelo `metaforecast`:
   - `combined_df`: linhas de treino (`unique_id, ds, y, MODEL_1, …, MODEL_N`)
     com as previsões dos modelos sobre janelas históricas
   - `df_predictions`: linhas do horizonte de teste (sem `y`)
5. **`_ade_predict_one_series`** — para cada série, instancia `ADE(...)`,
   ajusta `meta_lags` ao tamanho disponível de treino, faz `fit` no histórico
   e `predict` no horizonte de teste.
6. **`aux.save_to_csv`** — calcula métricas e grava a linha de resultado no
   arquivo do experimento.

---

## Trocando para outros datasets/modelos

```bash
python -m combinations.ade --dataset ANP_MONTHLY
python -m combinations.ade --dataset M4_WEEKLY_DATASET
python -m combinations.ade --dataset US_BIRTHS_DATASET
python -m combinations.ade --dataset ETTH1 --models ARIMA ETS rf catboost --exp-name ADE_4models
```

### Adicionando um dataset que ainda não existe

Um dataset não registrado **já roda**: `resolve_spec` mede o passo entre os
timestamps e infere a frequência, imprimindo o que inferiu. Só a sazonalidade
(usada pelo FFORMA, não pelo ADE) é palpite. Para fixar, adicione uma linha em
`DATASET_SPECS` (`combinations/dataset_specs.py`):

```python
"MEU_DATASET": DatasetSpec("MEU_DATASET", freq="D", freq_ade="D", seasonality=7),
```

O horizonte nunca precisa ser declarado — sai da coluna `horizon` do CSV.
Se a frequência declarada não bater com os timestamps reais, `resolve_spec`
aborta com a diferença explícita (foi assim que o problema de resolução do ETTM
teria sido pego).

---

## Instalação (uma vez)

```bash
conda env create -f ade_environment.yml
```

`metaforecast` puxa `lightgbm`, `neuralforecast`, `pytorch-lightning` e `torch`
— por isso o env é separado do `agno`.

O yml pina `setuptools<81` de propósito: a partir da 81 o `pkg_resources` foi
removido, e o `lightning_fabric` ainda chama `pkg_resources.declare_namespace()`
no import. Sem o pin, `from metaforecast.ensembles import ADE` quebra com
`ModuleNotFoundError: No module named 'pkg_resources'`.

---

## Troubleshooting

- **`FileNotFoundError: Os modelos abaixo não possuem resultados...`** — você
  passou um nome de modelo que não tem CSV. Confira a lista em `resultados/`.
- **`ValueError: Modelo 'X' tem N janelas para dataset_index=K, mas ...`** —
  algum modelo tem um número de linhas diferente para alguma série. Reexecute
  o experimento que gerou o CSV ou remova o modelo da lista.
- **`KeyError: 'h'` no ADE** — você passou uma freq não suportada.
  O script já converte `'h'` → `'H'`, mas se passar algo exótico, o erro vem
  da `metaforecast`. Use uma freq da tabela acima.
- **SMAPE muito alto** — pode indicar que algum modelo base tem previsões
  absurdamente grandes (overflow numérico). Inspecione com `mean`/`dba`
  primeiro: se eles dão SMAPE ≈ 2.0, há um modelo base ruim na lista.
