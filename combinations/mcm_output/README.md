# Multi-Comparison Matrix — Guia de Uso

Gerado por `combinations/multi_comparison_matrix.py` usando a biblioteca
[multi-comp-matrix](https://github.com/MSD-IRIMAS/Multi_Comparison_Matrix).

---

## Como rodar

```bash
# Ativar o ambiente e rodar o script padrão
conda run -n agno python multi_comparison_matrix.py
```

O script padrão roda:
1. Uma MCM individual para cada dataset em `DATASETS`
2. Uma MCM combinando todos os datasets juntos

Os resultados ficam em subpastas dentro de `mcm_output/`:
- `mcm_output/ETTH1/` — individual ETTH1
- `mcm_output/ETTH2/` — individual ETTH2
- `mcm_output/combined_ETTH1_ETTH2/` — análise combinada

Cada pasta contém `.png`, `.pdf`, `.csv` e `analysis.json`.

---

## Como adicionar outros métodos

Abra `multi_comparison_matrix.py` e edite o dicionário `METHODS`:

```python
METHODS = {
    "dba":          "/caminho/para/resultados/dba",
    "mean":         "/caminho/para/resultados/mean",
    "median":       "/caminho/para/resultados/median",
    "orchestrator": "/caminho/para/resultados/orchestrator_llm_v1_pattern",
    # Adicione seu novo método aqui:
    "meu_modelo":   "/caminho/para/resultados/meu_modelo",
}
```

**Requisito:** cada pasta deve conter um arquivo `<DATASET_NAME>.csv`
separado por `;` com pelo menos as colunas `dataset_index` e `smape`.

---

## Como adicionar outros datasets

Edite a lista `DATASETS`:

```python
DATASETS = ["ETTH1", "ETTH2", "NN5_WEEKLY_DATASET", "US_BIRTHS_DATASET"]
```

---

## Como usar as funções diretamente (sem editar o script)

```python
from multi_comparison_matrix import run, run_combined

# MCM para um único dataset
run("ETTH1")

# MCM para dois datasets separados
run("ETTH1")
run("ETTH2")

# MCM combinando múltiplos datasets em uma única análise
run_combined(["ETTH1", "ETTH2"])

# Combinado com label personalizado
run_combined(["ETTH1", "ETTH2"], label="minha_analise")

# Usando métodos customizados sem alterar o arquivo
meus_metodos = {
    "modelo_a": "/path/a",
    "modelo_b": "/path/b",
}
run_combined(["ETTH1", "ETTH2"], methods=meus_metodos)
run("ETTH1", methods=meus_metodos)
```

---

## Estrutura esperada dos CSVs

O arquivo deve ser separado por `;` e ter ao menos:

| dataset_index | smape  | (outras colunas são ignoradas) |
|---------------|--------|-------------------------------|
| 0             | 0.1285 | ...                           |
| 1             | 0.1394 | ...                           |

Se houver linhas duplicadas para o mesmo `dataset_index`, só a primeira é usada.

---

## Interpretação da matriz

- **Azul**: método na linha é significativamente **melhor** que o da coluna
- **Vermelho**: método na linha é significativamente **pior** que o da coluna
- **Cinza**: sem diferença estatisticamente significativa (p ≥ 0.05)
- **Negrito**: p-value < 0.05 (diferença significativa)
- Cada célula mostra: diferença média de SMAPE / win-tie-loss / probabilidades bayesianas / p-value
- Como a métrica é SMAPE (menor = melhor), valores negativos na diagonal azul indicam que a linha tem SMAPE menor

---

## Instalação das dependências

```bash
conda run -n agno pip install multi-comp-matrix --no-deps
conda run -n agno pip install scipy matplotlib tqdm baycomp pandas
```
