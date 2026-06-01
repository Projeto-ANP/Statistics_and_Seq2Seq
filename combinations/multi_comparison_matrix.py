"""
conda activate metrics
cd Statistics_and_Seq2Seq
python -m combinations.multi_comparison_matrix
"""
import os
import warnings

import pandas as pd
from multi_comp_matrix import MCM

# ---------------------------------------------------------------------------
# Configuração: adicione/remova métodos e datasets aqui
# ---------------------------------------------------------------------------

BASE = "/home/anp/Documents/lucas_mestrado/Statistics_and_Seq2Seq/timeseries/mestrado/resultados"

METHODS = {
    "dba":          f"{BASE}/dba",
    "mean":         f"{BASE}/mean",
    "median":       f"{BASE}/median",
    # "orchestrator_v1": f"{BASE}/orchestrator_llm_v1_pattern",
    "orchestrator_v3": f"{BASE}/orchestrator_llm_v3_pruning",
    "FFORMA":       f"{BASE}/FFORMA",
    "ADE":        f"{BASE}/ADE",
    
}

DATASETS = ["NN5_WEEKLY_DATASET", "ETTH1", "ETTH2", "ETTM1", "ETTM2", "ANP_MONTHLY"]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcm_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def load_smape(csv_path: str, method_name: str) -> pd.Series:
    """Lê CSV de um método e retorna Series de smape por dataset_index."""
    df = pd.read_csv(csv_path, sep=";", engine="python")
    df = df[["dataset_index", "smape"]].drop_duplicates(subset="dataset_index", keep="first")
    s = df.set_index("dataset_index")["smape"]
    s.name = method_name
    return s


def build_results_df(dataset_names: list[str], methods: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Constrói DataFrame wide (linhas = séries temporais, colunas = métodos).

    Cada dataset_name contribui com suas séries, prefixadas com o nome do
    dataset para evitar colisão de índices quando combinados.

    Args:
        dataset_names: lista de nomes de dataset (ex: ['ETTH1', 'ETTH2'])
        methods: dict {nome_método: pasta}. Se None, usa METHODS global.
    """
    if methods is None:
        methods = METHODS

    frames = []
    for ds_name in dataset_names:
        series = []
        for method_name, folder in methods.items():
            csv_path = os.path.join(folder, f"{ds_name}.csv")
            s = load_smape(csv_path, method_name)
            # prefixo para identificar a origem no DataFrame combinado
            s.index = [f"{ds_name}_{i}" for i in s.index]
            series.append(s)
        df_ds = pd.concat(series, axis=1)
        frames.append(df_ds)

    df = pd.concat(frames, axis=0)
    df.index.name = "serie"
    df = df.reset_index(drop=True)
    return df


def _run_mcm(df: pd.DataFrame, save_name: str, out_prefix: str) -> None:
    """Chama MCM.compare e salva os artefatos."""
    os.makedirs(out_prefix, exist_ok=True)
    # MCM concatena output_dir + savename sem separador, então a barra é necessária
    out_dir_with_sep = out_prefix + os.sep

    print(df.to_string())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MCM.compare(
            df_results=df,
            output_dir=out_dir_with_sep,
            png_savename=save_name,
            pdf_savename=save_name,
            csv_savename=save_name,
            used_statistic="SMAPE",
            order_better="increasing",   # menor SMAPE = melhor
            include_ProbaWinTieLoss=True,
            pvalue_test="wilcoxon",
        )

    print(f"Salvo em: {out_prefix}/")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def run(dataset_name: str, methods: dict[str, str] | None = None) -> None:
    """Gera MCM para um único dataset."""
    df = build_results_df([dataset_name], methods)
    out_prefix = os.path.join(OUTPUT_DIR, dataset_name)
    print(f"\n=== {dataset_name} ===")
    _run_mcm(df, save_name=f"mcm_{dataset_name}", out_prefix=out_prefix)


def run_combined(
    dataset_names: list[str] | None = None,
    label: str | None = None,
    methods: dict[str, str] | None = None,
) -> None:
    """
    Gera MCM combinando múltiplos datasets em uma única análise.

    As séries de todos os datasets são empilhadas: cada linha representa
    uma série temporal, independentemente de qual dataset ela veio.

    Args:
        dataset_names: lista de datasets a combinar. Se None, usa DATASETS global.
        label: nome do arquivo de saída. Se None, junta os nomes dos datasets.
        methods: dict {nome_método: pasta}. Se None, usa METHODS global.
    """
    if dataset_names is None:
        dataset_names = DATASETS

    if label is None:
        label = "_".join(dataset_names)

    df = build_results_df(dataset_names, methods)
    out_prefix = os.path.join(OUTPUT_DIR, f"combined_{label}")
    print(f"\n=== Combinado: {dataset_names} ===")
    _run_mcm(df, save_name=f"mcm_{label}", out_prefix=out_prefix)


# ---------------------------------------------------------------------------
# Execução padrão
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Roda individualmente cada dataset
    for ds in DATASETS:
        run(ds)

    # Roda todos os datasets combinados em uma única análise
    run_combined(DATASETS)

    print("\nPronto! Resultados em:", OUTPUT_DIR)
