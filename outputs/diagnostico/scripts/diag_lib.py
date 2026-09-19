"""Utilidades compartilhadas pelos scripts passoXX_*.py da investigacao de
previsoes divergentes. Reaproveita outputs/resultados/scripts/common.py (o
mesmo filtro de janela de teste, resolucao de caminho, parsing de listas)
em vez de duplicar essa logica.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGNOSTICO_BASE = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(DIAGNOSTICO_BASE))
RESULTADOS_SCRIPTS = os.path.join(REPO_ROOT, "outputs", "resultados", "scripts")
sys.path.insert(0, RESULTADOS_SCRIPTS)
import common as C  # noqa: E402

CATBOOST_FAMILY = [
    "catboost", "CWT_catboost", "DWT_catboost", "FT_catboost",
    "ONLY_CWT_catboost", "ONLY_DWT_catboost", "ONLY_FT_catboost",
]
RF_FAMILY = [
    "rf", "CWT_rf", "DWT_rf", "FT_rf",
    "ONLY_CWT_rf", "ONLY_DWT_rf", "ONLY_FT_rf",
]
ALL_REGRESSORS = CATBOOST_FAMILY + RF_FAMILY

# familia/representacao usadas no passo 4
FAMILY_OF = {r: "catboost" for r in CATBOOST_FAMILY}
FAMILY_OF.update({r: "rf" for r in RF_FAMILY})

REPRESENTACAO_OF = {
    "catboost": "original", "rf": "original",
    "CWT_catboost": "concatenada", "DWT_catboost": "concatenada", "FT_catboost": "concatenada",
    "CWT_rf": "concatenada", "DWT_rf": "concatenada", "FT_rf": "concatenada",
    "ONLY_CWT_catboost": "only", "ONLY_DWT_catboost": "only", "ONLY_FT_catboost": "only",
    "ONLY_CWT_rf": "only", "ONLY_DWT_rf": "only", "ONLY_FT_rf": "only",
}

DIVERGENCE_THRESHOLD = 100.0


def divergence_reference(row) -> float:
    """max(abs(train)) se a coluna existir no CSV de origem, senao
    max(abs(test)) da propria serie (nenhum dos CSVs de origem usados aqui
    tem coluna de treino -- o fallback e o caminho sempre executado na
    pratica, mas o codigo checa a coluna por seguranca)."""
    if "train" in row.index:
        train_vals = C.parse_num_list(row["train"])
        if train_vals:
            return max(abs(v) for v in train_vals)
    test_vals = C.parse_num_list(row["test"])
    return max(abs(v) for v in test_vals) if test_vals else 0.0


def is_divergent(row, threshold: float = DIVERGENCE_THRESHOLD) -> bool:
    pred_vals = C.parse_num_list(row["predictions"])
    if not pred_vals:
        return False
    max_pred = max(abs(v) for v in pred_vals)
    reference = divergence_reference(row)
    if reference == 0:
        return max_pred > 0
    return max_pred > threshold * reference


def tsf_series_values(dataset: str):
    """Retorna list[list[float]]: valores completos de cada serie do .tsf
    original de `dataset`, na mesma ordem (== dataset_index) usada em todo o
    projeto. Mesma extracao robusta de diag_lib.tsf_series_lengths (ultimo
    ':' separa metadata da lista de valores), so que devolvendo os valores
    em vez de so o comprimento.
    """
    tsf_path = os.path.join(C.TSF_BASE, C.TSF_MAP[dataset])
    series = []
    in_data = False
    with open(tsf_path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if in_data:
                values_part = line.rsplit(":", 1)[-1]
                series.append([float(v) for v in values_part.split(",") if v.strip() != ""])
            elif line.lower() == "@data":
                in_data = True
    return series


def tsf_series_lengths(dataset: str):
    """Retorna list[int]: comprimento total (n de pontos) de cada serie do
    .tsf original de `dataset`, na ordem em que aparecem apos '@data' (essa
    ordem e a mesma usada por dataset_index em todo o projeto). Note que e o
    comprimento da serie INTEIRA no .tsf (nao so a porcao de treino -- os
    CSVs de origem usados aqui nao tem uma coluna de treino explicita nem
    metadata suficiente em todo dataset para recalcular o corte exato de
    treino/teste de forma uniforme entre os 7 formatos de .tsf).
    """
    tsf_path = os.path.join(C.TSF_BASE, C.TSF_MAP[dataset])
    lengths = []
    in_data = False
    with open(tsf_path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if in_data:
                values_part = line.rsplit(":", 1)[-1]
                lengths.append(len([v for v in values_part.split(",") if v.strip() != ""]))
            elif line.lower() == "@data":
                in_data = True
    return lengths
