"""Ablacao -- Passo 3: MCM entre CREST (com agente) e CREST sem agente.

Mesma chamada de multi_comp_matrix.MCM.compare do item_07 (SMAPE: increasing;
POCID: decreasing; ProbaWinTieLoss + Wilcoxon), so que com dois comparates e as
680 series dos 7 datasets como uma unica populacao.

Requer o ambiente conda `metrics`:
    conda run -n metrics python outputs/resultados/scripts/ablacao_mcm.py

Saidas em outputs/resultados/ablacao/:
    mcm_input_agente_vs_sem_agente_{smape,pocid}.csv
    mcm_agente_vs_sem_agente_{smape,pocid}.{csv,png,pdf}
"""
import contextlib
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mfig
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

try:
    from multi_comp_matrix import MCM
except ImportError as exc:
    raise SystemExit(
        "multi_comp_matrix nao encontrado. Rode com: conda run -n metrics python " + os.path.abspath(__file__)
    ) from exc

METRICS = [
    ("smape", "SMAPE", "increasing"),
    ("pocid", "POCID", "decreasing"),
]
MAPPING = {"CREST": C.ORCHESTRATOR_GPTOSS, "CREST sem agente": C.ORCHESTRATOR_BASELINE}

# Figura maior e fontes maiores (o padrao "auto" para 2 comparates sai ~7x3 pol. com fonte 8).
FIG_SIZE = "16,8"
FONT_SIZE = 18


@contextlib.contextmanager
def colorbar_below():
    """MCM.py forca colorbar_orientation="vertical" em matrizes 2x2, ignorando o
    parametro; aqui a barra de cores passa a ser horizontal, abaixo da matriz,
    so durante a chamada."""
    original = mfig.Figure.colorbar

    def patched(self, mappable, *args, **kwargs):
        kwargs.update(orientation="horizontal", pad=0.14, shrink=0.7, aspect=35)
        cbar = original(self, mappable, *args, **kwargs)
        # com fonte 16, marcas a cada 0,0002 se sobrepoem; 5 marcas bastam
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
        return cbar

    mfig.Figure.colorbar = patched
    try:
        yield
    finally:
        mfig.Figure.colorbar = original


def main():
    out_dir = os.path.join(C.OUTPUTS_BASE, "ablacao")
    os.makedirs(out_dir, exist_ok=True)
    out_dir_sep = out_dir + os.sep
    all_warnings = []

    for metric, used_statistic, order_better in METRICS:
        wide, warns = C.build_wide_population(C.DATASETS, MAPPING, metric)
        wide = wide[list(MAPPING)]
        all_warnings.extend(warns)
        print(f"{metric}: {len(wide)} series")

        wide.to_csv(os.path.join(out_dir, f"mcm_input_agente_vs_sem_agente_{metric}.csv"), index=True)

        savename = f"mcm_agente_vs_sem_agente_{metric}"
        with warnings.catch_warnings(), colorbar_below():
            warnings.simplefilter("ignore")
            MCM.compare(
                df_results=wide.reset_index(drop=True),
                output_dir=out_dir_sep,
                png_savename=savename,
                pdf_savename=savename,
                csv_savename=savename,
                used_statistic=used_statistic,
                order_better=order_better,
                include_ProbaWinTieLoss=True,
                pvalue_test="wilcoxon",
                save_as_json=False,
                fig_size=FIG_SIZE,
                font_size=FONT_SIZE,
            )
        print(f"MCM ({metric}) salva em {out_dir}/{savename}.png", flush=True)

    C.merge_validation_warnings(all_warnings)


if __name__ == "__main__":
    main()
