"""Estatistica pareada por dataset, com correcao de Holm.

Compartilhado pelo item 01 (`analyze_search.py`) e pelo item 03
(`stats_por_dataset.py`). A diferenca em relacao aos scripts antigos: em vez de
um Wilcoxon agregado sobre 680 series de 7 datasets, cada comparacao e feita
dentro do dataset, e o conjunto de comparacoes de um dataset recebe correcao de
multiplicidade (Holm).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

LOWER_IS_BETTER = {"smape", "rmse", "mape", "msmape", "mae"}


def pairwise_table(methods, reference, dataset, metrics=("smape", "pocid", "rmse"), tol=1e-9):
    """Compara cada metodo contra `reference`, pareado por dataset_index."""
    rows = []
    ref = methods.get(reference)
    if ref is None:
        return rows
    for name, m in methods.items():
        if name == reference:
            continue
        idx = ref.index.intersection(m.index)
        if len(idx) == 0:
            continue
        for metric in metrics:
            if metric not in ref.columns or metric not in m.columns:
                continue
            a = ref.loc[idx, metric].astype(float).values
            b = m.loc[idx, metric].astype(float).values
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if a.size == 0:
                continue
            d = b - a
            lower = metric in LOWER_IS_BETTER
            wins = int(np.sum((b < a - 1e-6) if lower else (b > a + 1e-6)))
            losses = int(np.sum((b > a + 1e-6) if lower else (b < a - 1e-6)))
            ties = int(a.size - wins - losses)
            nz = d[np.abs(d) > tol]
            p = float(wilcoxon(nz).pvalue) if nz.size >= 5 else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "reference": reference,
                    "metric": metric,
                    "n_series": int(a.size),
                    "mean_method": float(np.mean(b)),
                    "mean_reference": float(np.mean(a)),
                    "mean_diff": float(np.mean(d)),
                    "median_diff": float(np.median(d)),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "p_value": p,
                }
            )
    return rows


def holm(rows, p_key="p_value", out_key="p_holm", alpha=0.05):
    """Correcao de Holm dentro de um mesmo dataset (in place, retorna rows)."""
    valid = [r for r in rows if r.get(p_key) is not None and np.isfinite(r[p_key])]
    valid.sort(key=lambda r: r[p_key])
    m = len(valid)
    prev = 0.0
    for i, row in enumerate(valid):
        adjusted = min(1.0, (m - i) * float(row[p_key]))
        adjusted = max(adjusted, prev)  # monotonico
        prev = adjusted
        row[out_key] = float(adjusted)
        row["significant"] = bool(adjusted < alpha)
    for row in rows:
        row.setdefault(out_key, float("nan"))
        row.setdefault("significant", False)
    return rows
