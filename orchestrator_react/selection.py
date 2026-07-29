"""Pool recipes — *how* a pool was chosen, so it can be re-chosen per fold.

This module exists because of a measured defect. Until it was added, a pool was a
frozen list of model indices: `select_top_k` looked at all three validation
windows, picked the five lowest-error models, and the backtest then scored those
five models on **the same three windows**. The number that ranked a strategy had
already seen the selection it was ranking.

On the 111 NN5 series that made the validation score *anti-predictive*. Ranking
sixteen fixed rules by their in-sample validation score and by their blind test
score gives Spearman **-0.718**: looking better in validation predicted looking
*worse* on the test window. Rebuilding the pool per fold — the protocol here —
turns that into **+0.288**.

The design mirrors `weighting.WeightsRecipe`, which already had this property for
weights: store the recipe, not the numbers, and re-fit it under the anti-leakage
protocol. `PoolRecipe` does the same for membership.

Every selector takes:
    y_true  : (n_fit, horizon)
    y_preds : (n_fit, n_models, horizon)
and returns model indices into the full pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from orchestrator_react.weighting import per_model_error


#: Recipes that can be re-fit on a subset of windows. Anything else (an explicit
#: list of models the agent named) is constant across folds by definition.
REFITTABLE_METHODS = ("top_k", "stable", "prune_redundant")


def top_k_indices(
    y_true: np.ndarray, y_preds: np.ndarray, k: int, metric: str = "rmse"
) -> np.ndarray:
    """The k lowest-error models over the given windows."""
    k = int(np.clip(int(k), 1, y_preds.shape[1]))
    err = per_model_error(y_true, y_preds, metric=metric)
    return np.argsort(err)[:k]


def stable_indices(
    y_true: np.ndarray, y_preds: np.ndarray, k: int, metric: str = "rmse"
) -> np.ndarray:
    """The k most consistent models: lowest `mean rank + std of rank` across windows.

    A model ranked 1st in one window and 20th in another loses to one ranked 4th
    in all of them. With a single window there is no variability to measure, so
    this degenerates to `top_k_indices` — which is stated here rather than left
    for the caller to discover.
    """
    n_fit, n_models = y_preds.shape[0], y_preds.shape[1]
    k = int(np.clip(int(k), 1, n_models))
    if n_fit < 2:
        return top_k_indices(y_true, y_preds, k, metric)

    ranks = np.zeros((n_fit, n_models))
    for i in range(n_fit):
        err = per_model_error(y_true[i : i + 1], y_preds[i : i + 1], metric=metric)
        ranks[i] = np.argsort(np.argsort(err)) + 1
    penalised = ranks.mean(axis=0) + ranks.std(axis=0)
    return np.argsort(penalised)[:k]


def rank_table(
    y_true: np.ndarray, y_preds: np.ndarray, metric: str = "rmse"
) -> np.ndarray:
    """`(n_fit, n_models)` of per-window ranks — for reporting, not selection."""
    n_fit, n_models = y_preds.shape[0], y_preds.shape[1]
    ranks = np.zeros((n_fit, n_models))
    for i in range(n_fit):
        err = per_model_error(y_true[i : i + 1], y_preds[i : i + 1], metric=metric)
        ranks[i] = np.argsort(np.argsort(err)) + 1
    return ranks


def redundant_groups(
    y_true: np.ndarray,
    y_preds: np.ndarray,
    candidates: Sequence[int],
    threshold: float = 0.95,
) -> List[List[int]]:
    """Groups of models whose *error series* correlate above `threshold`.

    Correlating the errors rather than the forecasts is deliberate: two models can
    track the same series closely and still fail on different windows, which is
    exactly the diversity a combination needs.
    """
    cand = [int(c) for c in candidates]
    if len(cand) < 2:
        return []
    resid = (y_preds[:, cand, :] - y_true[:, None, :]).transpose(1, 0, 2)
    resid = resid.reshape(len(cand), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(resid)
    corr = np.nan_to_num(corr, nan=0.0)

    seen: set = set()
    groups: List[List[int]] = []
    for a in range(len(cand)):
        if a in seen:
            continue
        members = [a] + [
            b for b in range(a + 1, len(cand))
            if b not in seen and corr[a, b] >= float(threshold)
        ]
        if len(members) > 1:
            seen.update(members)
            groups.append([cand[m] for m in members])
    return groups


def prune_redundant_indices(
    y_true: np.ndarray,
    y_preds: np.ndarray,
    base: Sequence[int],
    corr_threshold: float = 0.95,
    metric: str = "rmse",
) -> np.ndarray:
    """Keeps the lowest-error member of each redundant group, drops the rest."""
    base = [int(b) for b in base]
    groups = redundant_groups(y_true, y_preds, base, threshold=corr_threshold)
    if not groups:
        return np.asarray(base, dtype=int)

    err = per_model_error(y_true, y_preds, metric=metric)
    dropped: set = set()
    for grp in groups:
        best = min(grp, key=lambda m: err[m])
        dropped.update(m for m in grp if m != best)
    kept = [b for b in base if b not in dropped]
    # Pruning everything is a configuration error, not a valid empty pool; the
    # caller gets the base back rather than a crash mid-backtest.
    return np.asarray(kept or base, dtype=int)


@dataclass
class PoolRecipe:
    """Reproducible description of how a pool was chosen.

    `resolved` holds the membership fit on all available windows — what the agent
    sees, what `apply_to_test` uses, and what goes in the CSV. Under nested
    selection the backtest ignores it and re-fits per fold.
    """

    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    #: Base pool for recipes that filter an existing one, as explicit indices so a
    #: fold can be resolved without walking a chain of handles.
    base: Optional[Tuple[int, ...]] = None
    resolved: Tuple[int, ...] = ()

    @property
    def refittable(self) -> bool:
        return self.method in REFITTABLE_METHODS

    def spec(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "params": dict(self.params),
            "n_base": None if self.base is None else len(self.base),
        }


def resolve_pool_recipe(
    recipe: PoolRecipe, y_true: np.ndarray, y_preds: np.ndarray
) -> np.ndarray:
    """Re-runs a recipe over the given windows.

    It does not decide which windows those are — `state.py` owns the anti-leakage
    protocol, exactly as it does for weight recipes.
    """
    method = str(recipe.method)
    p = dict(recipe.params)

    if not recipe.refittable or y_true.shape[0] == 0:
        return np.asarray(recipe.resolved, dtype=int)

    if method == "top_k":
        return top_k_indices(y_true, y_preds, int(p.get("k", 5)), str(p.get("metric", "rmse")))
    if method == "stable":
        return stable_indices(y_true, y_preds, int(p.get("k", 5)), str(p.get("metric", "rmse")))
    if method == "prune_redundant":
        base = recipe.base if recipe.base is not None else tuple(range(y_preds.shape[1]))
        return prune_redundant_indices(
            y_true, y_preds, base,
            corr_threshold=float(p.get("corr_threshold", 0.95)),
            metric=str(p.get("metric", "rmse")),
        )
    return np.asarray(recipe.resolved, dtype=int)
