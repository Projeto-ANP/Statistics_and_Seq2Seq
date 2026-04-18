"""Model Confidence Set (Hansen, Lunde, Nason 2011) for forecast selection.

The MCS procedure delivers a *set-valued* answer to the question
"which forecasts are statistically indistinguishable from the best?":
given a collection of M models and a loss series per model, MCS
iteratively eliminates the worst-performing model until the equal-
predictive-ability null can no longer be rejected at level α.  The
surviving subset M̂_{1-α} is the Model Confidence Set.

Why MCS in HALMOC?
    Picking a single "winner" by mean loss is brittle to noise.  MCS
    gives the LLM Judge a *defensible candidate pool* rather than a
    point selection — it is the statistical analogue of beam search and
    matches the way verifier-based test-time scaling is presented in
    Snell et al. (2024).  See also the recent forecasting literature
    (e.g., M5, M6 retrospectives) where MCS is the standard tool for
    evaluating a competition leaderboard.

Algorithm sketch (T_R range statistic; Hansen et al. 2011, §3.1.1):
    1. For each pair (i, j) in the surviving set M, compute the time
       series of loss differentials d_{ij,t} = L_{i,t} - L_{j,t}.
    2. The t-statistic is
           t_{ij} = mean_t(d_{ij,t}) / sqrt(var(mean_t(d_{ij,t})))
       where the variance is estimated by a stationary block bootstrap
       (Politis & Romano 1994 JASA 89:1303-1313) to handle serial
       correlation.
    3. The MCS test statistic is T_R = max_{i,j in M} |t_{ij}|.
    4. Bootstrap the null distribution of T_R by resampling the loss
       differentials with the same blocks, recentre, and compare.
    5. If p > α, return M.  Otherwise, eliminate
           e_M = argmax_i max_{j in M} t_{ij}
       and repeat.

References:
    - Hansen, P. R., Lunde, A., Nason, J. M. (2011).
      *The Model Confidence Set*.  Econometrica 79(2):453-497.
    - Politis, D. N., & Romano, J. P. (1994).
      *The Stationary Bootstrap*.  JASA 89(428):1303-1313.
    - Bernardi, M., & Catania, L. (2018).  R package `MCS`.
      Comput. Stat. & Data Analysis 96:55-72 (algorithmic reference).

This is a pure-numpy implementation (no statsmodels dependency).  It is
sufficient for the HALMOC pipeline's typical sizes (n_windows ≤ 10³,
n_models ≤ 10²).  For larger problems, switch to the R package or to
arch.bootstrap.MCS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# Stationary block bootstrap
# ──────────────────────────────────────────────────────────────────────────


def stationary_bootstrap_indices(
    n: int,
    block_size: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a `(n_boot, n)` matrix of stationary-bootstrap indices.

    Politis & Romano (1994): at each step, with probability `1/block_size`,
    start a new block at a uniformly-drawn index; otherwise, advance the
    current index by one (mod n).

    Args:
        n: length of the original series.
        block_size: expected block length (must be ≥ 1).
        n_boot: number of bootstrap replicates.
        rng: numpy generator (deterministic for reproducibility).
    """

    if n <= 0:
        return np.zeros((n_boot, 0), dtype=int)
    p = 1.0 / max(1.0, float(block_size))
    out = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(0, n)
        new_block = rng.random(n - 1) < p
        starts = rng.integers(0, n, size=n - 1)
        for t in range(1, n):
            if new_block[t - 1]:
                idx[t] = starts[t - 1]
            else:
                idx[t] = (idx[t - 1] + 1) % n
        out[b] = idx
    return out


def _auto_block_size(d: np.ndarray) -> float:
    """Politis-White (2004) plug-in rule, simplified.

    Falls back to `n^{1/3}` when the rule is degenerate.
    """

    n = len(d)
    if n < 8:
        return max(1.0, float(n) ** (1.0 / 3.0))
    # Correlogram out to floor(min(n/2, 8 * log10(n)))
    L = int(min(n // 2, max(2, np.floor(8.0 * np.log10(n)))))
    d = np.asarray(d, dtype=float) - float(np.mean(d))
    sd = float(np.std(d))
    if sd < EPS:
        return max(1.0, float(n) ** (1.0 / 3.0))
    rho = np.array(
        [float(np.dot(d[: n - k], d[k:])) / (n * sd * sd) for k in range(1, L + 1)]
    )
    # Andrews-Buchinsky truncation: take significant lags
    sig = np.where(np.abs(rho) > 2.0 / np.sqrt(n))[0]
    m = int(sig.max() + 1) if sig.size > 0 else 1
    # Plug-in (Politis & White 2004, eq. 11)
    g = 0.0
    G = 0.0
    for k in range(1, 2 * m + 1):
        if k - 1 >= len(rho):
            break
        w = 1.0 if abs(k) <= m else 2.0 - abs(k) / m  # flat-top kernel
        g += w * rho[k - 1]
        G += w * abs(k) * rho[k - 1]
    g = 2.0 * g
    G = 2.0 * G
    denom = max(1.0 + g, EPS)
    bopt = ((2.0 * G ** 2) / (denom ** 2)) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))
    return float(np.clip(bopt, 1.0, max(1.0, n / 2.0)))


# ──────────────────────────────────────────────────────────────────────────
# MCS core
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MCSConfig:
    """Tuning knobs for the Model Confidence Set procedure.

    Attributes:
        alpha: significance level (1 - α confidence in the surviving set).
        n_boot: bootstrap replicates for variance/p-value estimation.
        block_size: stationary-bootstrap expected block length.  If
            `None` or `<= 0`, auto-select via Politis-White.
        statistic: `"range"` (T_R) or `"max"` (T_max) — both from
            Hansen et al. (2011).  T_R is faster and more popular.
        random_state: seed for the bootstrap RNG (deterministic).
        min_models: stop elimination when the surviving set reaches this
            size, regardless of p-value.
    """

    alpha: float = 0.10
    n_boot: int = 999
    block_size: Optional[float] = None
    statistic: Literal["range", "max"] = "range"
    random_state: Optional[int] = 0
    min_models: int = 1


@dataclass
class MCSResult:
    """Summary returned by `model_confidence_set`."""

    surviving: List[str]
    eliminated_order: List[Tuple[str, float]]  # (model, p-value at elimination)
    p_values: Dict[str, float]  # MCS p-value per model
    config: MCSConfig
    block_size_used: float
    n_obs: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "surviving": list(self.surviving),
            "eliminated_order": [(m, float(p)) for m, p in self.eliminated_order],
            "p_values": {k: float(v) for k, v in self.p_values.items()},
            "alpha": float(self.config.alpha),
            "n_boot": int(self.config.n_boot),
            "statistic": str(self.config.statistic),
            "block_size_used": float(self.block_size_used),
            "n_obs": int(self.n_obs),
        }


def model_confidence_set(
    losses: np.ndarray,
    model_names: Sequence[str],
    config: Optional[MCSConfig] = None,
) -> MCSResult:
    """Compute the Model Confidence Set on a (T, M) loss matrix.

    Args:
        losses: shape `(T, M)` — per-observation loss for each model
            (same loss function across columns; e.g. squared error per
            window-step pair flattened to a single time index).
        model_names: length-M iterable of model identifiers.
        config: MCSConfig knobs.

    Returns:
        `MCSResult` listing surviving models, elimination order, and
        per-model MCS p-values.
    """

    cfg = config or MCSConfig()
    L = np.asarray(losses, dtype=float)
    if L.ndim != 2:
        raise ValueError(f"losses must be 2-D (T, M), got shape={L.shape}")
    T, M = L.shape
    if len(model_names) != M:
        raise ValueError(
            f"model_names length {len(model_names)} != n_models {M}"
        )
    if M == 0 or T == 0:
        return MCSResult(
            surviving=list(model_names),
            eliminated_order=[],
            p_values={n: float("nan") for n in model_names},
            config=cfg,
            block_size_used=0.0,
            n_obs=T,
        )
    if M == 1:
        return MCSResult(
            surviving=list(model_names),
            eliminated_order=[],
            p_values={model_names[0]: 1.0},
            config=cfg,
            block_size_used=0.0,
            n_obs=T,
        )

    # NaN handling: drop rows with any NaN (conservative, avoids subtle
    # bias from row-wise imputation on dependent samples).
    finite = np.all(np.isfinite(L), axis=1)
    L = L[finite]
    T = L.shape[0]
    if T < 4:
        return MCSResult(
            surviving=list(model_names),
            eliminated_order=[],
            p_values={n: float("nan") for n in model_names},
            config=cfg,
            block_size_used=0.0,
            n_obs=T,
        )

    # Decide block size from the *pooled* loss differential variability
    if cfg.block_size is None or cfg.block_size <= 0:
        # heuristic: average column means as a single proxy series
        block = _auto_block_size(L.mean(axis=1))
    else:
        block = float(cfg.block_size)

    rng = np.random.default_rng(cfg.random_state)
    boot_idx = stationary_bootstrap_indices(T, block, cfg.n_boot, rng)

    surviving = list(range(M))
    eliminated: List[Tuple[str, float]] = []
    pvals: Dict[str, float] = {n: 1.0 for n in model_names}
    cumulative_p = 0.0

    while len(surviving) > max(1, int(cfg.min_models)):
        sub = L[:, surviving]
        t_stat, p_value, worst_local = _mcs_iteration(
            sub, boot_idx, statistic=cfg.statistic
        )
        cumulative_p = max(cumulative_p, p_value)
        worst_global = surviving[worst_local]
        if p_value > cfg.alpha:
            # cannot reject EPA — stop, but ensure remaining models
            # retain p-values >= the strictest level reached.
            for j in surviving:
                pvals[model_names[j]] = max(pvals[model_names[j]], cumulative_p)
            break

        # Eliminate worst, record monotone-adjusted p-value
        adj_p = cumulative_p
        eliminated.append((model_names[worst_global], adj_p))
        pvals[model_names[worst_global]] = adj_p
        surviving.remove(worst_global)

    # Models still surviving at termination get the final cumulative_p
    for j in surviving:
        pvals[model_names[j]] = max(pvals[model_names[j]], cumulative_p)

    return MCSResult(
        surviving=[model_names[j] for j in surviving],
        eliminated_order=eliminated,
        p_values=pvals,
        config=cfg,
        block_size_used=block,
        n_obs=T,
    )


# ──────────────────────────────────────────────────────────────────────────
# Inner step
# ──────────────────────────────────────────────────────────────────────────


def _mcs_iteration(
    L_sub: np.ndarray,
    boot_idx: np.ndarray,
    statistic: str = "range",
) -> Tuple[float, float, int]:
    """One iteration of the MCS elimination loop.

    Args:
        L_sub: (T, k) loss matrix for the currently-surviving models.
        boot_idx: (n_boot, T) bootstrap index matrix (precomputed for
            the full T to keep block structure stable across iterations).
        statistic: `"range"` or `"max"`.

    Returns:
        (test_statistic, p_value, index_of_worst_in_L_sub)
    """

    T, k = L_sub.shape
    if k <= 1:
        return 0.0, 1.0, 0

    # Loss differentials d_ij,t = L_i - L_j for all i != j
    # mean over j != i of d_ij gives the relative loss of model i
    col_mean = L_sub.mean(axis=1, keepdims=True)  # (T, 1)
    rel_loss = L_sub - col_mean  # (T, k); mean across rows is column-mean of L - row-mean of L
    # mean relative loss per model
    d_bar = rel_loss.mean(axis=0)  # (k,)

    # Bootstrap means of relative losses (recentred for null distribution)
    # boot_idx is (n_boot, T). Sample.
    n_boot = boot_idx.shape[0]
    # gather: rel_loss[boot_idx, :] -> (n_boot, T, k)
    boot_d = rel_loss[boot_idx].mean(axis=1)  # (n_boot, k)
    # Studentise using bootstrap variance of d_bar
    var_d = boot_d.var(axis=0, ddof=1)  # (k,)
    var_d = np.where(var_d > EPS, var_d, EPS)
    sd_d = np.sqrt(var_d)

    # t-stats
    t_obs = d_bar / sd_d

    # Bootstrap distribution: recentre by subtracting the original mean
    boot_t = (boot_d - d_bar[None, :]) / sd_d[None, :]

    if statistic == "max":
        # T_max = max_i (d_bar_i / sd_i)
        T_obs = float(np.max(t_obs))
        T_boot = boot_t.max(axis=1)
    else:
        # T_R = max_i |t_obs_i|  (range / equivalence form)
        T_obs = float(np.max(np.abs(t_obs)))
        T_boot = np.max(np.abs(boot_t), axis=1)

    # MCS p-value: probability under bootstrap null that T_boot >= T_obs
    p = float(np.mean(T_boot >= T_obs))
    # smoothing: include +1/(n_boot+1) protection from MC degenerate 0
    p = max(p, 1.0 / (n_boot + 1))

    worst = int(np.argmax(t_obs))  # eliminate model with largest *positive* relative loss
    return T_obs, p, worst


# ──────────────────────────────────────────────────────────────────────────
# Convenience: build losses from (n_windows, n_models, horizon) preds
# ──────────────────────────────────────────────────────────────────────────


def squared_error_loss(
    y_true: np.ndarray,
    y_preds: np.ndarray,
) -> np.ndarray:
    """Return a (n_windows * horizon, n_models) loss matrix of squared errors.

    Args:
        y_true:  (n_windows, horizon).
        y_preds: (n_windows, n_models, horizon).
    """

    y_true = np.asarray(y_true, dtype=float)
    y_preds = np.asarray(y_preds, dtype=float)
    if y_true.ndim != 2 or y_preds.ndim != 3:
        raise ValueError("y_true must be 2-D and y_preds 3-D")
    n_windows, horizon = y_true.shape
    if y_preds.shape[0] != n_windows or y_preds.shape[2] != horizon:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape}, y_preds {y_preds.shape}"
        )
    n_models = y_preds.shape[1]
    diff = y_preds - y_true[:, None, :]  # (n_windows, n_models, horizon)
    losses = (diff ** 2).transpose(0, 2, 1).reshape(n_windows * horizon, n_models)
    return losses


def absolute_error_loss(y_true: np.ndarray, y_preds: np.ndarray) -> np.ndarray:
    """Return a (n_windows * horizon, n_models) loss matrix of absolute errors."""

    y_true = np.asarray(y_true, dtype=float)
    y_preds = np.asarray(y_preds, dtype=float)
    diff = np.abs(y_preds - y_true[:, None, :])
    n_windows, n_models, horizon = diff.shape
    return diff.transpose(0, 2, 1).reshape(n_windows * horizon, n_models)
