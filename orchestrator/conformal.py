"""Conformal prediction wrappers for the HALMOC pipeline.

Provides distribution-free prediction intervals with empirical coverage
≥ 1−α, even under regime change.  Two complementary methods:

- **Split conformal** (Vovk, Gammerman, Shafer 2005; Lei et al. 2018
  *JASA* 113(523):1094–1111): the textbook split-conformal estimator
  on absolute residuals — fast, exchangeable-data baseline.
- **EnbPI** (Xu & Xie 2021 *ICML*; 2022 *IEEE TPAMI* 44(11):8682–8695):
  ensemble out-of-bag style residuals, designed for time series.
- **ACI** (Adaptive Conformal Inference — Gibbs & Candès 2021 *NeurIPS*;
  2022): online α adaptation that corrects miscoverage under
  distribution shift.

All three use the same minimal interface so they are interchangeable
in `halmoc_pipeline.py`:

```python
wrapper = ConformalWrapper.from_name("aci", alpha=0.10)
wrapper.calibrate(y_true_calib, y_pred_calib)        # 1-D arrays
lo, hi = wrapper.interval(y_pred_test)               # 1-D arrays
wrapper.update(y_true_obs, y_pred_obs)               # online, ACI only
```

References (in addition to the above):
    - Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman,
      L. (2018). Distribution-Free Predictive Inference for Regression.
      JASA 113(523):1094–1111.
    - Vovk, V., Gammerman, A., & Shafer, G. (2005).
      *Algorithmic Learning in a Random World*. Springer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np


EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# Base interface + registry
# ──────────────────────────────────────────────────────────────────────────


class ConformalWrapper(ABC):
    """Abstract base; all conformal wrappers expose the same 3 methods."""

    name: str = "base"

    def __init__(self, alpha: float = 0.10, **params: Any) -> None:
        self.alpha = float(alpha)
        self.params: Dict[str, Any] = dict(params)
        self._fitted: bool = False

    @abstractmethod
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalWrapper":
        ...

    @abstractmethod
    def interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Online update.  No-op for static methods."""

        return None

    @staticmethod
    def _flat(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(-1)


CONFORMAL_REGISTRY: Dict[str, Type[ConformalWrapper]] = {}


def register_conformal(name: str) -> Callable[[Type[ConformalWrapper]], Type[ConformalWrapper]]:
    key = name.strip().lower()

    def _decorate(cls: Type[ConformalWrapper]) -> Type[ConformalWrapper]:
        cls.name = key
        CONFORMAL_REGISTRY[key] = cls
        return cls

    return _decorate


def make_conformal(name: str, **params: Any) -> ConformalWrapper:
    key = name.strip().lower()
    if key not in CONFORMAL_REGISTRY:
        raise KeyError(
            f"Unknown conformal wrapper '{name}'. Available: {sorted(CONFORMAL_REGISTRY)}"
        )
    return CONFORMAL_REGISTRY[key](**params)


# Convenience alias used in the docstring example
ConformalWrapper.from_name = staticmethod(make_conformal)  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────
# Split conformal
# ──────────────────────────────────────────────────────────────────────────


@register_conformal("split")
class SplitConformal(ConformalWrapper):
    """Standard split conformal on absolute residuals (Lei et al. 2018).

    Given calibration residuals r_i = |y_i - ŷ_i|, the (1−α)-quantile q
    yields the symmetric interval [ŷ − q, ŷ + q] for any new ŷ, with
    finite-sample marginal coverage ≥ 1−α under exchangeability.
    """

    def __init__(self, alpha: float = 0.10, **params: Any) -> None:
        super().__init__(alpha=alpha, **params)
        self._q: float = 0.0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformal":
        y_t = self._flat(y_true)
        y_p = self._flat(y_pred)
        if y_t.size != y_p.size or y_t.size == 0:
            raise ValueError(f"size mismatch or empty: {y_t.size} vs {y_p.size}")
        residuals = np.abs(y_t - y_p)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            self._q = 0.0
        else:
            n = residuals.size
            # finite-sample correction: ceil((n+1)(1-α)) / n
            level = float(np.ceil((n + 1) * (1 - self.alpha))) / n
            level = float(np.clip(level, 0.0, 1.0))
            self._q = float(np.quantile(residuals, level))
        self._fitted = True
        return self

    def interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("SplitConformal.interval called before calibrate")
        y_p = self._flat(y_pred)
        return y_p - self._q, y_p + self._q


# ──────────────────────────────────────────────────────────────────────────
# EnbPI
# ──────────────────────────────────────────────────────────────────────────


@register_conformal("enbpi")
class EnbPIConformal(ConformalWrapper):
    """EnbPI-flavoured wrapper on a single ensemble prediction stream.

    The original EnbPI (Xu & Xie 2021) bootstraps the base model B
    times and uses leave-one-out residuals.  Here we accept a stream of
    *already-aggregated* validation residuals and treat them as if they
    were OOB — appropriate when the meta-combiner is fit per-window with
    expanding-origin training (no in-fold leakage).  This keeps the
    interface identical to the other wrappers.

    Parameter `mode`:
        - `"sliding"` (default): use only the most recent
          `window_size` residuals.  Adapts to slow regime change.
        - `"all"`: use all calibration residuals.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        window_size: int = 200,
        mode: str = "sliding",
        **params: Any,
    ) -> None:
        super().__init__(alpha=alpha, window_size=window_size, mode=mode, **params)
        self._residuals: np.ndarray = np.empty(0, dtype=float)
        self._q: float = 0.0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "EnbPIConformal":
        y_t = self._flat(y_true)
        y_p = self._flat(y_pred)
        if y_t.size != y_p.size or y_t.size == 0:
            raise ValueError(f"size mismatch or empty: {y_t.size} vs {y_p.size}")
        r = np.abs(y_t - y_p)
        r = r[np.isfinite(r)]
        self._residuals = r
        self._update_q()
        self._fitted = True
        return self

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_t = self._flat(y_true)
        y_p = self._flat(y_pred)
        new_r = np.abs(y_t - y_p)
        new_r = new_r[np.isfinite(new_r)]
        self._residuals = np.concatenate([self._residuals, new_r])
        self._update_q()

    def _update_q(self) -> None:
        ws = int(self.params["window_size"])
        if str(self.params["mode"]).lower() == "sliding" and ws > 0:
            r = self._residuals[-ws:]
        else:
            r = self._residuals
        if r.size == 0:
            self._q = 0.0
            return
        n = r.size
        level = float(np.ceil((n + 1) * (1 - self.alpha))) / n
        level = float(np.clip(level, 0.0, 1.0))
        self._q = float(np.quantile(r, level))

    def interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("EnbPIConformal.interval called before calibrate")
        y_p = self._flat(y_pred)
        return y_p - self._q, y_p + self._q


# ──────────────────────────────────────────────────────────────────────────
# Adaptive Conformal Inference (ACI)
# ──────────────────────────────────────────────────────────────────────────


@register_conformal("aci")
class ACIConformal(ConformalWrapper):
    """Adaptive Conformal Inference (Gibbs & Candès 2021 NeurIPS).

    Maintains an online α_t that is corrected after each observed
    miscoverage:

        α_{t+1} = α_t + γ · (α − err_t)

    where err_t = 1 if the previous interval missed and 0 otherwise.
    This guarantees long-run coverage = 1−α even under arbitrary
    distribution shift.

    `gamma` controls adaptation speed (Gibbs & Candès use 0.005 for
    daily data).  A larger γ reacts faster but is noisier.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.01,
        window_size: int = 200,
        **params: Any,
    ) -> None:
        super().__init__(alpha=alpha, gamma=gamma, window_size=window_size, **params)
        self._residuals: np.ndarray = np.empty(0, dtype=float)
        self._alpha_t: float = float(alpha)
        self._q: float = 0.0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ACIConformal":
        y_t = self._flat(y_true)
        y_p = self._flat(y_pred)
        if y_t.size != y_p.size or y_t.size == 0:
            raise ValueError(f"size mismatch or empty: {y_t.size} vs {y_p.size}")
        r = np.abs(y_t - y_p)
        r = r[np.isfinite(r)]
        self._residuals = r
        self._alpha_t = float(self.alpha)
        self._update_q()
        self._fitted = True
        return self

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Online: append residual and adapt α_t for the next call."""

        y_t = self._flat(y_true)
        y_p = self._flat(y_pred)
        new_r = np.abs(y_t - y_p)
        for i, r_obs in enumerate(new_r):
            if not np.isfinite(r_obs):
                continue
            # Did the most-recent interval cover y_t[i]?
            err = 1.0 if r_obs > self._q else 0.0
            self._alpha_t = float(
                np.clip(self._alpha_t + self.params["gamma"] * (self.alpha - err), 0.0, 1.0)
            )
            self._residuals = np.append(self._residuals, r_obs)
            self._update_q()

    def _update_q(self) -> None:
        ws = int(self.params["window_size"])
        r = self._residuals[-ws:] if ws > 0 else self._residuals
        if r.size == 0:
            self._q = 0.0
            return
        n = r.size
        level = float(np.clip(1.0 - self._alpha_t, 0.0, 1.0))
        # use plain (n+1) finite-sample quantile
        level = min(1.0, float(np.ceil((n + 1) * level)) / n)
        self._q = float(np.quantile(r, level))

    def interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("ACIConformal.interval called before calibrate")
        y_p = self._flat(y_pred)
        return y_p - self._q, y_p + self._q

    @property
    def current_alpha(self) -> float:
        """Currently effective α_t (useful for diagnostics)."""

        return float(self._alpha_t)


# ──────────────────────────────────────────────────────────────────────────
# Convenience: turn the (n_windows, horizon) calibration tensor into 1-D
# ──────────────────────────────────────────────────────────────────────────


def calibrate_from_validation(
    wrapper: ConformalWrapper,
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
) -> ConformalWrapper:
    """Calibrate by flattening (n_windows, horizon) arrays.

    Useful when the meta-combiner is fit per-horizon and one wants a
    single coverage guarantee over all horizons.  For per-horizon
    intervals, instantiate one wrapper per step.
    """

    return wrapper.calibrate(np.asarray(y_true_2d).reshape(-1),
                              np.asarray(y_pred_2d).reshape(-1))
