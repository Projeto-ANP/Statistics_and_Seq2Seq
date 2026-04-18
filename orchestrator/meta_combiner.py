"""Pluggable meta-combiner registry for forecast combination.

This module is the **numeric** layer of the HALMOC architecture (Tier D).
While the LLM Council proposes *strategy families* (e.g. "stacking",
"horizon-specialist selection") and the LLM Judge selects a subset, the
actual numeric weights are learned by a *meta-combiner*: a deterministic
machine-learning model that maps base-forecaster predictions to the
target.

Why a swappable registry?
    - **Reproducibility**: every combiner is registered by name and
      instantiated through a single factory; experiments are
      self-describing.
    - **Empirical neutrality**: arxiv 2504.08940 (Combining Forecasts
      using Meta-Learning, April 2025) shows Random Forest beats
      Linear/kNN/MLP/LSTM on 35 EU electricity series (MAPE 1.52 %), but
      results are dataset-dependent.  Users must be able to swap in
      Ridge, Lasso, GBM, LightGBM, XGBoost, or a custom estimator with
      one config change.
    - **Family-safety**: predictions from FT/CWT/DWT base models exhibit
      strong intra-family redundancy.  An L2-regularised linear model
      (ridge) is the safe default; tree models capture non-linear
      regime interactions.

Design:
    All combiners implement a small `MetaCombinerBase` interface
    (`fit`, `predict`, `get_weights`).  `register_meta_combiner("name")`
    decorates a class to add it to the registry; `make_meta_combiner`
    is the single factory.  `PerHorizonEnsemble` is a thin facade that
    fits one independent combiner per forecast step — recommended for
    multi-horizon forecasts because horizon-specific dynamics differ
    (Gaillard & Goude 2015; Timmermann 2006 *Handbook of Economic
    Forecasting* ch. 4).

Citations:
    - Combining Forecasts using Meta-Learning (Sobczak et al. 2025),
      arxiv 2504.08940 — RF wins among LR/kNN/MLP/RF/LSTM.
    - Timmermann (2006) — handbook treatment of forecast combination;
      OLS/ridge as canonical linear baselines.
    - Gaillard & Goude (2015) — horizon-specific specialist combination
      for electricity load.
    - Breiman (2001) — Random Forests.
    - Friedman (2001) — Gradient Boosting Machines.

The module is dependency-light by default (numpy + scikit-learn).  XGB,
LightGBM, etc. are loaded lazily and degrade gracefully if missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np


EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# Base interface
# ──────────────────────────────────────────────────────────────────────────


class MetaCombinerBase(ABC):
    """Abstract base class every meta-combiner must implement.

    Contract:
        - `fit(X, y)` accepts an (n_samples, n_base_models) feature
          matrix of base predictions and an (n_samples,) target.
        - `predict(X)` returns an (n_samples,) vector of combined
          forecasts.
        - `get_weights()` returns either an (n_base_models,) vector of
          model weights (linear combiners) or `None` (non-linear
          combiners that have no closed-form weight interpretation).

    Implementations should be deterministic given a fixed `random_state`
    where applicable.
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)
        self._fitted: bool = False
        self._n_features: Optional[int] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaCombinerBase":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def get_weights(self) -> Optional[np.ndarray]:
        return None

    # ----- Helpers shared by subclasses -----

    @staticmethod
    def _check_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape={X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"row mismatch: X has {X.shape[0]} rows but y has {y.shape[0]}"
            )
        # Replace NaN/inf with column means (a defensive step; the caller
        # should ideally pre-clean).
        if not np.all(np.isfinite(X)):
            col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            X = np.where(np.isfinite(X), X, col_mean)
        if not np.all(np.isfinite(y)):
            y = np.where(np.isfinite(y), y, np.nanmean(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0)
        return X, y

    @staticmethod
    def _check_x(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape={X.shape}")
        if not np.all(np.isfinite(X)):
            col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            X = np.where(np.isfinite(X), X, col_mean)
        return X


# ──────────────────────────────────────────────────────────────────────────
# Registry + factory
# ──────────────────────────────────────────────────────────────────────────


META_COMBINER_REGISTRY: Dict[str, Type[MetaCombinerBase]] = {}


def register_meta_combiner(
    name: str,
) -> Callable[[Type[MetaCombinerBase]], Type[MetaCombinerBase]]:
    """Decorator: add `cls` to `META_COMBINER_REGISTRY` under `name`.

    Names are case-insensitive.  Re-registering the same name overwrites
    the previous entry (useful for hot-swapping in notebooks).
    """

    key = name.strip().lower()

    def _decorate(cls: Type[MetaCombinerBase]) -> Type[MetaCombinerBase]:
        cls.name = key
        META_COMBINER_REGISTRY[key] = cls
        return cls

    return _decorate


def list_meta_combiners() -> List[str]:
    """Return the sorted list of registered meta-combiner names."""

    return sorted(META_COMBINER_REGISTRY.keys())


def make_meta_combiner(name: str, **params: Any) -> MetaCombinerBase:
    """Instantiate a registered meta-combiner by name.

    Raises:
        KeyError: if `name` is not registered.
    """

    key = name.strip().lower()
    if key not in META_COMBINER_REGISTRY:
        raise KeyError(
            f"Unknown meta-combiner '{name}'. Available: {list_meta_combiners()}"
        )
    return META_COMBINER_REGISTRY[key](**params)


# ──────────────────────────────────────────────────────────────────────────
# Built-in combiners
# ──────────────────────────────────────────────────────────────────────────


@register_meta_combiner("simple_average")
class SimpleAverageCombiner(MetaCombinerBase):
    """Equal-weight mean across base models.  No fit needed.

    Included as a baseline (Clemen 1989; Stock & Watson 2004 — "forecast
    combination puzzle": equal weights are hard to beat).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleAverageCombiner":
        X, _ = self._check_xy(X, y)
        self._n_features = X.shape[1]
        self._weights = np.ones(self._n_features) / float(self._n_features)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        return X.mean(axis=1)

    def get_weights(self) -> Optional[np.ndarray]:
        return self._weights if self._fitted else None


@register_meta_combiner("ols")
class OLSMetaCombiner(MetaCombinerBase):
    """Ordinary Least Squares with optional non-negativity + simplex.

    Granger & Ramanathan (1984) Method A baseline.  Without
    constraints, OLS may produce negative or > 1 weights — useful as a
    reference, risky in production.
    """

    def __init__(
        self,
        non_negative: bool = False,
        project_simplex: bool = False,
        fit_intercept: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            non_negative=non_negative,
            project_simplex=project_simplex,
            fit_intercept=fit_intercept,
            **params,
        )
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OLSMetaCombiner":
        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]

        if self.params["non_negative"]:
            try:
                from scipy.optimize import nnls

                Xa = (
                    np.hstack([X, np.ones((X.shape[0], 1))])
                    if self.params["fit_intercept"]
                    else X
                )
                coefs, _ = nnls(Xa, y)
                if self.params["fit_intercept"]:
                    self._coef = coefs[:-1]
                    self._intercept = float(coefs[-1])
                else:
                    self._coef = coefs
                    self._intercept = 0.0
            except Exception:
                # Fall back to lstsq if scipy missing
                Xa = (
                    np.hstack([X, np.ones((X.shape[0], 1))])
                    if self.params["fit_intercept"]
                    else X
                )
                coefs, *_ = np.linalg.lstsq(Xa, y, rcond=None)
                if self.params["fit_intercept"]:
                    self._coef = np.maximum(coefs[:-1], 0.0)
                    self._intercept = float(coefs[-1])
                else:
                    self._coef = np.maximum(coefs, 0.0)
        else:
            Xa = (
                np.hstack([X, np.ones((X.shape[0], 1))])
                if self.params["fit_intercept"]
                else X
            )
            coefs, *_ = np.linalg.lstsq(Xa, y, rcond=None)
            if self.params["fit_intercept"]:
                self._coef = coefs[:-1]
                self._intercept = float(coefs[-1])
            else:
                self._coef = coefs
                self._intercept = 0.0

        if self.params["project_simplex"]:
            self._coef = _project_simplex(np.asarray(self._coef, dtype=float))
            self._intercept = 0.0

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._coef is None:
            raise RuntimeError("OLSMetaCombiner.predict called before fit")
        return X @ self._coef + self._intercept

    def get_weights(self) -> Optional[np.ndarray]:
        return None if self._coef is None else np.asarray(self._coef, dtype=float)


@register_meta_combiner("ridge")
class RidgeMetaCombiner(MetaCombinerBase):
    """L2-regularised linear combiner — recommended default.

    Why ridge?  Base predictions from FT/CWT/DWT families are often
    highly collinear; OLS coefficients become unstable.  Ridge
    regularisation shrinks weights toward 0 (equivalent under a
    Gaussian prior) and produces well-conditioned solutions.

    Reference: Hoerl & Kennard (1970); Diebold & Pauly (1990) for
    forecast-combination context.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        project_simplex: bool = False,
        random_state: Optional[int] = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            project_simplex=project_simplex,
            random_state=random_state,
            **params,
        )
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeMetaCombiner":
        from sklearn.linear_model import Ridge

        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]
        self._estimator = Ridge(
            alpha=float(self.params["alpha"]),
            fit_intercept=bool(self.params["fit_intercept"]),
            random_state=self.params["random_state"],
        )
        self._estimator.fit(X, y)
        if self.params["project_simplex"]:
            w = _project_simplex(np.asarray(self._estimator.coef_, dtype=float))
            self._estimator.coef_ = w
            self._estimator.intercept_ = 0.0
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._estimator is None:
            raise RuntimeError("RidgeMetaCombiner.predict called before fit")
        return self._estimator.predict(X)

    def get_weights(self) -> Optional[np.ndarray]:
        if self._estimator is None:
            return None
        return np.asarray(self._estimator.coef_, dtype=float)


@register_meta_combiner("lasso")
class LassoMetaCombiner(MetaCombinerBase):
    """L1-regularised linear combiner (encourages sparse model selection).

    Reference: Tibshirani (1996); Diebold & Shin (2019) — sparse
    forecast combination.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        fit_intercept: bool = True,
        project_simplex: bool = False,
        max_iter: int = 5000,
        random_state: Optional[int] = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            project_simplex=project_simplex,
            max_iter=max_iter,
            random_state=random_state,
            **params,
        )
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoMetaCombiner":
        from sklearn.linear_model import Lasso

        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]
        self._estimator = Lasso(
            alpha=float(self.params["alpha"]),
            fit_intercept=bool(self.params["fit_intercept"]),
            max_iter=int(self.params["max_iter"]),
            random_state=self.params["random_state"],
        )
        self._estimator.fit(X, y)
        if self.params["project_simplex"]:
            w = _project_simplex(np.asarray(self._estimator.coef_, dtype=float))
            self._estimator.coef_ = w
            self._estimator.intercept_ = 0.0
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._estimator is None:
            raise RuntimeError("LassoMetaCombiner.predict called before fit")
        return self._estimator.predict(X)

    def get_weights(self) -> Optional[np.ndarray]:
        if self._estimator is None:
            return None
        return np.asarray(self._estimator.coef_, dtype=float)


@register_meta_combiner("elasticnet")
class ElasticNetMetaCombiner(MetaCombinerBase):
    """L1+L2 elastic-net (Zou & Hastie 2005)."""

    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 5000,
        random_state: Optional[int] = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=random_state,
            **params,
        )
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetMetaCombiner":
        from sklearn.linear_model import ElasticNet

        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]
        self._estimator = ElasticNet(
            alpha=float(self.params["alpha"]),
            l1_ratio=float(self.params["l1_ratio"]),
            fit_intercept=bool(self.params["fit_intercept"]),
            max_iter=int(self.params["max_iter"]),
            random_state=self.params["random_state"],
        )
        self._estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._estimator is None:
            raise RuntimeError("ElasticNetMetaCombiner.predict called before fit")
        return self._estimator.predict(X)

    def get_weights(self) -> Optional[np.ndarray]:
        if self._estimator is None:
            return None
        return np.asarray(self._estimator.coef_, dtype=float)


@register_meta_combiner("random_forest")
class RandomForestMetaCombiner(MetaCombinerBase):
    """Random Forest meta-combiner.

    Per arxiv 2504.08940 (April 2025), RF was the strongest meta-learner
    on 35 EU electricity series (MAPE 1.52 % vs LR/kNN/MLP/LSTM).
    Captures non-linear interactions between base predictions.

    Reference: Breiman (2001) Machine Learning 45:5–32.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 2,
        n_jobs: int = -1,
        random_state: Optional[int] = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            **params,
        )
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestMetaCombiner":
        from sklearn.ensemble import RandomForestRegressor

        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]
        self._estimator = RandomForestRegressor(
            n_estimators=int(self.params["n_estimators"]),
            max_depth=self.params["max_depth"],
            min_samples_leaf=int(self.params["min_samples_leaf"]),
            n_jobs=int(self.params["n_jobs"]),
            random_state=self.params["random_state"],
        )
        self._estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._estimator is None:
            raise RuntimeError("RandomForestMetaCombiner.predict called before fit")
        return self._estimator.predict(X)

    def get_weights(self) -> Optional[np.ndarray]:
        """Surrogate: feature importances normalised to sum 1.

        Not weights in the linear-combination sense, but a useful
        interpretability proxy when paired with `simple_average` as a
        sanity comparison.
        """

        if self._estimator is None:
            return None
        imp = np.asarray(self._estimator.feature_importances_, dtype=float)
        s = imp.sum()
        return imp / s if s > 0 else imp


@register_meta_combiner("gbm")
class GradientBoostingMetaCombiner(MetaCombinerBase):
    """sklearn GradientBoostingRegressor meta-combiner.

    Reference: Friedman (2001) Annals of Statistics 29(5):1189–1232.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: Optional[int] = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            **params,
        )
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingMetaCombiner":
        from sklearn.ensemble import GradientBoostingRegressor

        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]
        self._estimator = GradientBoostingRegressor(
            n_estimators=int(self.params["n_estimators"]),
            learning_rate=float(self.params["learning_rate"]),
            max_depth=int(self.params["max_depth"]),
            subsample=float(self.params["subsample"]),
            random_state=self.params["random_state"],
        )
        self._estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._estimator is None:
            raise RuntimeError("GradientBoostingMetaCombiner.predict called before fit")
        return self._estimator.predict(X)

    def get_weights(self) -> Optional[np.ndarray]:
        if self._estimator is None:
            return None
        imp = np.asarray(self._estimator.feature_importances_, dtype=float)
        s = imp.sum()
        return imp / s if s > 0 else imp


@register_meta_combiner("knn")
class KNNMetaCombiner(MetaCombinerBase):
    """k-Nearest-Neighbours meta-combiner.

    A non-parametric local combiner.  Effective when the prediction
    landscape has clear regimes; weak when the n_models dimensionality
    is high relative to training rows.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        **params: Any,
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, weights=weights, **params)
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNMetaCombiner":
        from sklearn.neighbors import KNeighborsRegressor

        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]
        k = max(1, min(int(self.params["n_neighbors"]), X.shape[0]))
        self._estimator = KNeighborsRegressor(
            n_neighbors=k,
            weights=str(self.params["weights"]),
        )
        self._estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._estimator is None:
            raise RuntimeError("KNNMetaCombiner.predict called before fit")
        return self._estimator.predict(X)


@register_meta_combiner("mlp")
class MLPMetaCombiner(MetaCombinerBase):
    """Multi-Layer Perceptron meta-combiner.

    Useful for capturing non-linear regime interactions when the
    training set is large.  Not recommended below ~100 windows.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (32, 16),
        activation: str = "relu",
        max_iter: int = 500,
        learning_rate_init: float = 1e-3,
        random_state: Optional[int] = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            activation=activation,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            random_state=random_state,
            **params,
        )
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPMetaCombiner":
        from sklearn.neural_network import MLPRegressor

        X, y = self._check_xy(X, y)
        self._n_features = X.shape[1]
        self._estimator = MLPRegressor(
            hidden_layer_sizes=tuple(self.params["hidden_layer_sizes"]),
            activation=str(self.params["activation"]),
            max_iter=int(self.params["max_iter"]),
            learning_rate_init=float(self.params["learning_rate_init"]),
            random_state=self.params["random_state"],
        )
        self._estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self._estimator is None:
            raise RuntimeError("MLPMetaCombiner.predict called before fit")
        return self._estimator.predict(X)


# Optional gradient-boosting backends — registered only if importable.

try:  # pragma: no cover - depends on optional dep
    import xgboost  # noqa: F401

    @register_meta_combiner("xgboost")
    class XGBoostMetaCombiner(MetaCombinerBase):
        """XGBoost meta-combiner (registered iff `xgboost` is installed).

        Reference: Chen & Guestrin (2016) KDD.
        """

        def __init__(
            self,
            n_estimators: int = 300,
            learning_rate: float = 0.05,
            max_depth: int = 4,
            subsample: float = 0.9,
            colsample_bytree: float = 0.9,
            random_state: Optional[int] = 0,
            **params: Any,
        ) -> None:
            super().__init__(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                **params,
            )
            self._estimator = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostMetaCombiner":
            from xgboost import XGBRegressor

            X, y = self._check_xy(X, y)
            self._n_features = X.shape[1]
            self._estimator = XGBRegressor(
                n_estimators=int(self.params["n_estimators"]),
                learning_rate=float(self.params["learning_rate"]),
                max_depth=int(self.params["max_depth"]),
                subsample=float(self.params["subsample"]),
                colsample_bytree=float(self.params["colsample_bytree"]),
                random_state=self.params["random_state"],
                tree_method="hist",
                verbosity=0,
            )
            self._estimator.fit(X, y)
            self._fitted = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X = self._check_x(X)
            if self._estimator is None:
                raise RuntimeError("XGBoostMetaCombiner.predict called before fit")
            return self._estimator.predict(X)

        def get_weights(self) -> Optional[np.ndarray]:
            if self._estimator is None:
                return None
            try:
                imp = np.asarray(self._estimator.feature_importances_, dtype=float)
                s = imp.sum()
                return imp / s if s > 0 else imp
            except Exception:
                return None
except ImportError:  # pragma: no cover
    pass


try:  # pragma: no cover
    import lightgbm  # noqa: F401

    @register_meta_combiner("lightgbm")
    class LightGBMMetaCombiner(MetaCombinerBase):
        """LightGBM meta-combiner (registered iff `lightgbm` is installed).

        Reference: Ke et al. (2017) NeurIPS.
        """

        def __init__(
            self,
            n_estimators: int = 300,
            learning_rate: float = 0.05,
            num_leaves: int = 31,
            min_child_samples: int = 5,
            subsample: float = 0.9,
            random_state: Optional[int] = 0,
            **params: Any,
        ) -> None:
            super().__init__(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                random_state=random_state,
                **params,
            )
            self._estimator = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMMetaCombiner":
            from lightgbm import LGBMRegressor

            X, y = self._check_xy(X, y)
            self._n_features = X.shape[1]
            self._estimator = LGBMRegressor(
                n_estimators=int(self.params["n_estimators"]),
                learning_rate=float(self.params["learning_rate"]),
                num_leaves=int(self.params["num_leaves"]),
                min_child_samples=int(self.params["min_child_samples"]),
                subsample=float(self.params["subsample"]),
                random_state=self.params["random_state"],
                verbosity=-1,
            )
            self._estimator.fit(X, y)
            self._fitted = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X = self._check_x(X)
            if self._estimator is None:
                raise RuntimeError("LightGBMMetaCombiner.predict called before fit")
            return self._estimator.predict(X)

        def get_weights(self) -> Optional[np.ndarray]:
            if self._estimator is None:
                return None
            try:
                imp = np.asarray(self._estimator.feature_importances_, dtype=float)
                s = imp.sum()
                return imp / s if s > 0 else imp
            except Exception:
                return None
except ImportError:  # pragma: no cover
    pass


# ──────────────────────────────────────────────────────────────────────────
# Per-horizon ensemble facade
# ──────────────────────────────────────────────────────────────────────────


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Local copy of the simplex projection — keeps `meta_combiner` self-contained.

    Logic mirrors `orchestrator.strategies._project_simplex`.
    """

    v = np.asarray(v, dtype=float).reshape(-1)
    if np.all(v <= 0):
        return np.ones_like(v) / len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones_like(v) / len(v)
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(v) / len(v)


@dataclass
class PerHorizonEnsembleConfig:
    """Configuration knobs for `PerHorizonEnsemble`.

    Attributes:
        combiner_name: name in `META_COMBINER_REGISTRY`.
        combiner_params: kwargs forwarded to the combiner constructor.
        share_combiner_across_horizons: if True, fit a single combiner on
            all (window, horizon) pairs flattened.  If False, fit one
            independent combiner per horizon (recommended).
        project_predictions_simplex: if True, project per-row predictions
            onto the unit simplex (only meaningful for linear combiners
            with weight-sum=1 interpretation).
    """

    combiner_name: str = "ridge"
    combiner_params: Dict[str, Any] = field(default_factory=dict)
    share_combiner_across_horizons: bool = False
    project_predictions_simplex: bool = False


class PerHorizonEnsemble:
    """Facade that fits one meta-combiner per forecast horizon.

    Inputs are 3-D arrays shaped `(n_windows, n_models, horizon)` for
    base predictions and `(n_windows, horizon)` for the target.  This
    matches the `ValidationData` contract in
    `orchestrator/data_contract.py`.

    Args:
        config: `PerHorizonEnsembleConfig` (use the dataclass for type
            safety) or a plain dict.
    """

    def __init__(self, config: Optional[PerHorizonEnsembleConfig] = None) -> None:
        self.cfg = config or PerHorizonEnsembleConfig()
        self._combiners: List[MetaCombinerBase] = []
        self._horizon: Optional[int] = None
        self._n_models: Optional[int] = None
        self._fitted: bool = False

    # ----- Public API -----

    def fit(self, y_preds: np.ndarray, y_true: np.ndarray) -> "PerHorizonEnsemble":
        """Fit per-horizon (or shared) combiners.

        Args:
            y_preds: shape (n_windows, n_models, horizon).
            y_true:  shape (n_windows, horizon).
        """

        y_preds = np.asarray(y_preds, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if y_preds.ndim != 3:
            raise ValueError(f"y_preds must be 3-D, got shape={y_preds.shape}")
        if y_true.ndim != 2:
            raise ValueError(f"y_true must be 2-D, got shape={y_true.shape}")
        n_windows, n_models, horizon = y_preds.shape
        if y_true.shape != (n_windows, horizon):
            raise ValueError(
                f"y_true shape {y_true.shape} incompatible with y_preds {y_preds.shape}"
            )

        self._horizon = horizon
        self._n_models = n_models
        self._combiners = []

        if self.cfg.share_combiner_across_horizons:
            X = y_preds.transpose(0, 2, 1).reshape(n_windows * horizon, n_models)
            y = y_true.reshape(n_windows * horizon)
            comb = make_meta_combiner(self.cfg.combiner_name, **self.cfg.combiner_params)
            comb.fit(X, y)
            self._combiners = [comb] * horizon
        else:
            for h in range(horizon):
                X_h = y_preds[:, :, h]
                y_h = y_true[:, h]
                comb = make_meta_combiner(
                    self.cfg.combiner_name, **self.cfg.combiner_params
                )
                comb.fit(X_h, y_h)
                self._combiners.append(comb)

        self._fitted = True
        return self

    def predict(self, y_preds: np.ndarray) -> np.ndarray:
        """Combine predictions per horizon.

        Args:
            y_preds: shape (n_windows, n_models, horizon).

        Returns:
            shape (n_windows, horizon).
        """

        if not self._fitted:
            raise RuntimeError("PerHorizonEnsemble.predict called before fit")
        y_preds = np.asarray(y_preds, dtype=float)
        if y_preds.ndim != 3:
            raise ValueError(f"y_preds must be 3-D, got shape={y_preds.shape}")
        n_windows, n_models, horizon = y_preds.shape
        if horizon != self._horizon or n_models != self._n_models:
            raise ValueError(
                f"shape mismatch at predict: expected (*, {self._n_models}, {self._horizon}), "
                f"got (*, {n_models}, {horizon})"
            )

        out = np.empty((n_windows, horizon), dtype=float)
        for h in range(horizon):
            comb = self._combiners[h]
            X_h = y_preds[:, :, h]
            if self.cfg.project_predictions_simplex:
                w = comb.get_weights()
                if w is not None and w.size == n_models:
                    wp = _project_simplex(w)
                    out[:, h] = X_h @ wp
                    continue
            out[:, h] = comb.predict(X_h)
        return out

    def get_weights_per_horizon(self) -> Optional[np.ndarray]:
        """Return (horizon, n_models) weights matrix when all combiners
        expose weights; otherwise `None`.
        """

        if not self._fitted:
            return None
        rows: List[np.ndarray] = []
        for comb in self._combiners:
            w = comb.get_weights()
            if w is None:
                return None
            rows.append(np.asarray(w, dtype=float))
        return np.vstack(rows)

    # ----- Convenience constructors -----

    @classmethod
    def from_name(
        cls,
        combiner_name: str,
        combiner_params: Optional[Dict[str, Any]] = None,
        share_across_horizons: bool = False,
        project_simplex: bool = False,
    ) -> "PerHorizonEnsemble":
        cfg = PerHorizonEnsembleConfig(
            combiner_name=combiner_name,
            combiner_params=dict(combiner_params or {}),
            share_combiner_across_horizons=bool(share_across_horizons),
            project_predictions_simplex=bool(project_simplex),
        )
        return cls(cfg)
