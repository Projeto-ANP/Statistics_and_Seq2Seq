"""Application state: raw data, handles and attempt history.

Section 3.2, principle 1: the full series and the forecast matrices live here and
**never** enter the prompt. The tools in `tools.py` read this state and return
compact summaries.

Principle 3: `evaluate_strategy` is the only way a strategy enters the history, and
it always goes through the validation-window backtest first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from orchestrator_react import metrics as M
from orchestrator_react.combiners import COMBINE_METHODS, apply_combination
from orchestrator_react.config import ReactConfig
from orchestrator_react.weighting import WeightsRecipe, resolve_recipe, summarize_weights


FULL_POOL = "pool_full"


@dataclass
class Attempt:
    """A strategy evaluated by backtest. Enters the ranked history."""

    attempt_id: str
    spec: Dict[str, Any]
    origin: str  # "baseline" | "agent"
    aggregate: Dict[str, float]
    per_window: List[Dict[str, float]]
    score: float
    rationale: str = ""
    iteration: Optional[int] = None
    n_models: int = 0

    def brief(self, include_rationale: bool = True) -> Dict[str, Any]:
        """Compact history row, exactly as the agent sees it."""
        out: Dict[str, Any] = {
            "id": self.attempt_id,
            "strategy": _spec_label(self.spec),
            "n_models": self.n_models,
            "score": round(float(self.score), 4),
            "rmse": _r(self.aggregate.get("RMSE")),
            "smape": _r(self.aggregate.get("SMAPE")),
            "pocid": _r(self.aggregate.get("POCID"), 1),
            "rmse_per_window": [_r(w.get("RMSE")) for w in self.per_window],
            "origin": self.origin,
        }
        if include_rationale and self.rationale:
            out["rationale"] = self.rationale[:220]
        return out


def _r(x: Any, nd: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _spec_label(spec: Dict[str, Any]) -> str:
    method = spec.get("combine", "?")
    bits = [str(method)]
    if spec.get("pool") and spec.get("pool") != FULL_POOL:
        bits.append(f"pool={spec['pool']}")
    if spec.get("weights"):
        bits.append(f"w={spec['weights']}")
    if method == "trimmed_mean":
        bits.append(f"trim={spec.get('trim_pct', 0.2)}")
    if method == "best_single":
        bits.append(str(spec.get("model", "?")))
    return " ".join(bits)


def _canonical(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)


class ReactState:
    """Owner of the data and of the evaluation protocol.

    Args:
        y_true: `(n_windows, horizon)` — actual values of the validation windows.
        y_preds: `(n_windows, n_models, horizon)` — pool forecasts on those windows.
        test_preds: `(n_models, horizon)` — final test forecasts (no y_true).
        model_names: names in the same order as the model axis.
        train_series: historical series without the test period (may be None).
        config: run configuration.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_preds: np.ndarray,
        test_preds: np.ndarray,
        model_names: Sequence[str],
        train_series: Optional[np.ndarray] = None,
        config: Optional[ReactConfig] = None,
        dataset_index: Optional[int] = None,
        freq: str = "",
        seasonal_period: Optional[int] = None,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_preds = np.asarray(y_preds, dtype=float)
        self.test_preds = np.asarray(test_preds, dtype=float)
        self.model_names = [str(m) for m in model_names]
        self.train_series = None if train_series is None else np.asarray(train_series, dtype=float)
        self.config = config or ReactConfig()
        self.dataset_index = dataset_index
        self.freq = freq
        self.seasonal_period = seasonal_period or self.config.seasonal_period

        self._validate_shapes()

        # handle registries
        self.pools: Dict[str, List[int]] = {FULL_POOL: list(range(self.n_models))}
        self.pool_meta: Dict[str, Dict[str, Any]] = {
            FULL_POOL: {"origin": "full", "k": self.n_models}
        }
        self.weights: Dict[str, WeightsRecipe] = {}
        self._pool_seq = 0
        self._weight_seq = 0

        # history and trace
        self.attempts: List[Attempt] = []
        self._attempt_by_spec: Dict[str, Attempt] = {}
        self._attempt_seq = 0
        self.tools_called: List[Dict[str, Any]] = []
        self.tool_errors: List[Dict[str, Any]] = []

        self._baseline_agg: Optional[Dict[str, float]] = None
        self._contiguous: Optional[bool] = None

    # ── validation ───────────────────────────────────────────────────────────

    def _validate_shapes(self) -> None:
        if self.y_true.ndim != 2:
            raise ValueError(f"y_true must be (n_windows, horizon), got {self.y_true.shape}")
        if self.y_preds.ndim != 3:
            raise ValueError(
                f"y_preds must be (n_windows, n_models, horizon), got {self.y_preds.shape}"
            )
        if self.test_preds.ndim != 2:
            raise ValueError(f"test_preds must be (n_models, horizon), got {self.test_preds.shape}")
        nw, h = self.y_true.shape
        if self.y_preds.shape[0] != nw or self.y_preds.shape[2] != h:
            raise ValueError(
                f"y_preds {self.y_preds.shape} incompatible with y_true {self.y_true.shape}"
            )
        if self.test_preds.shape[0] != self.y_preds.shape[1]:
            raise ValueError(
                f"test_preds has {self.test_preds.shape[0]} models, "
                f"y_preds has {self.y_preds.shape[1]}"
            )
        if len(self.model_names) != self.y_preds.shape[1]:
            raise ValueError(f"{len(self.model_names)} names for {self.y_preds.shape[1]} models")

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def n_windows(self) -> int:
        return int(self.y_true.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.y_true.shape[1])

    @property
    def n_models(self) -> int:
        return int(self.y_preds.shape[1])

    def windows_are_contiguous(self, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
        """Do the validation windows tile a contiguous stretch of the series?

        Concatenating the windows is only legitimate if they are adjacent in time,
        which is how the generation pipeline peels them off (`series[:-h]`, then
        again, ...). When `train_series` is available this is verified rather than
        assumed: the concatenation must equal its tail. Verified on real ANP data
        with an exact match.
        """
        if self._contiguous is not None:
            return self._contiguous
        if self.train_series is None or self.n_windows < 2:
            self._contiguous = True  # nothing to contradict it
            return self._contiguous
        concat = self.y_true.reshape(-1)
        if self.train_series.size < concat.size:
            self._contiguous = False
            return self._contiguous
        tail = self.train_series[-concat.size :]
        self._contiguous = bool(np.allclose(concat, tail, rtol=rtol, atol=atol, equal_nan=True))
        return self._contiguous

    def model_index(self, name: str) -> int:
        try:
            return self.model_names.index(str(name))
        except ValueError:
            raise KeyError(f"unknown model: {name!r}. Available: {self.model_names}") from None

    # ── pool handles ─────────────────────────────────────────────────────────

    def register_pool(self, indices: Sequence[int], origin: str, **meta: Any) -> str:
        idx = sorted({int(i) for i in indices})
        if not idx:
            raise ValueError("empty pool")
        bad = [i for i in idx if not (0 <= i < self.n_models)]
        if bad:
            raise ValueError(f"model indices out of range: {bad}")
        # Reuse an identical handle so the name space does not blow up.
        for handle, existing in self.pools.items():
            if existing == idx:
                return handle
        self._pool_seq += 1
        handle = f"pool{self._pool_seq}"
        self.pools[handle] = idx
        self.pool_meta[handle] = {"origin": origin, "k": len(idx), **meta}
        return handle

    def get_pool(self, handle: str) -> List[int]:
        if handle not in self.pools:
            raise KeyError(f"unknown pool: {handle!r}. Available: {sorted(self.pools)}")
        return list(self.pools[handle])

    def pool_names(self, handle: str) -> List[str]:
        return [self.model_names[i] for i in self.get_pool(handle)]

    # ── weight handles ───────────────────────────────────────────────────────

    def register_weights(self, recipe: WeightsRecipe) -> str:
        self.get_pool(recipe.pool_handle)  # validates
        self._weight_seq += 1
        handle = f"w{self._weight_seq}"
        idx = self.get_pool(recipe.pool_handle)
        fit = self._fit_windows(recipe.fit_windows, exclude=None)
        resolved, meta = resolve_recipe(recipe, self.y_true[fit], self.y_preds[np.ix_(fit, idx)])
        recipe.resolved = np.asarray(resolved, dtype=float)
        recipe.meta = {**meta, "fit_windows": [int(i) for i in fit]}
        self.weights[handle] = recipe
        return handle

    def get_weights_recipe(self, handle: str) -> WeightsRecipe:
        if handle not in self.weights:
            raise KeyError(f"unknown weights handle: {handle!r}. Available: {sorted(self.weights)}")
        return self.weights[handle]

    def weights_summary(self, handle: str) -> Dict[str, Any]:
        recipe = self.get_weights_recipe(handle)
        return summarize_weights(recipe.resolved, self.pool_names(recipe.pool_handle))

    def resolved_weights_map(self, handle: str) -> Dict[str, Any]:
        """Numeric weights keyed by model name — feeds `weights_handle_resolved`."""
        recipe = self.get_weights_recipe(handle)
        names = self.pool_names(recipe.pool_handle)
        w = np.asarray(recipe.resolved, dtype=float)
        if w.ndim == 1:
            return {"per_horizon": False, "weights": {n: float(v) for n, v in zip(names, w)}}
        return {
            "per_horizon": True,
            "weights": {
                str(h): {n: float(w[j, h]) for j, n in enumerate(names)}
                for h in range(w.shape[1])
            },
        }

    # ── anti-leakage protocol ────────────────────────────────────────────────

    def _fit_windows(
        self, requested: Optional[Sequence[int]], exclude: Optional[int]
    ) -> List[int]:
        """Windows used to fit weights.

        - `requested=None` => all available windows.
        - `exclude=i` applies the backtest protocol when evaluating window `i`:
            * `expanding` (default): only windows strictly before `i`.
            * `loo`: all windows except `i`.
          This guarantees no strategy ever sees the target it is forecasting.
        """
        pool = list(range(self.n_windows)) if requested is None else [int(i) for i in requested]
        pool = [i for i in pool if 0 <= i < self.n_windows]
        if exclude is None:
            return pool
        if self.config.backtest_mode == "loo":
            return [i for i in pool if i != exclude]
        return [i for i in pool if i < exclude]

    # ── strategy execution ───────────────────────────────────────────────────

    def normalize_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and canonicalises a strategy spec."""
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object")
        method = str(spec.get("combine", "")).strip().lower()
        if method not in COMBINE_METHODS:
            raise ValueError(f"combine={method!r} is invalid. Valid: {list(COMBINE_METHODS)}")

        out: Dict[str, Any] = {"combine": method}

        if method == "best_single":
            name = spec.get("model") or spec.get("model_id")
            if not name:
                raise ValueError("combine='best_single' requires the 'model' field")
            self.model_index(str(name))  # validates
            out["model"] = str(name)
            out["pool"] = FULL_POOL
            return out

        pool = str(spec.get("pool") or FULL_POOL)
        self.get_pool(pool)
        out["pool"] = pool

        if method == "weighted":
            wh = spec.get("weights")
            if not wh:
                raise ValueError("combine='weighted' requires the 'weights' field (a handle)")
            recipe = self.get_weights_recipe(str(wh))
            if recipe.pool_handle != pool:
                raise ValueError(
                    f"weights {wh!r} were computed over {recipe.pool_handle!r}, "
                    f"but the strategy uses pool {pool!r}"
                )
            out["weights"] = str(wh)
        elif method == "trimmed_mean":
            trim = float(spec.get("trim_pct", 0.2))
            if not (0.0 <= trim < 0.5):
                raise ValueError(f"trim_pct={trim} outside [0, 0.5)")
            out["trim_pct"] = trim
        elif method == "dba":
            out["dba_max_iter"] = int(spec.get("dba_max_iter", 30))
        return out

    def _combine_window(
        self, spec: Dict[str, Any], preds: np.ndarray, exclude_window: Optional[int]
    ) -> np.ndarray:
        """Applies the strategy to one `(n_pool, horizon)` window matrix."""
        method = spec["combine"]
        if method == "best_single":
            return apply_combination(preds, "best_single", model_pos=0)

        weights = None
        if method == "weighted":
            recipe = self.get_weights_recipe(spec["weights"])
            idx = self.get_pool(recipe.pool_handle)
            fit = self._fit_windows(recipe.fit_windows, exclude=exclude_window)
            if not fit:
                # No admissible history (window 0 in expanding mode): uniform weights.
                weights = np.ones(len(idx), dtype=float) / len(idx)
            else:
                weights, _ = resolve_recipe(
                    recipe, self.y_true[fit], self.y_preds[np.ix_(fit, idx)]
                )
        return apply_combination(
            preds,
            method,
            weights=weights,
            trim_pct=float(spec.get("trim_pct", 0.2)),
            dba_max_iter=int(spec.get("dba_max_iter", 30)),
        )

    def backtest(self, spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Runs the strategy over the validation windows. Returns `(n_windows, horizon)`."""
        spec = self.normalize_spec(spec)
        idx = (
            [self.model_index(spec["model"])]
            if spec["combine"] == "best_single"
            else self.get_pool(spec["pool"])
        )
        out = np.full((self.n_windows, self.horizon), np.nan, dtype=float)
        for i in range(self.n_windows):
            window_preds = self.y_preds[i][idx, :]
            out[i, :] = self._combine_window(spec, window_preds, exclude_window=i)
        return out, spec

    def apply_to_test(self, spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Phase 4: applies the strategy to the test forecasts.

        Uses `apply_combination` — the same function as the backtest. The only
        difference is that weights are fit on **all** windows the recipe asks for,
        since there is no target window to exclude (the test is blind).
        """
        spec = self.normalize_spec(spec)
        debug: Dict[str, Any] = {"method": spec["combine"], "spec": dict(spec)}

        if spec["combine"] == "best_single":
            pos = self.model_index(spec["model"])
            debug["chosen_model"] = spec["model"]
            return self.test_preds[pos, :].copy(), debug

        idx = self.get_pool(spec["pool"])
        debug["pool_models"] = [self.model_names[i] for i in idx]
        preds = self.test_preds[idx, :]
        combined = self._combine_window(spec, preds, exclude_window=None)
        if spec["combine"] == "weighted":
            debug["weights_handle"] = spec["weights"]
            debug["weights_resolved"] = self.resolved_weights_map(spec["weights"])
        return combined, debug

    # ── metrics and score ────────────────────────────────────────────────────

    def _score_metrics(
        self, combined: np.ndarray
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        cfg = self.config
        per_window = [
            M.all_metrics(self.y_true[i], combined[i], zero=cfg.mape_zero, epsilon=cfg.mape_epsilon)
            for i in range(self.n_windows)
        ]
        agg = M.all_metrics(
            self.y_true.reshape(-1),
            combined.reshape(-1),
            zero=cfg.mape_zero,
            epsilon=cfg.mape_epsilon,
        )
        # POCID is directional within a window; we average across windows, which is
        # the project convention (`pocid_within_sequence`).
        pocids = [w["POCID"] for w in per_window]
        agg["POCID"] = float(np.nanmean(pocids)) if pocids else float("nan")
        return agg, per_window

    def baseline_aggregate(self) -> Dict[str, float]:
        """Score normalisation anchor: plain mean of all models."""
        if self._baseline_agg is None:
            combined, _ = self.backtest({"combine": "mean", "pool": FULL_POOL})
            self._baseline_agg, _ = self._score_metrics(combined)
        return self._baseline_agg

    # ── history ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        spec: Dict[str, Any],
        rationale: str = "",
        origin: str = "agent",
        iteration: Optional[int] = None,
    ) -> Tuple[Attempt, bool]:
        """Evaluates and records. Returns `(attempt, is_new)`.

        Re-evaluating an already tested strategy does not create a duplicate entry —
        it returns the existing one with `is_new=False`, which lets the ReAct loop
        detect repetition.
        """
        combined, norm = self.backtest(spec)
        key = _canonical(norm)
        if key in self._attempt_by_spec:
            return self._attempt_by_spec[key], False

        agg, per_window = self._score_metrics(combined)
        score = M.composite_score(agg, self.baseline_aggregate(), self.config.score_weights())

        self._attempt_seq += 1
        n_models = 1 if norm["combine"] == "best_single" else len(self.get_pool(norm["pool"]))
        attempt = Attempt(
            attempt_id=f"a{self._attempt_seq}",
            spec=norm,
            origin=origin,
            aggregate=agg,
            per_window=per_window,
            score=score,
            rationale=rationale,
            iteration=iteration,
            n_models=n_models,
        )
        self.attempts.append(attempt)
        self._attempt_by_spec[key] = attempt
        return attempt, True

    def ranked_attempts(self) -> List[Attempt]:
        return sorted(
            self.attempts,
            key=lambda a: (float("inf") if not np.isfinite(a.score) else a.score),
        )

    def best_attempt(self) -> Optional[Attempt]:
        ranked = self.ranked_attempts()
        return ranked[0] if ranked else None

    def rank_of(self, attempt: Attempt) -> int:
        return self.ranked_attempts().index(attempt) + 1

    # ── tool trace ───────────────────────────────────────────────────────────

    def log_tool(
        self, name: str, args: Dict[str, Any], ok: bool, error: str = "", kind: str = ""
    ) -> None:
        entry: Dict[str, Any] = {"tool": name, "args": args, "ok": bool(ok)}
        if error:
            entry["error"] = error
            entry["kind"] = kind
        self.tools_called.append(entry)
        if not ok:
            self.tool_errors.append({"tool": name, "args": args, "error": error, "kind": kind})
