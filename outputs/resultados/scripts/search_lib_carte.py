"""Buscas deterministicas no protocolo atual do CARTE (agent-only).

Diferencas em relacao ao `aprovacao_solida/01_busca_nao_llm/search_lib.py`:

- o catalogo e o do CARTE atual: sem `prune_redundant` e sem `best_single`;
- os bracos partem de um historico vazio, como o CARTE agent-only;
- `greedy` e `ucb` avaliam um primeiro candidato aleatorio para iniciar;
- o orcamento e o mesmo do CARTE (`--evals`, default 12).

O espaco de acoes e:
    pools   : full, select_stable k, select_top_k k, select_forward
    pesos   : inverse, error_trend, meta, best_per_horizon,
              topk_per_horizon, forward
    combinar: mean, median, trimmed_mean, dba, weighted
"""
from __future__ import annotations

import json

import numpy as np

from orchestrator_react import tools as T
from orchestrator_react.state import FULL_POOL
from orchestrator_react.weighting import per_model_error

COMBINES = ("mean", "median", "trimmed_mean", "dba", "weighted")
FAMILIES = COMBINES

_WEIGHT_METHODS = {
    "inverse_error": "inverse",
    "error_trend": "trend",
    "pooled_meta_model": "meta",
    "feature_based": "meta",
    "best_per_horizon": "best_per_horizon",
    "topk_mean_per_horizon": "topk_per_horizon",
    "forward_selection": "forward",
}


def _spec_key(spec):
    return json.dumps(spec, sort_keys=True, default=str)


def allowed_weight_methods(state):
    methods = ["inverse", "trend", "best_per_horizon", "topk_per_horizon", "forward"]
    if getattr(state, "pooled_meta_model", None) is not None:
        methods.append("meta")
    return tuple(methods)


def candidate_pools(state):
    pools = [("full", FULL_POOL)]
    for k in (3, 5, 7, 9, 11, 13, 15):
        if k < state.n_models:
            try:
                pools.append((f"stable{k}", T.select_stable(state, k)["pool"]))
            except Exception:
                pass
    for k in (3, 5, 8):
        if k < state.n_models:
            try:
                pools.append((f"topk{k}", T.select_top_k(state, k)["pool"]))
            except Exception:
                pass
    try:
        pools.append(("forward", T.select_forward(state, max_steps=20, metric="rmse")["pool"]))
    except Exception:
        pass
    return pools


class SpecFactory:
    def __init__(self, state, seed=7):
        self.state = state
        self.rng = np.random.default_rng(int(seed))
        self.pools = candidate_pools(state)
        self._weights = {}

    def weight_handle(self, method, pool):
        key = (method, pool)
        if key in self._weights:
            return self._weights[key]
        st = self.state
        try:
            if method == "inverse":
                handle = T.weights_inverse_error(st, pool=pool)["weights"]
            elif method == "trend":
                handle = T.weights_error_trend(st, pool=pool)["weights"]
            elif method == "meta":
                scope = "dataset" if getattr(st, "pooled_meta_model", None) is not None else "series"
                handle = T.weights_meta_model(st, pool=pool, scope=scope)["weights"]
            elif method == "best_per_horizon":
                handle = T.weights_best_per_horizon(st, pool=pool)["weights"]
            elif method == "topk_per_horizon":
                handle = T.weights_topk_per_horizon(st, pool=pool, k=3)["weights"]
            elif method == "forward":
                handle = T.weights_forward(st, pool=pool, max_steps=20)["weights"]
            else:
                raise ValueError(f"metodo de peso desconhecido: {method!r}")
        except Exception:
            return None
        self._weights[key] = handle
        return handle

    def top_models(self, pool, n=3):
        st = self.state
        idx = st.get_pool(pool)
        err = per_model_error(st.y_true, st.y_preds[:, idx, :], metric="rmse")
        order = np.argsort(err)[: max(1, int(n))]
        return [st.model_names[idx[int(i)]] for i in order]

    def spec_for(self, combine, pool, method=None):
        st = self.state
        if combine == "weighted":
            methods = list(allowed_weight_methods(st))
            if method is None:
                method = str(self.rng.choice(methods))
            methods = [method] + [m for m in methods if m != method]
            for candidate in methods:
                handle = self.weight_handle(candidate, pool)
                if handle is not None:
                    return {"combine": "weighted", "pool": pool, "weights": handle}
            return {"combine": "mean", "pool": pool}
        spec = {"combine": combine, "pool": pool}
        if combine == "trimmed_mean":
            spec["trim_pct"] = 0.2
        return spec

    def random_spec(self):
        combine = str(self.rng.choice(COMBINES))
        _label, pool = self.pools[int(self.rng.integers(len(self.pools)))]
        return self.spec_for(combine, pool)

    def family_spec(self, family, i):
        st = self.state
        if family == "weighted":
            methods = list(allowed_weight_methods(st))
            method = methods[i % len(methods)]
            pool = self.pools[(i // max(1, len(methods))) % len(self.pools)][1]
            return self.spec_for("weighted", pool, method=method)
        pool = self.pools[i % len(self.pools)][1]
        return self.spec_for(family, pool)


def _neighbors(state, factory, spec):
    combine = str(spec.get("combine", "mean"))
    pool = str(spec.get("pool", FULL_POOL))
    pools = [p for _label, p in factory.pools]
    out = []

    for p in pools:
        if p == pool:
            continue
        if combine == "weighted":
            recipe = state.get_weights_recipe(spec["weights"])
            method = _WEIGHT_METHODS.get(recipe.method)
            if method:
                out.append(factory.spec_for("weighted", p, method=method))
        else:
            out.append(factory.spec_for(combine, p))

    for c in ("mean", "median", "trimmed_mean", "dba"):
        if c != combine:
            out.append(factory.spec_for(c, pool))
    for m in allowed_weight_methods(state):
        out.append(factory.spec_for("weighted", pool, method=m))
    return out


def _initial_attempt(state, factory, rationale):
    """Avalia um primeiro candidato quando o historico esta vazio."""
    spec = factory.random_spec()
    attempt, _ = state.evaluate(spec, rationale=rationale, origin="search")
    return attempt


def run_random(state, budget, seed=7):
    factory = SpecFactory(state, seed=seed)
    seen = {_spec_key(a.spec) for a in state.attempts}
    n = 0
    guard = 0
    while n < int(budget) and guard < 500:
        guard += 1
        spec = factory.random_spec()
        key = _spec_key(spec)
        if key in seen:
            continue
        seen.add(key)
        state.evaluate(spec, rationale="busca aleatoria", origin="search")
        n += 1
    return n


def run_greedy(state, budget, seed=7):
    factory = SpecFactory(state, seed=seed)
    if not state.attempts:
        if int(budget) <= 0:
            return 0
        _initial_attempt(state, factory, "busca gulosa")
        budget = int(budget) - 1
    current = state.best_attempt()
    seen = {_spec_key(a.spec) for a in state.attempts}
    n = 0
    while n < int(budget) and current is not None:
        improved = False
        for spec in _neighbors(state, factory, current.spec):
            if n >= int(budget):
                break
            key = _spec_key(spec)
            if key in seen:
                continue
            seen.add(key)
            attempt, _ = state.evaluate(spec, rationale="busca gulosa", origin="search")
            n += 1
            if np.isfinite(attempt.score) and (
                not np.isfinite(current.score) or attempt.score < current.score
            ):
                current = attempt
                improved = True
                break
        if not improved:
            break
    return n


def run_ucb(state, budget, seed=7):
    factory = SpecFactory(state, seed=seed)
    if not state.attempts:
        if int(budget) <= 0:
            return 0
        _initial_attempt(state, factory, "busca ucb")
        budget = int(budget) - 1
    counts = {f: 0 for f in FAMILIES}
    sums = {f: 0.0 for f in FAMILIES}
    pulls = {f: 0 for f in FAMILIES}
    seen = {_spec_key(a.spec) for a in state.attempts}
    n = 0
    guard = 0
    while n < int(budget) and guard < 600:
        guard += 1
        anchor = state.best_attempt()
        scale = abs(float(anchor.score)) if anchor is not None and np.isfinite(anchor.score) else 1.0
        scale = scale or 1.0
        total = max(1, n)
        arm = None
        best_u = -np.inf
        for f in FAMILIES:
            if counts[f] == 0:
                u = np.inf
            else:
                u = sums[f] / counts[f] + 1.0 * scale * np.sqrt(np.log(total + 1) / counts[f])
            if u > best_u:
                best_u = u
                arm = f
        spec = factory.family_spec(arm, pulls[arm])
        pulls[arm] += 1
        key = _spec_key(spec)
        if key in seen:
            continue
        seen.add(key)
        attempt, _ = state.evaluate(spec, rationale="busca ucb", origin="search")
        if anchor is not None and np.isfinite(attempt.score):
            reward = (float(anchor.score) - float(attempt.score)) / scale
        else:
            reward = 0.0
        counts[arm] += 1
        sums[arm] += reward
        n += 1
    return n


def run_method(state, method, budget, seed=7):
    method = str(method).strip().lower()
    if method == "random":
        return run_random(state, budget, seed=seed)
    if method == "greedy":
        return run_greedy(state, budget, seed=seed)
    if method == "ucb":
        return run_ucb(state, budget, seed=seed)
    raise ValueError(f"metodo desconhecido: {method!r} (use random,greedy,ucb)")
