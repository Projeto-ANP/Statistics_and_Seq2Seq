"""Buscas deterministicas sobre o mesmo catalogo do agente (sem LLM).

Objetivo cientifico: isolar o efeito do LLM. A ablacao do trabalho compara o
agente contra "nenhuma busca" (a melhor semente). Se uma busca simples — gulosa,
aleatoria ou bandit — com o MESMO orcamento empatar ou vencer o agente, entao o
que nao agrega e a busca em geral (e o LLM e apenas um search caro). Se o agente
vencer as buscas simples, a contribuicao do LLM fica isolada.

Espaco de acoes identico ao catalogo final do agente (catalog v2, com as
receitas extra ligadas), para que a unica diferenca entre os bracos seja o
mecanismo de busca:
    pools   : full, select_stable k, select_top_k k, prune_redundant, select_forward
    pesos   : inverse, error_trend, meta_model (escopo por conjunto), best_per_horizon,
              topk_per_horizon, forward, ols*
    combinar: mean, median, trimmed_mean, dba, weighted, best_single

(*) `ols` so entra quando ha janelas suficientes (`min_windows_for_ols`), igual ao
catalogo do agente; com as tres janelas do protocolo, nunca entra.
"""

from __future__ import annotations

import json

import numpy as np

from orchestrator_react import tools as T
from orchestrator_react.state import FULL_POOL
from orchestrator_react.weighting import per_model_error

COMBINES = ("mean", "median", "trimmed_mean", "dba", "weighted", "best_single")
FAMILIES = COMBINES

_WEIGHT_METHODS = {
    "inverse_error": "inverse",
    "error_trend": "trend",
    "pooled_meta_model": "meta",
    "feature_based": "meta",
    "best_per_horizon": "best_per_horizon",
    "topk_mean_per_horizon": "topk_per_horizon",
    "forward_selection": "forward",
    "ols": "ols",
}


def _spec_key(spec):
    return json.dumps(spec, sort_keys=True, default=str)


def allowed_weight_methods(state):
    methods = ["inverse", "trend", "meta", "best_per_horizon", "topk_per_horizon", "forward"]
    if state.n_windows >= int(getattr(state.config, "min_windows_for_ols", 5)):
        methods.append("ols")
    return tuple(methods)


def candidate_pools(state):
    pools = [("full", FULL_POOL)]
    for k in (3, 5, 7, 9, 11, 13, 15):
        if k < state.n_models:
            pools.append((f"stable{k}", T.select_stable(state, k)["pool"]))
    for k in (3, 5, 8):
        if k < state.n_models:
            pools.append((f"topk{k}", T.select_top_k(state, k)["pool"]))
    try:
        pools.append(
            ("pruned", T.prune_redundant(state, pool=FULL_POOL, corr_threshold=0.95)["pool"])
        )
    except Exception:
        pass
    try:
        pools.append(("forward", T.select_forward(state, max_steps=20, metric="rmse")["pool"]))
    except Exception:
        pass
    return pools


class SpecFactory:
    """Monta strategies validas (com handles registrados no state)."""

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
        if method == "inverse":
            h = T.weights_inverse_error(st, pool=pool)["weights"]
        elif method == "trend":
            h = T.weights_error_trend(st, pool=pool)["weights"]
        elif method == "meta":
            scope = "dataset" if getattr(st, "pooled_meta_model", None) is not None else "series"
            h = T.weights_meta_model(st, pool=pool, scope=scope)["weights"]
        elif method == "best_per_horizon":
            h = T.weights_best_per_horizon(st, pool=pool)["weights"]
        elif method == "topk_per_horizon":
            h = T.weights_topk_per_horizon(st, pool=pool, k=3)["weights"]
        elif method == "forward":
            h = T.weights_forward(st, pool=pool, max_steps=20)["weights"]
        elif method == "ols":
            h = T.weights_ols(st, pool=pool)["weights"]
        else:
            raise ValueError(f"metodo de peso desconhecido: {method!r}")
        self._weights[key] = h
        return h

    def top_models(self, pool, n=3):
        st = self.state
        idx = st.get_pool(pool)
        err = per_model_error(st.y_true, st.y_preds[:, idx, :], metric="rmse")
        order = np.argsort(err)[: max(1, int(n))]
        return [st.model_names[idx[int(i)]] for i in order]

    def spec_for(self, combine, pool, model=None, method=None):
        st = self.state
        if combine == "best_single":
            if model is None:
                model = st.model_names[int(self.rng.integers(st.n_models))]
            return {"combine": "best_single", "model": model}
        if combine == "weighted":
            if method is None:
                method = str(self.rng.choice(allowed_weight_methods(st)))
            return {
                "combine": "weighted",
                "pool": pool,
                "weights": self.weight_handle(method, pool),
            }
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
        if family == "best_single":
            return {"combine": "best_single", "model": st.model_names[i % st.n_models]}
        if family == "weighted":
            methods = allowed_weight_methods(st)
            method = methods[i % len(methods)]
            pool = self.pools[(i // max(1, len(methods))) % len(self.pools)][1]
            return self.spec_for("weighted", pool, method=method)
        pool = self.pools[i % len(self.pools)][1]
        return self.spec_for(family, pool)


def _neighbors(state, factory, spec):
    """Vizinhos de 1 passo, em ordem deterministica."""
    combine = str(spec.get("combine", "mean"))
    pool = str(spec.get("pool", FULL_POOL))
    pools = [p for _label, p in factory.pools]
    out = []

    if combine == "best_single":
        for model in factory.top_models(FULL_POOL, n=5):
            if model != spec.get("model"):
                out.append({"combine": "best_single", "model": model})
        for p in pools:
            for c in ("mean", "median", "trimmed_mean", "dba"):
                out.append(
                    {
                        "combine": c,
                        "pool": p,
                        **({"trim_pct": 0.2} if c == "trimmed_mean" else {}),
                    }
                )
            for m in allowed_weight_methods(state):
                out.append({"combine": "weighted", "pool": p, "weights": factory.weight_handle(m, p)})
        return out

    # mesmo metodo em outros pools
    for p in pools:
        if p == pool:
            continue
        if combine == "weighted":
            recipe = state.get_weights_recipe(spec["weights"])
            method = _WEIGHT_METHODS.get(recipe.method)
            if method:
                out.append(
                    {"combine": "weighted", "pool": p, "weights": factory.weight_handle(method, p)}
                )
        else:
            out.append(
                {
                    "combine": combine,
                    "pool": p,
                    **({"trim_pct": spec.get("trim_pct", 0.2)} if combine == "trimmed_mean" else {}),
                }
            )

    # outros metodos no mesmo pool
    for c in ("mean", "median", "trimmed_mean", "dba"):
        if c != combine:
            out.append({"combine": c, "pool": pool, **({"trim_pct": 0.2} if c == "trimmed_mean" else {})})
    for m in allowed_weight_methods(state):
        out.append({"combine": "weighted", "pool": pool, "weights": factory.weight_handle(m, pool)})
    for model in factory.top_models(pool, n=3):
        out.append({"combine": "best_single", "model": model})
    return out


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


def run_greedy(state, budget):
    factory = SpecFactory(state, seed=0)
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
    counts = {f: 0 for f in FAMILIES}
    sums = {f: 0.0 for f in FAMILIES}
    pulls = {f: 0 for f in FAMILIES}
    seen = {_spec_key(a.spec) for a in state.attempts}
    anchor = state.best_attempt()
    scale = (
        abs(float(anchor.score))
        if anchor is not None and np.isfinite(anchor.score)
        else 1.0
    ) or 1.0
    n = 0
    guard = 0
    while n < int(budget) and guard < 600:
        guard += 1
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


def run_method(state, method, budget):
    """Roda um braco de busca. Retorna o numero de avaliacoes novas gastas."""
    method = str(method).strip().lower()
    if method == "seeds":
        return 0
    if method == "random":
        return run_random(state, budget)
    if method == "greedy":
        return run_greedy(state, budget)
    if method == "ucb":
        return run_ucb(state, budget)
    raise ValueError(f"metodo desconhecido: {method!r} (use seeds,random,greedy,ucb)")
