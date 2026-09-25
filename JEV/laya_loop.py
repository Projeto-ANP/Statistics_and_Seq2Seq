"""LAYA-driven combination loop — um "System One" no lugar do LLM gerador.

Em vez de gerar Thought/Action/Action Input em texto livre, o classificador
recebe um ESTADO compacto (card da série + card do pool + histórico ranqueado)
e uma pergunta `choice` cujas opções são AÇÕES CONCRETAS já montadas a partir do
estado (estratégias determinísticas sobre os pools existentes + aceitar).

Toda ação escolhida passa por `state.evaluate` — ou seja, o mesmo contrato do
loop ReAct original: a estratégia só entra no histórico depois do backtest nas
janelas de validação, e o resultado final é sempre a melhor tentativa do
histórico (princípio 5 preservado). A diferença é que aqui o espaço de decisão
é um MENU enumerado, não texto livre — não há geração, não há parse, não há
argumento inválido.

Uso (servidor, com `pip install laya`):

    python JEV/run_laya.py --datasets ETTM2 NN5_WEEKLY_DATASET --version laya_v0
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# O probe de TF do transformers pode travar na construção do modelo; o próprio
# README do laya manda usar USE_TF=0 nesse caso.
os.environ.setdefault("USE_TF", "0")

from orchestrator_react import prompts as P
from orchestrator_react import tools as T
from orchestrator_react.state import FULL_POOL, Attempt, ReactState

#: Métodos oferecidos no menu por pool. dba fica de fora do menu v0 (mais lento
#: com tslearn e redundante com as sementes, que já o cobrem).
MENU_METHODS = ("mean", "median", "trimmed_mean")

#: Quantos pools entram no menu (pool_full + os 3 pools estáveis menores).
MAX_POOLS_IN_MENU = 4

#: Quantos modelos viram candidatos best_single.
TOP_K_BEST_SINGLE = 3

_INSTRUCTIONS = (
    "You are searching for the best forecast combination, scored on 3 validation "
    "windows (lower score is better). Pick ONE strategy to evaluate, or accept. "
    "Prefer strategies that are NOT already in the history. If the history leader "
    "looks strong and no untested option is promising, accept."
)


class LayaAgent:
    """Wrapper mínimo sobre `laya.load`, com carga lazy e mensagens claras."""

    def __init__(self, checkpoint: str = "english", max_len: Optional[int] = None) -> None:
        if checkpoint not in ("english", "multilingual"):
            raise ValueError(f"checkpoint deve ser 'english' ou 'multilingual', got {checkpoint!r}")
        self.checkpoint = checkpoint
        self.max_len = max_len
        self._agent: Any = None
        self.name = f"laya-{checkpoint}"

    def load(self) -> None:
        if self._agent is not None:
            return
        try:
            import laya  # noqa: PLC0415 — importado só onde é usado
        except ImportError as exc:
            raise RuntimeError(
                "laya não está instalado: pip install laya  (e USE_TF=0 se a "
                "construção do modelo travar na importação do transformers)"
            ) from exc
        if self.checkpoint == "english":
            self._agent = laya.load("convaiinnovations/laya")
        else:
            self._agent = laya.load("convaiinnovations/laya", subfolder="multilingual")

    def predict(self, state_text: str, question: Dict[str, Any]) -> Dict[str, Any]:
        self.load()
        try:
            return self._agent.predict(state_text, question, max_len=self.max_len)
        except TypeError:
            # o agente direto pode não aceitar max_len; sem limite ele usa o
            # default do checkpoint (512 english / 1024 multilingual)
            return self._agent.predict(state_text, question)


@dataclass
class LayaLoopResult:
    final_attempt: Optional[Attempt]
    iterations_used: int = 0
    stop_reason: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "final_attempt": self.final_attempt.attempt_id if self.final_attempt else None,
            "strategy": self.final_attempt.spec if self.final_attempt else None,
            "iterations_used": self.iterations_used,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
        }


# ──────────────────────────────────────────────────────────────────────────────
# estado compacto e menu
# ──────────────────────────────────────────────────────────────────────────────


def _spec_key(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)


def _history_keys(state: ReactState) -> set:
    return {_spec_key(a.spec) for a in state.attempts}


def build_state_text(
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    state: ReactState,
    budget: int = 1400,
) -> str:
    """Estado que o classificador lê — só cartões compactos + histórico.

    Deliberadamente SEM o system prompt do agente ReAct: as opções carregam
    seus próprios critérios na pergunta, e o checkpoint english tem 512 tokens
    de orçamento. `budget` é em caracteres (≈4 chars/token).
    """
    ranked = state.ranked_attempts()[:6]
    hist = [a.brief(include_rationale=False) for a in ranked]
    payload = {
        "series": P._slim_series_card(series_card),
        "pool": P._slim_pool_card(pool_card),
        "history_best_first": hist,
    }
    return P._compact(payload, limit=budget)


def build_candidates(state: ReactState) -> Dict[str, Dict[str, Any]]:
    """Ações concretas do menu: estratégia -> label legível.

    Pools ordenados: pool_full primeiro, depois por tamanho (k=5,7,9 das
    sementes estáveis). Cada pool × {mean, median, trimmed_mean}, mais os
    top-3 modelos como best_single. `accept` não é estratégia — é tratado no
    loop.
    """
    handles = sorted(
        state.pools,
        key=lambda h: (h != FULL_POOL, len(state.pools.get(h, []) or []), h),
    )
    handles = handles[:MAX_POOLS_IN_MENU]
    known = _history_keys(state)
    cands: Dict[str, Dict[str, Any]] = {}
    for h in handles:
        n = len(state.pools[h])
        for method in MENU_METHODS:
            spec: Dict[str, Any] = {"combine": method, "pool": h}
            if method == "trimmed_mean":
                spec["trim_pct"] = 0.2
            label = f"{method}_{h}"
            cands[label] = {"spec": spec, "n_models": n,
                            "tested": _spec_key(spec) in known}
    try:
        top = T.select_top_k(state, k=min(TOP_K_BEST_SINGLE, state.n_models))
        for entry in top["models"]:
            name = entry if isinstance(entry, str) else entry["model"]
            spec = {"combine": "best_single", "model": str(name)}
            label = f"best_{name}"
            cands[label] = {"spec": spec, "n_models": 1,
                            "tested": _spec_key(spec) in known}
    except Exception:
        pass
    return cands


def _describe(label: str, info: Dict[str, Any]) -> str:
    spec = info["spec"]
    method = spec["combine"]
    n = info["n_models"]
    suffix = " (already tested)" if info["tested"] else ""
    if method == "best_single":
        return f"use only model {spec['model']} as the forecast{suffix}"
    what = {"mean": "average", "median": "median", "trimmed_mean": "trimmed mean"}[method]
    return f"{what} of the {n} models in pool {spec['pool']}{suffix}"


# ──────────────────────────────────────────────────────────────────────────────
# o loop
# ──────────────────────────────────────────────────────────────────────────────


def run_laya_loop(
    state: ReactState,
    agent: LayaAgent,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    max_iterations: int = 12,
    state_budget: int = 1400,
    patience: int = 3,
    on_step: Optional[Any] = None,
) -> LayaLoopResult:
    """Roda o loop de decisão do classificador e devolve a melhor tentativa.

    `state` já vem semeado pela Fase 2 (`pool.run_phase2`). O classificador só
    escolhe entre ações concretas; a execução continua determinística.
    """
    result = LayaLoopResult(final_attempt=state.best_attempt())
    if not state.attempts:
        raise RuntimeError("histórico vazio: rode pool.run_phase2 antes do loop")

    # O classificador não "aprende" com a observation como o ReAct: sem um freio
    # ele repete a mesma escolha até o orçamento acabar. `patience` turnos
    # seguidos sem informação nova encerram o loop (análogo ao early_stop_patience
    # do react_loop).
    patience = max(1, int(patience))
    stale = 0
    last_chosen: Optional[str] = None

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        result.iterations_used = iteration
        cands = build_candidates(state)
        if not cands:
            result.stop_reason = "no_candidates"
            break
        criteria: Dict[str, str] = {label: _describe(label, info) for label, info in cands.items()}
        criteria["accept"] = "stop now and keep the current best strategy"
        stext = build_state_text(series_card, pool_card, state, budget=state_budget)
        question = {
            "next_action": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": criteria,
            }
        }
        try:
            out = agent.predict(stext, question)
            answer = out["answers"]["next_action"]
            chosen = str(answer.get("choice") or "")
            conf = answer.get("confidence", answer.get("probability"))
            probs = answer.get("probabilities")
            usage = out.get("usage")
        except Exception as exc:
            result.errors.append(f"iteration {iteration}: {type(exc).__name__}: {exc}")
            result.stop_reason = "laya_error"
            break

        if chosen == "accept":
            result.stop_reason = "laya_accept"
            break
        if chosen not in cands:
            result.errors.append(f"iteration {iteration}: laya escolheu label fora do menu: {chosen!r}")
            result.stop_reason = "laya_error"
            break

        spec = cands[chosen]["spec"]
        attempt, is_new = state.evaluate(
            spec, rationale=f"laya selected {chosen}", origin="agent", iteration=iteration
        )
        entry = {
            "iteration": iteration,
            "action": chosen,
            "confidence": conf,
            "probability_of_chosen": (probs or {}).get(chosen),
            "probabilities": {k: round(float(v), 4) for k, v in (probs or {}).items()},
            "input_tokens": (usage or {}).get("input_tokens"),
            "score": round(float(attempt.score), 4) if attempt.score == attempt.score else None,
            "rank": state.ranked_attempts().index(attempt) + 1,
            "already_tested": not is_new,
        }
        result.trace.append(entry)
        if on_step is not None:
            try:
                on_step(state.dataset_index, entry)
            except Exception:
                pass

        # ── freio de repetição: turno sem informação nova ───────────────────
        if not is_new or chosen == last_chosen:
            stale += 1
        else:
            stale = 0
        last_chosen = chosen
        if stale >= patience:
            result.stop_reason = f"no new information in {stale} consecutive turns"
            break
    else:
        result.stop_reason = "iteration_budget_exhausted"

    result.final_attempt = state.best_attempt()
    return result
