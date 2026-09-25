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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

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
    "Prefer strategies that are NOT already in the history. Read 'what you tried "
    "so far': if an option you tried did not become the best, try a DIFFERENT kind "
    "of option (different method, different group recipe, different model) instead "
    "of repeating it. A typical good sequence: test a robust option like "
    "median_stable5; if it does not lead, test a weighted option or the best "
    "single model; then accept the leader. Rules: prefer small structured "
    "comparisons over one sweeping decision; if a seeded baseline is leading and "
    "nothing concrete suggests a deviation, accepting it is often the right move; "
    "an option is only valid for the group of models it names; a weighted option "
    "whose weights come out nearly equal gives the same forecast as the plain "
    "average of that group, so do not retry the same idea; when you accept, the "
    "justification must cite observable series characteristics, not just 'lowest "
    "error'."
)


class LayaAgent:
    """Wrapper mínimo sobre `laya.load`, com carga lazy e mensagens claras."""

    def __init__(self, checkpoint: str = "english", max_len: Optional[int] = None) -> None:
        # "english" | "multilingual" | caminho LOCAL de um checkpoint fine-tuned
        # (diretório com model.safetensors + rl_agent_config.json)
        if checkpoint not in ("english", "multilingual"):
            if not (os.path.isdir(checkpoint) or "/" in checkpoint or "\\" in checkpoint):
                raise ValueError(
                    f"checkpoint deve ser 'english', 'multilingual' ou um caminho "
                    f"local de checkpoint fine-tuned, got {checkpoint!r}"
                )
        self.checkpoint = checkpoint
        self.max_len = max_len
        self._agent: Any = None
        self.name = os.path.basename(str(checkpoint).rstrip("/")) if checkpoint not in ("english", "multilingual") else f"laya-{checkpoint}"

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
        elif self.checkpoint == "multilingual":
            self._agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
        else:
            self._agent = laya.load(self.checkpoint)  # diretório local fine-tuned

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
    #: Telemetria completa por turno (para debug/artifacts): estado ANTES,
    #: pergunta com todas as opções, resposta crua, ação executada, resultado,
    #: estado DEPOIS e tempos.
    step_details: List[Dict[str, Any]] = field(default_factory=list)

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


def _validation_landscape(state: ReactState) -> Dict[str, Any]:
    """O que a validação JÁ diz, de forma explícita: líder, gap, melhores modelos.

    O "oráculo de validação" é o líder do histórico — a única verdade disponível
    antes do teste, e é honesto mostrá-la (é o que o argmin usa).
    """
    ranked = state.ranked_attempts()
    leader = ranked[0] if ranked else None
    gap = None
    if len(ranked) > 1 and leader is not None:
        gap = round(float(ranked[1].score - leader.score), 4)
    es = T.error_summary(state, top_n=5)
    return {
        "leader_on_validation": (
            {"id": leader.attempt_id,
             "strategy": leader.brief(include_rationale=False)["strategy"],
             "score": round(float(leader.score), 4)}
            if leader else None
        ),
        "gap_to_second": gap,
        "best_models_by_error": es.get("top", []),
        "note": (
            "the leader is the best strategy found so far on the validation "
            "windows (the validation oracle). A proposal only replaces it if it "
            "scores strictly lower."
        ),
    }


def build_state_text(
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    state: ReactState,
    budget: int = 1400,
    scratchpad: Optional[List[Dict[str, Any]]] = None,
    dataset_card: Optional[Dict[str, Any]] = None,
) -> str:
    """Estado que o classificador lê: cartões + regime + paisagem + histórico.

    O bloco `regime` resume em uma frase as características da série
    (tendência, sazonalidade, estabilidade do ranking) e os campeões — o
    vínculo entre a forma da série e quais modelos tendem a funcionar.
    `validation_landscape` é o que a validação já diz: líder, gap e melhores
    modelos. `scratchpad` são as ações JÁ TENTADAS pelo próprio classificador
    com o resultado. `dataset_card` é o prior cross-series (LOO, validação-only).
    """
    ranked = state.ranked_attempts()[:6]
    hist = [a.brief(include_rationale=False) for a in ranked]
    stab = pool_card.get("ranking_stability") or {}
    tc = (series_card.get("trend_champion") or {}).get("model")
    sc = (series_card.get("seasonality_champion") or {}).get("model")
    regime = (
        f"trend={series_card.get('trend_strength')}, "
        f"seasonal={series_card.get('seasonal_strength')}, "
        f"model ranking {stab.get('verdict', '?')} "
        f"(tau={stab.get('mean_kendall_tau')})"
    )
    payload: Dict[str, Any] = {
        "series": P._slim_series_card(series_card),
        "pool": P._slim_pool_card(pool_card),
        "regime": {
            "summary": regime,
            "trend_champion": tc,
            "seasonality_champion": sc,
        },
        "validation_landscape": _validation_landscape(state),
        "history_best_first": hist,
    }
    if dataset_card:
        payload["dataset_card"] = dataset_card
    if scratchpad:
        payload["what_you_tried_so_far"] = scratchpad
    return P._compact(payload, limit=budget)


def model_evidence(state: ReactState, series_card: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Evidência POR MODELO: erro (médio e por janela), estabilidade de ranking,
    tendência do erro, top-3, campeões de tendência/sazonalidade."""
    es = T.error_summary(state, top_n=state.n_models)
    rs = T.ranking_stability(state)
    by_model = {r["model"]: r for r in es["top"]}
    movers = {m["model"]: m for m in rs.get("biggest_movers", [])}
    top3 = set(rs.get("always_top3", []))
    tc = (series_card.get("trend_champion") or {}).get("model")
    sc = (series_card.get("seasonality_champion") or {}).get("model")
    per_w = np.abs(state.y_preds - state.y_true[:, None, :]).mean(axis=2)  # (W, M)
    out: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(state.model_names):
        e = by_model.get(name)
        if e is None:
            continue
        werr = per_w[:, j]
        if werr[-1] < werr[0] - 1e-12:
            trend = "improving"
        elif werr[-1] > werr[0] + 1e-12:
            trend = "degrading"
        else:
            trend = "flat"
        m = movers.get(name)
        champion = "trend" if name == tc else ("seasonal" if name == sc else None)
        out[name] = {
            "error": e["error"], "rank": e["rank"],
            "per_window": [round(float(v), 4) for v in werr],
            "rank_spread": int(m["rank_spread"]) if m else 0,
            "ranks": m["ranks"] if m else None,
            "always_top3": name in top3,
            "trend": trend,
            "champion": champion,
        }
    return out


def _fmt_model(name: str, ev: Optional[Dict[str, Any]], fmt: str) -> str:
    """Evidência de um modelo no formato 'raw' (números) ou 'text' (resumo)."""
    if ev is None:
        return name
    if fmt == "raw":
        ranks = ev["ranks"] if ev["ranks"] else []
        return (
            f"{name}(err={ev['error']:.4f}, per_window={ev['per_window']}, "
            f"ranks={ranks}, rank_spread={ev['rank_spread']}, {ev['trend']}"
            f"{', ' + ev['champion'] + ' champion' if ev['champion'] else ''})"
        )
    bits = [f"{name}(err={ev['error']:.2f}"]
    if ev["rank"] == 1:
        bits.append("lowest error")
    if ev["always_top3"]:
        bits.append("always top3")
    elif ev["rank_spread"] > 0:
        bits.append("stable rank" if ev["rank_spread"] <= 4 else "unstable rank")
    if ev["trend"] == "improving":
        bits.append("improving")
    elif ev["trend"] == "degrading":
        bits.append("degrading")
    if ev["champion"] == "trend":
        bits.append("trend champion")
    elif ev["champion"] == "seasonal":
        bits.append("seasonality champion")
    return " ".join(bits) + ")"


def _group_recipes(state: ReactState) -> Dict[str, tuple]:
    """Receitas de grupo: (handle, membros). Determinísticas, reusam handles."""
    recipes: Dict[str, tuple] = {"full": (FULL_POOL, state.pool_names(FULL_POOL))}
    for k in (5, 7, 9):
        try:
            h = T.select_stable(state, k=k)["pool"]
            recipes[f"stable{k}"] = (h, state.pool_names(h))
        except Exception:
            pass
    try:
        h = T.select_top_k(state, k=5)["pool"]
        recipes["top5"] = (h, state.pool_names(h))
    except Exception:
        pass
    try:
        h = T.prune_redundant(state, pool=FULL_POOL, corr_threshold=0.95)["pool"]
        recipes["prune"] = (h, state.pool_names(h))
    except Exception:
        pass
    return recipes


#: Métodos do menu v2: combinações diretas + pesos sobre cada receita de grupo.
V2_METHODS = ("mean", "median", "trimmed_mean", "weighted_inverse", "weighted_softmax")

_METHOD_WORDS = {
    "mean": "average",
    "median": "median",
    "trimmed_mean": "trimmed mean (drop 20% from each tail)",
    "weighted_inverse": "weighted average with weights = 1/validation error",
    "weighted_softmax": "weighted average with weights = softmax(-validation error)",
}


def build_candidates(
    state: ReactState,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    no_seeds: bool = False,
    fmt: str = "text",
) -> Dict[str, Dict[str, Any]]:
    """Menu v2: receita de grupo × método, com EVIDÊNCIA por modelo.

    Cada opção é uma ação composta CONCRETA: a receita define o grupo (com a
    lista de membros e a evidência de cada um) e o método define a combinação.
    O classificador só escolhe; a montagem é determinística. `fmt` controla a
    apresentação da evidência: "raw" (números crus) ou "text" (resumo).

    `no_seeds=True` adiciona ações de ferramenta para construir pools do zero.
    """
    known = _history_keys(state)
    ev = model_evidence(state, series_card)
    cands: Dict[str, Dict[str, Any]] = {}

    if no_seeds:
        for k in (5, 7, 9):
            cands[f"select_stable_k{k}"] = {
                "kind": "tool", "tool": "select_stable", "args": {"k": k},
                "n_models": 0, "tested": False,
            }
        cands["select_top_k_k5"] = {
            "kind": "tool", "tool": "select_top_k", "args": {"k": 5},
            "n_models": 0, "tested": False,
        }
        cands["prune_redundant_full"] = {
            "kind": "tool", "tool": "prune_redundant", "args": {"pool": FULL_POOL},
            "n_models": 0, "tested": False,
        }

    recipes = _group_recipes(state)
    for rname, (handle, members) in recipes.items():
        n = len(members)
        for method in V2_METHODS:
            label = f"{method}_{rname}"
            info: Dict[str, Any] = {
                "kind": "weighted" if method.startswith("weighted_") else "strategy",
                "spec": {"combine": method, "pool": handle},
                "n_models": n, "members": members, "evidence": ev,
                "recipe": rname, "tested": False,
            }
            if method == "trimmed_mean":
                info["spec"]["trim_pct"] = 0.2
            if info["kind"] == "weighted":
                info["tested"] = any(
                    a.spec.get("combine") == "weighted" and a.spec.get("pool") == handle
                    for a in state.attempts
                )
            else:
                info["tested"] = _spec_key(info["spec"]) in known
            cands[label] = info

    top_models = sorted(ev, key=lambda name: ev[name]["rank"])[:TOP_K_BEST_SINGLE]
    for name in top_models:
        spec = {"combine": "best_single", "model": name}
        label = f"best_{name}"
        cands[label] = {
            "kind": "strategy", "spec": spec, "n_models": 1,
            "members": [name], "evidence": ev,
            "tested": _spec_key(spec) in known,
        }
    return cands


def build_staged_questions(
    state: ReactState,
    series_card: Dict[str, Any],
    no_seeds: bool = False,
    fmt: str = "text",
) -> Dict[str, Dict[str, Any]]:
    """Turno em ETAPAS CONDICIONAIS: chamada 1 decide o movimento; a chamada 2
    só faz a pergunta adequada ao movimento (receita+método OU modelo OU
    construção). A composição é feita pelo CÓDIGO com os argmax."""
    recipes = _group_recipes(state)
    ev = model_evidence(state, series_card)
    top_single = sorted(ev, key=lambda n: ev[n]["rank"])[:5]

    moves = {
        "combine": "test a combination over a group of models",
        "single": "test one model alone",
    }
    if state.attempts:
        moves["accept"] = "accept the current best strategy"
    if no_seeds:
        moves["build"] = "build a new group of models first (stable/top-k/pruned)"
    return {
        "move": {
            "type": "choice",
            "instructions": "Given the series and the current history, what kind of move should be tested now?",
            "criteria": moves,
        },
        "combine": {
            "recipe": {
                "type": "choice",
                "instructions": "Which group of models?",
                "criteria": {
                    rname: _fmt_recipe(rname, members, ev, fmt)
                    for rname, (_, members) in recipes.items()
                },
            },
            "method": {
                "type": "choice",
                "instructions": "Which combination method over that group?",
                "criteria": {
                    "mean": "plain average",
                    "median": "median (robust to an outlying model)",
                    "trimmed_mean": "trimmed mean (drop 20% from each tail)",
                    "weighted_inverse": "weighted average with weights = 1/validation error",
                    "weighted_softmax": "weighted average with weights = softmax(-validation error)",
                },
            },
        },
        "single": {
            "type": "choice",
            "instructions": "Which model alone?",
            "criteria": {n: _fmt_model(n, ev.get(n), fmt) for n in top_single},
        },
        "build": {
            "type": "choice",
            "instructions": "Which group recipe?",
            "criteria": {
                "stable5": "the 5 models with the most consistent ranking across windows",
                "stable7": "the 7 models with the most consistent ranking across windows",
                "stable9": "the 9 models with the most consistent ranking across windows",
                "top5": "the 5 models with the lowest validation error",
                "prune": "drop near-duplicate models (correlation > 0.95)",
            },
        },
    }


def _compose_staged(
    state: ReactState,
    answers: Dict[str, Any],
    recipes: Dict[str, tuple],
) -> tuple:
    """Traduz as respostas das etapas em uma ação concreta.

    Retorna (kind, spec_ou_tool, label) onde kind ∈ {"accept","tool","strategy",
    "weighted"}.
    """
    move = str((answers.get("move") or {}).get("choice") or "accept")
    if move == "accept":
        return ("accept", None, "accept")
    if move == "build":
        tool = str((answers.get("build") or {}).get("choice") or "stable5")
        if tool in BUILD_TOOLS:
            return ("tool", tool, f"build:{tool}")
        return ("accept", None, "accept")
    if move == "single":
        name = str((answers.get("single") or {}).get("choice") or "")
        if not name:
            return ("accept", None, "accept")
        return ("strategy", {"combine": "best_single", "model": name}, f"best_{name}")
    rname = str((answers.get("recipe") or {}).get("choice") or "full")
    method = str((answers.get("method") or {}).get("choice") or "mean")
    if rname not in recipes:
        rname = "full"
    handle, members = recipes[rname]
    spec: Dict[str, Any] = {"combine": method, "pool": handle}
    if method == "trimmed_mean":
        spec["trim_pct"] = 0.2
    kind = "weighted" if method.startswith("weighted_") else "strategy"
    return (kind, spec, f"{method}_{rname}")


def _fmt_recipe(rname: str, members: list, ev: dict, fmt: str) -> str:
    shown = members[:5]
    mtext = "; ".join(_fmt_model(m, ev.get(m), fmt) for m in shown)
    return f"the {len(members)} models [{mtext}]"


#: Receitas de construção do menu staged (no-seeds) -> ferramenta + args.
BUILD_TOOLS = {
    "stable5": ("select_stable", {"k": 5}),
    "stable7": ("select_stable", {"k": 7}),
    "stable9": ("select_stable", {"k": 9}),
    "top5": ("select_top_k", {"k": 5}),
    "prune": ("prune_redundant", {"pool": FULL_POOL}),
}


def _compose_staged(
    state: ReactState,
    answers: Dict[str, Any],
    recipes: Dict[str, tuple],
) -> tuple:
    """Traduz as respostas das etapas em uma ação concreta.

    Retorna (kind, spec_ou_tool, label) onde kind ∈ {"accept","tool","strategy",
    "weighted"}.
    """
    move = str((answers.get("move") or {}).get("choice") or "accept")
    if move == "accept":
        return ("accept", None, "accept")
    if move == "build":
        tool = str((answers.get("build") or {}).get("choice") or "stable5")
        if tool in BUILD_TOOLS:
            return ("tool", tool, f"build:{tool}")
        return ("accept", None, "accept")
    if move == "single":
        name = str((answers.get("single") or {}).get("choice") or "")
        if not name:
            return ("accept", None, "accept")
        return ("strategy", {"combine": "best_single", "model": name}, f"best_{name}")
    rname = str((answers.get("recipe") or {}).get("choice") or "full")
    method = str((answers.get("method") or {}).get("choice") or "mean")
    if rname not in recipes:
        rname = "full"
    handle, members = recipes[rname]
    spec: Dict[str, Any] = {"combine": method, "pool": handle}
    if method == "trimmed_mean":
        spec["trim_pct"] = 0.2
    kind = "weighted" if method.startswith("weighted_") else "strategy"
    return (kind, spec, f"{method}_{rname}")


def _leader_kind(state: ReactState) -> str:
    """Tipo da estratégia que lidera a validação agora (para a pergunta de
    OBSERVAÇÃO)."""
    leader = state.best_attempt()
    if leader is None:
        return "no_leader"
    method = leader.spec.get("combine")
    if method == "best_single":
        return "single"
    if method == "weighted":
        return "weighted"
    return "combination"


_OBS_CRITERIA = {
    "combination": "a robust combination over a group of models (mean/median/trimmed)",
    "weighted": "a weighted combination",
    "single": "one model alone",
    "no_leader": "the history is empty - nothing scored yet",
}

#: (tipo do líder) -> decisões oferecidas, com o prior anti-single embutido.
_DECISION_CRITERIA = {
    "combination": {
        "new_combination": "test a DIFFERENT combination (choose a group recipe and a method)",
        "diversify_single": "test a single model as a different kind of bet",
        "accept": "accept the leader",
    },
    "weighted": {
        "new_combination": "test a different combination (choose a group recipe and a method)",
        "diversify_single": "test a single model as a different kind of bet",
        "accept": "accept the leader",
    },
    "single": {
        "diversify_combination": (
            "test a COMBINATION over a group to reduce the risk of a single model"
        ),
        "another_single": "test another single model",
        "accept": "accept the leader",
    },
    "no_leader": {
        "build": "build a group of models first (stable/top-k/pruned)",
        "combine": "test a combination over a group of models",
        "single": "test a single model",
    },
}

_DECISION_INSTRUCTIONS = {
    "combination": (
        "The leader is a robust combination. Combinations over groups have "
        "generally transferred better than single models in past series; prefer "
        "them unless the history strongly suggests otherwise."
    ),
    "weighted": (
        "The leader is a weighted combination. Combinations over groups have "
        "generally transferred better than single models in past series; prefer "
        "them unless the history strongly suggests otherwise."
    ),
    "single": (
        "The leader is a single model. Single-model bets rarely transfer well; "
        "consider testing a combination over a group instead, unless the single "
        "model's evidence is overwhelming."
    ),
    "no_leader": (
        "Nothing has been scored yet. Start by building a group or testing a "
        "combination; combinations over groups generally transfer better than "
        "single models."
    ),
}


def build_reasoning_questions(
    state: ReactState,
    series_card: Dict[str, Any],
    no_seeds: bool = False,
    fmt: str = "text",
) -> Dict[str, Any]:
    """Turno em 3 PASSOS (reasoning por construção):

    1. q1 OBSERVAÇÃO: qual TIPO de estratégia lidera a validação agora?
    2. q2 DECISÃO condicionada à observação (prior anti-single embutido)
    3. q3 PARÂMETROS conforme a decisão (receita×método / modelo / construção)

    Cada resposta é registrada na telemetria — a cadeia vira o "raciocínio".
    """
    recipes = _group_recipes(state)
    ev = model_evidence(state, series_card)
    top_single = sorted(ev, key=lambda n: ev[n]["rank"])[:5]
    leader_kind = _leader_kind(state)

    q1 = {
        "type": "choice",
        "instructions": (
            "Read the current history. What KIND of strategy has the best "
            "validation score right now?"
        ),
        "criteria": _OBS_CRITERIA,
    }
    q2 = {
        "type": "choice",
        "instructions": _DECISION_INSTRUCTIONS[leader_kind],
        "criteria": _DECISION_CRITERIA[leader_kind],
    }
    q3 = {
        "combine": {
            "recipe": {
                "type": "choice",
                "instructions": "Which group of models?",
                "criteria": {
                    rname: _fmt_recipe(rname, members, ev, fmt)
                    for rname, (_, members) in recipes.items()
                },
            },
            "method": {
                "type": "choice",
                "instructions": "Which combination method over that group?",
                "criteria": {
                    "mean": "plain average",
                    "median": "median (robust to an outlying model)",
                    "trimmed_mean": "trimmed mean (drop 20% from each tail)",
                    "weighted_inverse": "weighted average with weights = 1/validation error",
                    "weighted_softmax": "weighted average with weights = softmax(-validation error)",
                },
            },
        },
        "single": {
            "type": "choice",
            "instructions": "Which model alone?",
            "criteria": {n: _fmt_model(n, ev.get(n), fmt) for n in top_single},
        },
        "build": {
            "type": "choice",
            "instructions": "Which group recipe?",
            "criteria": {
                "stable5": "the 5 models with the most consistent ranking across windows",
                "stable7": "the 7 models with the most consistent ranking across windows",
                "stable9": "the 9 models with the most consistent ranking across windows",
                "top5": "the 5 models with the lowest validation error",
                "prune": "drop near-duplicate models (correlation > 0.95)",
            },
        },
    }
    return {"q1": q1, "q2": q2, "q3": q3, "leader_kind": leader_kind}


#: decisão do q2 -> (tipo de composição, chave do q3)
_DECISION_TO_PATH = {
    "new_combination": "combine",
    "diversify_combination": "combine",
    "combine": "combine",
    "diversify_single": "single",
    "another_single": "single",
    "single": "single",
    "build": "build",
    "accept": "accept",
}


def _compose_reasoning(answers: Dict[str, Any], recipes: Dict[str, tuple]) -> tuple:
    """Compõe a ação a partir das respostas de q1/q2/q3."""
    decision = str((answers.get("q2") or {}).get("choice") or "accept")
    path = _DECISION_TO_PATH.get(decision, "accept")
    if path == "accept":
        return ("accept", None, "accept")
    if path == "build":
        tool = str((answers.get("build") or {}).get("choice") or "stable5")
        if tool in BUILD_TOOLS:
            return ("tool", tool, f"build:{tool}")
        return ("accept", None, "accept")
    if path == "single":
        name = str((answers.get("single") or {}).get("choice") or "")
        if not name:
            return ("accept", None, "accept")
        return ("strategy", {"combine": "best_single", "model": name}, f"best_{name}")
    rname = str((answers.get("recipe") or {}).get("choice") or "full")
    method = str((answers.get("method") or {}).get("choice") or "mean")
    if rname not in recipes:
        rname = "full"
    handle, members = recipes[rname]
    spec: Dict[str, Any] = {"combine": method, "pool": handle}
    if method == "trimmed_mean":
        spec["trim_pct"] = 0.2
    kind = "weighted" if method.startswith("weighted_") else "strategy"
    return (kind, spec, f"{method}_{rname}")


def _describe_option(label: str, info: Dict[str, Any], fmt: str = "text") -> str:
    if info.get("kind") == "tool":
        tool = info["tool"]
        args = info["args"]
        if tool == "select_stable":
            return (f"select the {args['k']} models with the most consistent ranking "
                    "across windows (new pool handle)")
        if tool == "select_top_k":
            return f"select the {args['k']} models with the lowest validation error (new pool handle)"
        return "drop near-duplicate models from the full pool (new pool handle)"
    spec = info["spec"]
    method = spec["combine"]
    ev = info.get("evidence", {})
    members = info.get("members", [])
    shown = members[:5]
    mtext = "; ".join(_fmt_model(m, ev.get(m), fmt) for m in shown)
    suffix = " (already tested)" if info["tested"] else ""
    if method == "best_single":
        return f"use only model {_fmt_model(spec['model'], ev.get(spec['model']), fmt)} as the forecast{suffix}"
    what = _METHOD_WORDS.get(method, method)
    return f"{what} of the {len(members)} models [{mtext}]{suffix}"


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
    no_seeds: bool = False,
    fmt: str = "text",
    dataset_card: Optional[Dict[str, Any]] = None,
    staged: bool = False,
    reasoning: bool = False,
    on_step: Optional[Any] = None,
) -> LayaLoopResult:
    """Roda o loop de decisão do classificador e devolve a melhor tentativa.

    `state` já vem semeado pela Fase 2 (`pool.run_phase2`) — exceto com
    `no_seeds=True`, onde o histórico começa vazio e o menu passa a incluir
    ações de ferramenta para construir pools do zero. O classificador só
    escolhe entre ações concretas; a execução continua determinística.

    `fmt="raw"` apresenta a evidência dos modelos como números crus;
    `fmt="text"` como resumo comparativo (para o A/B).

    Cada turno é registrado em `result.step_details` com telemetria completa:
    estado antes, pergunta+opções, resposta crua (probabilidades), ação
    executada, resultado, estado depois e tempos.
    """
    result = LayaLoopResult(final_attempt=state.best_attempt())
    if not state.attempts and not no_seeds:
        raise RuntimeError("histórico vazio: rode pool.run_phase2 antes do loop")

    patience = max(1, int(patience))
    stale = 0
    last_chosen: Optional[str] = None
    #: Opções já escolhidas; com >=2 tentativas saem do menu (força diversidade:
    #: um classificador determinístico repete a escolha se a entrada não muda).
    tried: Dict[str, int] = {}

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        result.iterations_used = iteration
        t_turn = time.perf_counter()

        # ── modo ETAPAS: uma chamada, perguntas independentes, composição ────
        if staged:
            scratch = [
                {"iter": e["iteration"], "action": e["action"],
                 "score": e.get("score"), "rank": e.get("rank")}
                for e in result.trace[-6:]
            ]
            stext = build_state_text(
                series_card, pool_card, state, budget=state_budget,
                scratchpad=scratch or None, dataset_card=dataset_card,
            )
            qs = build_staged_questions(state, series_card, no_seeds=no_seeds, fmt=fmt)
            # ── chamada 1: o MOVIMENTO ───────────────────────────────────
            t_pred = time.perf_counter()
            try:
                out1 = agent.predict(stext, {"move": qs["move"]})
                answers = {"move": out1["answers"].get("move") or {}}
                pred_error = None
            except Exception as exc:
                result.errors.append(f"iteration {iteration}: {type(exc).__name__}: {exc}")
                result.stop_reason = "laya_error"
                pred_error = f"{type(exc).__name__}: {exc}"
                answers = {}
            pred1_s = time.perf_counter() - t_pred
            move = str(answers["move"].get("choice") or "accept")
            # ── chamada 2 (condicional): só a pergunta do movimento ─────
            pred2_s = 0.0
            if pred_error is None and move in ("combine", "single", "build"):
                t_pred2 = time.perf_counter()
                # combine já é um dicionário de perguntas; single/build são uma
                # pergunta só e precisam ser embrulhadas {nome: pergunta}
                follow_q = qs[move] if move == "combine" else {move: qs[move]}
                try:
                    out2 = agent.predict(stext, follow_q)
                    answers.update(out2["answers"] or {})
                except Exception as exc:
                    result.errors.append(
                        f"iteration {iteration}: follow-up {move}: {type(exc).__name__}: {exc}"
                    )
                    pred_error = f"follow-up {move}: {type(exc).__name__}: {exc}"
                pred2_s = time.perf_counter() - t_pred2
            pred_s = pred1_s + pred2_s
            kind, spec_or_tool, label = _compose_staged(
                state, answers, _group_recipes(state),
            )
            step: Dict[str, Any] = {
                "iteration": iteration,
                "state_before": stext,
                "question": qs,
                "answer": {
                    qid: {"choice": a.get("choice"),
                          "confidence": a.get("confidence"),
                          "probabilities": a.get("probabilities")}
                    for qid, a in answers.items()
                },
                "pred_error": pred_error,
                "timing": {"predict_s": round(pred_s, 4),
                           "predict1_s": round(pred1_s, 4),
                           "predict2_s": round(pred2_s, 4)},
            }
            if pred_error is not None:
                result.step_details.append(step)
                break
            if kind == "accept":
                if not state.attempts:
                    # composição degenerada: nada a aceitar ainda — conta como
                    # turno sem informação e continua
                    result.errors.append(
                        f"iteration {iteration}: composed 'accept' with empty history"
                    )
                    stale += 1
                    step["executed"] = "accept ignored (empty history)"
                    result.step_details.append(step)
                    if stale >= patience:
                        result.stop_reason = f"no new information in {stale} consecutive turns"
                        break
                    continue
                result.stop_reason = "laya_accept"
                step["executed"] = "accept (keep current best)"
                step["state_after"] = _brief_history(state)
                step["timing"]["tool_s"] = 0.0
                result.step_details.append(step)
                break
            # freio de composições repetidas
            if tried.get(label, 0) >= 2:
                stale += 1
                result.trace.append({"iteration": iteration, "action": label,
                                     "skipped": "composition repeated twice"})
                step["executed"] = {"skipped": f"{label} já composto 2x"}
                result.step_details.append(step)
                if stale >= patience:
                    result.stop_reason = f"no new information in {stale} consecutive turns"
                    break
                continue
            tried[label] = tried.get(label, 0) + 1
            t_tool = time.perf_counter()
            entry: Dict[str, Any] = {
                "iteration": iteration, "action": label,
                "confidence": (answers.get("move") or {}).get("confidence"),
            }
            if kind == "tool":
                tool_fn_name, targs = BUILD_TOOLS[spec_or_tool]
                existing = set(state.pools)
                try:
                    obs = getattr(T, tool_fn_name)(state, **targs)
                    handle = obs.get("pool")
                    new_handle = handle is not None and handle not in existing
                    entry["observation"] = (
                        f"pool {handle} ({len(obs.get('models', []))} models)"
                        if handle else json.dumps(obs, default=str)[:120]
                    )
                except Exception as exc:
                    entry["observation"] = f"error: {exc}"
                    new_handle = False
                tool_s = time.perf_counter() - t_tool
                step.update({
                    "executed": {"tool": tool_fn_name, "args": targs,
                                 "observation": entry.get("observation")},
                    "state_after": _brief_history(state),
                    "timing": {"predict_s": step["timing"]["predict_s"],
                               "tool_s": round(tool_s, 4)},
                })
                result.step_details.append(step)
                result.trace.append(entry)
                if new_handle:
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    result.stop_reason = f"no new information in {stale} consecutive turns"
                    break
                continue
            spec = dict(spec_or_tool)
            if kind == "weighted":
                method = spec["combine"]
                wres = (
                    T.weights_inverse_error(state, pool=spec["pool"])
                    if method == "weighted_inverse"
                    else T.weights_softmax_neg_error(state, pool=spec["pool"])
                )
                spec = {"combine": "weighted", "pool": spec["pool"],
                        "weights": wres.get("weights")}
            attempt, is_new = state.evaluate(
                spec, rationale=f"laya composed {label}", origin="agent", iteration=iteration
            )
            tool_s = time.perf_counter() - t_tool
            entry.update({
                "score": round(float(attempt.score), 4) if attempt.score == attempt.score else None,
                "rank": state.ranked_attempts().index(attempt) + 1,
                "already_tested": not is_new,
            })
            step.update({
                "executed": {"spec": spec},
                "result": {"score": entry["score"], "rank": entry["rank"],
                           "already_tested": not is_new,
                           "is_best": attempt is state.best_attempt()},
                "state_after": _brief_history(state),
                "timing": {"predict_s": step["timing"]["predict_s"],
                           "tool_s": round(tool_s, 4),
                           "turn_s": round(time.perf_counter() - t_turn, 4)},
            })
            result.step_details.append(step)
            result.trace.append(entry)
            if not is_new or label == last_chosen:
                stale += 1
            else:
                stale = 0
            last_chosen = label
            if stale >= patience:
                result.stop_reason = f"no new information in {stale} consecutive turns"
                break
            continue

        # ── modo REASONING (3 passos): observação → decisão → parâmetros ────
        if reasoning:
            scratch = [
                {"iter": e["iteration"], "action": e["action"],
                 "score": e.get("score"), "rank": e.get("rank")}
                for e in result.trace[-6:]
            ]
            stext = build_state_text(
                series_card, pool_card, state, budget=state_budget,
                scratchpad=scratch or None, dataset_card=dataset_card,
            )
            rqs = build_reasoning_questions(state, series_card, no_seeds=no_seeds, fmt=fmt)
            pred_error = None
            answers: Dict[str, Any] = {}
            # ── chamada 1: OBSERVAÇÃO (ler o placar) ──────────────────────
            t1 = time.perf_counter()
            try:
                out1 = agent.predict(stext, {"q1": rqs["q1"]})
                answers["q1"] = out1["answers"].get("q1") or {}
            except Exception as exc:
                result.errors.append(f"iteration {iteration}: q1: {type(exc).__name__}: {exc}")
                result.stop_reason = "laya_error"
                pred_error = f"q1: {type(exc).__name__}: {exc}"
            pred1_s = time.perf_counter() - t1
            # ── chamada 2: DECISÃO (condicionada ao tipo do líder) ─────────
            pred2_s = 0.0
            if pred_error is None:
                t2 = time.perf_counter()
                try:
                    out2 = agent.predict(stext, {"q2": rqs["q2"]})
                    answers["q2"] = out2["answers"].get("q2") or {}
                except Exception as exc:
                    result.errors.append(f"iteration {iteration}: q2: {type(exc).__name__}: {exc}")
                    pred_error = f"q2: {type(exc).__name__}: {exc}"
                pred2_s = time.perf_counter() - t2
            # ── chamada 3: PARÂMETROS conforme a decisão ───────────────────
            pred3_s = 0.0
            if pred_error is None:
                path = _DECISION_TO_PATH.get(
                    str((answers.get("q2") or {}).get("choice") or "accept"), "accept",
                )
                if path in ("combine", "single", "build"):
                    t3 = time.perf_counter()
                    try:
                        out3 = agent.predict(stext, rqs["q3"][path])
                        answers.update(out3["answers"] or {})
                    except Exception as exc:
                        result.errors.append(
                            f"iteration {iteration}: q3/{path}: {type(exc).__name__}: {exc}"
                        )
                        pred_error = f"q3/{path}: {type(exc).__name__}: {exc}"
                    pred3_s = time.perf_counter() - t3
            pred_s = pred1_s + pred2_s + pred3_s
            kind, spec_or_tool, label = _compose_reasoning(
                answers, _group_recipes(state),
            )
            step: Dict[str, Any] = {
                "iteration": iteration,
                "state_before": stext,
                "question": rqs,
                "answer": {
                    qid: {"choice": a.get("choice"),
                          "confidence": a.get("confidence"),
                          "probabilities": a.get("probabilities")}
                    for qid, a in answers.items()
                },
                "pred_error": pred_error,
                "timing": {"predict_s": round(pred_s, 4),
                           "predict1_s": round(pred1_s, 4),
                           "predict2_s": round(pred2_s, 4),
                           "predict3_s": round(pred3_s, 4)},
            }
            if pred_error is not None:
                result.step_details.append(step)
                break
            if kind == "accept":
                if not state.attempts:
                    result.errors.append(
                        f"iteration {iteration}: composed 'accept' with empty history"
                    )
                    stale += 1
                    step["executed"] = "accept ignored (empty history)"
                    result.step_details.append(step)
                    if stale >= patience:
                        result.stop_reason = f"no new information in {stale} consecutive turns"
                        break
                    continue
                result.stop_reason = "laya_accept"
                step["executed"] = "accept (keep current best)"
                step["state_after"] = _brief_history(state)
                step["timing"]["tool_s"] = 0.0
                result.step_details.append(step)
                break
            if tried.get(label, 0) >= 2:
                stale += 1
                result.trace.append({"iteration": iteration, "action": label,
                                     "skipped": "composition repeated twice"})
                step["executed"] = {"skipped": f"{label} já composto 2x"}
                result.step_details.append(step)
                if stale >= patience:
                    result.stop_reason = f"no new information in {stale} consecutive turns"
                    break
                continue
            tried[label] = tried.get(label, 0) + 1
            t_tool = time.perf_counter()
            entry: Dict[str, Any] = {
                "iteration": iteration, "action": label,
                "confidence": (answers.get("q2") or {}).get("confidence"),
            }
            if kind == "tool":
                tool_fn_name, targs = BUILD_TOOLS[spec_or_tool]
                existing = set(state.pools)
                try:
                    obs = getattr(T, tool_fn_name)(state, **targs)
                    handle = obs.get("pool")
                    new_handle = handle is not None and handle not in existing
                    entry["observation"] = (
                        f"pool {handle} ({len(obs.get('models', []))} models)"
                        if handle else json.dumps(obs, default=str)[:120]
                    )
                except Exception as exc:
                    entry["observation"] = f"error: {exc}"
                    new_handle = False
                tool_s = time.perf_counter() - t_tool
                step.update({
                    "executed": {"tool": tool_fn_name, "args": targs,
                                 "observation": entry.get("observation")},
                    "state_after": _brief_history(state),
                    "timing": {"predict_s": step["timing"]["predict_s"],
                               "tool_s": round(tool_s, 4)},
                })
                result.step_details.append(step)
                result.trace.append(entry)
                if new_handle:
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    result.stop_reason = f"no new information in {stale} consecutive turns"
                    break
                continue
            spec = dict(spec_or_tool)
            if kind == "weighted":
                method = spec["combine"]
                wres = (
                    T.weights_inverse_error(state, pool=spec["pool"])
                    if method == "weighted_inverse"
                    else T.weights_softmax_neg_error(state, pool=spec["pool"])
                )
                spec = {"combine": "weighted", "pool": spec["pool"],
                        "weights": wres.get("weights")}
            attempt, is_new = state.evaluate(
                spec, rationale=f"laya reasoned {label}", origin="agent", iteration=iteration
            )
            tool_s = time.perf_counter() - t_tool
            entry.update({
                "score": round(float(attempt.score), 4) if attempt.score == attempt.score else None,
                "rank": state.ranked_attempts().index(attempt) + 1,
                "already_tested": not is_new,
            })
            step.update({
                "executed": {"spec": spec},
                "result": {"score": entry["score"], "rank": entry["rank"],
                           "already_tested": not is_new,
                           "is_best": attempt is state.best_attempt()},
                "state_after": _brief_history(state),
                "timing": {"predict_s": step["timing"]["predict_s"],
                           "tool_s": round(tool_s, 4),
                           "turn_s": round(time.perf_counter() - t_turn, 4)},
            })
            result.step_details.append(step)
            result.trace.append(entry)
            if not is_new or label == last_chosen:
                stale += 1
            else:
                stale = 0
            last_chosen = label
            if stale >= patience:
                result.stop_reason = f"no new information in {stale} consecutive turns"
                break
            continue

        cands = build_candidates(state, series_card, pool_card, no_seeds=no_seeds, fmt=fmt)
        for label in [l for l in list(cands) if tried.get(l, 0) >= 2]:
            del cands[label]
        if not cands:
            result.stop_reason = "menu_exhausted"
            break
        criteria: Dict[str, str] = {
            label: _describe_option(label, info, fmt) for label, info in cands.items()
        }
        if state.attempts:
            criteria["accept"] = "stop now and keep the current best strategy"
        scratch = [
            {"iter": e["iteration"], "action": e["action"],
             "score": e.get("score"), "rank": e.get("rank")}
            for e in result.trace[-6:]
        ]
        stext = build_state_text(
            series_card, pool_card, state, budget=state_budget,
            scratchpad=scratch or None, dataset_card=dataset_card,
        )
        question = {
            "next_action": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": criteria,
            }
        }

        t_pred = time.perf_counter()
        try:
            out = agent.predict(stext, question)
            answer = out["answers"]["next_action"]
            chosen = str(answer.get("choice") or "")
            conf = answer.get("confidence", answer.get("probability"))
            probs = answer.get("probabilities")
            usage = out.get("usage")
            pred_error = None
        except Exception as exc:
            result.errors.append(f"iteration {iteration}: {type(exc).__name__}: {exc}")
            result.stop_reason = "laya_error"
            pred_error = f"{type(exc).__name__}: {exc}"
            chosen, conf, probs, usage = "", None, None, None
        pred_s = time.perf_counter() - t_pred

        step: Dict[str, Any] = {
            "iteration": iteration,
            "state_before": stext,
            "question": question,
            "answer": {
                "choice": chosen,
                "confidence": conf,
                "probabilities": {k: round(float(v), 4) for k, v in (probs or {}).items()},
                "input_tokens": (usage or {}).get("input_tokens"),
            },
            "pred_error": pred_error,
            "timing": {"predict_s": round(pred_s, 4)},
        }

        if pred_error is not None:
            result.step_details.append(step)
            break
        if chosen == "accept":
            result.stop_reason = "laya_accept"
            step["executed"] = "accept (keep current best)"
            step["state_after"] = _brief_history(state)
            step["timing"]["tool_s"] = 0.0
            result.step_details.append(step)
            break
        if chosen not in cands:
            result.errors.append(f"iteration {iteration}: laya escolheu label fora do menu: {chosen!r}")
            result.stop_reason = "laya_error"
            result.step_details.append(step)
            break

        info = cands[chosen]
        tried[chosen] = tried.get(chosen, 0) + 1
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "action": chosen,
            "confidence": conf,
            "probability_of_chosen": (probs or {}).get(chosen),
            "probabilities": {k: round(float(v), 4) for k, v in (probs or {}).items()},
            "input_tokens": (usage or {}).get("input_tokens"),
        }
        t_tool = time.perf_counter()

        if info.get("kind") == "tool":
            # ação de ferramenta: executa, registra o handle e continua
            tool_fn = getattr(T, info["tool"])
            existing = set(state.pools)
            try:
                obs = tool_fn(state, **info["args"])
                handle = obs.get("pool")
                new_handle = handle is not None and handle not in existing
                entry["observation"] = (
                    f"pool {handle} ({len(obs.get('models', []))} models)"
                    if handle else json.dumps(obs, default=str)[:120]
                )
            except Exception as exc:
                entry["observation"] = f"error: {exc}"
                new_handle = False
            tool_s = time.perf_counter() - t_tool
            step.update({
                "executed": {"tool": info["tool"], "args": info["args"],
                             "observation": entry.get("observation")},
                "state_after": _brief_history(state),
                "timing": {"predict_s": step["timing"]["predict_s"],
                           "tool_s": round(tool_s, 4)},
            })
            result.step_details.append(step)
            result.trace.append(entry)
            if on_step is not None:
                try:
                    on_step(state.dataset_index, entry)
                except Exception:
                    pass
            if new_handle:
                stale = 0
            else:
                stale += 1
            last_chosen = chosen
            if stale >= patience:
                result.stop_reason = f"no new information in {stale} consecutive turns"
                break
            continue

        spec = dict(info["spec"])
        if info.get("kind") == "weighted":
            method = spec["combine"]
            wres = (
                T.weights_inverse_error(state, pool=spec["pool"])
                if method == "weighted_inverse"
                else T.weights_softmax_neg_error(state, pool=spec["pool"])
            )
            spec = {"combine": "weighted", "pool": spec["pool"],
                    "weights": wres.get("weights")}
        attempt, is_new = state.evaluate(
            spec, rationale=f"laya selected {chosen}", origin="agent", iteration=iteration
        )
        tool_s = time.perf_counter() - t_tool
        entry.update({
            "score": round(float(attempt.score), 4) if attempt.score == attempt.score else None,
            "rank": state.ranked_attempts().index(attempt) + 1,
            "already_tested": not is_new,
        })
        step.update({
            "executed": {"spec": spec},
            "result": {"score": entry["score"], "rank": entry["rank"],
                       "already_tested": not is_new,
                       "is_best": attempt is state.best_attempt()},
            "state_after": _brief_history(state),
            "timing": {"predict_s": step["timing"]["predict_s"],
                       "tool_s": round(tool_s, 4),
                       "turn_s": round(time.perf_counter() - t_turn, 4)},
        })
        result.step_details.append(step)
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


GATE_INSTRUCTIONS = (
    "Is this candidate strategy the best available strategy for this series on "
    "the validation windows (nested leave-one-out)? Answer yes only if you "
    "expect it to be strictly the best on validation."
)


def run_gate_pass(
    state: ReactState,
    agent: Any,
    series_card: Dict[str, Any],
    pool_card: Dict[str, Any],
    budget: int = 8000,
) -> List[Dict[str, Any]]:
    """H3: o gate pontua TODO o histórico (sementes + propostas).

    Para cada candidato único do histórico, monta o estado com a estratégia
    candidata e pergunta P("é o melhor na validação"). Mesmo formato do dataset
    de fine-tune (`build_finetune_dataset.py`), então treino e inferência casam.
    """
    seen = set()
    scored: List[Dict[str, Any]] = []
    for a in state.ranked_attempts():
        key = _spec_key(a.spec)
        if key in seen:
            continue
        seen.add(key)
        stext = build_state_text(series_card, pool_card, state, budget=budget)
        stext += "\nCANDIDATE STRATEGY: " + json.dumps(a.spec, sort_keys=True, default=str)
        try:
            out = agent.predict(stext, {
                "gate": {"type": "noul", "instructions": GATE_INSTRUCTIONS},
            })
            p = float(out["answers"]["gate"].get("noul", 0.5))
        except Exception:
            p = 0.5
        scored.append({
            "id": a.attempt_id, "spec": a.spec,
            "score_val": round(float(a.score), 6),
            "p_best": round(p, 4),
            "origin": a.origin,
        })
    return scored


def _brief_history(state: ReactState) -> List[Dict[str, Any]]:
    """Placar resumido pós-turno (para o 'estado depois' na telemetria)."""
    return [a.brief(include_rationale=False) for a in state.ranked_attempts()[:5]]
