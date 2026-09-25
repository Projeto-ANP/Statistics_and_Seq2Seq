#!/usr/bin/env python3
"""Gate H3 em regressão logística — treino <1s, sem GPU.

Treina leave-one-dataset-out sobre o gate_dataset.jsonl (o mesmo do fine-tune
do LAYA): o gate para o dataset X usa só exemplos dos OUTROS datasets. Na
inferência, pontua cada candidato do histórico com P("é o melhor na
validação") e o final é o argmax.

É o atalho pragmático enquanto o fine-tune do LAYA não termina — e o baseline
honesto do ganho que a camada de seleção aprendida pode dar.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURES = ["score_val", "rank", "margem_pct", "n_attempts", "tau", "n_models",
            "origin_agent", "turn"]


class LogisticGate:
    def __init__(self, data_path: str, holdout: str,
                 target: str = "label_val", seed: int = 0) -> None:
        rows: List[dict] = []
        with open(data_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("dataset") == holdout:
                    continue
                rows.append(r)
        if not rows:
            raise RuntimeError(
                f"sem exemplos de treino para o holdout {holdout!r} em {data_path}"
            )
        X, y = [], []
        for r in rows:
            feats = {
                "score_val": r.get("score_val"),
                "rank": r.get("rank"),
                "margem_pct": r.get("margem_pct"),
                "n_attempts": r.get("n_attempts"),
                "tau": r.get("tau"),
                "n_models": r.get("n_models"),
                "origin_agent": 1.0 if r.get("origin") == "agent" else 0.0,
                "turn": r.get("turn", 0),
            }
            if any(v is None for v in feats.values()):
                continue
            X.append([float(feats[f]) for f in FEATURES])
            y.append(int(r.get(target, 0)))
        if not X:
            raise RuntimeError(f"nenhum exemplo utilizável para o holdout {holdout!r}")
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=seed)
        self.model.fit(np.asarray(X), np.asarray(y))
        self.n_train = len(y)
        self.base_rate = float(np.mean(y))
        self.holdout = holdout
        self.target = target

    def score(self, feats: Dict[str, float]) -> float:
        X = np.asarray([[float(feats[f]) for f in FEATURES]], dtype=float)
        return float(self.model.predict_proba(X)[0, 1])


class LogisticGateW:
    """Gate logístico POR JANELA: 3 modelos (label_val_w0/1/2), LOO por dataset.

    Substitui o LAYA fine-tuned como gate do A2 (treino <1s, sem GPU). A nota
    final é a média dos 3 P — o mesmo formato p_windows/p_mean do veredito.
    """

    def __init__(self, data_path: str, holdout: str, seed: int = 0) -> None:
        rows: List[dict] = []
        with open(data_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("dataset") == holdout:
                    continue
                rows.append(r)
        if not rows:
            raise RuntimeError(
                f"sem exemplos de treino para o holdout {holdout!r} em {data_path}"
            )
        self.models = []
        self.n_train = []
        for w in range(3):
            X, y = [], []
            for r in rows:
                target = r.get(f"label_val_w{w}")
                if target is None:
                    continue
                feats = {
                    "score_val": r.get("score_val"),
                    "rank": r.get("rank"),
                    "margem_pct": r.get("margem_pct"),
                    "n_attempts": r.get("n_attempts"),
                    "tau": r.get("tau"),
                    "n_models": r.get("n_models"),
                    "origin_agent": 1.0 if r.get("origin") == "agent" else 0.0,
                    "turn": r.get("turn", 0),
                }
                if any(v is None for v in feats.values()):
                    continue
                X.append([float(feats[f]) for f in FEATURES])
                y.append(int(target))
            if not X:
                raise RuntimeError(f"sem exemplos para a janela {w} do holdout {holdout!r}")
            m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
            m.fit(np.asarray(X), np.asarray(y))
            self.models.append(m)
            self.n_train.append(len(y))
        self.holdout = holdout

    def score(self, feats: Dict[str, float]):
        X = np.asarray([[float(feats[f]) for f in FEATURES]], dtype=float)
        pw = [float(m.predict_proba(X)[0, 1]) for m in self.models]
        return pw, float(np.mean(pw))


def candidate_features(state: Any, attempt: Any, pool_card: Dict[str, Any]) -> Dict[str, float]:
    """Features de um candidato do histórico — tudo só-validação, computável
    na inferência sem tocar no teste."""
    ranked = state.ranked_attempts()
    rank = ranked.index(attempt) + 1 if attempt in ranked else None
    best = state.best_attempt()
    margem = None
    if best is not None:
        margem = 0.0 if best is attempt else float(
            (attempt.score - best.score) / (abs(best.score) or 1.0)
        )
    spec = attempt.spec
    n_models = (
        1 if spec.get("combine") == "best_single"
        else len(state.get_pool(spec.get("pool") or "pool_full"))
    )
    tau = (pool_card.get("ranking_stability") or {}).get("mean_kendall_tau")
    return {
        "score_val": float(attempt.score),
        "rank": float(rank) if rank is not None else 0.0,
        "margem_pct": float(margem) if margem is not None else 0.0,
        "n_attempts": float(len(state.attempts)),
        "tau": float(tau) if tau is not None else 0.0,
        "n_models": float(n_models),
        "origin_agent": 1.0 if attempt.origin == "agent" else 0.0,
        "turn": float(attempt.iteration or 0),
    }
