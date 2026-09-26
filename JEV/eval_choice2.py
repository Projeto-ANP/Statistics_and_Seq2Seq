#!/usr/bin/env python3
"""ESCOLHA pareada v2 — formato FIEL ao pipeline (garantir de vez).

A pergunta de escolha é construída EXATAMENTE como o pipeline faria:
  - estado = build_state_text(...) (o mesmo do gate/menu, budget 8000);
  - opções = as 4 melhores estratégias do histórico de sementes por score
    (= "o agente propôs 4 opções para combinar");
  - descrição de cada opção = rótulo + membros com EVIDÊNCIA por modelo
    (_fmt_model, o formato do menu laya_loop);
  - pergunta choice com criteria {letra: descrição}, em 2 ordens (normal +
    invertida) — permutation averaging: escolha final = maior probabilidade
    média entre ordens.

Gabaritos: (i) vencedor da validação (argmin das 4) e (ii) vencedor do TESTE
(smape via apply_to_test — avaliação apenas, não entra em nenhum treino).

Uso (servidor):
  python3 JEV/eval_choice2.py --datasets NN5_WEEKLY_DATASET --series 40 --laya \
    --endpoints kev4b=http://127.0.0.1:8009 kev08b=http://127.0.0.1:8010 \
    --out JEV/phase0_choice2.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "JEV"))
os.chdir(_ROOT)

from run_tsf_orchestrator import DEFAULT_MODELS  # noqa: E402
from orchestrator_react import ingest as I  # noqa: E402
from orchestrator_react import pool as POOL  # noqa: E402
from orchestrator_react import tools as T  # noqa: E402
from orchestrator_react.config import ReactConfig  # noqa: E402
from laya_loop import (  # noqa: E402
    LayaAgent,
    build_state_text,
    model_evidence,
    _fmt_model,
)
from run_laya import DATASET_SOURCES, DEFAULT_RESULTS_DIR, DEFAULT_SOURCE_DIR  # noqa: E402

LETTERS = "abcdefghij"


# ─────────────────────────── clientes (mesmos do eval_choice) ────────────────


def http_systemone(url: str, state: str, questions: Dict[str, Any],
                   model: str) -> Tuple[Dict[str, Any], float]:
    body = {"state": state, "model": model, "questions": questions}
    req = urllib.request.Request(
        f"{url}/v1/systemone",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return out, time.perf_counter() - t0


class LayaClient:
    name = "laya"

    def __init__(self) -> None:
        self.agent = LayaAgent(checkpoint="multilingual", max_len=8192)

    def ask(self, state: str, questions: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        return self.agent.predict(state, questions), time.perf_counter() - t0


class HttpClient:
    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url.rstrip("/")

    def ask(self, state: str, questions: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        return http_systemone(self.url, state, questions, "kev-latest")


def _probs_from_answer(ans: Any) -> Optional[Dict[str, float]]:
    if not isinstance(ans, dict):
        return None
    probs = ans.get("probabilities")
    if isinstance(probs, dict) and probs:
        return {str(k): float(v) for k, v in probs.items()}
    return None


# ─────────────────────────── descrição no formato do menu ────────────────────


def _describe(spec: Dict[str, Any], members: List[str],
              ev: Dict[str, Dict[str, Any]], brief: str) -> str:
    """Descrição da opção como o menu faz: rótulo + membros com evidência."""
    parts = ", ".join(_fmt_model(m, ev.get(m), "text") for m in members[:6])
    return f"{brief}: models {parts}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Escolha pareada v2 (formato fiel ao pipeline)")
    ap.add_argument("--datasets", nargs="+", default=["NN5_WEEKLY_DATASET"])
    ap.add_argument("--series", type=int, default=40)
    ap.add_argument("--endpoints", nargs="+", default=[])
    ap.add_argument("--laya", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="JEV/phase0_choice2.json")
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args(argv)

    clients: List[Any] = []
    if args.laya:
        clients.append(LayaClient())
    for spec in args.endpoints:
        name, url = spec.split("=", 1)
        clients.append(HttpClient(name, url))

    cfg = ReactConfig()
    rng = random.Random(args.seed)

    # ── amostrar séries de cada dataset ──────────────────────────────────────
    tasks: List[Tuple[str, int]] = []
    for dataset in args.datasets:
        models = [m for m in DEFAULT_MODELS
                  if os.path.exists(I.model_csv_path(m, dataset, args.results_dir))]
        frames = I.load_dataset_frames(models, dataset, args.results_dir)
        n_series = I.count_series(dataset, models[0], args.results_dir)
        try:
            bad = I.find_misaligned_models(models, dataset, 0,
                                           results_dir=args.results_dir,
                                           frames=frames, n_windows=3)
        except Exception:
            bad = {}
        models_eff = [m for m in models if m not in bad]
        idxs = list(range(n_series))
        rng.shuffle(idxs)
        for idx in idxs[: args.series]:
            tasks.append((dataset, idx, models_eff, frames))

    questions: List[Dict[str, Any]] = []
    for dataset, idx, models_eff, frames in tasks:
        try:
            ing = I.load_series(models=models_eff, dataset=dataset, dataset_index=idx,
                                config=cfg, results_dir=args.results_dir,
                                source_file=DATASET_SOURCES[dataset],
                                source_dir=args.source_dir, frames=frames)
            state = ing.state
            POOL.run_phase2(state, cfg)
            series_card = T.series_profile(state)
            pool_card = POOL.pool_report(state)
            ev = model_evidence(state, series_card)
            top4 = state.ranked_attempts()[:4]
            state_text = build_state_text(series_card, pool_card, state, budget=8000)
            specs = [a.spec for a in top4]
            briefs = [a.brief(include_rationale=False)["strategy"] for a in top4]
            members_l = []
            for a in top4:
                if a.spec.get("combine") == "best_single":
                    members_l.append([a.spec["model"]])
                else:
                    pool = a.spec.get("pool") or "pool_full"
                    try:
                        members_l.append(state.pool_names(pool))
                    except Exception:
                        members_l.append([])
            descs = [_describe(s, m, ev, b) for s, m, b in zip(specs, members_l, briefs)]
            truths = {}
            for a, spec in zip(top4, specs):
                forecast, _ = state.apply_to_test(spec)
                smape = float(np.nanmean(
                    2 * np.abs(forecast - np.asarray(ing.test_values)) /
                    (np.abs(forecast) + np.abs(np.asarray(ing.test_values)))))
                truths[json.dumps(spec, sort_keys=True)] = {
                    "score_val": float(a.score), "smape_test": smape,
                }
            variants = []
            for order in (list(range(4)), list(range(3, -1, -1))):
                variants.append({
                    "criteria": {LETTERS[i]: descs[order[i]] for i in range(4)},
                    "letter_to_spec": {LETTERS[i]: json.dumps(specs[order[i]], sort_keys=True)
                                       for i in range(4)},
                })
            questions.append({
                "dataset": dataset, "series": idx, "state": state_text,
                "variants": variants, "truths": truths,
            })
        except Exception as exc:
            print(f"[choice2] série {dataset}/{idx} falhou: {type(exc).__name__}: {exc}",
                  flush=True)

    val_winner_test = [min(q["truths"].values(), key=lambda t: t["score_val"])["smape_test"]
                       for q in questions]
    test_winner_test = [min(t["smape_test"] for t in q["truths"].values()) for q in questions]

    results: Dict[str, Any] = {
        "config": {"n_questions": len(questions), "seed": args.seed},
        "baselines": {
            "argmin_of_4_test_smape": round(float(np.mean(val_winner_test)), 4),
            "oracle_test_smape": round(float(np.mean(test_winner_test)), 4),
        },
        "models": {},
    }

    for client in clients:
        print(f"[choice2] avaliando {client.name} ...", flush=True)
        picks: List[str] = []
        lats: List[float] = []
        errors = 0
        for q in questions:
            spec_prob_sums: Dict[str, float] = {}
            spec_counts: Dict[str, int] = {}
            for variant in q["variants"]:
                question = {
                    "best": {
                        "type": "choice",
                        "instructions": (
                            "Given the series and the current history, which of "
                            "these options is the best strategy to apply to this "
                            "series? Choose exactly one letter."
                        ),
                        "criteria": variant["criteria"],
                    },
                }
                try:
                    out, lat = client.ask(q["state"], question)
                    lats.append(lat)
                    ans = (out.get("answers") or {}).get("best")
                    probs = _probs_from_answer(ans)
                    if probs:
                        for letter, p in probs.items():
                            spec = variant["letter_to_spec"].get(letter)
                            if spec is not None:
                                spec_prob_sums[spec] = spec_prob_sums.get(spec, 0.0) + p
                    c = (ans or {}).get("choice")
                    spec = variant["letter_to_spec"].get(str(c).strip() if c else "")
                    if spec is not None:
                        spec_counts[spec] = spec_counts.get(spec, 0) + 1
                except Exception as exc:
                    print(f"  erro: {type(exc).__name__}: {exc}", flush=True)
                    errors += 1
            if spec_prob_sums:
                picks.append(max(spec_prob_sums, key=spec_prob_sums.get))
            elif spec_counts:
                picks.append(max(spec_counts, key=spec_counts.get))
            else:
                picks.append(None)

        chosen_test: List[float] = []
        hit_val = hit_test = 0
        for q, spec in zip(questions, picks):
            if spec is None:
                continue
            t = q["truths"][spec]
            chosen_test.append(t["smape_test"])
            if t["score_val"] <= min(x["score_val"] for x in q["truths"].values()) + 1e-9:
                hit_val += 1
            if t["smape_test"] <= min(x["smape_test"] for x in q["truths"].values()) + 1e-9:
                hit_test += 1

        n = len(chosen_test)
        results["models"][client.name] = {
            "n_answered": n,
            "n_errors": errors,
            "picks_val_winner": round(hit_val / n, 4) if n else None,
            "picks_test_winner": round(hit_test / n, 4) if n else None,
            "random_baseline": 0.25,
            "mean_test_smape_chosen": round(float(np.mean(chosen_test)), 4) if chosen_test else None,
            "argmin_of_4_baseline": round(float(np.mean(val_winner_test)), 4),
            "chosen_vs_argmin": round(float(np.mean(chosen_test)) - float(np.mean(val_winner_test)), 4) if chosen_test else None,
            "latency_p50_s": round(float(np.median(lats)), 4) if lats else None,
        }
        print(f"[choice2]   {client.name}: {results['models'][client.name]}", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"[choice2] resultados -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
