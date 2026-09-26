#!/usr/bin/env python3
"""ESCOLHA pareada: LAYA vs kev — quem escolhe melhor entre N opções?

Cenário do usuário: o agente propõe 4 candidatos para combinar; o modelo
recebe o estado + as 4 opções e responde QUAL é a melhor (pergunta `choice`).
Medimos contra dois gabaritos: (i) o vencedor da VALIDAÇÃO (argmin das 4),
(ii) o vencedor do TESTE (smape_test mínimo) — e o sMAPE de teste médio da
opção escolhida, comparado com escolher o argmin das 4 e com o acaso (25%).

Universos: (dataset, série) do gate_dataset.jsonl; as 4 opções = top-4 por
score de validação (inclui o argmin — exatamente o que um agente entregaria).

Modelos (mesma pergunta `choice` com criteria = {letra: spec}):
  - laya          : LAYA multilingual local
  - name=URL      : servidor TypeSafe-compatível (kev4b :8009, kev08b :8010)

Uso (servidor):
  python3 JEV/eval_choice.py --laya \
    --endpoints kev4b=http://127.0.0.1:8009 kev08b=http://127.0.0.1:8010 \
    --universes 200 --out JEV/phase0_choice.json
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

LETTERS = "abcdefghij"


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
        from laya_loop import LayaAgent  # noqa: PLC0415
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


def _chosen(ans: Any) -> Optional[str]:
    if not isinstance(ans, dict):
        return None
    c = ans.get("choice")
    if c is not None:
        return str(c).strip().lower()
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Escolha pareada LAYA vs kev")
    ap.add_argument("--data", default="JEV/data/gate_dataset.jsonl")
    ap.add_argument("--endpoints", nargs="+", default=[])
    ap.add_argument("--laya", action="store_true")
    ap.add_argument("--universes", type=int, default=200)
    ap.add_argument("--n-options", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="JEV/phase0_choice.json")
    args = ap.parse_args(argv)

    # ── montar universos ─────────────────────────────────────────────────────
    records: Dict[Any, List[Dict[str, Any]]] = {}
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                records.setdefault((r["dataset"], r["series"]), []).append(r)
    keys = [k for k, v in records.items() if len(v) >= args.n_options]
    random.seed(args.seed)
    random.shuffle(keys)
    keys = keys[: args.universes]

    # ── construir as questões ────────────────────────────────────────────────
    questions: List[Dict[str, Any]] = []
    for key in keys:
        cands = records[key]
        cands = sorted(cands, key=lambda r: r["score_val"])[: args.n_options]
        criteria: Dict[str, str] = {}
        spec_by_letter: Dict[str, str] = {}
        for i, r in enumerate(cands):
            letter = LETTERS[i]
            spec_by_letter[letter] = json.dumps(r["spec"], sort_keys=True)
            criteria[letter] = spec_by_letter[letter][:120]
        questions.append({
            "key": key,
            "state": cands[0]["state"],
            "criteria": criteria,
            "spec_by_letter": spec_by_letter,
            "truth": {letter: {
                "score_val": next(c["score_val"] for c in cands if json.dumps(c["spec"], sort_keys=True) == spec_by_letter[letter]),
                "smape_test": next(c["smape_test"] for c in cands if json.dumps(c["spec"], sort_keys=True) == spec_by_letter[letter]),
            } for letter in spec_by_letter},
        })

    clients: List[Any] = []
    if args.laya:
        clients.append(LayaClient())
    for spec in args.endpoints:
        name, url = spec.split("=", 1)
        clients.append(HttpClient(name, url))

    # baselines por questão
    val_winner_test = []   # smape_test do argmin das 4
    test_winner_test = []  # oráculo
    for q in questions:
        truths = q["truth"]
        val_winner_test.append(min(t["score_val"] for t in truths.values()) and
                               min(truths.values(), key=lambda t: t["score_val"])["smape_test"])
        test_winner_test.append(min(t["smape_test"] for t in truths.values()))

    results: Dict[str, Any] = {"config": {"universes": len(questions),
                                          "n_options": args.n_options,
                                          "seed": args.seed},
                               "baselines": {
                                   "argmin_of_4_test_smape": round(float(np.mean(val_winner_test)), 4),
                                   "oracle_test_smape": round(float(np.mean(test_winner_test)), 4),
                               },
                               "models": {}}

    for client in clients:
        print(f"[choice] avaliando {client.name} ...", flush=True)
        picks: List[str] = []
        lats: List[float] = []
        errors = 0
        for q in questions:
            question = {
                "best": {
                    "type": "choice",
                    "instructions": (
                        "Which of these candidate strategies is the best for "
                        "this series? Choose exactly one letter."
                    ),
                    "criteria": q["criteria"],
                },
            }
            try:
                out, lat = client.ask(q["state"], question)
                lats.append(lat)
                ans = (out.get("answers") or {}).get("best")
                c = _chosen(ans)
                if c not in q["truth"]:
                    # aceitar "a)" etc.
                    c2 = c[:1] if c else None
                    c = c2 if c2 in q["truth"] else None
                picks.append(c)
            except Exception as exc:
                print(f"  erro: {type(exc).__name__}: {exc}", flush=True)
                errors += 1
                picks.append(None)

        chosen_test = []
        hit_val = hit_test = 0
        n = 0
        for q, c in zip(questions, picks):
            if c is None:
                continue
            n += 1
            t = q["truth"][c]
            chosen_test.append(t["smape_test"])
            if t["score_val"] <= min(x["score_val"] for x in q["truth"].values()) + 1e-9:
                hit_val += 1
            if t["smape_test"] <= min(x["smape_test"] for x in q["truth"].values()) + 1e-9:
                hit_test += 1

        results["models"][client.name] = {
            "n_answered": n,
            "n_errors": errors,
            "picks_val_winner": round(hit_val / n, 4) if n else None,
            "picks_test_winner": round(hit_test / n, 4) if n else None,
            "random_baseline": round(1.0 / args.n_options, 4),
            "mean_test_smape_chosen": round(float(np.mean(chosen_test)), 4) if chosen_test else None,
            "argmin_of_4_baseline": round(float(np.mean(val_winner_test)), 4),
            "chosen_vs_argmin": round(float(np.mean(chosen_test)) - float(np.mean(val_winner_test)), 4) if chosen_test else None,
            "latency_p50_s": round(float(np.median(lats)), 4) if lats else None,
            "picks": picks,
        }
        print(f"[choice]   {client.name}: {results['models'][client.name]}", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"[choice] resultados -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
