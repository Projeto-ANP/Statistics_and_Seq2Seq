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
    # Viés de posição medido: kev-4b responde 'a' 82%, kev-0.8b 'b' 52%, LAYA
    # 'a'+'d' — as letras viciam as escolhas (JevBench: opções invertidas
    # derrubam 72%→21% em modelos pequenos). Correção: cada universo é
    # perguntado em 2 ORDENS (normal + invertida); a escolha final do modelo
    # = spec com MAIOR PROBABILIDADE MÉDIA entre as ordens (permutation
    # averaging, TypeLLM). Só a discriminação de conteúdo sobrevive.
    questions: List[Dict[str, Any]] = []
    for key in keys:
        cands = records[key]
        cands = sorted(cands, key=lambda r: r["score_val"])[: args.n_options]
        specs = [json.dumps(r["spec"], sort_keys=True) for r in cands]
        truths = {
            spec: {"score_val": r["score_val"], "smape_test": r["smape_test"]}
            for r, spec in zip(cands, specs)
        }
        state = cands[0]["state"]
        orders = [specs, list(reversed(specs))]
        variants = []
        for order in orders:
            criteria = {LETTERS[i]: order[i][:120] for i in range(len(order))}
            variants.append({
                "criteria": criteria,
                "letter_to_spec": {LETTERS[i]: order[i] for i in range(len(order))},
            })
        questions.append({
            "key": key, "state": state, "variants": variants, "truths": truths,
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
        truths = q["truths"]
        val_winner_test.append(min(truths.values(), key=lambda t: t["score_val"])["smape_test"])
        test_winner_test.append(min(t["smape_test"] for t in truths.values()))

    results: Dict[str, Any] = {"config": {"universes": len(questions),
                                          "n_options": args.n_options,
                                          "seed": args.seed},
                               "baselines": {
                                   "argmin_of_4_test_smape": round(float(np.mean(val_winner_test)), 4),
                                   "oracle_test_smape": round(float(np.mean(test_winner_test)), 4),
                               },
                               "models": {}}

    def _probs_from_answer(ans: Any) -> Optional[Dict[str, float]]:
        if not isinstance(ans, dict):
            return None
        probs = ans.get("probabilities")
        if isinstance(probs, dict) and probs:
            return {str(k): float(v) for k, v in probs.items()}
        return None

    for client in clients:
        print(f"[choice] avaliando {client.name} ...", flush=True)
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
                            "Which of these candidate strategies is the best for "
                            "this series? Choose exactly one letter."
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
                    c = _chosen(ans)
                    spec = variant["letter_to_spec"].get(c or "")
                    if spec is not None:
                        spec_counts[spec] = spec_counts.get(spec, 0) + 1
                except Exception as exc:
                    print(f"  erro: {type(exc).__name__}: {exc}", flush=True)
                    errors += 1

            if spec_prob_sums:
                chosen_spec = max(spec_prob_sums, key=spec_prob_sums.get)
            elif spec_counts:
                chosen_spec = max(spec_counts, key=spec_counts.get)
            else:
                chosen_spec = None
            picks.append(chosen_spec)

        chosen_test = []
        hit_val = hit_test = 0
        n = 0
        for q, spec in zip(questions, picks):
            if spec is None:
                continue
            n += 1
            t = q["truths"][spec]
            chosen_test.append(t["smape_test"])
            if t["score_val"] <= min(x["score_val"] for x in q["truths"].values()) + 1e-9:
                hit_val += 1
            if t["smape_test"] <= min(x["smape_test"] for x in q["truths"].values()) + 1e-9:
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
        }
        print(f"[choice]   {client.name}: {results['models'][client.name]}", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"[choice] resultados -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
