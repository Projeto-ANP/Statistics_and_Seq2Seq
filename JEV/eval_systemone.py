#!/usr/bin/env python3
"""FASE 0 — avaliação de modelos System One (Jev-class) para o papel de GATE.

Mede, sobre o gate_dataset.jsonl (rótulos `label_val_w0/1/2` = vencedor LOO de
cada janela de validação), a qualidade de decisão de cada modelo:

  - acurácia (P>0.5 vs rótulo por janela)
  - Brier e ECE (calibração — o que o LAYA zero-shot saturou)
  - confiança média + fração de decisões confiantes (>=0.9) e erro delas
  - latência p50/p95 por decisão

Pergunta por registro (3 janelas em UMA passada, formato noul):

  "Will this candidate strategy be the winner of validation window w of 3,
   scored leave-one-out (everything else fitted on the other two windows)?"

Modelos:
  - `laya`           : LAYA multilingual zero-shot local (referência atual)
  - `name=URL`       : qualquer servidor TypeSafe-compatível (/v1/systemone):
                       kev   -> http://127.0.0.1:8009
                       eikos -> http://127.0.0.1:8000
  Normaliza os dois formatos de resposta noul ("noul" do kev/typesafe e
  "probability" do eikos).

Uso (no servidor):
  python3 JEV/eval_systemone.py --laya --endpoints kev4b=http://127.0.0.1:8009 \
      eikos4b=http://127.0.0.1:8000 --limit 500 --out JEV/phase0_results.json
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "JEV"))

WINDOWS = ("w0", "w1", "w2")


# ─────────────────────────── clientes ─────────────────────────────────────────


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
        out = self.agent.predict(state, questions)
        return out, time.perf_counter() - t0


class HttpClient:
    def __init__(self, name: str, url: str, model: str) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self.model = model

    def ask(self, state: str, questions: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        return http_systemone(self.url, state, questions, self.model)


def _p_yes(ans: Any) -> Optional[float]:
    if not isinstance(ans, dict):
        return None
    if "noul" in ans and isinstance(ans["noul"], (int, float)):
        return float(ans["noul"])
    if "probability" in ans and isinstance(ans["probability"], (int, float)):
        return float(ans["probability"])
    return None


def _conf(ans: Any) -> Optional[float]:
    if not isinstance(ans, dict):
        return None
    for key in ("confidence", "answer_confidence"):
        if isinstance(ans.get(key), (int, float)):
            return float(ans[key])
    return None


# ─────────────────────────── métricas ─────────────────────────────────────────


def ece(probs: List[float], labels: List[int], bins: int = 10) -> float:
    """Expected Calibration Error por bin de confiança."""
    if not probs:
        return float("nan")
    pairs = sorted(zip(probs, labels), key=lambda p: p[0])
    n = len(pairs)
    out = 0.0
    for b in range(bins):
        lo = b * n // bins
        hi = (b + 1) * n // bins
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        conf = sum(p for p, _ in chunk) / len(chunk)
        acc = sum(l for _, l in chunk) / len(chunk)
        out += len(chunk) / n * abs(conf - acc)
    return out


def evaluate(client: Any, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    probs: List[float] = []
    labels: List[int] = []
    confs: List[float] = []
    lats: List[float] = []
    errors: List[str] = []
    confident_taken = confident_wrong = 0
    decisions: List[Dict[str, Any]] = []  # por decisão (para ranqueamento/auditoria)

    for i, rec in enumerate(records):
        questions = {
            f"transfer_{w}": {
                "type": "noul",
                "instructions": (
                    f"Will this candidate strategy be the winner of validation "
                    f"window {int(w[1]) + 1} of 3, scored leave-one-out "
                    f"(everything else fitted on the other two windows)? "
                    f"Answer yes only if you expect it to win that window."
                ),
            }
            for w in WINDOWS
        }
        try:
            out, lat = client.ask(rec["state"], questions)
            lats.append(lat)
        except Exception as exc:
            errors.append(f"record {i}: {type(exc).__name__}: {exc}")
            continue
        answers = out.get("answers") or {}
        rec_p: Dict[str, float] = {}
        for w in WINDOWS:
            ans = answers.get(f"transfer_{w}")
            p = _p_yes(ans)
            label = int(rec.get(f"label_val_{w}", 0))
            if p is None:
                errors.append(f"record {i} {w}: no noul/probability in answer")
                continue
            probs.append(p)
            labels.append(label)
            rec_p[w] = p
            decisions.append({
                "dataset": rec.get("dataset"), "series": rec.get("series"),
                "spec": json.dumps(rec.get("spec"), sort_keys=True, default=str),
                "window": w, "p": round(p, 4), "label": label,
                "label_val": rec.get("label_val"),
                "universe_size": rec.get("universe_size"),
            })
            c = _conf(ans)
            if c is not None:
                confs.append(c)
                if c >= 0.9:
                    confident_taken += 1
                    confident_wrong += (1 if (p >= 0.5) != bool(label) else 0)

    n = len(probs)
    if not n:
        return {"n": 0, "errors": errors[:5]}
    acc = sum(1 for p, lab in zip(probs, labels) if (p >= 0.5) == bool(lab)) / n
    brier = sum((p - lab) ** 2 for p, lab in zip(probs, labels)) / n
    ce = ece(probs, labels)
    base_rate = sum(labels) / n
    always_no_acc = 1.0 - base_rate
    always_yes_acc = base_rate
    # Brier dos baselines constantes (P = taxa-base fixa)
    const_brier = sum((base_rate - lab) ** 2 for lab in labels) / n
    # ── ranqueamento: o argmax de P escolhe o vencedor do universo? ─────────
    groups: Dict[Any, Dict[str, Any]] = {}
    for d in decisions:
        key = (d["dataset"], d["series"])
        g = groups.setdefault(key, {"records": {}, "label_val": {}, "universe_size": 0})
        spec = d["spec"]
        g["records"].setdefault(spec, []).append(d["p"])
        g["label_val"][spec] = int(d["label_val"] or 0)
        g["universe_size"] = max(g["universe_size"], int(d["universe_size"] or 1))
    argmax_hit = top3_hit = 0
    n_groups = 0
    mean_size = 0.0
    for key, g in groups.items():
        if not g["records"]:
            continue
        n_groups += 1
        mean_size += g["universe_size"]
        ranked = sorted(g["records"].items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        winner_specs = [s for s, lv in g["label_val"].items() if lv == 1]
        if not winner_specs:
            continue
        argmax_hit += (1 if ranked[0][0] in winner_specs else 0)
        top3_hit += (1 if any(s in winner_specs for s, _ in ranked[:3]) else 0)
    mean_size = mean_size / n_groups if n_groups else 1.0
<<<<<<< HEAD
=======
    mean_conf = sum(confs) / len(confs) if confs else None
>>>>>>> b46f8c78812b28c02bac04555ae2b6b25786b21e
    lats_sorted = sorted(lats)
    p50 = lats_sorted[len(lats_sorted) // 2] if lats_sorted else float("nan")
    p95 = lats_sorted[min(int(len(lats_sorted) * 0.95), len(lats_sorted) - 1)] if lats_sorted else float("nan")
    return {
        "n_decisions": n,
        "n_requests": len(lats),
        "base_rate": round(base_rate, 4),
        "accuracy": round(acc, 4),
        "always_no_accuracy": round(always_no_acc, 4),
        "always_yes_accuracy": round(always_yes_acc, 4),
        "brier": round(brier, 4),
        "brier_constant_baseline": round(const_brier, 4),
        "ece": round(ce, 4),
        "n_groups": n_groups,
        "mean_universe_size": round(mean_size, 1),
        "random_argmax_hit": round(1.0 / mean_size, 4),
        "argmax_picks_winner": round(argmax_hit / n_groups, 4) if n_groups else None,
        "top3_contains_winner": round(top3_hit / n_groups, 4) if n_groups else None,
        "mean_confidence": round(mean_conf, 4) if confs else None,
        "confident_taken": confident_taken,
        "confident_wrong": confident_wrong,
        "confident_error_rate": round(confident_wrong / confident_taken, 4)
        if confident_taken else None,
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "errors": errors[:5],
        "n_errors": len(errors),
        "decisions": decisions,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fase 0: avaliação de System One para o gate")
    ap.add_argument("--data", default="JEV/data/gate_dataset.jsonl")
    ap.add_argument("--endpoints", nargs="+", default=[],
                    help="name=URL (servidor TypeSafe-compatível, ex. kev4b=http://127.0.0.1:8009)")
    ap.add_argument("--laya", action="store_true", help="incluir LAYA multilingual local")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="JEV/phase0_results.json")
    args = ap.parse_args(argv)

    records: List[Dict[str, Any]] = []
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # ── amostra por UNIVERSO COMPLETO (dataset, série): o ranqueamento
    #    (argmax escolhe o vencedor?) exige todos os candidatos do grupo ──────
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r.get("dataset"), r.get("series")), []).append(r)
    keys = sorted(groups)
    random.seed(args.seed)
    random.shuffle(keys)
    chosen: List[Dict[str, Any]] = []
    for k in keys:
        if args.limit and len(chosen) + len(groups[k]) > args.limit and chosen:
            break
        chosen.extend(groups[k])
    records = chosen
    print(f"[phase0] {len(records)} registros em {len(set((r.get('dataset'), r.get('series')) for r in records))} universos completos de {args.data}",
          flush=True)

    clients: List[Any] = []
    if args.laya:
        clients.append(LayaClient())
    for spec in args.endpoints:
        name, url = spec.split("=", 1)
        clients.append(HttpClient(name, url, model=name))

    results: Dict[str, Any] = {"config": {"limit": args.limit, "seed": args.seed,
                                          "n_records": len(records)},
                               "models": {}}
    for client in clients:
        print(f"[phase0] avaliando {client.name} ...", flush=True)
        t0 = time.perf_counter()
        results["models"][client.name] = evaluate(client, records)
        print(f"[phase0]   {client.name}: {results['models'][client.name]}",
              flush=True)
        print(f"[phase0]   tempo total: {time.perf_counter() - t0:.1f}s", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"[phase0] resultados -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
