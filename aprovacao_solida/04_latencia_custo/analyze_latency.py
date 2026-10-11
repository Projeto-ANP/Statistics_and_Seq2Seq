#!/usr/bin/env python3
"""Custo e latencia do agente, direto dos artefatos (sem rodar nada).

Le `llm_artifacts/<DATASET>/dataset_*.json` de uma ou mais versoes e resume:
tempo total e por chamada, numero de chamadas, iteracoes, tokens de prompt e de
avaliacao, caracteres de reasoning, respostas vazias, retries de erro e
fallbacks de `think=False`. Serve para a secao de custo-beneficio e para o A/B
de `--reasoning low` (rode um subconjunto com reasoning low numa versao nova e
compare as duas).

Uso:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    python3 aprovacao_solida/04_latencia_custo/analyze_latency.py \\
        --versions orchestrator_react_v5 orchestrator_react_v5_qwen
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import DATASETS, DEFAULT_RESULTS_DIR  # noqa: E402


def _num(x):
    return float(x) if isinstance(x, (int, float)) and x == x else None


def series_stats(payload):
    """Le os numeros de custo dos artefatos, em dois esquemas.

    Runs antigos (v5) gravaram apenas `trajectory` (com `llm_call_s` por turno)
    e um `summary` sem `llm_call_s_total`/`step_details`. Runs novos (codigo
    atual) gravam `step_details` com tokens e `thinking_chars`. O script aceita
    os dois; `artifacts_schema` diz qual foi usado, para ninguem comparar
    tokens de um esquema com zeros do outro.
    """
    react = payload.get("react") or {}
    summary = react.get("summary") or {}
    trajectory = react.get("trajectory") or []
    steps = react.get("step_details") or []
    full = bool(steps)
    n_calls = len(steps) if full else len(trajectory)

    def collect(key):
        vals = [s.get(key) for s in steps]
        return [float(v) for v in vals if isinstance(v, (int, float)) and v == v]

    eval_tokens = collect("ollama_eval_count")
    prompt_tokens = collect("ollama_prompt_eval_count")
    thinking = collect("ollama_thinking_chars")

    llm_s = _num(summary.get("llm_call_s_total"))
    if llm_s is None:
        calls = [float(e.get("llm_call_s") or 0.0) for e in trajectory]
        llm_s = sum(calls) if calls else None
    tool_s = _num(summary.get("tool_exec_s_total"))
    if tool_s is None:
        calls = [float(e.get("tool_exec_s") or 0.0) for e in trajectory]
        tool_s = sum(calls) if calls else None
    elapsed = _num(summary.get("elapsed_s"))

    done_reasons = {}
    for s in steps:
        reason = s.get("ollama_done_reason")
        if reason:
            done_reasons[str(reason)] = done_reasons.get(str(reason), 0) + 1

    return {
        "artifacts_schema": "full" if full else "legacy",
        "elapsed_s": elapsed,
        "iterations": _num(summary.get("iterations_used")),
        "llm_s": llm_s,
        "tool_s": tool_s,
        "n_calls": n_calls,
        "llm_s_per_call": (llm_s / n_calls) if (llm_s is not None and n_calls) else None,
        "eval_tokens": sum(eval_tokens) if eval_tokens else 0,
        "prompt_tokens": sum(prompt_tokens) if prompt_tokens else 0,
        "thinking_chars": sum(thinking) if thinking else 0,
        "empty_responses": _num(summary.get("empty_responses")),
        "llm_error_retries": _num(summary.get("llm_error_retries")),
        "think_fallbacks": sum(1 for s in steps if s.get("ollama_think_fallback_off")),
        "model": summary.get("llm_model"),
        "stop_reason": summary.get("stop_reason"),
        "done_reasons": done_reasons,
    }


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    p = argparse.ArgumentParser(description="Latencia e custo por versao/dataset.")
    p.add_argument("--versions", nargs="+", default=["orchestrator_react_v5", "orchestrator_react_v5_qwen"])
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--out", default="outputs/resultados/aprovacao_solida/latencia_custo.csv")
    args = p.parse_args(argv)

    all_rows = []
    for version in args.versions:
        base = os.path.join(args.results_dir, version, "llm_artifacts")
        if not os.path.isdir(base):
            print(f"{version}: sem llm_artifacts em {base}")
            continue
        print(f"\n######## {version} ########")
        print(
            f"{'dataset':<20}{'n':>5}{'iter':>6}{'calls':>6}{'s/call':>9}"
            f"{'llm_s':>9}{'tot_h':>7}{'think_k':>9}{'eval_tok':>10}{'prom_tok':>11}"
            f"{'empty':>7}{'erretr':>7}{'fallb':>6}"
        )
        for dataset in args.datasets:
            files = sorted(glob.glob(os.path.join(base, dataset, "dataset_*.json")))
            rows = []
            for path in files:
                try:
                    with open(path, encoding="utf-8") as fh:
                        rows.append(series_stats(json.load(fh)))
                except Exception:
                    continue
            if not rows:
                continue
            n = len(rows)
            per_call = _median([r["llm_s_per_call"] for r in rows])
            llm_s = _mean([r["llm_s"] for r in rows])
            total_h = sum(r["elapsed_s"] or 0 for r in rows) / 3600.0
            print(
                f"{dataset:<20}{n:>5}"
                f"{(_mean([r['iterations'] for r in rows]) or 0):>6.1f}"
                f"{(_median([r['n_calls'] for r in rows]) or 0):>6.1f}"
                f"{(per_call or 0):>9.1f}"
                f"{(llm_s or 0):>9.1f}"
                f"{total_h:>7.2f}"
                f"{(sum(r['thinking_chars'] for r in rows) / 1000.0):>9.1f}"
                f"{sum(r['eval_tokens'] for r in rows):>10d}"
                f"{sum(r['prompt_tokens'] for r in rows):>11d}"
                f"{sum(r['empty_responses'] or 0 for r in rows):>7.0f}"
                f"{sum(r['llm_error_retries'] or 0 for r in rows):>7.0f}"
                f"{sum(r['think_fallbacks'] for r in rows):>6.0f}"
            )
            done = {}
            for r in rows:
                for k, v in (r["done_reasons"] or {}).items():
                    done[k] = done.get(k, 0) + v
            row = {
                "version": version,
                "dataset": dataset,
                "n_series": n,
                "mean_iterations": _mean([r["iterations"] for r in rows]),
                "median_calls": _median([r["n_calls"] for r in rows]),
                "median_s_per_call": per_call,
                "mean_llm_s": llm_s,
                "total_hours": total_h,
                "total_thinking_kchars": sum(r["thinking_chars"] for r in rows) / 1000.0,
                "total_eval_tokens": sum(r["eval_tokens"] for r in rows),
                "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                "empty_responses": sum(r["empty_responses"] or 0 for r in rows),
                "llm_error_retries": sum(r["llm_error_retries"] or 0 for r in rows),
                "think_fallbacks": sum(r["think_fallbacks"] for r in rows),
                "models": ",".join(sorted({str(r["model"]) for r in rows})),
                "artifacts_schema": ",".join(sorted({r["artifacts_schema"] for r in rows})),
                "done_reasons": json.dumps(done, ensure_ascii=False),
            }
            all_rows.append(row)

    if all_rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.out, sep=";", index=False)
        print(f"\nsalvo em: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
