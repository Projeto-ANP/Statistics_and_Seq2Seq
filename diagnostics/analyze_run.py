#!/usr/bin/env python3
"""Analisador determinístico de um run do orchestrator ReAct (NÃO usa LLM).

Roda localmente sobre a pasta que o servidor devolveu
(./timeseries/mestrado/resultados/orchestrator_react_<version>/).

O que responde, por série e em agregado:
  1. FINAL vs MELHOR SEMENTE NO TESTE  — o agente (ou o argmin de validação)
     escolheu algo pior do que a melhor semente já disponível, medido na janela
     cega? (oráculo de referência: o piso que a Fase 2 já tinha em mãos, se
     tivesse como escolhê-la a posteriori)
  2. ORIGEM do vencedor (agent/baseline) e taxa de intervenção do agente.
  3. TRANSFERÊNCIA das propostas do agente: quando uma proposta vence a melhor
     semente na VALIDAÇÃO, com que frequência ela também vence no TESTE?
     (≈50% seria acaso; <50% é overfit sistemático)
  4. Spearman(score de validação, smape de teste) por série sobre as propostas.
  5. Comparação com o braço determinístico (--det) e distribuição dos verdicts.

Uso:
  python diagnostics/analyze_run.py --run orchestrator_react_v5 \
      --dataset NN5_WEEKLY_DATASET --tsf nn5_weekly_dataset.tsf \
      --det orchestrator_baseline_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from orchestrator_react import ingest as I
from orchestrator_react import pipeline as PL
from orchestrator_react import pool as POOL
from orchestrator_react.config import ReactConfig
from orchestrator_react.data_source import load_series_source
from orchestrator_react.registry import call_tool

BASE = "timeseries/mestrado/resultados"
SRC = os.path.expanduser("~/Documents/mestrado/forecasting_datasets")

ALL_MODELS = [
    "ARIMA", "ETS", "THETA", "rf", "catboost",
    "CWT_rf", "DWT_rf", "FT_rf",
    "CWT_catboost", "DWT_catboost", "FT_catboost",
    "ONLY_CWT_catboost", "ONLY_CWT_rf",
    "ONLY_DWT_catboost", "ONLY_DWT_rf",
    "ONLY_FT_catboost", "ONLY_FT_rf",
    "NaiveSeasonal", "NaiveMovingAverage",
]


def smape(preds, actual):
    p2 = np.asarray(preds, dtype=float).reshape(1, -1)
    t2 = np.asarray(actual, dtype=float).reshape(1, -1)
    return float(np.nanmean(2 * np.abs(p2 - t2) / (np.abs(p2) + np.abs(t2)), axis=1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pasta do run em resultados/")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tsf", required=True)
    ap.add_argument("--det", default=None, help="pasta do braço determinístico p/ comparar")
    ap.add_argument("--no-replay", action="store_true", help="pular o replay das propostas")
    ap.add_argument("--series", nargs="*", type=int, default=None)
    args = ap.parse_args()

    dataset, tsf, run = args.dataset, args.tsf, args.run
    models = [m for m in ALL_MODELS if os.path.exists(f"{BASE}/{m}/normal/{dataset}.csv")]
    frames = I.load_dataset_frames(models, dataset, BASE)
    n_series = I.count_series(dataset, models[0], BASE)
    source = load_series_source(tsf, n_expected_series=n_series, source_dir=SRC)
    cfg = ReactConfig()

    # mesma detecção de desalinhamento que run_dataset faz antes do loop
    try:
        bad = I.find_misaligned_models(
            models, dataset, 0, results_dir=BASE, frames=frames,
            n_windows=cfg.n_validation_windows,
        )
    except Exception:
        bad = {}
    drop = sorted(bad)
    models_eff = [m for m in models if m not in drop]
    if drop:
        print(f"modelos descartados por desalinhamento ({len(drop)}): {drop}")

    csv_path = f"{BASE}/{run}/{dataset}.csv"
    if not os.path.exists(csv_path):
        print(f"FALTANDO: {csv_path}")
        return 2
    df = pd.read_csv(csv_path, sep=";")

    art_dir = f"{BASE}/{run}/llm_artifacts/{dataset}"

    print("treinando pooled meta-models (LOO) ...")
    meta_models = PL._build_pooled_meta_models(
        models=models_eff, dataset=dataset, todo=list(range(n_series)), config=cfg,
        source=source, results_dir=BASE, frames=frames,
    )
    print(f"pooled: {len(meta_models)} séries atendidas "
          f"({'WITHHELD' if not meta_models else 'disponível'})\n")

    idxs = args.series if args.series is not None else list(range(n_series))
    finals, seed_tests, deltas, origins, verdicts = [], [], [], [], []
    n_worse = n_agent = 0
    print(f"{'idx':>4} {'origem':<9} {'final':>7} {'melhor_semente_teste':>21} {'delta':>8}")
    for idx in idxs:
        ing = I.load_series(
            models=models_eff, dataset=dataset, dataset_index=idx, config=cfg,
            results_dir=BASE, source_file=tsf, source_dir=SRC, frames=frames,
        )
        st = ing.state
        st.pooled_meta_model = meta_models.get(idx)
        seeds = POOL.seed_baselines(st, seed_pooled_meta_model=True)
        best_seed_test = min(
            smape(st.apply_to_test(a.spec)[0], ing.test_values) for a in seeds
        )
        row = df[df.dataset_index == idx].iloc[0]
        final_s = float(row.smape)
        desc = json.loads(row.description)
        origin = desc.get("origin")
        if origin == "agent":
            n_agent += 1
        delta = final_s - best_seed_test
        finals.append(final_s)
        seed_tests.append(best_seed_test)
        deltas.append(delta)
        origins.append(origin)
        if delta > 1e-6:
            n_worse += 1
        verdicts.append(str(row.get("selection_verdict", "?")))
        print(f"{idx:>4} {str(origin):<9} {final_s:>7.4f} {best_seed_test:>21.4f} {delta:>+8.4f}")

    n = len(finals)
    print(f"\n=== {run} / {dataset} ({n} séries) ===")
    print(f"origem agente: {n_agent}/{n}  |  final pior que melhor semente no TESTE: {n_worse}/{n}")
    print(f"média final       : {np.mean(finals):.4f}")
    print(f"média oráculo semente (teste): {np.mean(seed_tests):.4f}")
    print(f"média delta       : {np.mean(deltas):+.4f}")
    from collections import Counter
    print(f"verdicts: {dict(Counter(verdicts))}")

    if args.det:
        det_csv = f"{BASE}/{args.det}/{dataset}.csv"
        if os.path.exists(det_csv):
            det = pd.read_csv(det_csv, sep=";")
            det_m = det[det.dataset_index.isin(idxs)].smape.astype(float).mean()
            print(f"braço determinístico {args.det}: média {det_m:.4f} "
                  f"(agente: {np.mean(finals):.4f}, delta={np.mean(finals)-det_m:+.4f})")

    # ── replay das propostas do agente ────────────────────────────────────────
    if not args.no_replay and os.path.isdir(art_dir):
        print("\nreplay das propostas (determinístico, sem LLM) ...")
        all_prop, per_series_rho, transfer = [], [], []
        n_prop = n_fail = 0
        for idx in idxs:
            art_path = f"{art_dir}/dataset_{idx}.json"
            if not os.path.exists(art_path):
                continue
            art = json.load(open(art_path))
            ing = I.load_series(
                models=models_eff, dataset=dataset, dataset_index=idx, config=cfg,
                results_dir=BASE, source_file=tsf, source_dir=SRC, frames=frames,
            )
            st = ing.state
            st.pooled_meta_model = meta_models.get(idx)
            seeds = POOL.seed_baselines(st, seed_pooled_meta_model=True)
            best_seed_val = min(a.score for a in seeds)
            best_seed_test = min(
                smape(st.apply_to_test(a.spec)[0], ing.test_values) for a in seeds
            )
            props = []
            for e in art["react"]["trajectory"]:
                action = e["action"]
                a = dict(e.get("action_args") or {})
                if action != "evaluate_strategy":
                    try:
                        call_tool(st, action, a, withheld={})
                    except Exception:
                        pass
                    continue
                n_prop += 1
                try:
                    ok, _ = call_tool(st, action, a, withheld={})
                    if not ok:
                        n_fail += 1
                        continue
                    attempt = st.attempts[-1]
                    t = smape(st.apply_to_test(attempt.spec)[0], ing.test_values)
                    props.append((attempt.score, t, attempt.attempt_id))
                    all_prop.append((idx, attempt.score, t))
                except Exception:
                    n_fail += 1
            if len(props) >= 3:
                vals = np.array([p[0] for p in props])
                ts = np.array([p[1] for p in props])
                if len(set(vals.tolist())) > 1:
                    rho = np.corrcoef(np.argsort(vals), np.argsort(ts))[0, 1]
                    per_series_rho.append(rho)
            for (val, t, aid) in props:
                if val < best_seed_val - 1e-9:
                    transfer.append(t < best_seed_test)

        print(f"propostas replayed: {n_prop} (falhas: {n_fail})")
        if per_series_rho:
            rhos = np.array(per_series_rho)
            print(f"Spearman(val, teste) por série: média {np.mean(rhos):+.3f} "
                  f"mediana {np.median(rhos):+.3f} positivo {np.sum(rhos>0)}/{len(rhos)}")
        if transfer:
            print(f"transferência (venceu semente na val E no teste): "
                  f"{sum(transfer)}/{len(transfer)} ({sum(transfer)/len(transfer)*100:.0f}%)"
                  f"  [acaso seria ~50%]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
