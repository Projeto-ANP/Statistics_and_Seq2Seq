#!/usr/bin/env python3
"""Roda toda a bateria deterministica da aprovacao solida, em sequencia.

Gera, sem LLM, tudo o que falta para fechar a defesa:

    01 busca nao-LLM nos sete datasets, mais a analise pareada;
    02 replay vencedor versus melhor semente, por run disponivel;
    05 mediana nas sementes (resultado negativo ou positivo);
    06 sensibilidade da selecao aos pesos da pontuacao;
    07 novas combinacoes (por horizonte, Caruana, best_single) contra o piso;
    03 estatistica pareada por dataset com correcao de Holm;
    04 latencia e custo dos runs existentes.

Tudo e gravado em `outputs/resultados/aprovacao_solida/`, que e a pasta a
commitar. Os logs ficam em `logs/aprovacao_solida/`.

Uso no servidor:

    conda activate agno
    cd ~/Statistics_and_Seq2Seq
    python3 aprovacao_solida/run_all.py

Smoke (3 series de um dataset, ~2 min):

    python3 aprovacao_solida/run_all.py --datasets ETTH1 --limit 3

Em background:

    nohup python3 aprovacao_solida/run_all.py > logs/aprovacao_solida_all.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import DATASETS  # noqa: E402

DEFAULT_RESULTS_DIR = "./timeseries/mestrado/resultados"
DEFAULT_RUN = "orchestrator_react_v5"
EXTRA_RUNS = ("orchestrator_react_v5_qwen", "orchestrator_react_v5_gemma26")


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def run_step(name: str, cmd: list, log_dir: Path, dry_run: bool = False) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    _log(f"\n[{datetime.now():%H:%M:%S}] === {name} ===")
    _log("$ " + " ".join(cmd))
    if dry_run:
        return 0
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(_REPO), stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FALHOU (exit {proc.returncode})"
    _log(f"[{datetime.now():%H:%M:%S}] {name}: {status} em {elapsed:.0f}s  (log: {log_path})")
    if proc.returncode != 0:
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            _log("  ultimas linhas do log:")
            for line in tail:
                _log("    " + line)
        except Exception:
            pass
    return proc.returncode


def available_runs(results_dir: str, main_run: str) -> list:
    runs = [main_run]
    for extra in EXTRA_RUNS:
        if extra != main_run and os.path.isdir(os.path.join(results_dir, extra)):
            runs.append(extra)
    return runs


def main(argv=None):
    p = argparse.ArgumentParser(description="Bateria determinística da aprovação sólida.")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--limit", type=int, default=None, help="apenas as N primeiras series (smoke)")
    p.add_argument("--search-version", default="v1")
    p.add_argument("--run", default=DEFAULT_RUN, help="run principal (vencedor, replay, sensibilidade)")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", default="outputs/resultados/aprovacao_solida")
    p.add_argument("--log-dir", default="logs/aprovacao_solida")
    p.add_argument("--skip", nargs="*", default=[],
                   help="etapas a pular: busca replay medianas sensibilidade combos stats latencia")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    py = sys.executable
    out = Path(args.output_dir)
    logs = Path(args.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    skip = {s.strip().lower() for s in args.skip}
    limit_args = ["--limit", str(args.limit)] if args.limit else []
    runs = available_runs(args.results_dir, args.run)

    _log(f"datasets   : {args.datasets}")
    _log(f"runs       : {runs}")
    _log(f"saida      : {out}")
    _log(f"logs       : {logs}")
    _log(f"limit      : {args.limit}")
    _log(f"pular      : {sorted(skip) if skip else 'nada'}")

    results = []
    t0 = time.time()

    # 01 busca nao-LLM
    if "busca" not in skip:
        for ds in args.datasets:
            results.append((
                f"01_busca_{ds}",
                run_step(
                    f"01_busca_{ds}",
                    [py, "aprovacao_solida/01_busca_nao_llm/run_search.py",
                     "--dataset", ds, "--version", args.search_version,
                     "--methods", "seeds,random,greedy,ucb", *limit_args],
                    logs, args.dry_run,
                ),
            ))
        results.append((
            "01_analise",
            run_step(
                "01_analise",
                [py, "aprovacao_solida/01_busca_nao_llm/analyze_search.py",
                 "--version", args.search_version,
                 "--out", str(out / "analise_busca.csv")],
                logs, args.dry_run,
            ),
        ))
        if not args.dry_run:
            src_dir = os.path.join(args.results_dir, f"orchestrator_search_{args.search_version}")
            dst_dir = out / "busca_csv"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for ds in args.datasets:
                src = os.path.join(src_dir, f"{ds}.csv")
                if os.path.isfile(src):
                    shutil.copy2(src, dst_dir / f"{ds}.csv")
            _log(f"CSVs da busca copiados para {dst_dir}")

    # 02 replay
    if "replay" not in skip:
        for run in runs:
            results.append((
                f"02_replay_{run}",
                run_step(
                    f"02_replay_{run}",
                    [py, "aprovacao_solida/02_replay_winner_curse/replay_gate.py",
                     "--run", run, "--datasets", *args.datasets,
                     "--out", str(out / f"replay_gate_{run}.csv"), *limit_args],
                    logs, args.dry_run,
                ),
            ))

    # 05 medianas
    if "medianas" not in skip:
        results.append((
            "05_medianas",
            run_step(
                "05_medianas",
                [py, "aprovacao_solida/05_sementes_medianas/seed_medians.py",
                 "--datasets", *args.datasets,
                 "--out", str(out / "sementes_medianas.csv"), *limit_args],
                logs, args.dry_run,
            ),
        ))

    # 06 sensibilidade
    if "sensibilidade" not in skip:
        results.append((
            "06_sensibilidade",
            run_step(
                "06_sensibilidade",
                [py, "aprovacao_solida/06_sensibilidade_score/run_score_sensitivity.py",
                 "--run", args.run, "--datasets", *args.datasets,
                 "--out", str(out / "sensibilidade_score.csv"), *limit_args],
                logs, args.dry_run,
            ),
        ))

    # 07 novas combinacoes
    if "combos" not in skip:
        results.append((
            "07_novas_combinacoes",
            run_step(
                "07_novas_combinacoes",
                [py, "aprovacao_solida/07_novas_combinacoes/run_new_combos.py",
                 "--datasets", *args.datasets,
                 "--out", str(out / "novas_combinacoes.csv"), *limit_args],
                logs, args.dry_run,
            ),
        ))

    # 03 estatistica
    if "stats" not in skip:
        results.append((
            "03_stats",
            run_step(
                "03_stats",
                [py, "aprovacao_solida/03_estatistica_por_dataset/stats_por_dataset.py",
                 "--search-version", args.search_version,
                 "--out", str(out / "stats_por_dataset.csv")],
                logs, args.dry_run,
            ),
        ))

    # 04 latencia
    if "latencia" not in skip:
        results.append((
            "04_latencia",
            run_step(
                "04_latencia",
                [py, "aprovacao_solida/04_latencia_custo/analyze_latency.py",
                 "--versions", *runs,
                 "--out", str(out / "latencia_custo.csv")],
                logs, args.dry_run,
            ),
        ))

    elapsed = time.time() - t0
    failures = [name for name, code in results if code != 0]
    _log("\n" + "=" * 72)
    _log(f"terminado em {elapsed / 60:.1f} min | etapas: {len(results)} | falhas: {len(failures)}")
    for name, code in results:
        _log(f"  {name:<28} {'OK' if code == 0 else 'FALHOU'}")
    if failures:
        _log("veja os logs das etapas que falharam antes de commitar.")
    _log("")
    _log("saidas para commitar:")
    for path in sorted(out.rglob("*.csv")):
        _log(f"  {path}")
    _log("")
    _log("no servidor, depois de conferir:")
    _log(f"  git add {out}")
    _log('  git commit -m "resultados: validacao aprovacao solida"')
    _log("  git push")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
