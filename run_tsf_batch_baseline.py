"""
Wrapper que roda `run_tsf_baselines.py` (orquestrador SEM agente) em lote: vários
datasets, cada um com o seu próprio log. Mesma interface do `run_tsf_batch.py`,
só que sem `--combinators` (não há LLM).

Uma chamada:

    nohup python3 run_tsf_batch_baseline.py \\
        --datasets ETTM1 ETTM2 ETTH1 ETTH2 ANP_MONTHLY NN5_WEEKLY_DATASET M4_WEEKLY_DATASET \\
        --version v1 \\
        > logs/baseline_batch.log 2>&1 &

Roda os datasets em sequência, cada um com o modelo pool padrão do orquestrador
(19 modelos). Grava em `resultados/orchestrator_baseline_<version>/<DATASET>.csv`
e loga em `logs/baseline_<version>_<dataset>.log`.

Argumentos que o `run_tsf_baselines.py` aceita (--windows, --no-pooled-meta-model,
...) vão depois de `--` e são repassados para CADA chamada:

    python3 run_tsf_batch_baseline.py --datasets ETTH1 ETTH2 --version v1 -- --windows 3

Use `--plan` para só imprimir os comandos e sair, sem rodar nada.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from run_tsf_batch import DATASET_SOURCES, DEFAULT_LOG_DIR, DEFAULT_SOURCE_DIR

REPO_ROOT = Path(__file__).resolve().parent
BASELINES = REPO_ROOT / "run_tsf_baselines.py"


@dataclass
class Job:
    dataset: str
    source: str
    version: str
    log_path: Path


def build_jobs(datasets: list[str], version: str, log_dir: Path) -> list[Job]:
    jobs = []
    for dataset in datasets:
        source = DATASET_SOURCES.get(dataset)
        if source is None:
            known = ", ".join(sorted(DATASET_SOURCES))
            raise SystemExit(
                f"Dataset '{dataset}' não está em DATASET_SOURCES (run_tsf_batch.py). "
                f"Datasets conhecidos: {known}."
            )
        jobs.append(Job(
            dataset=dataset,
            source=source,
            version=version,
            log_path=log_dir / f"baseline_{version}_{dataset.lower()}.log",
        ))
    return jobs


def build_command(job: Job, source_dir: str, extra_args: list[str]) -> list[str]:
    return [
        sys.executable, str(BASELINES),
        "--dataset", job.dataset,
        "--source", job.source,
        "--source-dir", source_dir,
        "--version", job.version,
        *extra_args,
    ]


def run_job(job: Job, source_dir: str, extra_args: list[str]) -> tuple[int, float]:
    cmd = build_command(job, source_dir, extra_args)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> {job.dataset}  (log: {job.log_path})", flush=True)
    print(f"   {' '.join(cmd)}", flush=True)

    t0 = time.time()
    with open(job.log_path, "w") as log_file:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FALHOU (exit {proc.returncode})"
    print(f"   {status} em {elapsed:.0f}s", flush=True)
    return proc.returncode, elapsed


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        argv, extra_args = argv[:split], argv[split + 1:]
    else:
        extra_args = []

    p = argparse.ArgumentParser(
        prog="python run_tsf_batch_baseline.py",
        description="Roda run_tsf_baselines.py (sem agente) em lote, vários datasets.",
    )
    p.add_argument("--datasets", nargs="+", required=True, help="ex: ETTH1 ETTH2 ANP_MONTHLY")
    p.add_argument("--version", default="v1", help="sufixo da pasta orchestrator_baseline_<version> (default v1)")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    p.add_argument(
        "--stop-on-first-failure", action="store_true",
        help="aborta o lote no primeiro dataset que falhar (default: continua e reporta no final)",
    )
    p.add_argument("--plan", action="store_true", help="só imprime os comandos e sai, sem rodar nada")
    args = p.parse_args(argv)

    jobs = build_jobs(args.datasets, args.version, Path(args.log_dir))

    print(f"Plano: {len(jobs)} execução(ões) — {len(args.datasets)} dataset(s), sem LLM")
    for job in jobs:
        print(f"  {job.dataset:24s} -> {job.log_path}")
    if extra_args:
        print(f"Args extras repassados para cada chamada: {' '.join(extra_args)}")
    print()

    if args.plan:
        return 0

    results: list[tuple[Job, int, float]] = []
    for job in jobs:
        code, elapsed = run_job(job, args.source_dir, extra_args)
        results.append((job, code, elapsed))
        if code != 0 and args.stop_on_first_failure:
            print(f"\nParando: {job.dataset} falhou e --stop-on-first-failure está ativo.")
            break

    failures = [r for r in results if r[1] != 0]
    print(f"\n=== Resumo: {len(results)} rodada(s), {len(failures)} falha(s) ===")
    for job, code, elapsed in results:
        status = "OK" if code == 0 else f"FALHOU (exit {code})"
        print(f"  {job.dataset:24s} {status:18s} {elapsed:7.0f}s  {job.log_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
