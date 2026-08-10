"""
Wrapper que roda `run_tsf_orchestrator.py` em lote: vários datasets, um ou
mais combinadores (modelo LLM da Fase 3), cada combinação com o seu próprio
log — sem precisar copiar/colar um `nohup` por dataset.

Uma chamada:

    nohup python3 run_tsf_batch.py \\
        --datasets ETTH1 ETTH2 ANP_MONTHLY \\
        --combinators qwen3:30b-a3b \\
        --version v5_qwen \\
        > logs/v5_qwen_batch.log 2>&1 &

roda o que hoje seria três `nohup ... &` separados, um após o outro (sequencial
de propósito: todos os combinadores locais compartilham o mesmo servidor
Ollama, então rodar em paralelo faria os modelos disputarem a mesma GPU em vez
de ganhar tempo). Cada dataset grava no seu próprio arquivo, com o mesmo nome
que você já usava à mão:

    logs/v5_qwen_etth1.log
    logs/v5_qwen_etth2.log
    logs/v5_qwen_anp_monthly.log

Argumentos que o `run_tsf_orchestrator.py` já aceita (--reasoning,
--max-iterations, --windows, ...) continuam disponíveis — tudo depois de `--`
é repassado sem modificação para CADA chamada:

    python3 run_tsf_batch.py --datasets ETTH1 ETTH2 --combinators qwen3:30b-a3b \\
        --version v5_qwen -- --reasoning low --max-iterations 15

Comparando modelos (um contra o outro, mesmos datasets):

    python3 run_tsf_batch.py --datasets ETTH1 ETTH2 ANP_MONTHLY \\
        --combinators gpt-oss:20b qwen3:30b-a3b --version v5

roda os 2 modelos x 3 datasets = 6 execuções. Quando há mais de um combinador,
o `--version` recebe automaticamente o nome do combinador como sufixo (senão o
segundo modelo sobrescreveria o CSV de saída do primeiro, já que a pasta de
saída do orquestrador é só `orchestrator_react_<version>` — não inclui o nome
do modelo):

    resultados/orchestrator_react_v5_gpt-oss-20b/ETTH1.csv
    resultados/orchestrator_react_v5_qwen3-30b-a3b/ETTH1.csv

Use `--plan` para só imprimir os comandos e sair, sem rodar nada.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ORCHESTRATOR = REPO_ROOT / "run_tsf_orchestrator.py"
DEFAULT_SOURCE_DIR = "../forecasting_datasets"
DEFAULT_LOG_DIR = "logs"

# dataset (nome de resultados) -> arquivo .tsf de origem.
#
# Confirmado contra `run_tsf_regressors.py` (que gerou os CSVs base): ETTH1/2 e
# ETTM1/2 usam os arquivos com H/M MAIÚSCULOS — existem também `ETTh1.tsf` /
# `ETTm1.tsf` minúsculos no mesmo diretório, que são OUTRO arquivo (conteúdo
# diferente, verificado com `diff`). Usar o minúsculo por engano não dá erro
# na hora — dá `SeriesAlignmentError` (ou nada, se coincidir por acaso), então
# vale manter esse registro em vez de digitar o nome à mão.
DATASET_SOURCES: dict[str, str] = {
    "ETTH1": "ETTH1.tsf",
    "ETTH2": "ETTH2.tsf",
    "ETTM1": "ETTM1.tsf",
    "ETTM2": "ETTM2.tsf",
    "ANP_MONTHLY": "mes_11_venda_mensal.tsf",
    "NN5_WEEKLY_DATASET": "nn5_weekly_dataset.tsf",
    "M4_WEEKLY_DATASET": "m4_weekly_dataset.tsf",
    "US_BIRTHS_DATASET": "us_births_dataset.tsf",
}

# Datasets pequenos demais para o meta-modelo pooled (default
# pooled_meta_model_min_series=20 no orquestrador). Não é erro rodar mesmo
# assim, mas silenciosamente desliga uma tool inteira do catálogo do agente.
_SMALL_DATASET_WARNING = {
    "US_BIRTHS_DATASET": 1,
}


def _slug(text: str) -> str:
    """'qwen3:30b-a3b' -> 'qwen3-30b-a3b'; seguro para nome de arquivo/pasta."""
    return text.replace(":", "-").replace("/", "-").replace(" ", "-")


@dataclass
class Job:
    dataset: str
    source: str
    combinator: str
    version: str
    log_path: Path


def build_jobs(
    datasets: list[str],
    combinators: list[str],
    version: str,
    log_dir: Path,
    sources: dict[str, str],
) -> list[Job]:
    jobs = []
    multi_model = len(combinators) > 1
    for combinator in combinators:
        run_version = f"{version}_{_slug(combinator)}" if multi_model else version
        for dataset in datasets:
            source = sources.get(dataset)
            if source is None:
                known = ", ".join(sorted(sources))
                raise SystemExit(
                    f"Dataset '{dataset}' não está em DATASET_SOURCES. "
                    f"Datasets conhecidos: {known}. "
                    f"Adicione '{dataset}': '<arquivo>.tsf' no topo de run_tsf_batch.py."
                )
            log_name = f"{run_version}_{dataset.lower()}.log"
            jobs.append(Job(
                dataset=dataset,
                source=source,
                combinator=combinator,
                version=run_version,
                log_path=log_dir / log_name,
            ))
    return jobs


def build_command(job: Job, source_dir: str, extra_args: list[str]) -> list[str]:
    return [
        sys.executable, str(ORCHESTRATOR),
        "--dataset", job.dataset,
        "--source", job.source,
        "--source-dir", source_dir,
        "--combinator", job.combinator,
        "--version", job.version,
        *extra_args,
    ]


def run_job(job: Job, source_dir: str, extra_args: list[str]) -> tuple[int, float]:
    cmd = build_command(job, source_dir, extra_args)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> {job.dataset} / {job.combinator}  (log: {job.log_path})", flush=True)
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
        prog="python run_tsf_batch.py",
        description="Roda run_tsf_orchestrator.py em lote (vários datasets x um ou mais combinadores).",
    )
    p.add_argument("--datasets", nargs="+", required=True, help="ex: ETTH1 ETTH2 ANP_MONTHLY")
    p.add_argument("--combinators", nargs="+", required=True, help="ex: qwen3:30b-a3b")
    p.add_argument("--version", default="v5", help="sufixo da pasta de experimento (default v5)")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    p.add_argument(
        "--stop-on-first-failure", action="store_true",
        help="aborta o lote no primeiro dataset/combinador que falhar (default: continua e reporta no final)",
    )
    p.add_argument("--plan", action="store_true", help="só imprime os comandos e sai, sem rodar nada")
    args = p.parse_args(argv)

    log_dir = Path(args.log_dir)
    jobs = build_jobs(args.datasets, args.combinators, args.version, log_dir, DATASET_SOURCES)

    for ds in args.datasets:
        if ds in _SMALL_DATASET_WARNING:
            print(
                f"Aviso: '{ds}' tem {_SMALL_DATASET_WARNING[ds]} série(s) — abaixo do mínimo "
                f"para o meta-modelo pooled (20). A tool weights_pooled_meta_model fica "
                f"desligada nesse dataset; não é erro, mas mude se não for intencional."
            )

    print(f"Plano: {len(jobs)} execução(ões) — {len(args.datasets)} dataset(s) x {len(args.combinators)} combinador(es)")
    for job in jobs:
        print(f"  {job.dataset:24s} {job.combinator:20s} -> {job.log_path}")
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
            print(f"\nParando: {job.dataset}/{job.combinator} falhou e --stop-on-first-failure está ativo.")
            break

    failures = [r for r in results if r[1] != 0]
    print(f"\n=== Resumo: {len(results)} rodada(s), {len(failures)} falha(s) ===")
    for job, code, elapsed in results:
        status = "OK" if code == 0 else f"FALHOU (exit {code})"
        print(f"  {job.dataset:24s} {job.combinator:20s} {status:18s} {elapsed:7.0f}s  {job.log_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
