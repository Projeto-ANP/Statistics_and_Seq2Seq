#!/usr/bin/env python3
"""Driver de diagnóstico para rodar NO SERVIDOR (é onde está o Ollama).

Exemplos:

  # braço determinístico (sem LLM) — referência de piso
  python diagnostics/diag_entry.py --dataset NN5_WEEKLY_DATASET --source nn5_weekly_dataset.tsf \
      --version diag_det --use-llm 0

  # re-executar a config publicada (âncora de sanidade; deve reproduzir v5)
  python diagnostics/diag_entry.py --dataset NN5_WEEKLY_DATASET --source nn5_weekly_dataset.tsf \
      --version diag_control --seed 7

  # variar a semente do amostrador (medir variância do agente)
  python diagnostics/diag_entry.py --dataset NN5_WEEKLY_DATASET --source nn5_weekly_dataset.tsf \
      --version diag_seed13 --seed 13

  # reasoning off (remove o canal harmony do gpt-oss)
  python diagnostics/diag_entry.py --dataset NN5_WEEKLY_DATASET --source nn5_weekly_dataset.tsf \
      --version diag_reasoning_off --reasoning off

  # gate de calibração (pula o loop quando o ranking já está estável)
  python diagnostics/diag_entry.py --dataset NN5_WEEKLY_DATASET --source nn5_weekly_dataset.tsf \
      --version diag_calgate --config '{"calibration_gate": true}'

  # ablações de prompt/contexto
  python diagnostics/diag_entry.py --dataset NN5_WEEKLY_DATASET --source nn5_weekly_dataset.tsf \
      --version diag_nocard --config '{"dataset_card": false}'
  python diagnostics/diag_entry.py --dataset NN5_WEEKLY_DATASET --source nn5_weekly_dataset.tsf \
      --version diag_noseedpooled --config '{"seed_pooled_meta_model": false}'

  # só uma série (smoke) — ATENÇÃO: com <20 séries o pooled meta-model e o
  # dataset card somem; NÃO use runs parciais como braço de ablação real.
  python diagnostics/diag_entry.py --dataset ETTM2 --source ETTM2.tsf \
      --version smoke --indices 0

Saída: CSV + artifacts em ./timeseries/mestrado/resultados/orchestrator_react_<version>/.
Traga de volta a pasta inteira do run; `diagnostics/analyze_run.py` roda localmente
(determinístico, sem LLM) sobre ela.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from run_tsf_orchestrator import exec_dataset_orchestrator  # noqa: E402
from orchestrator_react.config import LLMRole, ReactConfig  # noqa: E402

DEFAULT_SOURCE_DIR = os.path.expanduser("~/forecasting_datasets")


def _parse_reasoning(raw: str | None):
    """None = default do servidor; off/false/no = False; resto = intensidade."""
    if raw is None:
        return None
    if str(raw).lower() in {"off", "false", "no"}:
        return False
    return str(raw).lower()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reasoning", default=None)
    p.add_argument("--use-llm", type=int, default=1)
    p.add_argument("--config", default="{}", help="JSON de overrides do ReactConfig")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--indices", nargs="*", type=int, default=None)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--max-llm-failures", type=int, default=5)
    args = p.parse_args()

    cfg = ReactConfig(**json.loads(args.config))

    combinator = LLMRole(
        model=args.model if args.use_llm else None,
        temperature=0.2,
        seed=args.seed,
    )
    combinator.reasoning = _parse_reasoning(args.reasoning)

    summary = exec_dataset_orchestrator(
        dataset=args.dataset,
        source_file=args.source,
        source_dir=args.source_dir,
        combinator_model=combinator,
        use_llm=bool(args.use_llm),
        version=args.version,
        n_windows=args.windows,
        max_iterations=args.max_iterations,
        max_llm_failures=args.max_llm_failures,
        indices=args.indices,
        config=cfg,
    )
    print(
        json.dumps(
            {
                "dataset": summary["dataset"],
                "experiment": summary["experiment"],
                "n_ok": summary["n_ok"],
                "n_failed": summary["n_failed"],
                "csv_path": summary["csv_path"],
                "artifacts_dir": summary["artifacts_dir"],
                "ablation": summary["ablation_config"],
            }
        )
    )
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
