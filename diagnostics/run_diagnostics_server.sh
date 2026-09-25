#!/bin/bash
# Batelada de diagnóstico para rodar NO SERVIDOR (onde está o Ollama).
#
# Saída de cada braço:
#   - resultados  -> ./timeseries/mestrado/resultados/orchestrator_react_<version>/
#   - log do braço -> ./diagnostics/logs/<version>_<dataset>.log
# Traga de volta as pastas inteiras (CSV + llm_artifacts) e rode
# `diagnostics/analyze_run.py` aqui na máquina local (não usa LLM).
#
# Custo estimado: NN5 ~24 min/run (111 séries), ETTM2 ~3 min/run (7 séries).
# Batelada completa: ~2,5 h.
set -u
set -o pipefail
source /home/lucas.castro/.conda/etc/profile.d/conda.sh
conda activate agno
cd /home/lucas.castro/Statistics_and_Seq2Seq
mkdir -p diagnostics/logs

run() {  # run <version> <dataset> <tsf> [args...]
  local v="$1" d="$2" s="$3"; shift 3
  echo "===== $v ($d) ====="
  python3 diagnostics/diag_entry.py --dataset "$d" --source "$s" --version "$v" "$@" \
    2>&1 | tee "diagnostics/logs/${v}_${d}.log" \
    || { echo "FALHOU $v ($d)"; echo "FALHOU $v ($d)" >> diagnostics/logs/_FAILURES.txt; }
}

D=NN5_WEEKLY_DATASET
S=nn5_weekly_dataset.tsf

# ── 0. Âncoras ────────────────────────────────────────────────────────────────
run diag_det_nn5      "$D" "$S" --use-llm 0   # piso determinístico (argmin das sementes)
run diag_control_nn5  "$D" "$S" --seed 7      # reproduz a config publicada (deve dar ~0.1177)

# ── 1. Variância do agente (3 sementes) ──────────────────────────────────────
run diag_seed13_nn5  "$D" "$S" --seed 13
run diag_seed42_nn5  "$D" "$S" --seed 42
run diag_seed99_nn5  "$D" "$S" --seed 99

# ── 2. reasoning (canal harmony do gpt-oss) ───────────────────────────────────
run diag_reasoning_off_nn5 "$D" "$S" --reasoning off
run diag_reasoning_low_nn5 "$D" "$S" --reasoning low

# ── 3. Gate de calibração (pula o loop quando o ranking já é estável) ────────
run diag_calgate_nn5 "$D" "$S" --config '{"calibration_gate": true}'

# ── 4. Atribuição dos levers de contexto ──────────────────────────────────────
run diag_nocard_nn5       "$D" "$S" --config '{"dataset_card": false}'
run diag_noseedpooled_nn5 "$D" "$S" --config '{"seed_pooled_meta_model": false}'

# ── 5. ETTM2 (7 séries, o dataset dos runs iso_*) ─────────────────────────────
D2=ETTM2
S2=ETTM2.tsf
run diag_det_ettm2          "$D2" "$S2" --use-llm 0
run diag_control_ettm2      "$D2" "$S2" --seed 7
run diag_reasoning_off_ettm2 "$D2" "$S2" --reasoning off
run diag_calgate_ettm2      "$D2" "$S2" --config '{"calibration_gate": true}'

echo "===== TODOS OS RUNS DE DIAGNÓSTICO CONCLUÍDOS ====="
