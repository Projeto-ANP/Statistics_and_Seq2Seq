#!/bin/bash
# Fine-tune do GATE no servidor (GPU 5090) — v3: treino SÓ com janelas de validação.
#
# O teste fica intocado até a avaliação final: os rótulos de treino
# (label_val / label_val_seed) vêm do score ANINHADO das 3 janelas de validação.
#
#   nohup bash JEV/run_finetune_server.sh > JEV/logs/finetune_batch.log 2>&1 &
#
# Etapas:
#   1. dataset completo (7 datasets; rótulos de validação p/ treino + de teste
#      apenas para análise posterior)
#   2. baseline logístico LOO nos alvos de validação (instantâneo)
#   3-4. para cada holdout (ETTM2, NN5): fine-tune + avaliação com
#        label_val (ranqueador dinâmico) e label_val_seed (gate vs sementes)
set -u
set -o pipefail
source /home/lucas.castro/.conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate agno 2>/dev/null || true
cd /home/lucas.castro/Statistics_and_Seq2Seq
mkdir -p JEV/logs JEV/data JEV/models

echo "===== 1/4 dataset de decisões (rótulos de validação + análise) ====="
python3 JEV/build_finetune_dataset.py 2>&1 | tee JEV/logs/build_dataset.log \
  || { echo "FALHOU build do dataset"; exit 1; }

echo "===== 2/4 baseline logístico (LOO, alvos de validação) ====="
for T in label_val label_val_seed; do
  echo "----- alvo: $T -----"
  python3 JEV/run_gate_baseline.py --target "$T" 2>&1 | tee "JEV/logs/gate_baseline_${T}.log"
done

echo "===== 3-4/4 fine-tune + avaliação (2 holdouts x 2 alvos) ====="
for H in ETTM2 NN5_WEEKLY_DATASET; do
  SLUG=$(echo "$H" | tr '[:upper:]' '[:lower:]')
  for T in label_val label_val_seed; do
    TAG=$([ "$T" = "label_val" ] && echo "val" || echo "valseed")
    echo "===== holdout $H / alvo $T ====="
    python3 JEV/finetune_laya_gate.py --holdout "$H" --target "$T" \
      2>&1 | tee "JEV/logs/finetune_${SLUG}_${TAG}.log" \
      || { echo "FALHOU fine-tune $H/$T"; continue; }
    python3 JEV/eval_gate.py --checkpoint "JEV/models/laya_gate_${SLUG}_${TAG}" \
      --holdout "$H" --target "$T" 2>&1 | tee "JEV/logs/eval_${SLUG}_${TAG}.log"
  done
done

echo "===== FINE-TUNE CONCLUÍDO ====="
