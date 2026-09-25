#!/bin/bash
# Fine-tune do GATE no servidor (GPU 5090) — Tasks 1-4 do PLANO_FINETUNE.md.
#
# Rode UMA vez, depois de sincronizar os arquivos novos de JEV/:
#
#   nohup bash JEV/run_finetune_server.sh > JEV/logs/finetune_batch.log 2>&1 &
#
# O que ele faz, em sequência:
#   1. build do dataset completo (7 datasets, todas as fontes) -> JEV/data/gate_dataset.jsonl
#   2. baseline de regressão logística LOO (instantâneo)
#   3. fine-tune do LAYA nos dois holdouts principais (ETTM2, NN5) -> JEV/models/laya_gate_*/
#   4. avaliação do gate em cada holdout (AUC + sweep de limiar vs piso)
#
# Tempo estimado: ~5 min (dataset) + segundos (baseline) + ~10-20 min por
# holdout na 5090. Logs por etapa em JEV/logs/.
set -u
set -o pipefail
source /home/lucas.castro/.conda/etc/profile.d/conda.sh
conda activate agno
cd /home/lucas.castro/Statistics_and_Seq2Seq
mkdir -p JEV/logs JEV/data JEV/models

echo "===== 1/4 dataset de decisões rotuladas ====="
python3 JEV/build_finetune_dataset.py 2>&1 | tee JEV/logs/build_dataset.log \
  || { echo "FALHOU build do dataset"; exit 1; }

echo "===== 2/4 baseline logístico (LOO) ====="
python3 JEV/run_gate_baseline.py 2>&1 | tee JEV/logs/gate_baseline.log

echo "===== 3-4/4 fine-tune + avaliação (ETTM2, NN5) ====="
for H in ETTM2 NN5_WEEKLY_DATASET; do
  SLUG=$(echo "$H" | tr '[:upper:]' '[:lower:]')
  python3 JEV/finetune_laya_gate.py --holdout "$H" 2>&1 | tee "JEV/logs/finetune_${SLUG}.log" \
    || { echo "FALHOU fine-tune $H"; continue; }
  python3 JEV/eval_gate.py --checkpoint "JEV/models/laya_gate_${SLUG}" \
      --holdout "$H" 2>&1 | tee "JEV/logs/eval_${SLUG}.log"
done

echo "===== FINE-TUNE CONCLUÍDO ====="
