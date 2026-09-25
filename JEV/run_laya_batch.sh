#!/bin/bash
# Batelada do agente LAYA (classificador System One) — estilo do run_tsf_batch.py.
#
# Rode NO SERVIDOR (é onde estão os dados e o env agno):
#
#   nohup bash JEV/run_laya_batch.sh > logs/laya_v0_batch.log 2>&1 &
#
# Pré-requisito no servidor (uma vez):
#   pip install laya
#
# Cada dataset gera:
#   resultados/ -> ./timeseries/mestrado/resultados/orchestrator_laya_<version>/<DATASET>.csv
#   log         -> ./logs/<version>_<dataset>.log  (DATASET SUMMARY com this run + baseline + externos)
set -u
set -o pipefail
source /home/lucas.castro/.conda/etc/profile.d/conda.sh
conda activate agno
cd /home/lucas.castro/Statistics_and_Seq2Seq
mkdir -p logs

VERSION="${VERSION:-laya_v0}"
CHECKPOINT="${CHECKPOINT:-english}"
DATASETS="${DATASETS:-ETTM2 NN5_WEEKLY_DATASET}"
MAX_ITERATIONS="${MAX_ITERATIONS:-12}"

for d in $DATASETS; do
  echo "===== $d ($CHECKPOINT) -> version $VERSION ====="
  python3 JEV/run_laya.py \
    --datasets "$d" \
    --version "$VERSION" \
    --checkpoint "$CHECKPOINT" \
    --max-iterations "$MAX_ITERATIONS" \
    2>&1 | tee "logs/${VERSION}_${d}.log" \
    || { echo "FALHOU $d ($CHECKPOINT)"; echo "FALHOU $d ($CHECKPOINT)" >> logs/_laya_failures.txt; }
done

echo "===== BATELADA LAYA CONCLUÍDA ====="
