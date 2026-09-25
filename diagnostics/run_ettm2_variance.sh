#!/bin/bash
# Variância de amostragem no ETTM2: default vs --reasoning low, 5 sementes cada.
#
# Rode DEPOIS da batelada principal (competir por GPU com ela só embaralha os
# tempos). Cada run ~3 min; total ~30 min.
#
# Por que existe: um único run de ETTM2 (7 séries) não separa "efeito da flag"
# de "sorte de amostragem" — a série 5 decide a média sozinha. Com 5 sementes
# por configuração comparamos DISTRIBUIÇÕES, não runs únicos.
set -u
set -o pipefail
source /home/lucas.castro/.conda/etc/profile.d/conda.sh
conda activate agno
cd /home/lucas.castro/Statistics_and_Seq2Seq
mkdir -p diagnostics/logs

D=ETTM2
S=ETTM2.tsf

run() {  # run <version> <seed> [extra args...]
  local v="$1" seed="$2"; shift 2
  echo "===== $v (seed=$seed) ====="
  python3 diagnostics/diag_entry.py --dataset "$D" --source "$S" --version "$v" \
    --seed "$seed" "$@" 2>&1 | tee "diagnostics/logs/${v}.log" \
    || { echo "FALHOU $v"; echo "FALHOU $v" >> diagnostics/logs/_FAILURES.txt; }
}

# 5 sementes × default do servidor
run diagv_default_seed7 7
run diagv_default_seed13 13
run diagv_default_seed42 42
run diagv_default_seed99 99
run diagv_default_seed123 123

# 5 sementes × reasoning low
run diagv_low_seed7 7 --reasoning low
run diagv_low_seed13 13 --reasoning low
run diagv_low_seed42 42 --reasoning low
run diagv_low_seed99 99 --reasoning low
run diagv_low_seed123 123 --reasoning low

echo "===== VARIÂNCIA ETTM2 CONCLUÍDA ====="
echo "Comparação local (sem LLM):"
for v in diagv_default_seed7 diagv_default_seed13 diagv_default_seed42 diagv_default_seed99 diagv_default_seed123 \
         diagv_low_seed7 diagv_low_seed13 diagv_low_seed42 diagv_low_seed99 diagv_low_seed123; do
  echo "python diagnostics/analyze_run.py --run orchestrator_react_$v --dataset ETTM2 --tsf ETTM2.tsf --det orchestrator_react_diag_det_ettm2"
done
