#!/bin/bash
# N=5 por condição na série 5 do ETTM2 (sem nenhuma flag), + 2 âncoras de dataset inteiro.
# None = sem --reasoning (default do servidor, como o publicado); low = --reasoning low.
set -u
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate agno 2>/dev/null || true
cd /home/lucas.castro/Statistics_and_Seq2Seq

echo "===== CORE: 5x série 5 reasoning=None ====="
for i in 01 02 03 04 05; do
  echo "----- iso_s5_none_$i -----"
  python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_s5_none_$i -- --indices 5 || echo "FALHOU iso_s5_none_$i"
done

echo "===== CORE: 5x série 5 reasoning=low ====="
for i in 01 02 03 04 05; do
  echo "----- iso_s5_low_$i -----"
  python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_s5_low_$i -- --indices 5 --reasoning low || echo "FALHOU iso_s5_low_$i"
done

echo "===== ANCORA: dataset inteiro reasoning=None ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_s5_anchor_none || echo "FALHOU iso_s5_anchor_none"

echo "===== ANCORA: dataset inteiro reasoning=low ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_s5_anchor_low -- --reasoning low || echo "FALHOU iso_s5_anchor_low"

echo "===== TODOS OS RUNS DA SERIE 5 CONCLUIDOS ====="
