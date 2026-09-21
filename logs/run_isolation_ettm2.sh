#!/bin/bash
# Passo 1: isolamento das variáveis no ETTM2.
# Passo 0 mostrou que o run publicado (orchestrator_react_v5) usou reasoning=None
# (default do servidor), NÃO --reasoning low. Portanto os controles rodam SEM
# --reasoning, e o braço extra iso_reasoning_low isola a 4a variável que o run
# combinado (v2_gpt_low_nivel) também mudou.
set -u
source /home/lucas.castro/.conda/etc/profile.d/conda.sh
conda activate agno
cd /home/lucas.castro/Statistics_and_Seq2Seq

echo "===== 1/5 iso_control_repeat (nenhuma flag, reproduz config publicada) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_control_repeat || echo "FALHOU iso_control_repeat"

echo "===== 2/5 iso_reorder (so --reorder-weight-tools) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_reorder -- --reorder-weight-tools || echo "FALHOU iso_reorder"

echo "===== 3/5 iso_dropcomb (so --drop-redundant-combine) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_dropcomb -- --drop-redundant-combine || echo "FALHOU iso_dropcomb"

echo "===== 4/5 iso_reducedseed (so --reduced-seeding) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_reducedseed -- --reduced-seeding || echo "FALHOU iso_reducedseed"

echo "===== 5/5 iso_reasoning_low (so --reasoning low; braco extra do Passo 0) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_reasoning_low -- --reasoning low || echo "FALHOU iso_reasoning_low"

echo "===== TODOS OS RUNS CONCLUIDOS ====="
