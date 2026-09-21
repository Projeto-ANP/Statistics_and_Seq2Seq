#!/bin/bash
# Passo 1 (redesenhado): braços de flag medidos com --reasoning low (contexto do run
# combinado), tendo o iso_reasoning_low como controle deles. O controle sem reasoning
# (iso_control_repeat, órfão PID 3723410) termina as 7 séries por conta própria.
set -u
cd /home/lucas.castro/Statistics_and_Seq2Seq

# Espera o controle (sem reasoning) terminar para não disputar a GPU com ele.
while ps -p 3723410 > /dev/null 2>&1; do sleep 30; done
sleep 5
NROWS=$(grep -c '^' timeseries/mestrado/resultados/orchestrator_react_iso_control_repeat/ETTM2.csv 2>/dev/null || echo 0)
echo "Controle terminou. Linhas no CSV (esperado 8 = header + 7 series): $NROWS"
if [ "$NROWS" -lt 8 ]; then echo "ERRO: controle incompleto, abortando"; exit 1; fi

echo "===== 1/4 iso_reasoning_low (so --reasoning low) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_reasoning_low -- --reasoning low || echo "FALHOU iso_reasoning_low"

echo "===== 2/4 iso_reorder_low (--reasoning low + --reorder-weight-tools) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_reorder_low -- --reasoning low --reorder-weight-tools || echo "FALHOU iso_reorder_low"

echo "===== 3/4 iso_dropcomb_low (--reasoning low + --drop-redundant-combine) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_dropcomb_low -- --reasoning low --drop-redundant-combine || echo "FALHOU iso_dropcomb_low"

echo "===== 4/4 iso_reducedseed_low (--reasoning low + --reduced-seeding) ====="
python3 run_tsf_batch.py --datasets ETTM2 --combinators gpt-oss:20b --version iso_reducedseed_low -- --reasoning low --reduced-seeding || echo "FALHOU iso_reducedseed_low"

echo "===== TODOS OS RUNS LOW CONCLUIDOS ====="
