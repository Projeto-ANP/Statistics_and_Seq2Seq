#!/usr/bin/env bash
# Ablação do CARTE: buscas sem LLM e oráculos ex-post.
#
# Uso no servidor:
#   conda activate agno
#   cd ~/Statistics_and_Seq2Seq
#   mkdir -p logs
#   nohup bash outputs/resultados/scripts/run_ablacao_carte.sh \
#       > logs/ablacao_carte_master.log 2>&1 &
#
# O script grava um log por etapa em logs/ e pode ser reiniciado com o mesmo
# comando: a busca usa --resume e não refaz os pares (série, braço) já
# presentes nos CSVs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO"

mkdir -p logs
LOG_SEARCH="logs/ablacao_search_carte.log"
LOG_ORACLE="logs/ablacao_oraculos_carte.log"
LOG_ORACLE_U="logs/ablacao_oraculo_u_carte.log"
LOG_ORACLE_RESUMO="logs/ablacao_oraculos_resumo.log"
LOG_RESUMO="logs/ablacao_search_resumo.log"

export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] repo: $REPO"
echo "[$(date '+%F %T')] python: $(command -v python3)"
python3 -c "import orchestrator_react, pandas, numpy; print('imports ok')"

MEAN_DIR="timeseries/mestrado/resultados/orchestrator_react_consist_seed11_20_mean"
if [ ! -f "$MEAN_DIR/ANP_MONTHLY.csv" ]; then
    echo "ERRO: pasta da média das dez execuções ausente: $MEAN_DIR" >&2
    exit 1
fi
for seed in 11 12 13 14 15 16 17 18 19 20; do
    run_dir="timeseries/mestrado/resultados/orchestrator_react_consist_seed${seed}"
    if [ ! -f "$run_dir/ANP_MONTHLY.csv" ]; then
        echo "ERRO: execução ausente: $run_dir" >&2
        exit 1
    fi
done
echo "[$(date '+%F %T')] dez execuções e média encontradas"

echo "[$(date '+%F %T')] etapa 1/5: buscas sem LLM (random 7,8,9; greedy; ucb; 12 avaliações)"
python3 outputs/resultados/scripts/ablacao_search_carte.py \
    --methods random,greedy,ucb \
    --random-seeds 7 8 9 \
    --evals 12 \
    --resume \
    2>&1 | tee "$LOG_SEARCH"

echo "[$(date '+%F %T')] etapa 2/5: oráculos das tentativas do agente e das buscas"
python3 outputs/resultados/scripts/ablacao_oraculos_carte.py \
    2>&1 | tee "$LOG_ORACLE"

echo "[$(date '+%F %T')] etapa 3/5: oráculo amplo sobre o universo U"
python3 outputs/resultados/scripts/ablacao_oraculo_u_carte.py \
    --resume \
    2>&1 | tee "$LOG_ORACLE_U"

echo "[$(date '+%F %T')] etapa 4/5: consolidação dos oráculos"
python3 outputs/resultados/scripts/ablacao_oraculos_resumo.py \
    2>&1 | tee "$LOG_ORACLE_RESUMO"

echo "[$(date '+%F %T')] etapa 5/5: resumo das buscas e testes pareados"
python3 outputs/resultados/scripts/ablacao_search_resumo.py \
    2>&1 | tee "$LOG_RESUMO"

echo "[$(date '+%F %T')] concluído"
echo "  buscas:    timeseries/mestrado/resultados/orchestrator_search_carte/"
echo "  oráculos:  outputs/resultados/ablacao_oraculos_carte/"
echo "  oráculo U: outputs/resultados/ablacao_oraculo_u_carte/"
echo "  resumo:    outputs/resultados/ablacao_oraculos_resumo/ e ablacao_search_carte_resumo/"
