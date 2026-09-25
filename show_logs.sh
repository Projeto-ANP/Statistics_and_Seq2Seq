#!/bin/bash
# Mostra o estado de TODOS os logs das bateladas (diagnóstico + LAYA).
# Uso no servidor:
#   bash show_logs.sh              # visão geral: processos + resumo de cada log
#   bash show_logs.sh -f <arquivo> # segue um log ao vivo (tail -f)
#   bash show_logs.sh -a           # imprime os DATASET SUMMARY de todos os logs
set -u

HEAD() { printf "\n===== %s =====\n" "$1"; }

if [ "${1:-}" = "-f" ] && [ -n "${2:-}" ]; then
  exec tail -f "$2"
fi

# ── 0. o que está rodando agora ──────────────────────────────────────────────
HEAD "PROCESSOS ATIVOS"
ps aux | grep -E "diag_entry|run_diagnostics|run_laya|run_tsf_batch" | grep -v grep \
  || echo "  (nenhum processo de batelada rodando)"

# ── 1. status de cada log ────────────────────────────────────────────────────
HEAD "LOGS (última linha de cada um)"
FOUND=0
for f in diagnostics/logs/*.log logs/laya_v0_*.log logs/v2_gpt_batch_low.log logs/isolation_batch.log; do
  [ -f "$f" ] || continue
  FOUND=1
  last=$(tail -1 "$f" 2>/dev/null | cut -c1-90)
  size=$(du -h "$f" 2>/dev/null | cut -f1)
  printf "  %-58s %6s  %s\n" "$f" "$size" "$last"
done
[ "$FOUND" = 0 ] && echo "  (nenhum .log encontrado em diagnostics/logs/ nem logs/laya_v0_*)"

# ── 2. falhas registradas ────────────────────────────────────────────────────
HEAD "FALHAS"
if [ -s diagnostics/logs/_FAILURES.txt ] || [ -s logs/_laya_failures.txt ]; then
  cat diagnostics/logs/_FAILURES.txt logs/_laya_failures.txt 2>/dev/null
else
  echo "  (nenhuma falha registrada)"
fi

# ── 3. DATASET SUMMARY de cada log que terminou um dataset ───────────────────
if [ "${1:-}" = "-a" ]; then
  HEAD "DATASET SUMMARIES"
  for f in diagnostics/logs/*.log logs/laya_v0_*.log; do
    [ -f "$f" ] || continue
    if grep -q "DATASET SUMMARY" "$f"; then
      HEAD "$f"
      awk '/DATASET SUMMARY/{p=1} p&&/csv:/{print; exit} p{print}' "$f" | head -12
    fi
  done
else
  echo ""
  echo "  para os DATASET SUMMARY completos de todos os logs: bash show_logs.sh -a"
fi
