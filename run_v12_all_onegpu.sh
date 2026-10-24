#!/usr/bin/env bash
# v12 — seasonal guard + top-k per horizon (direct-preparation), gap 0.10.
#
# Two queues, one per GPU:
#   GPU 0: gpt-oss low  -> gpt-oss medium
#   GPU 1: qwen medium  -> gemma on
# Within each queue the models run sequentially; the two queues can run in
# parallel. Both Ollama servers must be on distinct GPUs and on ports 11434/11435.
#
# Usage:
#   nohup bash run_v12_all.sh parallel   > logs/v12_all.log 2>&1 &
#   nohup bash run_v12_all.sh sequential > logs/v12_all.log 2>&1 &
#
# `parallel` (default) runs queue 0 and queue 1 at the same time.
# `sequential` runs queue 0 and then queue 1, still using the assigned GPU.
#
# The model tags below are the repo defaults; confirm with `ollama list` and edit
# if your server uses different tags.

set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs

GPTOSS="gpt-oss:20b"
QWEN="qwen3.8:27b"
GEMMA="gemma4:26b"

DATASETS=(
  ANP_MONTHLY
  ETTH1
  ETTH2
  ETTM1
  ETTM2
  NN5_WEEKLY_DATASET
  M4_WEEKLY_DATASET
)

V12_FLAGS=(
  --llm-seed 11
  --prompt-v2
  --catalog-v2
  --no-early-stop
  --scoring-budget
  --direct-preparation
  --max-iterations 12
  --max-total-turns 24
  --temperature 0.7
  --score-preset smape_only
  --score-window-mode weighted
  --no-agent-score-preset
  --no-seed-attempts
  --no-best-single
  --final-strategy seasonal_guard
  --final-selector-max-candidates 10
  --seasonal-guard-min-gap 0.10
)

CHECK_GPU=${CHECK_GPU:-1}

log() { echo "[$(date '+%F %T')] $*"; }

check_server() {
  local url="$1"
  if ! curl -fsS --max-time 10 "${url}/api/tags" >/dev/null; then
    log "ERROR: Ollama not reachable at ${url}"
    return 1
  fi
  return 0
}

check_model() {
  local url="$1" model="$2"
  if ! curl -fsS --max-time 10 "${url}/api/tags" | grep -q "\"name\":\"${model}\""; then
    log "WARNING: ${model} not found at ${url} (check: ollama list / pull)"
  fi
}

preflight_gpu() {
  log "GPU compute apps:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid \
    --format=csv,noheader || true
  log "GPU map (index,uuid):"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader || true
  log "If you do not see one llama-server on each GPU, stop before running."
  log "The client --gpu N only chooses the URL; the Ollama server must be started"
  log "with CUDA_VISIBLE_DEVICES=N (or in a container with --gpus device=N)."
}

run_model() {
  local gpu="$1" model="$2" reasoning="$3" version="$4"
  local logfile="logs/${version}_batch.log"
  log "START gpu=${gpu} model=${model} reasoning=${reasoning} version=${version}"
  python3 run_tsf_batch.py \
    --datasets "${DATASETS[@]}" \
    --combinators "${model}" \
    --gpu "${gpu}" \
    --version "${version}" \
    -- "${V12_FLAGS[@]}" --reasoning "${reasoning}" \
    > "${logfile}" 2>&1
  local rc=$?
  log "END   gpu=${gpu} model=${model} version=${version} rc=${rc} log=${logfile}"
  return "${rc}"
}

queue_gpu0() {
  run_model 0 "${GPTOSS}" low    v12_gptoss_low
  run_model 0 "${GPTOSS}" medium v12_gptoss_medium
}

queue_gpu1() {
  run_model 0 "${QWEN}"  medium v12_qwen_medium
  run_model 0 "${GEMMA}" on     v12_gemma_on
}

log "v12 runner: GPTOSS=${GPTOSS} QWEN=${QWEN} GEMMA=${GEMMA}"
log "datasets: ${DATASETS[*]}"

check_server "http://127.0.0.1:11434" || exit 1
check_server "http://127.0.0.1:11435" || exit 1
check_model "http://127.0.0.1:11434" "${GPTOSS}"
check_model "http://127.0.0.1:11434" "${QWEN}"
check_model "http://127.0.0.1:11434" "${GEMMA}"
check_model "http://127.0.0.1:11435" "${GPTOSS}"
check_model "http://127.0.0.1:11435" "${QWEN}"
check_model "http://127.0.0.1:11435" "${GEMMA}"

if [ "${CHECK_GPU}" = "1" ]; then
  preflight_gpu
fi

MODE="${1:-parallel}"
if [ "${MODE}" = "sequential" ]; then
  queue_gpu0
  rc0=$?
  queue_gpu1
  rc1=$?
else
  queue_gpu0 & p0=$!
  queue_gpu1 & p1=$!
  wait "${p0}"; rc0=$?
  wait "${p1}"; rc1=$?
fi

log "done: queue_gpu0 rc=${rc0} queue_gpu1 rc=${rc1}"
exit $(( rc0 != 0 || rc1 != 0 ))
