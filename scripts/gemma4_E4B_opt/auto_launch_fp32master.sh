#!/usr/bin/env bash
# Wait for GPUs to be free, then auto-launch fp32-master experiment:
#   1) 50-step bench validation (verifies dtype config + memory + step1 loss)
#   2) Full 2-epoch run if (1) succeeds and user did not pass SKIP_FULL=1
#
# Use:
#   nohup bash scripts/gemma4_E4B_opt/auto_launch_fp32master.sh \
#     > /tmp/auto_fp32master.log 2>&1 &

set -uo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
SKIP_BENCH="${SKIP_BENCH:-0}"
SKIP_FULL="${SKIP_FULL:-0}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "AUTO-LAUNCH waiting for GPUs to free up..."

# Wait for all GPUs to be free (memory.used < 200 MiB)
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    BUSY=0
    for u in $USED; do
        if [ "$u" -gt 200 ]; then BUSY=$((BUSY+1)); fi
    done
    if [ "$BUSY" -eq 0 ]; then
        log "All 8 GPUs free, proceeding"
        break
    fi
    sleep 60
done

# Stage 1: 50-step bench validation
if [ "${SKIP_BENCH}" != "1" ]; then
    log "================================================="
    log "Stage 1: 50-step fp32-master bench validation"
    log "================================================="
    cd "${REPO}"
    LABEL="bench_fp32master_a3_pf_50step" \
    TEMPLATE="gemma4" \
    WEIGHT_DECAY="0.1" \
    MAX_STEPS=50 \
    FULL_SCHED_STOP=1 \
    TORCH_DTYPE="float32" \
    EXTRA_ENV="GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_KV_SHARE_DETACH=1" \
    EXTRA_ARGS="--bf16 true --fp16 false --padding_free true --max_grad_norm 1.0" \
    bash scripts/gemma4_E4B_opt/bench_variant.sh 2>&1 | tee /tmp/bench_fp32master.log

    BENCH_EXIT=$?
    BENCH_STATUS=$(grep -oE "STATUS = (SUCCESS|FAILED)[^,]*" /tmp/bench_fp32master.log | tail -1)
    log "Bench result: ${BENCH_STATUS:-(no status line)}, exit=${BENCH_EXIT}"

    if [ "${BENCH_EXIT}" != "0" ]; then
        log "Bench FAILED, aborting before full run"
        exit 1
    fi

    log "Bench succeeded — first-step loss should be ~2.226 if numerical setup is correct"
    grep -E "step 1\s|loss = 2\.|loss = 1\.|peak mem" /tmp/bench_fp32master.log | tail -10 | while IFS= read -r line; do log "  $line"; done
fi

# Stage 2: Full 2-epoch run
if [ "${SKIP_FULL}" != "1" ]; then
    log "================================================="
    log "Stage 2: full 2-epoch fp32-master run"
    log "================================================="
    cd "${REPO}"
    bash scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh 2>&1 | tee /tmp/fp32master_full.log
    log "Full run completed"
fi

log "AUTO-LAUNCH all done"
