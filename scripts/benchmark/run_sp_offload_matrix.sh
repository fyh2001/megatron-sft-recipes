#!/usr/bin/env bash
# run_sp_offload_matrix.sh
#   Run the Qwen3.5-9B SP/Ulysses/offload × 3-backend benchmark matrix.
#
#   Each group runs TOTAL_STEPS=5 WARMUP_BENCH=1 at MAX_LEN=16384 MBS=1 GAS=1.
#   Short by design: swift 4.1.2 × transformers 5.5.4 under Ulysses SP trips
#   a dataset-ordering bug around steps 3–5 that propagates NaN grads and
#   eventually corrupts the CUDA context after ~15 forwards. At 5 steps we
#   still collect 4 valid peak-mem + step-time samples before that window
#   closes — enough for a "can-it-run" + memory-cost characterisation, which
#   is the explicit goal of this benchmark round.
#
#   Output layout: ${BENCH_DIR}/<group_name>/{bench.jsonl, report.json,
#   train.log, gpu_metrics.jsonl, dcgm_tc.tsv, v0-*/logging.jsonl}
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${BENCH_DIR:=/home/ubuntu/perf_opt/megatron_output/bench_sp_offload}"
mkdir -p "${BENCH_DIR}"

: "${TOTAL_STEPS:=5}"
: "${WARMUP_BENCH:=1}"
: "${MAX_LEN:=16384}"
: "${MBS:=1}"
: "${GAS:=1}"

MODEL=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=8

log() { printf '\n[%(%F %T)T] %s\n' -1 "$*" >&2; }

run_swift_group() {
    local name="$1"; shift
    log "=== Group: ${name} ==="
    env \
        BENCH_DIR="${BENCH_DIR}" \
        RUN_NAME="${name}" \
        TOTAL_STEPS="${TOTAL_STEPS}" \
        WARMUP_BENCH="${WARMUP_BENCH}" \
        MAX_LEN="${MAX_LEN}" \
        MBS="${MBS}" \
        GAS="${GAS}" \
        MODEL="${MODEL}" \
        "$@" \
        bash "${SCRIPT_DIR}/bench_swift_sp.sh" \
        || log "!! Group ${name} failed, continuing"
}

run_megatron_group() {
    local name="$1"; shift
    log "=== Group: ${name} (Megatron) ==="
    local OUT="${BENCH_DIR}/${name}"
    mkdir -p "${OUT}"
    env \
        BENCH_DIR="${BENCH_DIR}" \
        USE_MEGATRON_BACKEND=true \
        MODEL="${MODEL}" \
        MBS="${MBS}" \
        GBS=$(( MBS * 8 * GAS )) \
        MAX_LEN="${MAX_LEN}" \
        TOTAL_STEPS="${TOTAL_STEPS}" \
        WARMUP_BENCH="${WARMUP_BENCH}" \
        FREEZE_VIT=true \
        "$@" \
        bash "${SCRIPT_DIR}/bench_megatron.sh" \
        || log "!! Group ${name} failed, continuing"
    # bench_megatron.sh writes into ${BENCH_DIR}/megatron — rename so each
    # megatron group has its own subdir matching ${name}.
    if [ -d "${BENCH_DIR}/megatron" ] && [ "${BENCH_DIR}/megatron" != "${OUT}" ]; then
        mv "${BENCH_DIR}/megatron" "${OUT}.raw" 2>/dev/null || true
        # keep both; ${OUT} stays empty if bench_megatron.sh errored
    fi
}

# ===== Swift-based groups (DS / FSDP2) =====
# DS ZeRO-3 + SP=2 (no offload)
run_swift_group ds_sp2_no_off \
    BACKEND=ds SP=2 \
    DS_CONFIG="${SCRIPT_DIR}/sp_offload_configs/zero3_nopin.json"

# DS ZeRO-3 + SP=2 + optimizer CPU offload
run_swift_group ds_sp2_off_opt \
    BACKEND=ds SP=2 \
    DS_CONFIG="${SCRIPT_DIR}/sp_offload_configs/zero3_offload_opt.json"

# DS ZeRO-3 + SP=4 (no offload). DP shrinks to 2.
run_swift_group ds_sp4_no_off \
    BACKEND=ds SP=4 \
    DS_CONFIG="${SCRIPT_DIR}/sp_offload_configs/zero3_nopin.json"

# FSDP2 + SP=2
run_swift_group fsdp2_sp2 \
    BACKEND=fsdp2 SP=2

# FSDP2 + SP=4
run_swift_group fsdp2_sp4 \
    BACKEND=fsdp2 SP=4

# ===== Megatron group(s) =====
# Megatron TP=4 SP RECOMPUTE=none (the task's validated "best" config)
run_megatron_group megatron_tp4_sp \
    TP=4 PP=1 CP=1 RECOMPUTE=none

# Megatron TP=2 SP RECOMPUTE=selective (comparator)
run_megatron_group megatron_tp2_sp_sel \
    TP=2 PP=1 CP=1 RECOMPUTE=selective

log "=== Matrix complete; outputs in ${BENCH_DIR} ==="
ls -la "${BENCH_DIR}"
