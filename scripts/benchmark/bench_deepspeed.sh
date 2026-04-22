#!/usr/bin/env bash
# bench_deepspeed.sh
#   DeepSpeed ZeRO-3 backend benchmark, reusing scripts/fsdp/train.py as the
#   training entry. ~90% of the logic is identical to bench_fsdp.sh — only
#   the accelerate config (DS stage-3 instead of FSDP2) and a forced
#   COMPILE=false (DeepSpeed 0.18 + torch.compile is known unstable) differ.
#
# Needs to run inside the `fsdp_sft` container.
#
# 用法：
#   MODEL=... bash scripts/benchmark/bench_deepspeed.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# DeepSpeed also needs multi-stream CUDA for ZeRO-3 allgather + compute
# overlap. Override _common.sh's Megatron-flavoured default.
export CUDA_DEVICE_MAX_CONNECTIONS=8

# ===== Benchmark 配置 =====
: "${MODEL:=mistralai/Mistral-Nemo-Instruct-2407}"
: "${MBS:=1}"
: "${GAS:=1}"
: "${MAX_LEN:=4096}"
: "${TOTAL_STEPS:=50}"
: "${WARMUP_BENCH:=20}"
# DeepSpeed 0.18 × torch.compile is flaky on hybrid models (Gated DeltaNet
# kernels trigger graph breaks; ZeRO-3 param bucket hook conflicts with
# inductor's fx graph). We always force compile off for this backend and
# rely on ZeRO-3 stream prefetch for overlap instead.
COMPILE=false
: "${GRAD_CKPT:=true}"
: "${PAD_TO_MAX:=true}"
: "${ATTN_IMPL:=flash_attention_2}"
: "${FREEZE_VISION:=false}"

: "${BENCH_DIR:=${OUTPUT_ROOT}/benchmark}"
: "${BENCH_OUTPUT:=${BENCH_DIR}/deepspeed}"
mkdir -p "${BENCH_OUTPUT}"

GPU_LOG="${BENCH_OUTPUT}/gpu_metrics.jsonl"
BENCH_LOG="${BENCH_OUTPUT}/bench.jsonl"

GBS_EFF=$(( MBS * NPROC_PER_NODE * GAS ))

# ===== 动态探测模型：num_params（wrap 类对 DS 不需要）=====
if [ -z "${NUM_PARAMS:-}" ]; then
    log "Introspecting model to derive num_params..."
    _INSPECT_OUT="$(python "${SCRIPT_DIR}/_inspect_model.py" "${MODEL}")"
    eval "${_INSPECT_OUT}"
    log "Model info:"
    printf '%s\n' "${_INSPECT_OUT}" | sed 's/^/  /'
fi

# Render run-local accelerate DS config — keeping a copy alongside the
# bench outputs makes the run reproducible without chasing upstream edits.
RENDERED_DS_CFG="${BENCH_OUTPUT}/accelerate_ds_zero3.rendered.yaml"
cp "${SCRIPT_DIR}/../fsdp/accelerate_ds_zero3.yaml" "${RENDERED_DS_CFG}"

log "=== DeepSpeed ZeRO-3 Benchmark ==="
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    num_params "${NUM_PARAMS}" \
    MBS/GAS/GBS "${MBS}/${GAS}/${GBS_EFF}" \
    max_length "${MAX_LEN}" \
    total_steps "${TOTAL_STEPS}" \
    warmup_steps "${WARMUP_BENCH}" \
    compile "${COMPILE} (forced)" \
    grad_ckpt "${GRAD_CKPT}" \
    freeze_vision "${FREEZE_VISION}" \
    output "${BENCH_OUTPUT}"

python "${SCRIPT_DIR}/gpu_monitor.py" --output "${GPU_LOG}" &
GPU_MON_PID=$!
trap 'kill ${GPU_MON_PID} 2>/dev/null || true' EXIT

COMPILE_FLAG="--no_compile"  # forced
GRAD_CKPT_FLAG="--gradient_checkpointing"
if [ "${GRAD_CKPT}" = "false" ]; then GRAD_CKPT_FLAG="--no_gradient_checkpointing"; fi
SYNTHETIC_FLAG=""
if [ "${SYNTHETIC:-false}" = "true" ]; then SYNTHETIC_FLAG="--synthetic"; fi
PAD_FLAG=""
if [ "${PAD_TO_MAX}" = "true" ]; then PAD_FLAG="--pad_to_max"; fi
FREEZE_VISION_FLAG=""
if [ "${FREEZE_VISION}" = "true" ]; then FREEZE_VISION_FLAG="--freeze_vision"; fi

: "${PROFILE:=false}"
: "${PROFILE_START_STEP:=5}"
: "${PROFILE_END_STEP:=9}"
NSYS_BIN="/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys"
LAUNCH_PREFIX=""
PROFILE_FLAG=""
if [ "${PROFILE}" = "true" ]; then
    if [ ! -x "${NSYS_BIN}" ]; then
        log "ERROR: nsys not found at ${NSYS_BIN}; set NSYS_BIN env var." >&2
        exit 2
    fi
    PROFILE_FLAG="--profile --profile_start_step ${PROFILE_START_STEP} --profile_end_step ${PROFILE_END_STEP}"
    LAUNCH_PREFIX="${NSYS_BIN} profile \
        --trace=cuda,nvtx,cublas,cudnn \
        --trace-fork-before-exec=true \
        --cuda-memory-usage=false \
        --sample=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output=${BENCH_OUTPUT}/profile.nsys-rep \
        --force-overwrite=true"
    log "nsys profiling enabled, recording steps ${PROFILE_START_STEP}..${PROFILE_END_STEP}"
fi

log "Starting training (${TOTAL_STEPS} steps)..."

${LAUNCH_PREFIX} accelerate launch \
    --config_file "${RENDERED_DS_CFG}" \
    --num_processes "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/../fsdp/train.py" \
    --model_name_or_path "${MODEL}" \
    --train_file "${TRAIN_JSONL}" \
    --output_dir "${BENCH_OUTPUT}" \
    --per_device_train_batch_size "${MBS}" \
    --gradient_accumulation_steps "${GAS}" \
    --max_length "${MAX_LEN}" \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.1 \
    --num_train_epochs 999 \
    --max_steps "${TOTAL_STEPS}" \
    --logging_steps 1 \
    --benchmark \
    --benchmark_log "${BENCH_LOG}" \
    --warmup_steps_bench "${WARMUP_BENCH}" \
    --attn_implementation "${ATTN_IMPL}" \
    ${COMPILE_FLAG} ${GRAD_CKPT_FLAG} ${SYNTHETIC_FLAG} ${PAD_FLAG} ${FREEZE_VISION_FLAG} ${PROFILE_FLAG} \
    2>&1 | tee "${BENCH_OUTPUT}/train.log"

kill ${GPU_MON_PID} 2>/dev/null || true
wait ${GPU_MON_PID} 2>/dev/null || true

log "Benchmark complete. Generating report..."

python "${SCRIPT_DIR}/report.py" \
    --framework fsdp \
    --bench_log "${BENCH_LOG}" \
    --gpu_log "${GPU_LOG}" \
    --warmup_steps "${WARMUP_BENCH}" \
    --num_params "${NUM_PARAMS}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --output "${BENCH_OUTPUT}/report.json"

log "Report saved to ${BENCH_OUTPUT}/report.json"
