#!/usr/bin/env bash
# bench_swift_sp.sh
#   Unified benchmark runner for ms-swift `swift sft` paths under
#   Ulysses sequence-parallelism. Works with two `BACKEND=` modes:
#     ds    — DeepSpeed ZeRO-3 (optionally with optimizer offload)
#     fsdp2 — PyTorch native FSDP2 (accelerate plugin, activation ckpt)
#
#   Both routes share:
#     - swift_sft_patched.py as the entry (applies the swift 4.1.2 ↔
#       transformers 5.5.4 Ulysses mask-signature shim before sft_main)
#     - logging.jsonl for per-step metrics
#     - gpu_monitor.py (nvidia-smi 1Hz) + dcgm_scrape.py for hardware counters
#
# Env overrides (with sensible defaults):
#   MODEL        full path to Qwen3.5-9B (or HF repo id)
#   BACKEND      ds | fsdp2        [required]
#   SP           sequence_parallel_size (int)
#   MBS / GAS    micro-batch, gradient-accumulation-steps (GBS = MBS*NPROC*GAS)
#   MAX_LEN      per-sample max_length
#   TOTAL_STEPS  optimizer steps
#   WARMUP_BENCH optimizer steps ignored by report.py
#   DS_CONFIG    deepspeed json config (only used when BACKEND=ds)
#   RUN_NAME     subdir under ${BENCH_DIR}
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# Ulysses SP + compile does not need Megatron's single-stream CUDA; revert.
export CUDA_DEVICE_MAX_CONNECTIONS=8

: "${MODEL:=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B}"
: "${MODEL_TYPE:=qwen3_5}"
: "${BACKEND:?BACKEND must be set to ds|fsdp2}"
: "${SP:=2}"
: "${MBS:=1}"
: "${GAS:=16}"
: "${MAX_LEN:=16384}"
: "${TOTAL_STEPS:=15}"
: "${WARMUP_BENCH:=5}"
: "${DS_CONFIG:=${SCRIPT_DIR}/sp_offload_configs/zero3_nopin.json}"
: "${FREEZE_VIT:=true}"
# When BACKEND=ds we drive memory via HF's gradient_checkpointing hook.
# When BACKEND=fsdp2 the preset's fsdp_config.activation_checkpointing=true
# already installs per-layer ckpt; passing --gradient_checkpointing would
# double-wrap and swift warns + disables one path, so leave it off.
: "${GRAD_CKPT:=auto}"

: "${BENCH_DIR:=${OUTPUT_ROOT}/bench_sp_offload}"
: "${RUN_NAME:=${BACKEND}_sp${SP}}"
BENCH_OUTPUT="${BENCH_DIR}/${RUN_NAME}"
mkdir -p "${BENCH_OUTPUT}"

GPU_LOG="${BENCH_OUTPUT}/gpu_metrics.jsonl"
DCGM_LOG="${BENCH_OUTPUT}/dcgm_tc.tsv"
TRAIN_LOG="${BENCH_OUTPUT}/train.log"

GBS=$(( MBS * NPROC_PER_NODE * GAS ))

# ===== Derive grad_ckpt default from BACKEND =====
if [ "${GRAD_CKPT}" = "auto" ]; then
    if [ "${BACKEND}" = "fsdp2" ]; then
        GRAD_CKPT=false
    else
        GRAD_CKPT=true
    fi
fi

log "=== swift sft + SP Benchmark ==="
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    backend "${BACKEND}" \
    SP "${SP}" \
    MBS/GAS/GBS "${MBS}/${GAS}/${GBS}" \
    max_length "${MAX_LEN}" \
    total_steps "${TOTAL_STEPS}" \
    warmup_steps "${WARMUP_BENCH}" \
    grad_ckpt "${GRAD_CKPT}" \
    freeze_vit "${FREEZE_VIT}" \
    ds_config "${DS_CONFIG}" \
    output "${BENCH_OUTPUT}"

# ===== Monitors =====
python "${SCRIPT_DIR}/gpu_monitor.py" --output "${GPU_LOG}" &
GPU_MON_PID=$!
# dcgm_scrape.py pins to localhost:9500 by default (our 1Hz exporter).
# Run in the host namespace too (the container shares --net=host).
python "${SCRIPT_DIR}/dcgm_scrape.py" "${DCGM_LOG}" "http://localhost:9500/metrics" &
DCGM_PID=$!
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

# ===== Backend-specific flags =====
BACKEND_FLAGS=()
GRAD_CKPT_FLAGS=()
if [ "${BACKEND}" = "ds" ]; then
    BACKEND_FLAGS+=(--deepspeed "${DS_CONFIG}")
elif [ "${BACKEND}" = "fsdp2" ]; then
    BACKEND_FLAGS+=(--fsdp fsdp2)
else
    log "ERROR: BACKEND must be 'ds' or 'fsdp2', got '${BACKEND}'"
    exit 2
fi

if [ "${GRAD_CKPT}" = "true" ]; then
    GRAD_CKPT_FLAGS+=(--gradient_checkpointing true)
fi

FREEZE_FLAGS=()
if [ "${FREEZE_VIT}" = "true" ]; then
    FREEZE_FLAGS+=(--freeze_vit true --freeze_aligner true)
fi

log "Starting training (${TOTAL_STEPS} steps)..."

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/swift_sft_patched.py" \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${TRAIN_JSONL}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --max_length "${MAX_LEN}" \
    --truncation_strategy delete \
    --per_device_train_batch_size "${MBS}" \
    --gradient_accumulation_steps "${GAS}" \
    --max_steps "${TOTAL_STEPS}" \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    "${GRAD_CKPT_FLAGS[@]}" \
    "${FREEZE_FLAGS[@]}" \
    "${BACKEND_FLAGS[@]}" \
    --sequence_parallel_size "${SP}" \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --save_strategy no \
    --logging_steps 1 \
    --output_dir "${BENCH_OUTPUT}" \
    2>&1 | tee "${TRAIN_LOG}"

kill "${GPU_MON_PID}" "${DCGM_PID}" 2>/dev/null || true
wait "${GPU_MON_PID}" 2>/dev/null || true
wait "${DCGM_PID}" 2>/dev/null || true

# swift writes a timestamped subdir (v0-...) that holds logging.jsonl.
# Pick the newest one (each run creates a fresh dir).
LATEST_VDIR="$(ls -dt "${BENCH_OUTPUT}"/v*-* 2>/dev/null | head -n 1 || true)"
LOGGING_JSONL=""
if [ -n "${LATEST_VDIR}" ]; then
    LOGGING_JSONL="${LATEST_VDIR}/logging.jsonl"
fi

if [ -z "${LOGGING_JSONL}" ] || [ ! -f "${LOGGING_JSONL}" ]; then
    log "WARN: no swift logging.jsonl found; skipping report."
    exit 0
fi

log "Parsing ${LOGGING_JSONL} into bench.jsonl + report.json..."

python "${SCRIPT_DIR}/report_swift_sp.py" \
    --logging_jsonl "${LOGGING_JSONL}" \
    --gpu_log "${GPU_LOG}" \
    --warmup_steps "${WARMUP_BENCH}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --gbs "${GBS}" \
    --max_len "${MAX_LEN}" \
    --backend "${BACKEND}" \
    --sp "${SP}" \
    --bench_jsonl_out "${BENCH_OUTPUT}/bench.jsonl" \
    --report_out "${BENCH_OUTPUT}/report.json"

log "Report saved to ${BENCH_OUTPUT}/report.json"
