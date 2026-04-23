#!/usr/bin/env bash
# bench_swift_sp_profile.sh
#   Short 5-step run of `swift sft` with torch.profiler hooked on one step,
#   to measure real per-kernel time distribution (GEMM vs FlashAttn vs NCCL vs
#   Triton/GDN vs Elementwise) on Qwen3.5-9B + FSDP2 + Ulysses SP=2.
#
# Env overrides:
#   BACKEND=fsdp2|ds       default fsdp2
#   SP=2|4                 default 2
#   TOTAL_STEPS            default 5 (profile adds ~1s overhead; keep short)
#   PROFILE_STEP           default 3 (0-indexed; skip compile cold start)
#   RUN_NAME               default profile_${BACKEND}_sp${SP}
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

export CUDA_DEVICE_MAX_CONNECTIONS=8

: "${MODEL:=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B}"
: "${MODEL_TYPE:=qwen3_5}"
: "${BACKEND:=fsdp2}"
: "${SP:=2}"
: "${MBS:=1}"
: "${GAS:=1}"
: "${MAX_LEN:=16384}"
: "${TOTAL_STEPS:=5}"
: "${PROFILE_STEP:=3}"
: "${DS_CONFIG:=${SCRIPT_DIR}/sp_offload_configs/zero3_nopin.json}"
: "${FREEZE_VIT:=true}"
: "${GRAD_CKPT:=auto}"

# Same FSDP / compile knobs as bench_swift_sp_v2.sh
: "${TORCH_COMPILE:=false}"
: "${NO_AC:=false}"
: "${FSDP_RESHARD:=true}"
: "${FSDP_WRAP_POLICY:=}"
: "${FSDP_MIN_NUM_PARAMS:=}"

: "${BENCH_DIR:=${OUTPUT_ROOT}/bench_sp_offload_profile}"
: "${RUN_NAME:=profile_${BACKEND}_sp${SP}}"
BENCH_OUTPUT="${BENCH_DIR}/${RUN_NAME}"
mkdir -p "${BENCH_OUTPUT}"

GBS=$(( MBS * NPROC_PER_NODE * GAS ))

if [ "${GRAD_CKPT}" = "auto" ]; then
    if [ "${BACKEND}" = "fsdp2" ]; then GRAD_CKPT=false; else GRAD_CKPT=true; fi
fi

log "=== swift sft torch.profiler profile run ==="
printf '  %-22s = %s\n' \
    swift_version "$(docker exec fsdp_sft python -c 'import swift;print(swift.__version__)' 2>/dev/null || echo '?')" \
    backend "${BACKEND}" \
    SP "${SP}" \
    max_length "${MAX_LEN}" \
    total_steps "${TOTAL_STEPS}" \
    profile_step "${PROFILE_STEP}" \
    output "${BENCH_OUTPUT}"

BACKEND_FLAGS=()
if [ "${BACKEND}" = "ds" ]; then
    BACKEND_FLAGS+=(--deepspeed "${DS_CONFIG}")
elif [ "${BACKEND}" = "fsdp2" ]; then
    NEED_OVERRIDE=false
    [ "${NO_AC}" = "true" ] && NEED_OVERRIDE=true
    [ "${FSDP_RESHARD}" = "false" ] && NEED_OVERRIDE=true
    [ -n "${FSDP_WRAP_POLICY}" ] && NEED_OVERRIDE=true
    if [ "${NEED_OVERRIDE}" = "true" ]; then
        FSDP_OVERRIDE="${BENCH_OUTPUT}/fsdp_override.json"
        AC_VAL=true; [ "${NO_AC}" = "true" ] && AC_VAL=false
        RESHARD_VAL=true; [ "${FSDP_RESHARD}" = "false" ] && RESHARD_VAL=false
        WRAP_VAL="${FSDP_WRAP_POLICY:-TRANSFORMER_BASED_WRAP}"
        EXTRA_WRAP=""
        if [ "${WRAP_VAL}" = "SIZE_BASED_WRAP" ] && [ -n "${FSDP_MIN_NUM_PARAMS}" ]; then
            EXTRA_WRAP=", \"min_num_params\": ${FSDP_MIN_NUM_PARAMS}"
        fi
        cat > "${FSDP_OVERRIDE}" <<EOF
{
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": ${RESHARD_VAL},
        "auto_wrap_policy": "${WRAP_VAL}",
        "cpu_ram_efficient_loading": true,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": ${AC_VAL}${EXTRA_WRAP}
    }
}
EOF
        BACKEND_FLAGS+=(--fsdp "${FSDP_OVERRIDE}")
        log "FSDP override: ${FSDP_OVERRIDE}"
    else
        BACKEND_FLAGS+=(--fsdp fsdp2)
    fi
fi

GRAD_CKPT_FLAGS=()
if [ "${GRAD_CKPT}" = "true" ]; then
    GRAD_CKPT_FLAGS+=(--gradient_checkpointing true)
fi

COMPILE_FLAGS=()
if [ "${TORCH_COMPILE}" = "true" ]; then
    COMPILE_FLAGS+=(--torch_compile true)
fi

FREEZE_FLAGS=()
if [ "${FREEZE_VIT}" = "true" ]; then
    FREEZE_FLAGS+=(--freeze_vit true --freeze_aligner true)
fi

TRAIN_LOG="${BENCH_OUTPUT}/train.log"

log "Starting training (profiling step ${PROFILE_STEP}, rank 0 only)..."

: "${MASTER_PORT:=29533}"
TORCH_PROFILE_STEP="${PROFILE_STEP}" \
TORCH_PROFILE_OUT="${BENCH_OUTPUT}" \
TORCH_PROFILE_RANK=0 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" "${SCRIPT_DIR}/swift_sft_profiled.py" \
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
    "${COMPILE_FLAGS[@]}" \
    "${FREEZE_FLAGS[@]}" \
    "${BACKEND_FLAGS[@]}" \
    --sequence_parallel_size "${SP}" \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --save_strategy no \
    --logging_steps 1 \
    --output_dir "${BENCH_OUTPUT}" \
    2>&1 | tee "${TRAIN_LOG}"

log "Analyzing trace..."
TRACE="$(ls "${BENCH_OUTPUT}"/trace_rank0_step*.json 2>/dev/null | head -1 || true)"
if [ -z "${TRACE}" ]; then
    log "WARN: no trace file found in ${BENCH_OUTPUT}"
    exit 0
fi

python "${SCRIPT_DIR}/analyze_torch_profile.py" "${TRACE}" \
    | tee "${BENCH_OUTPUT}/analysis.txt"

log "Done. See ${BENCH_OUTPUT}/analysis.txt"
