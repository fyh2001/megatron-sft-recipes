#!/usr/bin/env bash
# sft_gemma4_31b.sh
#   Gemma 4 31B Dense SFT，单机 8×80GB H100 极限配置。
#   FSDP2 + torch.compile 后端。
#
#   31B bf16 = 62GB 模型。FSDP FULL_SHARD 切 8 卡：
#     模型 ~7.75GB/卡 + optimizer states ~31GB/卡 = ~39GB static + activations。
#     需要 gradient checkpointing + CPU offload optimizer 才安全。
#
# 如果 OOM，依次考虑：
#   1. MAX_LEN 4096 -> 2048
#   2. MBS 1 已经是最小
#   3. 加 --cpu_offload（默认已开）
#   4. 改 LoRA（需扩展 train.py）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${MODEL:=google/gemma-4-31b-it}"
: "${MBS:=1}"
: "${GAS:=8}"
: "${LR:=5e-6}"
: "${MIN_LR:=5e-7}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=1}"
: "${WARMUP:=0.05}"
: "${SAVE_STEPS:=500}"
: "${COMPILE:=true}"
: "${GRAD_CKPT:=true}"
: "${CPU_OFFLOAD:=true}"
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/fsdp_gemma4_31b}"

GBS_EFF=$(( MBS * NPROC_PER_NODE * GAS ))

log "launching Gemma 4 31B Dense SFT (FSDP2 + compile)"
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    dataset "${TRAIN_JSONL}" \
    val_dataset "${VALID_JSONL}" \
    DP/MBS/GAS "${NPROC_PER_NODE}/${MBS}/${GAS}" \
    GBS_eff "${GBS_EFF}" \
    max_length "${MAX_LEN}" \
    LR "${LR}" \
    epochs "${EPOCHS}" \
    compile "${COMPILE}" \
    grad_ckpt "${GRAD_CKPT}" \
    cpu_offload "${CPU_OFFLOAD}" \
    output_dir "${OUTPUT_DIR}"

COMPILE_FLAG=""
if [ "${COMPILE}" = "true" ]; then COMPILE_FLAG="--compile"; fi
GRAD_CKPT_FLAG=""
if [ "${GRAD_CKPT}" = "true" ]; then GRAD_CKPT_FLAG="--gradient_checkpointing"; fi
CPU_OFFLOAD_FLAG=""
if [ "${CPU_OFFLOAD}" = "true" ]; then CPU_OFFLOAD_FLAG="--cpu_offload"; fi

accelerate launch \
    --config_file "${SCRIPT_DIR}/accelerate_config.yaml" \
    --num_processes "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train.py" \
    --model_name_or_path "${MODEL}" \
    --train_file "${TRAIN_JSONL}" \
    --valid_file "${VALID_JSONL}" \
    --per_device_train_batch_size "${MBS}" \
    --gradient_accumulation_steps "${GAS}" \
    --max_length "${MAX_LEN}" \
    --learning_rate "${LR}" \
    --min_lr "${MIN_LR}" \
    --warmup_ratio "${WARMUP}" \
    --num_train_epochs "${EPOCHS}" \
    --save_steps "${SAVE_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${COMPILE_FLAG} ${GRAD_CKPT_FLAG} ${CPU_OFFLOAD_FLAG}

log "training finished, checkpoints under ${OUTPUT_DIR}"
