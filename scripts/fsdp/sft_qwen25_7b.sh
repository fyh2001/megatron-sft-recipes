#!/usr/bin/env bash
# sft_qwen25_7b.sh
#   Qwen2.5-7B 全参数 SFT，单机 8 卡 A100/H100 80GB。
#   FSDP2 + torch.compile 后端。
#
# 前置：
#   - 已执行 scripts/fsdp/setup_env.sh，容器就绪
#   - 已执行 scripts/02_convert_data.py，sft-data/ 就绪
#
# 在容器内执行：
#   docker exec -it fsdp_sft bash
#   bash scripts/fsdp/sft_qwen25_7b.sh
#
# 环境变量覆盖示例：
#   MBS=1 GAS=8 LR=2e-5 bash scripts/fsdp/sft_qwen25_7b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# ===== 训练参数 =====
# H100 80GB × 8，FSDP FULL_SHARD（7B bf16 = 14GB，切 8 卡 = ~1.75GB/卡 + optimizer ~7GB/卡）。
# 显存非常宽裕，MBS=2 + GAS=4 → GBS_eff = 2*8*4 = 64。
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${MBS:=2}"
: "${GAS:=4}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=2}"
: "${WARMUP:=0.05}"
: "${SAVE_STEPS:=500}"
: "${COMPILE:=true}"
: "${GRAD_CKPT:=true}"
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/fsdp_qwen25_7b}"

GBS_EFF=$(( MBS * NPROC_PER_NODE * GAS ))

log "launching Qwen2.5-7B SFT (FSDP2 + compile)"
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
    output_dir "${OUTPUT_DIR}"

COMPILE_FLAG=""
if [ "${COMPILE}" = "true" ]; then COMPILE_FLAG="--compile"; fi
GRAD_CKPT_FLAG=""
if [ "${GRAD_CKPT}" = "true" ]; then GRAD_CKPT_FLAG="--gradient_checkpointing"; fi

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
    ${COMPILE_FLAG} ${GRAD_CKPT_FLAG}

log "training finished, checkpoints under ${OUTPUT_DIR}"
