#!/usr/bin/env bash
# sft_ministral3_3b.sh
#   Ministral-3 3B Instruct（VLM）全参数 SFT，冻结 vision tower。
#   FSDP2 + torch.compile 后端。
#
#   模型是多模态 VLM（3.4B Mistral LM + 0.4B Pixtral ViT）。
#   你的数据纯文本 → vision tower 冻结，只训 LM 权重。
#
# 前置：
#   - 已执行 scripts/fsdp/setup_env.sh
#   - 已执行 scripts/02_convert_data.py
#
# 在容器内执行：
#   docker exec -it fsdp_sft bash
#   bash scripts/fsdp/sft_ministral3_3b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# HF 下载比 ModelScope 快（Ministral 在 ModelScope CDN 只有 9MB/s）
export USE_HF=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# ===== 训练参数 =====
# freeze_vit 后 trainable ~3B，H100 80GB × 8 非常宽裕。
# Ministral 的 chat template 在部分框架下不支持 packing → MAX_LEN=2048 减少 padding 浪费。
: "${MODEL:=mistralai/Ministral-3-3B-Instruct-2512-BF16}"
: "${MBS:=4}"
: "${GAS:=2}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${MAX_LEN:=2048}"
: "${EPOCHS:=2}"
: "${WARMUP:=0.05}"
: "${SAVE_STEPS:=500}"
: "${COMPILE:=true}"
: "${GRAD_CKPT:=true}"
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/fsdp_ministral3_3b}"

GBS_EFF=$(( MBS * NPROC_PER_NODE * GAS ))

log "launching Ministral-3 3B VLM SFT (FSDP2 + compile, freeze_vision)"
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    dataset "${TRAIN_JSONL}" \
    val_dataset "${VALID_JSONL}" \
    DP/MBS/GAS "${NPROC_PER_NODE}/${MBS}/${GAS}" \
    GBS_eff "${GBS_EFF}" \
    max_length "${MAX_LEN}" \
    LR "${LR}" \
    epochs "${EPOCHS}" \
    freeze_vision "true" \
    compile "${COMPILE}" \
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
    --freeze_vision \
    ${COMPILE_FLAG} ${GRAD_CKPT_FLAG}

log "training finished, checkpoints under ${OUTPUT_DIR}"
