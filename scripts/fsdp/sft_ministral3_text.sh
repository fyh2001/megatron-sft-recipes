#!/usr/bin/env bash
# sft_ministral3_text.sh
#   Ministral-3 3B 纯文本 SFT（伪装 Llama），单机 8 卡 H100 80GB。
#   FSDP2 + torch.compile 后端。
#
#   前提：已跑过 scripts/megatron/convert_ministral3_to_llama.py，
#   把 VLM 结构砍成纯文本 LlamaForCausalLM。
#
# 在容器内执行：
#   docker exec -it fsdp_sft bash
#   bash scripts/fsdp/sft_ministral3_text.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# ===== 训练参数 =====
# 纯文本 3B 模型，H100 80GB × 8 非常宽裕。
: "${MODEL:=/home/ubuntu/perf_opt/models/ministral3-3b-text-llama}"
: "${MBS:=4}"
: "${GAS:=2}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=2}"
: "${WARMUP:=0.05}"
: "${SAVE_STEPS:=500}"
: "${COMPILE:=true}"
: "${GRAD_CKPT:=true}"
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/fsdp_ministral3_text}"

GBS_EFF=$(( MBS * NPROC_PER_NODE * GAS ))

log "launching Ministral-3 3B text-only (as Llama) SFT (FSDP2 + compile)"
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
