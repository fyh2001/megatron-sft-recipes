#!/usr/bin/env bash
# 05_sft_gemma4_31b.sh
#   Gemma 4 31B Dense SFT，单机 8×80GB H100 极限配置。
#   需要：
#     - TP=4 PP=2（4-way tensor + 2-way pipeline）
#     - sequence_parallel
#     - full recompute（activation checkpoint）
#     - micro_batch_size=1
#   如果 OOM，依次考虑：
#     1. max_length 4096 -> 2048
#     2. GBS 16 -> 8
#     3. 开 --optimizer_cpu_offload true
#     4. 改 LoRA：--train_type lora --lora_rank 16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/../_common.sh"

: "${MODEL:=LLM-Research/gemma-4-31b-it}"
: "${TP:=4}"
: "${PP:=2}"
: "${MBS:=1}"
: "${GBS:=16}"
: "${LR:=5e-6}"
: "${MIN_LR:=5e-7}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=1}"
: "${SAVE_STEPS:=500}"
: "${RECOMPUTE:=full}"
: "${RECOMPUTE_METHOD:=uniform}"
: "${RECOMPUTE_NUM_LAYERS:=1}"
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/gemma4_31b}"

log "launching Gemma 4 31B Dense SFT"
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    dataset "${TRAIN_JSONL}" \
    val_dataset "${VALID_JSONL}" \
    TP/PP "${TP}/${PP}" \
    MBS/GBS "${MBS}/${GBS}" \
    max_length "${MAX_LEN}" \
    LR "${LR}" \
    epochs "${EPOCHS}" \
    recompute "${RECOMPUTE}/${RECOMPUTE_METHOD}/${RECOMPUTE_NUM_LAYERS}" \
    output_dir "${OUTPUT_DIR}"

# shellcheck disable=SC2086
NPROC_PER_NODE="${NPROC_PER_NODE}" \
megatron sft \
    --model "${MODEL}" \
    --dataset "${TRAIN_JSONL}" \
    --val_dataset "${VALID_JSONL}" \
    --save_safetensors true \
    --tensor_model_parallel_size "${TP}" \
    --pipeline_model_parallel_size "${PP}" \
    --sequence_parallel true \
    --micro_batch_size "${MBS}" \
    --global_batch_size "${GBS}" \
    --packing true \
    --max_length "${MAX_LEN}" \
    --lr "${LR}" --min_lr "${MIN_LR}" \
    --lr_warmup_fraction 0.05 \
    --lr_decay_style cosine \
    --num_train_epochs "${EPOCHS}" \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --recompute_granularity "${RECOMPUTE}" \
    --recompute_method "${RECOMPUTE_METHOD}" \
    --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps "${SAVE_STEPS}" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true --no_save_rng true

log "training finished, checkpoints under ${OUTPUT_DIR}"
