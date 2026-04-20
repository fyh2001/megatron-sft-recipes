#!/usr/bin/env bash
# 03_sft_qwen25_7b.sh
#   Qwen2.5-7B 全参数 SFT，单机 8 卡 A100/H100 80GB。
#   用 ms-swift 的 Mcore-Bridge 模式：--model 直接指 HF 名，
#   框架自动下载 + 转 mcore 格式，不需要单独跑权重转换脚本。
#
# 前置：
#   - 已执行 01_setup_env.sh，容器就绪
#   - 已执行 02_convert_data.py，sft-data/train.jsonl + valid.jsonl 就绪
#
# 在容器内执行（source .venv/bin/activate 之后）：
#   bash /home/ubuntu/perf_opt/scripts/03_sft_qwen25_7b.sh
#
# 环境变量覆盖示例：
#   MBS=2 GBS=64 LR=2e-5 EPOCHS=3 bash scripts/03_sft_qwen25_7b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# ===== 训练参数（可通过环境变量覆盖）=====
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${TP:=2}"
: "${PP:=1}"
: "${MBS:=1}"
: "${GBS:=32}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=2}"
: "${SAVE_STEPS:=500}"
: "${RECOMPUTE:=selective}"     # none | selective | full
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/qwen25_7b}"

log "launching Qwen2.5-7B SFT"
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    dataset "${TRAIN_JSONL}" \
    val_dataset "${VALID_JSONL}" \
    TP/PP "${TP}/${PP}" \
    MBS/GBS "${MBS}/${GBS}" \
    max_length "${MAX_LEN}" \
    LR "${LR}" \
    epochs "${EPOCHS}" \
    recompute "${RECOMPUTE}" \
    output_dir "${OUTPUT_DIR}"

# 注意：ms-swift 的 per-message `loss` 字段会自动覆盖 --loss_scale，
# 不需要额外传 loss_scale 参数。
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
    --lr_scheduler_type cosine \
    --num_train_epochs "${EPOCHS}" \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --recompute_granularity "${RECOMPUTE}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps "${SAVE_STEPS}" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true --no_save_rng true

log "training finished, checkpoints under ${OUTPUT_DIR}"
