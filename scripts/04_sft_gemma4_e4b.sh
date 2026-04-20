#!/usr/bin/env bash
# 04_sft_gemma4_e4b.sh
#   Gemma 4 E4B (~4.5B effective params) SFT，单机 8 卡 A100/H100 80GB 轻松跑。
#   主要用来验证"切模型流程"——和 03 脚本相比只改了 --model 和并行度。
#
# 注意：需要 ms-swift >= 4.1.0（官方 PR #8508 加入 Gemma 4 支持），01_setup_env.sh 已装。
#
# 运行方式：
#   bash /home/ubuntu/perf_opt/scripts/04_sft_gemma4_e4b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# ===== 训练参数 =====
# Gemma 4 E4B 在 modelscope 上的官方 ID（LLM-Research 是 modelscope 上常用的镜像组织）
# 如果拉不到可以换成 HF: --model google/gemma-4-e4b-it
: "${MODEL:=LLM-Research/gemma-4-e4b-it}"
: "${TP:=1}"
: "${PP:=1}"
: "${MBS:=2}"
: "${GBS:=32}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=2}"
: "${SAVE_STEPS:=500}"
: "${RECOMPUTE:=selective}"
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/gemma4_e4b}"

log "launching Gemma 4 E4B SFT"
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    dataset "${TRAIN_JSONL}" \
    val_dataset "${VALID_JSONL}" \
    TP/PP "${TP}/${PP}" \
    MBS/GBS "${MBS}/${GBS}" \
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
    --sequence_parallel false \
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
