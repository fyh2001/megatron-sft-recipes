#!/usr/bin/env bash
# 07_sft_ministral3_3b.sh
#   Ministral-3 3B Instruct 全参数 SFT，单机 8 卡 H100 80GB。
#   走 ms-swift 的 HF Transformers 后端（`swift sft`），不走 Megatron 后端：
#   mcore_bridge 目前没有 mistral3 converter，megatron-core 也没适配 Pixtral ViT。
#
#   模型是多模态 VLM（3.4B Mistral LM + 0.4B Pixtral ViT）。
#   你的 RP 数据纯文本 → vision tower 冻结，只训 LM 和 projector。
#   用官方 BF16 反量化权重（`-BF16` 后缀）省掉 FP8 dequantize 环节。
#
# 前置：
#   - 已执行 01_setup_env.sh，容器就绪
#   - 已执行 02_convert_data.py，sft-data/train.jsonl + valid.jsonl 就绪
#   - mistral-common + hf_transfer 已装（首次 smoke 验证 Ministral 推理时已装）
#
# 在容器内执行：
#   docker exec -it swift_sft bash
#   bash /home/ubuntu/perf_opt/megatron-sft-recipes/scripts/07_sft_ministral3_3b.sh
#
# 环境变量覆盖示例：
#   MBS=2 GAS=4 LR=2e-5 EPOCHS=3 bash scripts/07_sft_ministral3_3b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/../_common.sh"

# ===== 数据源配置：走 HF + hf_transfer（Ministral 在 ModelScope 的 CDN 只有 9MB/s；HF 能到 600MB/s） =====
export USE_HF=1
export HF_HUB_ENABLE_HF_TRANSFER=1
# HF_HOME 已由 _common.sh 设到挂载卷

# ===== 训练参数（可通过环境变量覆盖）=====
# H100 80GB × 8 的推荐配比（Ministral-3 3B BF16 + DDP + flash_attn + gradient checkpointing）：
#   全参数 SFT 显存估算（单卡，DP=8 每卡都有完整模型/梯度/optimizer）：
#     模型 7.7GB + 梯度 7.7GB + Adam optim 30.8GB = 46GB static
#     freeze_vit 省 0.45B × 16 bytes = 7GB → 39GB static
#     + activation (MBS=4, seq=2048) with GC ≈ 4GB
#     = ≈ 43GB/卡，很安全
#   GBS = MBS × DP × GAS = 4 × 8 × 2 = 64
#
# 注意：
#   Ministral-3 的 chat template `mistral_2512` 在 ms-swift 4.1.2 里暂**不支持
#   packing / padding_free** → 只能用传统 right-pad 训练。你的对话平均 ~1100 token，
#   MAX_LEN=2048 够用且 padding 浪费约 50%（如果 4096 会浪费 73%）。
#   加 --group_by_length 让 batch 内长度接近，进一步减少 padding 浪费。
# 如果 OOM，按这顺序降：MBS 4→2、GAS 2→4（保 GBS 不变）、MAX_LEN 2048→1536。
: "${MODEL:=mistralai/Ministral-3-3B-Instruct-2512-BF16}"
: "${MBS:=4}"                   # per_device_train_batch_size
: "${GAS:=2}"                   # gradient_accumulation_steps
: "${LR:=1e-5}"
: "${MAX_LEN:=2048}"
: "${EPOCHS:=2}"
: "${WARMUP:=0.05}"
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/ministral3_3b}"

# effective global batch size (供用户参考)
_DP="${NPROC_PER_NODE}"
GBS_EFF=$(( MBS * _DP * GAS ))

log "launching Ministral-3 3B Instruct SFT (HF backend, full-parameter)"
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    dataset "${TRAIN_JSONL}" \
    val_dataset "${VALID_JSONL}" \
    DP/MBS/GAS "${_DP}/${MBS}/${GAS}" \
    GBS_eff "${GBS_EFF}" \
    max_length "${MAX_LEN}" \
    LR "${LR}" \
    epochs "${EPOCHS}" \
    freeze_vit "true" \
    output_dir "${OUTPUT_DIR}"

# ms-swift 的 per-message `loss` 字段会自动覆盖 --loss_scale，跟 Megatron 后端一致。
# shellcheck disable=SC2086
NPROC_PER_NODE="${NPROC_PER_NODE}" \
USE_HF=1 \
HF_HUB_ENABLE_HF_TRANSFER=1 \
swift sft \
    --model "${MODEL}" \
    --tuner_type full \
    --freeze_vit true \
    --dataset "${TRAIN_JSONL}" \
    --val_dataset "${VALID_JSONL}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --max_length "${MAX_LEN}" \
    --truncation_strategy delete \
    --group_by_length true \
    --per_device_train_batch_size "${MBS}" \
    --gradient_accumulation_steps "${GAS}" \
    --learning_rate "${LR}" \
    --lr_scheduler_type cosine \
    --warmup_ratio "${WARMUP}" \
    --num_train_epochs "${EPOCHS}" \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --save_strategy epoch \
    --eval_strategy no \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8

log "training finished, checkpoints under ${OUTPUT_DIR}"
