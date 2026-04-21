#!/usr/bin/env bash
# 08_sft_ministral3_text.sh
#   Ministral-3 3B Instruct 纯文本 SFT，单机 8 卡 H100 80GB，走 Megatron 后端。
#
#   前提：已跑过 scripts/convert_ministral3_to_llama.py，把 VLM 结构砍成纯文本
#   LlamaForCausalLM（vision_tower + multi_modal_projector 权重删掉，config 改成
#   architectures=["LlamaForCausalLM"]、model_type="llama"、关掉 YaRN scaling）。
#   这样 mcore_bridge 的 Llama 路径就能直接认，避免改 bridge 源码的 2-5 人周工作量。
#
#   Ministral-3 text backbone 本质是 GQA Llama-ish：
#     hidden=3072, layers=26, heads=32, kv_heads=8, head_dim=128, vocab=131072
#     RMSNorm + RoPE(θ=1e6) + SwiGLU + tie_word_embeddings=True
#     没有 sliding_window，可以完全走 Llama 路径
#
# 前置：
#   - 已执行 01_setup_env.sh
#   - 已执行 02_convert_data.py
#   - 已执行 scripts/convert_ministral3_to_llama.py（产出在
#     /home/ubuntu/perf_opt/models/ministral3-3b-text-llama）
#
# 在容器内执行：
#   docker exec -it swift_sft bash
#   bash /home/ubuntu/perf_opt/megatron-sft-recipes/scripts/08_sft_ministral3_text.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/../_common.sh"

# ===== 训练参数（可通过环境变量覆盖）=====
# H100 80GB × 8 的推荐配比（Ministral-3 3B BF16 + TP=2/PP=1 + distributed optimizer + packing=4096）：
#   显存估算（单卡，TP=2 把模型切两卡）：
#     模型 3.4GB + 梯度 3.4GB + optim/DP 13.7/4 = 3.4GB ≈ 10GB static
#     + CE fp32 buffer (vocab=131072, MBS=4, seq=4096) / TP=2 = 4.3GB
#     + activation (MBS=4, seq=4096) with FlashAttn ≈ 15GB
#     = ≈ 30-35GB/卡，非常宽裕
#   Ministral vocab=131072 比 Qwen2.5 的 152064 小 14%，CE OOM 压力小一些，
#   但 TP=1 + MBS=4 仍可能撞顶；保守用 TP=2。
# GBS=64 与 Qwen 脚本对齐；DP=4/MBS=4 → accum=4。
# 默认指向 convert_ministral3_to_llama.py 的默认输出位置；
# _common.sh 的 HOST_MOUNT 已动态按仓库路径推导，老部署仍可 HOST_MOUNT=/home/ubuntu/perf_opt 覆盖。
: "${MODELS_DIR:=${HOST_MOUNT}/models}"
: "${MODEL:=${MODELS_DIR}/ministral3-3b-text-llama}"
: "${TP:=2}"
: "${PP:=1}"
: "${MBS:=4}"
: "${GBS:=64}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=2}"
: "${SAVE_STEPS:=500}"
: "${RECOMPUTE:=none}"          # none | selective | full
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/ministral3_3b_text}"

# 本地模型路径不在 ms-swift 的 MODEL_MAPPING 里，必须显式传 model_type / template
: "${MODEL_TYPE:=llama}"        # 伪装 Llama
: "${TEMPLATE:=llama}"          # Mistral 的 [INST]...[/INST] 格式在 swift 里就叫 llama template

log "launching Ministral-3 3B (text-only, disguised as Llama) SFT via Megatron"
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    model_type "${MODEL_TYPE}" \
    template "${TEMPLATE}" \
    dataset "${TRAIN_JSONL}" \
    val_dataset "${VALID_JSONL}" \
    TP/PP "${TP}/${PP}" \
    MBS/GBS "${MBS}/${GBS}" \
    max_length "${MAX_LEN}" \
    LR "${LR}" \
    epochs "${EPOCHS}" \
    recompute "${RECOMPUTE}" \
    output_dir "${OUTPUT_DIR}"

# shellcheck disable=SC2086
NPROC_PER_NODE="${NPROC_PER_NODE}" \
megatron sft \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --template "${TEMPLATE}" \
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
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps "${SAVE_STEPS}" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true --no_save_rng true

log "training finished, checkpoints under ${OUTPUT_DIR}"
