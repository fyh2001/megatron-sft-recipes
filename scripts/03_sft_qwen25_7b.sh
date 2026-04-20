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
# 在容器内执行（业务包已由 01_setup_env.sh 装进系统 Python，无需 venv）：
#   docker exec -it swift_sft bash
#   bash /home/ubuntu/perf_opt/megatron-sft-recipes/scripts/03_sft_qwen25_7b.sh
#
# 环境变量覆盖示例：
#   MBS=2 GBS=64 LR=2e-5 EPOCHS=3 bash scripts/03_sft_qwen25_7b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# ===== 训练参数（可通过环境变量覆盖）=====
# H100 80GB × 8 的推荐配比（Qwen2.5-7B BF16 + distributed optimizer + packing=4096）：
#   TP=2 PP=1  : Qwen2.5 vocab=152064 太大，megatron-core 默认的 native CE
#                fusion 会分配 (B*S, V) × fp32 临时 buffer = 4.6GB（MBS=2）/ 9.3GB
#                (MBS=4)，TP=1 时压爆 80GB。TP=2 把 vocab 切两卡 → buffer 减半。
#                （ms-swift 在 TE>=2.8 时会自动用 te 实现没这问题，但 NGC 25.03 自带
#                 TE 2.1，所以只能用 TP 切 vocab 这条路。）
#   MBS=2      : TP=2 下 MBS=2 实测约 52GB，安全边际充足
#   GBS=64     : 262k tokens/batch，7B SFT 的健康区间；DP=4/MBS=2 → accum=8
#   RECOMPUTE=none : TE FlashAttention 是 fused 的，selective 几乎不省显存
# 如果还 OOM，按这顺序降：GBS 64→32、RECOMPUTE none→selective→full、最后 MBS 2→1。
# 想冲 TP=1：要么换镜像到 TE>=2.8（NGC 25.10+ 或自己升 TE，会破坏 NGC 镜像
#   预编译 Apex/FlashAttention 的 ABI 一致性，谨慎），要么 MBS=1 降 CE buffer。
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${TP:=2}"
: "${PP:=1}"
: "${MBS:=2}"
: "${GBS:=64}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${MAX_LEN:=4096}"
: "${EPOCHS:=2}"
: "${SAVE_STEPS:=500}"
: "${RECOMPUTE:=none}"          # none | selective | full
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
