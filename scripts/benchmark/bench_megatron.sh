#!/usr/bin/env bash
# bench_megatron.sh
#   Megatron (ms-swift mcore-bridge) 后端的性能基准测试。
#   运行 50 个 step（前 20 步 warmup 不计入，后 30 步计入指标）。
#
# 默认模型：Mistral-Nemo-12B (12.2B params)
#   注意：Mistral-Nemo 需要通过 convert_ministral3_to_llama.py 转换后才能跑 Megatron，
#   或直接用 Qwen2.5-14B（Megatron 原生支持）。这里默认 Qwen2.5-14B 以保持可比性。
#   如果已有转好的 Mistral-Nemo Llama 权重，设 MODEL=<local_path> MODEL_TYPE=llama。
#
# 需要在 swift_sft 容器内执行。
#
# 用法：
#   bash scripts/benchmark/bench_megatron.sh
#   MODEL=Qwen/Qwen2.5-14B-Instruct bash scripts/benchmark/bench_megatron.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# ===== Benchmark 配置 =====
# 默认用可直接作为对比的模型配置
# Mistral-Nemo-12B 在 Megatron 端需要转换成 Llama，所以这里提供两种选择：
#   1. 如果有转好的 Mistral-Nemo（MODEL=<local_path> MODEL_TYPE=llama TEMPLATE=llama）
#   2. 直接用 mistralai/Mistral-Nemo-Instruct-2407 走 swift sft 的 HF 后端做对照
# 为了和 FSDP 脚本对齐模型，默认走 HF 后端（swift sft），这样两边都是相同模型。
: "${MODEL:=mistralai/Mistral-Nemo-Instruct-2407}"
: "${USE_MEGATRON_BACKEND:=false}"
: "${TP:=2}"
: "${PP:=1}"
: "${MBS:=1}"
: "${GBS:=8}"
: "${MAX_LEN:=4096}"
: "${TOTAL_STEPS:=50}"
: "${WARMUP_BENCH:=20}"
: "${RECOMPUTE:=selective}"

: "${BENCH_DIR:=${OUTPUT_ROOT}/benchmark}"
BENCH_OUTPUT="${BENCH_DIR}/megatron"
mkdir -p "${BENCH_OUTPUT}"

GPU_LOG="${BENCH_OUTPUT}/gpu_metrics.jsonl"
TRAIN_LOG="${BENCH_OUTPUT}/train.log"

log "=== Megatron / ms-swift Benchmark ==="
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    backend "${USE_MEGATRON_BACKEND}" \
    TP/PP "${TP}/${PP}" \
    MBS/GBS "${MBS}/${GBS}" \
    max_length "${MAX_LEN}" \
    total_steps "${TOTAL_STEPS}" \
    warmup_steps "${WARMUP_BENCH}" \
    recompute "${RECOMPUTE}" \
    output "${BENCH_OUTPUT}"

# 启动 GPU 监控
python "${SCRIPT_DIR}/gpu_monitor.py" --output "${GPU_LOG}" &
GPU_MON_PID=$!
trap 'kill ${GPU_MON_PID} 2>/dev/null || true' EXIT

log "Starting training (${TOTAL_STEPS} steps)..."

if [ "${USE_MEGATRON_BACKEND}" = "true" ]; then
    # Megatron 后端（需要模型在 mcore-bridge 的支持列表里）
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    megatron sft \
        --model "${MODEL}" \
        --dataset "${TRAIN_JSONL}" \
        --save_safetensors true \
        --tensor_model_parallel_size "${TP}" \
        --pipeline_model_parallel_size "${PP}" \
        --sequence_parallel true \
        --micro_batch_size "${MBS}" \
        --global_batch_size "${GBS}" \
        --packing true \
        --max_length "${MAX_LEN}" \
        --lr 1e-5 --min_lr 1e-6 \
        --lr_warmup_fraction 0.1 \
        --lr_decay_style cosine \
        --num_train_epochs 999 \
        --max_steps "${TOTAL_STEPS}" \
        --finetune true \
        --cross_entropy_loss_fusion true \
        --recompute_granularity "${RECOMPUTE}" \
        --use_distributed_optimizer true \
        --overlap_grad_reduce true \
        --overlap_param_gather true \
        --output_dir "${BENCH_OUTPUT}/ckpt" \
        --save_steps 99999 \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --no_save_optim true --no_save_rng true \
        2>&1 | tee "${TRAIN_LOG}"
else
    # HF Transformers 后端（swift sft）—— 更通用，任何 HF 模型都能跑
    GAS=$(( GBS / (MBS * NPROC_PER_NODE) ))
    if [ "${GAS}" -lt 1 ]; then GAS=1; fi

    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    swift sft \
        --model "${MODEL}" \
        --tuner_type full \
        --dataset "${TRAIN_JSONL}" \
        --torch_dtype bfloat16 \
        --attn_impl flash_attention_2 \
        --max_length "${MAX_LEN}" \
        --truncation_strategy delete \
        --per_device_train_batch_size "${MBS}" \
        --gradient_accumulation_steps "${GAS}" \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --num_train_epochs 999 \
        --max_steps "${TOTAL_STEPS}" \
        --gradient_checkpointing true \
        --logging_steps 1 \
        --save_strategy no \
        --eval_strategy no \
        --output_dir "${BENCH_OUTPUT}/ckpt" \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        2>&1 | tee "${TRAIN_LOG}"
fi

# 停止 GPU 监控
kill ${GPU_MON_PID} 2>/dev/null || true
wait ${GPU_MON_PID} 2>/dev/null || true

log "Benchmark complete. Generating report..."

python "${SCRIPT_DIR}/report.py" \
    --framework megatron \
    --train_log "${TRAIN_LOG}" \
    --gpu_log "${GPU_LOG}" \
    --warmup_steps "${WARMUP_BENCH}" \
    --num_params 12.2e9 \
    --num_gpus "${NPROC_PER_NODE}" \
    --output "${BENCH_OUTPUT}/report.json"

log "Report saved to ${BENCH_OUTPUT}/report.json"
