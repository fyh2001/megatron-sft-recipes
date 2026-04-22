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
# When RECOMPUTE=full, mcore requires a --recompute_method (block / uniform)
# and --recompute_num_layers N. Defaults here match ms-swift's upstream
# qwen3_next/mcore.sh recipe: uniform across all layers, N=1 layer per group
# (most aggressive memory savings). Only applied when RECOMPUTE=full.
: "${RECOMPUTE_METHOD:=uniform}"
: "${RECOMPUTE_NUM_LAYERS:=1}"
# For VLM backbones (Qwen3.5-9B, etc.), freeze the vision tower + aligner so
# only the text backbone is trained. ms-swift's `megatron sft` path honours
# --freeze_vit / --freeze_aligner; the swift sft (HF backend) branch ignores.
: "${FREEZE_VIT:=false}"

: "${BENCH_DIR:=${OUTPUT_ROOT}/benchmark}"
BENCH_OUTPUT="${BENCH_DIR}/megatron"
mkdir -p "${BENCH_OUTPUT}"

GPU_LOG="${BENCH_OUTPUT}/gpu_metrics.jsonl"
TRAIN_LOG="${BENCH_OUTPUT}/train.log"

# ===== 动态 num_params =====
# 以前写死 7.6e9（Qwen2.5-7B），换模型必错。现在跑 _inspect_model.py 在
# meta-device 上构建模型、数 params，几百毫秒就能出准确值。
# 支持 NUM_PARAMS 环境变量显式覆盖（用于还没支持 AutoConfig 的冷门模型）。
if [ -z "${NUM_PARAMS:-}" ]; then
    SCRIPT_BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    log "Introspecting model to derive num_params..."
    _INSPECT_OUT="$(python "${SCRIPT_BENCH_DIR}/_inspect_model.py" "${MODEL}")"
    eval "${_INSPECT_OUT}"
    log "Model info:"
    printf '%s\n' "${_INSPECT_OUT}" | sed 's/^/  /'
fi

log "=== Megatron / ms-swift Benchmark ==="
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    num_params "${NUM_PARAMS}" \
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

# PROFILE=true wraps the training command under `nsys profile`. Megatron
# does not expose cudaProfilerStart/Stop hooks, so we rely on nsys's
# --delay/--duration (in seconds) to only record a time window in which
# training is at steady state (after compile + dataset packing).
: "${PROFILE:=false}"
: "${PROFILE_DELAY:=110}"
: "${PROFILE_DURATION:=5}"
NSYS_BIN="/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys"
LAUNCH_PREFIX=""
if [ "${PROFILE}" = "true" ]; then
    if [ ! -x "${NSYS_BIN}" ]; then
        log "ERROR: nsys not found at ${NSYS_BIN}; set NSYS_BIN env var." >&2
        exit 2
    fi
    # --trace-fork-before-exec=true: swift → megatron sft → torchrun fork
    # chain, nsys only sees the rank subprocesses if this is enabled.
    LAUNCH_PREFIX="${NSYS_BIN} profile \
        --trace=cuda,nvtx,cublas,cudnn \
        --trace-fork-before-exec=true \
        --cuda-memory-usage=false \
        --sample=none \
        --delay=${PROFILE_DELAY} \
        --duration=${PROFILE_DURATION} \
        --output=${BENCH_OUTPUT}/profile.nsys-rep \
        --force-overwrite=true"
    log "nsys profiling enabled, delay=${PROFILE_DELAY}s duration=${PROFILE_DURATION}s"
fi

log "Starting training (${TOTAL_STEPS} steps)..."

FREEZE_VIT_ARGS=""
if [ "${FREEZE_VIT}" = "true" ]; then
    FREEZE_VIT_ARGS="--freeze_vit true --freeze_aligner true"
fi

RECOMPUTE_METHOD_ARGS=""
if [ "${RECOMPUTE}" = "full" ]; then
    RECOMPUTE_METHOD_ARGS="--recompute_method ${RECOMPUTE_METHOD} --recompute_num_layers ${RECOMPUTE_NUM_LAYERS}"
fi

if [ "${USE_MEGATRON_BACKEND}" = "true" ]; then
    # Megatron 后端（需要模型在 mcore-bridge 的支持列表里）
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    ${LAUNCH_PREFIX} megatron sft \
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
        --train_iters "${TOTAL_STEPS}" \
        --finetune true \
        --cross_entropy_loss_fusion true \
        --recompute_granularity "${RECOMPUTE}" \
        ${RECOMPUTE_METHOD_ARGS} \
        --use_distributed_optimizer true \
        --overlap_grad_reduce true \
        --overlap_param_gather true \
        ${FREEZE_VIT_ARGS} \
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
    ${LAUNCH_PREFIX} swift sft \
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

# packing + fixed MAX_LEN 下每步吞吐的 token 数是 GBS * MAX_LEN；
# ms-swift megatron 日志里只有 iteration / elapsed_time / train_speed，不直接给 tokens。
TOKENS_PER_STEP=$(( GBS * MAX_LEN ))

python "${SCRIPT_DIR}/report.py" \
    --framework megatron \
    --train_log "${TRAIN_LOG}" \
    --gpu_log "${GPU_LOG}" \
    --warmup_steps "${WARMUP_BENCH}" \
    --num_params "${NUM_PARAMS}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --tokens_per_step "${TOKENS_PER_STEP}" \
    --output "${BENCH_OUTPUT}/report.json"

log "Report saved to ${BENCH_OUTPUT}/report.json"
