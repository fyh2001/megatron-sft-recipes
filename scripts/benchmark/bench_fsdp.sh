#!/usr/bin/env bash
# bench_fsdp.sh
#   FSDP2 + compile 后端的性能基准测试。
#   运行 50 个 optimizer step（前 20 步 warmup 不计入，后 30 步计入指标）。
#
# 默认模型：Mistral-Nemo-12B (12.2B params)
# 需要在 fsdp_sft 容器内执行。
#
# 用法：
#   bash scripts/benchmark/bench_fsdp.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/benchmark/bench_fsdp.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# ===== Benchmark 配置 =====
: "${MODEL:=mistralai/Mistral-Nemo-Instruct-2407}"
: "${MBS:=1}"
: "${GAS:=1}"
: "${MAX_LEN:=4096}"
: "${TOTAL_STEPS:=50}"
: "${WARMUP_BENCH:=20}"
: "${COMPILE:=true}"
: "${GRAD_CKPT:=true}"
: "${PAD_TO_MAX:=true}"

: "${BENCH_DIR:=${OUTPUT_ROOT}/benchmark}"
BENCH_OUTPUT="${BENCH_DIR}/fsdp"
mkdir -p "${BENCH_OUTPUT}"

GPU_LOG="${BENCH_OUTPUT}/gpu_metrics.jsonl"
BENCH_LOG="${BENCH_OUTPUT}/bench.jsonl"

GBS_EFF=$(( MBS * NPROC_PER_NODE * GAS ))

log "=== FSDP2 + compile Benchmark ==="
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    MBS/GAS/GBS "${MBS}/${GAS}/${GBS_EFF}" \
    max_length "${MAX_LEN}" \
    total_steps "${TOTAL_STEPS}" \
    warmup_steps "${WARMUP_BENCH}" \
    compile "${COMPILE}" \
    grad_ckpt "${GRAD_CKPT}" \
    output "${BENCH_OUTPUT}"

# 启动 GPU 监控
python "${SCRIPT_DIR}/gpu_monitor.py" --output "${GPU_LOG}" &
GPU_MON_PID=$!
trap 'kill ${GPU_MON_PID} 2>/dev/null || true' EXIT

COMPILE_FLAG="--compile"
if [ "${COMPILE}" = "false" ]; then COMPILE_FLAG="--no_compile"; fi
GRAD_CKPT_FLAG="--gradient_checkpointing"
if [ "${GRAD_CKPT}" = "false" ]; then GRAD_CKPT_FLAG="--no_gradient_checkpointing"; fi
SYNTHETIC_FLAG=""
if [ "${SYNTHETIC:-false}" = "true" ]; then SYNTHETIC_FLAG="--synthetic"; fi
PAD_FLAG=""
if [ "${PAD_TO_MAX}" = "true" ]; then PAD_FLAG="--pad_to_max"; fi

log "Starting training (${TOTAL_STEPS} steps)..."

accelerate launch \
    --config_file "${SCRIPT_DIR}/../fsdp/accelerate_config.yaml" \
    --num_processes "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/../fsdp/train.py" \
    --model_name_or_path "${MODEL}" \
    --train_file "${TRAIN_JSONL}" \
    --output_dir "${BENCH_OUTPUT}" \
    --per_device_train_batch_size "${MBS}" \
    --gradient_accumulation_steps "${GAS}" \
    --max_length "${MAX_LEN}" \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.1 \
    --num_train_epochs 999 \
    --max_steps "${TOTAL_STEPS}" \
    --logging_steps 1 \
    --benchmark \
    --benchmark_log "${BENCH_LOG}" \
    --warmup_steps_bench "${WARMUP_BENCH}" \
    ${COMPILE_FLAG} ${GRAD_CKPT_FLAG} ${SYNTHETIC_FLAG} ${PAD_FLAG} \
    2>&1 | tee "${BENCH_OUTPUT}/train.log"

# 停止 GPU 監控
kill ${GPU_MON_PID} 2>/dev/null || true
wait ${GPU_MON_PID} 2>/dev/null || true

log "Benchmark complete. Generating report..."

python "${SCRIPT_DIR}/report.py" \
    --framework fsdp \
    --bench_log "${BENCH_LOG}" \
    --gpu_log "${GPU_LOG}" \
    --warmup_steps "${WARMUP_BENCH}" \
    --num_params 7.6e9 \
    --num_gpus "${NPROC_PER_NODE}" \
    --output "${BENCH_OUTPUT}/report.json"

log "Report saved to ${BENCH_OUTPUT}/report.json"
