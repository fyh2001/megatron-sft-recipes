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

# FSDP does not need Megatron's TP-sequence-parallel ordering guarantee, and
# setting MAX_CONNECTIONS=1 stalls it. Override _common.sh's default back to
# the CUDA default (8) so per-layer all_gather and compute can overlap.
export CUDA_DEVICE_MAX_CONNECTIONS=8

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
# Attention backend passed to train.py --attn_implementation.
# Useful knob for debugging FSDP2+compile stalls that might come from the
# flash_attn varlen/padding-free path interacting badly with DTensor.
: "${ATTN_IMPL:=flash_attention_2}"
# VLM backbones (e.g. Qwen3.5-9B) ship a vision tower we don't want to train
# in text-only SFT. When true, train.py zeros requires_grad on any parameter
# whose name contains "vision"/"visual"/"projector".
: "${FREEZE_VISION:=false}"

# Activation checkpointing has two implementation paths inside this repo:
#   (a) accelerate FSDP2 plugin -- fsdp_activation_checkpointing: true in the
#       rendered yaml. Preferred when it works, because it wraps ckpt AFTER
#       fully_shard inside accelerator.prepare(), preserving TRANSFORMER_BASED_WRAP.
#   (b) transformers-native model.gradient_checkpointing_enable() -- used when
#       the FSDP plugin path triggers the known flash_attn_2 + Qwen2 KV
#       CheckpointError metadata mismatch (see docs/benchmark_result.md §4.3
#       item 5). Fires BEFORE accelerator.prepare(); in recent transformers
#       the module rewrite is benign to FSDP_AUTO_WRAP by class name.
# Default: follow GRAD_CKPT (plugin path). Users can override to FSDP_ACX=false
# + GRAD_CKPT=true to force the transformers-native escape hatch.
: "${FSDP_ACX:=${GRAD_CKPT}}"

: "${BENCH_DIR:=${OUTPUT_ROOT}/benchmark}"
BENCH_OUTPUT="${BENCH_DIR}/fsdp"
mkdir -p "${BENCH_OUTPUT}"

GPU_LOG="${BENCH_OUTPUT}/gpu_metrics.jsonl"
BENCH_LOG="${BENCH_OUTPUT}/bench.jsonl"

GBS_EFF=$(( MBS * NPROC_PER_NODE * GAS ))

# ===== 动态探测模型：num_params 和 FSDP wrap 类 =====
# 让 bench 脚本脱离「一个模型改三处常量」的陷阱：
#   * --num_params 以前硬编码 7.6e9（Qwen2.5-7B）。
#   * accelerate_config.yaml 里 fsdp_transformer_layer_cls_to_wrap 以前也
#     硬编码 Qwen2DecoderLayer，换模型必忘改。
# 这里跑 _inspect_model.py：加载 AutoConfig + 在 meta device 上构造模型，
# 既不下载权重也不占 GPU，几百毫秒内即可拿到 num_params 和 decoder 类名。
# 想覆盖仍然可以用 NUM_PARAMS / FSDP_WRAP_CLS 环境变量显式指定。
if [ -z "${NUM_PARAMS:-}" ] || [ -z "${FSDP_WRAP_CLS:-}" ]; then
    log "Introspecting model to derive num_params / wrap class..."
    _INSPECT_OUT="$(python "${SCRIPT_DIR}/_inspect_model.py" "${MODEL}")"
    eval "${_INSPECT_OUT}"
    log "Model info:"
    printf '%s\n' "${_INSPECT_OUT}" | sed 's/^/  /'
fi

if [ -z "${FSDP_WRAP_CLS:-}" ]; then
    log "ERROR: could not auto-detect FSDP_WRAP_CLS for ${MODEL}; set it manually." >&2
    exit 2
fi

# Render a run-local accelerate config with
#   (a) the auto-detected wrap class, and
#   (b) fsdp_activation_checkpointing forced to match GRAD_CKPT.
# The second rewrite matters because accelerate honours the yaml flag
# regardless of whether train.py skips its own gradient_checkpointing_enable
# call; if both paths install ckpt the user ends up with double wrapping, and
# if only the yaml flag is true while train.py thinks grad_ckpt is off there
# is no way for the user to turn ckpt off from the command line.
RENDERED_ACCEL_CFG="${BENCH_OUTPUT}/accelerate_config.rendered.yaml"
sed -E \
    -e "s|^([[:space:]]*fsdp_transformer_layer_cls_to_wrap:[[:space:]]*).*|\\1${FSDP_WRAP_CLS}|" \
    -e "s|^([[:space:]]*fsdp_activation_checkpointing:[[:space:]]*).*|\\1${FSDP_ACX}|" \
    "${SCRIPT_DIR}/../fsdp/accelerate_config.yaml" > "${RENDERED_ACCEL_CFG}"

log "=== FSDP2 + compile Benchmark ==="
printf '  %-22s = %s\n' \
    model "${MODEL}" \
    num_params "${NUM_PARAMS}" \
    wrap_cls "${FSDP_WRAP_CLS}" \
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
FREEZE_VISION_FLAG=""
if [ "${FREEZE_VISION}" = "true" ]; then FREEZE_VISION_FLAG="--freeze_vision"; fi

# PROFILE=true wraps accelerate launch under `nsys profile` with capture
# controlled by cudaProfilerStart/Stop from train.py (see --profile flag
# and --profile_start_step / --profile_end_step in scripts/fsdp/train.py).
# Only records the requested step window, so the resulting nsys-rep stays
# manageable (~300-500 MB).
: "${PROFILE:=false}"
: "${PROFILE_START_STEP:=5}"
: "${PROFILE_END_STEP:=9}"
NSYS_BIN="/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys"
LAUNCH_PREFIX=""
PROFILE_FLAG=""
if [ "${PROFILE}" = "true" ]; then
    if [ ! -x "${NSYS_BIN}" ]; then
        log "ERROR: nsys not found at ${NSYS_BIN}; set NSYS_BIN env var." >&2
        exit 2
    fi
    PROFILE_FLAG="--profile --profile_start_step ${PROFILE_START_STEP} --profile_end_step ${PROFILE_END_STEP}"
    # --trace-fork-before-exec=true: accelerate launch forks 8 rank
    # subprocesses; without this flag nsys only traces the parent coordinator
    # and cudaProfilerStart calls inside train.py never get seen.
    # --capture-range-end=stop (not stop-shutdown): only stop recording on
    # cudaProfilerStop; don't kill the process tree, since non-rank-0 ranks
    # keep running FSDP collectives — killing them triggers NCCL errors.
    LAUNCH_PREFIX="${NSYS_BIN} profile \
        --trace=cuda,nvtx,cublas,cudnn \
        --trace-fork-before-exec=true \
        --cuda-memory-usage=false \
        --sample=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output=${BENCH_OUTPUT}/profile.nsys-rep \
        --force-overwrite=true"
    log "nsys profiling enabled, recording steps ${PROFILE_START_STEP}..${PROFILE_END_STEP}"
fi

log "Starting training (${TOTAL_STEPS} steps)..."

${LAUNCH_PREFIX} accelerate launch \
    --config_file "${RENDERED_ACCEL_CFG}" \
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
    --attn_implementation "${ATTN_IMPL}" \
    ${COMPILE_FLAG} ${GRAD_CKPT_FLAG} ${SYNTHETIC_FLAG} ${PAD_FLAG} ${FREEZE_VISION_FLAG} ${PROFILE_FLAG} \
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
    --num_params "${NUM_PARAMS}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --output "${BENCH_OUTPUT}/report.json"

log "Report saved to ${BENCH_OUTPUT}/report.json"
