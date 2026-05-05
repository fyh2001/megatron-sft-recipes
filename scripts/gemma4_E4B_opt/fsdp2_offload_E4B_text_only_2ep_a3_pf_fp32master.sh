#!/usr/bin/env bash
# Gemma-4-E4B-it text-only SFT — fp32-master + bf16 mixed precision experiment
#
# WHY: §24 cossim experiment showed FSDP2's `--torch_dtype bfloat16` leads to
# bf16 sharded master (no fp32 master). This means optimizer step is bf16,
# accumulating rounding over 800 steps → drift of ~64% direction at layer 0.
#
# DS3 baseline keeps fp32 master via `bf16.enabled=auto` + `contiguous_gradients`.
# To match that with FSDP2:
#   * load model fp32 (`--torch_dtype float32`) → sharded fp32 master kept
#   * force bf16 mixed precision (`--bf16 true --fp16 false`) → forward/backward
#     compute still bf16 (cast from fp32 master at unshard); optimizer step on
#     fp32 sharded master.
#
# Reference: PyTorch FSDP2 MixedPrecisionPolicy doc says:
#   "The sharded parameters stay in original dtype. The optimizer step uses
#    the sharded parameter in the original dtype."
#
# Memory delta vs bf16-master config:
#   * sharded fp32 master vs bf16: +2 byte × 8B params / 8 ranks = +2 GB/rank
#   * with cpu_offload, sharded master is on CPU → ~0 GPU impact
#   * unshard buffer dtype: param_dtype=bf16 → unshard is bf16 (no change)
#   * grad reduce-scatter dtype: reduce_dtype=bf16 → no change
#
# All other config matches the production §15 run.

set -euo pipefail

# Runtime paths — overridable via env vars so the same script works
# both on the host (default) and inside the docker image (override).
REPO="${REPO:-/home/ubuntu/fyh/megatron-sft-recipes}"
OUT_ROOT="${OUT_ROOT:-${REPO}/experiments/gemma4_E4B_alt_offload}"
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:-fsdp2_offload_a3_pf_2ep_fp32master}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/run_${TS}_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL="${MODEL:-/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it}"
DATASET_PATH="${DATASET_PATH:-/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl}"
# Used only for steps_per_epoch echo / report metadata; safe default if unknown.
DATASET_SIZE="${DATASET_SIZE:-51557}"

NPROC="${NPROC:-8}"
SP="${SP:-1}"
MBS="${MBS:-1}"
GBS_TARGET=128
if [ -z "${GAS:-}" ]; then
    GAS=$(( GBS_TARGET / (MBS * NPROC / SP) ))
fi
GBS=$(( MBS * NPROC * GAS / SP ))
NUM_EPOCHS=2
LR=2e-5
WARMUP_RATIO=0.05
MAX_LEN=16384
WARMUP_BENCH_FOR_REPORT=5

NUM_PARAMS=7.518e9
NUM_ACTIVE_PARAMS=4.62e9

MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

echo "=========================================="
echo "FSDP2 offload A3+padding_free — fp32 master + bf16 mp"
echo "Run dir   : ${RUN_DIR}"
echo "Dataset   : ${DATASET_PATH} (${DATASET_SIZE} samples)"
echo "Engine    : FSDP2 (full_shard auto_wrap offload), AC=on"
echo "A3        : GEMMA4_FSDP_WRAP_PLE=1 (PLE separate FSDP unit)"
echo "Padding   : padding_free=true, packing=false"
echo "Precision : torch_dtype=fp32 (master) + bf16 mp (compute)"
echo "MBS / GAS / SP / DP / GBS : ${MBS} / ${GAS} / ${SP} / $((NPROC/SP)) / ${GBS}"
echo "Epochs / LR / Warmup     : ${NUM_EPOCHS} / ${LR} / ${WARMUP_RATIO}"
echo "Steps (computed)         : per_epoch ≈ ceil(${DATASET_SIZE} / ${GBS}) ≈ $(( (DATASET_SIZE + GBS - 1) / GBS )) ; total ≈ $(( ((DATASET_SIZE + GBS - 1) / GBS) * NUM_EPOCHS ))"
echo "Port      : ${MASTER_PORT}"
echo "=========================================="

FSDP_OVERRIDE="${RUN_DIR}/fsdp_override.json"
ACT_CPU_OFFLOAD="${ACT_CPU_OFFLOAD:-true}"
cat > "${FSDP_OVERRIDE}" <<EOF
{
    "fsdp": "full_shard auto_wrap offload",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "FULL_STATE_DICT",
        "activation_checkpointing": true,
        "activation_cpu_offload": ${ACT_CPU_OFFLOAD},
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    }
}
EOF
echo "FSDP override written:"
cat "${FSDP_OVERRIDE}"

GPU_LOG="${RUN_DIR}/gpu_metrics.jsonl"
DCGM_LOG="${RUN_DIR}/dcgm_tc.tsv"
TRAIN_LOG="${RUN_DIR}/train.log"
STATUS_FILE="${RUN_DIR}/STATUS"

# GPU monitors — optional. Disable by setting GPU_MONITOR=0 / DCGM_SCRAPE=0
# (e.g. inside docker image where dcgm-exporter is not reachable).
GPU_MON_PID=
DCGM_PID=
if [ "${GPU_MONITOR:-1}" = "1" ]; then
    python3 "${REPO}/scripts/benchmark/gpu_monitor.py" --output "${GPU_LOG}" &
    GPU_MON_PID=$!
fi
if [ "${DCGM_SCRAPE:-1}" = "1" ]; then
    python3 "${REPO}/scripts/benchmark/dcgm_scrape.py" "${DCGM_LOG}" "${DCGM_URL:-http://localhost:9500/metrics}" &
    DCGM_PID=$!
fi
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

set +e
# Default behaviour on the host: re-enter the swift docker container.
# Inside the docker image (where the script is already running with the
# right env), set DOCKER_EXEC_WRAP="bash -lc" to skip the docker-exec layer.
DOCKER_EXEC_WRAP="${DOCKER_EXEC_WRAP:-docker exec fsdp_sft bash -lc}"
${DOCKER_EXEC_WRAP} "
cd ${REPO} && \
PYTHONPATH=${REPO}/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
GEMMA4_FSDP_WRAP_PLE=1 \
GEMMA4_KV_SHARE_DETACH=${GEMMA4_KV_SHARE_DETACH:-1} \
GEMMA4_FSDP_REDUCE_FP32_NCCL=${GEMMA4_FSDP_REDUCE_FP32_NCCL:-1} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
FSDP_STATE_DICT_TYPE=FULL_STATE_DICT \
${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}} \
NPROC_PER_NODE=${NPROC} MASTER_PORT=${MASTER_PORT} \
swift sft \
    --model ${MODEL} \
    --model_type gemma4 \
    --template gemma4 \
    --dataset ${DATASET_PATH} \
    --tuner_type full \
    --torch_dtype float32 \
    --bf16 true \
    --fp16 false \
    --attn_impl flash_attention_2 \
    --max_length ${MAX_LEN} \
    --truncation_strategy right \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${GAS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LR} \
    --lr_scheduler_type cosine \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay 0.1 \
    --use_liger_kernel true \
    --gradient_checkpointing true \
    --freeze_vit true --freeze_aligner true \
    --fsdp ${FSDP_OVERRIDE} \
    --sequence_parallel_size ${SP} \
    --packing false \
    --padding_free ${PADDING_FREE:-true} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy epoch \
    --save_total_limit 3 \
    --save_only_model true \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --output_dir ${RUN_DIR} \
    --logging_dir ${RUN_DIR}/runs \
" 2>&1 | tee "${TRAIN_LOG}"
EXIT=${PIPESTATUS[0]}
set -e

kill "${GPU_MON_PID}" "${DCGM_PID}" 2>/dev/null || true
wait "${GPU_MON_PID}" 2>/dev/null || true
wait "${DCGM_PID}" 2>/dev/null || true

VDIR=$(ls -dt ${RUN_DIR}/v*-* 2>/dev/null | head -n 1 || true)
if [ -n "${VDIR}" ] && [ -f "${VDIR}/logging.jsonl" ]; then
    ln -sf "${VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
fi

if [ "${EXIT}" = "0" ]; then
    STATUS="SUCCESS"
    SUMMARY="see report.json"
else
    if grep -q "CUDA out of memory" "${TRAIN_LOG}" 2>/dev/null; then
        STATUS="FAILED"
        SUMMARY="OOM"
    else
        STATUS="FAILED"
        SUMMARY="exit=${EXIT}"
    fi
fi
echo "${STATUS} — ${SUMMARY}" > "${STATUS_FILE}"

if [ -f "${RUN_DIR}/logging.jsonl" ]; then
    python3 "${REPO}/scripts/benchmark/report_swift_sp.py" \
        --logging_jsonl "${RUN_DIR}/logging.jsonl" \
        --gpu_log "${GPU_LOG}" \
        --warmup_steps "${WARMUP_BENCH_FOR_REPORT}" \
        --num_gpus "${NPROC}" \
        --gbs "${GBS}" \
        --grad_accum "${GAS}" \
        --max_len "${MAX_LEN}" \
        --backend fsdp2 \
        --sp "${SP}" \
        --num_params "${NUM_PARAMS}" \
        --num_active_params "${NUM_ACTIVE_PARAMS}" \
        --dataset_size "${DATASET_SIZE}" \
        --bench_jsonl_out "${RUN_DIR}/bench.jsonl" \
        --report_out "${RUN_DIR}/report.json" || true

    python3 "${REPO}/scripts/benchmark/extract_loss_curve.py" \
        "${RUN_DIR}/logging.jsonl" "${RUN_DIR}" || true
fi

if [ -f "${RUN_DIR}/report.json" ]; then
    echo "=========================================="
    echo "Final summary"
    echo "=========================================="
    python3 -c "
import json
d = json.load(open('${RUN_DIR}/report.json'))
keys = ['mean_step_time_ms','median_step_time_ms','p99_step_time_ms','micro_step_time_ms',
        'tokens_per_sec_per_gpu','achieved_tflops_per_gpu','achieved_tflops_per_gpu_active',
        'mfu_pct','mfu_pct_active_params',
        'peak_mem_gb','peak_mem_gib_from_swift_log',
        'avg_gpu_util_pct','avg_power_w','peak_power_w',
        'actual_total_wall_min','steps_per_epoch_for_dataset',
        'loss_first_step','loss_last_step']
for k in keys:
    print(f'  {k:32s} = {d.get(k)}')
"
fi

# Best-effort cleanup of stray python workers (host: via docker exec; inside image: direct pkill)
if [ -z "${DOCKER_EXEC_WRAP+x}" ] || [ "${DOCKER_EXEC_WRAP}" = "docker exec fsdp_sft bash -lc" ]; then
    docker exec fsdp_sft pkill -9 python 2>/dev/null || true
else
    pkill -9 python 2>/dev/null || true
fi

exit ${EXIT}
