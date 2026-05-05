#!/usr/bin/env bash
# Gemma-4-E4B-it text-only SFT — final optimized full run candidate
#
# Candidate: A3 + padding_free
#   * A3: GEMMA4_FSDP_WRAP_PLE=1
#         sitecustomize patch (13/13) pulls
#         `model.language_model.embed_tokens_per_layer` (PLE, ~2.82B params)
#         out of the root FSDP unit into its own FSDP unit.  This avoids the
#         5.6+ GiB root-unit gather peak that caused earlier OOMs.
#   * padding_free=true
#         skips padded tokens in attention.  In the fair 806-scheduler/50-step
#         bench, this stacked cleanly with A3.
#
# Fair-loss short bench results (all use real 2-epoch/806-step scheduler,
# stopped by callback at step 50):
#   B0  FSDP2+offload baseline  : mean=36.83 s/it, peak=77.36 GiB, step50 loss=1.702132
#   A3  PLE separate wrap       : mean=34.88 s/it, peak=76.53 GiB, step50 loss=1.702442
#   A3+padding_free (this)      : mean=34.61 s/it, peak=75.07 GiB, step50 loss=1.699745
#
# Final training config:
#   * Model             : google/gemma-4-E4B-it
#   * Dataset           : sft-data/SFT_0424_2.jsonl (51,557 rows)
#   * Engine            : FSDP2 full_shard auto_wrap offload
#   * NPROC/SP/MBS/GAS  : 8 / 1 / 1 / 16
#   * GBS               : 128
#   * Epochs            : 2 (~806 optimizer steps)
#   * LR / warmup       : 2e-5 / 0.05, cosine
#   * max_length        : 16384
#   * truncation        : right
#   * AC                : on
#   * padding_free      : true
#   * packing           : false
#   * save              : epoch, save_only_model=true, save_total_limit=3

set -euo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
OUT_ROOT=${REPO}/experiments/gemma4_E4B_alt_offload
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:-fsdp2_offload_a3_pf_2ep_gbs128}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/run_${TS}_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it
DATASET_PATH=/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl
DATASET_SIZE=51557

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
echo "FSDP2 offload A3+padding_free — Gemma-4-E4B-it text-only 2-epoch"
echo "Run dir   : ${RUN_DIR}"
echo "Dataset   : ${DATASET_PATH} (${DATASET_SIZE} samples)"
echo "Engine    : FSDP2 (full_shard auto_wrap offload), AC=on"
echo "A3        : GEMMA4_FSDP_WRAP_PLE=1 (PLE separate FSDP unit)"
echo "Padding   : padding_free=true, packing=false"
echo "MBS / GAS / SP / DP / GBS : ${MBS} / ${GAS} / ${SP} / $((NPROC/SP)) / ${GBS}"
echo "Epochs / LR / Warmup     : ${NUM_EPOCHS} / ${LR} / ${WARMUP_RATIO}"
echo "Steps (computed)         : per_epoch ≈ ceil(${DATASET_SIZE} / ${GBS}) ≈ $(( (DATASET_SIZE + GBS - 1) / GBS )) ; total ≈ $(( ((DATASET_SIZE + GBS - 1) / GBS) * NUM_EPOCHS ))"
echo "Port      : ${MASTER_PORT}"
echo "=========================================="

FSDP_OVERRIDE="${RUN_DIR}/fsdp_override.json"
cat > "${FSDP_OVERRIDE}" <<'EOF'
{
    "fsdp": "full_shard auto_wrap offload",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "FULL_STATE_DICT",
        "activation_checkpointing": true,
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

python3 "${REPO}/scripts/benchmark/gpu_monitor.py" --output "${GPU_LOG}" &
GPU_MON_PID=$!
python3 "${REPO}/scripts/benchmark/dcgm_scrape.py" "${DCGM_LOG}" "http://localhost:9500/metrics" &
DCGM_PID=$!
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

set +e
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
GEMMA4_FSDP_WRAP_PLE=1 \
GEMMA4_KV_SHARE_DETACH=${GEMMA4_KV_SHARE_DETACH:-1} \
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
    --torch_dtype bfloat16 \
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
    --padding_free true \
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

docker exec fsdp_sft pkill -9 python 2>/dev/null || true

exit ${EXIT}
