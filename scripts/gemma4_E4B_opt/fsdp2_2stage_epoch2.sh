#!/usr/bin/env bash
# 2-stage training: stage 2 of 2.
#
# Hypothesis: FSDP2's epoch-2 loss stagnation is dominated by Adam optimizer
# state inheriting biased momentum/variance from epoch 1's inflated gradients.
# Fix: launch stage 2 as a FRESH training run, loading model weights from
# stage-1 checkpoint (epoch-1 end) but starting with fresh optimizer state.
#
# Stage 1 = the existing 1-epoch run results.
# Stage 2 = this script, runs 1 epoch with fresh optimizer.
#
# Net effect = 2 epochs of total training, but with the bad momentum from
# epoch 1 reset before epoch 2 starts.

set -euo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
OUT_ROOT=${REPO}/experiments/gemma4_E4B_alt_offload
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:-fsdp2_2stage_epoch2}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/run_${TS}_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

# Stage 1's checkpoint — using no-detach run's epoch-1 checkpoint (loss 1.61)
STAGE1_CKPT="${STAGE1_CKPT:-/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_E4B_alt_offload/run_20260501_024401_fsdp2_offload_a3_pf_2ep_NO_detach/v0-20260501-104443/checkpoint-403}"

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
NUM_EPOCHS=1   # stage 2 = 1 epoch

# LR for stage 2 — start from where stage 1 effectively ended (mid-cosine).
# At step 403/806 in original schedule, LR ≈ peak * cos(0.5 * pi * (403-40)/(806-40))/2 ≈ 1e-5.
# Use peak=1e-5 with shorter warmup (5 steps) + cosine decay over the rest.
LR="${LR:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.012}"  # ~5 steps warmup over 403 steps
MAX_LEN=16384

NUM_PARAMS=7.518e9
NUM_ACTIVE_PARAMS=4.62e9

MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

echo "=========================================="
echo "FSDP2 2-stage epoch 2 — fresh optimizer"
echo "Run dir       : ${RUN_DIR}"
echo "Stage 1 ckpt  : ${STAGE1_CKPT}"
echo "Dataset       : ${DATASET_PATH} (${DATASET_SIZE} samples)"
echo "Stage 2 epochs: ${NUM_EPOCHS} (~$(( (DATASET_SIZE + GBS - 1) / GBS )) steps)"
echo "LR            : ${LR} (warmup_ratio ${WARMUP_RATIO})"
echo "MBS/GAS/SP/DP/GBS : ${MBS}/${GAS}/${SP}/$((NPROC/SP))/${GBS}"
echo "Port          : ${MASTER_PORT}"
echo "=========================================="

# Render fsdp_override.json (same as stage 1)
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
docker exec -e PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble \
  -e GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
  -e GEMMA4_FSDP_WRAP_PLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e FSDP_STATE_DICT_TYPE=FULL_STATE_DICT \
  -e NPROC_PER_NODE=${NPROC} \
  -e MASTER_PORT=${MASTER_PORT} \
  fsdp_sft \
  swift sft \
    --model "${STAGE1_CKPT}" \
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
    --data_seed 43 \
    --seed 43 \
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
    --save_only_model true \
    --save_total_limit 2 \
    --max_grad_norm 1.0 \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --output_dir "${RUN_DIR}" \
    --logging_dir "${RUN_DIR}/runs" 2>&1 | tee "${TRAIN_LOG}"
EXIT=${PIPESTATUS[0]}
set -e

kill "${GPU_MON_PID}" "${DCGM_PID}" 2>/dev/null || true

if [ "${EXIT}" = "0" ]; then
    echo "SUCCESS — see logging.jsonl in v0-*/" > "${STATUS_FILE}"
else
    echo "FAILED — exit=${EXIT}" > "${STATUS_FILE}"
fi
echo "STATUS: $(cat ${STATUS_FILE})"

docker exec fsdp_sft pkill -9 python 2>/dev/null || true
exit ${EXIT}
