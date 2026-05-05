#!/usr/bin/env bash
# DS3 bench for layer-by-layer numerical comparison vs FSDP2.
# Mirrors bench_variant.sh but uses DeepSpeed ZeRO-3 with colleague's config.
set -euo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
OUT_ROOT=${REPO}/experiments/gemma4_E4B_alt_offload/bench
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:?LABEL is required}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/run_${TS}_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it
DATASET_PATH=/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl

NPROC="${NPROC:-8}"
MBS="${MBS:-1}"
GAS="${GAS:-16}"
SP="${SP:-1}"
GBS=$(( MBS * NPROC * GAS / SP ))
NUM_EPOCHS=2
LR=2e-5
WARMUP_RATIO=0.05
MAX_LEN=16384
MAX_STEPS="${MAX_STEPS:-5}"

EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
USE_LIGER="${USE_LIGER:-true}"
GRAD_CKPT="${GRAD_CKPT:-true}"

DS_CONFIG=/home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/sp_offload_configs/zero3_colleague.json

MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

echo "=========================================="
echo "DS3 ZeRO-3 — Gemma-4-E4B-it text-only bench"
echo "Run dir : ${RUN_DIR}"
echo "DS conf : ${DS_CONFIG}"
echo "MBS/GAS/SP/NPROC/GBS : ${MBS}/${GAS}/${SP}/${NPROC}/${GBS}"
echo "Max steps (StopAfter): ${MAX_STEPS}"
echo "=========================================="

TRAIN_LOG="${RUN_DIR}/train.log"
STATUS_FILE="${RUN_DIR}/STATUS"

set +e
# Build env exports as a single block; avoids bash param-expansion bug with nested ${PYTHONPATH:-}
ENV_PREAMBLE="export PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-};
export GEMMA4_FORCE_MEM_EFFICIENT_SDP=1;
export GEMMA4_STOP_AFTER_STEPS=${MAX_STEPS};
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True';
${EXTRA_ENV:+$(printf 'export %s; ' ${EXTRA_ENV})}"
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
${ENV_PREAMBLE} \
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
    --use_liger_kernel ${USE_LIGER} \
    --gradient_checkpointing ${GRAD_CKPT} \
    --freeze_vit true --freeze_aligner true \
    --deepspeed ${DS_CONFIG} \
    --sequence_parallel_size ${SP} \
    --packing false \
    --padding_free false \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --logging_steps 1 \
    --save_strategy no \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --max_grad_norm 1.0 \
    --report_to none \
    --output_dir ${RUN_DIR} \
    --logging_dir ${RUN_DIR}/runs \
    ${EXTRA_ARGS}
" 2>&1 | tee "${TRAIN_LOG}"
EXIT=${PIPESTATUS[0]}
set -e

if [ "${EXIT}" = "0" ]; then
    echo "SUCCESS — DS3 bench finished" > "${STATUS_FILE}"
else
    if grep -q "CUDA out of memory" "${TRAIN_LOG}" 2>/dev/null; then
        echo "FAILED — OOM" > "${STATUS_FILE}"
    else
        echo "FAILED — exit=${EXIT}" > "${STATUS_FILE}"
    fi
fi

docker exec fsdp_sft pkill -9 python 2>/dev/null || true
exit ${EXIT}
