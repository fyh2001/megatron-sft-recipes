#!/usr/bin/env bash
# Short-run bench harness for E4B optimization study.
#
# Runs the same E4B + FSDP2 + cpu_offload baseline as the §15 run but with a
# single variant change applied; stops at MAX_STEPS (default 50) and emits a
# clean report so we can do apples-to-apples loss / s/it / peak-mem
# comparisons across variants without spending 8 h per attempt.
#
# Baseline (B0 = §15) data point — DO NOT change this file's defaults except
# for the variant under test:
#   FSDP2 + cpu_offload + AC=on + padding_free=false + no cache_steps + standard wrap
#   GBS=128 (1×8×16), seq=16384, lr=2e-5, warmup_ratio=0.05
#   measured: mean s/it 37.49, peak mem 77.54 GiB, step 50 loss 1.7030
#
# Usage:
#   LABEL="A1_padding_free" \
#   EXTRA_ARGS="--padding_free true" \
#   bash scripts/gemma4_E4B_opt/bench_variant.sh
#
# Available env knobs:
#   LABEL              required, str — short tag, becomes part of run dir name
#   EXTRA_ARGS         str          — extra CLI flags appended to `swift sft`
#   EXTRA_ENV          str          — extra env vars (e.g. NCCL_DEBUG=info)
#   FSDP_OFFLOAD_ON    {1,0}        — toggle "offload" suffix in fsdp string (default 1 = baseline)
#   FSDP_AC_ON         {true,false} — toggle activation_checkpointing in fsdp_config (default true)
#   FSDP_WRAP_EXTRA    str          — JSON-fragment to merge into fsdp_config (for custom wrap policies, A3)
#   MAX_STEPS          int          — stop after N optimizer steps (default 50)
#   FULL_SCHED_STOP    {1,0}        — if 1, keep real 2-epoch/806-step scheduler and stop at MAX_STEPS via callback

set -euo pipefail

LABEL="${LABEL:?LABEL is required (e.g. LABEL=A1_padding_free)}"

REPO=/home/ubuntu/fyh/megatron-sft-recipes
OUT_ROOT=${REPO}/experiments/gemma4_E4B_alt_offload/bench
mkdir -p "${OUT_ROOT}"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/run_${TS}_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it
DATASET_PATH=/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl
DATASET_SIZE=51557

NPROC=${NPROC:-8}
SP=${SP:-1}
MBS=${MBS:-1}
GAS=${GAS:-16}
GBS=$(( MBS * NPROC * GAS / SP ))
LR=2e-5
WARMUP_RATIO=0.05
MAX_LEN=16384
MAX_STEPS="${MAX_STEPS:-50}"
WARMUP_BENCH_FOR_REPORT=10  # skip first 10 steps for steady-state stats

NUM_PARAMS=7.518e9
NUM_ACTIVE_PARAMS=4.62e9

MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

# Allow toggling offload + AC for variants like A3/A4
FSDP_OFFLOAD_ON="${FSDP_OFFLOAD_ON:-1}"
FSDP_AC_ON="${FSDP_AC_ON:-true}"
FSDP_WRAP_EXTRA="${FSDP_WRAP_EXTRA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
FULL_SCHED_STOP="${FULL_SCHED_STOP:-0}"
TEMPLATE="${TEMPLATE:-gemma4_nothinking}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
USE_LIGER="${USE_LIGER:-true}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Render fsdp string
if [ "${FSDP_OFFLOAD_ON}" = "1" ]; then
    FSDP_STR="full_shard auto_wrap offload"
else
    FSDP_STR="full_shard auto_wrap"
fi

# Render gradient_checkpointing CLI flag (must mirror activation_checkpointing in JSON)
if [ "${FSDP_AC_ON}" = "true" ]; then
    GC_FLAG="--gradient_checkpointing true"
else
    GC_FLAG="--gradient_checkpointing false"
fi

if [ "${FULL_SCHED_STOP}" = "1" ]; then
    STEP_ARGS="--num_train_epochs 2"
    EXTRA_ENV="${EXTRA_ENV} GEMMA4_STOP_AFTER_STEPS=${MAX_STEPS}"
    SCHED_MODE="full-2ep-scheduler-stop-at-${MAX_STEPS}"
else
    STEP_ARGS="--max_steps ${MAX_STEPS} --num_train_epochs 99"
    SCHED_MODE="max_steps-${MAX_STEPS}-scheduler"
fi

echo "=========================================="
echo "BENCH variant: ${LABEL}"
echo "Run dir       : ${RUN_DIR}"
echo "MAX_STEPS     : ${MAX_STEPS}"
echo "SCHED_MODE    : ${SCHED_MODE}"
echo "FSDP_OFFLOAD  : ${FSDP_OFFLOAD_ON}"
echo "FSDP_AC       : ${FSDP_AC_ON}"
echo "EXTRA_ARGS    : ${EXTRA_ARGS}"
echo "EXTRA_ENV     : ${EXTRA_ENV}"
echo "GBS           : ${GBS} (= ${MBS} × ${NPROC} × ${GAS} / ${SP})"
echo "Port          : ${MASTER_PORT}"
echo "=========================================="

# Render fsdp_override.json
FSDP_OVERRIDE="${RUN_DIR}/fsdp_override.json"
FSDP_STR_VAR="${FSDP_STR}" \
FSDP_AC_ON_VAR="${FSDP_AC_ON}" \
FSDP_WRAP_EXTRA_VAR="${FSDP_WRAP_EXTRA}" \
python3 - <<'PYEOF' > "${FSDP_OVERRIDE}"
import json, os
cfg = {
    "fsdp": os.environ["FSDP_STR_VAR"],
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": True,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": False,
        "state_dict_type": "FULL_STATE_DICT",
        "activation_checkpointing": os.environ["FSDP_AC_ON_VAR"] == "true",
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"],
    }
}
extra = os.environ.get("FSDP_WRAP_EXTRA_VAR", "").strip()
if extra:
    cfg["fsdp_config"].update(json.loads(extra))
print(json.dumps(cfg, indent=4))
PYEOF
echo "FSDP override written:"
cat "${FSDP_OVERRIDE}"

# Background monitors
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
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
FSDP_STATE_DICT_TYPE=FULL_STATE_DICT \
${EXTRA_ENV} \
NPROC_PER_NODE=${NPROC} MASTER_PORT=${MASTER_PORT} \
swift sft \
    --model ${MODEL} \
    --model_type gemma4 \
    --template ${TEMPLATE} \
    --dataset ${DATASET_PATH} \
    --tuner_type full \
    --torch_dtype ${TORCH_DTYPE} \
    --attn_impl flash_attention_2 \
    --max_length ${MAX_LEN} \
    --truncation_strategy right \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${GAS} \
    ${STEP_ARGS} \
    --learning_rate ${LR} \
    --lr_scheduler_type cosine \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --use_liger_kernel ${USE_LIGER} \
    ${GC_FLAG} \
    --freeze_vit true --freeze_aligner true \
    --fsdp ${FSDP_OVERRIDE} \
    --sequence_parallel_size ${SP} \
    --packing false \
    --padding_free false \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy no \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --output_dir ${RUN_DIR} \
    --logging_dir ${RUN_DIR}/runs \
    ${EXTRA_ARGS} \
" 2>&1 | tee "${TRAIN_LOG}"
EXIT=${PIPESTATUS[0]}
set -e

kill "${GPU_MON_PID}" "${DCGM_PID}" 2>/dev/null || true
wait "${GPU_MON_PID}" 2>/dev/null || true
wait "${DCGM_PID}" 2>/dev/null || true

# Resolve swift's auto-named v*-* sub-dir
VDIR=$(ls -dt ${RUN_DIR}/v*-* 2>/dev/null | head -n 1 || true)
if [ -n "${VDIR}" ] && [ -f "${VDIR}/logging.jsonl" ]; then
    ln -sf "${VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
fi

# Categorise outcome
if [ "${EXIT}" = "0" ]; then
    STATUS="SUCCESS"
    SUMMARY="bench finished ${MAX_STEPS} steps (${SCHED_MODE})"
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
echo "STATUS = ${STATUS} — ${SUMMARY}"

# Quick summary table
if [ -f "${RUN_DIR}/logging.jsonl" ]; then
    echo "=========================================="
    echo "Variant ${LABEL} — bench summary"
    echo "Baseline (B0 §15): mean s/it 37.49, peak 77.54 GiB, step 50 loss 1.7030"
    echo "=========================================="
    python3 - <<PYEOF
import json, statistics
rows = [json.loads(l) for l in open("${RUN_DIR}/logging.jsonl") if l.startswith('{')]
n = len(rows)
print(f"  total steps logged: {n}")
if n > 0:
    rows = [r for r in rows if 'train_speed(s/it)' in r]
    n = len(rows)
    sit = [r['train_speed(s/it)'] for r in rows[10:]]
    mem = [r['memory(GiB)'] for r in rows]
    print(f"  training-step rows: {n}")
    print(f"  step 1   loss = {rows[0]['loss']:.4f}, mem = {rows[0]['memory(GiB)']} GiB, s/it = {rows[0]['train_speed(s/it)']:.2f}, lr = {rows[0]['learning_rate']}")
    if n >= 30:
        print(f"  step 30  loss = {rows[29]['loss']:.4f}, mem = {rows[29]['memory(GiB)']} GiB, s/it = {rows[29]['train_speed(s/it)']:.2f}, lr = {rows[29]['learning_rate']}")
    if n >= 50:
        print(f"  step 50  loss = {rows[49]['loss']:.4f}, mem = {rows[49]['memory(GiB)']} GiB, s/it = {rows[49]['train_speed(s/it)']:.2f}, lr = {rows[49]['learning_rate']}")
    if len(sit) > 0:
        print(f"  steady s/it (step >=10, n={len(sit)}):")
        print(f"    mean   = {statistics.mean(sit):.2f}")
        print(f"    median = {statistics.median(sit):.2f}")
    print(f"  peak mem (swift): {max(mem)} GiB")
PYEOF
fi

# Cleanup any straggler workers
docker exec fsdp_sft pkill -9 python 2>/dev/null || true

exit ${EXIT}
