#!/usr/bin/env bash
# P0b (plan B): FSDP2 + native PyTorch Context Parallel via HF accelerate
# (skips swift entirely — swift's Ulysses SP has GQA shape bug on gemma4
# num_global_key_value_heads=2; accelerate's CP uses ring attention + SDPA
# which doesn't collide with FA head_dim=512 issue).
#
# Expected config for apples-to-apples with Qwen3.5 SP=2:
#   cp_size=2 (equivalent to Ulysses SP=2), dp_shard=4, seq=16384, MBS=1
#   AC=on (gemma4 26B MoE is too big for AC=off even with CP)
#
# Output: /home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p0_baseline_fsdp2_cp/run_NN_*/
set -euo pipefail

RUN_LABEL="${RUN_LABEL:-first_try}"
OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p0_baseline_fsdp2_cp"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${RUN_LABEL}"
mkdir -p "${RUN_DIR}"

cp "$0" "${RUN_DIR}/cmd.sh"
cp /home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/train_fsdp2_cp.py "${RUN_DIR}/train_fsdp2_cp.py"

NUM_STEPS="${NUM_STEPS:-40}"
WARMUP="${WARMUP:-5}"
CP_SIZE="${CP_SIZE:-2}"
DP_SHARD="${DP_SHARD:-4}"
MBS="${MBS:-1}"
SEQ_LEN="${SEQ_LEN:-16384}"

echo "Run dir: ${RUN_DIR}"
echo "Config: cp_size=${CP_SIZE} dp_shard=${DP_SHARD} mbs=${MBS} seq=${SEQ_LEN} steps=${NUM_STEPS}"

# Start background GPU + DCGM monitors
python /home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/gpu_monitor.py \
    --output "${RUN_DIR}/gpu_metrics.jsonl" &
GPU_MON_PID=$!
python /home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/dcgm_scrape.py \
    "${RUN_DIR}/dcgm_tc.tsv" "http://localhost:9500/metrics" &
DCGM_PID=$!
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

docker exec fsdp_sft bash -lc "cd /home/ubuntu/fyh/megatron-sft-recipes && \
    accelerate launch \
        --num-processes 8 --num-machines 1 \
        --mixed-precision bf16 \
        --main-process-port 29555 \
        scripts/gemma4_opt/train_fsdp2_cp.py \
            --cp-size ${CP_SIZE} \
            --dp-shard-size ${DP_SHARD} \
            --mbs ${MBS} \
            --seq-len ${SEQ_LEN} \
            --num-steps ${NUM_STEPS} \
            --warmup-steps ${WARMUP} \
            --output-dir ${RUN_DIR} \
            --metrics-out ${RUN_DIR}/metrics.jsonl \
    " 2>&1 | tee "${RUN_DIR}/stdout.log"
EXIT=${PIPESTATUS[0]}

kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true
wait ${GPU_MON_PID} 2>/dev/null || true
wait ${DCGM_PID} 2>/dev/null || true

if [ "${EXIT}" = "0" ]; then
    echo "SUCCESS" > "${RUN_DIR}/STATUS"
else
    echo "FAILED exit_code=${EXIT}" > "${RUN_DIR}/STATUS"
fi
exit ${EXIT}
