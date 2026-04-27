#!/usr/bin/env bash
# P0c: DeepSpeed production-config baseline for gemma-4-26B-A4B-it
#
# This is your online SFT command (provided 2026-04-23), with only these tweaks
# required to make it a repeatable bench run on this machine:
#   --model     -> /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
#   --dataset   -> our local sft-data (production data not on this host)
#   --output_dir-> bench output root
#   --max_steps 40 added (original had --num_train_epochs 1 which is 13h)
#   --save_strategy no added (original saves every 50 steps, pollutes timing)
#
# Everything else preserved byte-for-byte: DS ZeRO-3 + offload(opt+param) CPU,
# SP=2, MBS=1 grad_accum=16 (GBS=64), AC=on, overlap_comm=false, bucket=5e7, etc.
#
# Expected wall: ~20 min for 40 optimizer steps (each step = 16 micro-steps).
# Output: /home/ubuntu/fyh/megatron_output/gemma4_opt/p0_baseline_ds_prod/run_NN_*/
set -euo pipefail

RUN_LABEL="${RUN_LABEL:-first_try}"
OUT_ROOT="/home/ubuntu/fyh/megatron_output/gemma4_opt/p0_baseline_ds_prod"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${RUN_LABEL}"
mkdir -p "${RUN_DIR}"

# Save the exact cmd + ds config with this run
cp "$0" "${RUN_DIR}/cmd.sh"
cp "$(dirname "$0")/zero3_offload_nopin.json" "${RUN_DIR}/zero3_offload_nopin.json"

DS_CONFIG="${RUN_DIR}/zero3_offload_nopin.json"

echo "Run dir: ${RUN_DIR}"

# Start GPU monitor + DCGM scraper in background so we get the same
# telemetry as FSDP2 runs.
python /home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/gpu_monitor.py \
    --output "${RUN_DIR}/gpu_metrics.jsonl" &
GPU_MON_PID=$!
python /home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/dcgm_scrape.py \
    "${RUN_DIR}/dcgm_tc.tsv" "http://localhost:9500/metrics" &
DCGM_PID=$!
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

# Run the DS training. Everything below is the user's original command with
# only the three tweaks noted above.
docker exec fsdp_sft bash -lc "
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=29501 \
swift sft \
    --model /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it \
    --model_type gemma4 \
    --template gemma4 \
    --output_dir ${RUN_DIR}/swift_out \
    --tuner_type full \
    --dataset /home/ubuntu/fyh/megatron-sft-recipes/sft-data/train.jsonl \
    --load_from_cache_file false \
    --truncation_strategy right \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --max_steps 40 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --save_strategy no \
    --logging_steps 1 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory false \
    --deepspeed ${DS_CONFIG} \
    --sequence_parallel_size 2 \
    --attn_impl flash_attention_2 \
    --use_liger_kernel true \
    --gradient_checkpointing true \
    --torch_empty_cache_steps 10 \
    --report_to tensorboard \
    --save_only_model true \
    --padding_free false
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

# Point at the logging.jsonl for parse_swift_log.py
LATEST_VDIR="$(ls -dt ${RUN_DIR}/swift_out/v*-* 2>/dev/null | head -n 1 || true)"
if [ -n "${LATEST_VDIR}" ] && [ -f "${LATEST_VDIR}/logging.jsonl" ]; then
    ln -sf "${LATEST_VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
fi

exit ${EXIT}
