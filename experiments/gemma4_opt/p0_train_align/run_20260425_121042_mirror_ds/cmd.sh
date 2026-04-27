#!/usr/bin/env bash
# P0g: FSDP2 ↔ DS training-curve alignment run for gemma-4-26B-A4B-it.
#
# Goal: re-run FSDP2 with the *exact* training hyperparams of the DS production
# baseline (p0_baseline_ds_prod) so per-step loss / grad_norm curves can be
# compared head-to-head.  Forward bitwise alignment was already confirmed by
# p0_forward_align (FSDP2 = single-GPU bitwise; DS bf16 mode 1.5% diff via
# autocast).  This run extends that to a *training* alignment.
#
# Diffs vs p0_baseline_fsdp2.sh (these were causing the apples-vs-oranges
# comparison the user flagged):
#
#   knob                      | FSDP2 baseline   | DS baseline       | THIS RUN (mirror DS)
#   ──────────────────────────|──────────────────|───────────────────|──────────────────────
#   per_device_train_batch_size  1                  1                   1
#   gradient_accumulation_steps  1                  16                  16  ← key
#   GBS (= MBS × DP × GAS)       4 (DP=4)           64                  64  ← key
#   learning_rate                1e-5               2e-5                2e-5
#   warmup_ratio                 0.1                0.05                0.05
#   truncation_strategy          delete (resample)  right (truncate)    right ← key
#   freeze_vit / freeze_aligner  true / true        (not set)           (not set)
#   dataloader_num_workers       2                  4                   4
#   dataloader_pin_memory        (default true)     false               false
#   torch_empty_cache_steps      (default None)     10                  10
#   template (explicit)          (auto)             gemma4              gemma4
#   max_steps (head only)        2353 (full epoch)  40 opt-steps        40 opt-steps
#
# Backend specifics (FSDP2-only; need our gemma4 modeling patch):
#   - --fsdp <override.json> with full_shard auto_wrap, fsdp_version=2,
#     reshard_after_forward=true (matches DS ZeRO-3 full-resharding semantics),
#     activation_checkpointing=true (matches DS --gradient_checkpointing true),
#     cpu_ram_efficient_loading=false (gemma4 root-level VLM params bug, see
#     gemma4_debug_log A4),
#     transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer,
#                                    Gemma4VisionEncoderLayer] (gemma4
#     _no_split_modules contains a non-existent Gemma4AudioLayer, see
#     gemma4_debug_log A1).
#   - DS uses --gradient_checkpointing true; FSDP2 path puts AC in
#     fsdp_config.activation_checkpointing instead.  Both paths AC every layer
#     so memory + activation-recompute behaviour is equivalent.
#
# Required env: depends on the patched transformers/models/gemma4/modeling_gemma4.py
# in the fsdp_sft container.  Verify with:
#   docker exec fsdp_sft md5sum /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py
# Expected md5: 39ebf386a992fea9eac0883f459ac658
#
# Expected wall: ~25 min for 40 opt-steps (16 micros × ~2 s/micro × 40 = 21 min,
# plus ~3 min for model load + setup).
#
# Output: /home/ubuntu/fyh/megatron_output/gemma4_opt/p0_train_align/run_NN_*/
#   ├── cmd.sh                 (this script, copied)
#   ├── stdout.log             (full tee'd log)
#   ├── STATUS                 (SUCCESS/FAILED)
#   ├── fsdp_override.json     (FSDP2 config used)
#   ├── gpu_metrics.jsonl      (gpu_monitor.py 1Hz)
#   ├── dcgm_tc.tsv            (DCGM tensor-core active 10Hz)
#   ├── swift_out/v0-*/        (swift trainer artefacts incl. logging.jsonl)
#   ├── logging.jsonl          (symlink to swift_out/v*/logging.jsonl for tools)
#   ├── loss_curve.tsv         (extracted by extract_loss_curve.py)
#   └── loss_curve.txt         (ASCII sparkline)
set -euo pipefail

RUN_LABEL="${RUN_LABEL:-mirror_ds}"
OUT_ROOT="/home/ubuntu/fyh/megatron_output/gemma4_opt/p0_train_align"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${RUN_LABEL}"
MAX_STEPS="${MAX_STEPS:-40}"

mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

FSDP_OVERRIDE="${RUN_DIR}/fsdp_override.json"
cat > "${FSDP_OVERRIDE}" <<'EOF'
{
    "_purpose": "FSDP2 align run mirroring DS prod-baseline. ZeRO-3 (full reshard) + AC=on, gemma4-specific workarounds.",
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": true,
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]
    }
}
EOF

echo "Run dir:        ${RUN_DIR}"
echo "FSDP override:  ${FSDP_OVERRIDE}"
echo "Max opt steps:  ${MAX_STEPS}"

# Background: GPU monitor + DCGM scrape (same telemetry as DS run, for a fair
# per-rank power/util comparison if desired).  Both daemons exit when killed.
python /home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/gpu_monitor.py \
    --output "${RUN_DIR}/gpu_metrics.jsonl" &
GPU_MON_PID=$!
python /home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/dcgm_scrape.py \
    "${RUN_DIR}/dcgm_tc.tsv" "http://localhost:9500/metrics" &
DCGM_PID=$!
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

# All flags below mirror p0_baseline_ds_prod.sh byte-for-byte EXCEPT:
#   --deepspeed → --fsdp (backend swap)
#   --gradient_checkpointing true → omitted (FSDP2 AC is in override JSON)
#   added explicit --seed 42 --data_seed 42 (defaults but written for clarity)
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
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --save_strategy no \
    --logging_steps 1 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory false \
    --fsdp ${FSDP_OVERRIDE} \
    --sequence_parallel_size 2 \
    --attn_impl flash_attention_2 \
    --use_liger_kernel true \
    --torch_empty_cache_steps 10 \
    --report_to tensorboard \
    --save_only_model true \
    --padding_free false \
    --seed 42 \
    --data_seed 42
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

# Symlink logging.jsonl + extract loss curve so the run dir is self-contained.
LATEST_VDIR="$(ls -dt ${RUN_DIR}/swift_out/v*-* 2>/dev/null | head -n 1 || true)"
if [ -n "${LATEST_VDIR}" ] && [ -f "${LATEST_VDIR}/logging.jsonl" ]; then
    ln -sf "${LATEST_VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
    python /home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark/extract_loss_curve.py \
        "${LATEST_VDIR}/logging.jsonl" "${RUN_DIR}" || true
fi

exit ${EXIT}
