#!/usr/bin/env bash
# bench_swift_sp_v2.sh
#   Same matrix concept as bench_swift_sp.sh but WITHOUT the swift_sp_patch
#   shim — ms-swift >=4.2.0.dev0 (git main post 2026-04-21) already carries
#   the transformers 5.5 signature fix (PR #9167) and the Qwen3.5 GDN
#   Ulysses SP hook (PR #9162 / #9189). Keep this v2 runner separate from v1
#   so we can diff the two generations side-by-side in the report.
#
# Run after:
#   docker exec fsdp_sft pip install --force-reinstall --no-deps \
#       git+https://github.com/modelscope/ms-swift.git@main
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

export CUDA_DEVICE_MAX_CONNECTIONS=8

: "${MODEL:=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B}"
: "${MODEL_TYPE:=qwen3_5}"
: "${BACKEND:?BACKEND must be ds|fsdp2}"
: "${SP:=2}"
: "${MBS:=1}"
: "${GAS:=1}"
: "${MAX_LEN:=16384}"
: "${TOTAL_STEPS:=30}"
: "${WARMUP_BENCH:=5}"
: "${DS_CONFIG:=${SCRIPT_DIR}/sp_offload_configs/zero3_nopin.json}"
: "${FREEZE_VIT:=true}"
: "${GRAD_CKPT:=auto}"

# --- FSDP2 / compile knobs (only effective when BACKEND=fsdp2) ---
# Default empty preserves the prior behaviour (uses swift's fsdp2.json preset).
: "${TORCH_COMPILE:=false}"          # true → adds --torch_compile true
: "${NO_AC:=false}"                  # true → activation_checkpointing=false
: "${FSDP_RESHARD:=true}"            # false → reshard_after_forward=false
: "${FSDP_WRAP_POLICY:=}"            # TRANSFORMER_BASED_WRAP | SIZE_BASED_WRAP (empty=preset default)
: "${FSDP_MIN_NUM_PARAMS:=}"         # only used when FSDP_WRAP_POLICY=SIZE_BASED_WRAP

# --- Dataset / compute-density knobs ---
: "${PACKING:=false}"                # true → --packing true (pack short samples into seq=MAX_LEN)
: "${USE_LIGER:=false}"              # true → --use_liger_kernel true
: "${DATALOADER_WORKERS:=2}"         # passed as --dataloader_num_workers
: "${DATASET_NUM_PROC:=4}"           # passed as --dataset_num_proc
: "${LAZY_TOKENIZE:=true}"           # false → --lazy_tokenize false (pre-tokenise whole dataset)

: "${BENCH_DIR:=${OUTPUT_ROOT}/bench_sp_offload_v2}"
: "${RUN_NAME:=${BACKEND}_sp${SP}_v2}"
BENCH_OUTPUT="${BENCH_DIR}/${RUN_NAME}"
mkdir -p "${BENCH_OUTPUT}"

GPU_LOG="${BENCH_OUTPUT}/gpu_metrics.jsonl"
DCGM_LOG="${BENCH_OUTPUT}/dcgm_tc.tsv"
TRAIN_LOG="${BENCH_OUTPUT}/train.log"

GBS=$(( MBS * NPROC_PER_NODE * GAS ))

if [ "${GRAD_CKPT}" = "auto" ]; then
    if [ "${BACKEND}" = "fsdp2" ]; then GRAD_CKPT=false; else GRAD_CKPT=true; fi
fi

log "=== swift sft + SP Benchmark (v2 — ms-swift main) ==="
printf '  %-22s = %s\n' \
    swift_version "$(docker exec fsdp_sft python -c 'import swift;print(swift.__version__)' 2>/dev/null || echo '?')" \
    model "${MODEL}" \
    backend "${BACKEND}" \
    SP "${SP}" \
    MBS/GAS/GBS "${MBS}/${GAS}/${GBS}" \
    max_length "${MAX_LEN}" \
    total_steps "${TOTAL_STEPS}" \
    warmup_steps "${WARMUP_BENCH}" \
    grad_ckpt "${GRAD_CKPT}" \
    torch_compile "${TORCH_COMPILE}" \
    no_ac "${NO_AC}" \
    fsdp_reshard "${FSDP_RESHARD}" \
    fsdp_wrap_policy "${FSDP_WRAP_POLICY:-(preset)}" \
    output "${BENCH_OUTPUT}"

python "${SCRIPT_DIR}/gpu_monitor.py" --output "${GPU_LOG}" &
GPU_MON_PID=$!
python "${SCRIPT_DIR}/dcgm_scrape.py" "${DCGM_LOG}" "http://localhost:9500/metrics" &
DCGM_PID=$!
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

BACKEND_FLAGS=()
GRAD_CKPT_FLAGS=()
if [ "${BACKEND}" = "ds" ]; then
    BACKEND_FLAGS+=(--deepspeed "${DS_CONFIG}")
elif [ "${BACKEND}" = "fsdp2" ]; then
    # If any FSDP knob is non-default, materialise an override JSON and pass that as --fsdp
    NEED_OVERRIDE=false
    [ "${NO_AC}" = "true" ] && NEED_OVERRIDE=true
    [ "${FSDP_RESHARD}" = "false" ] && NEED_OVERRIDE=true
    [ -n "${FSDP_WRAP_POLICY}" ] && NEED_OVERRIDE=true

    if [ "${NEED_OVERRIDE}" = "true" ]; then
        FSDP_OVERRIDE="${BENCH_OUTPUT}/fsdp_override.json"
        AC_VAL=true; [ "${NO_AC}" = "true" ] && AC_VAL=false
        RESHARD_VAL=true; [ "${FSDP_RESHARD}" = "false" ] && RESHARD_VAL=false
        WRAP_VAL="${FSDP_WRAP_POLICY:-TRANSFORMER_BASED_WRAP}"
        EXTRA_WRAP=""
        if [ "${WRAP_VAL}" = "SIZE_BASED_WRAP" ] && [ -n "${FSDP_MIN_NUM_PARAMS}" ]; then
            EXTRA_WRAP=", \"min_num_params\": ${FSDP_MIN_NUM_PARAMS}"
        fi
        cat > "${FSDP_OVERRIDE}" <<EOF
{
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": ${RESHARD_VAL},
        "auto_wrap_policy": "${WRAP_VAL}",
        "cpu_ram_efficient_loading": true,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": ${AC_VAL}${EXTRA_WRAP}
    }
}
EOF
        BACKEND_FLAGS+=(--fsdp "${FSDP_OVERRIDE}")
        log "FSDP override written: ${FSDP_OVERRIDE}"
        cat "${FSDP_OVERRIDE}"
    else
        BACKEND_FLAGS+=(--fsdp fsdp2)
    fi
fi

if [ "${GRAD_CKPT}" = "true" ]; then
    GRAD_CKPT_FLAGS+=(--gradient_checkpointing true)
fi

COMPILE_FLAGS=()
if [ "${TORCH_COMPILE}" = "true" ]; then
    COMPILE_FLAGS+=(--torch_compile true)
fi

DENSITY_FLAGS=()
if [ "${PACKING}" = "true" ]; then
    DENSITY_FLAGS+=(--packing true)
fi
if [ "${USE_LIGER}" = "true" ]; then
    DENSITY_FLAGS+=(--use_liger_kernel true)
fi
if [ "${LAZY_TOKENIZE}" = "false" ]; then
    DENSITY_FLAGS+=(--lazy_tokenize false)
fi

FREEZE_FLAGS=()
if [ "${FREEZE_VIT}" = "true" ]; then
    FREEZE_FLAGS+=(--freeze_vit true --freeze_aligner true)
fi

log "Starting training (${TOTAL_STEPS} steps, plain swift sft)..."

NPROC_PER_NODE="${NPROC_PER_NODE}" swift sft \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${TRAIN_JSONL}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --max_length "${MAX_LEN}" \
    --truncation_strategy delete \
    --per_device_train_batch_size "${MBS}" \
    --gradient_accumulation_steps "${GAS}" \
    --max_steps "${TOTAL_STEPS}" \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    "${GRAD_CKPT_FLAGS[@]}" \
    "${COMPILE_FLAGS[@]}" \
    "${DENSITY_FLAGS[@]}" \
    "${FREEZE_FLAGS[@]}" \
    "${BACKEND_FLAGS[@]}" \
    --sequence_parallel_size "${SP}" \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --save_strategy no \
    --logging_steps 1 \
    --output_dir "${BENCH_OUTPUT}" \
    2>&1 | tee "${TRAIN_LOG}"

kill "${GPU_MON_PID}" "${DCGM_PID}" 2>/dev/null || true
wait "${GPU_MON_PID}" 2>/dev/null || true
wait "${DCGM_PID}" 2>/dev/null || true

LATEST_VDIR="$(ls -dt "${BENCH_OUTPUT}"/v*-* 2>/dev/null | head -n 1 || true)"
if [ -z "${LATEST_VDIR}" ] || [ ! -f "${LATEST_VDIR}/logging.jsonl" ]; then
    log "WARN: no logging.jsonl"
    exit 0
fi

python "${SCRIPT_DIR}/report_swift_sp.py" \
    --logging_jsonl "${LATEST_VDIR}/logging.jsonl" \
    --gpu_log "${GPU_LOG}" \
    --warmup_steps "${WARMUP_BENCH}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --gbs "${GBS}" \
    --max_len "${MAX_LEN}" \
    --backend "${BACKEND}" \
    --sp "${SP}" \
    --bench_jsonl_out "${BENCH_OUTPUT}/bench.jsonl" \
    --report_out "${BENCH_OUTPUT}/report.json"
log "Report saved to ${BENCH_OUTPUT}/report.json"
