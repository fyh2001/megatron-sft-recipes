#!/usr/bin/env bash
# P2: activation_checkpointing=false at the P1 peak (GBS=4 native).
#
# Hypothesis: with AC=off, every decoder layer's intermediate activations stay
# resident on GPU until backward.  For gemma4-26B (30 layers, hidden=2816,
# seq=16384, MoE intermediate dims), naive estimate is ~3 GB activations per
# layer × 30 = ~90 GB extra above AC=on baseline (P1 was 65 GiB).  Therefore
# native AC=off almost certainly OOMs on 80 GB H100.
#
# Plan:
#   Run 1: GBS=4 native + NO_AC=true  (expect OOM, document the ceiling)
#   Run 2: GBS=4 offload + NO_AC=true  (offload params/grads/master to CPU,
#                                       see if activations alone fit + measure
#                                       step time vs P1 native AC=on)
#
# Locked baseline (from §0.4 walkthrough): truncation=right, mem_eff SDPA,
# GQA repeat_kv, cpu:gloo+cuda:nccl, FA2, USE_LIGER=true silent, freeze_vit.
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p2_no_ac"
mkdir -p "${OUT_ROOT}"
ATTEMPTS_FILE="${OUT_ROOT}/attempts.md"
if [ ! -f "${ATTEMPTS_FILE}" ]; then
    cat > "${ATTEMPTS_FILE}" <<EOF
# P2 NO_AC sweep — attempts timeline

> Goal: measure step-time / peak-mem trade-off when disabling FSDP2's wrap-level activation_checkpointing at the P1 peak (GBS=4 native).
>
> Hypothesis: ~90 GB extra activations from 30 gemma4 decoder layers (no recompute) → OOM on 80 GB without offload.

EOF
fi

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-30}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"

# Two configurations: native first (likely OOM), offload second (always fits)
SWEEP=(
    "native    NO_AC=true                       gbs4_no_ac_native"
    "offload   NO_AC=true FSDP_OFFLOAD=offload  gbs4_no_ac_offload"
)

for entry in "${SWEEP[@]}"; do
    MODE="$(echo "$entry" | awk '{print $1}')"
    LABEL="$(echo "$entry" | awk '{print $NF}')"
    EXTRA_ENV="$(echo "$entry" | sed -e "s/^${MODE} *//" -e "s/ *${LABEL}\$//")"

    RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
    mkdir -p "${RUN_DIR}"
    cp "$0" "${RUN_DIR}/cmd.sh"

    # When offload is on, also use --optim adamw_torch (bench script auto-adds
    # this when FSDP_OFFLOAD env is set).
    EXTRA_ENV_NEEDS_OPTIM=""
    if echo "${EXTRA_ENV}" | grep -q FSDP_OFFLOAD; then
        EXTRA_ENV_NEEDS_OPTIM="(adamw_torch auto-applied)"
    fi

    MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

    echo "=========================================="
    echo "P2 [${LABEL}]: mode=${MODE} ${EXTRA_ENV} ${EXTRA_ENV_NEEDS_OPTIM} port=${MASTER_PORT}"
    echo "Run dir: ${RUN_DIR}"
    echo "=========================================="

    set +e
    docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 MBS=1 GAS=1 \
${EXTRA_ENV} FSDP_RESHARD=true \
PACKING=false USE_LIGER=true \
TRUNCATION_STRATEGY=right \
MODEL=${MODEL} \
MODEL_TYPE=gemma4 \
FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer \
FSDP_CPU_RAM_EFFICIENT=false \
NUM_PARAMS=25.2e9 NUM_ACTIVE_PARAMS=3.8e9 \
DATASET_SIZE=18819 \
TOTAL_STEPS=${TOTAL_STEPS} WARMUP_BENCH=${WARMUP_BENCH} \
RUN_NAME=run_${LABEL} \
BENCH_DIR=${OUT_ROOT}/_bench \
bash scripts/benchmark/bench_swift_sp_v2.sh
" 2>&1 | tee "${RUN_DIR}/stdout.log"
    EXIT=${PIPESTATUS[0]}
    set -e

    if [ "${EXIT}" = "0" ]; then
        STATUS="SUCCESS"
        SUMMARY="see report.json"
    else
        if grep -q "CUDA out of memory" "${RUN_DIR}/stdout.log" 2>/dev/null; then
            STATUS="FAILED"
            SUMMARY="OOM ($(grep -oE 'Tried to allocate [0-9.]+ [GM]iB.* free' "${RUN_DIR}/stdout.log" | head -1))"
        else
            STATUS="FAILED"
            SUMMARY="exit=${EXIT}"
        fi
    fi
    echo "${STATUS} — ${SUMMARY}" > "${RUN_DIR}/STATUS"

    BENCH_OUT="${OUT_ROOT}/_bench/run_${LABEL}"
    if [ -d "${BENCH_OUT}" ]; then
        for f in report.json bench.jsonl dcgm_tc.tsv fsdp_override.json gpu_metrics.jsonl; do
            [ -f "${BENCH_OUT}/${f}" ] && cp "${BENCH_OUT}/${f}" "${RUN_DIR}/" || true
        done
        VDIR=$(ls -dt ${BENCH_OUT}/v*-* 2>/dev/null | head -n 1 || true)
        if [ -n "${VDIR}" ] && [ -f "${VDIR}/logging.jsonl" ]; then
            ln -sf "${VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
        fi
    fi

    printf "| run_%s | %s mode | %s | %s |\n" \
        "$(basename ${RUN_DIR})" "${MODE}" "${STATUS}" "${SUMMARY}" >> "${ATTEMPTS_FILE}"

    if [ -f "${RUN_DIR}/report.json" ]; then
        echo "    summary: $(python3 -c "
import json
d = json.load(open('${RUN_DIR}/report.json'))
print(f\"step={d.get('mean_step_time_ms', '?')}ms tokens/s/gpu={d.get('tokens_per_sec_per_gpu', '?')} peak_mem={d.get('peak_mem_gib_from_swift_log', '?')}GiB MFU={d.get('mfu_pct_active_params', '?')}%\")
" 2>/dev/null)"
    fi

    # Aggressive cleanup before next iteration
    docker exec fsdp_sft pkill -9 python 2>/dev/null || true
    set +e
    for i in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{gsub(/ /,""); print}')
        if [ -n "${USED}" ] && [ "${USED}" -lt 100 ] 2>/dev/null; then break; fi
    done
    set -e
    sleep 2
done

echo "=========================================="
echo "P2 NO_AC complete. Run dirs: ${OUT_ROOT}/run_*"
echo "=========================================="
