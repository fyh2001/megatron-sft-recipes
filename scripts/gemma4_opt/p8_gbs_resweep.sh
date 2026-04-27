#!/usr/bin/env bash
# P8: GBS resweep on the optimized stack (P5 peak: native + AC=off + packing + Liger).
#
# P1 found GBS=4 native = peak when stack was bare (no mem_eff SDPA, no packing,
# no Liger).  After P2-P5 wins, recompute peak on the full optimized stack.
#
# Sweep matrix (3 points around P5 peak):
#   - GBS=4   native (P5 peak, reference)
#   - GBS=8   offload (MBS=1 GAS=2; tests if 2× GBS density beats offload IO penalty)
#   - GBS=64  offload (MBS=1 GAS=16; matches DS prod GBS, max sample-based GBS)
#
# MBS=2 paths skipped — still blocked by swift Ulysses VLM dict-attention_mask
# bug (P1 attempts.md), packing won't fix that (it changes mask shape but not
# the dict-typing).
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p8_gbs_resweep"
mkdir -p "${OUT_ROOT}"
ATTEMPTS_FILE="${OUT_ROOT}/attempts.md"
if [ ! -f "${ATTEMPTS_FILE}" ]; then
    cat > "${ATTEMPTS_FILE}" <<EOF
# P8 GBS resweep — attempts timeline

> Goal: verify P5 peak (GBS=4 native + AC=off + packing + Liger) is still
> peak after the full optimized stack is in place.

EOF
fi

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-30}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"

# Each entry: extra-env LABEL
SWEEP=(
    "MBS=1 GAS=1                                 gbs4_native_full_stack"
    "MBS=1 GAS=2 FSDP_OFFLOAD=offload            gbs8_offload_full_stack"
    "MBS=1 GAS=16 FSDP_OFFLOAD=offload           gbs64_offload_full_stack"
)

for entry in "${SWEEP[@]}"; do
    LABEL="$(echo "$entry" | awk '{print $NF}')"
    EXTRA_ENV="$(echo "$entry" | sed -e "s/ *${LABEL}\$//")"

    RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
    mkdir -p "${RUN_DIR}"
    cp "$0" "${RUN_DIR}/cmd.sh"
    MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

    echo "=========================================="
    echo "P8 [${LABEL}]: ${EXTRA_ENV} packing=true Liger·on AC=off · port=${MASTER_PORT}"
    echo "Run dir: ${RUN_DIR}"
    echo "=========================================="

    set +e
    docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 \
${EXTRA_ENV} \
NO_AC=true FSDP_RESHARD=true \
PACKING=true USE_LIGER=true \
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

    printf "| run_%s | %s | %s | %s |\n" \
        "$(basename ${RUN_DIR})" "${EXTRA_ENV}" "${STATUS}" "${SUMMARY}" >> "${ATTEMPTS_FILE}"

    if [ -f "${RUN_DIR}/report.json" ]; then
        echo "    summary: $(python3 -c "
import json
d = json.load(open('${RUN_DIR}/report.json'))
print(f\"step={d.get('mean_step_time_ms', '?')}ms tokens/s/gpu={d.get('tokens_per_sec_per_gpu', '?')} peak_mem={d.get('peak_mem_gib_from_swift_log', '?')}GiB MFU={d.get('mfu_pct_active_params', '?')}%\")
" 2>/dev/null)"
    fi

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
echo "P8 GBS resweep complete. Run dirs: ${OUT_ROOT}/run_*"
echo "=========================================="
