#!/usr/bin/env bash
# P4: --packing true at the P2 peak (GBS=4 native + AC=off).
#
# Packing concatenates multiple short samples up to max_length=16384 → every
# micro is dense-packed, eliminating padding waste.  Real-token density:
# avg 2823 tok / 16384 = 17% → packing gives ~5.8× density.  Expected step
# time stays similar, so tokens/s/GPU should jump ~5×.
#
# Memory consideration: with packing each micro IS at 16k, so per-layer
# activation footprint scales 5.8× vs P2's variable-length avg 2823.  P2 was
# 64.94 GiB peak.  Estimate +20-50 GiB → 85-115 GiB → likely needs AC=on
# or offload.
#
# Plan (escalate on OOM):
#   Run 1: GBS=4 native + AC=off + packing=true  (best case if it fits)
#   Run 2: GBS=4 native + AC=on  + packing=true  (fallback when AC=off OOMs)
#   Run 3: GBS=4 offload + AC=on + packing=true  (fallback when both above OOM)
#
# Locked baseline: MBS=1 GAS=1, mem_eff SDPA + GQA repeat_kv, FA2,
# truncation=right, USE_LIGER=true (silent until P5).
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron_output/gemma4_opt/p4_packing"
mkdir -p "${OUT_ROOT}"
ATTEMPTS_FILE="${OUT_ROOT}/attempts.md"
if [ ! -f "${ATTEMPTS_FILE}" ]; then
    cat > "${ATTEMPTS_FILE}" <<EOF
# P4 packing sweep — attempts timeline

> Goal: enable --packing true at the P2 peak. Expected ~5× tokens/s/GPU.
> Auto-escalate AC=off → AC=on → offload on OOM.

EOF
fi

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-30}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"

# Each entry: extra-env LABEL
SWEEP=(
    "NO_AC=true                                gbs4_pack_no_ac_native"
    "NO_AC=false                               gbs4_pack_ac_on_native"
    "NO_AC=false FSDP_OFFLOAD=offload          gbs4_pack_ac_on_offload"
)

for entry in "${SWEEP[@]}"; do
    LABEL="$(echo "$entry" | awk '{print $NF}')"
    EXTRA_ENV="$(echo "$entry" | sed -e "s/ *${LABEL}\$//")"

    RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
    mkdir -p "${RUN_DIR}"
    cp "$0" "${RUN_DIR}/cmd.sh"
    MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

    echo "=========================================="
    echo "P4 [${LABEL}]: ${EXTRA_ENV} PACKING=true port=${MASTER_PORT}"
    echo "Run dir: ${RUN_DIR}"
    echo "=========================================="

    set +e
    docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 MBS=1 GAS=1 \
${EXTRA_ENV} FSDP_RESHARD=true \
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
        # Stop escalation as soon as we get a SUCCESS
        echo "P4 [${LABEL}] succeeded — stopping further escalation."
        docker exec fsdp_sft pkill -9 python 2>/dev/null || true
        sleep 3
        break
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
echo "P4 packing complete. Run dirs: ${OUT_ROOT}/run_*"
echo "=========================================="
