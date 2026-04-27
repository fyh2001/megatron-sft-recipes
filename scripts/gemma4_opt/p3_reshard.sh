#!/usr/bin/env bash
# P3: reshard_after_forward=false (ZeRO-2 mode) at the P2 peak (GBS=4 native + AC=off).
#
# Hypothesis: with reshard_after_forward=false, each layer's params stay
# unsharded after forward → backward doesn't need to all_gather again →
# fewer NCCL ops, faster step.  Cost: at peak (end of forward, start of
# backward), ALL 30 layers' params are unsharded simultaneously on each rank
# → 26B × 2 bytes = ~52 GB on each rank (vs 6.5 GB sharded).
#
# Plan:
#   Run 1: native + reshard=false  (almost certainly OOM, ~52 GB params alone
#                                   plus activations + opt + ... > 80 GB)
#   Run 2: offload + reshard=false  (offload still keeps unsharded view on GPU
#                                    during fwd+bwd window; uncertain if fits)
#
# Locked baseline: GBS=4 native, MBS=1 GAS=1, NO_AC=true, truncation=right,
# mem_eff SDPA + GQA repeat_kv, FA2.
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p3_reshard"
mkdir -p "${OUT_ROOT}"
ATTEMPTS_FILE="${OUT_ROOT}/attempts.md"
if [ ! -f "${ATTEMPTS_FILE}" ]; then
    cat > "${ATTEMPTS_FILE}" <<EOF
# P3 reshard_after_forward=false sweep — attempts timeline

> Goal: measure step-time / peak-mem trade-off when switching FSDP2 from
> ZeRO-3 (reshard=true) to ZeRO-2 (reshard=false) at the P2 peak.
>
> Hypothesis: native OOMs (full model unsharded = 52 GB / rank > 80 GB ceiling
> after activations + opt state); offload may or may not fit.

EOF
fi

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-30}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"

SWEEP=(
    "native    NO_AC=true FSDP_RESHARD=false                         gbs4_no_ac_zero2_native"
    "offload   NO_AC=true FSDP_RESHARD=false FSDP_OFFLOAD=offload    gbs4_no_ac_zero2_offload"
)

for entry in "${SWEEP[@]}"; do
    MODE="$(echo "$entry" | awk '{print $1}')"
    LABEL="$(echo "$entry" | awk '{print $NF}')"
    EXTRA_ENV="$(echo "$entry" | sed -e "s/^${MODE} *//" -e "s/ *${LABEL}\$//")"

    RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
    mkdir -p "${RUN_DIR}"
    cp "$0" "${RUN_DIR}/cmd.sh"
    MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

    echo "=========================================="
    echo "P3 [${LABEL}]: mode=${MODE} ${EXTRA_ENV} port=${MASTER_PORT}"
    echo "Run dir: ${RUN_DIR}"
    echo "=========================================="

    set +e
    docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 MBS=1 GAS=1 \
${EXTRA_ENV} \
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
echo "P3 reshard sweep complete. Run dirs: ${OUT_ROOT}/run_*"
echo "=========================================="
