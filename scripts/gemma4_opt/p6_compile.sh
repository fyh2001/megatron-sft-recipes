#!/usr/bin/env bash
# P6: torch.compile at the P5 peak (GBS=4 native + AC=off + packing + Liger).
#
# Plan §6 prediction: compile 5-15% step time reduction (Inductor fusion).
# Risk per qwen3.5 walkthrough §7.1: torch 2.10 + Inductor + TF32 API
# interaction has historical bugs that break compile mid-warmup.
#
# Single point smoke test, 30 min wall cap.  If compile errors out or fails
# warmup, document and skip to P7 MoE tuning.
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p6_compile"
mkdir -p "${OUT_ROOT}"
ATTEMPTS_FILE="${OUT_ROOT}/attempts.md"
if [ ! -f "${ATTEMPTS_FILE}" ]; then
    cat > "${ATTEMPTS_FILE}" <<EOF
# P6 torch.compile sweep — attempts timeline

> Goal: smoke test torch.compile at P5 peak.  Predicted +5-15% step time.
> Known risk: torch 2.10 + Inductor + TF32 API bug from qwen3.5 era may break
> compile.  If broken, document & skip to P7.

EOF
fi

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-30}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"

LABEL="gbs4_pack_no_ac_native_liger_compile"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"
MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

echo "=========================================="
echo "P6 [${LABEL}]: P5 + torch.compile · port=${MASTER_PORT}"
echo "Run dir: ${RUN_DIR}"
echo "=========================================="

set +e
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 MBS=1 GAS=1 \
NO_AC=true FSDP_RESHARD=true \
PACKING=true USE_LIGER=true \
TORCH_COMPILE=true \
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
    elif grep -qE "torch._dynamo|Inductor|recompile" "${RUN_DIR}/stdout.log" 2>/dev/null; then
        STATUS="FAILED"
        SUMMARY="compile error: $(grep -oE 'torch._dynamo[^\"]+|Inductor[^\"]+' "${RUN_DIR}/stdout.log" | head -1 | head -c 200)"
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

printf "| run_%s | %s | %s |\n" \
    "$(basename ${RUN_DIR})" "${STATUS}" "${SUMMARY}" >> "${ATTEMPTS_FILE}"

if [ -f "${RUN_DIR}/report.json" ]; then
    echo "    summary: $(python3 -c "
import json
d = json.load(open('${RUN_DIR}/report.json'))
print(f\"step={d.get('mean_step_time_ms', '?')}ms tokens/s/gpu={d.get('tokens_per_sec_per_gpu', '?')} peak_mem={d.get('peak_mem_gib_from_swift_log', '?')}GiB MFU={d.get('mfu_pct_active_params', '?')}%\")
" 2>/dev/null)"
fi

docker exec fsdp_sft pkill -9 python 2>/dev/null || true
echo "=========================================="
echo "P6 torch.compile complete. Run dir: ${RUN_DIR}"
echo "=========================================="
