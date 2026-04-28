#!/usr/bin/env bash
# P6b: torch.compile smoke on the alt-offload stack (GBS=64, AC=on, offload).
#
# History context:
#   - P6 v1: GBS=4 + AC=off + packing + Liger + compile → InductorError TF32 (mitigated)
#   - P6 v2: same + TF32 fix → InductorError Triton codegen numerical assertion
#     (suspected Liger triton kernel × Inductor triton codegen conflict)
#
# Hypothesis: Liger triton + Inductor triton at the same time is the breaker.
# Test by isolating compile alone on the alt-offload stack:
#   - GBS=64, MBS=1, GAS=16 (alt-offload baseline, just verified 45s/step)
#   - AC=on (matches DS prod)
#   - offload=on (matches DS prod)
#   - packing=false (matches DS prod)
#   - **USE_LIGER=false** (key change vs P6: kill Liger to remove triton conflict)
#   - TORCH_COMPILE=true
#
# Smoke goal: 10 steps to see if compile finishes warmup + first few measured.
# If it works: extend, then add Liger back to see if THAT is the breaker.
# If it fails: capture error, decide next mitigation.
#
# Wall budget: ~12 min (10 steps × ~45s + compile cold start ~2-3min).
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p6b_compile_alt_offload"
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:-no_liger}"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-10}"
WARMUP_BENCH="${WARMUP_BENCH:-3}"
USE_LIGER_FLAG="${USE_LIGER_FLAG:-false}"
MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

BENCH_DIR="${OUT_ROOT}/_bench"
RUN_NAME="run_compile_${LABEL}"

echo "=========================================="
echo "P6b compile smoke on alt-offload"
echo "Run dir: ${RUN_DIR}"
echo "USE_LIGER=${USE_LIGER_FLAG}, TORCH_COMPILE=true"
echo "Steps: ${TOTAL_STEPS} (${WARMUP_BENCH} warmup)"
echo "=========================================="

set +e
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 MBS=1 GAS=16 \
NO_AC=false FSDP_RESHARD=true \
PACKING=false USE_LIGER=${USE_LIGER_FLAG} \
TORCH_COMPILE=true \
TRUNCATION_STRATEGY=right \
FSDP_OFFLOAD=offload \
MODEL=${MODEL} \
MODEL_TYPE=gemma4 \
FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer \
FSDP_CPU_RAM_EFFICIENT=false \
NUM_PARAMS=25.2e9 NUM_ACTIVE_PARAMS=3.8e9 \
DATASET_SIZE=18819 \
TOTAL_STEPS=${TOTAL_STEPS} WARMUP_BENCH=${WARMUP_BENCH} \
RUN_NAME=${RUN_NAME} \
BENCH_DIR=${BENCH_DIR} \
bash scripts/benchmark/bench_swift_sp_v2.sh
" 2>&1 | tee "${RUN_DIR}/stdout.log"
EXIT=${PIPESTATUS[0]}
set -e

if [ "${EXIT}" = "0" ]; then
    STATUS="SUCCESS"
    SUMMARY="see report.json"
else
    if grep -q "CUDA out of memory" "${RUN_DIR}/stdout.log" 2>/dev/null; then
        STATUS="FAILED"; SUMMARY="OOM"
    elif grep -qE "TF32|legacy" "${RUN_DIR}/stdout.log" 2>/dev/null; then
        STATUS="FAILED"; SUMMARY="TF32 InductorError"
    elif grep -qE "AssertionError.*[0-9]+/[0-9]+" "${RUN_DIR}/stdout.log" 2>/dev/null; then
        STATUS="FAILED"; SUMMARY="Triton codegen AssertionError"
    elif grep -qE "torch._dynamo|Inductor|recompile" "${RUN_DIR}/stdout.log" 2>/dev/null; then
        STATUS="FAILED"; SUMMARY="$(grep -oE 'torch._dynamo[^\"]+|Inductor[^\"]+|InductorError[^\"]+' "${RUN_DIR}/stdout.log" | head -1 | head -c 200)"
    else
        STATUS="FAILED"; SUMMARY="exit=${EXIT}"
    fi
fi
echo "${STATUS} — ${SUMMARY}" > "${RUN_DIR}/STATUS"
echo "STATUS: ${STATUS} — ${SUMMARY}"

BENCH_OUT="${BENCH_DIR}/${RUN_NAME}"
if [ -d "${BENCH_OUT}" ]; then
    for f in report.json bench.jsonl dcgm_tc.tsv fsdp_override.json gpu_metrics.jsonl; do
        [ -f "${BENCH_OUT}/${f}" ] && cp "${BENCH_OUT}/${f}" "${RUN_DIR}/" || true
    done
    VDIR=$(ls -dt ${BENCH_OUT}/v*-* 2>/dev/null | head -n 1 || true)
    if [ -n "${VDIR}" ] && [ -f "${VDIR}/logging.jsonl" ]; then
        ln -sf "${VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
    fi
fi

if [ -f "${RUN_DIR}/report.json" ]; then
    echo ""
    echo "=== Summary ==="
    python3 -c "
import json
d = json.load(open('${RUN_DIR}/report.json'))
for k in ['mean_step_time_ms','median_step_time_ms','peak_mem_gib_from_swift_log','tokens_per_sec_per_gpu','mfu_pct_active_params','peak_power_w','actual_steps_completed']:
    print(f'  {k:32s} = {d.get(k)}')
"
fi

docker exec fsdp_sft pkill -9 python 2>/dev/null || true
exit ${EXIT}
