#!/usr/bin/env bash
# P5: gemma4 Liger dispatch at the P4 peak (GBS=4 native + AC=off + packing).
#
# What changes from P4:
#   - Liger Kernel now actually fires for gemma4 (sitecustomize patch 5
#     registers `gemma4 → _apply_liger_kernel_to_gemma4` in Liger's
#     MODEL_TYPE_TO_APPLY_LIGER_FN).  Replaces:
#       * Gemma4RMSNorm → LigerRMSNorm (offset=0, casting=gemma, no in_place)
#         — fuses 4-7 RMSNorm calls per layer × 30 layers = 120-210 RMSNorms/forward
#       * Gemma4TextMLP.forward → LigerGEGLUMLP.forward
#         — fuses gate_proj·act·up_proj into Triton kernel
#       * Gemma4ForConditionalGeneration.forward → gemma3-style causal_forward
#         using LigerForCausalLMLoss — fused lm_head + CE, no [B,N,V] logits
#         materialized.  Replaces the chunked CE workaround in modeling_gemma4
#         patch (which still materializes [B,N,V] bf16 = 8.6 GiB).
#
# Expected (vs P4):
#   - step time: -5~10% (fewer elementwise kernels for RMSNorm, fused GeGLU)
#   - peak mem: lower (no 8.6 GiB logits tensor + no chunked-CE float upcast)
#   - real MFU: +2~4 pp
#   - loss: bit-comparable to P4 (FLCE math equivalent to chunked CE)
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron_output/gemma4_opt/p5_liger"
mkdir -p "${OUT_ROOT}"
ATTEMPTS_FILE="${OUT_ROOT}/attempts.md"
if [ ! -f "${ATTEMPTS_FILE}" ]; then
    cat > "${ATTEMPTS_FILE}" <<EOF
# P5 Liger gemma4 dispatch — attempts timeline

> Goal: write Liger gemma4 dispatch (RMSNorm + GeGLU + fused linear-CE) and
> measure step / mem / MFU vs P4 (no Liger).  Loss should be bit-comparable.
> Pre-recon: gemma4 RMSNorm = output*weight (offset=0), MLP = GeGLU.

EOF
fi

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-30}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"

LABEL="gbs4_pack_no_ac_native_liger"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"
MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

echo "=========================================="
echo "P5 [${LABEL}]: P4 config + Liger gemma4 dispatch · port=${MASTER_PORT}"
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
echo "P5 Liger complete. Run dir: ${RUN_DIR}"
echo "=========================================="
