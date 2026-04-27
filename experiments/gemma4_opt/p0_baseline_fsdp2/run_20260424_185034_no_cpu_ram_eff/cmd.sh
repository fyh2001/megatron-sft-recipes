#!/usr/bin/env bash
# P0b: FSDP2 default baseline for gemma-4-26B-A4B-it
#
# Config semantics (swift default fsdp2 preset + SP=2 + MBS=1):
#   - full_shard (ZeRO-3), auto_wrap by transformer decoder layer
#   - activation_checkpointing = true (swift FSDP2 preset default)
#   - reshard_after_forward = true (ZeRO-3 full)
#   - SP=2 Ulysses (hard cap on gemma4: num_global_key_value_heads=2)
#   - MBS=1, grad_accum=1, DP=4 -> GBS=4
#   - packing=false (raw padded samples)
#   - --use_liger_kernel true (silent no-op on gemma4 until P5 adds dispatch)
#
# Expected wall: ~10 min for 40 steps incl. setup.
# Output: /home/ubuntu/fyh/megatron_output/gemma4_opt/p0_baseline_fsdp2/run_NN_*/
set -euo pipefail

RUN_LABEL="${RUN_LABEL:-first_try}"
OUT_ROOT="/home/ubuntu/fyh/megatron_output/gemma4_opt/p0_baseline_fsdp2"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${RUN_LABEL}"
mkdir -p "${RUN_DIR}"

# Save the exact cmd that produced this run, for reproducibility.
cp "$0" "${RUN_DIR}/cmd.sh"

echo "Run dir: ${RUN_DIR}"

docker exec fsdp_sft bash -lc "cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MASTER_PORT=29555 BACKEND=fsdp2 SP=2 MBS=1 \
  NO_AC=false FSDP_RESHARD=true \
  PACKING=false USE_LIGER=true \
  MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it \
  MODEL_TYPE=gemma4 \
  FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer \
  TOTAL_STEPS=40 WARMUP_BENCH=5 \
  RUN_NAME=run_${RUN_LABEL} \
  BENCH_DIR=${OUT_ROOT} \
  bash scripts/benchmark/bench_swift_sp_v2.sh" 2>&1 | tee "${RUN_DIR}/stdout.log"

EXIT=${PIPESTATUS[0]}

# Post-run bookkeeping
if [ "${EXIT}" = "0" ]; then
    echo "SUCCESS" > "${RUN_DIR}/STATUS"
    # Copy artifacts from bench dir into run dir
    [ -f "${OUT_ROOT}/run_${RUN_LABEL}/report.json" ] && \
        cp "${OUT_ROOT}/run_${RUN_LABEL}/report.json" "${RUN_DIR}/"
    [ -f "${OUT_ROOT}/run_${RUN_LABEL}/dcgm_tc.tsv" ] && \
        cp "${OUT_ROOT}/run_${RUN_LABEL}/dcgm_tc.tsv" "${RUN_DIR}/"
    [ -d "${OUT_ROOT}/run_${RUN_LABEL}" ] && \
        cp -r "${OUT_ROOT}/run_${RUN_LABEL}"/v*-* "${RUN_DIR}/" 2>/dev/null || true
else
    echo "FAILED exit_code=${EXIT}" > "${RUN_DIR}/STATUS"
fi

exit ${EXIT}
