#!/usr/bin/env bash
# P1 alt-offload (DS-equivalent) — single-config bench
#
# Goal: reproduce the DeepSpeed prod-baseline TOPOLOGY on FSDP2 and report
# every performance metric we collect so this run can be used as a stable
# operations baseline ("DS-equivalent on FSDP2").
#
# Why this config:
#   - DS prod (zero3 + offload(opt+param) + SP=2 + GBS=64 + AC=on) is the
#     online recipe we have to match for memory budget.  P1 native (GBS=4,
#     no offload) is faster but uses a different effective GBS — that
#     changes the optimizer trajectory and therefore loss curve.  This run
#     keeps GBS=64 / AC=on / SP=2 identical to DS so the only swap is the
#     parallelism engine (DS ZeRO-3 → FSDP2 + CPU offload).
#
# Configuration locked here (all aligned to gemma4_sft_0423.log production cmd):
#   - MBS=1 / GAS=16 → GBS = MBS × DP × GAS = 1 × 4 × 16 = 64
#     (SP=2 → DP=4 since NPROC=8 / SP=2 = 4)
#   - activation_checkpointing=true (AC=on, matches DS --gradient_checkpointing true)
#   - FSDP CPU offload (opt + grad + param all offloaded; matches DS
#     offload_optimizer + offload_param)
#   - reshard_after_forward=true (FSDP2 default)
#   - --truncation_strategy right (matches DS prod)
#   - --learning_rate 2e-5 --warmup_ratio 0.05 (matches DS prod, NOT bench
#     defaults of 1e-5/0.1; lr/warmup don't affect throughput but here we
#     want byte-for-byte topology parity)
#   - --use_liger_kernel true (silent no-op for gemma4 unless liger
#     dispatch is registered via sitecustomize; we leave it on so config
#     diff vs DS prod is purely "engine swap")
#   - --freeze_vit true --freeze_aligner true
#   - --attn_impl flash_attention_2 (modeling_gemma4 patch falls back
#     to mem_efficient SDPA on global head_dim=512 layers)
#   - max_length=16384, dtype=bfloat16
#
# Bench: 40 optimizer steps (5 warmup + 35 measured) — matches DS prod
# baseline's 40-step bench window so wall/throughput numbers are directly
# diff-able row-for-row.
#
# Output: experiments/gemma4_opt/p1_alt_offload/run_<TIMESTAMP>/
#   ├── cmd.sh                   (this script)
#   ├── stdout.log               (full bench wrapper output)
#   ├── STATUS                   (SUCCESS — see report.json / FAILED — reason)
#   ├── report.json              (aggregate metrics from report_swift_sp.py)
#   ├── bench.jsonl              (per-step metrics)
#   ├── dcgm_tc.tsv              (DCGM TC metrics, per ~5s)
#   ├── gpu_metrics.jsonl        (per-GPU util/mem/power, per ~1s)
#   ├── fsdp_override.json       (the actual FSDP JSON used)
#   └── logging.jsonl            (symlink into v*-* dir, raw swift trainer log)
#
# Wall time: ~30 min (40 step × ~45 s).
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p1_alt_offload"
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:-ds_equiv}"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-40}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"
MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

# Bench scaffold writes its own files into BENCH_DIR/RUN_NAME, then we
# copy the artefacts into RUN_DIR.  This mirrors p1_gbs_sweep.sh behaviour.
BENCH_DIR="${OUT_ROOT}/_bench"
RUN_NAME="run_alt_offload_ds_equiv"

echo "=========================================="
echo "P1 alt-offload (DS-equivalent on FSDP2)"
echo "Run dir: ${RUN_DIR}"
echo "Bench  : ${BENCH_DIR}/${RUN_NAME}"
echo "Steps  : ${TOTAL_STEPS} (${WARMUP_BENCH} warmup + $((TOTAL_STEPS - WARMUP_BENCH)) measured)"
echo "Port   : ${MASTER_PORT}"
echo "=========================================="

# Disable set -e while running so we capture exit code & write STATUS.
set +e
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 MBS=1 GAS=16 \
NO_AC=false FSDP_RESHARD=true \
PACKING=false USE_LIGER=true \
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

# Categorise outcome
if [ "${EXIT}" = "0" ]; then
    STATUS="SUCCESS"
    SUMMARY="see report.json"
else
    if grep -q "CUDA out of memory" "${RUN_DIR}/stdout.log" 2>/dev/null; then
        STATUS="FAILED"
        SUMMARY="OOM"
    else
        STATUS="FAILED"
        SUMMARY="exit=${EXIT}"
    fi
fi
echo "${STATUS} — ${SUMMARY}" > "${RUN_DIR}/STATUS"

# Copy bench artefacts into self-contained run dir
BENCH_OUT="${BENCH_DIR}/${RUN_NAME}"
if [ -d "${BENCH_OUT}" ]; then
    for f in report.json bench.jsonl dcgm_tc.tsv fsdp_override.json gpu_metrics.jsonl train.log; do
        [ -f "${BENCH_OUT}/${f}" ] && cp "${BENCH_OUT}/${f}" "${RUN_DIR}/" || true
    done
    VDIR=$(ls -dt ${BENCH_OUT}/v*-* 2>/dev/null | head -n 1 || true)
    if [ -n "${VDIR}" ] && [ -f "${VDIR}/logging.jsonl" ]; then
        ln -sf "${VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
    fi
fi

# Show final report.json summary
if [ -f "${RUN_DIR}/report.json" ]; then
    echo "=========================================="
    echo "Final summary"
    echo "=========================================="
    python3 -c "
import json
d = json.load(open('${RUN_DIR}/report.json'))
keys = ['mean_step_time_ms','median_step_time_ms','p99_step_time_ms','micro_step_time_ms',
        'tokens_per_sec_per_gpu','achieved_tflops_per_gpu','achieved_tflops_per_gpu_active',
        'mfu_pct','mfu_pct_active_params',
        'peak_mem_gb','peak_mem_gib_from_swift_log',
        'avg_gpu_util_pct','avg_power_w','peak_power_w',
        'actual_total_wall_min','steps_per_epoch_for_dataset',
        'loss_first_step','loss_last_step']
for k in keys:
    print(f'  {k:32s} = {d.get(k)}')
"
fi

# Cleanup
docker exec fsdp_sft pkill -9 python 2>/dev/null || true

exit ${EXIT}
