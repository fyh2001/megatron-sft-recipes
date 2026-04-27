#!/usr/bin/env bash
# P1: GBS/MBS 二维扫盘 (FSDP2 baseline preset + P0g essential patches).
#
# Goal: sweep MBS × GAS to find peak throughput GBS for FSDP2 to fix for P2-P7.
#
# Baseline (locked across the sweep, only MBS / GAS vary):
#   - swift FSDP2 preset: full_shard auto_wrap, AC=on (fsdp_config), reshard=true
#   - Ulysses SP=2 (gemma4 hard cap; num_global_kv_heads=2)
#   - --truncation_strategy delete (P0b default; resamples >max_length samples)
#   - --learning_rate 1e-5 --warmup_ratio 0.1 (P0b lr schedule)
#   - --use_liger_kernel true (silent no-op until P5)
#   - --freeze_vit true --freeze_aligner true (matches P0b; saves trainable params)
#   - --attn_impl flash_attention_2 (bench_swift_sp_v2.sh default)
#   - max_length=16384, dtype=bfloat16
#   - bench: 30 steps (5 warmup + 25 measure)
#
# P0g essential patches (always on, zero numeric cost):
#   - sitecustomize patch 1: SDPA mem_efficient backend forced (saves 8 GB
#     transient on global head_dim=512 layers, no-cost vs default)
#   - sitecustomize patch 2: use_gqa_in_sdpa→False (forces repeat_kv expansion,
#     unblocks mem_efficient backend with GQA)
# NOT loaded: sitecustomize patch 3 (mixed gloo backend) — only needed when
# CPU offload is on, which we DON'T enable here (we want native FSDP2 perf).
#
# Sweep matrix (7 points, ~5 min wall each):
#   MBS  GAS  GBS  notes
#   1    1    4    P0b baseline equivalent (with mem_eff patches)
#   1    2    8    GAS scale (no peak-mem change)
#   1    4    16   Qwen3.5 reference value
#   1    8    32   larger GBS
#   1    16   64   match DS production (P0g run_06 OOM'd by ~3 GB without offload)
#   2    1    8    MBS scale (activations ~2×; may OOM for global SDPA matrix)
#   2    2    16  ditto
#
# Note: we do NOT enable fsdp 'offload' here. The whole point of P1 is finding
# native FSDP2 peak throughput. Points that OOM are documented as "blocked
# without offload"; if a candidate peak GBS turns out to need offload, P2 onwards
# decides whether to accept the slowdown.
#
# Output: megatron_output/gemma4_opt/p1_gbs_sweep/run_<TIMESTAMP>_<LABEL>/
#   ├── cmd.sh              (this script)
#   ├── stdout.log          (full bench output)
#   ├── STATUS              (SUCCESS/FAILED with reason)
#   ├── report.json         (from bench wrapper, only if SUCCESS)
#   ├── bench.jsonl
#   ├── dcgm_tc.tsv
#   ├── fsdp_override.json
#   └── logging.jsonl       (symlink into v*-* dir)
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron_output/gemma4_opt/p1_gbs_sweep"
mkdir -p "${OUT_ROOT}"
ATTEMPTS_FILE="${OUT_ROOT}/attempts.md"
if [ ! -f "${ATTEMPTS_FILE}" ]; then
    cat > "${ATTEMPTS_FILE}" <<EOF
# P1 GBS/MBS sweep — attempts timeline

> Goal: find peak FSDP2 throughput by sweeping MBS × GAS at fixed baseline preset (truncation=delete, lr=1e-5, warmup=0.1, freeze_vit=true, USE_LIGER=true, no offload, mem_efficient SDPA + GQA repeat_kv patches always on).
>
> Each row: \`run_dir · MBS / GAS / GBS · STATUS · summary\`

EOF
fi

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
TOTAL_STEPS="${TOTAL_STEPS:-30}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"

# Sweep matrix: each row = "MBS GAS LABEL"
SWEEP=(
    "1  1   mbs1_ga1_gbs4"
    "1  2   mbs1_ga2_gbs8"
    "1  4   mbs1_ga4_gbs16"
    "1  8   mbs1_ga8_gbs32"
    "1  16  mbs1_ga16_gbs64"
    "2  1   mbs2_ga1_gbs8"
    "2  2   mbs2_ga2_gbs16"
)

# If user passed a specific config in args (e.g. for smoke), only run that
if [ "$#" -gt 0 ]; then
    case "$1" in
        --only)
            shift
            SWEEP=("$@")
            ;;
        --smoke)
            SWEEP=("1  1   mbs1_ga1_gbs4")
            ;;
    esac
fi

for entry in "${SWEEP[@]}"; do
    read -r MBS GAS LABEL <<< "${entry}"
    GBS=$(( MBS * 4 * GAS ))   # SP=2 → DP=4 → GBS = MBS × DP × GAS
    RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
    mkdir -p "${RUN_DIR}"
    cp "$0" "${RUN_DIR}/cmd.sh"

    # Auto-enable CPU offload for GAS>=2.  Empirical: PyTorch FSDP2 no_sync
    # mode (used during gradient accumulation) holds full unsharded grads on
    # each rank between micros (only reduce-scatters at last micro).  For
    # 26B/8 ranks that's ~+45 GiB above the GAS=1 baseline 65 GiB → OOM at
    # GAS=2 even with GBS=8.  CPU offload migrates these to host RAM at the
    # cost of D2H/H2D throughput.  For GAS=1 we stay native (peak ~65 GiB).
    # Verified: GAS=2 GBS=8 without offload → OOM at 78.8 GiB peak (run
    # mbs1_ga2_gbs8 abandoned).
    if [ "${GAS}" -ge 2 ]; then
        OFFLOAD_FLAG="offload"
        BENCH_LABEL_SUFFIX="_offload"
    else
        OFFLOAD_FLAG=""
        BENCH_LABEL_SUFFIX=""
    fi

    # Pick a unique MASTER_PORT per iteration to avoid EADDRINUSE from prior
    # torchrun's TCPStore lingering in TIME_WAIT.
    MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

    echo "=========================================="
    echo "P1 sweep [${LABEL}]: MBS=${MBS} GAS=${GAS} → GBS=${GBS}  offload=${OFFLOAD_FLAG:-none}  port=${MASTER_PORT}"
    echo "Run dir: ${RUN_DIR}"
    echo "=========================================="

    # Disable set -e so an OOM doesn't abort the entire sweep; capture exit code
    set +e
    docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=${MASTER_PORT} BACKEND=fsdp2 SP=2 MBS=${MBS} GAS=${GAS} \
NO_AC=${NO_AC:-true} FSDP_RESHARD=true \
PACKING=false USE_LIGER=true \
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-right} \
FSDP_OFFLOAD=${OFFLOAD_FLAG} \
MODEL=${MODEL} \
MODEL_TYPE=gemma4 \
FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer \
FSDP_CPU_RAM_EFFICIENT=false \
NUM_PARAMS=25.2e9 NUM_ACTIVE_PARAMS=3.8e9 \
DATASET_SIZE=18819 \
TOTAL_STEPS=${TOTAL_STEPS} WARMUP_BENCH=${WARMUP_BENCH} \
RUN_NAME=run_${LABEL}${BENCH_LABEL_SUFFIX} \
BENCH_DIR=${OUT_ROOT}/_bench \
bash scripts/benchmark/bench_swift_sp_v2.sh
" 2>&1 | tee "${RUN_DIR}/stdout.log"
    EXIT=${PIPESTATUS[0]}
    set -e

    # Categorise outcome
    if [ "${EXIT}" = "0" ]; then
        STATUS="SUCCESS"
        SUMMARY="see report.json"
    else
        # Try to detect OOM specifically
        if grep -q "CUDA out of memory" "${RUN_DIR}/stdout.log" 2>/dev/null; then
            STATUS="FAILED"
            SUMMARY="OOM ($(grep -oE 'GPU [0-9]+ has a total capacity of [0-9.]+ GiB of which [0-9.]+ [GM]iB is free' "${RUN_DIR}/stdout.log" | head -1))"
        else
            STATUS="FAILED"
            SUMMARY="exit=${EXIT}"
        fi
    fi
    echo "${STATUS} — ${SUMMARY}" > "${RUN_DIR}/STATUS"

    # Copy bench artefacts into self-contained run dir
    BENCH_OUT="${OUT_ROOT}/_bench/run_${LABEL}${BENCH_LABEL_SUFFIX}"
    if [ -d "${BENCH_OUT}" ]; then
        for f in report.json bench.jsonl dcgm_tc.tsv fsdp_override.json gpu_metrics.jsonl; do
            [ -f "${BENCH_OUT}/${f}" ] && cp "${BENCH_OUT}/${f}" "${RUN_DIR}/" || true
        done
        # Symlink swift logging.jsonl
        VDIR=$(ls -dt ${BENCH_OUT}/v*-* 2>/dev/null | head -n 1 || true)
        if [ -n "${VDIR}" ] && [ -f "${VDIR}/logging.jsonl" ]; then
            ln -sf "${VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
        fi
    fi

    # Append to attempts.md
    printf "| run_%s | MBS=%d GAS=%d GBS=%d | %s | %s |\n" \
        "$(basename ${RUN_DIR})" "${MBS}" "${GAS}" "${GBS}" "${STATUS}" "${SUMMARY}" >> "${ATTEMPTS_FILE}"

    # Pull a quick summary line from report.json if available
    if [ -f "${RUN_DIR}/report.json" ]; then
        echo "    summary: $(python3 -c "
import json, sys
d = json.load(open('${RUN_DIR}/report.json'))
print(f\"step={d.get('mean_step_time_ms', '?')}ms tokens/s/gpu={d.get('tokens_per_sec_per_gpu', '?')} peak_mem={d.get('peak_mem_gib_from_swift_log', '?')}GiB\")
" 2>/dev/null)"
    fi

    # Always cleanup leftover python / torchrun children to free GPU before next config.
    # Be aggressive: kill all python in the container (we run nothing else there during sweep).
    # Then poll until GPU 0 is empty (or timeout) so the next iteration starts on a clean slate.
    docker exec fsdp_sft pkill -9 python 2>/dev/null || true
    set +e  # protect against pipefail in the polling loop
    for i in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{gsub(/ /,""); print}')
        if [ -n "${USED}" ] && [ "${USED}" -lt 100 ] 2>/dev/null; then
            break
        fi
    done
    set -e
    sleep 2
done

echo "=========================================="
echo "P1 sweep complete. Run dirs: ${OUT_ROOT}/run_*"
echo "Attempts log: ${ATTEMPTS_FILE}"
echo "=========================================="
