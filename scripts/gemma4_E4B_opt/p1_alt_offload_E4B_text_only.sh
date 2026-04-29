#!/usr/bin/env bash
# P1 alt-offload (DS-equivalent) on Gemma-4-E4B-it — text-only SFT bench
#
# Goal: reproduce the P1 alt-offload topology
#   (FSDP2 + CPUOffloadPolicy(opt+grad+param), GBS=64, AC=on, MBS=1/GAS=16,
#    SP=2, truncation=right, packing=false, max_length=16384)
# but on `google/gemma-4-E4B-it` (dense + PLE, 4B effective ≈ 7.5B trainable
# when freeze_vit/freeze_aligner are on for text-only training), feeding the
# `sft-data/SFT_0424_2.jsonl` dataset (51557 samples).
#
# Architectural notes (vs gemma-4-26B-A4B-it baseline):
#   * model_type: 'gemma4'           ← same dispatch path
#   * dense (enable_moe_block=false) ← no MoE; NUM_ACTIVE_PARAMS = compute params
#   * 42 text decoder layers (Gemma4TextDecoderLayer)
#   * 16 vision encoder layers (Gemma4VisionEncoderLayer)
#   * 12 audio layers (Gemma4AudioLayer) ← present in E-series, absent in 26B-A4B
#   * global_head_dim=512 still ⇒ same SDPA mem_efficient pin (sitecustomize 1/3)
#   * vocab_size=262144 still ⇒ chunked CE patch (modeling_gemma4.py) needed
#   * tie_word_embeddings=true; lm_head shares weight with embed_tokens
#   * Per-Layer Embeddings (PLE): 2.82B lookup-only (no FLOPs);
#     transformer compute ≈ 3.95B + tied lm_head 0.67B = 4.62B
#
# Text-only enforcement:
#   * `--freeze_vit true --freeze_aligner true` follows ModelArch.gemma3n:
#     vision_tower=[model.vision_tower, model.audio_tower],
#     aligner=[model.embed_vision, model.embed_audio]
#     ⇒ both vision and audio towers frozen; only language_model + tied
#     lm_head are trainable.
#   * FSDP `transformer_layer_cls_to_wrap` includes all three layer classes
#     because FSDP must still wrap frozen modules (no_grad ≠ no_wrap).
#
# Required env (already prepared):
#   - container fsdp_sft up
#   - sitecustomize patch + GEMMA4_FORCE_MEM_EFFICIENT_SDP=1
#   - modeling_gemma4 patch (md5 39ebf386a992fea9eac0883f459ac658)
#   - liger gemma4 dispatch registered (silent on E4B since RMSNorm/GeGLU
#     subset matches, but doesn't change throughput envelope significantly)
#
# Output dir: experiments/gemma4_E4B_alt_offload/run_<TIMESTAMP>_text_only/
set -euo pipefail

OUT_ROOT="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_E4B_alt_offload"
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:-text_only}"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it
DATASET_PATH=/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl
DATASET_SIZE=51557

TOTAL_STEPS="${TOTAL_STEPS:-40}"
WARMUP_BENCH="${WARMUP_BENCH:-5}"
MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

# Trainable / compute params (text-only):
#   * trainable (full backprop, used for 6N FLOP estimate including PLE
#     lookup which sees no real GEMM): 7.518e9 (text language_model)
#   * compute-only (excludes PLE lookup table 2.82B and embedding lookups
#     that are pure scatter/gather, leaves transformer + tied lm_head): 4.62e9
NUM_PARAMS="${NUM_PARAMS:-7.518e9}"
NUM_ACTIVE_PARAMS="${NUM_ACTIVE_PARAMS:-4.62e9}"

BENCH_DIR="${OUT_ROOT}/_bench"
RUN_NAME="run_alt_offload_E4B_${LABEL}"

echo "=========================================="
echo "P1 alt-offload (DS-equivalent on FSDP2) — Gemma-4-E4B-it text-only"
echo "Run dir   : ${RUN_DIR}"
echo "Bench out : ${BENCH_DIR}/${RUN_NAME}"
echo "Dataset   : ${DATASET_PATH} (${DATASET_SIZE} samples)"
echo "Steps     : ${TOTAL_STEPS} (${WARMUP_BENCH} warmup + $((TOTAL_STEPS - WARMUP_BENCH)) measured)"
echo "GBS / MBS / GAS : 64 / 1 / 16  (SP=2 → DP=4)"
echo "Port      : ${MASTER_PORT}"
echo "=========================================="

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
TRAIN_JSONL=${DATASET_PATH} \
FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer,Gemma4AudioLayer \
FSDP_CPU_RAM_EFFICIENT=false \
NUM_PARAMS=${NUM_PARAMS} NUM_ACTIVE_PARAMS=${NUM_ACTIVE_PARAMS} \
DATASET_SIZE=${DATASET_SIZE} \
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
        STATUS="FAILED"
        SUMMARY="OOM"
    else
        STATUS="FAILED"
        SUMMARY="exit=${EXIT}"
    fi
fi
echo "${STATUS} — ${SUMMARY}" > "${RUN_DIR}/STATUS"

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

docker exec fsdp_sft pkill -9 python 2>/dev/null || true

exit ${EXIT}
